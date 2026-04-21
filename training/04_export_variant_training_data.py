#!/usr/bin/env python3
"""
Step 4 (Part B): Export variant-label training data for the foil-pattern classifier.

Reads cards directly from Postgres — NOT from ChromaDB — because `variant_label`
and `set_slug` are authoritative in the DB and may not have been backfilled into
ChromaDB yet.

Emits:
    training_data/variant_classifier/variant_manifest_train.jsonl
    training_data/variant_classifier/variant_manifest_val.jsonl
    training_data/variant_classifier/label_map.json

Class key format: "<set_slug>::<variant_label_lower_stripped>". Classes with
fewer than --min-samples examples are routed to a single "__rare__" bucket at
train time; they are still useful for eval coverage numbers.

Run from the Scraperv2 directory so config/database/embeddings_dinov2 import:
    python training/04_export_variant_training_data.py
    python training/04_export_variant_training_data.py --min-samples 50 --val-ratio 0.1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

# Make the project root importable (for config + database)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2.extras
from rich.console import Console

import database as db

console = Console()

RARE_CLASS = "__rare__"


def class_key(set_slug: str, variant_label: str) -> str:
    v = (variant_label or "").strip().lower()
    v = re.sub(r"\s+", " ", v)
    return f"{(set_slug or '').strip().lower()}::{v}"


def fetch_rows(limit: int | None = None) -> list[dict]:
    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql = """
        SELECT product_id, set_slug, variant_label, image_path, product_name
          FROM cards
         WHERE status = 'downloaded'
           AND image_path IS NOT NULL
           AND image_path <> ''
           AND variant_label IS NOT NULL
           AND variant_label <> ''
         ORDER BY id
    """
    if limit:
        sql += " LIMIT %s"
        cur.execute(sql, (limit,))
    else:
        cur.execute(sql)
    rows = cur.fetchall()
    cur.close()
    db.put_connection(conn)
    return [dict(r) for r in rows]


def export(output_dir: str, min_samples: int, val_ratio: float, seed: int,
           verify_images: bool, limit: int | None):
    console.print("\n[bold]Exporting variant-classifier training data[/bold]")
    console.print(f"  min_samples: [cyan]{min_samples}[/cyan]   val_ratio: [cyan]{val_ratio}[/cyan]   seed: [cyan]{seed}[/cyan]\n")

    rows = fetch_rows(limit=limit)
    console.print(f"  Rows with variant_label + image_path: [cyan]{len(rows):,}[/cyan]")

    if verify_images:
        before = len(rows)
        rows = [r for r in rows if os.path.exists(r["image_path"])]
        missing = before - len(rows)
        if missing:
            console.print(f"  Skipped missing image files: [yellow]{missing:,}[/yellow]")

    if not rows:
        console.print("[red]No eligible rows. Did you backfill variant_label yet?[/red]")
        sys.exit(1)

    class_counts = Counter(class_key(r["set_slug"], r["variant_label"]) for r in rows)
    kept = {k for k, c in class_counts.items() if c >= min_samples}
    rare_keys = {k for k in class_counts if k not in kept}
    rare_rows = sum(class_counts[k] for k in rare_keys)

    console.print(f"  Total class keys: [cyan]{len(class_counts):,}[/cyan]")
    console.print(f"  Kept classes (>= {min_samples}): [green]{len(kept):,}[/green]")
    console.print(f"  Rare classes -> {RARE_CLASS}: [yellow]{len(rare_keys):,}[/yellow]  ({rare_rows:,} rows)")

    # Build per-class sample lists
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        k = class_key(r["set_slug"], r["variant_label"])
        cls = k if k in kept else RARE_CLASS
        by_class[cls].append({
            "product_id":    r["product_id"],
            "image_path":    r["image_path"],
            "set_slug":      r["set_slug"],
            "variant_label": r["variant_label"],
            "class":         cls,
        })

    # Stratified split — at least 1 val sample per kept class when possible
    rng = random.Random(seed)
    train, val = [], []
    for cls, samples in by_class.items():
        rng.shuffle(samples)
        n_val = max(1, int(len(samples) * val_ratio)) if len(samples) >= 10 else 0
        val.extend(samples[:n_val])
        train.extend(samples[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)

    label_list = sorted(by_class.keys())
    label_map = {name: idx for idx, name in enumerate(label_list)}

    # Set-index table, used by the hierarchical variant classifier and the
    # set-stratified batch sampler. Derived here rather than at train time so
    # the checkpoint embeds a stable set ordering.
    set_labels = sorted({
        (rec["set_slug"] or "").strip().lower()
        for recs in by_class.values() for rec in recs
    })
    set_to_idx = {s: i for i, s in enumerate(set_labels)}

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "variant_manifest_train.jsonl")
    val_path   = os.path.join(output_dir, "variant_manifest_val.jsonl")
    lm_path    = os.path.join(output_dir, "label_map.json")

    def _dump(records, path):
        with open(path, "w") as f:
            for rec in records:
                rec = dict(rec)
                rec["label_idx"] = label_map[rec["class"]]
                rec["set_idx"]   = set_to_idx.get(
                    (rec.get("set_slug") or "").strip().lower(), 0
                )
                f.write(json.dumps(rec) + "\n")

    _dump(train, train_path)
    _dump(val, val_path)

    with open(lm_path, "w") as f:
        json.dump({
            "labels": label_list,
            "set_labels": set_labels,
            "rare_class": RARE_CLASS,
            "min_samples": min_samples,
            "num_classes": len(label_list),
            "num_sets": len(set_labels),
        }, f, indent=2)

    console.print(f"\n  [green]Wrote {train_path}[/green]  ({len(train):,} rows)")
    console.print(f"  [green]Wrote {val_path}[/green]  ({len(val):,} rows)")
    console.print(f"  [green]Wrote {lm_path}[/green]  ({len(label_list):,} labels)")

    # Top classes for sanity
    console.print("\n  Top 15 classes by sample count:")
    for cls, n in sorted(((c, len(s)) for c, s in by_class.items()),
                         key=lambda x: x[1], reverse=True)[:15]:
        console.print(f"    {cls:<60} {n:>6,}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="./training_data/variant_classifier")
    ap.add_argument("--min-samples", type=int, default=10)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-verify-images", action="store_true",
                    help="Skip checking that each image_path exists on disk")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap rows for quick experiments (default: no cap)")
    args = ap.parse_args()

    export(
        output_dir=args.output,
        min_samples=args.min_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
        verify_images=not args.no_verify_images,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
