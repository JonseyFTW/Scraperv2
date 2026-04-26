#!/usr/bin/env python3
"""
Step 8 (rec 6): Build the per-set color-ROI reference table.

For each set in ``color_roi_lib.COLOR_SWAP_REGISTRY``, this script scans the
training manifest (output of step 4), crops the per-set ROI from each card
image, computes mean LAB, and aggregates a per-variant reference vector.

Output:
    color_roi_references.json     (the references — ship this to RunPod)
    color_roi_eval.json           (val-set accuracy + per-set diagnostics)

Run from the Scraperv2 directory:
    python training/08_train_color_roi.py
    python training/08_train_color_roi.py --data-dir ./training_data/variant_classifier
    python training/08_train_color_roi.py --margin 4.0 --top-confusion-pairs 20

The reference JSON is small (a few KB) and doesn't change between training
runs unless the registry or training data does. After it's built locally:

    1. Copy ``color_roi_references.json`` to your RunPod volume.
    2. Set ``COLOR_ROI_REFERENCES`` env var on the endpoint to its path.
    3. Redeploy the handler container.

The handler will load it on cold start and use it to override the variant MLP
when (a) the requested set is in the registry, and (b) the ROI classifier is
confident (i.e. the second-best LAB distance is at least ``--margin`` away).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Repo root + training/ on the path so imports work no matter where the user
# launches this from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from variant_classifier_lib import load_jsonl
from color_roi_lib import (
    COLOR_SWAP_REGISTRY,
    extract_roi_lab,
    classify_with_confidence,
    get_candidates,
)

console = Console()


def _normalize_variant(label: str) -> str:
    s = (label or "").strip().lower()
    return re.sub(r"\s+", " ", s)


def _records_in_registry(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by set_slug, keeping only sets in the registry."""
    out: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        slug = (rec.get("set_slug") or "").strip().lower()
        if slug in COLOR_SWAP_REGISTRY:
            out[slug].append(rec)
    return out


def _build_references(grouped: dict[str, list[dict]], max_per_class: int):
    """Compute per-variant mean LAB in each set's ROI."""
    from PIL import Image

    references: dict[str, dict[str, list[float]]] = {}
    sample_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total = sum(min(len(v), max_per_class * 50) for v in grouped.values())
    with Progress(
        TextColumn("Building references"), BarColumn(),
        TextColumn("{task.completed}/{task.total}"), TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("ref", total=total)

        for set_slug, recs in grouped.items():
            per_variant_lab: dict[str, list[np.ndarray]] = defaultdict(list)
            registered = {_normalize_variant(c) for c in get_candidates(set_slug)}

            for rec in recs:
                variant_norm = _normalize_variant(rec.get("variant_label", ""))
                # If the registry pinned candidates, only use those samples for
                # references — anything else is noise (different variant).
                if registered and variant_norm not in registered:
                    progress.advance(task)
                    continue
                if sample_counts[set_slug][variant_norm] >= max_per_class:
                    progress.advance(task)
                    continue
                try:
                    img = Image.open(rec["image_path"]).convert("RGB")
                except Exception:
                    progress.advance(task)
                    continue
                lab = extract_roi_lab(img, set_slug)
                if lab is None:
                    progress.advance(task)
                    continue
                per_variant_lab[variant_norm].append(lab)
                sample_counts[set_slug][variant_norm] += 1
                progress.advance(task)

            set_refs = {}
            for variant, labs in per_variant_lab.items():
                if not labs:
                    continue
                arr = np.stack(labs, axis=0)
                set_refs[variant] = arr.mean(axis=0).tolist()
            if set_refs:
                references[set_slug] = set_refs

    return references, sample_counts


def _eval_against_val(val_records: list[dict], references: dict, margin: float):
    """Per-set top-1 accuracy of the ROI classifier vs. ground truth."""
    from PIL import Image

    grouped = _records_in_registry(val_records)
    out: dict[str, dict] = {}
    overall_attempted = 0
    overall_decided   = 0
    overall_correct   = 0

    for set_slug, recs in grouped.items():
        attempted = 0  # records with a valid ROI extraction
        decided   = 0  # records where the classifier was confident enough
        correct   = 0  # decided correctly
        confusion: Counter[tuple[str, str]] = Counter()

        for rec in recs:
            attempted += 1
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
            except Exception:
                continue
            true_variant = _normalize_variant(rec.get("variant_label", ""))
            result = classify_with_confidence(img, set_slug, references, margin=margin)
            if result is None:
                continue
            pred, _dist, _all = result
            decided += 1
            if pred == true_variant:
                correct += 1
            else:
                confusion[(true_variant, pred)] += 1

        if attempted:
            out[set_slug] = {
                "attempted":    attempted,
                "decided":      decided,
                "decision_pct": round(100 * decided / attempted, 1),
                "correct":      correct,
                "accuracy_when_decided": round(correct / max(decided, 1), 4),
                "top_confusions": [
                    {"true": t, "pred": p, "count": c}
                    for (t, p), c in confusion.most_common(10)
                ],
            }
            overall_attempted += attempted
            overall_decided   += decided
            overall_correct   += correct

    out["__overall__"] = {
        "attempted": overall_attempted,
        "decided":   overall_decided,
        "decision_pct": round(100 * overall_decided / max(overall_attempted, 1), 1),
        "correct":   overall_correct,
        "accuracy_when_decided": round(overall_correct / max(overall_decided, 1), 4),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./training_data/variant_classifier")
    ap.add_argument("--output-dir", default="./eval/color_roi")
    ap.add_argument("--max-per-class", type=int, default=200,
                    help="Cap samples per (set, variant) when building references "
                         "so well-represented classes don't drown out rare ones.")
    ap.add_argument("--margin", type=float, default=4.0,
                    help="LAB distance margin between best and second-best for "
                         "the classifier to commit. ~4.0 ≈ just-noticeable difference.")
    ap.add_argument("--no-eval", action="store_true",
                    help="Skip val-set accuracy reporting.")
    args = ap.parse_args()

    train_path = Path(args.data_dir) / "variant_manifest_train.jsonl"
    val_path   = Path(args.data_dir) / "variant_manifest_val.jsonl"
    if not train_path.exists():
        console.print(f"[red]Missing {train_path} — run step 4 first[/red]")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Building color-ROI references[/bold] for {len(COLOR_SWAP_REGISTRY)} registered sets")
    train = load_jsonl(str(train_path))
    grouped_train = _records_in_registry(train)
    for slug, recs in sorted(grouped_train.items()):
        console.print(f"  {slug:60s}  {len(recs):>6,} train rows")

    references, sample_counts = _build_references(grouped_train, args.max_per_class)

    refs_path = out_dir / "color_roi_references.json"
    refs_path.write_text(json.dumps({
        "references":      references,
        "registry":        {k: v for k, v in COLOR_SWAP_REGISTRY.items() if k in references},
        "sample_counts":   {k: dict(v) for k, v in sample_counts.items()},
        "schema_version":  1,
    }, indent=2))
    console.print(f"\n[green]Wrote {refs_path}[/green]   ({len(references)} sets)")

    if not args.no_eval and val_path.exists():
        console.print("\n[bold]Evaluating ROI classifier on val[/bold]")
        val = load_jsonl(str(val_path))
        eval_out = _eval_against_val(val, references, args.margin)
        eval_path = out_dir / "color_roi_eval.json"
        eval_path.write_text(json.dumps(eval_out, indent=2))
        ov = eval_out["__overall__"]
        console.print(
            f"  Attempted:           [cyan]{ov['attempted']:,}[/cyan]\n"
            f"  Decided (confident): [cyan]{ov['decided']:,}[/cyan]  ({ov['decision_pct']}%)\n"
            f"  Accuracy when decided: [green]{ov['accuracy_when_decided']:.4f}[/green]"
        )
        console.print(f"  [green]{eval_path}[/green]")
        console.print(
            "\nNext step: copy color_roi_references.json to the RunPod volume and "
            "set COLOR_ROI_REFERENCES=/runpod-volume/color_roi_references.json on "
            "the endpoint, then redeploy the container."
        )


if __name__ == "__main__":
    main()
