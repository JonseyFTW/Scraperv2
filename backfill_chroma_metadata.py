#!/usr/bin/env python3
"""Backfill ChromaDB metadata for existing embeddings — no re-embedding.

Walks the local DINOv2 collection, looks up each product_id in Postgres,
builds the new metadata dict (shared with embeddings_dinov2._build_card_metadata),
and calls collection.update(ids, metadatas=...) in batches.

After this runs locally, use `python embeddings_dinov2.py sync` to push the
updated metadata to the RunPod ChromaDB instance.

Usage:
    python backfill_chroma_metadata.py                # update local Chroma
    python backfill_chroma_metadata.py --batch 2000   # custom batch size
    python backfill_chroma_metadata.py --dry-run      # report only
"""
from __future__ import annotations

import argparse
import time

import psycopg2.extras
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import database as db
from embeddings_dinov2 import _build_card_metadata, get_collection

console = Console()


def _chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def _fetch_card_rows(product_ids: list[str]) -> dict[str, dict]:
    """Return product_id -> card-row dict for the given IDs."""
    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT product_id, product_name, set_slug, image_path,
               gcs_image_url, gcs_thumb_url, card_number, print_run,
               player_name, variant_label, loose_price
          FROM cards
         WHERE product_id = ANY(%s)
    """, (product_ids,))
    rows = cur.fetchall()
    cur.close()
    db.put_connection(conn)
    return {r["product_id"]: dict(r) for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    collection = get_collection()
    total_in_collection = collection.count()
    console.print(f"[bold]Collection:[/bold] {collection.name} ({total_in_collection:,} items)")

    # Pull all IDs up front. ~600K IDs ≈ 50 MB of strings, fine in RAM.
    console.print("  Loading existing IDs...")
    all_ids = collection.get(include=[])["ids"]
    console.print(f"  Got {len(all_ids):,} IDs")

    if args.dry_run:
        # Sample 5 to show what the new metadata will look like
        sample_map = _fetch_card_rows(all_ids[:5])
        for pid in all_ids[:5]:
            row = sample_map.get(pid)
            if row:
                console.print(f"  {pid}: {_build_card_metadata(row)}")
            else:
                console.print(f"  {pid}: [red]not found in Postgres[/red]")
        return

    t0 = time.time()
    updated = 0
    missing = 0

    with Progress(TextColumn("{task.description}"), BarColumn(),
                  TextColumn("{task.completed}/{task.total}"),
                  TimeRemainingColumn()) as progress:
        task = progress.add_task("update", total=len(all_ids))

        for chunk in _chunked(all_ids, args.batch):
            row_map = _fetch_card_rows(chunk)
            ids, metas = [], []
            for pid in chunk:
                row = row_map.get(pid)
                if row is None:
                    missing += 1
                    continue
                ids.append(pid)
                metas.append(_build_card_metadata(row))
            if ids:
                collection.update(ids=ids, metadatas=metas)
                updated += len(ids)
            progress.advance(task, advance=len(chunk))

    console.print(
        f"[green]Updated {updated:,} metadatas[/green] in {time.time()-t0:.1f}s; "
        f"{missing:,} IDs had no matching row in Postgres."
    )


if __name__ == "__main__":
    main()
