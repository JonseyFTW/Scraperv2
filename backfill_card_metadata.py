#!/usr/bin/env python3
"""Backfill `cards` metadata columns introduced in PRD_parallel_disambiguation.md.

Three passes, each idempotent and resumable:

    Pass 1a — derive gcs_image_url / gcs_thumb_url from existing image_url
              (pure SQL, no HTTP, covers ~95% of rows)
    Pass 1b — parse player_name / card_number / print_run / variant_label
              from product_name (in-process, chunked UPDATE)
    Pass 2  — network fetch for rows still missing gcs_image_url
              (rate-limited, reuses scraper_v3.SessionManager)

Usage:
    python backfill_card_metadata.py --pass 1a
    python backfill_card_metadata.py --pass 1b
    python backfill_card_metadata.py --pass 1         # runs 1a then 1b
    python backfill_card_metadata.py --pass 2 --limit 1000
    python backfill_card_metadata.py --stats
"""
from __future__ import annotations

import argparse
import sys
import time

import psycopg2.extras
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

import database as db
from card_name_parser import parse_product_name

console = Console()


def ensure_schema():
    """Idempotently add the parallel-disambiguation columns if they don't exist yet.

    The service's `database.init_db()` adds them, but this script can be run
    standalone on a fresh checkout where init_db() hasn't been called since the
    schema additions shipped. Calling init_db() is safe — every statement is
    guarded with IF NOT EXISTS / DO $$ ... duplicate_column.
    """
    console.print("[dim]Ensuring schema is up to date (idempotent init_db)...[/dim]")
    db.init_db()


# ── Pass 1a: SQL-only GCS URL derivation ──────────────────────────────────

GCS_LIKE = "%storage.googleapis.com/images.pricecharting.com/%"

PASS_1A_SQL = """
UPDATE cards
   SET gcs_image_url = regexp_replace(image_url,
         '(storage\\.googleapis\\.com/images\\.pricecharting\\.com/[^/]+)/\\d+[^"'' ]*',
         '\\1/1600.jpg'),
       gcs_thumb_url = regexp_replace(image_url,
         '(storage\\.googleapis\\.com/images\\.pricecharting\\.com/[^/]+)/\\d+[^"'' ]*',
         '\\1/240.jpg')
 WHERE image_url LIKE %s
   AND gcs_image_url IS NULL
"""


def pass_1a():
    console.print("[bold]Pass 1a:[/bold] SQL derive gcs_image_url / gcs_thumb_url from image_url")
    conn = db.get_connection()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM cards WHERE image_url LIKE %s AND gcs_image_url IS NULL",
                (GCS_LIKE,))
    before = cur.fetchone()[0]
    console.print(f"  Rows needing derivation: [cyan]{before:,}[/cyan]")

    t0 = time.time()
    cur.execute(PASS_1A_SQL, (GCS_LIKE,))
    updated = cur.rowcount
    conn.commit()
    cur.close()
    db.put_connection(conn)
    console.print(f"  [green]Updated {updated:,} rows[/green] in {time.time()-t0:.1f}s")


# ── Pass 1b: in-process parse of product_name ─────────────────────────────

def pass_1b(chunk_size: int = 10_000):
    console.print("[bold]Pass 1b:[/bold] parse product_name -> player/card_number/print_run/variant_label")
    total = 0
    # Cursor-based pagination by id so rows where the parser yields player_name=None
    # (e.g. product_names starting with '#' or '[') don't get re-selected forever.
    last_id = 0
    while True:
        conn = db.get_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("""
            SELECT id, product_name FROM cards
             WHERE id > %s
               AND player_name IS NULL
               AND product_name IS NOT NULL
               AND product_name <> ''
             ORDER BY id
             LIMIT %s
        """, (last_id, chunk_size))
        rows = cur.fetchall()
        cur.close()

        if not rows:
            db.put_connection(conn)
            break
        last_id = rows[-1]["id"]

        updates = []
        for r in rows:
            p = parse_product_name(r["product_name"])
            updates.append((p.player_name, p.card_number, p.print_run, p.variant_label, r["id"]))

        cur = conn.cursor()
        psycopg2.extras.execute_batch(cur, """
            UPDATE cards
               SET player_name=%s, card_number=%s, print_run=%s, variant_label=%s
             WHERE id=%s
        """, updates)
        conn.commit()
        cur.close()
        db.put_connection(conn)

        total += len(rows)
        console.print(f"  parsed +{len(rows):,} (total {total:,})")

    console.print(f"  [green]Pass 1b complete: {total:,} rows updated[/green]")


# ── Pass 2: network fetch for missing GCS URLs ────────────────────────────
# Opt-in, imports scraper_v3 lazily so the script works without curl_cffi installed.

def pass_2(limit: int, concurrency: int):
    try:
        import asyncio
        from scraper_v3 import SessionManager, fetch_image_url, gcs_urls_from_any
    except ImportError as e:
        console.print(f"[red]Pass 2 requires scraper_v3 dependencies: {e}[/red]")
        sys.exit(1)

    console.print(f"[bold]Pass 2:[/bold] network fetch (limit={limit}, concurrency={concurrency})")

    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT id, product_id, full_url FROM cards
         WHERE gcs_image_url IS NULL
           AND full_url IS NOT NULL
           AND full_url <> ''
         ORDER BY id
         LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    db.put_connection(conn)

    if not rows:
        console.print("  Nothing to backfill — every row has gcs_image_url or no full_url")
        return

    console.print(f"  Fetching {len(rows):,} product pages...")

    async def _run():
        session_mgr = SessionManager()
        ok = 0
        miss = 0
        err = 0

        with Progress(TextColumn("{task.description}"), BarColumn(),
                      TextColumn("{task.completed}/{task.total}"),
                      TimeRemainingColumn()) as progress:
            task = progress.add_task("backfill", total=len(rows))

            for i in range(0, len(rows), concurrency):
                batch = rows[i:i+concurrency]
                session = await session_mgr.get_session()
                results = await asyncio.gather(
                    *[fetch_image_url(session, dict(r)) for r in batch],
                    return_exceptions=True,
                )

                for r, res in zip(batch, results):
                    if isinstance(res, Exception) or getattr(res, "error", None):
                        err += 1
                    elif getattr(res, "image_url", None):
                        gcs_full, gcs_thumb = gcs_urls_from_any(res.image_url)
                        db.update_card_image_url(
                            r["product_id"], res.image_url, gcs_full, gcs_thumb,
                        )
                        ok += 1
                    else:
                        miss += 1
                    progress.advance(task)

        console.print(f"  [green]OK: {ok:,}[/green]  [yellow]no image: {miss:,}[/yellow]  [red]error: {err:,}[/red]")

    asyncio.run(_run())


# ── Stats ─────────────────────────────────────────────────────────────────

def print_stats():
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cards")
    total = cur.fetchone()[0]

    queries = {
        "gcs_image_url populated":   "SELECT COUNT(*) FROM cards WHERE gcs_image_url IS NOT NULL",
        "gcs_thumb_url populated":   "SELECT COUNT(*) FROM cards WHERE gcs_thumb_url IS NOT NULL",
        "player_name populated":     "SELECT COUNT(*) FROM cards WHERE player_name IS NOT NULL",
        "card_number populated":     "SELECT COUNT(*) FROM cards WHERE card_number IS NOT NULL",
        "print_run populated":       "SELECT COUNT(*) FROM cards WHERE print_run IS NOT NULL",
        "variant_label populated":   "SELECT COUNT(*) FROM cards WHERE variant_label IS NOT NULL",
        "GCS-eligible missing gcs_image_url":
            f"SELECT COUNT(*) FROM cards WHERE image_url LIKE '{GCS_LIKE}' AND gcs_image_url IS NULL",
    }

    console.print(f"[bold]Cards total:[/bold] {total:,}")
    for label, q in queries.items():
        cur.execute(q)
        n = cur.fetchone()[0]
        pct = (n / total * 100) if total else 0
        console.print(f"  {label:<40} {n:>10,}  ({pct:5.1f}%)")
    cur.close()
    db.put_connection(conn)


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="phase", choices=["1", "1a", "1b", "2"],
                    help="Which pass to run (default: stats only)")
    ap.add_argument("--limit", type=int, default=1000, help="Pass 2 row limit")
    ap.add_argument("--concurrency", type=int, default=20, help="Pass 2 concurrency")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    # Every path below touches the new columns, so make sure they exist.
    ensure_schema()

    if args.phase in ("1", "1a"):
        pass_1a()
    if args.phase in ("1", "1b"):
        pass_1b()
    if args.phase == "2":
        pass_2(args.limit, args.concurrency)
    if args.stats or not args.phase:
        print_stats()


if __name__ == "__main__":
    main()
