#!/usr/bin/env python3
"""
TCGPlayer Pokemon Scraper — fetches card catalog from TCGCSV (free, no auth)
and downloads high-resolution images from TCGPlayer's CDN.

This captures ALL variants and printings that TCGPlayer tracks, including
sets like "Base Set (Shadowless)" that TCGdex doesn't have.

Usage:
    python tcgplayer_scraper.py fetch                # Fetch all sets + cards from TCGCSV
    python tcgplayer_scraper.py fetch --set 1663     # Fetch one set (by groupId)
    python tcgplayer_scraper.py download             # Download all pending images
    python tcgplayer_scraper.py download --limit 500 # Download up to 500 images
    python tcgplayer_scraper.py stats                # Show progress
    python tcgplayer_scraper.py sets                 # List all available sets
    python tcgplayer_scraper.py run                  # Full pipeline: fetch + download
    python tcgplayer_scraper.py run --limit 500      # Full pipeline, limited

Requirements:
    pip install requests rich psycopg2-binary
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeRemainingColumn,
)
from rich.table import Table

import config
import database as db

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_folder_name(name: str) -> str:
    """Sanitize a set name into a safe directory name."""
    for ch in ('/', '\\', ':', '*', '?', '"', '<', '>', '|'):
        name = name.replace(ch, '-')
    return name.strip(". ") or "unknown"


def _image_path_for_card(card: dict) -> str:
    """Build local image path: TCGPLAYER_IMAGE_DIR/SetName/productId.jpg"""
    folder = _safe_folder_name(card.get("group_name") or "unknown")
    return os.path.join(config.TCGPLAYER_IMAGE_DIR, folder, f"{card['product_id']}.jpg")


def _build_image_url(product_id: int) -> str:
    """Build CDN image URL for a product."""
    return f"{config.TCGPLAYER_IMAGE_CDN}/{product_id}_{config.TCGPLAYER_IMAGE_SIZE}.jpg"


def _extract_extended(ext_data: list[dict], field: str) -> str:
    """Extract a field from TCGPlayer extendedData array."""
    if not ext_data:
        return ""
    for item in ext_data:
        if item.get("name") == field:
            return item.get("value", "")
    return ""


# ---------------------------------------------------------------------------
# Fetch sets and cards from TCGCSV
# ---------------------------------------------------------------------------

def fetch_sets(set_filter: int | None = None) -> list[dict]:
    """Fetch all Pokemon TCG sets (groups) from TCGCSV."""
    console.print("[cyan]Fetching Pokemon sets from TCGCSV...[/cyan]")

    url = f"{config.TCGCSV_BASE}/{config.TCGPLAYER_CATEGORY_ID}/groups"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # TCGCSV returns {"results": [...]} or just a list
    groups = data if isinstance(data, list) else data.get("results", [])

    if set_filter:
        groups = [g for g in groups if g.get("groupId") == set_filter]
        if not groups:
            console.print(f"[red]Set groupId '{set_filter}' not found[/red]")
            return []

    console.print(f"Found [cyan]{len(groups)}[/cyan] sets")
    return groups


def fetch_cards(set_filter: int | None = None):
    """Fetch all card metadata from TCGCSV and store in PostgreSQL."""
    db.init_db()
    groups = fetch_sets(set_filter)
    if not groups:
        return 0

    console.print(f"\n[bold]Fetching card details for {len(groups)} sets...[/bold]\n")

    total_cards = 0
    failed_sets = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console,
    ) as progress:
        task = progress.add_task("Fetching sets", total=len(groups))

        for g in groups:
            group_id = g.get("groupId")
            group_name = g.get("name", str(group_id))
            progress.update(task, description=f"{group_name[:50]}")

            try:
                # Upsert the set
                db.upsert_tcgplayer_set({
                    "group_id": group_id,
                    "name": group_name,
                    "abbreviation": g.get("abbreviation", ""),
                    "is_supplemental": g.get("isSupplemental", False),
                    "published_on": g.get("publishedOn", ""),
                    "modified_on": g.get("modifiedOn", ""),
                    "card_count": 0,  # updated after fetching products
                })

                # Fetch products for this set
                prod_url = f"{config.TCGCSV_BASE}/{config.TCGPLAYER_CATEGORY_ID}/{group_id}/products"
                prod_resp = requests.get(prod_url, timeout=30)
                prod_resp.raise_for_status()
                prod_data = prod_resp.json()
                products = prod_data if isinstance(prod_data, list) else prod_data.get("results", [])

                # Filter to only "Cards" type products (skip sealed product, accessories, etc.)
                cards_batch = []
                for p in products:
                    # Build card record
                    ext = p.get("extendedData", [])
                    card_number = _extract_extended(ext, "Number")
                    rarity = _extract_extended(ext, "Rarity")
                    card_type = _extract_extended(ext, "CardType")

                    cards_batch.append({
                        "product_id": p["productId"],
                        "name": p.get("name", "Unknown"),
                        "clean_name": p.get("cleanName", ""),
                        "group_id": group_id,
                        "group_name": group_name,
                        "image_url": _build_image_url(p["productId"]),
                        "product_url": p.get("url", ""),
                        "card_number": card_number,
                        "rarity": rarity,
                        "card_type": card_type,
                        "ext_data": json.dumps(ext) if ext else None,
                        "status": "pending",
                    })

                if cards_batch:
                    db.upsert_tcgplayer_cards_bulk(cards_batch)
                    total_cards += len(cards_batch)

                    # Update set card count
                    db.upsert_tcgplayer_set({
                        "group_id": group_id,
                        "name": group_name,
                        "abbreviation": g.get("abbreviation", ""),
                        "is_supplemental": g.get("isSupplemental", False),
                        "published_on": g.get("publishedOn", ""),
                        "modified_on": g.get("modifiedOn", ""),
                        "card_count": len(cards_batch),
                    })

            except Exception as e:
                failed_sets.append((group_id, group_name, str(e)))

            progress.advance(task)
            time.sleep(config.TCGPLAYER_REQUEST_DELAY)

    console.print(f"\n[green]Fetched {total_cards} cards from {len(groups)} sets[/green]")
    if failed_sets:
        console.print(f"[yellow]{len(failed_sets)} sets failed:[/yellow]")
        for gid, gname, err in failed_sets[:10]:
            console.print(f"  [dim]{gname} ({gid}): {err}[/dim]")

    db.log_event("tcgplayer_fetch", f"Fetched {total_cards} cards from {len(groups)} sets")
    return total_cards


# ---------------------------------------------------------------------------
# Download images
# ---------------------------------------------------------------------------

def _download_one(card: dict, session: requests.Session) -> tuple[int, bool, str]:
    """Download a single card image. Returns (product_id, success, filepath)."""
    image_url = card.get("image_url")
    if not image_url:
        return card["product_id"], False, ""

    filepath = _image_path_for_card(card)

    # Skip if already exists on disk
    if os.path.exists(filepath):
        return card["product_id"], True, filepath

    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        resp = session.get(image_url, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return card["product_id"], True, filepath
        elif resp.status_code == 404:
            # No image available on CDN — try smaller size as fallback
            fallback_url = f"{config.TCGPLAYER_IMAGE_CDN}/{card['product_id']}_200w.jpg"
            resp2 = session.get(fallback_url, timeout=20)
            if resp2.status_code == 200 and len(resp2.content) > 1000:
                with open(filepath, "wb") as f:
                    f.write(resp2.content)
                return card["product_id"], True, filepath
    except Exception:
        pass
    return card["product_id"], False, ""


def download_images(limit: int = 0, retry_errors: bool = False):
    """Download images for all pending TCGPlayer Pokemon cards."""
    db.init_db()

    if retry_errors:
        conn = db.get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE tcgplayer_cards SET status='pending', error_msg=NULL WHERE status='error'")
        reset_count = cur.rowcount
        conn.commit()
        cur.close()
        db.put_connection(conn)
        if reset_count:
            console.print(f"[cyan]Reset {reset_count} errored cards back to pending[/cyan]")

    pending = db.get_tcgplayer_cards_by_status("pending", limit=limit)
    if not pending:
        console.print("[green]No pending TCGPlayer images to download.[/green]")
        return 0

    console.print(f"\n[bold]Downloading {len(pending)} TCGPlayer Pokemon card images[/bold]")
    console.print(f"[dim]Destination: {config.TCGPLAYER_IMAGE_DIR}[/dim]")
    console.print(f"[dim]Workers: {config.TCGPLAYER_DOWNLOAD_WORKERS}[/dim]\n")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    })
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=config.TCGPLAYER_DOWNLOAD_WORKERS,
        pool_maxsize=config.TCGPLAYER_DOWNLOAD_WORKERS,
        max_retries=config.MAX_RETRIES,
    )
    session.mount("https://", adapter)

    downloaded = 0
    errors = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console,
    ) as progress:
        task = progress.add_task("Downloading", total=len(pending))

        with ThreadPoolExecutor(max_workers=config.TCGPLAYER_DOWNLOAD_WORKERS) as pool:
            futures = {
                pool.submit(_download_one, card, session): card
                for card in pending
            }
            for future in as_completed(futures):
                product_id, ok, filepath = future.result()
                if ok:
                    db.tcgplayer_mark_downloaded(product_id, filepath)
                    downloaded += 1
                else:
                    db.tcgplayer_mark_error(product_id, "download_failed")
                    errors += 1
                progress.advance(task)

    console.print(f"\n[green]Downloaded {downloaded} images[/green]")
    if errors:
        console.print(f"[yellow]{errors} errors (retry with: UPDATE tcgplayer_cards SET status='pending' WHERE status='error')[/yellow]")

    db.log_event("tcgplayer_download", f"Downloaded {downloaded}, errors {errors}")
    return downloaded


# ---------------------------------------------------------------------------
# Stats & listing
# ---------------------------------------------------------------------------

def show_stats():
    """Display TCGPlayer scraping progress."""
    db.init_db()
    s = db.get_tcgplayer_stats()

    console.print("\n[bold]TCGPlayer Pokemon Scraper Stats[/bold]\n")
    console.print(f"  Sets:         [cyan]{s['sets']}[/cyan]")
    console.print(f"  Total cards:  [cyan]{s['total']}[/cyan]")
    console.print(f"  Downloaded:   [green]{s['downloaded']}[/green]")
    console.print(f"  Pending:      [yellow]{s['pending']}[/yellow]")
    console.print(f"  Errors:       [red]{s['error']}[/red]")
    console.print(f"  Skipped:      [dim]{s['skipped']}[/dim]")

    if s["total"] > 0:
        pct = (s["downloaded"] / s["total"]) * 100
        console.print(f"  Progress:     [green]{pct:.1f}%[/green]")

    # Disk usage
    if os.path.exists(config.TCGPLAYER_IMAGE_DIR):
        total_size = 0
        total_files = 0
        for root, dirs, files in os.walk(config.TCGPLAYER_IMAGE_DIR):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
                    total_files += 1
        console.print(f"  Images on disk: [dim]{total_files} files, {total_size / (1024**2):.1f} MB[/dim]")

    console.print(f"  Image dir:    [dim]{config.TCGPLAYER_IMAGE_DIR}[/dim]")
    console.print()


def list_sets():
    """List all available TCGPlayer Pokemon sets."""
    db.init_db()
    sets = db.get_tcgplayer_sets()

    if not sets:
        console.print("[yellow]No sets in DB yet. Run 'fetch' first.[/yellow]")
        return

    table = Table(title="TCGPlayer Pokemon Sets")
    table.add_column("Group ID", style="cyan", justify="right")
    table.add_column("Name", style="white")
    table.add_column("Abbr", style="dim")
    table.add_column("Cards", style="green", justify="right")

    for s in sets:
        table.add_row(
            str(s["group_id"]),
            s["name"],
            s.get("abbreviation") or "",
            str(s.get("card_count", 0)),
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(sets)} sets[/dim]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TCGPlayer Pokemon Scraper (via TCGCSV)")
    subparsers = parser.add_subparsers(dest="command")

    fetch_p = subparsers.add_parser("fetch", help="Fetch card metadata from TCGCSV")
    fetch_p.add_argument("--set", type=int, help="Only fetch a specific set (groupId, e.g. 1663)")

    dl_p = subparsers.add_parser("download", help="Download pending card images")
    dl_p.add_argument("--limit", type=int, default=0, help="Max images to download (0=all)")
    dl_p.add_argument("--retry-errors", action="store_true", help="Reset errored cards and retry them")

    subparsers.add_parser("stats", help="Show scraping progress")
    subparsers.add_parser("sets", help="List all available sets")

    run_p = subparsers.add_parser("run", help="Full pipeline: fetch + download")
    run_p.add_argument("--set", type=int, help="Only process a specific set (groupId)")
    run_p.add_argument("--limit", type=int, default=0, help="Max images to download")
    run_p.add_argument("--retry-errors", action="store_true", help="Reset errored cards and retry them")

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_cards(set_filter=args.set)
    elif args.command == "download":
        download_images(limit=args.limit, retry_errors=args.retry_errors)
    elif args.command == "stats":
        show_stats()
    elif args.command == "sets":
        list_sets()
    elif args.command == "run":
        console.print("\n[bold]===== TCGPLAYER POKEMON FULL PIPELINE =====[/bold]\n")
        fetch_cards(set_filter=args.set)
        download_images(limit=args.limit, retry_errors=args.retry_errors)
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
