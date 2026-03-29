#!/usr/bin/env python3
"""
Pokemon TCG Scraper — integrated into SportsCardPro Scraper v2

Fetches all Pokemon TCG card metadata from TCGdex (free, no API key) and
downloads high-resolution card images.  Card metadata is stored in PostgreSQL
(pokemon_sets / pokemon_cards tables) alongside your sports card data.

Usage:
    python pokemon_scraper.py fetch                # Fetch metadata from TCGdex
    python pokemon_scraper.py fetch --set base1    # Fetch one set only
    python pokemon_scraper.py download             # Download all pending images
    python pokemon_scraper.py download --limit 500 # Download up to 500 images
    python pokemon_scraper.py stats                # Show progress
    python pokemon_scraper.py run                  # Full pipeline: fetch + download
    python pokemon_scraper.py run --limit 500      # Full pipeline, limited

Requirements:
    pip install requests rich psycopg2-binary
"""
import argparse
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
    """Build local image path: POKEMON_IMAGE_DIR/Set Name/cardid.png"""
    safe_id = card["id"].replace("/", "-")
    folder = _safe_folder_name(card.get("set_name") or "unknown")
    return os.path.join(config.POKEMON_IMAGE_DIR, folder, f"{safe_id}.{config.POKEMON_IMAGE_FORMAT}")


# ---------------------------------------------------------------------------
# Fetch metadata from TCGdex
# ---------------------------------------------------------------------------

def fetch_sets(set_filter: str | None = None) -> list[dict]:
    """Fetch all Pokemon TCG sets from TCGdex and upsert into PostgreSQL."""
    console.print("[cyan]Fetching sets from TCGdex...[/cyan]")

    resp = requests.get(f"{config.TCGDEX_BASE}/sets", timeout=config.IMAGE_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    sets = resp.json()

    if set_filter:
        sets = [s for s in sets if s["id"] == set_filter]
        if not sets:
            console.print(f"[red]Set '{set_filter}' not found[/red]")
            return []

    console.print(f"Found [cyan]{len(sets)}[/cyan] sets")
    return sets


def fetch_cards(set_filter: str | None = None):
    """Fetch all card metadata from TCGdex (one API call per set) and store in PostgreSQL."""
    db.init_db()
    sets = fetch_sets(set_filter)
    if not sets:
        return 0

    console.print(f"\n[bold]Fetching card details for {len(sets)} sets...[/bold]\n")

    total_cards = 0
    failed_sets = []

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console,
    ) as progress:
        task = progress.add_task("Fetching sets", total=len(sets))

        for s in sets:
            set_id = s["id"]
            set_name = s.get("name", set_id)
            progress.update(task, description=f"{set_name[:40]}")

            try:
                set_resp = requests.get(
                    f"{config.TCGDEX_BASE}/sets/{set_id}",
                    timeout=config.IMAGE_DOWNLOAD_TIMEOUT,
                )
                set_resp.raise_for_status()
                set_data = set_resp.json()

                # Upsert the set itself
                db.upsert_pokemon_set({
                    "id": set_id,
                    "name": set_name,
                    "series": set_data.get("serie", {}).get("name", ""),
                    "total": set_data.get("cardCount", {}).get("total", 0),
                    "release_date": set_data.get("releaseDate", ""),
                    "images_symbol": (set_data.get("logo", "") or ""),
                    "images_logo": (set_data.get("symbol", "") or ""),
                })

                # Build card records
                cards_batch = []
                for card in set_data.get("cards", []):
                    cards_batch.append({
                        "id": card["id"],
                        "name": card.get("name", "Unknown"),
                        "local_id": card.get("localId", ""),
                        "set_id": set_id,
                        "set_name": set_name,
                        "category": card.get("category", ""),
                        "image_url": card.get("image", ""),
                        "status": "pending",
                    })

                if cards_batch:
                    db.upsert_pokemon_cards_bulk(cards_batch)
                    total_cards += len(cards_batch)

            except Exception as e:
                failed_sets.append((set_id, str(e)))

            progress.advance(task)
            time.sleep(config.POKEMON_REQUEST_DELAY)

    console.print(f"\n[green]Fetched {total_cards} cards from {len(sets)} sets[/green]")
    if failed_sets:
        console.print(f"[yellow]{len(failed_sets)} sets failed:[/yellow]")
        for sid, err in failed_sets[:5]:
            console.print(f"  [dim]{sid}: {err}[/dim]")

    db.log_event("pokemon_fetch", f"Fetched {total_cards} cards from {len(sets)} sets")
    return total_cards


# ---------------------------------------------------------------------------
# Download images
# ---------------------------------------------------------------------------

def _download_one(card: dict, session: requests.Session) -> tuple[str, bool, str]:
    """Download a single card image. Returns (card_id, success, filepath)."""
    image_url = card.get("image_url")
    if not image_url:
        return card["id"], False, ""

    filepath = _image_path_for_card(card)

    if os.path.exists(filepath):
        return card["id"], True, filepath

    url = f"{image_url}/{config.POKEMON_IMAGE_QUALITY}.{config.POKEMON_IMAGE_FORMAT}"
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        resp = session.get(url, timeout=20)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            return card["id"], True, filepath
    except Exception:
        pass
    return card["id"], False, ""


def download_images(limit: int = 0):
    """Download images for all pending Pokemon cards using parallel workers."""
    db.init_db()

    pending = db.get_pokemon_cards_by_status("pending", limit=limit)
    if not pending:
        console.print("[green]No pending Pokemon images to download.[/green]")
        return 0

    console.print(f"\n[bold]Downloading {len(pending)} Pokemon card images[/bold]")
    console.print(f"[dim]Destination: {config.POKEMON_IMAGE_DIR}[/dim]")
    console.print(f"[dim]Workers: {config.POKEMON_DOWNLOAD_WORKERS}[/dim]\n")

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=config.POKEMON_DOWNLOAD_WORKERS,
        pool_maxsize=config.POKEMON_DOWNLOAD_WORKERS,
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

        with ThreadPoolExecutor(max_workers=config.POKEMON_DOWNLOAD_WORKERS) as pool:
            futures = {
                pool.submit(_download_one, card, session): card
                for card in pending
            }
            for future in as_completed(futures):
                card_id, ok, filepath = future.result()
                if ok:
                    db.pokemon_mark_downloaded(card_id, filepath)
                    downloaded += 1
                else:
                    db.pokemon_mark_error(card_id, "download_failed")
                    errors += 1
                progress.advance(task)

    console.print(f"\n[green]Downloaded {downloaded} images[/green]")
    if errors:
        console.print(f"[yellow]{errors} errors (retry with: UPDATE pokemon_cards SET status='pending' WHERE status='error')[/yellow]")

    db.log_event("pokemon_download", f"Downloaded {downloaded}, errors {errors}")
    return downloaded


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def show_stats():
    """Display Pokemon scraping progress."""
    db.init_db()
    s = db.get_pokemon_stats()

    console.print("\n[bold]Pokemon TCG Scraper Stats[/bold]\n")
    console.print(f"  Sets:         [cyan]{s['sets']}[/cyan]")
    console.print(f"  Total cards:  [cyan]{s['total']}[/cyan]")
    console.print(f"  Downloaded:   [green]{s['downloaded']}[/green]")
    console.print(f"  Pending:      [yellow]{s['pending']}[/yellow]")
    console.print(f"  Errors:       [red]{s['error']}[/red]")

    if s["total"] > 0:
        pct = (s["downloaded"] / s["total"]) * 100
        console.print(f"  Progress:     [green]{pct:.1f}%[/green]")

    # Disk usage
    if os.path.exists(config.POKEMON_IMAGE_DIR):
        total_size = 0
        total_files = 0
        for root, dirs, files in os.walk(config.POKEMON_IMAGE_DIR):
            for f in files:
                fp = os.path.join(root, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
                    total_files += 1
        console.print(f"  Images on disk: [dim]{total_files} files, {total_size / (1024**2):.1f} MB[/dim]")

    console.print(f"  Image dir:    [dim]{config.POKEMON_IMAGE_DIR}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pokemon TCG Scraper (TCGdex)")
    subparsers = parser.add_subparsers(dest="command")

    fetch_p = subparsers.add_parser("fetch", help="Fetch card metadata from TCGdex")
    fetch_p.add_argument("--set", type=str, help="Only fetch a specific set (e.g. base1)")

    dl_p = subparsers.add_parser("download", help="Download pending card images")
    dl_p.add_argument("--limit", type=int, default=0, help="Max images to download (0=all)")

    subparsers.add_parser("stats", help="Show scraping progress")

    run_p = subparsers.add_parser("run", help="Full pipeline: fetch + download")
    run_p.add_argument("--set", type=str, help="Only process a specific set")
    run_p.add_argument("--limit", type=int, default=0, help="Max images to download")

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_cards(set_filter=args.set)
    elif args.command == "download":
        download_images(limit=args.limit)
    elif args.command == "stats":
        show_stats()
    elif args.command == "run":
        console.print("\n[bold]===== POKEMON TCG FULL PIPELINE =====[/bold]\n")
        fetch_cards(set_filter=args.set)
        download_images(limit=args.limit)
        show_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
