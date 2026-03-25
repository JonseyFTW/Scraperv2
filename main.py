#!/usr/bin/env python3
"""
SportsCardPro Scraper v2 - CSV-First Approach

Usage:
    python main.py                              # Full pipeline
    python main.py --sport baseball             # One sport only
    python main.py --phase 1                    # Discover sets only
    python main.py --phase 2                    # Download CSVs only
    python main.py --phase 3                    # Parse CSVs only (no browser)
    python main.py --phase 4                    # Scrape card pages for images
    python main.py --phase 5                    # Download images only (no browser)
    python main.py --stats                      # Show progress
    python main.py --reset-errors               # Retry failed items
    python main.py --headed                     # Show browser window
    python main.py --limit 100                  # Process max N cards (phase 4/5)

Environment variables:
    SCP_EMAIL       Your SportsCardsPro login email
    SCP_PASSWORD    Your SportsCardsPro password
"""
import argparse
import asyncio

from rich.console import Console
from rich.table import Table

console = Console()


def print_banner():
    console.print("""
[bold blue]╔══════════════════════════════════════════════════════╗
║       SportsCardPro Scraper v2 — CSV-First           ║
║  Discover → CSV Download → Parse → Images → Done     ║
╚══════════════════════════════════════════════════════╝[/bold blue]
""")


def show_stats():
    import database as db
    db.init_db()
    stats = db.get_stats()

    table = Table(title="Scrape Progress", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="white", min_width=25)
    table.add_column("Count", justify="right", style="green", min_width=10)

    table.add_section()
    table.add_row("Sets (total)", str(stats["total_sets"]))
    table.add_row("  CSV pending", str(stats["sets_csv_pending"]))
    table.add_row("  CSV downloaded", str(stats["sets_csv_downloaded"]))
    table.add_row("  CSV parsed", str(stats["sets_csv_parsed"]))
    table.add_row("  CSV errors", str(stats["sets_csv_error"]))

    table.add_section()
    table.add_row("Cards (total)", str(stats["total_cards"]))
    table.add_row("  Pending image scrape", str(stats["cards_pending"]))
    table.add_row("  Image URL found", str(stats["cards_image_found"]))
    table.add_row("  Downloaded", str(stats["cards_downloaded"]))
    table.add_row("  No image available", str(stats["cards_no_image"]))
    table.add_row("  Errors", str(stats["cards_error"]))

    total = stats["total_cards"]
    if total > 0:
        done = stats["cards_downloaded"]
        pct = (done / total) * 100
        table.add_section()
        table.add_row("Image completion", f"{pct:.1f}%")

    # Per-sport breakdown
    sport_keys = [k for k in stats if k.startswith("cards_") and k not in (
        "cards_pending", "cards_image_found", "cards_downloaded", "cards_no_image", "cards_error"
    )]
    if sport_keys:
        table.add_section()
        for k in sorted(sport_keys):
            sport = k.replace("cards_", "")
            table.add_row(f"  {sport} cards", str(stats[k]))

    console.print(table)


async def run_phase(phase: int, sport: str = None, limit: int = 0):
    import database as db
    import scraper

    db.init_db()

    if phase == 3:
        # No browser needed
        scraper.parse_csvs()
        return

    if phase == 5:
        # No browser needed
        await scraper.download_images(limit)
        return

    from playwright.async_api import async_playwright
    async with async_playwright() as pw:
        browser, context = await scraper.create_browser(pw)
        page = await scraper.new_stealth_page(context)
        try:
            if phase in (2, 4):
                await scraper.login(page)

            if phase == 1:
                await scraper.discover_sets(page, sport)
            elif phase == 2:
                await scraper.download_csvs(page, sport)
            elif phase == 4:
                await scraper.scrape_card_images(page, limit)
        finally:
            await context.close()
            await browser.close()


def main():
    parser = argparse.ArgumentParser(description="SportsCardPro Scraper v2")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="1=discover sets, 2=download CSVs, 3=parse CSVs, 4=scrape images, 5=download images")
    parser.add_argument("--sport", type=str,
                        choices=["baseball", "basketball", "football", "hockey",
                                 "racing", "soccer", "wrestling", "ufc"],
                        help="Limit to one sport")
    parser.add_argument("--limit", type=int, default=0, help="Max cards to process")
    parser.add_argument("--stats", action="store_true", help="Show progress")
    parser.add_argument("--reset-errors", action="store_true", help="Reset errors to retry")
    parser.add_argument("--reset-no-image", action="store_true", help="Reset 'no image' cards to retry")
    parser.add_argument("--failures", action="store_true", help="Show image failure breakdown")
    parser.add_argument("--headed", action="store_true", help="Show browser window")
    parser.add_argument("--proxy", type=str, default="",
                        help="SOCKS5 proxy URL, e.g. socks5://user:pass@us5580.nordvpn.com:1080")

    args = parser.parse_args()
    print_banner()

    if args.stats:
        show_stats()
        return

    if args.proxy:
        import config as cfg
        cfg.PROXY_URL = args.proxy

    if args.reset_errors:
        import database as db
        db.init_db()
        db.reset_errors()
        console.print("[green]Errors reset to pending.[/green]")
        show_stats()
        return

    if args.reset_no_image:
        import database as db
        db.init_db()
        count = db.reset_no_image()
        console.print(f"[green]Reset {count} 'no image' cards back to pending for retry.[/green]")
        show_stats()
        return

    if args.failures:
        import database as db
        db.init_db()
        stats = db.get_image_failure_stats()
        if not stats:
            console.print("[green]No failures found.[/green]")
        else:
            table = Table(title="Image Failure Breakdown", show_header=True, header_style="bold cyan")
            table.add_column("Reason", style="white", min_width=30)
            table.add_column("Count", justify="right", style="yellow", min_width=10)
            for reason, count in stats.items():
                table.add_row(reason, str(count))
            console.print(table)
        return

    if args.headed:
        import config as cfg
        cfg.HEADLESS = False

    if args.phase:
        console.print(f"Running phase {args.phase}")
        asyncio.run(run_phase(args.phase, args.sport, args.limit))
    else:
        import scraper
        asyncio.run(scraper.run_full_pipeline(args.sport, args.limit))

    show_stats()


if __name__ == "__main__":
    main()
