#!/usr/bin/env python3
"""
SportsCardPro Scraper v3 - Optimized Version
Major improvements over v2:
- curl_cffi for Cloudflare bypass (no browser needed for most operations)
- CDN pattern discovery to eliminate Phase 4
- Adaptive rate limiting and session rotation
- Optional Redis task queue support
- Scrapling for stubborn Cloudflare challenges

Usage:
    python main_v3.py                    # Full optimized pipeline
    python main_v3.py --phase 1          # Discover sets (curl_cffi)
    python main_v3.py --phase 2          # Download CSVs (with auth)
    python main_v3.py --phase 3          # Parse CSVs (no browser)
    python main_v3.py --phase 4          # Smart image URL discovery
    python main_v3.py --phase 5          # Download images (curl_cffi)
    python main_v3.py --stats            # Show progress
    python main_v3.py --cdn-test         # Test CDN pattern discovery
    python main_v3.py --use-v2           # Fallback to v2 scraper
"""
import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def print_banner():
    console.print(Panel.fit("""
[bold blue]SportsCardPro Scraper v3 — Optimized Edition[/bold blue]
[green]✓ curl_cffi for Cloudflare bypass
✓ CDN pattern exploitation  
✓ Adaptive rate limiting
✓ Session rotation
✓ Redis queue support[/green]
    """, title="[bold]Scraper v3[/bold]"))


def check_dependencies():
    """Check if required packages are installed"""
    missing = []
    
    try:
        import curl_cffi
    except ImportError:
        missing.append("curl-cffi")
        
    try:
        import scrapling
        has_scrapling = True
    except ImportError:
        has_scrapling = False
        console.print("[yellow]Scrapling not installed - will use Playwright fallback for auth[/yellow]")
        
    try:
        import redis
        has_redis = True
    except ImportError:
        has_redis = False
        console.print("[yellow]Redis not installed - using PostgreSQL queue[/yellow]")
        
    if missing:
        console.print(f"[red]Missing required packages: {', '.join(missing)}[/red]")
        console.print("[yellow]Install with: pip install " + " ".join(missing) + "[/yellow]")
        return False
        
    return True


async def run_phase_v3(phase: int, sport: str = None, limit: int = 0):
    """Run a specific phase with v3 optimizations"""
    import database as db
    import scraper_v3
    
    db.init_db()
    
    if phase == 1:
        await scraper_v3.discover_sets_v3(sport)
    elif phase == 2:
        await scraper_v3.download_csvs_v3(sport)
    elif phase == 3:
        # Reuse v2 CSV parser
        from scraper import parse_csvs
        parse_csvs()
    elif phase == 4:
        await scraper_v3.scrape_card_images_v3(limit)
    elif phase == 5:
        await scraper_v3.download_images_v3(limit)
    else:
        console.print("[red]Invalid phase[/red]")


async def test_cdn_pattern():
    """Test CDN pattern discovery with sample cards"""
    import database as db
    from scraper_v3 import CDNPatternEngine
    
    db.init_db()
    
    console.print("[bold]Testing CDN Pattern Discovery[/bold]\n")
    
    # Get sample cards
    sample_cards = db.get_cards_needing_images(20)
    if not sample_cards:
        console.print("[red]No cards available for testing. Run phases 1-3 first.[/red]")
        return
        
    engine = CDNPatternEngine()
    pattern = await engine.test_cdn_patterns(sample_cards)
    
    if pattern:
        console.print(f"\n[bold green]Success! Found working CDN pattern:[/bold green]")
        console.print(f"[cyan]{pattern}[/cyan]")
        console.print("\n[green]This means Phase 4 can be eliminated entirely![/green]")
        console.print("[green]All image URLs can be generated without any web requests![/green]")
        
        # Estimate time savings
        total_cards = len(db.get_cards_needing_images(999999))
        saved_requests = total_cards
        saved_time = (saved_requests * 3) / 60  # 3 seconds per request average
        
        console.print(f"\n[bold]Impact:[/bold]")
        console.print(f"  • Cards needing images: {total_cards:,}")
        console.print(f"  • Web requests saved: {saved_requests:,}")
        console.print(f"  • Estimated time saved: {saved_time:.0f} minutes")
    else:
        console.print("\n[yellow]No direct CDN pattern found.[/yellow]")
        console.print("[yellow]Will need to scrape card pages for image URLs.[/yellow]")


def show_stats():
    """Enhanced stats display"""
    import database as db
    db.init_db()
    stats = db.get_stats()
    
    # Main stats table
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
        
    console.print(table)
    
    # Performance comparison
    if total > 0 and done > 0:
        console.print("\n[bold]Performance Comparison (estimated):[/bold]")
        v2_time = (total * 3) / 60  # 3 sec per card with Playwright
        v3_time = (total * 0.5) / 60  # 0.5 sec per card with curl_cffi
        
        perf_table = Table(show_header=True, header_style="bold magenta")
        perf_table.add_column("Method", style="white")
        perf_table.add_column("Time", justify="right", style="yellow")
        perf_table.add_column("Improvement", justify="right", style="green")
        
        perf_table.add_row("Scraper v2 (Playwright)", f"{v2_time:.0f} min", "baseline")
        perf_table.add_row("Scraper v3 (curl_cffi)", f"{v3_time:.0f} min", f"{(v2_time/v3_time):.1f}x faster")
        
        if stats["cards_pending"] == 0:  # CDN pattern worked
            perf_table.add_row("CDN Pattern", "instant", "∞x faster")
            
        console.print(perf_table)


def main():
    parser = argparse.ArgumentParser(description="SportsCardPro Scraper v3 - Optimized")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run specific phase")
    parser.add_argument("--sport", type=str,
                        choices=["baseball", "basketball", "football", "hockey",
                                 "racing", "soccer", "wrestling", "ufc"],
                        help="Limit to one sport")
    parser.add_argument("--limit", type=int, default=0, help="Max items to process")
    parser.add_argument("--stats", action="store_true", help="Show progress stats")
    parser.add_argument("--cdn-test", action="store_true", help="Test CDN pattern discovery")
    parser.add_argument("--use-v2", action="store_true", help="Use v2 scraper (fallback)")
    parser.add_argument("--reset-errors", action="store_true", help="Reset errors to retry")
    parser.add_argument("--headed", action="store_true", help="Show browser (when needed)")
    
    args = parser.parse_args()
    
    if not args.use_v2:
        print_banner()
        
    if args.stats:
        show_stats()
        return
        
    if args.cdn_test:
        asyncio.run(test_cdn_pattern())
        return
        
    if args.reset_errors:
        import database as db
        db.init_db()
        db.reset_errors()
        console.print("[green]Errors reset to pending.[/green]")
        show_stats()
        return
        
    if args.headed:
        import config
        config.HEADLESS = False
        
    if args.use_v2:
        # Fall back to v2 scraper
        console.print("[yellow]Using v2 scraper (Playwright)[/yellow]")
        from main import main as main_v2
        sys.argv = [sys.argv[0]] + [arg for arg in sys.argv[1:] if arg != "--use-v2"]
        main_v2()
        return
        
    # Check dependencies for v3
    if not check_dependencies():
        console.print("\n[yellow]Falling back to v2 scraper...[/yellow]")
        from main import main as main_v2
        main_v2()
        return
        
    if args.phase:
        console.print(f"[bold]Running phase {args.phase} (optimized)[/bold]")
        asyncio.run(run_phase_v3(args.phase, args.sport, args.limit))
    else:
        # Run full pipeline
        import scraper_v3
        asyncio.run(scraper_v3.run_full_pipeline_v3(args.sport, args.limit))
        
    show_stats()


if __name__ == "__main__":
    main()