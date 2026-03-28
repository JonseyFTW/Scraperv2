#!/usr/bin/env python3
"""
LXC Container Card Stats
Shows how many cards each container has processed/is processing.

Usage:
    python lxc_stats.py              # Show per-container card counts
    python lxc_stats.py --monitor    # Live refresh every 10s
    python lxc_stats.py --json       # JSON output
"""
import argparse
import json
import os
import time
from datetime import datetime

import psycopg2
import psycopg2.extras
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:changeme@192.168.1.14:5433/sportscards",
)

# Track previous snapshot for rate calculation
_prev_snapshot = {"time": None, "done": None, "per_worker": {}}


def get_worker_stats():
    """Get card counts grouped by worker/container."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Per-worker breakdown
    cur.execute("""
        SELECT
            COALESCE(worker_id, 'unassigned') AS worker,
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
            SUM(CASE WHEN status = 'image_found' THEN 1 ELSE 0 END) AS image_found,
            SUM(CASE WHEN status = 'downloading' THEN 1 ELSE 0 END) AS downloading,
            SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
            SUM(CASE WHEN status = 'no_image' THEN 1 ELSE 0 END) AS no_image
        FROM cards
        GROUP BY worker_id
        ORDER BY total DESC
    """)
    workers = [dict(r) for r in cur.fetchall()]

    # Overall totals
    cur.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
            SUM(CASE WHEN status = 'downloading' THEN 1 ELSE 0 END) AS downloading,
            SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN status = 'image_found' THEN 1 ELSE 0 END) AS image_found,
            SUM(CASE WHEN status = 'no_image' THEN 1 ELSE 0 END) AS no_image,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
        FROM cards
    """)
    totals = dict(cur.fetchone())

    cur.close()
    conn.close()
    return workers, totals


def calc_rates(workers, totals):
    """Calculate cards/sec and ETA based on previous snapshot."""
    global _prev_snapshot
    now = time.time()

    # Count all processed cards (anything no longer pending/processing)
    total_cards = totals.get("total", 0) or 0
    pending = (totals.get("pending", 0) or 0)
    processing = (totals.get("processing", 0) or 0)
    done_now = total_cards - pending - processing

    # Per-worker done counts
    worker_done = {}
    for w in workers:
        name = w["worker"]
        worker_done[name] = (w["downloaded"] or 0) + (w["image_found"] or 0) + (w["errors"] or 0) + (w["no_image"] or 0)

    overall_rate = None
    worker_rates = {}
    eta_str = None

    if _prev_snapshot["time"] is not None:
        elapsed = now - _prev_snapshot["time"]
        if elapsed > 0:
            delta = done_now - _prev_snapshot["done"]
            overall_rate = delta / elapsed

            # Per-worker rates
            for name, done in worker_done.items():
                prev_done = _prev_snapshot["per_worker"].get(name, 0)
                w_delta = done - prev_done
                worker_rates[name] = w_delta / elapsed

            # ETA based on pending + processing
            remaining = (totals.get("pending", 0) or 0) + (totals.get("processing", 0) or 0)
            if overall_rate > 0:
                eta_seconds = remaining / overall_rate
                eta_str = format_duration(eta_seconds)

    # Save snapshot
    _prev_snapshot = {"time": now, "done": done_now, "per_worker": worker_done}

    return overall_rate, worker_rates, eta_str


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds < 0:
        return "-"
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    mins = int((seconds % 3600) // 60)
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    elif hours > 0:
        return f"{hours}h {mins}m"
    elif mins > 0:
        return f"{mins}m"
    return "<1m"


def display_stats():
    """Show per-container card stats."""
    workers, totals = get_worker_stats()
    overall_rate, worker_rates, eta_str = calc_rates(workers, totals)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Worker table
    table = Table(
        title=f"Cards per Container  ({now})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Container", style="bold white", min_width=20)
    table.add_column("Total", justify="right", min_width=8)
    table.add_column("Processing", justify="right", style="yellow", min_width=10)
    table.add_column("Downloading", justify="right", style="blue", min_width=11)
    table.add_column("Downloaded", justify="right", style="green", min_width=10)
    table.add_column("Image Found", justify="right", min_width=11)
    table.add_column("Errors", justify="right", style="red", min_width=8)
    table.add_column("No Image", justify="right", style="dim", min_width=8)
    table.add_column("Cards/s", justify="right", style="magenta", min_width=8)

    for w in workers:
        name = w["worker"]
        if name == "unassigned":
            name_text = Text("(unassigned)", style="dim italic")
        else:
            name_text = Text(name, style="bold white")

        # Active indicator
        active = (w["processing"] or 0) + (w["downloading"] or 0)
        if active > 0:
            name_text.append("  ACTIVE", style="bold green")

        # Per-worker rate
        rate = worker_rates.get(name)
        rate_str = f"{rate:.1f}" if rate is not None else "-"

        table.add_row(
            name_text,
            f"{w['total']:,}",
            f"{w['processing'] or 0:,}",
            f"{w['downloading'] or 0:,}",
            f"{w['downloaded'] or 0:,}",
            f"{w['image_found'] or 0:,}",
            f"{w['errors'] or 0:,}",
            f"{w['no_image'] or 0:,}",
            rate_str,
        )

    # Totals row
    table.add_section()
    overall_rate_str = f"{overall_rate:.1f}" if overall_rate is not None else "-"
    table.add_row(
        Text("TOTAL", style="bold"),
        f"{totals['total']:,}",
        f"{totals['processing'] or 0:,}",
        f"{totals['downloading'] or 0:,}",
        f"{totals['downloaded'] or 0:,}",
        f"{totals['image_found'] or 0:,}",
        f"{totals['errors'] or 0:,}",
        f"{totals['no_image'] or 0:,}",
        overall_rate_str,
    )

    console.print(table)

    # Summary line
    pending = totals.get("pending", 0) or 0
    processing = totals.get("processing", 0) or 0
    downloading = totals.get("downloading", 0) or 0
    done = (totals.get("downloaded", 0) or 0) + (totals.get("image_found", 0) or 0) + (totals.get("no_image", 0) or 0)
    total = totals["total"]
    pct = (done / total * 100) if total > 0 else 0

    active_workers = sum(
        1 for w in workers
        if w["worker"] != "unassigned" and ((w["processing"] or 0) + (w["downloading"] or 0)) > 0
    )
    total_workers = sum(1 for w in workers if w["worker"] != "unassigned")

    summary = (
        f"\n  Workers: [green]{active_workers}[/green]/{total_workers} active  |  "
        f"Pending: [yellow]{pending:,}[/yellow]  |  "
        f"In-flight: [blue]{processing + downloading:,}[/blue]  |  "
        f"Done: [green]{done:,}[/green] ({pct:.1f}%)"
    )

    if overall_rate is not None:
        summary += f"  |  Rate: [magenta]{overall_rate:.1f} cards/s[/magenta]"
    if eta_str:
        summary += f"  |  ETA: [cyan]{eta_str}[/cyan]"

    console.print(summary)


def monitor_mode(interval=10):
    """Live monitoring with periodic refresh."""
    console.print("[bold]Live monitor[/bold] — Ctrl+C to stop\n")

    # First snapshot (no rate yet)
    console.clear()
    display_stats()
    console.print(f"\n[dim]Calculating rate... refreshing in {interval}s[/dim]")
    time.sleep(interval)

    try:
        while True:
            console.clear()
            display_stats()
            console.print(f"\n[dim]Refreshing every {interval}s... (Ctrl+C to stop)[/dim]")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopped.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Per-container card stats")
    parser.add_argument("--monitor", action="store_true", help="Live refresh mode")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval (default: 10s)")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    if args.json:
        workers, totals = get_worker_stats()
        print(json.dumps({"workers": workers, "totals": totals}, indent=2, default=str))
        return

    if args.monitor:
        monitor_mode(args.interval)
    else:
        display_stats()


if __name__ == "__main__":
    main()
