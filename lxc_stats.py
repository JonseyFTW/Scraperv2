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
import sys
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
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors
        FROM cards
    """)
    totals = dict(cur.fetchone())

    cur.close()
    conn.close()
    return workers, totals


def display_stats():
    """Show per-container card stats."""
    workers, totals = get_worker_stats()
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

        table.add_row(
            name_text,
            f"{w['total']:,}",
            f"{w['processing'] or 0:,}",
            f"{w['downloading'] or 0:,}",
            f"{w['downloaded'] or 0:,}",
            f"{w['image_found'] or 0:,}",
            f"{w['errors'] or 0:,}",
            f"{w['no_image'] or 0:,}",
        )

    # Totals row
    table.add_section()
    table.add_row(
        Text("TOTAL", style="bold"),
        f"{totals['total']:,}",
        f"{totals['processing'] or 0:,}",
        f"{totals['downloading'] or 0:,}",
        f"{totals['downloaded'] or 0:,}",
        "",
        f"{totals['errors'] or 0:,}",
        "",
    )

    console.print(table)

    # Summary line
    pending = totals.get("pending", 0) or 0
    processing = totals.get("processing", 0) or 0
    downloading = totals.get("downloading", 0) or 0
    done = totals.get("downloaded", 0) or 0
    total = totals["total"]
    pct = (done / total * 100) if total > 0 else 0

    active_workers = sum(
        1 for w in workers
        if w["worker"] != "unassigned" and ((w["processing"] or 0) + (w["downloading"] or 0)) > 0
    )
    total_workers = sum(1 for w in workers if w["worker"] != "unassigned")

    console.print(
        f"\n  Workers: [green]{active_workers}[/green]/{total_workers} active  |  "
        f"Pending: [yellow]{pending:,}[/yellow]  |  "
        f"In-flight: [blue]{processing + downloading:,}[/blue]  |  "
        f"Done: [green]{done:,}[/green] ({pct:.1f}%)"
    )


def monitor_mode(interval=10):
    """Live monitoring with periodic refresh."""
    console.print("[bold]Live monitor[/bold] — Ctrl+C to stop\n")
    try:
        while True:
            console.clear()
            display_stats()
            console.print(f"\n[dim]Refreshing every {interval}s...[/dim]")
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
