#!/usr/bin/env python3
"""
LXC Container Stats Monitor
Shows Proxmox LXC container status, resource usage, and scraping progress.

Usage:
    python lxc_stats.py                    # Show all container stats
    python lxc_stats.py --monitor          # Live monitoring (refreshes every 10s)
    python lxc_stats.py --proxmox-host IP  # Override Proxmox host
    python lxc_stats.py --db-only          # Only show database stats (no Proxmox)

Environment variables:
    PROXMOX_HOST     - Proxmox host IP/hostname (default: 192.168.1.14)
    PROXMOX_USER     - API user (default: root@pam)
    PROXMOX_PASSWORD - API password
    PROXMOX_TOKEN_NAME - API token name (alternative to password)
    PROXMOX_TOKEN_VALUE - API token value
    DATABASE_URL     - PostgreSQL connection string
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import psycopg2
import psycopg2.extras
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text

console = Console()

# ── Configuration ────────────────────────────────────────────────────────
PROXMOX_HOST = os.environ.get("PROXMOX_HOST", "192.168.1.14")
PROXMOX_PORT = int(os.environ.get("PROXMOX_PORT", "8006"))
PROXMOX_USER = os.environ.get("PROXMOX_USER", "root@pam")
PROXMOX_PASSWORD = os.environ.get("PROXMOX_PASSWORD", "")
PROXMOX_TOKEN_NAME = os.environ.get("PROXMOX_TOKEN_NAME", "")
PROXMOX_TOKEN_VALUE = os.environ.get("PROXMOX_TOKEN_VALUE", "")
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:changeme@192.168.1.14:5433/sportscards",
)
PROXMOX_NODE = os.environ.get("PROXMOX_NODE", "")  # Auto-detected if empty


# ── Proxmox API Client ──────────────────────────────────────────────────
class ProxmoxClient:
    """Lightweight Proxmox API client using urllib (no extra dependencies)."""

    def __init__(self, host, port=8006, user="root@pam", password="",
                 token_name="", token_value="", verify_ssl=False):
        self.base_url = f"https://{host}:{port}/api2/json"
        self.user = user
        self.password = password
        self.token_name = token_name
        self.token_value = token_value
        self.verify_ssl = verify_ssl
        self.ticket = None
        self.csrf_token = None
        self._node = None

        import ssl
        import urllib.request
        self._ssl_ctx = ssl.create_default_context()
        if not verify_ssl:
            self._ssl_ctx.check_hostname = False
            self._ssl_ctx.verify_mode = ssl.CERT_NONE
        self._opener = urllib.request.build_opener(
            urllib.request.HTTPSHandler(context=self._ssl_ctx)
        )

    def _request(self, method, path, data=None):
        import urllib.request
        import urllib.parse

        url = f"{self.base_url}{path}"
        headers = {}

        if self.token_name and self.token_value:
            headers["Authorization"] = (
                f"PVEAPIToken={self.user}!{self.token_name}={self.token_value}"
            )
        elif self.ticket:
            headers["Cookie"] = f"PVEAuthCookie={self.ticket}"
            if method != "GET":
                headers["CSRFPreventionToken"] = self.csrf_token

        body = None
        if data:
            body = urllib.parse.urlencode(data).encode()

        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = self._opener.open(req, timeout=10)
            return json.loads(resp.read().decode())["data"]
        except Exception as e:
            raise ConnectionError(f"Proxmox API error: {e}")

    def authenticate(self):
        """Authenticate with password to get a ticket."""
        if self.token_name and self.token_value:
            return True
        if not self.password:
            raise ValueError(
                "No Proxmox credentials. Set PROXMOX_PASSWORD or "
                "PROXMOX_TOKEN_NAME + PROXMOX_TOKEN_VALUE environment variables."
            )
        result = self._request("POST", "/access/ticket", {
            "username": self.user,
            "password": self.password,
        })
        self.ticket = result["ticket"]
        self.csrf_token = result["CSRFPreventionToken"]
        return True

    def get_node(self):
        """Get the first (or configured) node name."""
        if self._node:
            return self._node
        nodes = self._request("GET", "/nodes")
        if PROXMOX_NODE:
            self._node = PROXMOX_NODE
        else:
            self._node = nodes[0]["node"]
        return self._node

    def get_containers(self):
        """Get all LXC containers with status and resource usage."""
        node = self.get_node()
        return self._request("GET", f"/nodes/{node}/lxc")

    def get_container_status(self, vmid):
        """Get detailed status for a specific container."""
        node = self.get_node()
        return self._request("GET", f"/nodes/{node}/lxc/{vmid}/status/current")

    def get_container_config(self, vmid):
        """Get container configuration."""
        node = self.get_node()
        return self._request("GET", f"/nodes/{node}/lxc/{vmid}/config")


# ── Local pct fallback ───────────────────────────────────────────────────
def get_containers_via_pct():
    """Fallback: get container info using local pct commands (on Proxmox host)."""
    try:
        result = subprocess.run(
            ["pct", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    containers = []
    for line in result.stdout.strip().split("\n")[1:]:  # Skip header
        parts = line.split()
        if len(parts) >= 3:
            vmid = parts[0]
            status = parts[1]
            name = parts[2] if len(parts) > 2 else ""

            info = {
                "vmid": vmid,
                "name": name,
                "status": status.lower(),
                "cpu": 0,
                "mem": 0,
                "maxmem": 0,
                "disk": 0,
                "maxdisk": 0,
                "netin": 0,
                "netout": 0,
                "uptime": 0,
                "cpus": 0,
            }

            # Get detailed stats if running
            if status.lower() == "running":
                try:
                    detail = subprocess.run(
                        ["pct", "status", vmid, "--verbose"],
                        capture_output=True, text=True, timeout=10,
                    )
                    for dline in detail.stdout.strip().split("\n"):
                        if ":" in dline:
                            key, val = dline.split(":", 1)
                            key = key.strip().lower()
                            val = val.strip()
                            if key == "cpu(s)":
                                info["cpus"] = int(val)
                except (subprocess.TimeoutExpired, ValueError):
                    pass

            containers.append(info)

    return containers


# ── Database Stats ───────────────────────────────────────────────────────
def get_db_stats():
    """Get comprehensive scraping stats from the database."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    except Exception as e:
        return {"error": str(e)}

    stats = {}

    # Overall card counts by status
    cur.execute("""
        SELECT status, COUNT(*) as count
        FROM cards
        GROUP BY status
        ORDER BY status
    """)
    stats["card_statuses"] = {row["status"]: row["count"] for row in cur.fetchall()}

    # Total cards
    cur.execute("SELECT COUNT(*) as c FROM cards")
    stats["total_cards"] = cur.fetchone()["c"]

    # Currently processing (active work)
    stats["processing"] = stats["card_statuses"].get("processing", 0)
    stats["downloading"] = stats["card_statuses"].get("downloading", 0)

    # Per-sport breakdown
    cur.execute("""
        SELECT s.sport, c.status, COUNT(*) as count
        FROM cards c
        JOIN sets s ON c.set_slug = s.slug
        GROUP BY s.sport, c.status
        ORDER BY s.sport, c.status
    """)
    sport_stats = {}
    for row in cur.fetchall():
        sport = row["sport"]
        if sport not in sport_stats:
            sport_stats[sport] = {}
        sport_stats[sport][row["status"]] = row["count"]
    stats["by_sport"] = sport_stats

    # Set stats
    cur.execute("""
        SELECT csv_status, COUNT(*) as count
        FROM sets
        GROUP BY csv_status
    """)
    stats["set_statuses"] = {row["csv_status"]: row["count"] for row in cur.fetchall()}

    # Recent activity (last 5 minutes)
    cur.execute("""
        SELECT COUNT(*) as c
        FROM scrape_log
        WHERE timestamp > (NOW() - INTERVAL '5 minutes')::text
    """)
    stats["recent_events"] = cur.fetchone()["c"]

    # Cards per sport totals
    cur.execute("""
        SELECT s.sport, COUNT(*) as total,
               SUM(CASE WHEN c.status = 'downloaded' THEN 1 ELSE 0 END) as done,
               SUM(CASE WHEN c.status = 'processing' THEN 1 ELSE 0 END) as active,
               SUM(CASE WHEN c.status = 'pending' THEN 1 ELSE 0 END) as pending,
               SUM(CASE WHEN c.status = 'error' THEN 1 ELSE 0 END) as errors
        FROM cards c
        JOIN sets s ON c.set_slug = s.slug
        GROUP BY s.sport
        ORDER BY s.sport
    """)
    stats["sport_summary"] = [dict(r) for r in cur.fetchall()]

    cur.close()
    conn.close()
    return stats


# ── Display Functions ────────────────────────────────────────────────────
def format_bytes(n):
    """Format bytes to human-readable."""
    if n is None or n == 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def format_uptime(seconds):
    """Format seconds to human-readable uptime."""
    if not seconds:
        return "-"
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    mins = (seconds % 3600) // 60
    if days > 0:
        return f"{days}d {hours}h {mins}m"
    elif hours > 0:
        return f"{hours}h {mins}m"
    return f"{mins}m"


def is_scraper_container(name):
    """Check if a container name matches scraper naming convention."""
    name_lower = (name or "").lower()
    return any(kw in name_lower for kw in ("scraper", "scp", "sportcard", "card"))


def build_container_table(containers):
    """Build Rich table for container status."""
    table = Table(
        title="LXC Container Status",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("CTID", style="bold", justify="right", min_width=6)
    table.add_column("Name", style="white", min_width=20)
    table.add_column("Status", min_width=10)
    table.add_column("CPU", justify="right", min_width=8)
    table.add_column("Memory", justify="right", min_width=14)
    table.add_column("Disk", justify="right", min_width=14)
    table.add_column("Net In/Out", justify="right", min_width=16)
    table.add_column("Uptime", justify="right", min_width=10)

    # Sort: scraper containers first, then by CTID
    sorted_containers = sorted(
        containers,
        key=lambda c: (
            0 if is_scraper_container(c.get("name", "")) else 1,
            int(c.get("vmid", 0)),
        ),
    )

    for ct in sorted_containers:
        vmid = str(ct.get("vmid", ""))
        name = ct.get("name", "unknown")
        status = ct.get("status", "unknown")
        cpus = ct.get("cpus", 0)

        # Status styling
        if status == "running":
            status_text = Text("RUNNING", style="bold green")
        elif status == "stopped":
            status_text = Text("STOPPED", style="bold red")
        else:
            status_text = Text(status.upper(), style="bold yellow")

        # CPU usage
        cpu_val = ct.get("cpu", 0)
        if status == "running" and cpu_val:
            cpu_pct = cpu_val * 100
            cpu_str = f"{cpu_pct:.1f}%"
            if cpus:
                cpu_str += f" / {cpus}c"
        elif status == "running" and cpus:
            cpu_str = f"- / {cpus}c"
        else:
            cpu_str = "-"

        # Memory
        mem = ct.get("mem", 0)
        maxmem = ct.get("maxmem", 0)
        if status == "running" and maxmem > 0:
            mem_pct = (mem / maxmem) * 100
            mem_str = f"{format_bytes(mem)} / {format_bytes(maxmem)} ({mem_pct:.0f}%)"
        elif maxmem > 0:
            mem_str = f"- / {format_bytes(maxmem)}"
        else:
            mem_str = "-"

        # Disk
        disk = ct.get("disk", 0)
        maxdisk = ct.get("maxdisk", 0)
        if maxdisk > 0 and disk > 0:
            disk_pct = (disk / maxdisk) * 100
            disk_str = f"{format_bytes(disk)} / {format_bytes(maxdisk)} ({disk_pct:.0f}%)"
        elif maxdisk > 0:
            disk_str = f"- / {format_bytes(maxdisk)}"
        else:
            disk_str = "-"

        # Network
        netin = ct.get("netin", 0)
        netout = ct.get("netout", 0)
        if status == "running" and (netin or netout):
            net_str = f"{format_bytes(netin)} / {format_bytes(netout)}"
        else:
            net_str = "-"

        # Uptime
        uptime = format_uptime(ct.get("uptime", 0))

        # Highlight scraper containers
        name_style = "bold yellow" if is_scraper_container(name) else "white"
        name_text = Text(name, style=name_style)

        table.add_row(vmid, name_text, status_text, cpu_str, mem_str,
                       disk_str, net_str, uptime)

    return table


def build_db_stats_table(stats):
    """Build Rich table for database scraping stats."""
    if "error" in stats:
        return Panel(
            f"[red]Database connection error: {stats['error']}[/red]",
            title="Database Stats",
        )

    # Main stats
    table = Table(
        title="Scraping Progress",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("Status", style="white", min_width=16)
    table.add_column("Cards", justify="right", style="green", min_width=10)
    table.add_column("", min_width=20)

    total = stats.get("total_cards", 0)

    for status in ("pending", "processing", "image_found", "downloading",
                    "downloaded", "no_image", "error"):
        count = stats.get("card_statuses", {}).get(status, 0)
        pct = (count / total * 100) if total > 0 else 0

        # Progress bar
        bar_width = 20
        filled = int(bar_width * pct / 100)
        bar = "[green]" + "█" * filled + "[/green]" + "░" * (bar_width - filled)
        pct_str = f"{bar} {pct:.1f}%"

        # Color code active statuses
        if status == "processing":
            style = "bold yellow"
        elif status == "downloading":
            style = "bold blue"
        elif status == "downloaded":
            style = "bold green"
        elif status == "error":
            style = "bold red"
        else:
            style = "white"

        table.add_row(
            Text(status, style=style),
            f"{count:,}",
            pct_str,
        )

    table.add_section()
    table.add_row(Text("TOTAL", style="bold"), f"{total:,}", "")

    return table


def build_sport_table(stats):
    """Build per-sport breakdown table."""
    sport_summary = stats.get("sport_summary", [])
    if not sport_summary:
        return None

    table = Table(
        title="Cards by Sport",
        show_header=True,
        header_style="bold cyan",
        title_style="bold white",
    )
    table.add_column("Sport", style="bold white", min_width=12)
    table.add_column("Total", justify="right", min_width=8)
    table.add_column("Done", justify="right", style="green", min_width=8)
    table.add_column("Active", justify="right", style="yellow", min_width=8)
    table.add_column("Pending", justify="right", min_width=8)
    table.add_column("Errors", justify="right", style="red", min_width=8)
    table.add_column("Progress", min_width=16)

    for sport in sport_summary:
        total = sport["total"]
        done = sport["done"] or 0
        pct = (done / total * 100) if total > 0 else 0

        bar_width = 12
        filled = int(bar_width * pct / 100)
        bar = "[green]" + "█" * filled + "[/green]" + "░" * (bar_width - filled)

        table.add_row(
            sport["sport"].title(),
            f"{total:,}",
            f"{done:,}",
            f"{sport['active'] or 0:,}",
            f"{sport['pending'] or 0:,}",
            f"{sport['errors'] or 0:,}",
            f"{bar} {pct:.0f}%",
        )

    return table


# ── Main Logic ───────────────────────────────────────────────────────────
def fetch_container_data(host=None):
    """Try to get container data via API, then fallback to pct."""
    target_host = host or PROXMOX_HOST

    # Try Proxmox API first
    try:
        client = ProxmoxClient(
            host=target_host,
            port=PROXMOX_PORT,
            user=PROXMOX_USER,
            password=PROXMOX_PASSWORD,
            token_name=PROXMOX_TOKEN_NAME,
            token_value=PROXMOX_TOKEN_VALUE,
        )
        client.authenticate()
        containers = client.get_containers()
        return containers, "api"
    except Exception as api_err:
        console.print(f"[dim]API connection failed: {api_err}[/dim]")

    # Fallback to local pct
    containers = get_containers_via_pct()
    if containers is not None:
        return containers, "pct"

    return None, None


def display_stats(host=None, db_only=False):
    """Display all stats."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    console.print(Panel.fit(
        f"[bold]LXC Container & Scraping Monitor[/bold]\n"
        f"[dim]{timestamp}[/dim]",
        border_style="blue",
    ))

    # Container stats
    if not db_only:
        containers, method = fetch_container_data(host)
        if containers:
            source = "Proxmox API" if method == "api" else "pct (local)"
            console.print(f"\n[dim]Source: {source} @ {host or PROXMOX_HOST}[/dim]")

            # Filter to show scraper containers prominently
            scraper_cts = [c for c in containers if is_scraper_container(c.get("name", ""))]
            other_cts = [c for c in containers if not is_scraper_container(c.get("name", ""))]

            running_scrapers = sum(1 for c in scraper_cts if c.get("status") == "running")
            total_scrapers = len(scraper_cts)

            console.print(
                f"\n[bold]Scraper Containers: "
                f"[green]{running_scrapers}[/green] / {total_scrapers} running[/bold]"
            )

            console.print(build_container_table(containers))
        else:
            console.print(
                "\n[yellow]Could not connect to Proxmox.[/yellow]"
                "\n[dim]Set PROXMOX_HOST, PROXMOX_PASSWORD (or TOKEN) env vars, "
                "or run on the Proxmox host directly.[/dim]\n"
                "[dim]Showing database stats only...[/dim]"
            )

    # Database stats
    console.print()
    db_stats = get_db_stats()
    console.print(build_db_stats_table(db_stats))

    sport_table = build_sport_table(db_stats)
    if sport_table:
        console.print()
        console.print(sport_table)

    # Active work summary
    processing = db_stats.get("processing", 0)
    downloading = db_stats.get("downloading", 0)
    if processing > 0 or downloading > 0:
        console.print(
            f"\n[bold yellow]Active Work:[/bold yellow] "
            f"{processing:,} cards being processed, "
            f"{downloading:,} images downloading"
        )
    else:
        console.print("\n[dim]No active processing detected.[/dim]")


def monitor_mode(host=None, db_only=False, interval=10):
    """Live monitoring with periodic refresh."""
    console.print("[bold]Starting live monitor[/bold] (press Ctrl+C to stop)\n")

    try:
        while True:
            console.clear()
            display_stats(host, db_only)
            console.print(f"\n[dim]Refreshing every {interval}s... (Ctrl+C to stop)[/dim]")
            time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped.[/yellow]")


def main():
    parser = argparse.ArgumentParser(
        description="LXC Container Stats Monitor for SportsCardPro Scraper"
    )
    parser.add_argument("--monitor", action="store_true",
                        help="Live monitoring mode (refreshes periodically)")
    parser.add_argument("--interval", type=int, default=10,
                        help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--proxmox-host", type=str, default=None,
                        help="Proxmox host IP (default: from env or 192.168.1.14)")
    parser.add_argument("--db-only", action="store_true",
                        help="Only show database stats, skip Proxmox connection")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON (for scripting)")
    args = parser.parse_args()

    if args.json:
        output = {}
        if not args.db_only:
            containers, method = fetch_container_data(args.proxmox_host)
            if containers:
                output["containers"] = containers
                output["source"] = method
        output["database"] = get_db_stats()
        print(json.dumps(output, indent=2, default=str))
        return

    if args.monitor:
        monitor_mode(args.proxmox_host, args.db_only, args.interval)
    else:
        display_stats(args.proxmox_host, args.db_only)


if __name__ == "__main__":
    main()
