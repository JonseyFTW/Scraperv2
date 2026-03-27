#!/usr/bin/env python3
"""
Migrate ChromaDB embeddings from local to a remote ChromaDB (e.g. Railway).

Usage:
    python migrate_chroma.py --target http://your-railway-chroma-url:8000
    python migrate_chroma.py --target http://your-railway-chroma-url:8000 --batch-size 500
    python migrate_chroma.py --target http://your-railway-chroma-url:8000 --dry-run

Environment variables (optional):
    CHROMA_TARGET_URL       Remote ChromaDB URL (alternative to --target)
    CHROMA_TARGET_TOKEN     Auth token/header for remote ChromaDB (if required)
"""
import argparse
import sys

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.panel import Panel
from rich.table import Table

import config

console = Console()

COLLECTION_NAME = "card_images"


def get_local_client():
    """Connect to local persistent ChromaDB.
    Forces SQLite WAL checkpoint so we see the latest embeddings."""
    import sqlite3
    import glob
    # Force WAL checkpoint on ChromaDB's SQLite before opening
    for db_file in glob.glob(f"{config.CHROMA_DIR}/**/chroma.sqlite3", recursive=True):
        try:
            conn = sqlite3.connect(db_file)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
        except Exception:
            pass
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def get_remote_client(target_url: str, token: str = None):
    """Connect to remote ChromaDB over HTTP."""
    from urllib.parse import urlparse
    parsed = urlparse(target_url if "://" in target_url else f"https://{target_url}")

    host = parsed.hostname
    port = parsed.port
    use_ssl = parsed.scheme == "https"

    # Default ports
    if port is None:
        port = 443 if use_ssl else 8000

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
        headers["X-Chroma-Token"] = token

    return chromadb.HttpClient(
        host=host,
        port=port,
        ssl=use_ssl,
        headers=headers if headers else None,
    )


def migrate(target_url: str, token: str = None, batch_size: int = 200, dry_run: bool = False):
    console.print(f"\n[bold]ChromaDB Migration: Local → Remote[/bold]\n")

    # Connect to source
    console.print(f"  Source: [cyan]{config.CHROMA_DIR}[/cyan]")
    local = get_local_client()
    try:
        local_col = local.get_collection(COLLECTION_NAME)
    except Exception:
        console.print(f"  [red]No '{COLLECTION_NAME}' collection found locally. Nothing to migrate.[/red]")
        return

    local_count = local_col.count()
    console.print(f"  Local embeddings: [green]{local_count:,}[/green]")

    if local_count == 0:
        console.print("  [yellow]No embeddings to migrate.[/yellow]")
        return

    # Connect to target
    console.print(f"  Target: [cyan]{target_url}[/cyan]")
    if dry_run:
        console.print("  [yellow]DRY RUN — no data will be written[/yellow]\n")
    else:
        remote = get_remote_client(target_url, token)
        try:
            remote.heartbeat()
        except Exception as e:
            console.print(f"  [red]Cannot connect to remote ChromaDB: {e}[/red]")
            return

        remote_col = remote.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        remote_count_before = remote_col.count()
        console.print(f"  Remote embeddings (before): [yellow]{remote_count_before:,}[/yellow]")

    # Fetch all local IDs to figure out what needs migrating
    console.print(f"\n  Fetching local IDs...")
    all_local_ids = local_col.get(include=[])["ids"]

    if not dry_run:
        # Check what already exists on remote
        existing_remote = set()
        if remote_col.count() > 0:
            existing_remote = set(remote_col.get(include=[])["ids"])
        ids_to_migrate = [i for i in all_local_ids if i not in existing_remote]
    else:
        ids_to_migrate = all_local_ids

    console.print(f"  IDs to migrate: [cyan]{len(ids_to_migrate):,}[/cyan] "
                  f"({local_count - len(ids_to_migrate):,} already on remote)\n")

    if not ids_to_migrate:
        console.print("  [green]Remote is already up to date![/green]")
        return

    # Migrate in batches
    migrated = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Migrating", total=len(ids_to_migrate))

        for i in range(0, len(ids_to_migrate), batch_size):
            batch_ids = ids_to_migrate[i:i + batch_size]

            # Fetch full data from local
            local_data = local_col.get(
                ids=batch_ids,
                include=["embeddings", "metadatas"],
            )

            if dry_run:
                migrated += len(batch_ids)
                progress.advance(task, len(batch_ids))
                continue

            try:
                remote_col.upsert(
                    ids=local_data["ids"],
                    embeddings=local_data["embeddings"],
                    metadatas=local_data["metadatas"],
                )
                migrated += len(batch_ids)
            except Exception as e:
                console.print(f"\n  [red]Batch error: {e}[/red]")
                failed += len(batch_ids)

            progress.advance(task, len(batch_ids))

    # Summary
    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_row("[bold]Migration Complete[/bold]")
    summary.add_row(
        f"  [green]✓ {migrated:,}[/green] migrated",
        f"[red]✗ {failed:,}[/red] failed",
    )
    if not dry_run:
        remote_count_after = remote_col.count()
        summary.add_row(f"  Remote total: [bold]{remote_count_after:,}[/bold] embeddings")
    console.print(Panel(summary, border_style="green" if failed == 0 else "yellow"))


def main():
    import os

    parser = argparse.ArgumentParser(description="Migrate ChromaDB embeddings to remote")
    parser.add_argument("--target", type=str,
                        default=os.environ.get("CHROMA_TARGET_URL", ""),
                        help="Remote ChromaDB URL (e.g. http://host:8000)")
    parser.add_argument("--token", type=str,
                        default=os.environ.get("CHROMA_TARGET_TOKEN", ""),
                        help="Auth token for remote ChromaDB")
    parser.add_argument("--batch-size", type=int, default=200,
                        help="Embeddings per batch (default: 200)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview migration without writing")

    args = parser.parse_args()

    if not args.target:
        console.print("[red]Error: --target URL required (or set CHROMA_TARGET_URL env var)[/red]")
        console.print("  Example: python migrate_chroma.py --target http://your-railway-host:8000")
        sys.exit(1)

    migrate(
        target_url=args.target,
        token=args.token,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
