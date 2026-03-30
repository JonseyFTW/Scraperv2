#!/usr/bin/env python3
"""
Sync local ChromaDB to RunPod Network Volume via S3 API.

Uploads your local ChromaDB directory to the RunPod network volume so your
serverless workers can access the embeddings.  Works for both sports card
and Pokemon card collections — uploads the entire chromadb directory.

Setup:
    1. Go to RunPod dashboard → Storage → "+ Create S3 API key"
    2. Set environment variables:
         $env:RUNPOD_S3_ACCESS_KEY = "your-access-key"
         $env:RUNPOD_S3_SECRET_KEY = "your-secret-key"

    Or edit config.py directly.

Usage:
    python sync_to_runpod.py              # Upload entire chromadb directory
    python sync_to_runpod.py --dry-run    # Show what would be uploaded
    python sync_to_runpod.py --status     # Check what's on RunPod volume

Requirements:
    pip install boto3 rich
"""
import argparse
import os
import sys

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, FileSizeColumn
from rich.table import Table

import config

console = Console()


def _get_s3_client():
    """Create an S3 client for RunPod network volume."""
    try:
        import boto3
    except ImportError:
        console.print("[red]Missing boto3. Install with:[/red]")
        console.print("  pip install boto3")
        sys.exit(1)

    if not config.RUNPOD_S3_ACCESS_KEY or not config.RUNPOD_S3_SECRET_KEY:
        console.print("[red]RunPod S3 credentials not set.[/red]")
        console.print()
        console.print("Set them via environment variables:")
        console.print('  $env:RUNPOD_S3_ACCESS_KEY = "your-access-key"')
        console.print('  $env:RUNPOD_S3_SECRET_KEY = "your-secret-key"')
        console.print()
        console.print("Or get them from RunPod dashboard → Storage → '+ Create S3 API key'")
        sys.exit(1)

    return boto3.client(
        "s3",
        endpoint_url=config.RUNPOD_S3_ENDPOINT,
        aws_access_key_id=config.RUNPOD_S3_ACCESS_KEY,
        aws_secret_access_key=config.RUNPOD_S3_SECRET_KEY,
        region_name="us-nc-1",
    )


def _collect_files(local_dir: str) -> list[tuple[str, str, int]]:
    """Collect all files to upload. Returns list of (local_path, s3_key, size)."""
    files = []
    for root, dirs, filenames in os.walk(local_dir):
        for filename in filenames:
            local_path = os.path.join(root, filename)
            # S3 key: chromadb/... (relative to DATA_DIR parent)
            rel_path = os.path.relpath(local_path, os.path.dirname(local_dir))
            s3_key = rel_path.replace("\\", "/")  # Windows → S3 paths
            size = os.path.getsize(local_path)
            files.append((local_path, s3_key, size))
    return files


def sync(dry_run: bool = False):
    """Upload local ChromaDB to RunPod network volume via S3."""
    chroma_dir = config.CHROMA_DIR

    if not os.path.exists(chroma_dir):
        console.print(f"[red]ChromaDB directory not found: {chroma_dir}[/red]")
        return

    files = _collect_files(chroma_dir)
    if not files:
        console.print("[yellow]No files found in ChromaDB directory.[/yellow]")
        return

    total_size = sum(f[2] for f in files)
    console.print(f"\n[bold]Sync ChromaDB → RunPod S3[/bold]\n")
    console.print(f"  Local:     [cyan]{chroma_dir}[/cyan]")
    console.print(f"  Endpoint:  [cyan]{config.RUNPOD_S3_ENDPOINT}[/cyan]")
    console.print(f"  Bucket:    [cyan]{config.RUNPOD_S3_BUCKET}[/cyan]")
    console.print(f"  Files:     [cyan]{len(files)}[/cyan]")
    console.print(f"  Total size: [cyan]{total_size / (1024**2):.1f} MB[/cyan]")
    console.print()

    if dry_run:
        table = Table(title="Files to upload (dry run)", show_header=True)
        table.add_column("S3 Key", style="cyan")
        table.add_column("Size", justify="right")
        for _, s3_key, size in files[:50]:
            table.add_row(s3_key, f"{size / 1024:.1f} KB")
        if len(files) > 50:
            table.add_row(f"... and {len(files) - 50} more", "")
        console.print(table)
        return

    s3 = _get_s3_client()
    uploaded = 0
    errors = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), FileSizeColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading", total=total_size)

        for local_path, s3_key, size in files:
            progress.update(task, description=f"{s3_key[-50:]}")
            try:
                s3.upload_file(
                    local_path,
                    config.RUNPOD_S3_BUCKET,
                    s3_key,
                )
                uploaded += 1
            except Exception as e:
                console.print(f"[red]Failed: {s3_key}: {e}[/red]")
                errors += 1

            progress.advance(task, advance=size)

    console.print(f"\n[green]Uploaded {uploaded} files ({total_size / (1024**2):.1f} MB)[/green]")
    if errors:
        console.print(f"[yellow]{errors} files failed[/yellow]")
    console.print()
    console.print("[dim]Your serverless workers will see the new data on next cold start.[/dim]")
    console.print("[dim]To force: scale active workers to 0 in RunPod dashboard, then back up.[/dim]")


def show_status():
    """Show what's currently on the RunPod volume."""
    s3 = _get_s3_client()

    console.print(f"\n[bold]RunPod Volume Contents[/bold]")
    console.print(f"  Bucket: [cyan]{config.RUNPOD_S3_BUCKET}[/cyan]\n")

    try:
        paginator = s3.get_paginator("list_objects_v2")
        total_files = 0
        total_size = 0
        prefixes = {}

        for page in paginator.paginate(Bucket=config.RUNPOD_S3_BUCKET):
            for obj in page.get("Contents", []):
                total_files += 1
                total_size += obj["Size"]
                # Group by top-level directory
                parts = obj["Key"].split("/")
                prefix = parts[0] if len(parts) > 1 else "(root)"
                prefixes[prefix] = prefixes.get(prefix, 0) + 1

        if total_files == 0:
            console.print("[yellow]Volume is empty.[/yellow]")
            return

        table = Table(title="Files by directory", show_header=True)
        table.add_column("Directory", style="cyan")
        table.add_column("Files", justify="right")
        for prefix, count in sorted(prefixes.items()):
            table.add_row(prefix, str(count))

        console.print(table)
        console.print(f"\n  Total: [cyan]{total_files}[/cyan] files, [cyan]{total_size / (1024**2):.1f} MB[/cyan]")

    except Exception as e:
        console.print(f"[red]Error listing volume: {e}[/red]")


def verify():
    """Compare local ChromaDB collection counts against RunPod endpoint."""
    import requests as http_requests

    if not config.RUNPOD_API_KEY:
        console.print("[red]RUNPOD_API_KEY not set. Export it first.[/red]")
        return
    if not config.RUNPOD_ENDPOINT_ID:
        console.print("[red]RUNPOD_ENDPOINT_ID not set.[/red]")
        return

    console.print(f"\n[bold]Verifying Local vs RunPod ChromaDB[/bold]\n")

    # --- Local counts ---
    try:
        local_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        local_collections = {}
        for col in local_client.list_collections():
            name = col.name if hasattr(col, 'name') else col
            c = local_client.get_collection(name)
            local_collections[name] = c.count()
    except Exception as e:
        console.print(f"[red]Error reading local ChromaDB: {e}[/red]")
        return

    # --- RunPod counts (via health endpoint) ---
    endpoint_url = f"https://api.runpod.ai/v2/{config.RUNPOD_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    console.print("[cyan]Querying RunPod endpoint health...[/cyan]")
    remote_count = None
    try:
        resp = http_requests.post(
            endpoint_url,
            json={"input": {"action": "health"}},
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        result = resp.json()
        if result.get("status") == "COMPLETED":
            output = result.get("output", {})
            remote_count = output.get("embedding_count", 0)
            remote_collection = output.get("collection", "card_embeddings_dinov2")
            console.print(f"[green]RunPod endpoint is healthy ({output.get('device', 'unknown')})[/green]")
        else:
            console.print(f"[yellow]RunPod status: {result.get('status')} — endpoint may be cold starting[/yellow]")
            console.print("[dim]Try again in 30-60 seconds if the endpoint was sleeping.[/dim]")
    except Exception as e:
        console.print(f"[red]Error querying RunPod: {e}[/red]")

    # --- Display comparison ---
    table = Table(title="ChromaDB Collection Comparison")
    table.add_column("Collection", style="cyan")
    table.add_column("Local", justify="right", style="green")
    table.add_column("RunPod", justify="right", style="yellow")
    table.add_column("Status", style="white")

    for name, local_count in sorted(local_collections.items()):
        if name == "card_embeddings_dinov2" and remote_count is not None:
            diff = local_count - remote_count
            if diff == 0:
                status = "[green]In sync[/green]"
            elif diff > 0:
                status = f"[yellow]RunPod missing {diff:,}[/yellow]"
            else:
                status = f"[yellow]Local missing {abs(diff):,}[/yellow]"
            table.add_row(name, f"{local_count:,}", f"{remote_count:,}", status)
        else:
            table.add_row(name, f"{local_count:,}", "[dim]—[/dim]",
                          "[dim]Not checked (only main collection queried)[/dim]")

    console.print(table)

    if remote_count is not None and local_collections.get("card_embeddings_dinov2", 0) != remote_count:
        console.print(f"\n[yellow]To fix: run 'python sync_to_runpod.py' to re-upload, "
                       "then restart RunPod workers.[/yellow]")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Sync ChromaDB to RunPod Network Volume (S3)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    parser.add_argument("--status", action="store_true", help="Show what's on the RunPod volume")
    parser.add_argument("--verify", action="store_true", help="Compare local vs RunPod embedding counts")
    args = parser.parse_args()

    if args.verify:
        verify()
    elif args.status:
        show_status()
    else:
        sync(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
