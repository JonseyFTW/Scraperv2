#!/usr/bin/env python3
"""
SportsCardPro Scraper - DINOv2 Embedding Generator (ChromaDB + RunPod)

Generates vector embeddings for downloaded card images using DINOv2-ViT-L/14
hosted on a RunPod serverless endpoint. Stores embeddings in ChromaDB for
fast indexed similarity search.

Drop-in replacement for embeddings.py — uses DINOv2 (1024-dim) instead of
CLIP ViT-B-32 (512-dim). Stores in a SEPARATE collection since dimensions differ.

Usage:
    python embeddings_dinov2.py generate              # Generate embeddings for all downloaded cards
    python embeddings_dinov2.py search image.jpg      # Find top matches for an input image
    python embeddings_dinov2.py stats                 # Show embedding coverage
    python embeddings_dinov2.py generate --limit 100  # Generate only 100 embeddings (for testing)

Requirements:
    pip install requests numpy chromadb rich

Environment variables:
    RUNPOD_API_KEY          Your RunPod API key
    RUNPOD_ENDPOINT_ID      Your serverless endpoint ID (default in config.py)

Note: DINOv2 is vision-only — no text search support (use CLIP for that).
"""
import argparse
import base64
import os
import sys
import time

import chromadb
import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

import config
import database as db

console = Console()

# Collection name — MUST be different from CLIP's "card_images" because
# DINOv2 produces 1024-dim embeddings vs CLIP's 512-dim
COLLECTION_NAME = "card_images_dinov2"

RUNPOD_TIMEOUT = 90  # seconds to wait for serverless response
RUNPOD_POLL_INTERVAL = 1.0  # seconds between status polls


def _runpod_url(path: str) -> str:
    """Build a RunPod serverless API URL."""
    return f"https://api.runpod.ai/v2/{config.RUNPOD_ENDPOINT_ID}/{path}"


def _runpod_headers() -> dict:
    """Build auth headers for RunPod API."""
    if not config.RUNPOD_API_KEY:
        console.print("[red]RUNPOD_API_KEY not set. Export it as an environment variable.[/red]")
        sys.exit(1)
    return {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _embed_image(image_path: str):
    """Send an image to RunPod DINOv2 endpoint and return the 1024-dim embedding."""
    try:
        with open(image_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        console.print(f"[red]Error reading {image_path}: {e}[/red]")
        return None

    payload = {
        "input": {
            "action": "embed",
            "image": image_b64,
        }
    }

    try:
        # Use runsync for simplicity — blocks until the worker returns
        resp = requests.post(
            _runpod_url("runsync"),
            json=payload,
            headers=_runpod_headers(),
            timeout=RUNPOD_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")

        # If the endpoint returned IN_QUEUE or IN_PROGRESS, poll for result
        if status in ("IN_QUEUE", "IN_PROGRESS"):
            job_id = data["id"]
            deadline = time.time() + RUNPOD_TIMEOUT
            while time.time() < deadline:
                time.sleep(RUNPOD_POLL_INTERVAL)
                poll_resp = requests.get(
                    _runpod_url(f"status/{job_id}"),
                    headers=_runpod_headers(),
                    timeout=30,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data.get("status")
                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    console.print(f"[red]RunPod job failed for {image_path}: {data}[/red]")
                    return None

            if status != "COMPLETED":
                console.print(f"[red]RunPod job timed out for {image_path}[/red]")
                return None

        if status == "COMPLETED":
            output = data.get("output")
            if output is None:
                console.print(f"[red]RunPod returned no output for {image_path}. Response: {data}[/red]")
                return None
            if "error" in output:
                console.print(f"[red]RunPod error for {image_path}: {output['error']}[/red]")
                return None
            embedding = output.get("embedding")
            if embedding is None:
                console.print(f"[red]No 'embedding' key in response for {image_path}. Keys: {list(output.keys())}[/red]")
                return None
            return embedding
        else:
            error_msg = data.get("error", "no error details")
            console.print(f"[red]RunPod status '{status}' for {image_path}: {error_msg}[/red]")
            console.print(f"[dim]{str(data)[:500]}[/dim]")
            return None

    except requests.exceptions.RequestException as e:
        console.print(f"[red]RunPod request error for {image_path}: {e}[/red]")
        return None


def _embed_image_batch(image_paths: list[str]) -> list:
    """Send a batch of images to RunPod and return embeddings.

    Falls back to single-image calls if the endpoint doesn't support batch.
    """
    # Try batch request first
    images_b64 = []
    valid_paths = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode("utf-8"))
                valid_paths.append(path)
        except Exception as e:
            console.print(f"[red]Error reading {path}: {e}[/red]")
            images_b64.append(None)

    payload = {
        "input": {
            "images": [img for img in images_b64 if img is not None],
        }
    }

    try:
        resp = requests.post(
            _runpod_url("runsync"),
            json=payload,
            headers=_runpod_headers(),
            timeout=RUNPOD_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        status = data.get("status")
        if status in ("IN_QUEUE", "IN_PROGRESS"):
            job_id = data["id"]
            deadline = time.time() + RUNPOD_TIMEOUT
            while time.time() < deadline:
                time.sleep(RUNPOD_POLL_INTERVAL)
                poll_resp = requests.get(
                    _runpod_url(f"status/{job_id}"),
                    headers=_runpod_headers(),
                    timeout=30,
                )
                poll_resp.raise_for_status()
                data = poll_resp.json()
                status = data.get("status")
                if status == "COMPLETED":
                    break
                if status == "FAILED":
                    break

        if status == "COMPLETED":
            output = data["output"]
            # Support both {"embeddings": [...]} and {"embedding": [...]}
            embeddings = output.get("embeddings", [output.get("embedding")])
            return embeddings

    except requests.exceptions.RequestException:
        pass

    # Fallback: send images one at a time
    results = []
    for path in valid_paths:
        emb = _embed_image(path)
        results.append(emb)
    return results


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_chroma_client = None
_chroma_collection = None


def get_collection():
    """Get or create the DINOv2 ChromaDB collection."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


_skipped_ids = set()


def _resolve_image_path(image_path: str) -> str:
    """Translate a DB image path to the local filesystem."""
    if not image_path:
        return image_path

    if os.path.exists(image_path):
        return image_path

    prefix = config.LINUX_DATA_PREFIX
    if prefix and image_path.startswith(prefix):
        translated = os.path.join(config.DATA_DIR, image_path[len(prefix):].lstrip("/\\"))
        return translated

    basename = os.path.basename(image_path)
    return os.path.join(config.IMAGE_DIR, basename)


def get_cards_needing_embeddings(limit: int = 500) -> list[dict]:
    """Get downloaded cards that don't have DINOv2 embeddings yet."""
    collection = get_collection()

    existing_ids = set()
    if collection.count() > 0:
        all_existing = collection.get(include=[])
        existing_ids = set(all_existing["ids"])

    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
    cur.execute("""
        SELECT * FROM cards
        WHERE status = 'downloaded' AND image_path IS NOT NULL
        ORDER BY id
    """)
    rows = cur.fetchall()
    cur.close()
    db.put_connection(conn)

    skip = existing_ids | _skipped_ids
    cards = [dict(r) for r in rows if str(r["product_id"]) not in skip]

    return cards[:limit]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def generate_embeddings(limit: int = 0):
    """Generate DINOv2 embeddings for all downloaded cards via RunPod."""
    batch_size = 100 if limit == 0 else min(limit, 100)
    total = 0

    console.print(f"\n[bold]Generating DINOv2 embeddings via RunPod (ChromaDB → {COLLECTION_NAME})[/bold]")
    console.print(f"  Endpoint: [cyan]{config.RUNPOD_ENDPOINT_ID}[/cyan]\n")

    collection = get_collection()

    while True:
        cards = get_cards_needing_embeddings(batch_size)
        if not cards:
            break

        batch_ids = []
        batch_embeddings = []
        batch_metadatas = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Embedding", total=len(cards))

            for card in cards:
                progress.update(task, description=f"Embed: {(card['product_name'] or 'img')[:30]}")

                resolved_path = _resolve_image_path(card.get("image_path", ""))
                if resolved_path and os.path.exists(resolved_path):
                    vec = _embed_image(resolved_path)
                    if vec is not None:
                        batch_ids.append(str(card["product_id"]))
                        batch_embeddings.append(vec if isinstance(vec, list) else vec.tolist())
                        batch_metadatas.append({
                            "product_name": card["product_name"] or "",
                            "set_slug": card["set_slug"] or "",
                            "image_path": card["image_path"] or "",
                            "loose_price": float(card["loose_price"] or 0),
                        })
                        total += 1
                else:
                    _skipped_ids.add(str(card["product_id"]))

                progress.advance(task)

                if limit > 0 and total >= limit:
                    break

        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

        if limit > 0 and total >= limit:
            console.print(f"\n[yellow]Reached limit of {limit}[/yellow]")
            break

    global _chroma_client, _chroma_collection
    final_count = collection.count()
    if _chroma_client is not None:
        del _chroma_collection
        del _chroma_client
        _chroma_collection = None
        _chroma_client = None

    console.print(f"\n[green]Generated {total} embeddings ({final_count} total in ChromaDB)[/green]")
    if _skipped_ids:
        console.print(f"[yellow]Skipped {len(_skipped_ids)} cards (image file not found on disk)[/yellow]")


def _display_results(results: dict, title: str):
    """Display ChromaDB query results in a rich table."""
    table = Table(title=title, show_header=True)
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Title", style="cyan")
    table.add_column("Set")
    table.add_column("Price", justify="right")
    table.add_column("Image Path", style="dim")

    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    for i, (card_id, dist, meta) in enumerate(zip(ids, distances, metadatas), 1):
        similarity = 1.0 - dist
        price = meta.get("loose_price", 0)
        price_str = f"${price:.2f}" if price else ""
        table.add_row(
            str(i),
            f"{similarity:.4f}",
            meta.get("product_name", "Unknown"),
            meta.get("set_slug", ""),
            price_str,
            meta.get("image_path", ""),
        )

    console.print(table)


def search_by_image(image_path: str, top_k: int = 10):
    """Find the top-K most similar cards to an input image."""
    if not os.path.exists(image_path):
        console.print(f"[red]Image not found: {image_path}[/red]")
        return

    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]No embeddings in database yet. Run 'generate' first.[/yellow]")
        return

    console.print(f"[cyan]Embedding query image via RunPod DINOv2...[/cyan]")
    query_vec = _embed_image(image_path)
    if query_vec is None:
        return

    query_list = query_vec if isinstance(query_vec, list) else query_vec.tolist()

    console.print(f"[cyan]Searching {top_k} closest matches...[/cyan]")
    results = collection.query(
        query_embeddings=[query_list],
        n_results=min(top_k, collection.count()),
    )

    _display_results(results, f"Top {top_k} DINOv2 Matches")


def show_embedding_stats():
    collection = get_collection()
    total_embs = collection.count()

    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cards WHERE status='downloaded'")
    total_cards = cur.fetchone()[0]
    cur.close()
    db.put_connection(conn)

    console.print(f"\n  Collection:       [cyan]{COLLECTION_NAME}[/cyan]")
    console.print(f"  Model:            [cyan]DINOv2-ViT-L/14 (1024-dim) via RunPod[/cyan]")
    console.print(f"  Endpoint:         [cyan]{config.RUNPOD_ENDPOINT_ID}[/cyan]")
    console.print(f"  Downloaded cards: [cyan]{total_cards}[/cyan]")
    console.print(f"  Embeddings:       [cyan]{total_embs}[/cyan]")
    if total_cards > 0:
        pct = (total_embs / total_cards) * 100
        console.print(f"  Coverage:         [green]{pct:.1f}%[/green]")
    console.print(f"  Storage:          [dim]{config.CHROMA_DIR}[/dim]")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Embedding Generator for Sports Cards (RunPod)")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate", help="Generate DINOv2 embeddings for downloaded cards")
    gen_parser.add_argument("--limit", type=int, default=0, help="Max embeddings to generate")

    img_parser = subparsers.add_parser("search", help="Search by image")
    img_parser.add_argument("image", help="Path to query image")
    img_parser.add_argument("--top", type=int, default=10, help="Number of results")

    subparsers.add_parser("stats", help="Show embedding stats")

    args = parser.parse_args()

    if args.command == "generate":
        generate_embeddings(args.limit)
    elif args.command == "search":
        search_by_image(args.image, args.top)
    elif args.command == "stats":
        show_embedding_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
