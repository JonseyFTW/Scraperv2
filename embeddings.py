#!/usr/bin/env python3
"""
SportsCardPro Scraper - CLIP Embedding Generator (ChromaDB)

Generates vector embeddings for downloaded card images using OpenCLIP.
Stores embeddings in ChromaDB for fast indexed similarity search.

Usage:
    python embeddings.py generate              # Generate embeddings for all downloaded cards
    python embeddings.py search image.jpg      # Find top matches for an input image
    python embeddings.py text-search "query"   # Find matches by text description
    python embeddings.py stats                 # Show embedding coverage
    python embeddings.py migrate               # Migrate embeddings from SQLite to ChromaDB

Requirements (install separately — these are large):
    pip install open-clip-torch torch torchvision numpy chromadb
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

import config
import database as db

console = Console()

# Lazy imports — these are heavy and not needed for the scraper itself
_model = None
_preprocess = None
_tokenizer = None


def _load_model():
    """Load CLIP model lazily on first use."""
    global _model, _preprocess, _tokenizer
    if _model is not None:
        return

    try:
        import open_clip
        import torch
    except ImportError:
        console.print("[red]Missing dependencies. Install with:[/red]")
        console.print("  pip install open-clip-torch torch torchvision")
        sys.exit(1)

    console.print("[cyan]Loading CLIP model (first time takes a minute)...[/cyan]")

    model_name = "ViT-B-32"
    pretrained = "laion2b_s34b_b79k"

    _model, _, _preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    _tokenizer = open_clip.get_tokenizer(model_name)
    _model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    console.print(f"[green]CLIP model loaded on {device}[/green]")


def _get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _embed_image(image_path: str):
    """Generate a CLIP embedding vector for a single image."""
    import torch
    from PIL import Image

    _load_model()
    device = _get_device()

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = _model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)

        return features.cpu().numpy().flatten()
    except Exception as e:
        console.print(f"[red]Error embedding {image_path}: {e}[/red]")
        return None


def _embed_text(text: str):
    """Generate a CLIP embedding for a text query."""
    import torch

    _load_model()
    device = _get_device()

    tokens = _tokenizer([text]).to(device)
    with torch.no_grad():
        features = _model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_chroma_client = None
_chroma_collection = None


def get_collection():
    """Get or create the ChromaDB collection for card embeddings."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name="card_images",
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def get_cards_needing_embeddings(limit: int = 500) -> list[dict]:
    """Get downloaded cards that don't have embeddings yet."""
    conn = db.get_connection()
    rows = conn.execute("""
        SELECT * FROM cards
        WHERE status = 'downloaded' AND image_path IS NOT NULL
        ORDER BY id
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    cards = [dict(r) for r in rows]
    if not cards:
        return []

    # Filter out cards already in ChromaDB
    collection = get_collection()
    card_ids = [str(c["product_id"]) for c in cards]
    existing = collection.get(ids=card_ids)
    existing_ids = set(existing["ids"])

    return [c for c in cards if str(c["product_id"]) not in existing_ids]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def generate_embeddings(limit: int = 0):
    """Generate CLIP embeddings for all downloaded cards."""
    batch_size = 100 if limit == 0 else min(limit, 100)
    total = 0

    console.print(f"\n[bold]Generating CLIP embeddings (ChromaDB)[/bold]\n")
    _load_model()

    collection = get_collection()

    while True:
        cards = get_cards_needing_embeddings(batch_size)
        if not cards:
            break

        # Batch buffers for ChromaDB upsert
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

                if card["image_path"] and os.path.exists(card["image_path"]):
                    vec = _embed_image(card["image_path"])
                    if vec is not None:
                        batch_ids.append(str(card["product_id"]))
                        batch_embeddings.append(vec.tolist())
                        batch_metadatas.append({
                            "product_name": card["product_name"] or "",
                            "set_slug": card["set_slug"] or "",
                            "image_path": card["image_path"] or "",
                            "loose_price": float(card["loose_price"] or 0),
                        })
                        total += 1

                progress.advance(task)

                if limit > 0 and total >= limit:
                    break

        # Flush batch to ChromaDB
        if batch_ids:
            collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadatas,
            )

        if limit > 0 and total >= limit:
            console.print(f"\n[yellow]Reached limit of {limit}[/yellow]")
            break

    console.print(f"\n[green]Generated {total} embeddings ({collection.count()} total in ChromaDB)[/green]")


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
        # ChromaDB cosine distance = 1 - similarity
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

    console.print(f"[cyan]Embedding query image...[/cyan]")
    query_vec = _embed_image(image_path)
    if query_vec is None:
        return

    console.print(f"[cyan]Searching {top_k} closest matches...[/cyan]")
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, collection.count()),
    )

    _display_results(results, f"Top {top_k} Matches")


def search_by_text(query: str, top_k: int = 10):
    """Find cards matching a text description using CLIP text encoding."""
    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]No embeddings in database yet.[/yellow]")
        return

    console.print(f"[cyan]Encoding text query: '{query}'[/cyan]")
    query_vec = _embed_text(query)

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, collection.count()),
    )

    _display_results(results, f"Top {top_k} Text Matches for '{query}'")


def show_embedding_stats():
    collection = get_collection()
    total_embs = collection.count()

    conn = db.get_connection()
    total_cards = conn.execute("SELECT COUNT(*) as c FROM cards WHERE status='downloaded'").fetchone()["c"]
    conn.close()

    console.print(f"\n  Downloaded cards: [cyan]{total_cards}[/cyan]")
    console.print(f"  Embeddings:       [cyan]{total_embs}[/cyan]")
    if total_cards > 0:
        pct = (total_embs / total_cards) * 100
        console.print(f"  Coverage:         [green]{pct:.1f}%[/green]")
    console.print(f"  Storage:          [dim]{config.CHROMA_DIR}[/dim]")


def migrate_from_sqlite():
    """Migrate existing embeddings from SQLite blob storage to ChromaDB."""
    import numpy as np

    conn = db.get_connection()

    # Check if the old embeddings table exists
    table_exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'"
    ).fetchone()

    if not table_exists:
        console.print("[yellow]No SQLite embeddings table found — nothing to migrate.[/yellow]")
        conn.close()
        return

    rows = conn.execute("""
        SELECT e.card_slug, e.vector, c.product_name, c.set_slug, c.image_path, c.loose_price
        FROM embeddings e
        LEFT JOIN cards c ON c.product_id = e.card_slug
    """).fetchall()
    conn.close()

    if not rows:
        console.print("[yellow]No embeddings found in SQLite — nothing to migrate.[/yellow]")
        return

    console.print(f"[cyan]Migrating {len(rows)} embeddings from SQLite to ChromaDB...[/cyan]")

    collection = get_collection()
    batch_ids = []
    batch_embeddings = []
    batch_metadatas = []

    for r in rows:
        vec = np.frombuffer(r["vector"], dtype=np.float32)
        batch_ids.append(str(r["card_slug"]))
        batch_embeddings.append(vec.tolist())
        batch_metadatas.append({
            "product_name": r["product_name"] or "",
            "set_slug": r["set_slug"] or "",
            "image_path": r["image_path"] or "",
            "loose_price": float(r["loose_price"] or 0),
        })

    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings,
        metadatas=batch_metadatas,
    )

    console.print(f"[green]Migrated {len(batch_ids)} embeddings to ChromaDB[/green]")


def main():
    parser = argparse.ArgumentParser(description="CLIP Embedding Generator for Sports Cards")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate", help="Generate embeddings for downloaded cards")
    gen_parser.add_argument("--limit", type=int, default=0, help="Max embeddings to generate")

    img_parser = subparsers.add_parser("search", help="Search by image")
    img_parser.add_argument("image", help="Path to query image")
    img_parser.add_argument("--top", type=int, default=10, help="Number of results")

    txt_parser = subparsers.add_parser("text-search", help="Search by text description")
    txt_parser.add_argument("query", help="Text description to search")
    txt_parser.add_argument("--top", type=int, default=10, help="Number of results")

    subparsers.add_parser("stats", help="Show embedding stats")
    subparsers.add_parser("migrate", help="Migrate embeddings from SQLite to ChromaDB")

    args = parser.parse_args()

    if args.command == "generate":
        generate_embeddings(args.limit)
    elif args.command == "search":
        search_by_image(args.image, args.top)
    elif args.command == "text-search":
        search_by_text(args.query, args.top)
    elif args.command == "stats":
        show_embedding_stats()
    elif args.command == "migrate":
        migrate_from_sqlite()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
