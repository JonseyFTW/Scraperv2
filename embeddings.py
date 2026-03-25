#!/usr/bin/env python3
"""
SportsCardPro Scraper - CLIP Embedding Generator

Generates vector embeddings for downloaded card images using OpenCLIP.
These embeddings enable fast visual similarity search via cosine distance.

Usage:
    python embeddings.py generate          # Generate embeddings for all downloaded cards
    python embeddings.py search image.jpg  # Find top matches for an input image
    python embeddings.py stats             # Show embedding coverage

Requirements (install separately — these are large):
    pip install open-clip-torch torch torchvision numpy --break-system-packages
"""
import argparse
import os
import sys
import sqlite3
import json
import struct
from datetime import datetime, timezone

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
        console.print("  pip install open-clip-torch torch torchvision --break-system-packages")
        sys.exit(1)

    console.print("[cyan]Loading CLIP model (first time takes a minute)...[/cyan]")

    # ViT-B/32 is a good balance of speed and accuracy
    # For better accuracy at the cost of speed, use ViT-L/14
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


def _vector_to_bytes(vec) -> bytes:
    """Serialize numpy vector to bytes for SQLite storage."""
    return vec.tobytes()


def _bytes_to_vector(data: bytes):
    """Deserialize bytes back to numpy vector."""
    import numpy as np
    return np.frombuffer(data, dtype=np.float32)


def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    import numpy as np
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Database: embeddings table
# ---------------------------------------------------------------------------

def init_embeddings_table():
    conn = db.get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS embeddings (
            card_slug   TEXT PRIMARY KEY,
            vector      BLOB NOT NULL,
            model       TEXT DEFAULT 'ViT-B-32',
            created_at  TEXT,
            FOREIGN KEY (card_slug) REFERENCES cards(slug)
        );
    """)
    conn.commit()
    conn.close()


def save_embedding(card_slug: str, vector, model_name: str = "ViT-B-32"):
    conn = db.get_connection()
    conn.execute("""
        INSERT INTO embeddings (card_slug, vector, model, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(card_slug) DO UPDATE SET vector=excluded.vector, created_at=excluded.created_at
    """, (card_slug, _vector_to_bytes(vector), model_name, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    conn.close()


def get_all_embeddings() -> list[tuple]:
    """Returns list of (card_slug, vector) tuples."""
    conn = db.get_connection()
    rows = conn.execute("SELECT card_slug, vector FROM embeddings").fetchall()
    conn.close()
    return [(r["card_slug"], _bytes_to_vector(r["vector"])) for r in rows]


def get_cards_needing_embeddings(limit: int = 500) -> list[dict]:
    """Get downloaded cards that don't have embeddings yet."""
    conn = db.get_connection()
    rows = conn.execute("""
        SELECT c.* FROM cards c
        LEFT JOIN embeddings e ON c.product_id = e.card_slug
        WHERE c.status = 'downloaded' AND c.image_path IS NOT NULL AND e.card_slug IS NULL
        ORDER BY c.id
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def generate_embeddings(limit: int = 0):
    """Generate CLIP embeddings for all downloaded cards."""
    init_embeddings_table()
    batch_size = 100 if limit == 0 else min(limit, 100)
    total = 0

    console.print(f"\n[bold]Generating CLIP embeddings[/bold]\n")
    _load_model()

    while True:
        cards = get_cards_needing_embeddings(batch_size)
        if not cards:
            break

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
                        save_embedding(card["product_id"], vec)
                        total += 1

                progress.advance(task)

                if limit > 0 and total >= limit:
                    console.print(f"\n[yellow]Reached limit of {limit}[/yellow]")
                    return

    console.print(f"\n[green]Generated {total} embeddings[/green]")


def search_by_image(image_path: str, top_k: int = 10):
    """Find the top-K most similar cards to an input image."""
    init_embeddings_table()

    if not os.path.exists(image_path):
        console.print(f"[red]Image not found: {image_path}[/red]")
        return

    console.print(f"[cyan]Embedding query image...[/cyan]")
    query_vec = _embed_image(image_path)
    if query_vec is None:
        return

    console.print(f"[cyan]Searching {top_k} closest matches...[/cyan]")

    all_embs = get_all_embeddings()
    if not all_embs:
        console.print("[yellow]No embeddings in database yet. Run 'generate' first.[/yellow]")
        return

    # Compute similarities
    results = []
    for slug, vec in all_embs:
        sim = _cosine_similarity(query_vec, vec)
        results.append((slug, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:top_k]

    # Fetch card details
    conn = db.get_connection()
    table = Table(title=f"Top {top_k} Matches", show_header=True)
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Title", style="cyan")
    table.add_column("Set")
    table.add_column("Image Path", style="dim")

    for i, (card_id, sim) in enumerate(top, 1):
        row = conn.execute(
            "SELECT product_name, set_slug, image_path FROM cards WHERE product_id=?", (card_id,)
        ).fetchone()
        if row:
            table.add_row(
                str(i),
                f"{sim:.4f}",
                row["product_name"] or "Unknown",
                row["set_slug"] or "",
                row["image_path"] or ""
            )

    conn.close()
    console.print(table)


def search_by_text(query: str, top_k: int = 10):
    """Find cards matching a text description using CLIP text encoding."""
    init_embeddings_table()

    console.print(f"[cyan]Encoding text query: '{query}'[/cyan]")
    query_vec = _embed_text(query)

    all_embs = get_all_embeddings()
    if not all_embs:
        console.print("[yellow]No embeddings in database yet.[/yellow]")
        return

    results = []
    for slug, vec in all_embs:
        sim = _cosine_similarity(query_vec, vec)
        results.append((slug, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    top = results[:top_k]

    conn = db.get_connection()
    table = Table(title=f"Top {top_k} Text Matches for '{query}'", show_header=True)
    table.add_column("Rank", width=5)
    table.add_column("Score", justify="right", width=10)
    table.add_column("Title", style="cyan")
    table.add_column("Set")

    for i, (card_id, sim) in enumerate(top, 1):
        row = conn.execute(
            "SELECT product_name, set_slug FROM cards WHERE product_id=?", (card_id,)
        ).fetchone()
        if row:
            table.add_row(str(i), f"{sim:.4f}", row["product_name"] or "?", row["set_slug"] or "")

    conn.close()
    console.print(table)


def show_embedding_stats():
    init_embeddings_table()
    conn = db.get_connection()

    total_cards = conn.execute("SELECT COUNT(*) as c FROM cards WHERE status='downloaded'").fetchone()["c"]
    total_embs = conn.execute("SELECT COUNT(*) as c FROM embeddings").fetchone()["c"]

    conn.close()

    console.print(f"\n  Downloaded cards: [cyan]{total_cards}[/cyan]")
    console.print(f"  Embeddings:       [cyan]{total_embs}[/cyan]")
    if total_cards > 0:
        pct = (total_embs / total_cards) * 100
        console.print(f"  Coverage:         [green]{pct:.1f}%[/green]")


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

    args = parser.parse_args()

    if args.command == "generate":
        generate_embeddings(args.limit)
    elif args.command == "search":
        search_by_image(args.image, args.top)
    elif args.command == "text-search":
        search_by_text(args.query, args.top)
    elif args.command == "stats":
        show_embedding_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
