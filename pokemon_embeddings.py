#!/usr/bin/env python3
"""
Pokemon TCG — DINOv2 Embedding Generator + RunPod Sync

Generates DINOv2-ViT-L/14 embeddings (1024-dim) for downloaded Pokemon card
images and stores them in ChromaDB.  Same model and dimensionality as the
sports card embeddings so they can coexist in the same RunPod endpoint.

Pokemon embeddings live in a separate collection: pokemon_embeddings_dinov2
Sports card embeddings live in:                    card_embeddings_dinov2

The `sync` command pushes local Pokemon embeddings to the RunPod ChromaDB
instance via the serverless endpoint's embed action.

Usage:
    python pokemon_embeddings.py generate              # Embed all downloaded Pokemon cards
    python pokemon_embeddings.py generate --limit 100  # Embed up to 100 cards
    python pokemon_embeddings.py generate --batch 64   # Custom GPU batch size
    python pokemon_embeddings.py search image.jpg      # Find similar Pokemon cards
    python pokemon_embeddings.py stats                 # Show embedding coverage
    python pokemon_embeddings.py sync                  # Push embeddings to RunPod ChromaDB
    python pokemon_embeddings.py sync --limit 1000     # Push up to 1000 at a time

Requirements:
    pip install torch torchvision chromadb rich psycopg2-binary requests
"""
import argparse
import base64
import io
import os
import sys
import time

import chromadb
import requests
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table

import config
import database as db

console = Console()

COLLECTION_NAME = config.POKEMON_CHROMA_COLLECTION  # "pokemon_embeddings_dinov2"

# Lazy-loaded model globals
_model = None
_transform = None
_device = None


def _load_model():
    """Load DINOv2-ViT-L/14 on GPU with fp16 for speed."""
    global _model, _transform, _device
    if _model is not None:
        return

    try:
        import torch
        from torchvision import transforms
    except ImportError:
        console.print("[red]Missing dependencies. Install with:[/red]")
        console.print("  pip install torch torchvision")
        sys.exit(1)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Loading DINOv2-ViT-L/14 on {_device}...[/cyan]")

    _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    _model.eval()
    _model = _model.to(_device)

    if _device == "cuda":
        _model = _model.half()
        console.print("[green]Using fp16 (half precision) for faster inference[/green]")

    _transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warm up
    dummy = torch.randn(1, 3, 518, 518).to(_device)
    if _device == "cuda":
        dummy = dummy.half()
    with torch.no_grad():
        _model(dummy)
    if _device == "cuda":
        torch.cuda.synchronize()

    param_count = sum(p.numel() for p in _model.parameters()) / 1e6
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if _device == "cuda" else 0
    console.print(f"[green]DINOv2-ViT-L/14 ready — {param_count:.0f}M params, 1024-dim, {vram:.1f}GB VRAM[/green]")


def _embed_single(image_path: str):
    """Generate a 1024-dim embedding for a single image (search queries)."""
    import torch
    from PIL import Image

    _load_model()

    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = _transform(img).unsqueeze(0).to(_device)
        if _device == "cuda":
            img_tensor = img_tensor.half()

        with torch.no_grad():
            features = _model(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.cpu().float().numpy().flatten()
    except Exception as e:
        console.print(f"[red]Error embedding {image_path}: {e}[/red]")
        return None


def _embed_batch(image_paths: list[str]) -> list:
    """Generate embeddings for a batch of images in one GPU pass."""
    import torch
    from PIL import Image

    _load_model()

    tensors = []
    valid_indices = []

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            tensor = _transform(img)
            tensors.append(tensor)
            valid_indices.append(i)
        except Exception as e:
            console.print(f"[red]Error loading {path}: {e}[/red]")

    if not tensors:
        return [None] * len(image_paths)

    batch = torch.stack(tensors).to(_device)
    if _device == "cuda":
        batch = batch.half()

    with torch.no_grad():
        features = _model(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    embeddings_np = features.cpu().float().numpy()

    results = [None] * len(image_paths)
    for idx, valid_idx in enumerate(valid_indices):
        results[valid_idx] = embeddings_np[idx].flatten().tolist()

    return results


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

_chroma_client = None
_chroma_collection = None


def get_collection():
    """Get or create the Pokemon DINOv2 ChromaDB collection."""
    global _chroma_client, _chroma_collection
    if _chroma_collection is not None:
        return _chroma_collection

    _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    _chroma_collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return _chroma_collection


def _get_existing_ids() -> set[str]:
    collection = get_collection()
    if collection.count() == 0:
        return set()
    all_existing = collection.get(include=[])
    return set(all_existing["ids"])


# ---------------------------------------------------------------------------
# Generate embeddings
# ---------------------------------------------------------------------------

_skipped_ids: set[str] = set()


def get_cards_needing_embeddings(limit: int = 0) -> list[dict]:
    """Get downloaded Pokemon cards that don't have DINOv2 embeddings yet."""
    existing_ids = _get_existing_ids()

    cards = db.get_pokemon_cards_by_status("downloaded")

    skip = existing_ids | _skipped_ids
    cards = [c for c in cards if c["id"] not in skip]

    if limit > 0:
        return cards[:limit]
    return cards


def generate_embeddings(limit: int = 0, batch_size: int = 32):
    """Generate DINOv2 embeddings for downloaded Pokemon card images."""
    import torch

    db.init_db()

    console.print(f"\n[bold]Generating DINOv2 embeddings for Pokemon cards — batch_size={batch_size}[/bold]")
    console.print(f"  Collection: [cyan]{COLLECTION_NAME}[/cyan]\n")

    _load_model()
    collection = get_collection()

    cards = get_cards_needing_embeddings(limit)
    if not cards:
        console.print("[green]All Pokemon cards already have embeddings![/green]")
        return

    console.print(f"  Cards to embed: [cyan]{len(cards)}[/cyan]\n")

    total = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(cards))

        for batch_start in range(0, len(cards), batch_size):
            batch_cards = cards[batch_start:batch_start + batch_size]

            paths = []
            valid_cards = []
            for card in batch_cards:
                image_path = card.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    paths.append(image_path)
                    valid_cards.append(card)
                else:
                    _skipped_ids.add(card["id"])

            if paths:
                embeddings = _embed_batch(paths)

                chroma_ids = []
                chroma_embeddings = []
                chroma_metadatas = []

                for card, emb in zip(valid_cards, embeddings):
                    if emb is not None:
                        full_title = f"{card['name']} ({card['set_name']} #{card['local_id']})"
                        chroma_ids.append(card["id"])
                        chroma_embeddings.append(emb)
                        chroma_metadatas.append({
                            "name": card["name"],
                            "full_title": full_title,
                            "set_id": card["set_id"] or "",
                            "set_name": card["set_name"] or "",
                            "local_id": card["local_id"] or "",
                            "category": card.get("category") or "",
                            "image_path": card["image_path"] or "",
                            "card_type": "pokemon",
                        })
                        total += 1

                if chroma_ids:
                    collection.upsert(
                        ids=chroma_ids,
                        embeddings=chroma_embeddings,
                        metadatas=chroma_metadatas,
                    )

            progress.update(task, advance=len(batch_cards),
                            description=f"Batch {batch_start // batch_size + 1} — {total} done")

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0

    final_count = collection.count()

    # Release ChromaDB resources
    global _chroma_client, _chroma_collection
    del _chroma_collection
    del _chroma_client
    _chroma_collection = None
    _chroma_client = None

    console.print(f"\n[green]Generated {total} embeddings in {elapsed:.1f}s ({rate:.1f} img/s)[/green]")
    console.print(f"[green]{final_count} total in '{COLLECTION_NAME}'[/green]")
    if _skipped_ids:
        console.print(f"[yellow]Skipped {len(_skipped_ids)} cards (image file not found)[/yellow]")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _display_results(results: dict, title: str):
    table = Table(title=title, show_header=True)
    table.add_column("Rank", style="bold", width=5)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Card", style="cyan")
    table.add_column("Set", style="green")
    table.add_column("Image", style="dim", max_width=50)

    for i, (card_id, dist, meta) in enumerate(
        zip(results["ids"][0], results["distances"][0], results["metadatas"][0]), 1
    ):
        similarity = 1.0 - dist
        table.add_row(
            str(i),
            f"{similarity:.4f}",
            meta.get("full_title", meta.get("name", card_id)),
            meta.get("set_name", ""),
            meta.get("image_path", ""),
        )

    console.print(table)


def search_by_image(image_path: str, top_k: int = 10):
    """Find the top-K most similar Pokemon cards to an input image."""
    if not os.path.exists(image_path):
        console.print(f"[red]Image not found: {image_path}[/red]")
        return

    collection = get_collection()
    if collection.count() == 0:
        console.print("[yellow]No embeddings yet. Run 'generate' first.[/yellow]")
        return

    console.print("[cyan]Embedding query image with DINOv2...[/cyan]")
    query_vec = _embed_single(image_path)
    if query_vec is None:
        return

    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=min(top_k, collection.count()),
    )
    _display_results(results, f"Top {top_k} Pokemon DINOv2 Matches")


# ---------------------------------------------------------------------------
# Sync to RunPod ChromaDB
# ---------------------------------------------------------------------------

def sync_to_runpod(limit: int = 0, batch_size: int = 100):
    """Push local Pokemon embeddings to the RunPod ChromaDB via the serverless endpoint.

    This sends pre-computed embeddings (not images) to RunPod so they can be
    upserted into the remote ChromaDB collection.  The RunPod handler needs an
    'upsert' action for this to work — if your handler only supports search/embed,
    you can alternatively rsync the ChromaDB directory to the RunPod volume.
    """
    if not config.RUNPOD_API_KEY:
        console.print("[red]RUNPOD_API_KEY not set — cannot sync.[/red]")
        return
    if not config.RUNPOD_ENDPOINT_ID:
        console.print("[red]RUNPOD_ENDPOINT_ID not set — cannot sync.[/red]")
        return

    collection = get_collection()
    total_count = collection.count()
    if total_count == 0:
        console.print("[yellow]No local embeddings to sync. Run 'generate' first.[/yellow]")
        return

    console.print(f"\n[bold]Syncing {total_count} Pokemon embeddings to RunPod[/bold]")
    console.print(f"  Endpoint: [cyan]{config.RUNPOD_ENDPOINT_ID}[/cyan]")
    console.print(f"  Collection: [cyan]{COLLECTION_NAME}[/cyan]\n")

    endpoint_url = f"https://api.runpod.ai/v2/{config.RUNPOD_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    # Read all embeddings from local ChromaDB
    offset = 0
    sent = 0
    read_batch = min(batch_size, 5000)

    if limit > 0:
        to_send = limit
    else:
        to_send = total_count

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Syncing", total=min(to_send, total_count))

        while offset < total_count and sent < to_send:
            results = collection.get(
                include=["embeddings", "metadatas"],
                limit=read_batch,
                offset=offset,
            )
            ids = results["ids"]
            if not ids:
                break

            # Send batch to RunPod
            payload = {
                "input": {
                    "action": "upsert",
                    "collection": COLLECTION_NAME,
                    "ids": ids,
                    "embeddings": results["embeddings"],
                    "metadatas": results["metadatas"],
                }
            }

            try:
                resp = requests.post(endpoint_url, json=payload, headers=headers, timeout=120)
                resp.raise_for_status()
                result = resp.json()
                if result.get("status") == "COMPLETED":
                    sent += len(ids)
                    console.print(f"  [dim]Sent batch of {len(ids)} — {sent} total[/dim]")
                else:
                    console.print(f"[yellow]RunPod returned status: {result.get('status')}[/yellow]")
                    # Still count as sent to avoid infinite loop
                    sent += len(ids)
            except requests.RequestException as e:
                console.print(f"[red]RunPod request failed: {e}[/red]")
                console.print("[yellow]Tip: You can alternatively rsync the ChromaDB directory to RunPod volume.[/yellow]")
                break

            offset += len(ids)
            progress.advance(task, advance=len(ids))

    console.print(f"\n[green]Synced {sent} embeddings to RunPod[/green]")
    console.print(f"[dim]Alternative: rsync {config.CHROMA_DIR} to RunPod /data/chromadb/[/dim]")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def show_stats():
    db.init_db()

    s = db.get_pokemon_stats()
    collection = get_collection()
    emb_count = collection.count()

    console.print(f"\n[bold]Pokemon Embedding Stats[/bold]\n")
    console.print(f"  Collection:       [cyan]{COLLECTION_NAME}[/cyan]")
    console.print(f"  Model:            [cyan]DINOv2-ViT-L/14 (1024-dim, local GPU)[/cyan]")
    console.print(f"  Downloaded cards: [cyan]{s['downloaded']}[/cyan]")
    console.print(f"  Embeddings:       [cyan]{emb_count}[/cyan]")
    if s["downloaded"] > 0:
        pct = (emb_count / s["downloaded"]) * 100
        console.print(f"  Coverage:         [green]{pct:.1f}%[/green]")
    console.print(f"  ChromaDB:         [dim]{config.CHROMA_DIR}[/dim]")
    console.print(f"  Image dir:        [dim]{config.POKEMON_IMAGE_DIR}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pokemon TCG — DINOv2 Embeddings + RunPod Sync"
    )
    subparsers = parser.add_subparsers(dest="command")

    gen_p = subparsers.add_parser("generate", help="Generate DINOv2 embeddings for Pokemon cards")
    gen_p.add_argument("--limit", type=int, default=0, help="Max embeddings to generate (0=all)")
    gen_p.add_argument("--batch", type=int, default=32, help="GPU batch size (default: 32)")

    img_p = subparsers.add_parser("search", help="Search by image")
    img_p.add_argument("image", help="Path to query image")
    img_p.add_argument("--top", type=int, default=10, help="Number of results")

    subparsers.add_parser("stats", help="Show embedding stats")

    sync_p = subparsers.add_parser("sync", help="Push embeddings to RunPod ChromaDB")
    sync_p.add_argument("--limit", type=int, default=0, help="Max embeddings to sync (0=all)")
    sync_p.add_argument("--batch", type=int, default=100, help="Batch size per request")

    args = parser.parse_args()

    if args.command == "generate":
        generate_embeddings(args.limit, args.batch)
    elif args.command == "search":
        search_by_image(args.image, args.top)
    elif args.command == "stats":
        show_stats()
    elif args.command == "sync":
        sync_to_runpod(limit=args.limit, batch_size=args.batch)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
