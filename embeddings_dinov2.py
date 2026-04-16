#!/usr/bin/env python3
"""
SportsCardPro Scraper - DINOv2 Embedding Generator (ChromaDB)

Generates vector embeddings for downloaded card images using DINOv2-ViT-L/14.
Runs locally on GPU with batched inference for maximum throughput.
Stores embeddings in ChromaDB for fast indexed similarity search.

Drop-in replacement for embeddings.py — uses DINOv2 (1024-dim) instead of
CLIP ViT-B-32 (512-dim). Stores in a SEPARATE collection since dimensions differ.

Usage:
    python embeddings_dinov2.py generate              # Generate embeddings for all downloaded cards
    python embeddings_dinov2.py generate --limit 100  # Generate only 100 embeddings (for testing)
    python embeddings_dinov2.py generate --batch 64   # Custom batch size (default: 32)
    python embeddings_dinov2.py search image.jpg      # Find top matches for an input image
    python embeddings_dinov2.py stats                 # Show embedding coverage

Requirements:
    pip install torch torchvision numpy chromadb rich

Note: DINOv2 is vision-only — no text search support (use CLIP for that).
"""
import argparse
import os
import sys
import time

import chromadb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

import config
import database as db

console = Console()

COLLECTION_NAME = "card_embeddings_dinov2_finetuned"

# Lazy-loaded model globals
_model = None
_transform = None
_device = None
_checkpoint_path = None


def _build_card_metadata(card: dict) -> dict:
    """Build the ChromaDB metadata dict for a card row.

    ChromaDB requires scalar values — coerce None -> "" or 0 as appropriate.
    The 0 sentinel for print_run means "unknown"; no real card has /0.
    """
    return {
        "product_name":  card.get("product_name")  or "",
        "set_slug":      card.get("set_slug")      or "",
        "image_path":    card.get("image_path")    or "",
        "gcs_image_url": card.get("gcs_image_url") or "",
        "gcs_thumb_url": card.get("gcs_thumb_url") or "",
        "card_number":   card.get("card_number")   or "",
        "print_run":     int(card["print_run"]) if card.get("print_run") else 0,
        "player_name":   (card.get("player_name") or "").lower(),
        "variant_label": card.get("variant_label") or "",
        "loose_price":   float(card.get("loose_price") or 0),
    }


def _load_model():
    """Load DINOv2-ViT-L/14 on GPU with fp16 for speed. Loads fine-tuned weights if --checkpoint was provided."""
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

    # Use register-token variant if loading fine-tuned weights (training uses dinov2_vitl14_reg)
    if _checkpoint_path:
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
        console.print(f"[cyan]Loading fine-tuned weights from {_checkpoint_path}...[/cyan]")
        state_dict = torch.load(_checkpoint_path, map_location=_device, weights_only=True)
        _model.load_state_dict(state_dict, strict=False)
        console.print(f"[green]Fine-tuned model loaded[/green]")
    else:
        _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    _model.eval()
    _model = _model.to(_device)

    # Use fp16 on GPU for ~2x speedup
    if _device == "cuda":
        _model = _model.half()
        console.print("[green]Using fp16 (half precision) for faster inference[/green]")

    # DINOv2 expects 518x518 images (14px patches * 37 = 518)
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
    """Generate a 1024-dim embedding for a single image. Used for search."""
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

    # Map back to original indices
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


def get_cards_needing_embeddings(limit: int = 0) -> list[dict]:
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

    if limit > 0:
        return cards[:limit]
    return cards


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def generate_embeddings(limit: int = 0, batch_size: int = 32):
    """Generate DINOv2 embeddings using local GPU with batched inference."""
    import torch

    console.print(f"\n[bold]Generating DINOv2 embeddings — local GPU, batch_size={batch_size}[/bold]")
    console.print(f"  Collection: [cyan]{COLLECTION_NAME}[/cyan]\n")

    _load_model()
    collection = get_collection()

    cards = get_cards_needing_embeddings(limit)
    if not cards:
        console.print("[green]All cards already have embeddings![/green]")
        return

    console.print(f"  Cards to embed: [cyan]{len(cards)}[/cyan]\n")

    total = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding", total=len(cards))

        # Process in batches
        for batch_start in range(0, len(cards), batch_size):
            batch_cards = cards[batch_start:batch_start + batch_size]

            # Resolve paths and filter
            paths = []
            valid_cards = []
            for card in batch_cards:
                resolved = _resolve_image_path(card.get("image_path", ""))
                if resolved and os.path.exists(resolved):
                    paths.append(resolved)
                    valid_cards.append(card)
                else:
                    _skipped_ids.add(str(card["product_id"]))

            if paths:
                embeddings = _embed_batch(paths)

                chroma_ids = []
                chroma_embeddings = []
                chroma_metadatas = []

                for card, emb in zip(valid_cards, embeddings):
                    if emb is not None:
                        chroma_ids.append(str(card["product_id"]))
                        chroma_embeddings.append(emb)
                        chroma_metadatas.append(_build_card_metadata(card))
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

    global _chroma_client, _chroma_collection
    final_count = collection.count()
    if _chroma_client is not None:
        del _chroma_collection
        del _chroma_client
        _chroma_collection = None
        _chroma_client = None

    console.print(f"\n[green]Generated {total} embeddings in {elapsed:.1f}s ({rate:.1f} img/s)[/green]")
    console.print(f"[green]{final_count} total in ChromaDB[/green]")
    if _skipped_ids:
        console.print(f"[yellow]Skipped {len(_skipped_ids)} cards (image file not found)[/yellow]")


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

    console.print(f"[cyan]Embedding query image with DINOv2...[/cyan]")
    query_vec = _embed_single(image_path)
    if query_vec is None:
        return

    console.print(f"[cyan]Searching {top_k} closest matches...[/cyan]")
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
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
    console.print(f"  Model:            [cyan]DINOv2-ViT-L/14 (1024-dim, local GPU)[/cyan]")
    console.print(f"  Downloaded cards: [cyan]{total_cards}[/cyan]")
    console.print(f"  Embeddings:       [cyan]{total_embs}[/cyan]")
    if total_cards > 0:
        pct = (total_embs / total_cards) * 100
        console.print(f"  Coverage:         [green]{pct:.1f}%[/green]")
    console.print(f"  Storage:          [dim]{config.CHROMA_DIR}[/dim]")


def migrate_collection():
    """Migrate embeddings from old collection name to the current one."""
    OLD_NAME = "card_images_dinov2"

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    try:
        old_col = client.get_collection(name=OLD_NAME)
    except Exception:
        console.print(f"[yellow]Old collection '{OLD_NAME}' not found — nothing to migrate.[/yellow]")
        return

    old_count = old_col.count()
    if old_count == 0:
        console.print(f"[yellow]Old collection '{OLD_NAME}' is empty.[/yellow]")
        return

    new_col = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing = set(new_col.get(include=[])["ids"]) if new_col.count() > 0 else set()

    console.print(f"[cyan]Migrating {old_count} embeddings: {OLD_NAME} → {COLLECTION_NAME}[/cyan]")

    batch_size = 5000
    migrated = 0

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  BarColumn(), TaskProgressColumn(), console=console) as progress:
        task = progress.add_task("Migrating", total=old_count)
        offset = 0

        while offset < old_count:
            results = old_col.get(
                include=["embeddings", "metadatas"],
                limit=batch_size,
                offset=offset,
            )
            ids = results["ids"]
            if not ids:
                break

            # Filter out already-migrated
            new_ids = []
            new_embs = []
            new_metas = []
            for i, cid in enumerate(ids):
                if cid not in existing:
                    new_ids.append(cid)
                    new_embs.append(results["embeddings"][i])
                    new_metas.append(results["metadatas"][i])

            if new_ids:
                new_col.upsert(ids=new_ids, embeddings=new_embs, metadatas=new_metas)
                migrated += len(new_ids)

            offset += len(ids)
            progress.update(task, advance=len(ids),
                            description=f"Migrated {migrated}")

    console.print(f"[green]Migrated {migrated} embeddings to '{COLLECTION_NAME}'[/green]")
    console.print(f"[green]{new_col.count()} total in new collection[/green]")

    # Delete old collection
    client.delete_collection(name=OLD_NAME)
    console.print(f"[dim]Deleted old collection '{OLD_NAME}'[/dim]")


def sync_to_runpod(limit: int = 0, batch_size: int = 500):
    """Push local sports card embeddings to RunPod ChromaDB via the serverless upsert action."""
    import requests

    if not config.RUNPOD_API_KEY:
        console.print("[red]RUNPOD_API_KEY not set. Export it first.[/red]")
        return
    if not config.RUNPOD_ENDPOINT_ID:
        console.print("[red]RUNPOD_ENDPOINT_ID not set.[/red]")
        return

    collection = get_collection()
    total_count = collection.count()
    if total_count == 0:
        console.print("[yellow]No local embeddings to sync. Run 'generate' first.[/yellow]")
        return

    to_send = limit if limit > 0 else total_count
    console.print(f"\n[bold]Syncing sports card embeddings to RunPod[/bold]")
    console.print(f"  Endpoint:   [cyan]{config.RUNPOD_ENDPOINT_ID}[/cyan]")
    console.print(f"  Collection: [cyan]{COLLECTION_NAME}[/cyan]")
    console.print(f"  Local count: [cyan]{total_count}[/cyan]")
    console.print(f"  Batch size: [cyan]{batch_size}[/cyan]\n")

    endpoint_url = f"https://api.runpod.ai/v2/{config.RUNPOD_ENDPOINT_ID}/runsync"
    headers = {
        "Authorization": f"Bearer {config.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
    offset = 0
    sent = 0

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Syncing", total=min(to_send, total_count))

        while offset < total_count and sent < to_send:
            chunk = min(batch_size, to_send - sent)
            results = collection.get(
                include=["embeddings", "metadatas"],
                limit=chunk,
                offset=offset,
            )
            ids = results["ids"]
            if not ids:
                break

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
                else:
                    console.print(f"[yellow]RunPod status: {result.get('status')} — continuing[/yellow]")
                    sent += len(ids)
            except requests.RequestException as e:
                console.print(f"[red]RunPod request failed: {e}[/red]")
                console.print("[yellow]Tip: You can also rsync the ChromaDB dir via a temporary GPU Pod.[/yellow]")
                break

            offset += len(ids)
            progress.update(task, advance=len(ids),
                            description=f"Sent {sent}/{min(to_send, total_count)}")

    console.print(f"\n[green]Synced {sent} embeddings to RunPod[/green]")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Embedding Generator for Sports Cards (Local GPU)")
    subparsers = parser.add_subparsers(dest="command")

    gen_parser = subparsers.add_parser("generate", help="Generate DINOv2 embeddings for downloaded cards")
    gen_parser.add_argument("--limit", type=int, default=0, help="Max embeddings to generate (0=all)")
    gen_parser.add_argument("--batch", type=int, default=32, help="GPU batch size (default: 32)")
    gen_parser.add_argument("--checkpoint", type=str, default=None,
                            help="Path to fine-tuned model checkpoint (uses base DINOv2 if not specified)")

    img_parser = subparsers.add_parser("search", help="Search by image")
    img_parser.add_argument("image", help="Path to query image")
    img_parser.add_argument("--top", type=int, default=10, help="Number of results")
    img_parser.add_argument("--checkpoint", type=str, default=None,
                            help="Path to fine-tuned model checkpoint (uses base DINOv2 if not specified)")

    subparsers.add_parser("stats", help="Show embedding stats")
    subparsers.add_parser("migrate", help="Migrate from old collection name (card_images_dinov2)")

    sync_parser = subparsers.add_parser("sync", help="Push embeddings to RunPod ChromaDB")
    sync_parser.add_argument("--limit", type=int, default=0, help="Max embeddings to sync (0=all)")
    sync_parser.add_argument("--batch", type=int, default=500, help="Batch size per API request (default: 500)")

    args = parser.parse_args()

    global _checkpoint_path
    if hasattr(args, "checkpoint") and args.checkpoint:
        _checkpoint_path = args.checkpoint

    if args.command == "generate":
        generate_embeddings(args.limit, args.batch)
    elif args.command == "search":
        search_by_image(args.image, args.top)
    elif args.command == "stats":
        show_embedding_stats()
    elif args.command == "migrate":
        migrate_collection()
    elif args.command == "sync":
        sync_to_runpod(limit=args.limit, batch_size=args.batch)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
