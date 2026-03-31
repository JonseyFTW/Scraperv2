#!/usr/bin/env python3
"""
Step 3: Re-Embed Collection with Fine-Tuned DINOv2

Loads the fine-tuned backbone weights and re-embeds all cards from the manifest
into a new ChromaDB collection. Preserves the exact same metadata format your
handler expects (name, full_title, set_name, image_path, etc.).

Run from the Scraperv2 directory:
    python training/03_reembed_collection.py
    python training/03_reembed_collection.py --checkpoint ./checkpoints/dinov2_finetuned_backbone.pt
    python training/03_reembed_collection.py --collection pokemon_embeddings_dinov2_finetuned
    python training/03_reembed_collection.py --batch 64  # faster on 4070 Ti Super
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
import torch
from PIL import Image
from torchvision import transforms
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn, BarColumn,
    TaskProgressColumn, TimeElapsedColumn, TimeRemainingColumn,
)

console = Console()


def load_finetuned_model(checkpoint_path: str, device: str):
    """Load DINOv2-ViT-L/14-reg with fine-tuned weights."""
    console.print(f"[cyan]Loading DINOv2-ViT-L/14-reg + fine-tuned weights...[/cyan]")
    console.print(f"  Checkpoint: [dim]{checkpoint_path}[/dim]")

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)

    if device == "cuda":
        model = model.half()
        console.print("[green]Using fp16 for inference[/green]")

    # Standard DINOv2 preprocessing
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Warmup
    dummy = torch.randn(1, 3, 518, 518).to(device)
    if device == "cuda":
        dummy = dummy.half()
    with torch.no_grad():
        model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()

    console.print("[green]Fine-tuned model ready[/green]\n")
    return model, transform


def embed_batch(model, transform, image_paths: list[str], device: str) -> list:
    """Embed a batch of images, returning list of embedding lists (or None for failures)."""
    tensors = []
    valid_indices = []

    for i, path in enumerate(image_paths):
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            tensors.append(tensor)
            valid_indices.append(i)
        except Exception as e:
            console.print(f"[red]Error loading {path}: {e}[/red]")

    if not tensors:
        return [None] * len(image_paths)

    batch = torch.stack(tensors).to(device)
    if device == "cuda":
        batch = batch.half()

    with torch.no_grad():
        features = model(batch)
        features = features / features.norm(dim=-1, keepdim=True)

    embeddings_np = features.cpu().float().numpy()

    results = [None] * len(image_paths)
    for idx, valid_idx in enumerate(valid_indices):
        results[valid_idx] = embeddings_np[idx].flatten().tolist()

    return results


def reembed(manifest_path: str, checkpoint_path: str, chromadb_path: str,
            collection_name: str, batch_size: int):
    """Re-embed all cards with fine-tuned model into a new ChromaDB collection."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        console.print(f"[green]GPU: {gpu_name} ({vram:.1f}GB VRAM)[/green]")

    # Load manifest
    with open(manifest_path) as f:
        manifest = json.load(f)
    console.print(f"  Cards in manifest: [cyan]{len(manifest):,}[/cyan]")

    # Load model
    model, transform = load_finetuned_model(checkpoint_path, device)

    # Setup ChromaDB collection
    console.print(f"  ChromaDB path:  [cyan]{chromadb_path}[/cyan]")
    console.print(f"  Collection:     [cyan]{collection_name}[/cyan]")

    client = chromadb.PersistentClient(path=chromadb_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    existing_count = collection.count()
    if existing_count > 0:
        console.print(f"  Existing entries: [yellow]{existing_count:,}[/yellow] (will upsert/overwrite)")

    # Check which cards already embedded (for resume support)
    existing_ids = set()
    if existing_count > 0:
        all_existing = collection.get(include=[])
        existing_ids = set(all_existing["ids"])
        console.print(f"  Already embedded: [dim]{len(existing_ids):,}[/dim]")

    # Filter to cards not yet embedded
    cards_to_embed = [c for c in manifest if c["id"] not in existing_ids]
    console.print(f"  Remaining:      [cyan]{len(cards_to_embed):,}[/cyan]\n")

    if not cards_to_embed:
        console.print("[green]All cards already re-embedded![/green]")
        return

    total = 0
    skipped = 0
    start_time = time.time()

    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Re-embedding", total=len(cards_to_embed))

        for batch_start in range(0, len(cards_to_embed), batch_size):
            batch_cards = cards_to_embed[batch_start:batch_start + batch_size]

            # Collect valid image paths
            paths = []
            valid_cards = []
            for card in batch_cards:
                image_path = card.get("image_path", "")
                if image_path and os.path.exists(image_path):
                    paths.append(image_path)
                    valid_cards.append(card)
                else:
                    skipped += 1

            if paths:
                embeddings = embed_batch(model, transform, paths, device)

                chroma_ids = []
                chroma_embeddings = []
                chroma_metadatas = []

                for card, emb in zip(valid_cards, embeddings):
                    if emb is not None:
                        # Rebuild metadata in same format as pokemon_embeddings.py
                        local_id = card.get("local_id", "")
                        set_name = card.get("set_name", "")
                        full_title = card.get("full_title", f"{card['name']} ({set_name} #{local_id})")

                        chroma_ids.append(card["id"])
                        chroma_embeddings.append(emb)
                        chroma_metadatas.append({
                            "name": card["name"],
                            "full_title": full_title,
                            "set_id": card.get("set_id", ""),
                            "set_name": set_name,
                            "local_id": local_id,
                            "category": card.get("category", ""),
                            "image_path": card.get("image_path", ""),
                            "card_type": card.get("card_type", "pokemon"),
                            "source": card.get("source", ""),
                        })
                        total += 1

                if chroma_ids:
                    collection.upsert(
                        ids=chroma_ids,
                        embeddings=chroma_embeddings,
                        metadatas=chroma_metadatas,
                    )

            progress.update(task, advance=len(batch_cards),
                            description=f"Batch {batch_start // batch_size + 1} — {total:,} done")

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    final_count = collection.count()

    console.print(f"\n[green]Re-embedded {total:,} cards in {elapsed:.1f}s ({rate:.1f} img/s)[/green]")
    console.print(f"[green]Total in '{collection_name}': {final_count:,}[/green]")
    if skipped:
        console.print(f"[yellow]Skipped {skipped:,} (image file not found)[/yellow]")

    console.print(f"\n[bold]Next steps:[/bold]")
    console.print(f"  1. Copy weights to RunPod:")
    console.print(f"     scp {checkpoint_path} runpod:/runpod-volume/")
    console.print(f"  2. Sync ChromaDB to RunPod:")
    console.print(f"     rsync -avz {chromadb_path}/ runpod:/runpod-volume/chromadb/")
    console.print(f"  3. Update handler to load fine-tuned weights:")
    console.print(f'     model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")')
    console.print(f'     model.load_state_dict(torch.load("/runpod-volume/dinov2_finetuned_backbone.pt"))')
    console.print(f"  4. Set COLLECTION_NAME={collection_name} in handler (or rename collection)")


def main():
    parser = argparse.ArgumentParser(description="Re-embed cards with fine-tuned DINOv2")
    parser.add_argument("--manifest", type=str, default="./training_data/manifest.json",
                        help="Path to manifest.json from Step 1")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/dinov2_finetuned_backbone.pt",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--chromadb", type=str, default=None,
                        help="ChromaDB path (default: auto-detect from config.py)")
    parser.add_argument("--collection", type=str, default="pokemon_embeddings_dinov2_finetuned",
                        help="New collection name for fine-tuned embeddings")
    parser.add_argument("--batch", type=int, default=32,
                        help="GPU batch size for inference (default: 32, try 64 on 4070 Ti Super)")

    args = parser.parse_args()

    # Auto-detect ChromaDB path
    chromadb_path = args.chromadb
    if chromadb_path is None:
        try:
            import config
            chromadb_path = config.CHROMA_DIR
            console.print(f"[dim]Auto-detected CHROMA_DIR = {chromadb_path}[/dim]")
        except ImportError:
            console.print("[red]Cannot import config.py — specify --chromadb path[/red]")
            sys.exit(1)

    if not os.path.exists(args.manifest):
        console.print(f"[red]Manifest not found: {args.manifest}[/red]")
        console.print("[yellow]Run Step 1 first: python training/01_export_training_data.py[/yellow]")
        sys.exit(1)
    if not os.path.exists(args.checkpoint):
        console.print(f"[red]Checkpoint not found: {args.checkpoint}[/red]")
        console.print("[yellow]Run Step 2 first: python training/02_finetune_dinov2.py[/yellow]")
        sys.exit(1)

    reembed(args.manifest, args.checkpoint, chromadb_path, args.collection, args.batch)


if __name__ == "__main__":
    main()
