#!/usr/bin/env python3
"""
Step 1: Export Training Data from ChromaDB

Reads your existing ChromaDB collection (pokemon_embeddings_dinov2) and builds:
  - training_data/manifest.json    — every card with id, name, image_path, set_name
  - training_data/hard_negatives.json — groups of same-character cards for contrastive training

Run from the Scraperv2 directory so config.py is importable:
    python training/01_export_training_data.py
    python training/01_export_training_data.py --chromadb /path/to/chromadb
    python training/01_export_training_data.py --collection card_embeddings_dinov2
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

# Add parent dir so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from rich.console import Console

console = Console()


def extract_character_name(card_name: str) -> str:
    """Extract the base character name for hard negative grouping.

    Examples:
        "Charizard ex"        -> "charizard"
        "Charizard V"         -> "charizard"
        "Charizard VMAX"      -> "charizard"
        "Charizard GX"        -> "charizard"
        "Dark Charizard"      -> "charizard"
        "Shining Charizard"   -> "charizard"
        "Pikachu"             -> "pikachu"
        "Detective Pikachu"   -> "pikachu"
    """
    name = card_name.lower().strip()

    # Remove common suffixes: ex, gx, v, vmax, vstar, ex, BREAK, etc.
    name = re.sub(r'\b(ex|gx|vmax|vstar|v|lv\.\s*x|break|prime|legend|radiant|shiny)\b', '', name)

    # Remove common prefixes
    name = re.sub(r'^(dark|light|shining|detective|team aqua\'?s?|team magma\'?s?|team rocket\'?s?|lt\.\s*surge\'?s?|brock\'?s?|misty\'?s?|erika\'?s?|sabrina\'?s?|blaine\'?s?|giovanni\'?s?|koga\'?s?|jasmine\'?s?|chuck\'?s?|pryce\'?s?|clair\'?s?|galarian|hisuian|alolan|paldean)\s+', '', name)

    # Remove trailing punctuation and whitespace
    name = re.sub(r'[^a-z0-9\s]', '', name).strip()
    name = re.sub(r'\s+', ' ', name)

    return name


def export_from_chromadb(chromadb_path: str, collection_name: str, output_dir: str):
    """Read all entries from ChromaDB and build training manifest + hard negatives."""
    console.print(f"\n[bold]Exporting training data from ChromaDB[/bold]")
    console.print(f"  ChromaDB path:  [cyan]{chromadb_path}[/cyan]")
    console.print(f"  Collection:     [cyan]{collection_name}[/cyan]")
    console.print(f"  Output dir:     [cyan]{output_dir}[/cyan]\n")

    client = chromadb.PersistentClient(path=chromadb_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        console.print(f"[red]Could not open collection '{collection_name}': {e}[/red]")
        console.print("[yellow]Available collections:[/yellow]")
        for col in client.list_collections():
            col_name = col.name if hasattr(col, 'name') else str(col)
            console.print(f"  - {col_name}")
        sys.exit(1)

    total = collection.count()
    console.print(f"  Total entries:  [cyan]{total:,}[/cyan]\n")

    if total == 0:
        console.print("[red]Collection is empty. Run embedding generation first.[/red]")
        sys.exit(1)

    # Read all entries in batches (ChromaDB get() has limits)
    manifest = []
    skipped = 0
    batch_size = 5000
    offset = 0

    while offset < total:
        results = collection.get(
            include=["metadatas"],
            limit=batch_size,
            offset=offset,
        )

        for card_id, meta in zip(results["ids"], results["metadatas"]):
            image_path = meta.get("image_path", "")
            # Support both Pokemon format ("name") and sports card format ("product_name")
            name = meta.get("name", "") or meta.get("product_name", "")

            if not image_path or not name:
                skipped += 1
                continue

            # Strip card number from product_name (e.g., "Ahmad Rashad #383" -> "Ahmad Rashad")
            clean_name = re.sub(r'\s*#\d+\s*$', '', name).strip()

            manifest.append({
                "id": card_id,
                "name": clean_name,
                "full_title": meta.get("full_title", name),
                "set_name": meta.get("set_name", "") or meta.get("set_slug", ""),
                "set_id": meta.get("set_id", ""),
                "local_id": meta.get("local_id", ""),
                "image_path": image_path,
                "category": meta.get("category", ""),
                "card_type": meta.get("card_type", ""),
                "source": meta.get("source", ""),
            })

        offset += len(results["ids"])
        console.print(f"  Read {min(offset, total):,} / {total:,} entries", end="\r")

    console.print(f"\n  Valid cards:    [green]{len(manifest):,}[/green]")
    if skipped:
        console.print(f"  Skipped:        [yellow]{skipped:,}[/yellow] (missing name or image_path)")

    # Verify images exist
    missing = 0
    valid_manifest = []
    for card in manifest:
        if os.path.exists(card["image_path"]):
            valid_manifest.append(card)
        else:
            missing += 1

    if missing:
        console.print(f"  Missing images: [yellow]{missing:,}[/yellow] (files not found on disk)")
    manifest = valid_manifest
    console.print(f"  Final count:    [green]{len(manifest):,}[/green]\n")

    # Build hard negatives — group cards by character name
    char_groups = defaultdict(list)
    for card in manifest:
        char_name = extract_character_name(card["name"])
        if char_name:
            char_groups[char_name].append(card["id"])

    # Only keep groups with 2+ cards (need at least 2 for hard negatives)
    hard_negatives = {
        name: ids for name, ids in char_groups.items()
        if len(ids) >= 2
    }

    # Stats
    total_in_groups = sum(len(ids) for ids in hard_negatives.values())
    console.print(f"[bold]Hard Negative Mining[/bold]")
    console.print(f"  Character groups:    [cyan]{len(hard_negatives):,}[/cyan]")
    console.print(f"  Cards in groups:     [cyan]{total_in_groups:,}[/cyan]")
    console.print(f"  Cards without group: [dim]{len(manifest) - total_in_groups:,}[/dim]")

    # Show top groups
    sorted_groups = sorted(hard_negatives.items(), key=lambda x: len(x[1]), reverse=True)
    console.print(f"\n  Top 10 hard negative groups:")
    for name, ids in sorted_groups[:10]:
        console.print(f"    {name}: [cyan]{len(ids)}[/cyan] cards")

    # Write output
    os.makedirs(output_dir, exist_ok=True)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    console.print(f"\n  Wrote [green]{manifest_path}[/green] ({len(manifest):,} cards)")

    hn_path = os.path.join(output_dir, "hard_negatives.json")
    with open(hn_path, "w") as f:
        json.dump(hard_negatives, f, indent=2)
    console.print(f"  Wrote [green]{hn_path}[/green] ({len(hard_negatives):,} groups)")

    console.print(f"\n[green]Done! Ready for Step 2: python training/02_finetune_dinov2.py[/green]")


def main():
    parser = argparse.ArgumentParser(description="Export training data from ChromaDB")
    parser.add_argument("--chromadb", type=str, default=None,
                        help="Path to ChromaDB directory (default: auto-detect from config.py)")
    parser.add_argument("--collection", type=str, default=None,
                        help="ChromaDB collection name (default: pokemon_embeddings_dinov2)")
    parser.add_argument("--output", type=str, default="./training_data",
                        help="Output directory for manifest + hard negatives")

    args = parser.parse_args()

    # Auto-detect paths from config.py if not specified
    chromadb_path = args.chromadb
    collection_name = args.collection

    if chromadb_path is None or collection_name is None:
        try:
            import config
            if chromadb_path is None:
                chromadb_path = config.CHROMA_DIR
            if collection_name is None:
                collection_name = config.POKEMON_CHROMA_COLLECTION
            console.print(f"[dim]Auto-detected from config.py:[/dim]")
            console.print(f"[dim]  CHROMA_DIR = {chromadb_path}[/dim]")
            console.print(f"[dim]  COLLECTION = {collection_name}[/dim]")
        except ImportError:
            if chromadb_path is None:
                console.print("[red]Cannot import config.py — specify --chromadb path[/red]")
                sys.exit(1)
            if collection_name is None:
                collection_name = "pokemon_embeddings_dinov2"

    export_from_chromadb(chromadb_path, collection_name, args.output)


if __name__ == "__main__":
    main()
