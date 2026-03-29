#!/usr/bin/env python3
"""
Test script: Pick a random card, run DINOv2 similarity search, show scores.

Supports both sports cards and Pokemon cards.  Randomly picks from whichever
collection has embeddings (or use --mode to force one).

Usage:
    python test_identify.py                        # Random card from any collection
    python test_identify.py --mode pokemon          # Random Pokemon card
    python test_identify.py --mode sports           # Random sports card
    python test_identify.py path/to/image.jpg       # Specific image (searches both)
    python test_identify.py --top 10                # Show top 10 matches
    python test_identify.py --gemini                # Also run Gemini vision ID

Requires:
    pip install torch torchvision chromadb rich psycopg2-binary
    pip install google-genai  (optional, for --gemini)

Set your API key for Gemini:
    export GEMINI_API_KEY=your-key-here
"""
import argparse
import base64
import os
import random
import sys

import chromadb
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
import database as db

console = Console()

# ---------------------------------------------------------------------------
# DINOv2 model (shared between sports + pokemon, same architecture)
# ---------------------------------------------------------------------------

_model = None
_transform = None
_device = None


def _load_dinov2():
    """Load DINOv2-ViT-L/14 once for all searches."""
    global _model, _transform, _device
    if _model is not None:
        return

    import torch
    from torchvision import transforms

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[cyan]Loading DINOv2-ViT-L/14 on {_device}...[/cyan]")

    _model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    _model.eval()
    _model = _model.to(_device)

    if _device == "cuda":
        _model = _model.half()

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

    console.print(f"[green]DINOv2 ready on {_device}[/green]")


def _embed_image(image_path: str):
    """Generate a 1024-dim DINOv2 embedding for a single image."""
    import torch
    from PIL import Image

    _load_dinov2()

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


# ---------------------------------------------------------------------------
# Card pickers
# ---------------------------------------------------------------------------

def _resolve_path(image_path: str) -> str:
    """Translate a DB image path to the local filesystem."""
    if not image_path:
        return image_path
    if os.path.exists(image_path):
        return image_path
    prefix = config.LINUX_DATA_PREFIX
    if prefix and image_path.startswith(prefix):
        return os.path.join(config.DATA_DIR, image_path[len(prefix):].lstrip("/\\"))
    return os.path.join(config.IMAGE_DIR, os.path.basename(image_path))


def pick_random_sports_card() -> dict | None:
    """Pick a random downloaded sports card with an image on disk."""
    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
    cur.execute("""
        SELECT * FROM cards
        WHERE status = 'downloaded' AND image_path IS NOT NULL
        ORDER BY RANDOM() LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    db.put_connection(conn)

    for r in rows:
        card = dict(r)
        resolved = _resolve_path(card["image_path"])
        if resolved and os.path.exists(resolved):
            card["_resolved_path"] = resolved
            card["_card_type"] = "sports"
            card["_display_name"] = card.get("product_name") or card.get("product_id") or "Unknown"
            card["_display_set"] = card.get("set_slug") or ""
            card["_display_price"] = f"${card.get('loose_price') or 0:.2f}"
            return card
    return None


def pick_random_pokemon_card() -> dict | None:
    """Pick a random downloaded Pokemon card with an image on disk."""
    conn = db.get_connection()
    cur = conn.cursor(cursor_factory=__import__('psycopg2').extras.RealDictCursor)
    cur.execute("""
        SELECT * FROM pokemon_cards
        WHERE status = 'downloaded' AND image_path IS NOT NULL
        ORDER BY RANDOM() LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    db.put_connection(conn)

    for r in rows:
        card = dict(r)
        if card["image_path"] and os.path.exists(card["image_path"]):
            card["_resolved_path"] = card["image_path"]
            card["_card_type"] = "pokemon"
            card["_display_name"] = f"{card.get('name', '?')} ({card.get('set_name', '?')} #{card.get('local_id', '?')})"
            card["_display_set"] = card.get("set_name") or ""
            card["_display_price"] = ""
            return card
    return None


def pick_random_card(mode: str = "any") -> dict | None:
    """Pick a random card. mode: 'sports', 'pokemon', or 'any'."""
    if mode == "sports":
        return pick_random_sports_card()
    if mode == "pokemon":
        return pick_random_pokemon_card()

    # 'any' — try both, pick randomly
    pickers = [pick_random_sports_card, pick_random_pokemon_card]
    random.shuffle(pickers)
    for picker in pickers:
        card = picker()
        if card:
            return card
    return None


# ---------------------------------------------------------------------------
# DINOv2 search
# ---------------------------------------------------------------------------

def dinov2_search(image_path: str, collection_name: str, top_k: int = 5) -> list[dict]:
    """Run DINOv2 similarity search against a ChromaDB collection."""
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return []

    if collection.count() == 0:
        return []

    vec = _embed_image(image_path)
    if vec is None:
        return []

    results = collection.query(
        query_embeddings=[vec.tolist()],
        n_results=min(top_k, collection.count()),
    )

    matches = []
    for card_id, dist, meta in zip(
        results["ids"][0], results["distances"][0], results["metadatas"][0]
    ):
        matches.append({
            "id": card_id,
            "similarity": 1.0 - dist,
            "name": meta.get("full_title") or meta.get("product_name") or meta.get("name") or card_id,
            "set": meta.get("set_name") or meta.get("set_slug") or "",
            "price": meta.get("loose_price", 0),
            "image_path": meta.get("image_path", ""),
            "card_type": meta.get("card_type", ""),
        })
    return matches


def display_matches(matches: list[dict], title: str):
    """Show search results in a rich table with color-coded similarity."""
    table = Table(title=title, show_header=True)
    table.add_column("Rank", width=5)
    table.add_column("Similarity", justify="right", width=10)
    table.add_column("Name", style="cyan", max_width=50)
    table.add_column("Set", style="green", max_width=30)
    table.add_column("Price", justify="right", width=8)
    table.add_column("Self?", width=5)

    for i, m in enumerate(matches, 1):
        sim = m["similarity"]
        if sim > 0.99:
            style = "bold green"
            label = "YES"
        elif sim > 0.95:
            style = "green"
            label = ""
        elif sim > 0.90:
            style = "yellow"
            label = ""
        else:
            style = "dim"
            label = ""

        price_str = f"${m['price']:.2f}" if m["price"] else ""
        table.add_row(
            str(i), f"{sim:.4f}", m["name"], m["set"], price_str, label,
            style=style,
        )

    console.print(table)

    # Summary
    if matches:
        top = matches[0]
        console.print()
        if top["similarity"] > 0.99:
            console.print("[bold green]Exact/near-exact match found (>0.99)[/bold green]")
        elif top["similarity"] > 0.95:
            console.print("[green]Strong match (>0.95) — very likely correct[/green]")
        elif top["similarity"] > 0.90:
            console.print("[yellow]Good match (>0.90) — probably correct[/yellow]")
        elif top["similarity"] > 0.80:
            console.print("[yellow]Moderate match (>0.80) — might be correct[/yellow]")
        else:
            console.print("[red]Weak match (<0.80) — card may not be in the database[/red]")


# ---------------------------------------------------------------------------
# Gemini (optional)
# ---------------------------------------------------------------------------

def gemini_identify(image_path: str, card_type: str = "sports") -> str:
    """Send image to Gemini and ask it to identify the card."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return "[ERROR] Set GEMINI_API_KEY environment variable first."

    try:
        from google import genai
    except ImportError:
        return "[ERROR] Install google-genai: pip install google-genai"

    client = genai.Client(api_key=api_key)

    with open(image_path, "rb") as f:
        image_data = f.read()

    mime = "image/jpeg"
    if image_path.lower().endswith(".png"):
        mime = "image/png"
    elif image_path.lower().endswith(".webp"):
        mime = "image/webp"

    if card_type == "pokemon":
        prompt = (
            "Identify this Pokemon Trading Card Game card. Provide:\n"
            "1. Pokemon name\n"
            "2. Set name\n"
            "3. Card number\n"
            "4. Rarity\n"
            "5. Any variant info (holo, reverse holo, full art, etc.)\n\n"
            "Be concise. If you can't determine something, say 'Unknown'."
        )
    else:
        prompt = (
            "Identify this sports trading card. Provide:\n"
            "1. Player name\n"
            "2. Year\n"
            "3. Brand/manufacturer (e.g. Topps, Panini, Upper Deck)\n"
            "4. Set name\n"
            "5. Card number (if visible)\n"
            "6. Any parallel/variant info\n\n"
            "Be concise. If you can't determine something, say 'Unknown'."
        )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": mime, "data": base64.b64encode(image_data).decode()}},
                    {"text": prompt},
                ],
            }
        ],
    )
    return response.text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test card identification: pick a random card, search with DINOv2, show scores"
    )
    parser.add_argument("image", nargs="?", help="Path to card image (default: random from DB)")
    parser.add_argument("--mode", choices=["sports", "pokemon", "any"], default="any",
                        help="Which card collection to test (default: any)")
    parser.add_argument("--top", type=int, default=5, help="Number of results to show")
    parser.add_argument("--gemini", action="store_true", help="Also run Gemini vision identification")
    args = parser.parse_args()

    db.init_db()

    # Pick image
    card = None
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            console.print(f"[red]File not found: {image_path}[/red]")
            sys.exit(1)
        card_type = args.mode if args.mode != "any" else "sports"
    else:
        card = pick_random_card(mode=args.mode)
        if not card:
            console.print(f"[red]No downloaded cards with images found (mode={args.mode}).[/red]")
            console.print("[yellow]Run 'python pokemon_scraper.py run' or check sports card downloads.[/yellow]")
            sys.exit(1)
        image_path = card["_resolved_path"]
        card_type = card["_card_type"]

    # Show the card info
    if card:
        info = (
            f"[bold]Image:[/bold] {image_path}\n"
            f"[bold]Type:[/bold] {card_type.upper()}\n"
            f"[bold]Name:[/bold] {card['_display_name']}\n"
            f"[bold]Set:[/bold] {card['_display_set']}"
        )
        if card["_display_price"]:
            info += f"\n[bold]Price:[/bold] {card['_display_price']}"
    else:
        info = (
            f"[bold]Image:[/bold] {image_path}\n"
            "[dim]External image (no DB record)[/dim]"
        )

    console.print(Panel.fit(info, title="Test Card"))

    # Determine which collections to search
    collections_to_search = []
    if card_type == "pokemon" or args.mode == "any":
        collections_to_search.append(("pokemon_embeddings_dinov2", "Pokemon DINOv2"))
    if card_type == "sports" or args.mode == "any":
        collections_to_search.append(("card_embeddings_dinov2", "Sports DINOv2"))

    # If a specific image was provided and mode=any, search both
    if args.image and args.mode == "any":
        collections_to_search = [
            ("pokemon_embeddings_dinov2", "Pokemon DINOv2"),
            ("card_embeddings_dinov2", "Sports DINOv2"),
        ]

    # Run DINOv2 search on each collection
    for coll_name, coll_label in collections_to_search:
        console.print(f"\n[bold cyan]{coll_label} Search[/bold cyan]")
        matches = dinov2_search(image_path, coll_name, top_k=args.top)

        if matches:
            display_matches(matches, f"Top {args.top} {coll_label} Matches")
        else:
            console.print(f"[yellow]No embeddings in '{coll_name}'. Run generate first.[/yellow]")

    # Gemini (optional)
    if args.gemini:
        console.print(f"\n[bold cyan]Gemini Vision Identification[/bold cyan]")
        gemini_result = gemini_identify(image_path, card_type)
        console.print(Panel(gemini_result, title="Gemini Says"))


if __name__ == "__main__":
    main()
