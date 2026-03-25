#!/usr/bin/env python3
"""
Test script: Pick a downloaded card, find CLIP matches, then ask Gemini to identify it.
Compares CLIP embedding results vs Gemini vision identification.

Usage:
    python test_identify.py                     # Random downloaded card
    python test_identify.py path/to/image.jpg   # Specific image
    python test_identify.py --top 5             # Show top 5 CLIP matches

Requires:
    pip install google-genai

Set your API key:
    set GEMINI_API_KEY=your-key-here   (Windows)
    export GEMINI_API_KEY=your-key-here (Linux/Mac)
"""
import argparse
import base64
import os
import random
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

import config
import database as db
import embeddings

console = Console()


def pick_random_card() -> dict:
    """Pick a random downloaded card that has an image on disk."""
    conn = db.get_connection()
    rows = conn.execute("""
        SELECT * FROM cards
        WHERE status = 'downloaded' AND image_path IS NOT NULL
        ORDER BY RANDOM() LIMIT 10
    """).fetchall()
    conn.close()

    for r in rows:
        card = dict(r)
        if card["image_path"] and os.path.exists(card["image_path"]):
            return card

    return None


def clip_search(image_path: str, top_k: int = 5) -> list[dict]:
    """Run CLIP similarity search, return top matches with scores."""
    embeddings._load_model()
    collection = embeddings.get_collection()

    if collection.count() == 0:
        console.print("[yellow]No embeddings in ChromaDB. Run 'python embeddings.py generate' first.[/yellow]")
        return []

    vec = embeddings._embed_image(image_path)
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
            "product_id": card_id,
            "similarity": 1.0 - dist,
            "product_name": meta.get("product_name", ""),
            "set_slug": meta.get("set_slug", ""),
            "loose_price": meta.get("loose_price", 0),
        })
    return matches


def gemini_identify(image_path: str) -> str:
    """Send image to Gemini and ask it to identify the sports card."""
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

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": base64.b64encode(image_data).decode(),
                        }
                    },
                    {
                        "text": (
                            "Identify this sports trading card. Provide:\n"
                            "1. Player name\n"
                            "2. Year\n"
                            "3. Brand/manufacturer (e.g. Topps, Panini, Upper Deck)\n"
                            "4. Set name\n"
                            "5. Card number (if visible)\n"
                            "6. Any parallel/variant info\n\n"
                            "Be concise. If you can't determine something, say 'Unknown'."
                        ),
                    },
                ],
            }
        ],
    )

    return response.text


def main():
    parser = argparse.ArgumentParser(description="Test card identification: CLIP vs Gemini")
    parser.add_argument("image", nargs="?", help="Path to card image (default: random from DB)")
    parser.add_argument("--top", type=int, default=5, help="Number of CLIP results")
    args = parser.parse_args()

    # Pick image
    card = None
    if args.image:
        image_path = args.image
        if not os.path.exists(image_path):
            console.print(f"[red]File not found: {image_path}[/red]")
            sys.exit(1)
    else:
        card = pick_random_card()
        if not card:
            console.print("[red]No downloaded cards with images found.[/red]")
            sys.exit(1)
        image_path = card["image_path"]

    # Show the card info if we know it
    console.print(Panel.fit(
        f"[bold]Image:[/bold] {image_path}\n"
        + (f"[bold]DB Name:[/bold] {card['product_name']}\n"
           f"[bold]Set:[/bold] {card['set_slug']}\n"
           f"[bold]Price:[/bold] ${card['loose_price'] or 0:.2f}"
           if card else "[dim]External image (no DB record)[/dim]"),
        title="Test Card",
    ))

    # CLIP search
    console.print("\n[bold cyan]CLIP Embedding Search[/bold cyan]")
    matches = clip_search(image_path, args.top)

    if matches:
        table = Table(show_header=True)
        table.add_column("Rank", width=5)
        table.add_column("Similarity", justify="right", width=10)
        table.add_column("Name", style="cyan")
        table.add_column("Set")
        table.add_column("Price", justify="right")

        for i, m in enumerate(matches, 1):
            style = "bold green" if m["similarity"] > 0.95 else ""
            table.add_row(
                str(i),
                f"{m['similarity']:.4f}",
                m["product_name"],
                m["set_slug"],
                f"${m['loose_price']:.2f}" if m["loose_price"] else "",
                style=style,
            )
        console.print(table)
    else:
        console.print("[yellow]No CLIP results.[/yellow]")

    # Gemini identification
    console.print("\n[bold cyan]Gemini Vision Identification[/bold cyan]")
    gemini_result = gemini_identify(image_path)
    console.print(Panel(gemini_result, title="Gemini Says"))

    # Comparison hint
    if matches and card:
        top_match = matches[0]
        console.print(f"\n[bold]Match?[/bold] Top CLIP result: [cyan]{top_match['product_name']}[/cyan] "
                      f"(similarity: {top_match['similarity']:.4f})")
        if top_match["similarity"] > 0.99:
            console.print("[green]CLIP found an exact/near-exact match.[/green]")
        elif top_match["similarity"] > 0.90:
            console.print("[yellow]CLIP found a strong match - likely correct.[/yellow]")
        else:
            console.print("[red]CLIP match is weak - Gemini result may be more reliable.[/red]")


if __name__ == "__main__":
    main()
