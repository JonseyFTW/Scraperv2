# CLIP Integration Guide for Trading Card Scanner

## Overview

This project has a pre-built CLIP embedding pipeline that can dramatically speed up card identification. Instead of sending every photo to Gemini (slow, expensive), use CLIP embeddings to find the closest match in your database first, and only fall back to Gemini when CLIP confidence is low.

## Architecture

```
User Photo → CLIP Embed (local, ~50ms) → ChromaDB Search (~10ms)
                                              │
                                    ┌─────────┴──────────┐
                                    │                      │
                              similarity > 0.90       similarity < 0.90
                                    │                      │
                              Return match           Send to Gemini Vision
                              (instant, free)        (slow, costs API call)
```

## What's Already Built

### Database (SQLite)

The scraper stores card data in `data/sportscards.db`:

```sql
cards (
    product_id      TEXT UNIQUE,    -- Use as lookup key
    set_slug        TEXT,           -- e.g. "1986-fleer-basketball"
    product_name    TEXT,           -- e.g. "Michael Jordan #57"
    console_name    TEXT,           -- Set display name
    image_path      TEXT,           -- Local path to reference image
    loose_price     REAL,           -- Ungraded price in USD
    cib_price       REAL,           -- Mid-grade price
    new_price       REAL,           -- Gem mint price
    graded_price    TEXT,           -- JSON of grade-specific prices
    status          TEXT            -- "downloaded" = has image + data
)
```

### ChromaDB Vector Store

Embeddings are stored in `data/chromadb/` using ChromaDB with persistent storage.

- **Collection name:** `card_images`
- **Distance metric:** Cosine (`hnsw:space: cosine`)
- **Vector dimensions:** 512 (from ViT-B-32)
- **ID format:** `product_id` as string

Each embedding has metadata:
```json
{
    "product_name": "Michael Jordan #57",
    "set_slug": "1986-fleer-basketball",
    "image_path": "images/1986-fleer-basketball/michael-jordan-57.jpg",
    "loose_price": 150.50
}
```

### CLIP Model

- **Model:** OpenCLIP `ViT-B-32`
- **Weights:** `laion2b_s34b_b79k`
- **Output:** 512-dimensional normalized float vector
- **Library:** `open-clip-torch`

### Image Storage

Images are saved at: `images/{set_slug}/{slugified-card-name}.{ext}`

Example: `images/1986-fleer-basketball/michael-jordan-57.jpg`

---

## How to Integrate Into Your App

### Step 1: Install Dependencies

```bash
pip install open-clip-torch torch torchvision chromadb numpy pillow
```

For Gemini fallback:
```bash
pip install google-genai
```

### Step 2: CLIP Search Function (Copy This)

This is the core function your app needs. It takes a photo and returns the best matches from the database:

```python
import open_clip
import torch
import chromadb
from PIL import Image

# --- Load once at app startup ---

model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k"
)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

chroma_client = chromadb.PersistentClient(path="data/chromadb")
collection = chroma_client.get_or_create_collection(
    name="card_images",
    metadata={"hnsw:space": "cosine"},
)


def find_card(image_path: str, top_k: int = 5) -> list[dict]:
    """
    Embed a user photo and find the closest matches in the database.

    Returns list of dicts:
        [{"product_id", "similarity", "product_name", "set_slug", "loose_price", "image_path"}, ...]
    """
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model.encode_image(img_tensor)
        features /= features.norm(dim=-1, keepdim=True)

    query_vec = features.cpu().numpy().flatten().tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, collection.count()),
    )

    matches = []
    for card_id, distance, meta in zip(
        results["ids"][0], results["distances"][0], results["metadatas"][0]
    ):
        matches.append({
            "product_id": card_id,
            "similarity": 1.0 - distance,  # cosine: 1.0 = identical
            "product_name": meta.get("product_name", ""),
            "set_slug": meta.get("set_slug", ""),
            "loose_price": meta.get("loose_price", 0),
            "image_path": meta.get("image_path", ""),
        })
    return matches
```

### Step 3: Gemini Fallback Function (Copy This)

When CLIP confidence is low, send the image to Gemini for identification:

```python
import base64
import os
from google import genai

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


def identify_with_gemini(image_path: str) -> str:
    """Send card image to Gemini for identification. Returns text description."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    ext = image_path.lower().rsplit(".", 1)[-1]
    mime = {"png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{
            "role": "user",
            "parts": [
                {"inline_data": {"mime_type": mime, "data": image_data}},
                {"text": (
                    "Identify this sports trading card. Provide:\n"
                    "1. Player name\n"
                    "2. Year\n"
                    "3. Brand/manufacturer (e.g. Topps, Panini, Upper Deck)\n"
                    "4. Set name\n"
                    "5. Card number (if visible)\n"
                    "6. Any parallel/variant info\n\n"
                    "Be concise. If unsure, say 'Unknown'."
                )},
            ],
        }],
    )
    return response.text
```

### Step 4: Combined Identification Flow (Copy This)

This is the main function your scanner should call:

```python
import sqlite3

CONFIDENCE_THRESHOLD = 0.90  # Adjust based on testing


def identify_card(image_path: str) -> dict:
    """
    Identify a trading card from a photo.

    Strategy:
        1. CLIP search for fast, free match (~60ms)
        2. If confidence > threshold, return the DB match with pricing
        3. If confidence < threshold, fall back to Gemini vision API

    Returns:
        {
            "method": "clip" | "gemini",
            "confidence": float,         # 0-1 for CLIP, None for Gemini
            "product_name": str,
            "set_slug": str,
            "loose_price": float | None,
            "gemini_response": str | None,  # Raw Gemini text if used
        }
    """
    # Step 1: Try CLIP
    matches = find_card(image_path, top_k=3)

    if matches and matches[0]["similarity"] >= CONFIDENCE_THRESHOLD:
        best = matches[0]

        # Optionally pull full pricing from SQLite
        prices = get_full_pricing(best["product_id"])

        return {
            "method": "clip",
            "confidence": best["similarity"],
            "product_id": best["product_id"],
            "product_name": best["product_name"],
            "set_slug": best["set_slug"],
            "loose_price": prices.get("loose_price"),
            "cib_price": prices.get("cib_price"),
            "new_price": prices.get("new_price"),
            "graded_price": prices.get("graded_price"),
            "gemini_response": None,
        }

    # Step 2: CLIP not confident enough, ask Gemini
    gemini_text = identify_with_gemini(image_path)

    return {
        "method": "gemini",
        "confidence": matches[0]["similarity"] if matches else 0,
        "product_id": None,
        "product_name": None,
        "set_slug": None,
        "loose_price": None,
        "cib_price": None,
        "new_price": None,
        "graded_price": None,
        "gemini_response": gemini_text,
    }


def get_full_pricing(product_id: str) -> dict:
    """Pull complete pricing from SQLite for a matched card."""
    conn = sqlite3.connect("data/sportscards.db")
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT loose_price, cib_price, new_price, graded_price FROM cards WHERE product_id=?",
        (product_id,)
    ).fetchone()
    conn.close()

    if row:
        return dict(row)
    return {}
```

---

## Similarity Score Interpretation

| Score | Meaning | Action |
|-------|---------|--------|
| **> 0.98** | Near-exact match (same card, same photo angle) | Trust fully |
| **0.93 - 0.98** | Strong match (correct card, different scan/photo) | Trust, maybe show user for confirmation |
| **0.90 - 0.93** | Likely match (right card, but could be a variant) | Show top 3 matches for user to pick |
| **0.80 - 0.90** | Weak match (similar card but probably wrong) | Fall back to Gemini |
| **< 0.80** | No match (card not in database or bad photo) | Fall back to Gemini |

Adjust `CONFIDENCE_THRESHOLD` based on your testing. Start at **0.90** and lower it if CLIP is accurate enough at lower scores.

---

## Performance Comparison

| Method | Speed | Cost | Accuracy |
|--------|-------|------|----------|
| **CLIP only** | ~60ms | Free (local) | High for exact matches, poor for cards not in DB |
| **Gemini only** | 2-5 sec | ~$0.002/call | Good general identification, but no pricing data |
| **CLIP + Gemini fallback** | 60ms (90%+ of cases) | Near-zero | Best of both worlds |

---

## Text Search (Bonus)

CLIP also supports text-to-image search. Your app can let users type a description:

```python
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def search_by_description(text: str, top_k: int = 10) -> list[dict]:
    """Find cards matching a text description like '1986 Fleer Michael Jordan rookie'."""
    tokens = tokenizer([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    query_vec = features.cpu().numpy().flatten().tolist()

    results = collection.query(
        query_embeddings=[query_vec],
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
```

---

## Key Implementation Notes

1. **Load the model once at startup** — it takes several seconds. Don't reload per request.
2. **ChromaDB path must match** — point to the same `data/chromadb/` directory the scraper populates.
3. **SQLite is for full data** — ChromaDB metadata only has name/set/price. Query SQLite by `product_id` for complete card info.
4. **Images must be RGB** — always `.convert("RGB")` before embedding.
5. **Vectors are L2-normalized** — ChromaDB cosine distance ranges 0-2; similarity = `1.0 - distance`.
6. **The embedding DB grows with scraping** — as more cards are scraped and embedded, CLIP accuracy improves. Re-run `python embeddings.py generate` after new downloads.
