"""
DINOv2-ViT-L/14 Embedding Search — RunPod Serverless Handler

Accepts a base64-encoded card image, generates a DINOv2 embedding (1024-dim),
queries ChromaDB for the closest matches, and returns full card data
from the SQLite database.

Deployed as a RunPod serverless endpoint. The container stays warm between
requests, so model loading only happens on cold start (~5-8s).
"""

import base64
import io
import json
import os
import random
import sqlite3
import time
from collections import Counter

import chromadb
import numpy as np
import runpod
import torch
from PIL import Image

# --- Configuration ---
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "/runpod-volume/chromadb")
SQLITE_PATH = os.environ.get("SQLITE_PATH", "/runpod-volume/sportscards.db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "card_embeddings_dinov2")
TOP_K = int(os.environ.get("DINOV2_TOP_K", "5"))

# --- Global state (loaded once on cold start) ---
model = None
transform = None
device = None
collection = None


def load_model():
    """Load DINOv2-ViT-L/14 model and ChromaDB collection."""
    global model, transform, device, collection

    print("[DINOv2] Loading DINOv2-ViT-L/14 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load DINOv2-ViT-L/14 from torch hub (Meta's official weights)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
    model.eval()
    model = model.to(device)
    print(f"[DINOv2] Model loaded on {device} — 1024-dim embeddings")

    # DINOv2 expects 518x518 images (14px patches * 37 patches = 518)
    # Using standard ImageNet normalization
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(518),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Connect to ChromaDB (local persistent storage on the RunPod volume)
    print(f"[DINOv2] Connecting to ChromaDB at {CHROMADB_PATH}...")
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    count = collection.count()
    print(f"[DINOv2] ChromaDB collection '{COLLECTION_NAME}' has {count:,} embeddings")

    # Warm up with dummy inference
    print("[DINOv2] Warming up model...")
    dummy = torch.randn(1, 3, 518, 518).to(device)
    with torch.no_grad():
        model(dummy)
    print("[DINOv2] Ready for requests")


def get_sqlite_data(product_id: str) -> dict | None:
    """Look up full card data from SQLite by product_id."""
    if not os.path.exists(SQLITE_PATH):
        return None
    try:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT product_name, set_slug, console_name, loose_price, cib_price, "
            "new_price, graded_price, image_path FROM cards WHERE product_id = ?",
            (product_id,),
        ).fetchone()
        conn.close()
        if row:
            result = dict(row)
            if result.get("graded_price"):
                try:
                    result["graded_price"] = json.loads(result["graded_price"])
                except (json.JSONDecodeError, TypeError):
                    result["graded_price"] = None
            return result
    except Exception as e:
        print(f"[DINOv2] SQLite lookup error: {e}")
    return None


def embed_image(image_bytes: bytes) -> list[float]:
    """Generate a 1024-dim DINOv2 embedding from raw image bytes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        # DINOv2 returns CLS token as the image-level embedding
        features = model(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def embed_pil_image(img: Image.Image) -> list[float]:
    """Generate a 1024-dim DINOv2 embedding from a PIL Image."""
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(img_tensor)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().flatten().tolist()


def create_augmented_variants(image_bytes: bytes) -> list[Image.Image]:
    """Create original + 2 augmented variants for triple-query majority voting."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = img.size
    variants = [img]

    for _ in range(2):
        crop_ratio = random.uniform(0.92, 0.97)
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        cropped = cropped.resize((w, h), Image.LANCZOS)
        variants.append(cropped)

    return variants


def triple_query_majority_vote(
    image_bytes: bytes, n_results: int
) -> tuple[list[str], list[float], list[dict]]:
    """Run 3 embedding queries (original + 2 augmented) and pick winner by majority vote."""
    variants = create_augmented_variants(image_bytes)

    all_query_results = []
    for variant in variants:
        query_vec = embed_pil_image(variant)
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=n_results,
        )
        all_query_results.append(result)

    # Count #1 results across all 3 runs
    top1_ids = [r["ids"][0][0] for r in all_query_results]
    top1_counts = Counter(top1_ids)

    top1_best_similarity = {}
    for result in all_query_results:
        card_id = result["ids"][0][0]
        similarity = 1.0 - result["distances"][0][0]
        if card_id not in top1_best_similarity or similarity > top1_best_similarity[card_id]:
            top1_best_similarity[card_id] = similarity

    winner_id = max(
        top1_counts.keys(),
        key=lambda cid: (top1_counts[cid], top1_best_similarity.get(cid, 0)),
    )

    # Find the best run for the winner
    best_run_idx = 0
    best_winner_sim = -1.0
    for i, result in enumerate(all_query_results):
        if result["ids"][0][0] == winner_id:
            sim = 1.0 - result["distances"][0][0]
            if sim > best_winner_sim:
                best_winner_sim = sim
                best_run_idx = i

    best_result = all_query_results[best_run_idx]
    vote_summary = {cid: count for cid, count in top1_counts.most_common()}
    print(f"[DINOv2] Triple vote: winner={winner_id} ({top1_counts[winner_id]}/3), "
          f"sim={best_winner_sim:.4f}, votes={vote_summary}")

    return best_result["ids"][0], best_result["distances"][0], best_result["metadatas"][0]


def get_or_create_collection(name: str):
    """Get or create a ChromaDB collection by name (for upsert to any collection)."""
    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return chroma_client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def handler(event):
    """
    RunPod serverless handler.

    Input (event["input"]):
      - action: "search" | "embed" | "health" | "upsert"
      - image: base64-encoded image (for search/embed)
      - mime_type: "image/jpeg" (optional)
      - top_k: number of results (optional, default 5)
      - collection: collection name (for upsert, defaults to COLLECTION_NAME)
      - ids: list of IDs (for upsert)
      - embeddings: list of 1024-dim vectors (for upsert)
      - metadatas: list of metadata dicts (for upsert)

    Returns:
      search  → matches, embedding_count, query_time_ms
      embed   → embedding (1024-dim float list)
      health  → status info
      upsert  → upserted count, collection, total count
    """
    if model is None:
        load_model()

    input_data = event.get("input", {})
    action = input_data.get("action", "search")

    if action == "health":
        count = collection.count() if collection else 0
        return {
            "status": "ok",
            "model": "DINOv2-ViT-L/14 (1024-dim)",
            "device": str(device),
            "embedding_count": count,
            "sqlite_available": os.path.exists(SQLITE_PATH),
        }

    if action == "embed":
        image_b64 = input_data.get("image")
        if not image_b64:
            return {"error": "Missing 'image' field (base64-encoded)"}

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}

        start = time.monotonic()
        embedding = embed_image(image_bytes)
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "embedding": embedding,
            "dimensions": len(embedding),
            "inference_ms": round(elapsed_ms, 1),
        }

    if action == "search":
        start = time.monotonic()

        image_b64 = input_data.get("image")
        if not image_b64:
            return {"error": "Missing 'image' field (base64-encoded)"}

        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}

        top_k = input_data.get("top_k", TOP_K)
        count = collection.count()
        n_results = min(top_k, count)

        if n_results == 0:
            return {"matches": [], "embedding_count": 0, "query_time_ms": 0}

        # Triple-query majority vote search
        ids, distances, metadatas = triple_query_majority_vote(image_bytes, n_results)

        # Build response with SQLite enrichment
        matches = []
        for card_id, distance, meta in zip(ids, distances, metadatas):
            similarity = 1.0 - distance
            sqlite_data = get_sqlite_data(card_id)

            match = {
                "product_id": card_id,
                "similarity": round(similarity, 6),
            }

            if sqlite_data:
                match.update({
                    "product_name": sqlite_data.get("product_name", meta.get("product_name", "")),
                    "set_slug": sqlite_data.get("set_slug", meta.get("set_slug", "")),
                    "console_name": sqlite_data.get("console_name"),
                    "loose_price": sqlite_data.get("loose_price"),
                    "cib_price": sqlite_data.get("cib_price"),
                    "new_price": sqlite_data.get("new_price"),
                    "graded_price": sqlite_data.get("graded_price"),
                    "image_path": sqlite_data.get("image_path"),
                })
            else:
                match.update({
                    "product_name": meta.get("product_name", ""),
                    "set_slug": meta.get("set_slug", ""),
                    "console_name": None,
                    "loose_price": meta.get("loose_price"),
                    "cib_price": None,
                    "new_price": None,
                    "graded_price": None,
                    "image_path": meta.get("image_path"),
                })

            matches.append(match)

        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "matches": matches,
            "embedding_count": count,
            "query_time_ms": round(elapsed_ms, 1),
        }

    if action == "upsert":
        ids = input_data.get("ids")
        embeddings = input_data.get("embeddings")
        metadatas = input_data.get("metadatas")

        if not ids or not embeddings:
            return {"error": "Missing 'ids' and/or 'embeddings' fields"}
        if len(ids) != len(embeddings):
            return {"error": f"ids ({len(ids)}) and embeddings ({len(embeddings)}) length mismatch"}

        coll_name = input_data.get("collection", COLLECTION_NAME)
        target = get_or_create_collection(coll_name)

        start = time.monotonic()
        upsert_kwargs = {"ids": ids, "embeddings": embeddings}
        if metadatas:
            upsert_kwargs["metadatas"] = metadatas
        target.upsert(**upsert_kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000

        new_count = target.count()
        print(f"[DINOv2] Upserted {len(ids)} embeddings into '{coll_name}' ({new_count:,} total)")

        return {
            "upserted": len(ids),
            "collection": coll_name,
            "total_count": new_count,
            "upsert_ms": round(elapsed_ms, 1),
        }

    return {"error": f"Unknown action: {action}"}


# Load model immediately when container starts (not on first request)
# This makes the container "warm" as soon as it's running
load_model()

runpod.serverless.start({"handler": handler})
