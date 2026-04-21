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

# --- Preprocess spec (must match embeddings_dinov2.PREPROCESS_SPEC) ---
PREPROCESS_SPEC = {
    "size": 518,
    "interpolation": "bicubic",
    "crop": "center",
    "mean": [0.485, 0.456, 0.406],
    "std":  [0.229, 0.224, 0.225],
}

# --- Variant-classifier config (Part B) ---
VARIANT_WEIGHTS_PATH = os.environ.get(
    "VARIANT_CLASSIFIER_WEIGHTS", "/runpod-volume/variant_classifier.pt"
)

# --- Global state (loaded once on cold start) ---
model = None
transform = None
device = None
collection = None

# Variant classifier — lazy-loaded on first classify_variant call
variant_head = None
variant_labels = None  # list[str] — index -> label
variant_label_to_idx = None  # dict[str, int]
variant_arch = None
variant_set_to_classes: dict[str, list[str]] | None = None
_variant_load_error = None


def load_model():
    """Load DINOv2-ViT-L/14 model and ChromaDB collection."""
    global model, transform, device, collection

    print("[DINOv2] Loading DINOv2-ViT-L/14 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load DINOv2-ViT-L/14 from torch hub (Meta's official weights)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")

    # Load fine-tuned weights if available
    finetuned_weights = os.environ.get(
        "FINETUNED_WEIGHTS", "/runpod-volume/dinov2_finetuned_backbone.pt"
    )
    if os.path.exists(finetuned_weights):
        state_dict = torch.load(finetuned_weights, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[DINOv2] Loaded fine-tuned weights from {finetuned_weights}")
    else:
        print(f"[DINOv2] No fine-tuned weights at {finetuned_weights}, using base model")

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
    image_bytes: bytes, n_results: int, where: dict | None = None
) -> tuple[list[str], list[float], list[dict]]:
    """Run 3 embedding queries (original + 2 augmented) and pick winner by majority vote.

    Optional `where` is a ChromaDB metadata filter (e.g. {"print_run": 185}) applied
    to every sub-query.
    """
    variants = create_augmented_variants(image_bytes)

    all_query_results = []
    for variant in variants:
        query_vec = embed_pil_image(variant)
        query_kwargs = {"query_embeddings": [query_vec], "n_results": n_results}
        if where:
            query_kwargs["where"] = where
        result = collection.query(**query_kwargs)
        all_query_results.append(result)

    # Drop runs that returned nothing (can happen when `where` is strict).
    all_query_results = [r for r in all_query_results if r["ids"] and r["ids"][0]]
    if not all_query_results:
        return [], [], []

    # Count #1 results across all remaining runs
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


def _build_variant_head(arch: str, input_dim: int, num_classes: int):
    """Mirrors training/05_train_variant_classifier.build_head."""
    import torch.nn as nn
    if arch == "linear":
        return nn.Linear(input_dim, num_classes)
    if arch == "mlp":
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )
    raise ValueError(f"Unknown variant arch: {arch}")


def load_variant_classifier():
    """Load the variant classifier head from the RunPod volume.

    Fails loudly if the checkpoint's preprocess spec doesn't match the handler's —
    that would silently produce garbage predictions.
    """
    global variant_head, variant_labels, variant_label_to_idx, variant_arch, \
        variant_set_to_classes, _variant_load_error

    if variant_head is not None or _variant_load_error is not None:
        return

    if not os.path.exists(VARIANT_WEIGHTS_PATH):
        _variant_load_error = f"weights not found at {VARIANT_WEIGHTS_PATH}"
        print(f"[Variant] {_variant_load_error}")
        return

    try:
        print(f"[Variant] Loading classifier from {VARIANT_WEIGHTS_PATH}...")
        ckpt = torch.load(VARIANT_WEIGHTS_PATH, map_location=device, weights_only=False)

        stored_spec = ckpt.get("preprocess")
        if stored_spec and stored_spec != PREPROCESS_SPEC:
            _variant_load_error = (
                f"preprocess mismatch — ckpt {stored_spec} vs handler {PREPROCESS_SPEC}"
            )
            print(f"[Variant] REFUSING TO LOAD: {_variant_load_error}")
            return

        variant_labels = ckpt["label_map"]
        variant_label_to_idx = {lab: i for i, lab in enumerate(variant_labels)}
        variant_arch = ckpt.get("arch", "linear")
        head = _build_variant_head(variant_arch, ckpt.get("input_dim", 1024), len(variant_labels))
        head.load_state_dict(ckpt["head_state_dict"])
        head.eval().to(device)
        variant_head = head

        # Build set_slug -> [class_label, ...] index so clients can pass
        # ``set_slug`` instead of the full candidate_labels shortlist.
        # Class keys are "<set_slug>::<variant>"; __rare__ is allowed for
        # every set to match the hierarchical training-time mask.
        variant_set_to_classes = {}
        for lab in variant_labels:
            if lab == "__rare__":
                continue
            set_slug = lab.split("::", 1)[0]
            variant_set_to_classes.setdefault(set_slug, []).append(lab)

        print(
            f"[Variant] Ready — arch={variant_arch}, labels={len(variant_labels)}, "
            f"sets={len(variant_set_to_classes)}, "
            f"hierarchical={ckpt.get('hierarchical', False)}, "
            f"val_top1={ckpt.get('val_top1')}"
        )
    except Exception as e:
        _variant_load_error = f"load failed: {e}"
        print(f"[Variant] {_variant_load_error}")


def handler(event):
    """
    RunPod serverless handler.

    Input (event["input"]):
      - action: "search" | "embed" | "health" | "upsert" | "classify_variant"
      - image: base64-encoded image (for search/embed/classify_variant)
      - mime_type: "image/jpeg" (optional)
      - top_k: number of results (optional, default 5)
      - where: ChromaDB metadata filter for search (optional), e.g. {"print_run": 185}
      - candidate_labels: shortlist for classify_variant (optional)
      - collection: collection name (for upsert, defaults to COLLECTION_NAME)
      - ids: list of IDs (for upsert)
      - embeddings: list of 1024-dim vectors (for upsert)
      - metadatas: list of metadata dicts (for upsert)

    Returns:
      search           → matches, embedding_count, query_time_ms
      embed            → embedding (1024-dim float list)
      health           → status info (incl. variant classifier state)
      upsert           → upserted count, collection, total count
      classify_variant → scores, top, top_score, inference_ms
    """
    if model is None:
        load_model()

    input_data = event.get("input", {})
    action = input_data.get("action", "search")

    if action == "health":
        count = collection.count() if collection else 0
        # List all collections with their counts
        all_collections = {}
        try:
            chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
            for col in chroma_client.list_collections():
                col_name = col.name if hasattr(col, 'name') else col
                c = chroma_client.get_collection(col_name)
                all_collections[col_name] = c.count()
        except Exception as e:
            all_collections = {"error": str(e)}
        finetuned_weights = os.environ.get(
            "FINETUNED_WEIGHTS", "/runpod-volume/dinov2_finetuned_backbone.pt"
        )
        # Attempt variant-classifier load once for health reporting
        load_variant_classifier()
        return {
            "status": "ok",
            "model": "DINOv2-ViT-L/14-reg (1024-dim)",
            "finetuned": os.path.exists(finetuned_weights),
            "finetuned_weights": finetuned_weights if os.path.exists(finetuned_weights) else None,
            "device": str(device),
            "embedding_count": count,
            "collections": all_collections,
            "sqlite_available": os.path.exists(SQLITE_PATH),
            "variant_classifier": {
                "loaded":     variant_head is not None,
                "arch":       variant_arch,
                "num_labels": len(variant_labels) if variant_labels else 0,
                "weights_path": VARIANT_WEIGHTS_PATH,
                "error":      _variant_load_error,
            },
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
        where = input_data.get("where")  # optional ChromaDB metadata filter
        count = collection.count()
        n_results = min(top_k, count)

        if n_results == 0:
            return {"matches": [], "embedding_count": 0, "query_time_ms": 0}

        # Triple-query majority vote search (optionally metadata-filtered)
        ids, distances, metadatas = triple_query_majority_vote(image_bytes, n_results, where)

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

    if action == "classify_variant":
        load_variant_classifier()
        if variant_head is None:
            return {"error": f"variant classifier unavailable: {_variant_load_error}"}

        image_b64 = input_data.get("image")
        if not image_b64:
            return {"error": "Missing 'image' field (base64-encoded)"}
        try:
            image_bytes = base64.b64decode(image_b64)
        except Exception:
            return {"error": "Invalid base64 image data"}

        candidate_labels = input_data.get("candidate_labels")
        # Convenience: ``set_slug`` auto-expands to that set's class labels
        # (plus __rare__). Equivalent to passing candidate_labels yourself,
        # and matches the hierarchical training-time set mask.
        set_slug = input_data.get("set_slug")
        if set_slug and not candidate_labels and variant_set_to_classes is not None:
            candidate_labels = list(variant_set_to_classes.get(set_slug, []))
            if "__rare__" in variant_label_to_idx:
                candidate_labels.append("__rare__")

        start = time.monotonic()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(img_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            logits = variant_head(features.float())
            probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()

        scores = {variant_labels[i]: float(probs[i]) for i in range(len(variant_labels))}

        if candidate_labels:
            # Restrict + renormalize over the supplied shortlist.
            restricted = {k: scores.get(k, 0.0) for k in candidate_labels}
            total = sum(restricted.values())
            if total > 0:
                restricted = {k: v / total for k, v in restricted.items()}
            scores = restricted

        if not scores:
            return {"error": "No matching labels in scores"}

        top = max(scores, key=scores.get)
        elapsed_ms = (time.monotonic() - start) * 1000
        return {
            "scores":       scores,
            "top":          top,
            "top_score":    scores[top],
            "inference_ms": round(elapsed_ms, 1),
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
