# PRD: Parallel Disambiguation Infrastructure — Scraperv2 Implementation Plan

**Status:** Implemented (Part A A1–A9 + Part B B1–B5) — pending backfill run + RunPod redeploy
**Branch:** `claude/parallel-disambiguation-infrastructure-azdkF`
**Scope:** Scraperv2 repo. **Includes classifier training (Part B) because the DINOv2 fine-tuning pipeline already lives here under `training/`.**
**Out of scope:** CardScanner web-app changes (A.1/A.2) — those ship from the web-app repo.
**Source PRD:** Parent "Parallel Disambiguation Infrastructure" PRD (Parts A + B).

---

## 0. Summary

**Part A — data plumbing:** six new columns on `cards`, parsing at scrape time, a two-pass backfill for the ~2M existing rows, and updated ChromaDB metadata (local + RunPod).

**Part B — foil-pattern classifier:** new training scripts alongside the existing DINOv2 fine-tuner in `training/`, reusing its manifest/orchestration conventions. Plus a new `classify_variant` action on the RunPod handler so the web-app can hit it.

**Out of scope (tracked in the web-app repo):**
- A.1 — replace SCP scrape with `gcs_image_url` read in `dinov2-disambig.ts`
- A.2 — structured hint matching on `card_number` / `print_run`
- B.2/B.3 — shadow-mode logging and Gemini-gating on scan requests

Optional Scraperv2 work:
- **A.3**: add `where` filter support to the RunPod handler's `search` action.
- **B.4**: ingest user-pick data exported from the web-app's `ScanLog` for continuous-learning retrains (needs web-app to emit the export).

---

## 1. Current State (verified)

| Concern | Location | Current behavior |
|---|---|---|
| `cards` schema | `database.py:107-124` | 15 columns + `worker_id` (added via idempotent `ALTER TABLE ... duplicate_column` block at `:134-139`). No `card_number`, `print_run`, `player_name`, `variant_label`, `gcs_*_url`. |
| GCS URL extraction | `scraper_v3.py:793-818` (`fetch_image_url`) | Regex `r'https://storage\.googleapis\.com/images\.pricecharting\.com/([^/\s"\'<>]+)/\d+'` already pulls the hash and stores the full `/1600.jpg` URL into `cards.image_url`. So for the vast majority of existing rows, the canonical GCS URL is already on disk — it just isn't split into `1600.jpg` / `240.jpg` variants and isn't labeled as such. |
| CDN-pattern fast path | `scraper_v3.py:55-140` (`CDNPatternEngine`) + `:587-625` | Same URL shape, skips the HTTP fetch entirely. |
| CSV parse → DB | `scraper.py:605-704` (`parse_csvs`, called by `main_v3.py:89`) | Stores `product_name` verbatim. No parsing. |
| Embedding metadata | `embeddings_dinov2.py:298-303` | Writes only `product_name`, `set_slug`, `image_path`, `loose_price`. |
| RunPod `search` | `runpod_handler/handler.py:288-352` + `:380-382` | `collection.query(query_embeddings=..., n_results=...)` — no `where` clause, no metadata filter. |
| ChromaDB in-place update | `migrate_chroma.py:159-163` demonstrates `.upsert(ids, embeddings, metadatas)`. ChromaDB also supports `.update(ids, metadatas=...)` with no `embeddings` re-upload — the fast path for backfill. |
| Statuses | `pending` → `processing` → `image_found` → `downloading` → `downloaded`, plus `no_image`/`error`. |

### Key insight that simplifies the backfill

`cards.image_url` already contains the canonical GCS `/1600.jpg` URL for every row scraped by v3 (and most of v2). **A SQL-only backfill can populate `gcs_image_url` and derive `gcs_thumb_url` for the entire table in minutes, without a single HTTP call.** The network-bound backfill is only needed for rows where `image_url` is NULL or non-GCS.

---

## 2. Changes

### 2.1 Schema — `database.py`

Append to the init block at `database.py:134` (same idempotent `DO $$ BEGIN ... EXCEPTION WHEN duplicate_column THEN NULL` pattern already used for `worker_id`). One `DO $$` block per column keeps the failure mode isolated.

```sql
ALTER TABLE cards ADD COLUMN gcs_image_url  TEXT;     -- canonical "<hash>/1600.jpg"
ALTER TABLE cards ADD COLUMN gcs_thumb_url  TEXT;     -- "<hash>/240.jpg"
ALTER TABLE cards ADD COLUMN card_number    TEXT;     -- "30", "RPA-25", "CON-1"
ALTER TABLE cards ADD COLUMN print_run      INTEGER;  -- null = unknown, 0 reserved
ALTER TABLE cards ADD COLUMN player_name    TEXT;
ALTER TABLE cards ADD COLUMN variant_label  TEXT;     -- contents of [brackets]
```

Indexes (add alongside existing ones at `database.py:210-215`):

```sql
CREATE INDEX IF NOT EXISTS idx_cards_print_run   ON cards(print_run);
CREATE INDEX IF NOT EXISTS idx_cards_card_number ON cards(card_number);
CREATE INDEX IF NOT EXISTS idx_cards_player_lower ON cards(lower(player_name));
```

No index on `variant_label` yet — low cardinality per-set, and the B.1 classifier will read it in bulk by set, not by lookup.

### 2.2 Parsing helper — new module `card_name_parser.py`

Single module, small and unit-tested. Used both at scrape time and in the backfill script. Shape:

```python
# card_name_parser.py
import re
from dataclasses import dataclass
from typing import Optional

_CARD_NUMBER_RE = re.compile(r"#\s*([A-Za-z0-9\-]+)")
_PRINT_RUN_RE   = re.compile(r"/\s*(\d{1,6})\b")
_BRACKET_RE     = re.compile(r"\[([^\]]+)\]")

@dataclass
class ParsedName:
    player_name:   Optional[str]
    card_number:   Optional[str]
    print_run:     Optional[int]
    variant_label: Optional[str]

def parse_product_name(product_name: str) -> ParsedName:
    if not product_name:
        return ParsedName(None, None, None, None)

    # variant_label = contents of [...]
    m = _BRACKET_RE.search(product_name)
    variant_label = m.group(1).strip() if m else None

    # card_number = token after first #
    m = _CARD_NUMBER_RE.search(product_name)
    card_number = m.group(1) if m else None

    # print_run = first integer after / (anywhere in the string)
    m = _PRINT_RUN_RE.search(product_name)
    print_run = int(m.group(1)) if m else None

    # player_name = everything before the first [ or #, trimmed
    split_at = min(
        [i for i in (product_name.find("["), product_name.find("#")) if i >= 0]
        or [len(product_name)]
    )
    player_name = product_name[:split_at].strip() or None

    return ParsedName(player_name, card_number, print_run, variant_label)
```

Unit tests (new file `tests/test_card_name_parser.py`) cover:

- `"Joe Burrow #30 [Yellow Pyramids] /185"` → all four fields set.
- `"Joe Burrow [Silver]"` → name + variant, no number/run.
- `"Topps Chrome Refractor #RPA-25"` → alphanumeric card_number.
- `"Base /99"` → print_run only; name falls back to `"Base"`.
- `""` / `None` → all fields `None`.
- Bracket *before* `#`: `"Joe Burrow [Silver] #30"` → still split on whichever comes first (`[`).
- No hash, no brackets, plain `"Joe Burrow"` → `player_name="Joe Burrow"`, rest `None`.

### 2.3 GCS URL helpers — extend `scraper_v3.py`

Add a tiny helper next to `fetch_image_url` (line 793):

```python
_GCS_HASH_RE = re.compile(
    r"storage\.googleapis\.com/images\.pricecharting\.com/([^/\s\"'<>]+)"
)

def gcs_urls_from_any(url: str) -> tuple[Optional[str], Optional[str]]:
    """Given any PriceCharting GCS URL (any size), return (1600_url, 240_url)."""
    if not url:
        return None, None
    m = _GCS_HASH_RE.search(url)
    if not m:
        return None, None
    h = m.group(1)
    return (
        f"https://storage.googleapis.com/images.pricecharting.com/{h}/1600.jpg",
        f"https://storage.googleapis.com/images.pricecharting.com/{h}/240.jpg",
    )
```

Use at:
1. `fetch_image_url` (line 806–808) — return both URLs, not just the `1600.jpg`.
2. Phase 4 `CDNPatternEngine` return path — when the pattern hits, derive the thumb URL too.
3. The backfill script (section 2.6).

### 2.4 Scraper write path

Two changes inside the existing flow; no new phase needed.

**(a) CSV parse (`scraper.py:673-701`)** — after `product_name` is extracted, call the parser and add to the insert dict:

```python
from card_name_parser import parse_product_name

parsed = parse_product_name(product_name)
row = {
    # ... existing fields ...
    "player_name":   parsed.player_name,
    "card_number":   parsed.card_number,
    "print_run":     parsed.print_run,
    "variant_label": parsed.variant_label,
}
```

Extend `bulk_insert_cards` (`database.py:336-351`) to include the four parsed columns in both the column list and `VALUES` clause. No change to the `ON CONFLICT DO NOTHING` — we don't want to clobber existing rows on a CSV re-parse; backfill handles those separately.

**(b) Phase 4 image-URL write (`scraper_v3.py` — wherever the result of `fetch_image_url` is persisted)** — update the DB write helper (currently `db.update_card_image_url` or equivalent; confirm symbol name during implementation) to take and store `gcs_image_url` + `gcs_thumb_url` alongside `image_url`. Simplest: set `image_url = gcs_image_url` and populate the two new columns from the helper. Keeps `image_url` working for backwards compatibility.

### 2.5 Embeddings metadata — `embeddings_dinov2.py:298-303`

Replace the metadata dict with:

```python
chroma_metadatas.append({
    "product_name":   card["product_name"]   or "",
    "set_slug":       card["set_slug"]       or "",
    "image_path":     card["image_path"]     or "",   # kept for backcompat
    "gcs_image_url":  card["gcs_image_url"]  or "",
    "gcs_thumb_url":  card["gcs_thumb_url"]  or "",
    "card_number":    card["card_number"]    or "",
    "print_run":      int(card["print_run"]) if card["print_run"] else 0,
    "player_name":    (card["player_name"]   or "").lower(),
    "variant_label":  card["variant_label"]  or "",
    "loose_price":    float(card["loose_price"] or 0),
})
```

Note: ChromaDB metadata values must be scalar (str/int/float/bool), not `None` — coerce to `""`/`0` (same convention as the existing `product_name` line). The `0` sentinel for missing `print_run` is acceptable because no real print run is 0; the web app will treat `0` as "unknown" (documented there).

`get_cards_needing_embeddings` at `embeddings_dinov2.py:209-234` already does `SELECT *` so no query changes needed — the new columns flow through automatically.

### 2.6 Backfill — new `backfill_card_metadata.py`

Two passes, both idempotent and resumable.

**Pass 1 — in-process parsing + SQL-only GCS URL populate. Fast (~10 min for 2M rows).**

```sql
-- 1a. derive gcs_image_url / gcs_thumb_url from existing image_url
UPDATE cards
SET gcs_image_url = regexp_replace(image_url,
                     '(storage\.googleapis\.com/images\.pricecharting\.com/[^/]+)/\d+.*',
                     '\1/1600.jpg'),
    gcs_thumb_url = regexp_replace(image_url,
                     '(storage\.googleapis\.com/images\.pricecharting\.com/[^/]+)/\d+.*',
                     '\1/240.jpg')
WHERE image_url LIKE '%storage.googleapis.com/images.pricecharting.com/%'
  AND gcs_image_url IS NULL;
```

Then a chunked Python loop for the parsed fields (can't do this in pure SQL because the player_name rule picks `min(pos('['), pos('#'))`):

```python
# pseudocode
while True:
    batch = SELECT id, product_name FROM cards
            WHERE player_name IS NULL AND product_name IS NOT NULL
            ORDER BY id LIMIT 10_000;
    if not batch: break
    updates = [(parse_product_name(r.product_name), r.id) for r in batch]
    execute_batch("""
        UPDATE cards
           SET player_name=%s, card_number=%s, print_run=%s, variant_label=%s
         WHERE id=%s
    """, [(p.player_name, p.card_number, p.print_run, p.variant_label, id)
          for (p, id) in updates])
```

**Pass 2 — network-bound backfill for rows still missing `gcs_image_url`.** Rate-limited, parallelized, resumable. Reuses `SessionManager` from `scraper_v3.py:231-296` to keep the curl_cffi / proxy / Cloudflare handling identical to production. Back-of-envelope: ~2M rows × ~500 ms / 20 workers ≈ 14 h as the parent PRD estimates, but we expect Pass 1 to cover ≥95% of rows.

Resume logic: always `WHERE gcs_image_url IS NULL AND full_url IS NOT NULL` — rerunning picks up where it left off.

**Pass 3 — ChromaDB metadata update, in-place, no re-embedding.**

ChromaDB's `collection.update(ids=..., metadatas=...)` replaces metadata without re-uploading embeddings. For 600K+ rows this runs in a few minutes. Then `sync` (already exists in `embeddings_dinov2.py`) pushes the updated metadata to RunPod.

Script outline:

```python
# backfill_chroma_metadata.py
def rebuild_metadata(batch_size=1000):
    collection = get_collection("card_embeddings_dinov2")
    existing_ids = set(collection.get()["ids"])  # ~600K strings; fine in RAM
    for batch in chunked(existing_ids, batch_size):
        rows = db.query("""
            SELECT product_id, product_name, set_slug, image_path,
                   gcs_image_url, gcs_thumb_url, card_number, print_run,
                   player_name, variant_label, loose_price
              FROM cards
             WHERE product_id = ANY(%s)
        """, (list(batch),))
        ids, metas = [], []
        for r in rows:
            ids.append(r["product_id"])
            metas.append(_build_metadata(r))  # same shape as 2.5
        collection.update(ids=ids, metadatas=metas)
```

### 2.7 (Optional, A.3) RunPod handler `where` filter

In `runpod_handler/handler.py`, change the `search` branch (lines 288-352) to accept an optional `where` dict and pass it through to `collection.query`:

```python
where = input_data.get("where")  # e.g. {"print_run": 185}
query_kwargs = {"query_embeddings": [query_vec], "n_results": n_results}
if where:
    query_kwargs["where"] = where
results = collection.query(**query_kwargs)
```

This is the only required server-side change for A.3; the web-app builds the `where` clause. Ship behind a version bump of the handler image so older web-app versions that don't send `where` are unaffected.

---

## Part B — Foil-Pattern Classifier

### 2.8 Approach

Reuse the existing DINOv2 ViT-L/14-reg backbone (loaded the same way as in `training/02_finetune_dinov2.py:304` and `embeddings_dinov2.py:67-73`) and train **a classification head on top of frozen features**. This is a ~5 MB model that ships alongside the existing fine-tuned backbone on RunPod and adds ~10 ms per inference.

Why a head on frozen features, not an end-to-end classifier:
- DINOv2 features are already high-quality for fine-grained visual tasks.
- A linear probe or 2-layer MLP can be trained on a laptop in an hour and updated weekly.
- It composes cleanly with the existing fine-tuned backbone — at inference time we compute the DINOv2 feature once and reuse it for both similarity search AND variant classification.

Model shape, smallest-first:

1. **V1 — Linear probe.** `nn.Linear(1024, num_classes)`. Train in ~30 min. Expected ~80–85% top-1.
2. **V2 — Small MLP.** `Linear(1024, 256) → GELU → Dropout(0.1) → Linear(256, num_classes)`. Train in ~1 h. Expected ~88–92%.
3. **V3 — Crop-stack averaging.** Run V2 on 5 crops per card (4 corners + center stripe, border-biased) and average logits. Train once on the crop stack with random-crop augmentation. Expected 90%+ top-1, ≥98% top-3.

Ship V1 first to validate the pipeline end-to-end; iterate to V2/V3 only if eval numbers miss the target.

### 2.9 Class taxonomy

Per the parent PRD's Open Questions, start with **raw `variant_label` strings, prefixed by `set_slug`**:

```
class_key = f"{set_slug}::{variant_label.strip().lower()}"
```

- Filters out nulls and ultra-rare classes (<10 examples) — record them but route to a single `__rare__` bucket at train time.
- Keeps the `panini-prizm` "Silver" distinct from `panini-phoenix` "Silver" (different foils).
- Canonicalization (typos, casing) deferred to V2 if accuracy suffers on common classes.

Store the label map in the checkpoint so inference always has a deterministic index-to-label mapping.

### 2.10 `training/04_export_variant_training_data.py` — new

Mirrors `01_export_training_data.py` but groups by `variant_label` instead of character name.

```python
# pseudocode
cur.execute("""
    SELECT product_id, set_slug, variant_label, image_path, product_name
      FROM cards
     WHERE status = 'downloaded'
       AND image_path IS NOT NULL
       AND variant_label IS NOT NULL
       AND variant_label <> ''
""")
rows = cur.fetchall()

class_counts = Counter(f"{r.set_slug}::{r.variant_label.lower().strip()}" for r in rows)
kept = {k for k, c in class_counts.items() if c >= MIN_SAMPLES_PER_CLASS}  # e.g. 10

manifest = []
for r in rows:
    key = f"{r.set_slug}::{r.variant_label.lower().strip()}"
    label = key if key in kept else "__rare__"
    manifest.append({
        "product_id": r.product_id,
        "image_path": r.image_path,
        "set_slug":   r.set_slug,
        "variant_label": r.variant_label,
        "class":      label,
    })

# Stratified 90/10 split, seeded
# Emit: variant_manifest_train.jsonl, variant_manifest_val.jsonl, label_map.json
```

Outputs dropped in `./training_data/variant_classifier/` next to the existing Pokemon/Sports exports.

Success metric for this script: ≥ 100 classes with ≥ 100 samples each, covering ≥ 80% of all scraped `variant_label` values.

### 2.11 `training/05_train_variant_classifier.py` — new

```python
# Load the *same* backbone the inference-time path uses
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg")
if args.finetuned_backbone:
    model.load_state_dict(torch.load(args.finetuned_backbone, weights_only=True))
model.eval()
for p in model.parameters(): p.requires_grad = False
model = model.to(device)

# V1 head
head = nn.Linear(1024, num_classes).to(device)

# fp16 feature extraction, fp32 head, CE with label smoothing 0.1
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optim = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
```

Training loop:

- Image load + transform: identical preprocessing to `embeddings_dinov2.py:83-88` (518×518, bicubic, ImageNet norm). Matches RunPod inference.
- Augmentation: reuse the phone-camera augmentation from `02_finetune_dinov2.py:46-189` (perspective, rotation ±5°, brightness/contrast, glare, JPEG). **Don't** augment borders away with aggressive random-crop in V1; the foil pattern often lives there.
- Batch size: 128 on a 4070 Super Ti with fp16 features.
- Epochs: 10–20 for the head (loss plateaus fast on frozen features).
- Eval: top-1 and top-3 per epoch on the val set. Save best checkpoint.

Checkpoint format (single `.pt`):

```python
torch.save({
    "head_state_dict": head.state_dict(),
    "label_map":       list_of_class_names,  # index -> string
    "backbone":        "dinov2_vitl14_reg",
    "finetuned_backbone_sha": hash_of_backbone_checkpoint_or_None,
    "arch":            "linear",             # or "mlp_256"
    "input_dim":       1024,
    "preprocess":      {"size": 518, "crop": "center"},
    "trained_at":      iso_timestamp,
    "train_samples":   len(train_set),
    "val_top1":        best_top1,
    "val_top3":        best_top3,
}, "./checkpoints/variant_classifier_v1.pt")
```

Including the preprocess spec and the backbone identifier in the checkpoint means the RunPod handler can refuse to load mismatched versions — a small guardrail against silent drift.

### 2.12 `runpod_handler/handler.py` — new `classify_variant` action

```python
elif action == "classify_variant":
    image_b64        = input_data["image"]
    candidate_labels = input_data.get("candidate_labels")  # optional short-list

    feat = embed_image(image_bytes)  # reuses the existing DINOv2 path

    logits = _variant_head(torch.from_numpy(feat).to(_device))
    probs  = torch.softmax(logits, dim=-1).cpu().numpy()
    scores = {_label_map[i]: float(probs[i]) for i in range(len(_label_map))}

    if candidate_labels:
        # Restrict + renormalize over the candidates the web-app supplied.
        # (Web-app passes the labels of the DINOv2 top-K candidates.)
        scores = {k: scores.get(k, 0.0) for k in candidate_labels}
        total = sum(scores.values()) or 1.0
        scores = {k: v / total for k, v in scores.items()}

    top = max(scores, key=scores.get)
    return {"scores": scores, "top": top, "top_score": scores[top]}
```

Model loading happens once at cold start, next to the existing backbone load. The head is a few MB; download time negligible.

Handler response matches the parent PRD:

```json
{"scores": {"Yellow Pyramids": 0.87, "Teal Pyramids": 0.12}, "top": "Yellow Pyramids", "top_score": 0.87}
```

### 2.13 Pipeline integration — `run_full_pipeline.py`

Add two steps to the existing 7-step pipeline (orchestration at `run_full_pipeline.py:514-549`):

- **Step 8:** `04_export_variant_training_data.py`
- **Step 9:** `05_train_variant_classifier.py`
- **Step 5 (existing) extended:** upload the variant classifier checkpoint to RunPod S3 alongside the DINOv2 checkpoint.

New CLI flags:

- `--skip-variant-classifier` (symmetry with `--skip-training`)
- `--variant-arch {linear,mlp}` (default `linear` for V1)
- `--variant-min-samples 10`

Resumable in the same way every existing step is.

### 2.14 Evaluation harness

Single script: `training/06_eval_variant_classifier.py`. Reads the val manifest, runs the checkpoint, reports:

- Overall top-1 / top-3.
- Per-class precision/recall (sorted by support) — surfaces which foil patterns are learned well and which aren't.
- Confusion matrix dumped to CSV for the top N confusions.
- A head-to-head sample: for 100 random ambiguous DINOv2 clusters (≥2 candidates within 0.02 similarity), does the classifier pick the same variant as `product_name` ground truth?

This is what determines whether we promote V1 → V2 → V3, and later whether the classifier is good enough to gate Gemini calls in the web app.

### 2.15 Continuous learning (B.4, deferred)

Placeholder — no implementation in this PR, just an interface:

- Web-app exports user picks from `ScanLog.rawResponse` where `method="dinov2-disambig-user-pick"` as JSONL: `{image_url_or_bytes, chosen_product_id, rejected_product_ids, timestamp}`.
- Scraperv2 ingests the JSONL, joins `chosen_product_id` → `cards.variant_label`, adds to the training manifest with a `source: "user_pick"` tag and higher sample weight.
- Retrain script extended to honor `sample_weight`.

Ship this once shadow-mode (B.2, web-app side) produces enough picks to be worth retraining on. Target: weekly retrain cadence.

---

## 3. Rollout Order

**Part A — data plumbing** (one PR, then run backfill separately):

| # | Change | Risk | Reversible? |
|---|---|---|---|
| A1 | Schema additions (`database.py`) + indexes | Low — nullable columns, idempotent `ALTER` | Yes — drop columns |
| A2 | `card_name_parser.py` + unit tests | None | Yes |
| A3 | Scraper write-path updates (Phase 3 + Phase 4) | Low — new rows only | Yes — new columns ignored |
| A4 | `embeddings_dinov2.py` metadata expansion | Low — only affects newly-embedded cards | Yes |
| A5 | `backfill_card_metadata.py` Pass 1 (SQL + in-proc parse) | Low — pure `UPDATE ... WHERE IS NULL` | Yes — set new cols back to NULL |
| A6 | `backfill_chroma_metadata.py` (`collection.update`) | Low — metadata-only, no embeddings touched | Yes — re-run from DB |
| A7 | `sync` to RunPod | Low — existing command | Yes |
| A8 | Pass 2 network backfill (long-running) | Medium — hits SCP at scale; follow existing rate limits | Yes — kill + resume |
| A9 | (Opt.) RunPod handler `where` support | Low — additive param | Yes — old image |

Ship A1–A7 as one PR. Run A8 in a screen session on a scraper LXC and monitor. Land A9 independently once the web-app needs it.

**Part B — foil-pattern classifier** (blocks on A completing):

| # | Change | Risk | Reversible? |
|---|---|---|---|
| B1 | `training/04_export_variant_training_data.py` | None — read-only export | Yes |
| B2 | `training/05_train_variant_classifier.py` (V1 linear probe) | None — offline, produces a new checkpoint file | Yes |
| B3 | `training/06_eval_variant_classifier.py` + per-class report | None | Yes |
| B4 | `run_full_pipeline.py` steps 8–9 integration | Low — gated by `--skip-variant-classifier` | Yes |
| B5 | `runpod_handler/handler.py` — `classify_variant` action + checkpoint load at cold-start | Medium — adds cold-start time, new code path; ship behind handler version bump | Yes — revert handler image |
| B6 | Upload checkpoint to RunPod S3, deploy new handler release | Medium — must match inference-time preprocessing exactly | Yes — roll back release |
| B7 | Iterate to V2 (MLP) and V3 (crop-stack) if V1 misses the 90% top-1 / 98% top-3 bar | None (offline) | Yes |
| B8 | (Deferred) B.4 continuous-learning ingestion from web-app user-pick export | None until web-app emits the export | Yes |

Ship B1–B3 as a second PR once A has completed and Pass 1 of the backfill has populated `variant_label` across the table. B4–B6 ship as a third PR gated on eval results.

---

## 4. Acceptance Criteria

**Part A:**
1. `\d+ cards` in `psql` shows `gcs_image_url`, `gcs_thumb_url`, `card_number`, `print_run`, `player_name`, `variant_label` columns; three new indexes visible.
2. `pytest tests/test_card_name_parser.py` passes all ~8 cases above.
3. A freshly-scraped set shows non-null `player_name` / `card_number` / `variant_label` / `gcs_*_url` on ≥95% of rows (100% for GCS URLs when the image actually exists).
4. `SELECT count(*) FROM cards WHERE gcs_image_url IS NULL AND image_url LIKE '%googleapis%'` returns 0 after Pass 1.
5. After Pass 3, `collection.get(limit=1)["metadatas"][0]` on the local Chroma collection contains all new keys.
6. After `sync`, the RunPod collection `card_embeddings_dinov2` shows the same new keys on a sample query.
7. Existing `embeddings_dinov2.py search <img>` keeps working with unchanged latency and top-1 result (no regression in similarity ranking — metadata doesn't affect the ANN index).
8. (If A.3 shipped) `curl` to the RunPod handler with `{"action":"search","image":..., "where":{"print_run":185}}` returns only matches with `print_run: 185` in their metadata.

**Part B:**
9. `04_export_variant_training_data.py` emits `variant_manifest_{train,val}.jsonl` + `label_map.json` with ≥ 100 classes, each with ≥ 100 samples, and an `__rare__` bucket for the tail.
10. `05_train_variant_classifier.py` produces a checkpoint at `./checkpoints/variant_classifier_v1.pt` containing `head_state_dict`, `label_map`, `backbone`, `arch`, `preprocess`, and validation metrics.
11. V1 linear-probe validation top-1 ≥ 80% on the held-out stratified split. V3 (if needed) ≥ 90% top-1, ≥ 98% top-3.
12. `06_eval_variant_classifier.py` produces a per-class precision/recall CSV and a top-N confusion CSV.
13. RunPod `classify_variant` action returns scores summing to ~1.0 when `candidate_labels` is supplied; returns full distribution when not.
14. End-to-end cold-start time on the RunPod worker doesn't regress by more than 500 ms (classifier head should add well under that).
15. `run_full_pipeline.py --step 8` runs the export; `--step 9` runs training; `--skip-variant-classifier` short-circuits both — no regression to the existing 7-step DINOv2 flow.

---

## 5. Risks & Mitigations

**Part A:**
- **`image_url` containing non-GCS URLs.** Pass 1's `WHERE image_url LIKE '%googleapis%'` skips them; Pass 2 handles them via HTTP. If a meaningful fraction is non-GCS, investigate before running Pass 2 at full scale.
- **ChromaDB memory during Pass 3.** `collection.get()` with no filter returns every ID; for 600K IDs that's fine (~50 MB of strings), but if we ever hit 10M+ we need paginated `get(limit=N, offset=M)`. Flag for future.
- **Regex false positives.** `"Upper Deck #1/1000"` — is `1/1000` a print run or `#1 of 1000`? Our rule says `print_run=1000`, `card_number="1"`. That's correct for SCP's convention; document it in the parser's docstring and add it to the test suite.
- **Variant label case/typos.** Per the parent PRD (Open Questions): start with raw labels, canonicalize later if the classifier's accuracy suffers. No normalization in this PR.
- **Backfill hitting SCP too hard.** Pass 2 reuses `SessionManager`'s existing delays (`REQUEST_DELAY_MIN/MAX`, `IMAGE_CONCURRENT_DOWNLOADS`) — same budget as normal scrapes. If SCP rate-limits, the existing error-handling paths take over.
- **Column name collision with `tcgplayer_cards.card_number`.** Different table, no collision; just note it in passing so future readers aren't confused.

**Part B:**
- **Class imbalance.** The long tail of rare variants (`/10`, `/5`, 1-of-1s) will starve the classifier. Route everything below MIN_SAMPLES to `__rare__` at train time; at inference time, rare-class predictions are low-confidence and fall through to Gemini. Document this as *expected* behavior.
- **Train/inference preprocessing drift.** The #1 silent-failure mode for this class of model. Mitigation: both paths share one helper (`preprocess_for_dinov2(img) -> tensor`) living in `embeddings_dinov2.py`. Training imports it. RunPod handler imports it. Checkpoint stores the preprocess spec and refuses to load on mismatch.
- **Backbone mismatch between train and serve.** The RunPod worker currently runs a fine-tuned DINOv2 checkpoint. The classifier head must be trained against the same checkpoint (hash stored in `finetuned_backbone_sha`). Retrain the head any time the backbone is retrained.
- **Scraped labels don't match user-uploaded scans.** The reference photos are clean studio shots; scans have glare and crops. That's exactly what the phone-camera augmentation from `02_finetune_dinov2.py:46-189` is designed to close, and why we reuse it.
- **Shadow-mode data not available yet.** We can't evaluate real-world accuracy until the web-app ships B.2. Workaround: for V1, evaluate on a held-out stratified split of scraped labels. Promote gating to a separate PR once shadow data lands.
- **Cold-start regression.** Loading the head is cheap (~5 MB), but the label_map and preprocess checks add a few hundred ms. If cold-start time is a concern, lazy-load the head on the first `classify_variant` call rather than at worker boot.

---

## 6. Out of Scope (tracked in the web-app repo)

- **A.1 web-app**: replace SCP scrape with `gcs_image_url` read.
- **A.2 web-app**: structured hint matching + "skip DINOv2 if print_run hint can't match" gate.
- **B.2 web-app**: shadow-mode logging (classifier result vs. Gemini result on every ambiguous scan).
- **B.3 web-app**: gate the Gemini call on `classifier_top_score < threshold`.
- **B.4 web-app**: export user picks from `ScanLog` as training data (Scraperv2 consumes on the other end once available).

---

## 7. File Change Map

**Part A:**

| File | Change |
|---|---|
| `database.py` | +6 `ALTER TABLE ... DO $$` blocks, +3 indexes, extend `bulk_insert_cards` column list |
| `card_name_parser.py` | **new** — ~40 LOC |
| `tests/test_card_name_parser.py` | **new** — ~60 LOC |
| `scraper.py` (`parse_csvs` ~:670) | call parser, pass 4 new fields into insert dict |
| `scraper_v3.py` | +`gcs_urls_from_any` helper; update `fetch_image_url` return shape; update DB write to store both URLs |
| `embeddings_dinov2.py` | expand metadata dict (+6 keys); factor preprocessing into a reusable helper that both training and RunPod can import |
| `backfill_card_metadata.py` | **new** — Pass 1 (SQL + in-proc parse), Pass 2 (network) |
| `backfill_chroma_metadata.py` | **new** — Pass 3 (`collection.update`) |
| `runpod_handler/handler.py` | (optional A.3) accept `where` in `search` action |
| `CLAUDE.md` | append a line under "Database" documenting the new columns + backfill commands |

**Part B:**

| File | Change |
|---|---|
| `training/04_export_variant_training_data.py` | **new** — ~150 LOC. Emits `variant_manifest_{train,val}.jsonl` + `label_map.json`. |
| `training/05_train_variant_classifier.py` | **new** — ~350 LOC. Linear-probe (V1) + MLP (V2) + crop-stack (V3) switches. |
| `training/06_eval_variant_classifier.py` | **new** — ~200 LOC. Top-1/top-3, per-class P/R, confusion CSV. |
| `training/requirements.txt` | add `scikit-learn` (stratified split, classification report) |
| `training/run_full_pipeline.py` | +2 steps (export + train variant classifier), +3 CLI flags (`--skip-variant-classifier`, `--variant-arch`, `--variant-min-samples`); extend step-5 S3 upload to include the variant checkpoint |
| `runpod_handler/handler.py` | +`classify_variant` action; cold-start head loader with label-map + preprocess-spec validation |
| `runpod_handler/requirements.txt` | no change (head is pure `nn.Linear` on existing torch) |
| `embeddings_dinov2.py` | expose a `preprocess_image(img) -> tensor` helper so training and serving share one code path |
| `CLAUDE.md` | append a "Foil-pattern classifier" subsection under "Architecture" |

Estimated net diff: Part A ~400 LOC, Part B ~750 LOC.
