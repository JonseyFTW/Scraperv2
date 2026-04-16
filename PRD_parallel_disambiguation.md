# PRD: Parallel Disambiguation Infrastructure — Scraperv2 Implementation Plan

**Status:** Draft
**Branch:** `claude/parallel-disambiguation-infrastructure-azdkF`
**Scope:** Scraperv2 repo only. CardScanner web-app + RunPod classifier training are tracked separately.
**Source PRD:** Parent "Parallel Disambiguation Infrastructure" PRD (Parts A + B).

---

## 0. Summary

Six new columns on `cards`, parsing at scrape time, a two-pass backfill for the ~2M existing rows, and updated ChromaDB metadata (local + RunPod). That's the entire Scraperv2 surface area.

The CardScanner consumer work (A.1 reference-image lookup, A.2 hint matching, B.1–B.4 classifier) is out of scope here — this PRD only guarantees that the data those features need is present and correct.

Optional Scraperv2 work that enables downstream features:
- **A.3**: add `where` filter support to the RunPod handler's `search` action.
- **B**: export `variant_label` ground-truth tables for offline classifier training (trivial SQL; no code changes needed).

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

## 3. Rollout Order

| # | Change | Risk | Reversible? |
|---|---|---|---|
| 1 | Schema additions (`database.py`) + indexes | Low — nullable columns, idempotent `ALTER` | Yes — drop columns |
| 2 | `card_name_parser.py` + unit tests | None | Yes |
| 3 | Scraper write-path updates (Phase 3 + Phase 4) | Low — new rows only | Yes — new columns ignored |
| 4 | `embeddings_dinov2.py` metadata expansion | Low — only affects newly-embedded cards | Yes |
| 5 | `backfill_card_metadata.py` Pass 1 (SQL + in-proc parse) | Low — pure `UPDATE ... WHERE IS NULL` | Yes — set new cols back to NULL |
| 6 | `backfill_chroma_metadata.py` (`collection.update`) | Low — metadata-only, no embeddings touched | Yes — re-run from DB |
| 7 | `sync` to RunPod | Low — existing command | Yes |
| 8 | Pass 2 network backfill (long-running) | Medium — hits SCP at scale; follow existing rate limits | Yes — kill + resume |
| 9 | (Opt.) RunPod handler `where` support | Low — additive param | Yes — old image |

Ship 1–7 as one PR. Run 8 in a screen session on a scraper LXC and monitor. Land 9 independently once A.1/A.2 on the web-app side actually need it.

---

## 4. Acceptance Criteria

1. `\d+ cards` in `psql` shows `gcs_image_url`, `gcs_thumb_url`, `card_number`, `print_run`, `player_name`, `variant_label` columns; three new indexes visible.
2. `pytest tests/test_card_name_parser.py` passes all ~8 cases above.
3. A freshly-scraped set shows non-null `player_name` / `card_number` / `variant_label` / `gcs_*_url` on ≥95% of rows (100% for GCS URLs when the image actually exists).
4. `SELECT count(*) FROM cards WHERE gcs_image_url IS NULL AND image_url LIKE '%googleapis%'` returns 0 after Pass 1.
5. After Pass 3, `collection.get(limit=1)["metadatas"][0]` on the local Chroma collection contains all new keys.
6. After `sync`, the RunPod collection `card_embeddings_dinov2` shows the same new keys on a sample query.
7. Existing `embeddings_dinov2.py search <img>` keeps working with unchanged latency and top-1 result (no regression in similarity ranking — metadata doesn't affect the ANN index).
8. (If A.3 shipped) `curl` to the RunPod handler with `{"action":"search","image":..., "where":{"print_run":185}}` returns only matches with `print_run: 185` in their metadata.

---

## 5. Risks & Mitigations

- **`image_url` containing non-GCS URLs.** Pass 1's `WHERE image_url LIKE '%googleapis%'` skips them; Pass 2 handles them via HTTP. If a meaningful fraction is non-GCS, investigate before running Pass 2 at full scale.
- **ChromaDB memory during Pass 3.** `collection.get()` with no filter returns every ID; for 600K IDs that's fine (~50 MB of strings), but if we ever hit 10M+ we need paginated `get(limit=N, offset=M)`. Flag for future.
- **Regex false positives.** `"Upper Deck #1/1000"` — is `1/1000` a print run or `#1 of 1000`? Our rule says `print_run=1000`, `card_number="1"`. That's correct for SCP's convention; document it in the parser's docstring and add it to the test suite.
- **Variant label case/typos.** Per the parent PRD (Open Questions): start with raw labels, canonicalize later if the classifier's accuracy suffers. No normalization in this PR.
- **Backfill hitting SCP too hard.** Pass 2 reuses `SessionManager`'s existing delays (`REQUEST_DELAY_MIN/MAX`, `IMAGE_CONCURRENT_DOWNLOADS`) — same budget as normal scrapes. If SCP rate-limits, the existing error-handling paths take over.
- **Column name collision with `tcgplayer_cards.card_number`.** Different table, no collision; just note it in passing so future readers aren't confused.

---

## 6. Out of Scope (tracked elsewhere)

- **A.1 web-app**: replace SCP scrape with `gcs_image_url` read.
- **A.2 web-app**: structured hint matching + "skip DINOv2 if print_run hint can't match" gate.
- **B.1 classifier training**: ResNet/DINOv2-head over `variant_label`-labeled crops.
- **B.2–B.4**: shadow mode, promotion, continuous learning from disambig picks.

These unblock as soon as this PR lands and the backfill finishes.

---

## 7. File Change Map

| File | Change |
|---|---|
| `database.py` | +6 `ALTER TABLE ... DO $$` blocks, +3 indexes, extend `bulk_insert_cards` column list |
| `card_name_parser.py` | **new** — ~40 LOC |
| `tests/test_card_name_parser.py` | **new** — ~60 LOC |
| `scraper.py` (`parse_csvs` ~:670) | call parser, pass 4 new fields into insert dict |
| `scraper_v3.py` | +`gcs_urls_from_any` helper; update `fetch_image_url` return shape; update DB write to store both URLs |
| `embeddings_dinov2.py:298-303` | expand metadata dict (+6 keys) |
| `backfill_card_metadata.py` | **new** — Pass 1 (SQL + in-proc parse), Pass 2 (network) |
| `backfill_chroma_metadata.py` | **new** — Pass 3 (`collection.update`) |
| `runpod_handler/handler.py` (optional) | accept `where` in `search` action |
| `CLAUDE.md` | append a line under "Database" documenting the new columns + backfill commands |

Estimated net diff: ~400 LOC added, ~20 LOC changed.
