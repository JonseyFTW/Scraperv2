# SportsCardPro Scraper v2 — CSV-First Approach

Scrapes **every card** from **every set** on SportsCardsPro.com, downloads card images, and generates CLIP vector embeddings for visual similarity search. Designed to run distributed across multiple Proxmox LXC containers with shared PostgreSQL and NFS storage.

**Scale:** ~1.4 million cards across 8 sports (baseball, basketball, football, hockey, racing, soccer, wrestling, UFC).

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  UGREEN NAS (192.168.1.x)                                      │
│  ┌─────────────────────┐  ┌──────────────────────┐             │
│  │  PostgreSQL :5433    │  │  ChromaDB :8000       │             │
│  │  (cards, sets, logs) │  │  (CLIP embeddings)    │             │
│  └─────────────────────┘  └──────────────────────┘             │
│  ┌─────────────────────────────────────────────┐               │
│  │  NFS Share: /volume1/Data/scraper           │               │
│  │  (CSVs + images shared across all workers)  │               │
│  └─────────────────────────────────────────────┘               │
│        docker-compose up -d                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ LAN
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
     │  LXC 1  │          │  LXC 2  │          │  LXC 3  │
     │ Phase 5 │          │ Phase 5 │          │ Phase 5 │
     │ NordVPN │          │ NordVPN │          │ NordVPN │
     └─────────┘          └─────────┘          └─────────┘

     ┌───────────┐        ┌─────────────────────────────┐
     │ Windows PC│        │  Railway (production)        │
     │ Phase 1-4 │        │  ChromaDB — trading app      │
     │ --headed  │        │  ← migrate_chroma.py syncs   │
     └───────────┘        └─────────────────────────────┘
```

### Component Roles

| Component | Purpose |
|-----------|---------|
| **PostgreSQL** | Central database for all scrape data (sets, cards, status tracking). Supports concurrent writes from multiple workers via `FOR UPDATE SKIP LOCKED`. |
| **ChromaDB (local)** | Stores CLIP image embeddings during generation on the Windows PC. |
| **ChromaDB (Railway)** | Production instance that the card trading app reads from. Synced from local via `migrate_chroma.py`. |
| **NFS Share** | Shared filesystem so all LXC containers and the Windows PC access the same CSV/image files. |
| **NordVPN** | Each LXC container connects through a different VPN server. Auto-rotates on Cloudflare blocks. |
| **Windows PC** | Runs Phase 1-4 in `--headed` mode (real browser visible) because Cloudflare blocks headless browsers. |
| **LXC Containers** | Run Phase 5 (HTTP image downloads) in parallel — no browser needed. |

---

## Pipeline (5 Phases)

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Embeddings → Migration
Browser    Browser    No browser  Browser    No browser  GPU          HTTP
```

### Phase 1: Discover Sets
- **File:** `scraper.py` → `discover_sets(page, sport)`
- **What:** Navigates to category pages (e.g. `/category/football-cards`), paginates through all pages, extracts set URLs
- **Output:** Inserts rows into `sets` table with `csv_status='pending'`
- **Browser:** Required (Playwright)

### Phase 2: Download CSVs
- **File:** `scraper.py` → `download_csvs(page, sport)`
- **What:** For each set, visits the set page and clicks "Download Price List" button to download a CSV file
- **Output:** CSV files saved to `{DATA_DIR}/csvs/`, sets marked `csv_status='downloaded'`
- **Browser:** Required (needs login for Retail+ access)
- **Rate:** One CSV per set, ~5s delay between downloads

### Phase 3: Parse CSVs
- **File:** `scraper.py` → `parse_csvs()`
- **What:** Reads downloaded CSV files, extracts card data, bulk-inserts into `cards` table
- **Output:** Cards inserted with `status='pending'`, sets marked `csv_status='parsed'`
- **Browser:** Not needed
- **CSV columns:** `id`, `product-name`, `console-name`, `loose-price`, `cib-price`, `new-price`

### Phase 4: Scrape Image URLs
- **File:** `scraper.py` → `scrape_card_images(page, limit)`
- **What:** Uses the browser's `fetch()` API to request card pages in parallel batches, extracts image URLs from HTML via regex
- **Output:** Cards updated with `image_url` and `status='image_found'`
- **Browser:** Required (uses in-browser fetch to inherit Cloudflare cookies)
- **Bottleneck:** Cloudflare rate limiting — uses adaptive throttling and VPN rotation
- **Image URL pattern:** `https://storage.googleapis.com/images.pricecharting.com/{hash}/1600.jpg`
- **Fallback:** Phase 4b retries errored cards via direct `page.goto()` browser visits

### Phase 5: Download Images
- **File:** `scraper.py` → `download_images(limit)`
- **What:** Pure HTTP downloads of image files from Google Cloud Storage CDN
- **Output:** Images saved to `{DATA_DIR}/images/{product_id}.jpg`, cards marked `status='downloaded'`
- **Browser:** Not needed — uses `aiohttp` with 25 concurrent downloads
- **Parallelizable:** Multiple LXC containers can run this simultaneously

### Embeddings (Post-Pipeline)
- **File:** `embeddings.py` → `generate_embeddings()`
- **What:** Generates CLIP vector embeddings (ViT-B-32) for each downloaded card image
- **Output:** Embeddings stored in local ChromaDB collection `card_images`
- **Requires:** GPU recommended (`open-clip-torch`, `torch`, `torchvision`)

### Migration (Post-Embeddings)
- **File:** `migrate_chroma.py` → `migrate()`
- **What:** Incrementally syncs local ChromaDB to Railway production instance
- **Auth:** Uses `Authorization: Bearer` and `X-Chroma-Token` headers for Railway auth proxy

---

## File-by-File Reference

### `config.py` — Central Configuration
All paths, URLs, credentials, and tuning parameters.

| Variable | Purpose | Default |
|----------|---------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:changeme@192.168.1.14:5433/sportscards` |
| `DATA_DIR` | Root for CSVs, images, ChromaDB | `./data` or `SCP_DATA_DIR` env var |
| `CSV_DIR` | Downloaded CSV files | `{DATA_DIR}/csvs` |
| `IMAGE_DIR` | Downloaded card images | `{DATA_DIR}/images` |
| `CHROMA_DIR` | Local ChromaDB persistent storage | `{DATA_DIR}/chromadb` |
| `LOGIN_EMAIL` / `LOGIN_PASSWORD` | SportsCardsPro credentials | `SCP_EMAIL` / `SCP_PASSWORD` env vars |
| `HEADLESS` | Run browser headless | `True` (override with `--headed`) |
| `IMAGE_CONCURRENT_DOWNLOADS` | Parallel downloads in Phase 5 | `25` |
| `CATEGORY_URLS` | Dict mapping sport names to category page URLs | 8 sports |
| `USER_AGENTS` | Rotating user agent strings | 3 Chrome/Firefox UAs |

### `database.py` — PostgreSQL Data Layer
All database operations. Uses `psycopg2` with `RealDictCursor` for dict-style row access.

**Schema:**

```sql
-- Sets table: one row per card set (e.g. "1986 Fleer Basketball")
sets (
    slug        TEXT PRIMARY KEY,    -- URL slug, e.g. "basketball-cards-1986-fleer"
    name        TEXT,                -- Display name
    sport       TEXT,                -- "baseball", "football", etc.
    url         TEXT NOT NULL,       -- Full URL to set page
    csv_status  TEXT DEFAULT 'pending',  -- pending → downloaded → parsed | error
    img_status  TEXT DEFAULT 'pending',  -- pending → scraped
    csv_path    TEXT,                -- Path to downloaded CSV file
    card_count  INTEGER DEFAULT 0,
    updated_at  TEXT
)

-- Cards table: one row per individual card (~1.4M rows)
cards (
    id              SERIAL PRIMARY KEY,
    product_id      TEXT UNIQUE,     -- SportsCardsPro product ID (used as ChromaDB ID too)
    set_slug        TEXT NOT NULL,   -- FK → sets.slug
    product_name    TEXT,            -- "Michael Jordan #57"
    console_name    TEXT,            -- "Basketball Cards 1986 Fleer"
    card_url_slug   TEXT,            -- URL slug for the card page
    full_url        TEXT,            -- Full URL to card detail page
    image_url       TEXT,            -- GCS CDN image URL (set in Phase 4)
    image_path      TEXT,            -- Local file path (set in Phase 5)
    loose_price     REAL,            -- Ungraded price in pennies
    cib_price       REAL,            -- Mid-grade price
    new_price       REAL,            -- High-grade price
    graded_price    TEXT,
    status          TEXT DEFAULT 'pending',  -- State machine (see below)
    error_msg       TEXT
)

-- Scrape log: event tracking
scrape_log (
    id          SERIAL PRIMARY KEY,
    timestamp   TEXT NOT NULL,
    event       TEXT NOT NULL,       -- "phase1_complete", "phase4_complete", etc.
    details     TEXT
)
```

**Card Status State Machine:**

```
pending → processing → image_found → downloading → downloaded
                    ↘ no_image
                    ↘ error (retryable via --reset-errors)
```

- `pending` — Needs Phase 4 (image URL scraping)
- `processing` — Claimed by a worker (Phase 4), prevents duplicate work
- `image_found` — Has image URL, ready for Phase 5
- `downloading` — Claimed by a worker (Phase 5)
- `downloaded` — Image file saved to disk, ready for embedding
- `no_image` — Card page confirmed to have no image
- `error` — Failed, stores reason in `error_msg`

**Concurrency Pattern:**
All claim operations use PostgreSQL `FOR UPDATE SKIP LOCKED`:
```sql
UPDATE cards SET status='processing'
WHERE id IN (
    SELECT id FROM cards WHERE status='pending'
    ORDER BY set_slug DESC, id    -- Newest sets first
    LIMIT 500
    FOR UPDATE SKIP LOCKED        -- Skip rows locked by other workers
)
RETURNING *
```
This ensures multiple LXC containers never process the same cards.

**Key Functions:**
- `get_cards_needing_images(limit)` — Atomically claim pending cards for Phase 4
- `get_cards_needing_download(limit)` — Atomically claim cards for Phase 5
- `get_errored_cards(limit)` — Claim errored cards for retry
- `bulk_insert_cards(cards)` — Batch insert from CSV parse (`ON CONFLICT DO NOTHING`)
- `count_pending_images()` — Count remaining for ETA calculation
- `reset_errors()` — Reset error/stuck cards back to retryable states

### `scraper.py` — Browser Automation & Scraping (largest file)
Contains all 5 phases plus browser setup, Cloudflare handling, and VPN rotation.

**Key Sections:**

1. **Browser Creation** (`create_browser`)
   - Tries **Camoufox** first (anti-detect Firefox with C++ level fingerprint spoofing)
   - Falls back to **Playwright Chromium** with stealth patches if Camoufox not installed
   - Camoufox uses `headless="virtual"` (Xvfb) on Linux — looks headed to Cloudflare
   - Configures proxy from `HTTPS_PROXY` env var if set

2. **Cloudflare Handling** (`_solve_cloudflare`)
   - Detects Turnstile challenge iframes from `challenges.cloudflare.com`
   - Tries 3 click strategies: checkbox input, label element, iframe body click
   - 5 retry attempts with randomized delays
   - 30-second manual solve window in headed mode

3. **VPN Rotation** (`rotate_vpn`, `maybe_rotate_vpn`)
   - Only activates if `nordvpn` CLI is available (LXC containers)
   - Tracks consecutive Cloudflare blocks via `_cf_block_count`
   - After 3 consecutive blocks: `nordvpn disconnect` → `nordvpn connect` (new server/IP)
   - Resets throttle multiplier after rotation (fresh IP = clean slate)

4. **Adaptive Throttling** (Phase 4)
   - `throttle_multiplier` starts at 1.0
   - On CF block: doubles (up to 8x max)
   - After 5 consecutive clean batches: halves back down
   - Applies to batch delay: `delay = random(1.5, 3.0) * throttle_multiplier`

5. **ETA Display**
   - Phase 4: Prints ETA every 2 minutes with cards/sec rate
   - Phase 5: Live ETA in the status table, updates per batch
   - Format: "3 days 12 hours 7 minutes"

**Key Functions:**
- `create_browser(playwright)` → `(Browser, BrowserContext)` — Stealth browser setup
- `new_stealth_page(context)` → `Page` — Creates page with stealth patches
- `safe_goto(page, url)` → `bool` — Navigation with CF detection, retry, VPN rotation
- `login(page)` → `bool` — SportsCardsPro login
- `discover_sets(page, sport)` — Phase 1
- `download_csvs(page, sport)` — Phase 2
- `parse_csvs()` — Phase 3 (no browser)
- `scrape_card_images(page, limit)` — Phase 4 (in-browser fetch, adaptive throttle)
- `download_images(limit)` — Phase 5 (aiohttp, 25 concurrent)
- `run_full_pipeline(sport, limit)` — Runs phases 1-3 sequentially

### `embeddings.py` — CLIP Embedding Generator
Generates and searches vector embeddings using OpenCLIP ViT-B-32 model.

**ChromaDB Collection:** `card_images`
- **Distance metric:** Cosine
- **ID:** `product_id` (string)
- **Metadata:** `product_name`, `set_slug`, `image_path`, `loose_price`
- **Embedding dim:** 512 (ViT-B-32)

**Key Functions:**
- `generate_embeddings(limit)` — Batch-processes downloaded cards, skips existing embeddings and missing files
- `search_by_image(path, top_k)` — Finds visually similar cards to a query image
- `search_by_text(query, top_k)` — CLIP text-to-image search (e.g. "red jersey football card")
- `get_cards_needing_embeddings(limit)` — Filters cards not yet in ChromaDB, tracks `_skipped_ids` for missing files
- `show_embedding_stats()` — Coverage report

### `migrate_chroma.py` — Local → Railway ChromaDB Sync
Incrementally copies embeddings from local PersistentClient to remote HttpClient.

- Compares local vs remote IDs, only sends new ones
- Sends auth as both `Authorization: Bearer` and `X-Chroma-Token` headers
- Supports `--dry-run`, `--batch-size`, `--token` flags
- Idempotent — safe to run repeatedly

### `main.py` — CLI Entry Point
Parses arguments and dispatches to appropriate phase/function.

**Arguments:**
| Flag | Purpose |
|------|---------|
| `--phase N` | Run specific phase (1-5) |
| `--sport NAME` | Limit to one sport |
| `--limit N` | Max cards to process |
| `--stats` | Show progress dashboard |
| `--reset-errors` | Reset failed items for retry |
| `--reset-no-image` | Reset no-image cards for retry |
| `--failures` | Show error breakdown |
| `--headed` | Show browser window (sets `config.HEADLESS = False`) |

### `setup_scraper_lxc.sh` — Proxmox LXC Setup Wizard
Interactive `whiptail` dialog wizard for creating scraper LXC containers.

- Creates Ubuntu 24.04 LXC container with configurable CPU/RAM/disk
- Installs Python, NordVPN, GTK dependencies (for Camoufox), Xvfb
- Clones repo, creates venv, installs pip dependencies
- Mounts NFS share from UGREEN NAS
- Creates `scraper` helper command (`/usr/local/bin/scraper`)
- Helper commands: `scraper update`, `scraper vpn`, `scraper --phase 5 ...`

### `docker-compose.yml` — Database Containers
Runs on the UGREEN NAS:
- **PostgreSQL 16** on port 5433 (5432 was in use by UGREEN's built-in PG)
- **ChromaDB** on port 8000
- Named volumes for persistence

### `check_dupes.py` — Diagnostic Script
Checks for duplicate `product_id` values in the cards table. (Result: no duplicates found across 1.4M cards.)

---

## Data Flow Diagram

```
SportsCardsPro.com
        │
   Phase 1: GET /category/{sport}-cards
        │ → discover set URLs
        ▼
   Phase 2: GET /console/{set-slug} → click "Download Price List"
        │ → save CSV to {DATA_DIR}/csvs/{slug}.csv
        ▼
   Phase 3: Parse CSV files (offline, no network)
        │ → INSERT INTO cards (...) ON CONFLICT DO NOTHING
        ▼
   Phase 4: browser.fetch(/game/{set}/{card-slug})
        │ → regex match GCS image URL from HTML
        │ → UPDATE cards SET image_url=..., status='image_found'
        ▼
   Phase 5: aiohttp.get(image_url) × 25 concurrent
        │ → save to {DATA_DIR}/images/{product_id}.jpg
        │ → UPDATE cards SET image_path=..., status='downloaded'
        ▼
   Embeddings: CLIP ViT-B-32 encode each image
        │ → upsert into local ChromaDB (collection: card_images)
        ▼
   Migration: migrate_chroma.py
        │ → incremental sync to Railway ChromaDB
        ▼
   Trading App reads from Railway ChromaDB
```

---

## Known Issues & Technical Debt

1. **Cloudflare blocking** — Phase 4 is the main bottleneck. The site aggressively blocks automated browsers. Current mitigations (Camoufox, VPN rotation, adaptive throttling) help but don't fully solve it. Running `--headed` on a Windows PC is most reliable.

2. **No connection pooling** — `database.py` opens/closes a new `psycopg2` connection for every operation. At 1.4M cards this is inefficient. Should use a connection pool (`psycopg2.pool` or `sqlalchemy`).

3. **Blocking DB calls in async code** — `scraper.py` is async but `database.py` uses synchronous psycopg2. DB calls block the event loop. Should use `asyncpg` or run DB ops in a thread executor.

4. **Large scraper.py** — This file contains all 5 phases, browser setup, CF handling, VPN rotation, and download logic (~1300 lines). Should be split into separate modules.

5. **Hardcoded NAS IP** — `192.168.1.14` appears in config defaults and setup scripts. Should be fully env-var driven.

6. **No retry/resume for Phase 2** — If CSV download fails mid-set, there's no granular resume. Sets are marked as `error` and need `--reset-errors`.

7. **Image paths platform-dependent** — Windows uses `Z:\...` paths, Linux uses `/mnt/...`. The `SCP_DATA_DIR` env var handles this but `image_path` stored in DB reflects whichever platform downloaded it.

---

## Environment Variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `SCP_EMAIL` | Yes | SportsCardsPro login email |
| `SCP_PASSWORD` | Yes | SportsCardsPro login password |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `SCP_DATA_DIR` | No | Override data directory (for NFS/shared drive) |
| `HTTPS_PROXY` | No | Proxy URL for browser (auto-detected) |

---

## Quick Start

```bash
# 1. Start databases on NAS
export POSTGRES_PASSWORD=your_secure_password
docker-compose up -d

# 2. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install camoufox[geoip] && python -m camoufox fetch
playwright install chromium && playwright install-deps chromium

# 3. Set credentials
export SCP_EMAIL="your@email.com"
export SCP_PASSWORD="yourpassword"
export DATABASE_URL="postgresql://postgres:pass@192.168.1.14:5433/sportscards"

# 4. Run phases
python main.py --phase 1 --sport football    # Discover sets
python main.py --phase 2 --sport football    # Download CSVs
python main.py --phase 3                     # Parse all CSVs
python main.py --phase 4 --headed            # Scrape image URLs (use --headed!)
python main.py --phase 5                     # Download images (can parallelize)

# 5. Generate embeddings (needs GPU)
pip install open-clip-torch torch torchvision numpy
python embeddings.py generate

# 6. Sync to Railway
python migrate_chroma.py --target https://your-railway-url --token YOUR_KEY

# 7. Monitor progress
python main.py --stats
```

---

## Dependencies

```
playwright>=1.52.0          # Browser automation
playwright-stealth>=1.0.6   # Anti-detection patches
camoufox[geoip]>=0.4.0      # Anti-detect Firefox (optional, recommended)
aiohttp>=3.9.0              # Async HTTP for image downloads
aiofiles>=23.0.0            # Async file I/O
rich>=13.0.0                # Terminal UI (progress bars, tables)
chromadb>=1.0.0              # Vector database for embeddings
psycopg2-binary>=2.9.0      # PostgreSQL driver

# For embeddings (install separately — large):
open-clip-torch torch torchvision numpy
```
