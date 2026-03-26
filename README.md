# SportsCardPro Scraper v2 — CSV-First Approach

Scrapes **every card** from **every set** on SportsCardsPro.com using a smarter approach:
download set CSVs (one request = all cards in a set) instead of visiting each card page individually.

## How It's Faster

| Approach | Requests for 600K cards across 3000 sets |
|----------|------------------------------------------|
| v1 (scrape each card page) | ~600,000 page loads → **30+ days** |
| **v2 (CSV download per set)** | ~3,000 CSV downloads → **~4 hours** for all metadata |

Images still need individual scraping (Phase 4), but you have all the data immediately after Phase 3.

## Pipeline

```
Phase 1: Discover sets     → Browse category pages, extract all set URLs
Phase 2: Download CSVs     → Click "Download Price List" on each set page
Phase 3: Parse CSVs        → Extract card data into PostgreSQL (no browser needed)
Phase 4: Scrape images     → Visit card pages to find image URLs
Phase 5: Download images   → HTTP download all images (no browser needed)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  UGREEN NAS (192.168.1.x)                                      │
│  ┌─────────────────────┐  ┌──────────────────────┐             │
│  │  PostgreSQL :5433    │  │  ChromaDB :8000       │             │
│  │  (cards, sets, logs) │  │  (CLIP embeddings)    │             │
│  └─────────────────────┘  └──────────────────────┘             │
│        docker-compose up -d                                     │
└──────────────────────────────┬──────────────────────────────────┘
                               │ LAN
          ┌────────────────────┼────────────────────┐
          │                    │                    │
     ┌────┴────┐          ┌────┴────┐          ┌────┴────┐
     │  VM 1   │          │  VM 2   │          │  VM 3   │
     │ scraper │          │ scraper │          │ scraper │
     └─────────┘          └─────────┘          └─────────┘

              ┌─────────────────────────────┐
              │  Railway (production)        │
              │  ChromaDB — your trading app │
              │  ← migrate_chroma.py syncs   │
              └─────────────────────────────┘
```

- **PostgreSQL** stores all scrape data (sets, cards, progress). Supports concurrent writes from multiple VMs.
- **ChromaDB (local)** stores CLIP embeddings during generation.
- **ChromaDB (Railway)** is the production instance your trading app reads from.

## Prerequisites

- **SportsCardsPro Retail+ subscription** — Required for CSV download access
- Python 3.11+
- Docker (on UGREEN/NAS for databases)
- ~50GB disk for images

## Infrastructure Setup

### 1. Start databases on UGREEN

Copy `docker-compose.yml` to your UGREEN and run:

```bash
export POSTGRES_PASSWORD=your_secure_password
docker-compose up -d
```

This starts PostgreSQL on port **5433** and ChromaDB on port **8000**.

### 2. Environment variables

Set these on each VM/machine that runs the scraper:

```bash
# Required
export SCP_EMAIL="your@email.com"
export SCP_PASSWORD="yourpassword"
export DATABASE_URL="postgresql://postgres:your_secure_password@192.168.1.14:5433/sportscards"

# Optional — shared drive for CSVs/images
export SCP_DATA_DIR="\\\\192.168.1.14\\Data\\scraper"   # Windows UNC
export SCP_DATA_DIR="/mnt/nas/scraper"                   # Linux mount
```

### 3. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
playwright install-deps chromium
```

## Quick Start

```bash
python main.py --headed    # First run: watch the browser to verify it works
python main.py             # Full auto after you're confident
```

## Usage Examples

```bash
# Full pipeline for one sport
python main.py --sport baseball

# Step by step
python main.py --phase 1                    # Discover all sets
python main.py --phase 2                    # Download all CSVs
python main.py --phase 3                    # Parse CSVs (instant, no browser)
python main.py --phase 4 --limit 100        # Test image scraping on 100 cards
python main.py --phase 5                    # Download all found images

# Monitor progress
python main.py --stats

# Retry failures
python main.py --reset-errors
python main.py
```

## Embedding Generation

Once you have images, generate CLIP embeddings into local ChromaDB:

```bash
pip install open-clip-torch torch torchvision numpy
python embeddings.py generate       # Embed all downloaded images
python embeddings.py search card.jpg  # Find matches for a photo
python embeddings.py stats           # Show embedding coverage
```

## Migrating Embeddings to Railway

Once embeddings are generated locally, sync them to your production ChromaDB on Railway.
Railway uses an Auth Proxy in front of ChromaDB — pass the API key with `--token`.

```bash
# Preview what would be migrated
python migrate_chroma.py --target https://auth-proxy-alpha.up.railway.app --token YOUR_API_KEY --dry-run

# Run the migration
python migrate_chroma.py --target https://auth-proxy-alpha.up.railway.app --token YOUR_API_KEY

# Custom batch size
python migrate_chroma.py --target https://auth-proxy-alpha.up.railway.app --token YOUR_API_KEY --batch-size 500
```

You can also set env vars instead of flags:

```bash
export CHROMA_TARGET_URL=https://auth-proxy-alpha.up.railway.app
export CHROMA_TARGET_TOKEN=YOUR_API_KEY
python migrate_chroma.py
```

The API key is found in Railway under: ChromaDB project → Auth Proxy service → Variables → `API_KEY`.

The migration is **incremental** — it skips embeddings already present on the remote, so you can run it repeatedly as you scrape new cards.

## LXC Container Management

### Creating a new LXC scraper container

Run the interactive setup wizard from the Proxmox host shell:

```bash
bash setup_scraper_lxc.sh
```

This walks you through CTID, CPU/RAM, NordVPN, database credentials, and NFS mount configuration.

### Updating a container

Pull the latest scraper code from git:

```bash
# From Proxmox host
pct exec <CTID> -- scraper update

# Or from inside the container
scraper update
```

### Running the scraper

```bash
# From Proxmox host
pct exec <CTID> -- scraper --phase 5 --sport football

# From inside the container
scraper --phase 5 --sport football
```

Multiple containers can run simultaneously — PostgreSQL row-level locking (`FOR UPDATE SKIP LOCKED`) ensures no two workers process the same cards.

### Other container commands

```bash
scraper vpn          # Show NordVPN connection status
scraper --stats      # Show scrape progress
scraper --reset-errors  # Reset failed cards for retry
```

> **Note:** Phase 4 (image URL scraping) requires a browser and should be run from your Windows PC with `--headed`. LXC containers are best for Phase 5 (HTTP image downloads) which doesn't need a browser.

## URL Patterns

The site follows consistent URL patterns:

```
Category:  /category/{sport}-cards
Brand:     /brand/{sport}-cards/{brand}     (e.g. /brand/baseball-cards/topps)
Set:       /console/{sport}-cards-{set}     (e.g. /console/football-cards-1979-topps)
Card:      /game/{sport}-cards-{set}/{slug} (e.g. /game/football-cards-1979-topps/earl-campbell-390)
```

The scraper uses the `/category/` pages to discover all sets, then `/console/` pages for CSV downloads.

## What the CSV Contains

Each CSV has one row per card with columns matching the API:
- `id` — Product ID (unique across the site)
- `product-name` — Card title (e.g. "Michael Jordan #57")
- `console-name` — Set name (e.g. "Basketball Cards 1986 Fleer")
- `loose-price` — Ungraded price in pennies
- `cib-price` — Mid-grade price in pennies
- `new-price` — High-grade price in pennies

## Image Scraping Strategy

After parsing CSVs, you have product IDs and card names. For images:

**Option A (what this script does):** Visit each card's detail page and extract the image URL from the DOM. This is the slowest part but most reliable.

**Option B (faster if the pattern holds):** SportsCardsPro/PriceCharting may serve images from a CDN with predictable URLs based on product ID. If you discover the pattern (check the image `src` on a few card pages), you could skip Phase 4 entirely and construct image URLs directly.

**Option C (use thumbnails from set pages):** Set pages often show thumbnail images in the card list. You could capture these during Phase 1/2 by scraping the set page DOM along with the CSV download. Thumbnails may be sufficient for CLIP embedding matching.

## Files

```
config.py           — URLs, credentials, delays, paths, database config
database.py         — PostgreSQL schema and operations
scraper.py          — Playwright automation (all 5 phases)
embeddings.py       — CLIP embedding generation and search (ChromaDB)
migrate_chroma.py   — Migrate local ChromaDB → remote (Railway)
docker-compose.yml  — PostgreSQL + ChromaDB containers for UGREEN
main.py             — CLI entry point
```
