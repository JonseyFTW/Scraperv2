# CLAUDE.md

## Project Overview
SportsCardPro Scraper v2 — scrapes sports card data, downloads images, and generates vector embeddings for similarity search.

## Architecture

### Embedding Pipeline
- **`embeddings.py`** — Original CLIP ViT-B-32 embeddings (512-dim). Uses `open-clip-torch`. Collection: `card_images`
- **`embeddings_dinov2.py`** — DINOv2 ViT-L/14 embeddings (1024-dim). Runs locally on GPU with batched fp16 inference. Collection: `card_embeddings_dinov2`
  - Supports `generate`, `search`, `stats`, `migrate` commands
  - `--batch` flag controls GPU batch size (default 32, 64 works well on 4070 Super Ti)
  - `migrate` command copies embeddings from old `card_images_dinov2` collection to `card_embeddings_dinov2`
  - DINOv2 is vision-only — no text search (use CLIP for that)

### RunPod Serverless Endpoint
- Endpoint ID: `m5et95n0vtnnmv` (env: `RUNPOD_ENDPOINT_ID`)
- Docker image: `ghcr.io/jonseyftw/dinov2-cardscanner:latest`
- Handler source: `runpod_handler/handler.py`
- Handler actions: `search` (triple-query majority vote), `embed` (raw 1024-dim vector), `health`
- ChromaDB collection on RunPod: `card_embeddings_dinov2` (must match local)
- To redeploy: rebuild Docker image in `C:\Scripts\CardScanner\dinov2-service\`, push to GHCR, trigger new release in RunPod dashboard

### Database
- PostgreSQL for card metadata (connection pooled via `database.py` — use `db.put_connection()` not `conn.close()`)
- ChromaDB for vector embeddings (local at `config.CHROMA_DIR`, separate instance on RunPod volume)
- Card statuses: `image_found` → `downloading` → `downloaded` → (embedding generated separately)
- To reset error cards: `UPDATE cards SET status = 'image_found' WHERE status = 'error'`

### Config
- `config.py` holds all settings: DB URL, paths, RunPod credentials
- `RUNPOD_API_KEY` — always from env var, never hardcoded
- `LINUX_DATA_PREFIX` — for translating Linux DB paths when running on Windows
- Images stored on network drive (Z:\ on Windows, /mnt/scraper-data on Linux)

## Environment
- Dev/generation machine: Windows with NVIDIA 4070 Super Ti (17.2GB VRAM)
- Scrapers: LXC containers on Proxmox (containers 127, 128, 19, 130, 131, 132, 133)
- PowerShell syntax: use `$env:VAR = "value"` not `export VAR=value`

### Proxmox LXC Services
- **Image Tracker** (container 133): `systemctl restart image-tracker`
  - Code path: `/home/user/Scraperv2/image_tracker.py`
  - Service file: `/etc/systemd/system/image-tracker.service`
  - Runs on port 5000 (`http://192.168.1.119:5000`)
  - Note: also has a stale copy at `/opt/scraperv2/` — the service uses `/home/user/Scraperv2/`
- **Scraper workers**: updated via `pct exec <CTID> -- scraper update` then `systemctl restart scraper`

## Key Decisions
- Local GPU for bulk embedding generation (21+ img/s) — RunPod endpoint only for search API
- fp16 half precision enabled for ~2x GPU speedup
- Collection name `card_embeddings_dinov2` is canonical — must match between local and RunPod
