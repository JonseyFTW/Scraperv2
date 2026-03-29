"""
SportsCardPro Scraper v2 - Configuration
CSV-first approach: download set CSVs, then scrape images
"""
import os

# Base URL
BASE_URL = "https://www.sportscardspro.com"

# ── Category discovery URLs ──────────────────────────────────────────────
# "All {sport}" pages list every set for that sport, across all brands.
# Brand pages (e.g. /brand/baseball-cards/topps) list sets per brand.
# We use the category "All" pages to get everything at once.
CATEGORY_URLS = {
    "baseball":    f"{BASE_URL}/category/baseball-cards",
    "basketball":  f"{BASE_URL}/category/basketball-cards",
    "football":    f"{BASE_URL}/category/football-cards",
    "hockey":      f"{BASE_URL}/category/hockey-cards",
    "racing":      f"{BASE_URL}/category/racing-cards",
    "soccer":      f"{BASE_URL}/category/soccer-cards",
    "wrestling":   f"{BASE_URL}/category/wrestling-cards",
    "ufc":         f"{BASE_URL}/category/ufc-cards",
    # "pokemon":   f"{BASE_URL}/category/pokemon-cards",
}

# ── Login ─────────────────────────────────────────────────────────────────
# Retail+ subscription required for CSV downloads.
# Set these via environment variables or edit directly.
LOGIN_EMAIL = os.environ.get("SCP_EMAIL", "mr.chadnjones@gmail.com")
LOGIN_PASSWORD = os.environ.get("SCP_PASSWORD", "LE4Ever!")
LOGIN_URL = f"{BASE_URL}/login"

# ── Database ──────────────────────────────────────────────────────────────
# PostgreSQL connection string. Set via env var or edit directly.
#   Example: postgresql://user:pass@192.168.1.14:5432/sportscards
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:changeme@192.168.1.14:5433/sportscards")

# ── Paths ─────────────────────────────────────────────────────────────────
# Set SCP_DATA_DIR env var to use a shared/network drive, e.g.:
#   Linux:   export SCP_DATA_DIR=/mnt/scraper-data
#   Windows: set SCP_DATA_DIR=Z:\
#   Windows: set SCP_DATA_DIR=\\192.168.1.14\Data\scraper
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SCP_DATA_DIR") or os.path.join(PROJECT_DIR, "data")
CSV_DIR = os.path.join(DATA_DIR, "csvs")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
IMG_DIR = IMAGE_DIR  # Alias for compatibility
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")

# ── Path mapping (for running embeddings on Windows against Linux DB paths)
# The DB stores Linux paths from the containers. When running on Windows,
# we need to translate them.  Set SCP_LINUX_DATA_PREFIX to the Linux data dir
# used by containers (e.g. /mnt/scraper-data).  The code will replace that
# prefix with DATA_DIR at runtime.
#   Example: export SCP_LINUX_DATA_PREFIX=/mnt/scraper-data
LINUX_DATA_PREFIX = os.environ.get("SCP_LINUX_DATA_PREFIX", "")

# ── RunPod (DINOv2 serverless endpoint) ──────────────────────────────────
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "m5et95n0vtnnmv")

# ── RunPod S3 (Network Volume access) ───────────────────────────────────
# Create an S3 API key in RunPod dashboard → Storage → "+ Create S3 API key"
RUNPOD_S3_ACCESS_KEY = os.environ.get("RUNPOD_S3_ACCESS_KEY", "")
RUNPOD_S3_SECRET_KEY = os.environ.get("RUNPOD_S3_SECRET_KEY", "")
RUNPOD_S3_ENDPOINT = os.environ.get("RUNPOD_S3_ENDPOINT", "https://s3api-us-nc-1.runpod.io")
RUNPOD_S3_BUCKET = os.environ.get("RUNPOD_S3_BUCKET", "fwlo3ibabg")

# ── Scraping behavior ─────────────────────────────────────────────────────
REQUEST_DELAY_MIN = 1.0        # Min seconds between page loads (per batch of 10)
REQUEST_DELAY_MAX = 3.0        # Max seconds between page loads
CSV_DOWNLOAD_DELAY = 5.0       # Delay between CSV downloads (rate limited)
CLOUDFLARE_WAIT = 8.0          # Seconds to wait for CF challenge
PAGE_LOAD_TIMEOUT = 30000      # Playwright timeout in ms
MAX_RETRIES = 3
RETRY_DELAY = 10.0
IMAGE_DOWNLOAD_TIMEOUT = 15
IMAGE_CONCURRENT_DOWNLOADS = 25  # Parallel image downloads (CDN, no CF)

# ── Browser ───────────────────────────────────────────────────────────────
HEADLESS = True
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
]

# ── Pokemon TCG (TCGdex API — free, no key required) ────────────────────
TCGDEX_BASE = "https://api.tcgdex.net/v2/en"
POKEMON_IMAGE_DIR = os.environ.get(
    "POKEMON_IMAGE_DIR", os.path.join(DATA_DIR, "pokemon_images")
)
POKEMON_IMAGE_QUALITY = "high"      # "high" or "low"
POKEMON_IMAGE_FORMAT = "png"        # png, jpg, or webp
POKEMON_DOWNLOAD_WORKERS = 8
POKEMON_REQUEST_DELAY = 0.05        # delay between API calls (be polite)
POKEMON_CHROMA_COLLECTION = "pokemon_embeddings_dinov2"

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(POKEMON_IMAGE_DIR, exist_ok=True)
