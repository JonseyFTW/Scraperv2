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
LOGIN_EMAIL = os.environ.get("SCP_EMAIL", "")
LOGIN_PASSWORD = os.environ.get("SCP_PASSWORD", "")
LOGIN_URL = f"{BASE_URL}/login"

# ── Database ──────────────────────────────────────────────────────────────
# PostgreSQL connection string. Set via env var or edit directly.
#   Example: postgresql://user:pass@192.168.1.14:5432/sportscards
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/sportscards")

# ── Paths ─────────────────────────────────────────────────────────────────
# Set SCP_DATA_DIR env var to use a shared/network drive, e.g.:
#   set SCP_DATA_DIR=\\192.168.1.14\Data\scraper
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("SCP_DATA_DIR") or os.path.join(PROJECT_DIR, "data")
CSV_DIR = os.path.join(DATA_DIR, "csvs")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
CHROMA_DIR = os.path.join(DATA_DIR, "chromadb")

# ── Scraping behavior ─────────────────────────────────────────────────────
REQUEST_DELAY_MIN = 3.0        # Min seconds between page loads
REQUEST_DELAY_MAX = 6.0        # Max seconds between page loads
CSV_DOWNLOAD_DELAY = 5.0       # Delay between CSV downloads (rate limited)
CLOUDFLARE_WAIT = 8.0          # Seconds to wait for CF challenge
PAGE_LOAD_TIMEOUT = 30000      # Playwright timeout in ms
MAX_RETRIES = 3
RETRY_DELAY = 10.0
IMAGE_DOWNLOAD_TIMEOUT = 15
IMAGE_CONCURRENT_DOWNLOADS = 5  # Parallel image downloads

# ── Browser ───────────────────────────────────────────────────────────────
HEADLESS = True
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) Gecko/20100101 Firefox/133.0",
]

# Ensure dirs exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
