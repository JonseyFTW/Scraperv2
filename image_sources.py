"""
Multi-source card image scraper.

Tries multiple image sources in priority order to find card images
when SportsCardPro is blocked by Cloudflare or missing the image.

Sources (in fallback order):
  1. Wayback Machine  — cached SportsCardPro pages (no Cloudflare)
  2. TCDB             — Trading Card Database (community-uploaded scans)
  3. COMC             — Check Out My Cards (high-res dealer scans)
  4. eBay Browse API   — listing photos from active/sold listings

Usage:
    from image_sources import MultiSourceImageFinder
    finder = MultiSourceImageFinder()
    image_url = await finder.find_image(card)
"""

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote, quote_plus, urljoin

import aiohttp
from rich.console import Console

import config

log = logging.getLogger(__name__)
console = Console()

# ── Shared helpers ────────────────────────────────────────────────────────

# Throttle state per source to avoid hammering any single site
_source_delays: dict[str, float] = {}

HEADERS = {
    "User-Agent": config.USER_AGENTS[0],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


def _clean_card_name(product_name: str) -> str:
    """Strip common noise from card names for search queries."""
    # Remove card number at the end (e.g. "#52")
    name = re.sub(r'\s*#\d+\s*$', '', product_name)
    # Remove parallel/variant info in brackets for broader search
    name_no_variant = re.sub(r'\[.*?\]', '', name).strip()
    return name_no_variant


def _build_search_query(card: dict) -> str:
    """Build a search string from card data: 'Year Brand Player #Number'."""
    console = card.get("console_name", "")
    product = card.get("product_name", "")
    # console_name is like "Football Cards 2025 Bowman Chrome University"
    # product_name is like "A.J. Turner [Aqua Refractor] #52"
    return f"{console} {product}".strip()


@dataclass
class SourceResult:
    """Result from an image source lookup."""
    source: str
    image_url: str | None = None
    error: str | None = None
    tried: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Source 1: Wayback Machine (cached SportsCardPro pages)
# ═══════════════════════════════════════════════════════════════════════════

# Same regex the main scraper uses to find GCS-hosted card images
_SCP_IMAGE_RE = re.compile(
    r'https://storage\.googleapis\.com/images\.pricecharting\.com/([^/\s"\'<>]+)/(\d+)(?:\.jpg)?'
)


async def search_wayback(session: aiohttp.ClientSession, card: dict) -> SourceResult:
    """
    Check if the Wayback Machine has a cached version of this card's
    SportsCardPro page, then extract the image URL from the cached HTML.
    """
    full_url = card.get("full_url")
    if not full_url:
        return SourceResult(source="wayback", error="no full_url for card")

    try:
        # Step 1: Check if the Wayback Machine has this URL archived
        avail_url = f"http://archive.org/wayback/available?url={quote(full_url, safe='')}"
        async with session.get(avail_url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                return SourceResult(source="wayback", error=f"availability API HTTP {resp.status}")
            data = await resp.json()

        snapshot = data.get("archived_snapshots", {}).get("closest")
        if not snapshot or not snapshot.get("available"):
            return SourceResult(source="wayback", error="not archived")

        # Step 2: Fetch the archived page (use id_ prefix to get raw content)
        archive_url = snapshot["url"]
        # Convert to raw mode: /web/TIMESTAMP/URL -> /web/TIMESTAMPid_/URL
        archive_url = re.sub(r'/web/(\d+)/', r'/web/\1id_/', archive_url)

        async with session.get(archive_url, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return SourceResult(source="wayback", error=f"archive fetch HTTP {resp.status}")
            html = await resp.text()

        # Step 3: Extract image URL using the same regex as the main scraper
        match = _SCP_IMAGE_RE.search(html)
        if match:
            img_hash = match.group(1)
            image_url = f"https://storage.googleapis.com/images.pricecharting.com/{img_hash}/1600.jpg"
            return SourceResult(source="wayback", image_url=image_url)

        return SourceResult(source="wayback", error="no image in archived page")

    except asyncio.TimeoutError:
        return SourceResult(source="wayback", error="timeout")
    except Exception as e:
        return SourceResult(source="wayback", error=str(e)[:200])


# ═══════════════════════════════════════════════════════════════════════════
# Source 2: TCDB (Trading Card Database)
# ═══════════════════════════════════════════════════════════════════════════

_TCDB_SEARCH_URL = "https://www.tcdb.com/DirectSearch"
_TCDB_IMAGE_RE = re.compile(
    r'https?://(?:images\.)?tcdb\.com/Images/Cards/[^\s"\'<>]+\.(?:jpg|png|jpeg|gif|webp)',
    re.IGNORECASE,
)
# Also match the card detail page image (usually /Images/Cards/...)
_TCDB_CARD_IMG_RE = re.compile(
    r'<img[^>]+src=["\']([^"\']*?/Images/Cards/[^"\']+)["\']',
    re.IGNORECASE,
)


async def search_tcdb(session: aiohttp.ClientSession, card: dict) -> SourceResult:
    """
    Search TCDB for a card by name, then extract the card image from
    the search results or the card detail page.
    """
    product_name = card.get("product_name", "")
    console_name = card.get("console_name", "")

    # Build a focused search: player name + year + set
    player_name = _clean_card_name(product_name)
    # Extract year from console_name (e.g. "Football Cards 2025 Bowman Chrome University")
    year_match = re.search(r'((?:19|20)\d{2})', console_name)
    year = year_match.group(1) if year_match else ""
    # Extract set name (remove "Football Cards YYYY" prefix)
    set_name = re.sub(r'^(?:Football|Baseball|Basketball|Hockey|Soccer|Racing|Wrestling|UFC)\s+Cards\s+\d{4}\s*', '', console_name, flags=re.IGNORECASE).strip()

    search_query = f"{player_name} {year} {set_name}".strip()
    if not search_query:
        return SourceResult(source="tcdb", error="empty search query")

    try:
        # TCDB uses a DirectSearch endpoint that redirects to results
        search_url = f"https://www.tcdb.com/ViewCard.cfm?q={quote_plus(search_query)}"

        # Try the advanced search form which returns HTML results
        params = {"searchtext": search_query}
        headers = {**HEADERS, "Referer": "https://www.tcdb.com/"}

        async with session.get(
            "https://www.tcdb.com/Search.cfm/fu/1",
            params=params,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=8),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return SourceResult(source="tcdb", error=f"search HTTP {resp.status}")
            html = await resp.text()

        # Look for card image URLs in search results
        # TCDB card images are at /Images/Cards/{path}.jpg
        img_matches = _TCDB_CARD_IMG_RE.findall(html)
        if img_matches:
            # Take the first card image found
            img_url = img_matches[0]
            if not img_url.startswith("http"):
                img_url = f"https://www.tcdb.com{img_url}"
            return SourceResult(source="tcdb", image_url=img_url)

        # Also check for direct image URLs in the page
        direct_matches = _TCDB_IMAGE_RE.findall(html)
        if direct_matches:
            return SourceResult(source="tcdb", image_url=direct_matches[0])

        # If search results page has card links, try the first one
        card_link_match = re.search(r'href="(/ViewCard\.cfm/[^"]+)"', html)
        if card_link_match:
            card_page_url = f"https://www.tcdb.com{card_link_match.group(1)}"
            return await _fetch_tcdb_card_page(session, card_page_url)

        return SourceResult(source="tcdb", error="no results found")

    except asyncio.TimeoutError:
        return SourceResult(source="tcdb", error="timeout")
    except Exception as e:
        return SourceResult(source="tcdb", error=str(e)[:200])


async def _fetch_tcdb_card_page(session: aiohttp.ClientSession, url: str) -> SourceResult:
    """Fetch a TCDB card detail page and extract the card image."""
    try:
        async with session.get(
            url,
            headers={**HEADERS, "Referer": "https://www.tcdb.com/"},
            timeout=aiohttp.ClientTimeout(total=8),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return SourceResult(source="tcdb", error=f"card page HTTP {resp.status}")
            html = await resp.text()

        img_matches = _TCDB_CARD_IMG_RE.findall(html)
        if img_matches:
            img_url = img_matches[0]
            if not img_url.startswith("http"):
                img_url = f"https://www.tcdb.com{img_url}"
            return SourceResult(source="tcdb", image_url=img_url)

        return SourceResult(source="tcdb", error="no image on card page")

    except asyncio.TimeoutError:
        return SourceResult(source="tcdb", error="card page timeout")
    except Exception as e:
        return SourceResult(source="tcdb", error=str(e)[:200])


# ═══════════════════════════════════════════════════════════════════════════
# Source 3: COMC (Check Out My Cards)
# ═══════════════════════════════════════════════════════════════════════════

_COMC_IMG_RE = re.compile(
    r'https?://img\.comc\.com/i/[^\s"\'<>]+\.(?:jpg|png|jpeg)',
    re.IGNORECASE,
)


async def search_comc(session: aiohttp.ClientSession, card: dict) -> SourceResult:
    """
    Search COMC for a card and extract the image URL.
    COMC has high-quality scans of every consigned card.
    """
    product_name = card.get("product_name", "")
    console_name = card.get("console_name", "")

    # Build search query
    search_query = _build_search_query(card)
    if not search_query:
        return SourceResult(source="comc", error="empty search query")

    try:
        # COMC search endpoint
        search_url = "https://www.comc.com/Cards"
        params = {"search": search_query}

        async with session.get(
            search_url,
            params=params,
            headers={**HEADERS, "Referer": "https://www.comc.com/"},
            timeout=aiohttp.ClientTimeout(total=8),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return SourceResult(source="comc", error=f"search HTTP {resp.status}")
            html = await resp.text()

        # Look for card images in the search results
        # COMC images are at img.comc.com/i/{path}.jpg
        img_matches = _COMC_IMG_RE.findall(html)
        if img_matches:
            # COMC image URLs have a ?size= param — request original size
            img_url = img_matches[0]
            # Strip any existing size param and request original
            if "?" in img_url:
                img_url = img_url.split("?")[0]
            img_url = f"{img_url}?size=original"
            return SourceResult(source="comc", image_url=img_url)

        # Try to find a card detail link and fetch it
        card_link = re.search(r'href="(/Cards/[^"]+/\d+)"', html)
        if card_link:
            card_url = f"https://www.comc.com{card_link.group(1)}"
            return await _fetch_comc_card_page(session, card_url)

        return SourceResult(source="comc", error="no results found")

    except asyncio.TimeoutError:
        return SourceResult(source="comc", error="timeout")
    except Exception as e:
        return SourceResult(source="comc", error=str(e)[:200])


async def _fetch_comc_card_page(session: aiohttp.ClientSession, url: str) -> SourceResult:
    """Fetch a COMC card detail page and extract the card image."""
    try:
        async with session.get(
            url,
            headers={**HEADERS, "Referer": "https://www.comc.com/"},
            timeout=aiohttp.ClientTimeout(total=8),
            allow_redirects=True,
        ) as resp:
            if resp.status != 200:
                return SourceResult(source="comc", error=f"card page HTTP {resp.status}")
            html = await resp.text()

        img_matches = _COMC_IMG_RE.findall(html)
        if img_matches:
            img_url = img_matches[0]
            if "?" in img_url:
                img_url = img_url.split("?")[0]
            img_url = f"{img_url}?size=original"
            return SourceResult(source="comc", image_url=img_url)

        return SourceResult(source="comc", error="no image on card page")

    except asyncio.TimeoutError:
        return SourceResult(source="comc", error="card page timeout")
    except Exception as e:
        return SourceResult(source="comc", error=str(e)[:200])


# ═══════════════════════════════════════════════════════════════════════════
# Source 4: eBay Browse API
# ═══════════════════════════════════════════════════════════════════════════

# eBay OAuth token cache
_ebay_token: dict = {"access_token": None, "expires_at": 0}


async def _get_ebay_token(session: aiohttp.ClientSession) -> str | None:
    """Get an eBay OAuth token using client credentials grant."""
    client_id = os.environ.get("EBAY_CLIENT_ID", "")
    client_secret = os.environ.get("EBAY_CLIENT_SECRET", "")

    if not client_id or not client_secret:
        return None

    # Return cached token if still valid
    if _ebay_token["access_token"] and time.time() < _ebay_token["expires_at"] - 60:
        return _ebay_token["access_token"]

    try:
        import base64
        credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()

        async with session.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                log.warning("eBay OAuth failed: HTTP %d", resp.status)
                return None
            token_data = await resp.json()

        _ebay_token["access_token"] = token_data["access_token"]
        _ebay_token["expires_at"] = time.time() + token_data.get("expires_in", 7200)
        return _ebay_token["access_token"]

    except Exception as e:
        log.warning("eBay OAuth error: %s", e)
        return None


async def search_ebay(session: aiohttp.ClientSession, card: dict) -> SourceResult:
    """
    Search eBay Browse API for a card listing and return the image URL.
    Requires EBAY_CLIENT_ID and EBAY_CLIENT_SECRET environment variables.
    """
    token = await _get_ebay_token(session)
    if not token:
        return SourceResult(source="ebay", error="no eBay credentials (set EBAY_CLIENT_ID and EBAY_CLIENT_SECRET)")

    search_query = _build_search_query(card)
    if not search_query:
        return SourceResult(source="ebay", error="empty search query")

    # Truncate to eBay's 100-char limit
    if len(search_query) > 100:
        search_query = search_query[:100]

    try:
        # Sports Trading Cards category = 261328
        params = {
            "q": search_query,
            "category_ids": "261328",
            "limit": "3",
        }

        async with session.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            params=params,
            headers={
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                "Accept": "application/json",
            },
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            if resp.status != 200:
                return SourceResult(source="ebay", error=f"search HTTP {resp.status}")
            data = await resp.json()

        items = data.get("itemSummaries", [])
        if not items:
            return SourceResult(source="ebay", error="no listings found")

        # Pick the first listing with an image
        for item in items:
            image = item.get("image", {})
            img_url = image.get("imageUrl")
            if img_url:
                return SourceResult(source="ebay", image_url=img_url)

        return SourceResult(source="ebay", error="listings found but no images")

    except asyncio.TimeoutError:
        return SourceResult(source="ebay", error="timeout")
    except Exception as e:
        return SourceResult(source="ebay", error=str(e)[:200])


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Source Fallback Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

# All available sources in priority order
ALL_SOURCES = [
    ("wayback", search_wayback),
    ("tcdb", search_tcdb),
    ("comc", search_comc),
    ("ebay", search_ebay),
]


@dataclass
class MultiSourceStats:
    """Track hit/miss stats per source."""
    found: dict[str, int] = field(default_factory=lambda: {s: 0 for s, _ in ALL_SOURCES})
    missed: dict[str, int] = field(default_factory=lambda: {s: 0 for s, _ in ALL_SOURCES})
    errors: dict[str, int] = field(default_factory=lambda: {s: 0 for s, _ in ALL_SOURCES})
    total_found: int = 0
    total_not_found: int = 0

    def summary(self) -> str:
        lines = ["Multi-source image search stats:"]
        for source, _ in ALL_SOURCES:
            f = self.found.get(source, 0)
            m = self.missed.get(source, 0)
            e = self.errors.get(source, 0)
            lines.append(f"  {source:10s}: {f} found, {m} missed, {e} errors")
        lines.append(f"  TOTAL: {self.total_found} found, {self.total_not_found} not found")
        return "\n".join(lines)


class MultiSourceImageFinder:
    """
    Tries multiple image sources in fallback order to find a card image.

    Usage:
        finder = MultiSourceImageFinder(sources=["wayback", "tcdb", "comc"])
        async with aiohttp.ClientSession() as session:
            result = await finder.find_image(session, card)
            if result.image_url:
                print(f"Found via {result.source}: {result.image_url}")
    """

    def __init__(
        self,
        sources: list[str] | None = None,
        delay_between_sources: float = 0.5,
    ):
        if sources:
            self.sources = [(name, fn) for name, fn in ALL_SOURCES if name in sources]
        else:
            self.sources = list(ALL_SOURCES)

        self.delay = delay_between_sources
        self.stats = MultiSourceStats()

    async def find_image(
        self,
        session: aiohttp.ClientSession,
        card: dict,
    ) -> SourceResult:
        """
        Fire all sources in parallel. First one to return an image wins,
        and the rest are cancelled.
        """
        card_name = card.get("product_name", "?")[:50]

        async def _run_source(name, fn):
            try:
                return await asyncio.wait_for(fn(session, card), timeout=12)
            except asyncio.TimeoutError:
                return SourceResult(source=name, error="timeout")
            except asyncio.CancelledError:
                return SourceResult(source=name, error="cancelled")

        # Launch all sources concurrently
        tasks = {
            name: asyncio.create_task(_run_source(name, fn))
            for name, fn in self.sources
        }

        winner = None
        last_result = SourceResult(source="none", error="no sources configured")

        # As each finishes, check if it found an image
        for coro in asyncio.as_completed(tasks.values()):
            result = await coro
            source_name = result.source

            if result.image_url and not winner:
                winner = result
                self.stats.found[source_name] = self.stats.found.get(source_name, 0) + 1
                self.stats.total_found += 1
                console.print(f"  [green]✓[/green] {source_name}: {card_name}")
                # Cancel all remaining tasks
                for name, task in tasks.items():
                    if not task.done():
                        task.cancel()
                break
            elif result.error and result.error != "cancelled":
                self.stats.errors[source_name] = self.stats.errors.get(source_name, 0) + 1
                last_result = result
            else:
                self.stats.missed[source_name] = self.stats.missed.get(source_name, 0) + 1
                last_result = result

        if winner:
            return winner

        self.stats.total_not_found += 1
        console.print(f"  [yellow]✗[/yellow] no image: {card_name}")
        return last_result

    async def find_images_batch(
        self,
        session: aiohttp.ClientSession,
        cards: list[dict],
        concurrency: int = 5,
    ) -> list[SourceResult]:
        """
        Find images for a batch of cards with limited concurrency.
        Returns a list of SourceResults in the same order as input cards.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _find_one(card):
            async with sem:
                return await self.find_image(session, card)

        tasks = [_find_one(card) for card in cards]
        return await asyncio.gather(*tasks)
