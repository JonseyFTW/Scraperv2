"""
Multi-source card image scraper.

Uses curl_cffi for TLS fingerprint spoofing (same as scraper_v3).
Sites like TCDB and COMC block plain Python HTTP clients but
curl_cffi impersonates Chrome's TLS handshake.

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
import base64
import logging
import os
import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote, quote_plus

from curl_cffi.requests import AsyncSession
from rich.console import Console

import config

log = logging.getLogger(__name__)
console = Console()

# Browser impersonation target for curl_cffi
IMPERSONATE = "chrome136"

# ── Shared helpers ────────────────────────────────────────────────────────


def _clean_card_name(product_name: str) -> str:
    """Strip variant info and card number for broader search."""
    name = re.sub(r'\s*#\d+\s*$', '', product_name)
    name = re.sub(r'\[.*?\]', '', name).strip()
    return name


def _build_search_query(card: dict) -> str:
    """Build a search string: 'SetName PlayerName #Number'."""
    console_name = card.get("console_name", "")
    product = card.get("product_name", "")
    return f"{console_name} {product}".strip()


@dataclass
class SourceResult:
    """Result from an image source lookup."""
    source: str
    image_url: str | None = None
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# Source 1: Wayback Machine (cached SportsCardPro pages)
# ═══════════════════════════════════════════════════════════════════════════

_SCP_IMAGE_RE = re.compile(
    r'https://storage\.googleapis\.com/images\.pricecharting\.com/([^/\s"\'<>]+)/(\d+)(?:\.jpg)?'
)


async def search_wayback(session: AsyncSession, card: dict) -> SourceResult:
    """Check Wayback Machine for cached SportsCardPro card page."""
    full_url = card.get("full_url")
    if not full_url:
        return SourceResult(source="wayback", error="no full_url")

    try:
        # Step 1: Check availability
        avail_url = f"http://archive.org/wayback/available?url={quote(full_url, safe='')}"
        resp = await session.get(avail_url, timeout=10)
        if resp.status_code != 200:
            return SourceResult(source="wayback", error=f"avail HTTP {resp.status_code}")
        data = resp.json()

        snapshot = data.get("archived_snapshots", {}).get("closest")
        if not snapshot or not snapshot.get("available"):
            return SourceResult(source="wayback", error="not archived")

        # Step 2: Fetch archived page (raw mode)
        archive_url = snapshot["url"]
        archive_url = re.sub(r'/web/(\d+)/', r'/web/\1id_/', archive_url)
        resp = await session.get(archive_url, timeout=10)
        if resp.status_code != 200:
            return SourceResult(source="wayback", error=f"fetch HTTP {resp.status_code}")

        # Step 3: Extract image URL
        match = _SCP_IMAGE_RE.search(resp.text)
        if match:
            img_hash = match.group(1)
            return SourceResult(
                source="wayback",
                image_url=f"https://storage.googleapis.com/images.pricecharting.com/{img_hash}/1600.jpg",
            )
        return SourceResult(source="wayback", error="no image in page")

    except Exception as e:
        return SourceResult(source="wayback", error=str(e)[:100])


# ═══════════════════════════════════════════════════════════════════════════
# Source 2: TCDB (Trading Card Database)
# ═══════════════════════════════════════════════════════════════════════════

_TCDB_CARD_IMG_RE = re.compile(
    r'<img[^>]+src=["\']([^"\']*?/Images/Cards/[^"\']+)["\']',
    re.IGNORECASE,
)


async def search_tcdb(session: AsyncSession, card: dict) -> SourceResult:
    """Search TCDB for card image using browser-like TLS fingerprint."""
    product_name = card.get("product_name", "")
    console_name = card.get("console_name", "")

    player_name = _clean_card_name(product_name)
    year_match = re.search(r'((?:19|20)\d{2})', console_name)
    year = year_match.group(1) if year_match else ""
    set_name = re.sub(
        r'^(?:Football|Baseball|Basketball|Hockey|Soccer|Racing|Wrestling|UFC)\s+Cards\s+\d{4}\s*',
        '', console_name, flags=re.IGNORECASE,
    ).strip()

    query = f"{player_name} {year} {set_name}".strip()
    if not query:
        return SourceResult(source="tcdb", error="empty query")

    try:
        resp = await session.get(
            "https://www.tcdb.com/Search.cfm/fu/1",
            params={"searchtext": query},
            timeout=10,
            allow_redirects=True,
            impersonate=IMPERSONATE,
        )
        if resp.status_code != 200:
            return SourceResult(source="tcdb", error=f"HTTP {resp.status_code}")

        html = resp.text

        # Look for card images in results
        img_matches = _TCDB_CARD_IMG_RE.findall(html)
        if img_matches:
            img_url = img_matches[0]
            if not img_url.startswith("http"):
                img_url = f"https://www.tcdb.com{img_url}"
            return SourceResult(source="tcdb", image_url=img_url)

        # Follow first card link if present
        card_link = re.search(r'href="(/ViewCard\.cfm/[^"]+)"', html)
        if card_link:
            resp2 = await session.get(
                f"https://www.tcdb.com{card_link.group(1)}",
                timeout=10,
                allow_redirects=True,
                impersonate=IMPERSONATE,
            )
            if resp2.status_code == 200:
                imgs = _TCDB_CARD_IMG_RE.findall(resp2.text)
                if imgs:
                    img_url = imgs[0]
                    if not img_url.startswith("http"):
                        img_url = f"https://www.tcdb.com{img_url}"
                    return SourceResult(source="tcdb", image_url=img_url)

        return SourceResult(source="tcdb", error="no results")

    except Exception as e:
        return SourceResult(source="tcdb", error=str(e)[:100])


# ═══════════════════════════════════════════════════════════════════════════
# Source 3: COMC (Check Out My Cards)
# ═══════════════════════════════════════════════════════════════════════════

_COMC_IMG_RE = re.compile(
    r'https?://img\.comc\.com/i/[^\s"\'<>]+\.(?:jpg|png|jpeg)',
    re.IGNORECASE,
)


async def search_comc(session: AsyncSession, card: dict) -> SourceResult:
    """Search COMC for card image using browser-like TLS fingerprint."""
    search_query = _build_search_query(card)
    if not search_query:
        return SourceResult(source="comc", error="empty query")

    try:
        resp = await session.get(
            "https://www.comc.com/Cards",
            params={"search": search_query},
            timeout=10,
            allow_redirects=True,
            impersonate=IMPERSONATE,
        )
        if resp.status_code != 200:
            return SourceResult(source="comc", error=f"HTTP {resp.status_code}")

        html = resp.text

        # Look for card images
        img_matches = _COMC_IMG_RE.findall(html)
        if img_matches:
            img_url = img_matches[0]
            if "?" in img_url:
                img_url = img_url.split("?")[0]
            return SourceResult(source="comc", image_url=f"{img_url}?size=original")

        # Follow first card detail link
        card_link = re.search(r'href="(/Cards/[^"]+/\d+)"', html)
        if card_link:
            resp2 = await session.get(
                f"https://www.comc.com{card_link.group(1)}",
                timeout=10,
                allow_redirects=True,
                impersonate=IMPERSONATE,
            )
            if resp2.status_code == 200:
                imgs = _COMC_IMG_RE.findall(resp2.text)
                if imgs:
                    img_url = imgs[0]
                    if "?" in img_url:
                        img_url = img_url.split("?")[0]
                    return SourceResult(source="comc", image_url=f"{img_url}?size=original")

        return SourceResult(source="comc", error="no results")

    except Exception as e:
        return SourceResult(source="comc", error=str(e)[:100])


# ═══════════════════════════════════════════════════════════════════════════
# Source 4: eBay Browse API
# ═══════════════════════════════════════════════════════════════════════════

_ebay_token: dict = {"access_token": None, "expires_at": 0}


async def _get_ebay_token(session: AsyncSession) -> str | None:
    """Get eBay OAuth token via client credentials grant."""
    client_id = os.environ.get("EBAY_CLIENT_ID", "")
    client_secret = os.environ.get("EBAY_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        return None

    if _ebay_token["access_token"] and time.time() < _ebay_token["expires_at"] - 60:
        return _ebay_token["access_token"]

    try:
        creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        resp = await session.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {creds}",
            },
            data="grant_type=client_credentials&scope=https://api.ebay.com/oauth/api_scope",
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        _ebay_token["access_token"] = data["access_token"]
        _ebay_token["expires_at"] = time.time() + data.get("expires_in", 7200)
        return _ebay_token["access_token"]
    except Exception:
        return None


async def search_ebay(session: AsyncSession, card: dict) -> SourceResult:
    """Search eBay Browse API for card listing images."""
    token = await _get_ebay_token(session)
    if not token:
        return SourceResult(source="ebay", error="no credentials")

    query = _build_search_query(card)
    if not query:
        return SourceResult(source="ebay", error="empty query")
    if len(query) > 100:
        query = query[:100]

    try:
        resp = await session.get(
            "https://api.ebay.com/buy/browse/v1/item_summary/search",
            params={"q": query, "category_ids": "261328", "limit": "3"},
            headers={
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                "Accept": "application/json",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return SourceResult(source="ebay", error=f"HTTP {resp.status_code}")

        items = resp.json().get("itemSummaries", [])
        for item in items:
            img_url = item.get("image", {}).get("imageUrl")
            if img_url:
                return SourceResult(source="ebay", image_url=img_url)

        return SourceResult(source="ebay", error="no images")

    except Exception as e:
        return SourceResult(source="ebay", error=str(e)[:100])


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Source Fallback Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

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
    Tries multiple image sources in parallel to find a card image.
    Uses curl_cffi with Chrome TLS fingerprint for anti-bot bypass.
    """

    def __init__(
        self,
        sources: list[str] | None = None,
    ):
        if sources:
            self.sources = [(name, fn) for name, fn in ALL_SOURCES if name in sources]
        else:
            self.sources = list(ALL_SOURCES)
        self.stats = MultiSourceStats()

    async def find_image(
        self,
        session: AsyncSession,
        card: dict,
    ) -> SourceResult:
        """
        Fire all sources in parallel. First one to return an image wins.
        """
        card_name = card.get("product_name", "?")[:50]

        async def _run_source(name, fn):
            try:
                result = await asyncio.wait_for(fn(session, card), timeout=15)
                return (name, result)
            except asyncio.TimeoutError:
                return (name, SourceResult(source=name, error="timeout"))
            except asyncio.CancelledError:
                return (name, SourceResult(source=name, error="cancelled"))
            except Exception as e:
                return (name, SourceResult(source=name, error=str(e)[:100]))

        tasks = [
            asyncio.create_task(_run_source(name, fn))
            for name, fn in self.sources
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        winner = None
        errors_for_card = []
        last_result = SourceResult(source="none", error="no sources")

        for item in results:
            if isinstance(item, Exception):
                continue
            name, result = item

            if result.image_url and not winner:
                winner = result
                self.stats.found[name] = self.stats.found.get(name, 0) + 1
                self.stats.total_found += 1
                console.print(f"  [green]✓ {name}[/green]: {card_name}")
            elif result.error and result.error != "cancelled":
                self.stats.errors[name] = self.stats.errors.get(name, 0) + 1
                errors_for_card.append(f"{name}={result.error[:30]}")
                last_result = result
            else:
                self.stats.missed[name] = self.stats.missed.get(name, 0) + 1
                last_result = result

        if winner:
            return winner

        self.stats.total_not_found += 1
        err_summary = " | ".join(errors_for_card) if errors_for_card else "no results"
        console.print(f"  [yellow]✗[/yellow] {card_name} [{err_summary}]")
        return last_result

    async def find_images_batch(
        self,
        session: AsyncSession,
        cards: list[dict],
        concurrency: int = 5,
    ) -> list[SourceResult]:
        """Find images for a batch of cards with limited concurrency."""
        sem = asyncio.Semaphore(concurrency)

        async def _find_one(card):
            async with sem:
                return await self.find_image(session, card)

        tasks = [_find_one(card) for card in cards]
        return await asyncio.gather(*tasks)
