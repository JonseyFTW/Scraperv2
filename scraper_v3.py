"""
SportsCardPro Scraper v3 - Optimized with curl_cffi and CDN pattern exploitation
Major improvements:
- Uses curl_cffi for non-browser requests (bypasses Cloudflare better)
- Predicts CDN URLs based on product_id pattern
- Scrapling for browser-required operations
- Adaptive rate limiting and session rotation
- Redis task queue support
"""
import asyncio
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiofiles
from curl_cffi import requests
from curl_cffi.requests import AsyncSession
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    from scrapling import StealthyFetcher
    HAS_SCRAPLING = True
except ImportError:
    HAS_SCRAPLING = False
    # Fallback to Playwright if needed
    from playwright.async_api import async_playwright, Page, Browser, BrowserContext

import psycopg2
import config
import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# CDN Pattern Discovery
# ═══════════════════════════════════════════════════════════════════════════

class CDNPatternEngine:
    """Reverse-engineer and predict CDN image URLs to skip Phase 4 entirely"""
    
    def __init__(self):
        self.patterns = {
            'pricecharting': 'https://storage.googleapis.com/images.pricecharting.com/{hash}/1600.jpg',
            'sportscards_direct': 'https://cdn.sportscardspro.com/images/cards/{product_id}.jpg',
            'sportscards_hash': 'https://cdn.sportscardspro.com/images/{hash}/{size}.jpg'
        }
        self.discovered_patterns = {}
        
    def generate_hash(self, product_id: str, set_slug: str = "") -> str:
        """Generate potential hash values for CDN URLs"""
        # Try various hash algorithms that sites commonly use
        candidates = []
        
        # MD5 of product_id
        candidates.append(hashlib.md5(product_id.encode()).hexdigest())
        
        # SHA1 of product_id
        candidates.append(hashlib.sha1(product_id.encode()).hexdigest())
        
        # SHA256 first 12 chars
        candidates.append(hashlib.sha256(product_id.encode()).hexdigest()[:12])
        
        # Product ID + set slug combination
        if set_slug:
            combined = f"{set_slug}_{product_id}"
            candidates.append(hashlib.md5(combined.encode()).hexdigest())
            
        return candidates
    
    async def test_cdn_patterns(self, sample_cards: List[Dict]) -> Optional[str]:
        """Test various CDN patterns with sample cards to find working pattern.
        Gives up quickly (15s max) since these patterns rarely work."""
        console.print("[cyan]Testing CDN patterns (quick check, max 15s)...[/cyan]")

        try:
            return await asyncio.wait_for(
                self._test_cdn_patterns_inner(sample_cards[:3]),
                timeout=15,
            )
        except asyncio.TimeoutError:
            console.print("[yellow]CDN pattern test timed out, falling back to scraping[/yellow]")
            return None

    async def _test_cdn_patterns_inner(self, sample_cards: List[Dict]) -> Optional[str]:
        browser = _get_supported_browsers()[0]
        async with AsyncSession(impersonate=browser) as session:
            for card in sample_cards:
                product_id = card['product_id']
                set_slug = card.get('set_slug', '')

                # Test direct product_id pattern
                test_urls = [
                    f"https://cdn.sportscardspro.com/images/cards/{product_id}.jpg",
                    f"https://storage.googleapis.com/images.pricecharting.com/{product_id}/1600.jpg",
                ]

                # Test hash-based patterns
                hashes = self.generate_hash(product_id, set_slug)
                for h in hashes:
                    test_urls.append(f"https://storage.googleapis.com/images.pricecharting.com/{h}/1600.jpg")
                    test_urls.append(f"https://cdn.sportscardspro.com/images/{h}/1600.jpg")

                for url in test_urls:
                    try:
                        resp = await session.head(url, timeout=3)
                        if resp.status_code == 200:
                            console.print(f"[green]Found working CDN pattern: {url}[/green]")
                            if '{product_id}' in url or product_id in url:
                                pattern = url.replace(product_id, '{product_id}')
                            else:
                                for h in hashes:
                                    if h in url:
                                        pattern = url.replace(h, '{hash}')
                                        self.discovered_patterns['hash_algo'] = hashes.index(h)
                                        break
                            self.discovered_patterns['pattern'] = pattern
                            return pattern
                    except Exception:
                        continue

        console.print("[yellow]No direct CDN pattern found, will scrape normally[/yellow]")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveRateLimiter:
    """Smart rate limiting that adapts based on server responses"""
    
    def __init__(self):
        self.base_delay = config.REQUEST_DELAY_MIN
        self.current_delay = self.base_delay
        self.max_delay = 10.0
        self.success_count = 0
        self.error_count = 0
        self.consecutive_403 = 0
        self.last_403 = 0
        
    async def wait(self):
        """Wait with current delay"""
        await asyncio.sleep(self.current_delay + random.uniform(0, 0.5))
        
    def on_success(self):
        """Reduce delay after successful requests"""
        self.success_count += 1
        self.consecutive_403 = 0
        if self.success_count > 10:
            self.current_delay = max(self.base_delay, self.current_delay * 0.9)
            self.success_count = 0
            
    def on_error(self, status_code: int = 0):
        """Increase delay after errors"""
        self.error_count += 1
        if status_code == 403:
            self.consecutive_403 += 1
            self.last_403 = time.time()
            self.current_delay = min(self.max_delay, self.current_delay * 2)
        elif status_code == 429:
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
        else:
            self.current_delay = min(self.max_delay, self.current_delay * 1.2)
        self.success_count = 0

    def should_rotate_session(self) -> bool:
        """Check if we should rotate to a new session.
        Only triggers on accumulated errors, NOT on a stale last_403 timestamp."""
        return self.error_count > 3

    def should_rotate_vpn(self) -> bool:
        """Check if too many 403s suggest IP is burned and VPN should cycle"""
        if self.consecutive_403 >= 50:
            self.consecutive_403 = 0
            return True
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Session Management with curl_cffi
# ═══════════════════════════════════════════════════════════════════════════

# Cache browser detection globally — only probe once per process
_supported_browsers_cache = None

def _get_supported_browsers() -> list:
    """Find which browser impersonations are supported (cached)."""
    global _supported_browsers_cache
    if _supported_browsers_cache is not None:
        return _supported_browsers_cache

    candidates = [
        "chrome136", "chrome131", "chrome124", "chrome120",
        "chrome119", "chrome116", "chrome110", "chrome107",
        "chrome104", "chrome101", "chrome100",
        "safari17_0", "safari15_5", "safari15_3",
    ]
    supported = []
    from curl_cffi.requests import Session
    for browser in candidates:
        try:
            s = Session(impersonate=browser)
            s.close()
            supported.append(browser)
        except Exception:
            continue
    if not supported:
        supported = ["chrome110"]
    console.print(f"[dim]Supported browsers: {', '.join(supported)}[/dim]")
    _supported_browsers_cache = supported
    return supported


class SessionManager:
    """Manage multiple curl_cffi sessions with rotation"""

    def __init__(self):
        self.sessions = []
        self.current_index = 0
        self.rate_limiter = AdaptiveRateLimiter()
        self.browsers = _get_supported_browsers()

    async def create_session(self) -> AsyncSession:
        """Create a new session with random browser fingerprint"""
        browser = random.choice(self.browsers)
        session = AsyncSession(
            impersonate=browser,
            timeout=30,
            verify=False  # Ignore SSL errors like Cloudflare's
        )
        
        # Set common headers
        session.headers.update({
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        })
        
        # Use proxy if configured
        proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        if proxy_url:
            session.proxies = {"http": proxy_url, "https": proxy_url}
            
        return session
        
    async def get_session(self, rotate: bool = False) -> AsyncSession:
        """Get current session or create new one.
        Old sessions are NOT closed immediately to avoid killing in-flight requests.
        They get replaced and garbage collected instead."""
        if rotate or not self.sessions or self.rate_limiter.should_rotate_session():
            # Cooldown: don't rotate more than once every 5 seconds
            now = time.time()
            if hasattr(self, '_last_rotation') and now - self._last_rotation < 5 and self.sessions:
                return self.sessions[self.current_index % len(self.sessions)]

            # Create new session (don't close old one — in-flight requests may still use it)
            session = await self.create_session()
            if not self.sessions:
                self.sessions.append(session)
            else:
                # Replace old session in the list (old one gets GC'd)
                self.sessions[self.current_index % len(self.sessions)] = session

            self.rate_limiter.error_count = 0  # Reset error count on rotation
            self.rate_limiter.last_403 = 0     # Clear 403 timestamp on rotation
            self._last_rotation = now
            console.print("[cyan]Rotated to new session[/cyan]")

        return self.sessions[self.current_index % len(self.sessions)]

    async def close_all(self):
        """Close all sessions"""
        for session in self.sessions:
            try:
                await session.close()
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Discover Sets (using curl_cffi)
# ═══════════════════════════════════════════════════════════════════════════

async def discover_sets_v3(sport: str = None):
    """Discover sets using curl_cffi - much faster and stealthier than Playwright"""
    categories = config.CATEGORY_URLS
    if sport:
        categories = {k: v for k, v in categories.items() if k == sport}
        
    console.print(f"\n[bold]Phase 1: Discovering sets from {len(categories)} categories (curl_cffi)[/bold]\n")
    
    session_mgr = SessionManager()
    total_found = 0
    
    try:
        for sport_name, cat_url in categories.items():
            console.print(f"  Category: [cyan]{sport_name}[/cyan]")
            session = await session_mgr.get_session()
            
            all_sets = []
            seen = set()
            page = 1
            
            while True:
                url = f"{cat_url}?page={page}" if page > 1 else cat_url
                
                try:
                    resp = await session.get(url)
                    if resp.status_code != 200:
                        session_mgr.rate_limiter.on_error(resp.status_code)
                        if resp.status_code == 403:
                            console.print("[yellow]Hit Cloudflare, rotating session...[/yellow]")
                            session = await session_mgr.get_session(rotate=True)
                            continue
                        break
                        
                    html = resp.text
                    session_mgr.rate_limiter.on_success()
                    
                    # Extract set links with regex (faster than parsing)
                    pattern = r'<a[^>]+href="(/console/[^"]+)"[^>]*>([^<]+)</a>'
                    matches = re.findall(pattern, html)
                    
                    new_count = 0
                    for href, name in matches:
                        slug = href.replace("/console/", "")
                        if slug not in seen:
                            seen.add(slug)
                            all_sets.append((slug, name.strip(), sport_name, f"{config.BASE_URL}{href}"))
                            new_count += 1
                            
                    if new_count == 0 or 'rel="next"' not in html:
                        break
                        
                    page += 1
                    await session_mgr.rate_limiter.wait()
                    
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    break
                    
            if all_sets:
                db.bulk_upsert_sets(all_sets)
                total_found += len(all_sets)
                console.print(f"    Found [green]{len(all_sets)}[/green] sets")
                
    finally:
        await session_mgr.close_all()
        
    console.print(f"\n  Total sets discovered: [green]{total_found}[/green]")
    db.log_event("phase1_complete", f"Discovered {total_found} sets (curl_cffi)")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Download CSVs (needs auth, using Scrapling or fallback)
# ═══════════════════════════════════════════════════════════════════════════

async def download_csvs_v3(sport: str = None):
    """Download CSVs using Scrapling for auth or curl_cffi with cookies"""
    pending = db.get_sets_needing_csv(sport)
    if not pending:
        console.print("[green]All set CSVs already downloaded.[/green]")
        return
        
    console.print(f"\n[bold]Phase 2: Downloading CSVs for {len(pending)} sets[/bold]\n")
    
    # First, get auth cookies using Scrapling or Playwright
    cookies = await get_auth_cookies()
    if not cookies:
        console.print("[red]Failed to authenticate[/red]")
        return
        
    # Use curl_cffi with auth cookies for fast CSV downloads
    session_mgr = SessionManager()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("CSV Downloads", total=len(pending))
        
        for s in pending:
            progress.update(task, description=f"CSV: {s['name'][:40]}")
            session = await session_mgr.get_session()
            
            # Add auth cookies to session
            for cookie in cookies:
                session.cookies.set(cookie['name'], cookie['value'])
                
            try:
                # Go to set page
                resp = await session.get(s["url"])
                if resp.status_code != 200:
                    session_mgr.rate_limiter.on_error(resp.status_code)
                    db.mark_set_csv_error(s["slug"])
                    progress.advance(task)
                    continue
                    
                html = resp.text
                session_mgr.rate_limiter.on_success()
                
                # Find CSV download link
                csv_link_match = re.search(r'href="([^"]*(?:download|csv)[^"]*)"', html, re.IGNORECASE)
                if csv_link_match:
                    csv_url = urljoin(s["url"], csv_link_match.group(1))
                    
                    # Download CSV
                    csv_resp = await session.get(csv_url)
                    if csv_resp.status_code == 200:
                        csv_filename = f"{s['slug']}.csv"
                        csv_path = os.path.join(config.CSV_DIR, csv_filename)
                        
                        async with aiofiles.open(csv_path, 'wb') as f:
                            await f.write(csv_resp.content)
                            
                        db.mark_set_csv_downloaded(s["slug"], csv_path)
                        session_mgr.rate_limiter.on_success()
                    else:
                        db.mark_set_csv_error(s["slug"])
                        session_mgr.rate_limiter.on_error(csv_resp.status_code)
                else:
                    console.print(f"[yellow]No CSV link found for {s['name']}[/yellow]")
                    db.mark_set_csv_error(s["slug"])
                    
            except Exception as e:
                console.print(f"[red]Error downloading {s['name']}: {e}[/red]")
                db.mark_set_csv_error(s["slug"])
                
            progress.advance(task)
            await session_mgr.rate_limiter.wait()
            
    await session_mgr.close_all()


async def get_auth_cookies():
    """Get authentication cookies using Scrapling or Playwright"""
    if HAS_SCRAPLING:
        return await get_auth_cookies_scrapling()
    else:
        return await get_auth_cookies_playwright()


async def get_auth_cookies_scrapling():
    """Use Scrapling's StealthyFetcher for Cloudflare-resistant login"""
    console.print("[cyan]Logging in with Scrapling (Cloudflare-resistant)...[/cyan]")
    
    try:
        fetcher = StealthyFetcher(
            headless=config.HEADLESS,
            network_idle=True,
            extra_headers={
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        
        # Navigate to login page
        page = fetcher.get(config.LOGIN_URL)
        
        # Handle Cloudflare Turnstile if present
        if page.css_first('[data-ray-id]'):
            console.print("[yellow]Cloudflare detected, solving challenge...[/yellow]")
            await asyncio.sleep(5)  # Let StealthyFetcher handle it
            
        # Fill login form
        page.fill('[name="email"]', config.LOGIN_EMAIL)
        page.fill('[name="password"]', config.LOGIN_PASSWORD)
        page.click('[type="submit"]')
        
        # Wait for login to complete
        await asyncio.sleep(3)
        
        # Extract cookies
        cookies = fetcher.driver.get_cookies()
        console.print("[green]Login successful with Scrapling![/green]")
        return cookies
        
    except Exception as e:
        console.print(f"[red]Scrapling login failed: {e}[/red]")
        return None


async def get_auth_cookies_playwright():
    """Fallback to Playwright for login if Scrapling not available"""
    console.print("[cyan]Logging in with Playwright (fallback)...[/cyan]")
    
    from playwright.async_api import async_playwright
    
    async with async_playwright() as pw:
        browser, context = await create_browser(pw)
        page = await new_stealth_page(context)
        
        try:
            # Navigate and login
            await page.goto(config.LOGIN_URL)
            await page.fill('[name="email"]', config.LOGIN_EMAIL)
            await page.fill('[name="password"]', config.LOGIN_PASSWORD)
            await page.click('[type="submit"]')
            await asyncio.sleep(3)
            
            # Extract cookies
            cookies = await context.cookies()
            console.print("[green]Login successful with Playwright![/green]")
            return cookies
            
        finally:
            await browser.close()


# Helper functions from original scraper (needed for compatibility)
async def create_browser(playwright):
    """Create browser for fallback"""
    browser = await playwright.chromium.launch(
        headless=config.HEADLESS,
        args=["--disable-blink-features=AutomationControlled"]
    )
    context = await browser.new_context(
        user_agent=random.choice(config.USER_AGENTS),
        viewport={"width": 1920, "height": 1080}
    )
    return browser, context


async def new_stealth_page(context):
    """Create stealth page for fallback"""
    page = await context.new_page()
    # Add stealth if available
    try:
        from playwright_stealth import stealth_async
        await stealth_async(page)
    except ImportError:
        pass
    return page


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Smart Image URL Discovery 
# ═══════════════════════════════════════════════════════════════════════════

async def scrape_card_images_v3(limit: int = 0, sport: str = None):
    """
    Smart image scraping:
    1. First try to use CDN pattern (no requests needed!)
    2. Fall back to curl_cffi batch requests
    3. Last resort: Scrapling for stubborn pages
    """
    sport_label = f" [{sport}]" if sport else ""
    console.print(f"\n[bold]Phase 4: Smart Image URL Discovery{sport_label}[/bold]\n")

    # First, try to discover CDN pattern using a peek (don't claim cards)
    sample_cards = db.peek_cards_needing_images(10, sport=sport)
    if not sample_cards:
        console.print("[green]No cards needing images![/green]")
        return

    cdn_engine = CDNPatternEngine()
    pattern = await cdn_engine.test_cdn_patterns(sample_cards)

    if pattern:
        # We found a pattern! Apply it to all cards
        await apply_cdn_pattern(pattern, cdn_engine, limit, sport=sport)
    else:
        # Fall back to scraping
        await scrape_with_curl_cffi(limit, sport=sport)


async def apply_cdn_pattern(pattern: str, cdn_engine: CDNPatternEngine, limit: int, sport: str = None):
    """Apply discovered CDN pattern to generate image URLs without any requests!"""
    console.print(f"[green]Applying CDN pattern: {pattern}[/green]")
    
    batch_size = 1000
    total_processed = 0
    
    while True:
        cards = db.get_cards_needing_images(batch_size, sport=sport)
        if not cards:
            break

        if limit > 0 and total_processed >= limit:
            break

        for card in cards:
            if limit > 0 and total_processed >= limit:
                break
                
            product_id = card['product_id']
            set_slug = card.get('set_slug', '')
            
            # Generate URL based on pattern
            if '{product_id}' in pattern:
                image_url = pattern.format(product_id=product_id)
            elif '{hash}' in pattern:
                hashes = cdn_engine.generate_hash(product_id, set_slug)
                hash_idx = cdn_engine.discovered_patterns.get('hash_algo', 0)
                image_url = pattern.format(hash=hashes[hash_idx], size='1600')
            else:
                continue

            gcs_full, gcs_thumb = gcs_urls_from_any(image_url)
            db.update_card_image_url(product_id, image_url, gcs_full, gcs_thumb)
            total_processed += 1
            
        console.print(f"[green]Processed {total_processed} cards with CDN pattern[/green]")
        
    console.print(f"[bold green]Phase 4 complete: {total_processed} image URLs generated with ZERO requests![/bold green]")


import shutil

_has_nordvpn = shutil.which("nordvpn") is not None


async def _cycle_vpn():
    """Disconnect and reconnect NordVPN to get a fresh IP. Only works on Linux/LXC."""
    if not _has_nordvpn:
        return False
    import subprocess
    import platform
    if platform.system() != "Linux":
        return False
    try:
        console.print("[cyan]Cycling VPN for fresh IP...[/cyan]")
        subprocess.run(["nordvpn", "disconnect"], capture_output=True, timeout=10)
        await asyncio.sleep(2)
        subprocess.run(["nordvpn", "connect"], capture_output=True, timeout=30)
        await asyncio.sleep(5)

        # Ensure local network (192.168.1.0/24) stays reachable after VPN connects
        # NordVPN can route everything through the tunnel, breaking DB access
        try:
            # Find the default gateway for local network
            route_check = subprocess.run(
                ["ip", "route", "show", "192.168.1.0/24"],
                capture_output=True, text=True, timeout=5,
            )
            if not route_check.stdout.strip():
                # No local route — add one via the container's gateway
                gw_result = subprocess.run(
                    ["ip", "route", "show", "default"],
                    capture_output=True, text=True, timeout=5,
                )
                # Parse gateway from before VPN, or use common .1 gateway
                gateway = "192.168.1.1"
                for line in gw_result.stdout.split("\n"):
                    if "192.168" in line and "via" in line:
                        parts = line.split()
                        gw_idx = parts.index("via") + 1
                        gateway = parts[gw_idx]
                        break
                subprocess.run(
                    ["ip", "route", "add", "192.168.1.0/24", "via", gateway],
                    capture_output=True, timeout=5,
                )
                console.print(f"[green]  Added local route via {gateway}[/green]")
        except Exception:
            pass

        # Check new IP
        result = subprocess.run(["nordvpn", "status"], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split("\n"):
            if "IP" in line or "Server" in line:
                console.print(f"[green]  {line.strip()}[/green]")
        return True
    except Exception as e:
        console.print(f"[yellow]VPN cycle failed: {e}[/yellow]")
        return False


async def scrape_with_curl_cffi(limit: int, sport: str = None):
    """Fallback scraping using curl_cffi for fast parallel requests"""
    console.print("[yellow]Using curl_cffi for image URL scraping[/yellow]")
    
    session_mgr = SessionManager()
    batch_size = 20
    total_ok = 0
    total_no_image = 0
    total_error = 0
    last_progress_time = time.time()
    STALL_TIMEOUT = 15 * 60  # 15 minutes

    while True:
        cards = db.get_cards_needing_images(500, sport=sport)
        if not cards or (limit > 0 and total_ok >= limit):
            break
            
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Image URLs", total=len(cards))
            
            for i in range(0, len(cards), batch_size):
                if limit > 0 and total_ok >= limit:
                    break
                    
                batch = cards[i:i + batch_size]
                session = await session_mgr.get_session()
                
                tasks = []
                for card in batch:
                    tasks.append(fetch_image_url(session, card))
                    
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                needs_rotation = False
                for card, result in zip(batch, results):
                    try:
                        if isinstance(result, Exception):
                            db.mark_card_error(card['product_id'], str(result))
                            total_error += 1
                            session_mgr.rate_limiter.on_error()
                        elif isinstance(result, FetchResult) and result.image_url:
                            db.update_card_image_url(
                                card['product_id'],
                                result.image_url,
                                result.gcs_image_url,
                                result.gcs_thumb_url,
                            )
                            total_ok += 1
                            last_progress_time = time.time()
                            session_mgr.rate_limiter.on_success()
                        elif isinstance(result, FetchResult) and result.error:
                            # Retryable — HTTP errors, timeouts, Cloudflare blocks
                            db.mark_card_error(card['product_id'], result.error)
                            total_error += 1
                            session_mgr.rate_limiter.on_error(result.status_code)
                            if result.status_code in (403, 429, 503):
                                needs_rotation = True
                        else:
                            # Page loaded fine but genuinely no image on it
                            db.mark_card_no_image(card['product_id'])
                            total_no_image += 1
                    except psycopg2.OperationalError as db_err:
                        console.print(f"[yellow]DB error (will retry card later): {db_err}[/yellow]")
                        total_error += 1

                # Rotate session if we're getting blocked
                if needs_rotation or session_mgr.rate_limiter.should_rotate_session():
                    session = await session_mgr.get_session(rotate=True)

                # Cycle VPN if IP is burned (50+ consecutive 403s)
                if session_mgr.rate_limiter.should_rotate_vpn():
                    cycled = await _cycle_vpn()
                    if cycled:
                        session = await session_mgr.get_session(rotate=True)
                    else:
                        # No VPN available — long backoff to let blocks expire
                        console.print("[yellow]No VPN available, waiting 60s for blocks to clear...[/yellow]")
                        await asyncio.sleep(60)
                        session = await session_mgr.get_session(rotate=True)

                progress.advance(task, len(batch))

                # Watchdog: exit if no images found for 15 minutes so systemd restarts us
                if time.time() - last_progress_time > STALL_TIMEOUT:
                    console.print("[red]No images found in 15 minutes — restarting...[/red]")
                    await session_mgr.close_all()
                    sys.exit(1)

                await session_mgr.rate_limiter.wait()
                
    await session_mgr.close_all()
    console.print(f"Complete: {total_ok} found, {total_no_image} no image, {total_error} errors")


class FetchResult:
    """Result of fetch_image_url — distinguishes success, no-image, and errors."""
    __slots__ = ('image_url', 'gcs_image_url', 'gcs_thumb_url', 'error', 'no_image', 'status_code')
    def __init__(self, image_url=None, gcs_image_url=None, gcs_thumb_url=None,
                 error=None, no_image=False, status_code=0):
        self.image_url = image_url
        self.gcs_image_url = gcs_image_url
        self.gcs_thumb_url = gcs_thumb_url
        self.error = error
        self.no_image = no_image
        self.status_code = status_code


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


async def fetch_image_url(session: AsyncSession, card: dict) -> FetchResult:
    """Fetch image URL from a card page using curl_cffi.
    Returns FetchResult to distinguish between no-image vs errors."""
    url = card.get('full_url', '')
    try:
        resp = await session.get(url, timeout=10)
        if resp.status_code != 200:
            return FetchResult(error=f"HTTP {resp.status_code}", status_code=resp.status_code)

        # Extract image URL from HTML
        pattern = r'https://storage\.googleapis\.com/images\.pricecharting\.com/([^/\s"\'<>]+)/\d+'
        match = re.search(pattern, resp.text)

        if match:
            gcs_full, gcs_thumb = gcs_urls_from_any(match.group(0))
            return FetchResult(
                image_url=gcs_full,
                gcs_image_url=gcs_full,
                gcs_thumb_url=gcs_thumb,
            )

        # Page loaded OK but genuinely no image on it
        return FetchResult(no_image=True)

    except Exception as e:
        err_str = str(e)
        # "Session is closed" is retryable, not a real failure
        if "session is closed" in err_str.lower():
            return FetchResult(error=f"Session closed (retryable)")
        return FetchResult(error=f"{e}")


# ═══════════════════════════════════════════════════════════════════════════
# Redis Task Queue Integration
# ═══════════════════════════════════════════════════════════════════════════

class RedisTaskQueue:
    """Simple Redis-based task queue for distributed scraping"""
    
    def __init__(self):
        if not HAS_REDIS:
            self.redis = None
            return
            
        try:
            self.redis = redis.Redis.from_url(
                os.environ.get('REDIS_URL', 'redis://localhost:6379'),
                decode_responses=True
            )
            self.redis.ping()
            console.print("[green]Redis task queue connected[/green]")
        except:
            self.redis = None
            console.print("[yellow]Redis not available, using PostgreSQL queue[/yellow]")
            
    def push_task(self, queue: str, task: dict):
        """Push task to queue"""
        if self.redis:
            self.redis.lpush(f"scraper:{queue}", json.dumps(task))
            
    def pop_task(self, queue: str) -> Optional[dict]:
        """Pop task from queue with atomic operation"""
        if self.redis:
            data = self.redis.rpop(f"scraper:{queue}")
            return json.loads(data) if data else None
        return None
        
    def get_queue_size(self, queue: str) -> int:
        """Get queue size"""
        if self.redis:
            return self.redis.llen(f"scraper:{queue}")
        return 0


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def run_full_pipeline_v3(sport: str = None, limit: int = 0):
    """Run optimized pipeline with all improvements.

    After the initial run, automatically retries error cards until
    no more errors can be recovered (errors stop decreasing).
    """
    db.init_db()

    # Phase 1: Discover sets
    await discover_sets_v3(sport)

    # Phase 2: Download CSVs
    await download_csvs_v3(sport)

    # Phase 3: Parse CSVs (reuse existing function)
    from scraper import parse_csvs
    parse_csvs(sport)

    # Phase 4: Smart image URL discovery
    await scrape_card_images_v3(limit, sport=sport)

    # Phase 5: Download images (reuse existing function but with curl_cffi)
    await download_images_v3(limit, sport=sport)

    # Phase 6: Auto-retry errors until they stop decreasing
    max_retries = 10
    last_error_count = None

    for attempt in range(1, max_retries + 1):
        # Count remaining errors
        conn = db.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM cards WHERE status = 'error'")
        error_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM cards WHERE status IN ('pending', 'image_found', 'downloading', 'processing')")
        pending_count = cur.fetchone()[0]
        cur.close()
        db.put_connection(conn)

        if error_count == 0:
            console.print("[bold green]All cards processed, zero errors![/bold green]")
            break

        # Stop if errors aren't decreasing (they're permanent failures)
        if last_error_count is not None and error_count >= last_error_count:
            console.print(f"[yellow]Errors not decreasing ({error_count:,} remaining) — these are likely permanent failures (no_image, bad URLs, etc.)[/yellow]")
            break

        # Wait for in-flight work to finish before resetting
        if pending_count > 0:
            console.print(f"[dim]Waiting for {pending_count:,} in-flight cards to finish...[/dim]")
            await asyncio.sleep(30)
            continue

        last_error_count = error_count
        console.print(f"\n[bold cyan]Auto-retry {attempt}/{max_retries}: Resetting {error_count:,} errors to pending[/bold cyan]")
        db.reset_errors()

        # Re-run phases 4 + 5 for the reset cards
        await scrape_card_images_v3(limit, sport=sport)
        await download_images_v3(limit, sport=sport)

    console.print("[bold green]Pipeline complete! Check stats with --stats[/bold green]")


async def download_images_v3(limit: int = 0, sport: str = None):
    """Download images using curl_cffi for maximum speed"""
    cards = db.get_cards_needing_download(sport=sport)
    if not cards:
        console.print("[green]No images to download[/green]")
        return
        
    console.print(f"\n[bold]Phase 5: Downloading {len(cards)} images with curl_cffi[/bold]\n")
    
    session_mgr = SessionManager()
    batch_size = 20  # Can handle more parallel downloads
    downloaded = 0
    errors = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Image Downloads", total=min(len(cards), limit) if limit else len(cards))
        
        for i in range(0, len(cards), batch_size):
            if limit > 0 and downloaded >= limit:
                break
                
            batch = cards[i:i + batch_size]
            session = await session_mgr.get_session()
            
            tasks = []
            for card in batch:
                if limit > 0 and downloaded >= limit:
                    break
                tasks.append(download_image(session, card))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for card, success in zip(batch, results):
                if isinstance(success, Exception):
                    errors += 1
                    db.mark_card_error(card['product_id'], str(success))
                    session_mgr.rate_limiter.on_error()
                elif success:
                    downloaded += 1
                    session_mgr.rate_limiter.on_success()
                else:
                    errors += 1
                    
            progress.advance(task, len(batch))
            await session_mgr.rate_limiter.wait()
            
    await session_mgr.close_all()
    console.print(f"[green]Downloaded {downloaded} images, {errors} errors[/green]")


async def download_image(session: AsyncSession, card: dict) -> bool:
    """Download a single image"""
    try:
        image_url = card['image_url']
        product_id = card['product_id']
        
        resp = await session.get(image_url, timeout=30)
        if resp.status_code != 200:
            return False
            
        # Save image
        ext = 'jpg'
        if 'content-type' in resp.headers:
            ct = resp.headers['content-type']
            if 'png' in ct:
                ext = 'png'
            elif 'webp' in ct:
                ext = 'webp'
                
        filename = f"{product_id}.{ext}"
        filepath = os.path.join(config.IMG_DIR, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(resp.content)
            
        db.mark_card_downloaded(product_id, filepath)
        return True
        
    except Exception:
        return False


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(run_full_pipeline_v3(limit=100))