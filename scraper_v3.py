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
        """Test various CDN patterns with sample cards to find working pattern"""
        console.print("[cyan]Testing CDN patterns to eliminate Phase 4...[/cyan]")
        
        async with AsyncSession(impersonate="chrome136") as session:
            for card in sample_cards[:10]:  # Test with 10 cards
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
                        resp = await session.head(url, timeout=5)
                        if resp.status_code == 200:
                            console.print(f"[green]✓ Found working pattern: {url}[/green]")
                            # Extract pattern
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
                    except:
                        continue
                        
        console.print("[yellow]No direct CDN pattern found, will need Phase 4[/yellow]")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Rate Limiting
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveRateLimiter:
    """Smart rate limiting that adapts based on server responses"""
    
    def __init__(self):
        self.base_delay = config.REQUEST_DELAY_MIN
        self.current_delay = self.base_delay
        self.max_delay = 30.0
        self.success_count = 0
        self.error_count = 0
        self.last_403 = 0
        
    async def wait(self):
        """Wait with current delay"""
        await asyncio.sleep(self.current_delay + random.uniform(0, 0.5))
        
    def on_success(self):
        """Reduce delay after successful requests"""
        self.success_count += 1
        if self.success_count > 10:
            self.current_delay = max(self.base_delay, self.current_delay * 0.9)
            self.success_count = 0
            
    def on_error(self, status_code: int = 0):
        """Increase delay after errors"""
        self.error_count += 1
        if status_code == 403:
            self.last_403 = time.time()
            self.current_delay = min(self.max_delay, self.current_delay * 2)
        elif status_code == 429:
            self.current_delay = min(self.max_delay, self.current_delay * 1.5)
        else:
            self.current_delay = min(self.max_delay, self.current_delay * 1.2)
        self.success_count = 0
        
    def should_rotate_session(self) -> bool:
        """Check if we should rotate to a new session"""
        return (self.error_count > 3 or 
                (self.last_403 and time.time() - self.last_403 < 300))


# ═══════════════════════════════════════════════════════════════════════════
# Session Management with curl_cffi
# ═══════════════════════════════════════════════════════════════════════════

class SessionManager:
    """Manage multiple curl_cffi sessions with rotation"""
    
    def __init__(self):
        self.sessions = []
        self.current_index = 0
        self.browsers = ["chrome136", "chrome135", "firefox120", "safari15_5"]
        self.rate_limiter = AdaptiveRateLimiter()
        
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
        """Get current session or create new one"""
        if rotate or not self.sessions or self.rate_limiter.should_rotate_session():
            # Close old session if rotating
            if self.sessions and self.current_index < len(self.sessions):
                await self.sessions[self.current_index].close()
                
            # Create new session
            session = await self.create_session()
            if rotate or not self.sessions:
                self.sessions.append(session)
            else:
                self.sessions[self.current_index] = session
                
            self.current_index = (self.current_index + 1) % max(1, len(self.sessions))
            self.rate_limiter.error_count = 0  # Reset error count on rotation
            console.print("[cyan]Rotated to new session[/cyan]")
            
        return self.sessions[self.current_index % len(self.sessions)]
        
    async def close_all(self):
        """Close all sessions"""
        for session in self.sessions:
            await session.close()


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
        page.click('button[type="submit"]')
        
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
            await page.click('button[type="submit"]')
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

async def scrape_card_images_v3(limit: int = 0):
    """
    Smart image scraping:
    1. First try to use CDN pattern (no requests needed!)
    2. Fall back to curl_cffi batch requests
    3. Last resort: Scrapling for stubborn pages
    """
    console.print(f"\n[bold]Phase 4: Smart Image URL Discovery[/bold]\n")
    
    # First, try to discover CDN pattern
    sample_cards = db.get_cards_needing_images(10)
    if not sample_cards:
        console.print("[green]No cards needing images![/green]")
        return
        
    cdn_engine = CDNPatternEngine()
    pattern = await cdn_engine.test_cdn_patterns(sample_cards)
    
    if pattern:
        # We found a pattern! Apply it to all cards
        await apply_cdn_pattern(pattern, cdn_engine, limit)
    else:
        # Fall back to scraping
        await scrape_with_curl_cffi(limit)


async def apply_cdn_pattern(pattern: str, cdn_engine: CDNPatternEngine, limit: int):
    """Apply discovered CDN pattern to generate image URLs without any requests!"""
    console.print(f"[green]Applying CDN pattern: {pattern}[/green]")
    
    batch_size = 1000
    total_processed = 0
    
    while True:
        cards = db.get_cards_needing_images(batch_size)
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
                
            db.update_card_image_url(product_id, image_url)
            total_processed += 1
            
        console.print(f"[green]Processed {total_processed} cards with CDN pattern[/green]")
        
    console.print(f"[bold green]Phase 4 complete: {total_processed} image URLs generated with ZERO requests![/bold green]")


async def scrape_with_curl_cffi(limit: int):
    """Fallback scraping using curl_cffi for fast parallel requests"""
    console.print("[yellow]Using curl_cffi for image URL scraping[/yellow]")
    
    session_mgr = SessionManager()
    batch_size = 10
    total_ok = 0
    total_no_image = 0
    total_error = 0
    
    while True:
        cards = db.get_cards_needing_images(100)
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
                    if isinstance(result, Exception):
                        db.mark_card_error(card['product_id'], str(result))
                        total_error += 1
                        session_mgr.rate_limiter.on_error()
                    elif isinstance(result, FetchResult) and result.image_url:
                        db.update_card_image_url(card['product_id'], result.image_url)
                        total_ok += 1
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

                # Rotate session if we're getting blocked
                if needs_rotation or session_mgr.rate_limiter.should_rotate_session():
                    session = await session_mgr.get_session(rotate=True)

                progress.advance(task, len(batch))
                await session_mgr.rate_limiter.wait()
                
    await session_mgr.close_all()
    console.print(f"Complete: {total_ok} found, {total_no_image} no image, {total_error} errors")


class FetchResult:
    """Result of fetch_image_url — distinguishes success, no-image, and errors."""
    __slots__ = ('image_url', 'error', 'no_image', 'status_code')
    def __init__(self, image_url=None, error=None, no_image=False, status_code=0):
        self.image_url = image_url
        self.error = error
        self.no_image = no_image
        self.status_code = status_code


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
            image_url = f"https://storage.googleapis.com/images.pricecharting.com/{match.group(1)}/1600.jpg"
            return FetchResult(image_url=image_url)

        # Page loaded OK but genuinely no image on it
        return FetchResult(no_image=True)

    except Exception as e:
        return FetchResult(error=f"{e} for {url}")


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
    """Run optimized pipeline with all improvements"""
    db.init_db()
    
    # Phase 1: Discover sets
    await discover_sets_v3(sport)
    
    # Phase 2: Download CSVs
    await download_csvs_v3(sport)
    
    # Phase 3: Parse CSVs (reuse existing function)
    from scraper import parse_csvs
    parse_csvs()
    
    # Phase 4: Smart image URL discovery
    await scrape_card_images_v3(limit)
    
    # Phase 5: Download images (reuse existing function but with curl_cffi)
    await download_images_v3(limit)
    
    console.print("[bold green]Pipeline complete! Check stats with --stats[/bold green]")


async def download_images_v3(limit: int = 0):
    """Download images using curl_cffi for maximum speed"""
    cards = db.get_cards_needing_download()
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