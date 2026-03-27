"""
SportsCardPro Scraper v2 - Playwright Automation

Strategy:
  Phase 1: Discover sets from category pages
  Phase 2: Download CSV for each set (one request per set = all cards)
  Phase 3: Parse CSVs into database
  Phase 4: Scrape card pages for image URLs (batched per set)
  Phase 5: Download images (no browser needed)
"""
import asyncio
import csv
import io
import os
import re
import random
import glob
import json
import subprocess
import shutil
from urllib.parse import urljoin, urlparse
from pathlib import Path

import aiohttp
import aiofiles
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.live import Live
from rich.table import Table
from rich.panel import Panel

import config
import database as db

console = Console()

# Try Camoufox first (best CF bypass), fall back to playwright-stealth
_USE_CAMOUFOX = False
try:
    from camoufox.async_api import AsyncCamoufox
    _USE_CAMOUFOX = True
except ImportError:
    AsyncCamoufox = None
    try:
        from playwright_stealth import stealth_async
    except ImportError:
        try:
            from playwright_stealth import Stealth
            _stealth_instance = Stealth()
            async def stealth_async(page):
                await _stealth_instance.apply_stealth_async(page)
        except ImportError:
            async def stealth_async(page):
                pass  # No stealth available


# ═══════════════════════════════════════════════════════════════════════════
# VPN rotation
# ═══════════════════════════════════════════════════════════════════════════

_vpn_available = shutil.which("nordvpn") is not None
_cf_block_count = 0  # Track consecutive CF blocks to trigger rotation

async def rotate_vpn():
    """Disconnect and reconnect NordVPN to get a fresh IP."""
    if not _vpn_available:
        return False
    try:
        console.print("  [cyan]Rotating VPN to fresh IP...[/cyan]")
        # Disconnect first
        proc = await asyncio.create_subprocess_exec(
            "nordvpn", "disconnect",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        await proc.wait()
        await asyncio.sleep(2)

        # Reconnect to a random server
        proc = await asyncio.create_subprocess_exec(
            "nordvpn", "connect",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode().strip()

        if proc.returncode == 0:
            # Extract server name from output if possible
            server = ""
            for line in output.splitlines():
                if "connected" in line.lower() or "server" in line.lower():
                    server = line.strip()
                    break
            console.print(f"  [green]VPN rotated: {server or 'connected'}[/green]")
            await asyncio.sleep(3)  # Let the connection stabilize
            return True
        else:
            console.print(f"  [yellow]VPN reconnect returned {proc.returncode}[/yellow]")
            return False
    except Exception as e:
        console.print(f"  [dim]VPN rotation failed: {e}[/dim]")
        return False


async def maybe_rotate_vpn(force: bool = False):
    """Rotate VPN after repeated CF blocks. Returns True if rotated."""
    global _cf_block_count
    if force:
        _cf_block_count = 0
        return await rotate_vpn()
    _cf_block_count += 1
    # Rotate after 3 consecutive CF blocks
    if _cf_block_count >= 3:
        _cf_block_count = 0
        return await rotate_vpn()
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Browser helpers
# ═══════════════════════════════════════════════════════════════════════════

async def create_browser(playwright) -> tuple[Browser, BrowserContext]:
    """Create a stealth browser. Uses Camoufox if available, else Playwright + stealth."""

    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    proxy_config = None
    if proxy_url:
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(proxy_url)
        proxy_config = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
        if parsed.username:
            proxy_config["username"] = parsed.username
        if parsed.password:
            proxy_config["password"] = parsed.password

    if _USE_CAMOUFOX:
        return await _create_camoufox_browser(playwright, proxy_config)
    else:
        return await _create_playwright_browser(playwright, proxy_config)


async def _create_camoufox_browser(playwright, proxy_config) -> tuple[Browser, BrowserContext]:
    """Launch Camoufox — anti-detect Firefox with C++ level fingerprint spoofing."""
    from camoufox import AsyncNewBrowser

    # On headless Linux (LXC), use virtual display; on Windows/headed use normal mode
    if config.HEADLESS:
        headless_mode = "virtual"  # Xvfb-backed, looks headed to CF
    else:
        headless_mode = False

    launch_opts = dict(
        headless=headless_mode,
        humanize=True,  # Human-like mouse movements
        os=("windows", "macos", "linux"),  # Random OS fingerprint
    )
    if proxy_config:
        launch_opts["proxy"] = proxy_config

    browser = await AsyncNewBrowser(playwright, **launch_opts)
    console.print("  [dim]Using Camoufox (anti-detect Firefox)[/dim]")

    context = browser.contexts[0] if browser.contexts else await browser.new_context()
    return browser, context


async def _create_playwright_browser(playwright, proxy_config) -> tuple[Browser, BrowserContext]:
    """Fallback: Playwright Chromium with stealth patches."""
    chrome_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--ignore-certificate-errors",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--window-size=1920,1080",
    ]

    launch_kwargs = dict(headless=config.HEADLESS, args=chrome_args)
    if proxy_config:
        launch_kwargs["proxy"] = proxy_config

    # Try system Chrome first, fall back to bundled Chromium
    try:
        browser = await playwright.chromium.launch(channel="chrome", **launch_kwargs)
        console.print("  [dim]Using system Chrome + stealth patches[/dim]")
    except Exception:
        import pathlib
        chromium_path = None
        pw_cache = pathlib.Path.home() / ".cache" / "ms-playwright"
        for d in sorted(pw_cache.glob("chromium-*"), reverse=True):
            candidate = d / "chrome-linux" / "chrome"
            if candidate.exists():
                chromium_path = str(candidate)
                break
        if chromium_path:
            launch_kwargs["executable_path"] = chromium_path
        browser = await playwright.chromium.launch(**launch_kwargs)
        console.print("  [dim]Using bundled Chromium + stealth (install camoufox for better CF bypass)[/dim]")

    ua = random.choice(config.USER_AGENTS)
    context = await browser.new_context(
        user_agent=ua,
        viewport={"width": 1920, "height": 1080},
        screen={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="America/Chicago",
        accept_downloads=True,
        ignore_https_errors=True,
        color_scheme="light",
        java_script_enabled=True,
        has_touch=False,
        is_mobile=False,
    )

    # JS-level anti-detection (backup — less effective than Camoufox C++ hooks)
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
        if (!window.chrome) {
            window.chrome = { runtime: {}, loadTimes: function(){}, csi: function(){} };
        }
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(parameter) {
            if (parameter === 37445) return 'Intel Inc.';
            if (parameter === 37446) return 'Intel Iris OpenGL Engine';
            return getParameter.apply(this, arguments);
        };
        delete navigator.__proto__.webdriver;
    """)

    return browser, context


async def new_stealth_page(context: BrowserContext) -> Page:
    page = await context.new_page()
    if not _USE_CAMOUFOX:
        await stealth_async(page)
    return page


async def _solve_cloudflare(page: Page):
    """Try to solve the Cloudflare Turnstile challenge automatically."""
    for attempt in range(5):
        try:
            # First check if challenge is already gone
            content = await page.content()
            if "challenge-platform" not in content and "Just a moment" not in content:
                console.print("  [green]Cloudflare challenge passed![/green]")
                return

            # Find the Turnstile iframe
            cf_frame = None
            for frame in page.frames:
                if "challenges.cloudflare.com" in (frame.url or ""):
                    cf_frame = frame
                    break

            if cf_frame:
                # Wait a moment for the widget to fully render
                await asyncio.sleep(random.uniform(1.5, 3.0))

                # Try multiple click strategies
                clicked = False

                # Strategy 1: input checkbox
                checkbox = cf_frame.locator("input[type='checkbox']")
                if await checkbox.count() > 0:
                    await checkbox.first.click(delay=random.uniform(50, 150))
                    clicked = True

                # Strategy 2: label element
                if not clicked:
                    label = cf_frame.locator("label, .cb-lb, .ctp-checkbox-label")
                    if await label.count() > 0:
                        await label.first.click(delay=random.uniform(50, 150))
                        clicked = True

                # Strategy 3: any clickable body element (Turnstile managed challenge)
                if not clicked:
                    body = cf_frame.locator("body")
                    if await body.count() > 0:
                        # Click center of the iframe body
                        bbox = await body.bounding_box()
                        if bbox:
                            await page.mouse.click(
                                bbox["x"] + bbox["width"] / 2,
                                bbox["y"] + bbox["height"] / 2,
                                delay=random.uniform(50, 150),
                            )
                            clicked = True

                if clicked:
                    console.print(f"  [cyan]Clicked Turnstile (attempt {attempt+1})[/cyan]")
            else:
                # No iframe found — might be a JS challenge, just wait
                console.print(f"  [dim]No Turnstile iframe found, waiting... (attempt {attempt+1})[/dim]")

            # Wait for the challenge to resolve
            await asyncio.sleep(config.CLOUDFLARE_WAIT + random.uniform(0, 3))

            # Re-check
            content = await page.content()
            if "challenge-platform" not in content and "Just a moment" not in content:
                console.print("  [green]Cloudflare challenge passed![/green]")
                return

        except Exception as e:
            console.print(f"  [dim]Turnstile attempt {attempt+1}: {e}[/dim]")
            await asyncio.sleep(random.uniform(2, 5))

    console.print("  [yellow]Could not auto-solve CF after 5 attempts[/yellow]")
    if not config.HEADLESS:
        console.print("  [yellow]Waiting 30s for manual solve in headed mode...[/yellow]")
        await asyncio.sleep(30)


async def safe_goto(page: Page, url: str, wait_for_cf: bool = False) -> bool:
    global _cf_block_count
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = await page.goto(url, timeout=config.PAGE_LOAD_TIMEOUT, wait_until="domcontentloaded")
            if wait_for_cf:
                await asyncio.sleep(config.CLOUDFLARE_WAIT)

            content = await page.content()
            if "challenge-platform" in content or "Just a moment" in content:
                console.print("  [yellow]Cloudflare challenge, attempting to solve...[/yellow]")
                await _solve_cloudflare(page)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                except:
                    pass
                # Check if challenge was solved
                content = await page.content()
                if "challenge-platform" in content or "Just a moment" in content:
                    # Still blocked — try VPN rotation
                    rotated = await maybe_rotate_vpn()
                    if rotated:
                        continue  # Retry with new IP

            if resp and resp.status in (200, 301, 302):
                _cf_block_count = 0  # Reset on success
                return True
            elif resp and resp.status == 403:
                console.print(f"  [red]403 on attempt {attempt+1}[/red]")
                rotated = await maybe_rotate_vpn()
                if not rotated:
                    await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
            else:
                _cf_block_count = 0
                return True
        except Exception as e:
            console.print(f"  [red]Error attempt {attempt+1}: {e}[/red]")
            await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
    return False


async def random_delay():
    await asyncio.sleep(random.uniform(config.REQUEST_DELAY_MIN, config.REQUEST_DELAY_MAX))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 0: Login
# ═══════════════════════════════════════════════════════════════════════════

async def login(page: Page) -> bool:
    """Log in to SportsCardsPro. Required for CSV downloads."""
    if not config.LOGIN_EMAIL or not config.LOGIN_PASSWORD:
        console.print("[red]Set SCP_EMAIL and SCP_PASSWORD env vars, or edit config.py[/red]")
        return False

    console.print(f"[cyan]Logging in as {config.LOGIN_EMAIL}...[/cyan]")
    ok = await safe_goto(page, config.LOGIN_URL, wait_for_cf=True)
    if not ok:
        return False

    try:
        # Fill email
        await page.fill('input[name="email"], input[type="email"], #email', config.LOGIN_EMAIL)
        await asyncio.sleep(0.5)
        # Fill password
        await page.fill('input[name="password"], input[type="password"], #password', config.LOGIN_PASSWORD)
        await asyncio.sleep(0.5)
        # Submit
        await page.click('button[type="submit"], input[type="submit"], button:has-text("Login"), button:has-text("Log In")')
        await page.wait_for_load_state("domcontentloaded", timeout=15000)
        await asyncio.sleep(3)

        # Verify login by checking for account link or logout URL
        content = await page.content()
        if "My Account" in content or "Logout" in content or "Log Out" in content or "logout" in content:
            console.print("[green]Login successful![/green]")
            return True
        else:
            console.print("[yellow]Login may have failed — proceeding anyway[/yellow]")
            return True  # Proceed anyway, might still work

    except Exception as e:
        console.print(f"[red]Login error: {e}[/red]")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Discover all sets
# ═══════════════════════════════════════════════════════════════════════════

async def discover_sets(page: Page, sport: str = None):
    """
    Visit each category page and extract all set links.
    Category pages list links like: /console/{sport}-cards-{set-slug}
    """
    categories = config.CATEGORY_URLS
    if sport:
        categories = {k: v for k, v in categories.items() if k == sport}

    console.print(f"\n[bold]Phase 1: Discovering sets from {len(categories)} categories[/bold]\n")

    first_load = True
    total_found = 0

    for sport_name, cat_url in categories.items():
        console.print(f"  Category: [cyan]{sport_name}[/cyan] → {cat_url}")

        ok = await safe_goto(page, cat_url, wait_for_cf=first_load)
        first_load = False
        if not ok:
            console.print(f"  [red]Failed to load {sport_name}[/red]")
            continue

        # Collect all set links, handling pagination
        all_sets = []
        seen = set()
        page_num = 0

        while True:
            page_num += 1
            sets_on_page = await page.evaluate("""() => {
                const links = document.querySelectorAll('a[href*="/console/"]');
                const results = [];
                for (const el of links) {
                    const href = el.getAttribute('href');
                    const text = el.textContent.trim();
                    if (href && href.startsWith('/console/') && text && text.length > 2) {
                        results.push({href, name: text});
                    }
                }
                return results;
            }""")

            new_count = 0
            for s in sets_on_page:
                slug = s["href"].replace("/console/", "")
                if slug not in seen:
                    seen.add(slug)
                    all_sets.append((slug, s["name"], sport_name, f"{config.BASE_URL}{s['href']}"))
                    new_count += 1

            # Try next page
            next_btn = await page.query_selector('a:has-text("Next"), a[rel="next"]')
            if next_btn and new_count > 0:
                try:
                    await next_btn.click()
                    await page.wait_for_load_state("domcontentloaded", timeout=10000)
                    await asyncio.sleep(2)
                except:
                    break
            else:
                break

        if all_sets:
            db.bulk_upsert_sets(all_sets)
            total_found += len(all_sets)
            console.print(f"    Found [green]{len(all_sets)}[/green] sets ({page_num} pages)")

        await random_delay()

    console.print(f"\n  Total sets discovered: [green]{total_found}[/green]")
    db.log_event("phase1_complete", f"Discovered {total_found} sets")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Download CSVs
# ═══════════════════════════════════════════════════════════════════════════

async def download_csvs(page: Page, sport: str = None):
    """
    Visit each set page and click "Download Price List" to get the CSV.
    Requires Retail+ subscription and being logged in.
    """
    pending = db.get_sets_needing_csv(sport)
    if not pending:
        console.print("[green]All set CSVs already downloaded.[/green]")
        return

    console.print(f"\n[bold]Phase 2: Downloading CSVs for {len(pending)} sets[/bold]\n")

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

            ok = await safe_goto(page, s["url"])
            if not ok:
                db.mark_set_csv_error(s["slug"])
                progress.advance(task)
                continue

            try:
                # Look for "Download Price List" link
                dl_link = await page.query_selector(
                    'a:has-text("Download Price List"), '
                    'a:has-text("Download"), '
                    'a[href*="download"], '
                    'a[href*="csv"]'
                )

                if dl_link:
                    csv_filename = f"{s['slug']}.csv"
                    csv_path = os.path.join(config.CSV_DIR, csv_filename)
                    downloaded = False

                    # Try up to 2 attempts — first may hit Cloudflare challenge
                    for dl_attempt in range(2):
                        try:
                            async with page.expect_download(timeout=30000) as dl_info:
                                await dl_link.click()
                            download = await dl_info.value
                            await download.save_as(csv_path)
                            downloaded = True
                            break
                        except Exception:
                            # Click may have navigated to a CF challenge page.
                            # Navigate back to the set page and retry.
                            if page.url != s["url"]:
                                await safe_goto(page, s["url"])
                                await asyncio.sleep(2)
                                dl_link = await page.query_selector(
                                    'a:has-text("Download Price List"), '
                                    'a:has-text("Download"), '
                                    'a[href*="download"], '
                                    'a[href*="csv"]'
                                )
                                if not dl_link:
                                    break

                    if downloaded:
                        db.mark_set_csv_downloaded(s["slug"], csv_path)
                        console.print(f"    [green]✓[/green] {s['slug']}")
                    else:
                        console.print(f"    [red]Download failed: {s['slug']}[/red]")
                        db.mark_set_csv_error(s["slug"])
                else:
                    # No download link found — might not have subscription
                    # or page structure different
                    console.print(f"    [yellow]No download link: {s['slug']}[/yellow]")
                    db.mark_set_csv_error(s["slug"])

            except Exception as e:
                err = str(e)[:100]
                console.print(f"    [red]Error: {err}[/red]")
                db.mark_set_csv_error(s["slug"])

            progress.advance(task)

            # CSV downloads are rate-limited to 1 per 10 minutes on the API,
            # but browser downloads may be less strict. Be cautious.
            await asyncio.sleep(config.CSV_DOWNLOAD_DELAY)

    db.log_event("phase2_complete", "CSV downloads finished")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Parse CSVs into database
# ═══════════════════════════════════════════════════════════════════════════

def parse_csvs():
    """
    Parse all downloaded CSVs and insert card records into the database.
    No browser needed — pure file processing.
    """
    pending = db.get_sets_needing_parse()

    # Also check for CSVs that exist on disk but aren't in DB yet
    existing_csvs = glob.glob(os.path.join(config.CSV_DIR, "*.csv"))
    console.print(f"\n[bold]Phase 3: Parsing {len(pending)} CSVs (+ {len(existing_csvs)} on disk)[/bold]\n")

    total_cards = 0

    for s in pending:
        csv_path = s.get("csv_path")
        if not csv_path or not os.path.exists(csv_path):
            console.print(f"  [yellow]CSV missing for {s['slug']}[/yellow]")
            continue

        try:
            cards = parse_single_csv(csv_path, s["slug"])
            if cards:
                db.bulk_insert_cards(cards)
                db.mark_set_csv_parsed(s["slug"], len(cards))
                total_cards += len(cards)
                console.print(f"  [green]✓[/green] {s['slug']}: {len(cards)} cards")
            else:
                console.print(f"  [yellow]Empty CSV: {s['slug']}[/yellow]")
                db.mark_set_csv_parsed(s["slug"], 0)
        except Exception as e:
            console.print(f"  [red]Parse error {s['slug']}: {e}[/red]")

    console.print(f"\n  Total cards parsed: [green]{total_cards}[/green]")
    db.log_event("phase3_complete", f"Parsed {total_cards} cards")


def parse_single_csv(csv_path: str, set_slug: str) -> list[dict]:
    """
    Parse a SportsCardsPro CSV file.
    
    Expected columns (based on API docs, CSV mirrors these):
    - id (product ID)
    - product-name / product_name
    - console-name / console_name (set name)
    - loose-price (ungraded, in pennies)
    - cib-price (mid grade, in pennies)
    - new-price (high grade, in pennies)
    - Plus potentially graded price columns
    """
    cards = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        # Try to detect delimiter
        sample = f.read(2048)
        f.seek(0)

        # SportsCardsPro CSVs are standard comma-delimited
        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
        except csv.Error:
            dialect = "excel"

        reader = csv.DictReader(f, dialect=dialect)

        for row in reader:
            # Normalize column names (handle hyphens vs underscores)
            norm = {k.strip().lower().replace("-", "_"): v.strip() for k, v in row.items() if k}

            product_id = norm.get("id", "")
            product_name = norm.get("product_name", norm.get("name", ""))
            console_name = norm.get("console_name", norm.get("console", ""))

            if not product_id or not product_name:
                continue

            # Build the card page URL slug
            # Pattern: /game/{set-slug}/{card-name-slug}
            card_name_slug = slugify(product_name)
            full_url = f"{config.BASE_URL}/game/{set_slug}/{card_name_slug}"

            # Parse prices (stored as dollars)
            loose = safe_price(norm.get("loose_price", norm.get("ungraded_price", "0")))
            cib = safe_price(norm.get("cib_price", norm.get("graded_price", "0")))
            new = safe_price(norm.get("new_price", norm.get("psa_10", "0")))

            cards.append({
                "product_id": product_id,
                "set_slug": set_slug,
                "product_name": product_name,
                "console_name": console_name,
                "card_url_slug": card_name_slug,
                "full_url": full_url,
                "loose_price": loose,
                "cib_price": cib,
                "new_price": new,
            })

    return cards


def slugify(name: str) -> str:
    """
    Convert a card name to URL slug matching SportsCardsPro's pattern.
    "Michael Jordan #57" → "michael-jordan-57"
    "Tom Brady [Red Refractor]" → "tom-brady-red-refractor"
    """
    s = name.lower()
    s = s.replace("#", "")           # Remove hash
    s = s.replace("[", "").replace("]", "")  # Remove brackets
    s = s.replace("'", "")           # Remove apostrophes
    s = s.replace(".", "")           # Remove periods
    s = s.replace(",", "")           # Remove commas
    s = s.replace("/", "-")          # Slash to dash
    s = re.sub(r'[^a-z0-9\s-]', '', s)  # Remove other special chars
    s = re.sub(r'\s+', '-', s.strip())   # Spaces to dashes
    s = re.sub(r'-+', '-', s)            # Collapse multiple dashes
    return s.strip("-")


def safe_price(val) -> float:
    """Parse a price string like '$5.00' into a float dollar value."""
    try:
        s = str(val).replace(",", "").replace("$", "").strip()
        return round(float(s), 2)
    except (ValueError, TypeError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Scrape card pages for image URLs
# ═══════════════════════════════════════════════════════════════════════════

# Image URL regex: matches the GCS-hosted card images in raw HTML
_IMAGE_RE = re.compile(
    r'https://storage\.googleapis\.com/images\.pricecharting\.com/([^/\s"\'<>]+)/(\d+)(?:\.jpg)?'
)


async def scrape_card_images(page: Page, limit: int = 0):
    """
    Fetch image URLs using in-browser fetch() — inherits Cloudflare clearance.

    Uses the browser's fetch() API to request card pages in parallel batches.
    This bypasses Cloudflare since the browser context already has CF cookies.
    Processes ~10-25 cards/sec depending on rate limiting.
    """
    FETCH_BATCH = 2           # parallel fetches per batch (lower = less suspicious)
    BATCH_DELAY_MIN = 4       # min seconds between batches
    BATCH_DELAY_MAX = 8       # max seconds between batches
    CF_REFRESH_EVERY = 25     # re-validate CF cookies every N batches
    DB_BATCH    = 500         # cards per DB fetch
    total_ok    = 0
    total_no_image = 0
    total_error = 0

    console.print(f"\n[bold]Phase 4: Scraping image URLs via in-browser fetch ({FETCH_BATCH}x parallel)[/bold]\n")

    # Navigate to a real SCP page to establish CF cookies + warm up
    console.print("  [cyan]Warming up browser session...[/cyan]")
    await safe_goto(page, f"{config.BASE_URL}/login", wait_for_cf=True)
    await asyncio.sleep(random.uniform(3, 6))  # Human-like pause after page load

    while True:
        cards = db.get_cards_needing_images(DB_BATCH)
        if not cards:
            console.print("[green]No more cards needing image URLs.[/green]")
            break

        if limit > 0:
            remaining = limit - total_ok
            if remaining <= 0:
                break
            cards = cards[:remaining]

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            ptask = progress.add_task("Images", total=len(cards))

            batch_count = 0
            for i in range(0, len(cards), FETCH_BATCH):
                batch = cards[i:i + FETCH_BATCH]
                urls = [c["full_url"] for c in batch]
                pids = [str(c["product_id"]) for c in batch]
                batch_count += 1

                # Periodically refresh CF cookies by navigating to a real page
                if batch_count % CF_REFRESH_EVERY == 0:
                    progress.update(ptask, description="Refreshing session...")
                    await safe_goto(page, f"{config.BASE_URL}/category/football-cards", wait_for_cf=True)
                    await asyncio.sleep(random.uniform(3, 6))

                progress.update(ptask, description=f"Img: batch {batch_count}")

                try:
                    results = await page.evaluate("""async (urls) => {
                        const re = /https:\\/\\/storage\\.googleapis\\.com\\/images\\.pricecharting\\.com\\/([^\\/\\s"'<>]+)\\/\\d+/;
                        return Promise.all(urls.map(async (url) => {
                            try {
                                const resp = await fetch(url);
                                if (resp.status === 403) return {hash: null, reason: 'cf_blocked'};
                                if (resp.status !== 200) return {hash: null, reason: 'http_' + resp.status};
                                const html = await resp.text();
                                // Detect CF challenge in response body
                                if (html.includes('challenge-platform') || html.includes('Just a moment')) {
                                    return {hash: null, reason: 'cf_challenge'};
                                }
                                const match = html.match(re);
                                if (match) return {hash: match[1], reason: null};
                                if (html.includes('pricecharting.com') || html.includes('sportscardspro.com')) {
                                    return {hash: null, reason: 'no_image_on_page'};
                                }
                                return {hash: null, reason: 'blocked_or_empty'};
                            } catch(e) { return {hash: null, reason: 'fetch_error'}; }
                        }));
                    }""", urls)
                except Exception:
                    results = [{"hash": None, "reason": "evaluate_error"}] * len(batch)

                # If we got CF blocked, pause and refresh the session
                cf_hits = sum(1 for r in results if isinstance(r, dict) and r.get("reason") in ("cf_blocked", "cf_challenge"))
                if cf_hits > 0:
                    console.print(f"  [yellow]Cloudflare blocked {cf_hits}/{len(batch)} — refreshing session...[/yellow]")
                    # Try rotating VPN if we keep getting blocked
                    rotated = await maybe_rotate_vpn()
                    if rotated:
                        await asyncio.sleep(random.uniform(5, 10))
                    await safe_goto(page, f"{config.BASE_URL}/category/football-cards", wait_for_cf=True)
                    await asyncio.sleep(random.uniform(10, 20))

                for card, result in zip(batch, results):
                    pid = card["product_id"]
                    img_hash = result.get("hash") if isinstance(result, dict) else result
                    reason = result.get("reason", "") if isinstance(result, dict) else ""

                    if img_hash:
                        image_url = f"https://storage.googleapis.com/images.pricecharting.com/{img_hash}/1600.jpg"
                        db.update_card_image_url(pid, image_url)
                        total_ok += 1
                    elif reason == "no_image_on_page":
                        # Page loaded fine but card genuinely has no image
                        db.mark_card_no_image(pid)
                        total_no_image += 1
                    else:
                        # Rate-limited or blocked — mark as error so it can be retried
                        db.mark_card_error(pid, reason or "unknown")
                        total_error += 1

                progress.advance(ptask, len(batch))
                await asyncio.sleep(random.uniform(BATCH_DELAY_MIN, BATCH_DELAY_MAX))

        console.print(f"  Batch done -- OK: [green]{total_ok}[/green], No image: [yellow]{total_no_image}[/yellow], Errors: [red]{total_error}[/red]")

        if limit > 0 and total_ok >= limit:
            break

    console.print(f"\n  Fast pass: [green]{total_ok}[/green] found, [yellow]{total_no_image}[/yellow] no image, [red]{total_error}[/red] errors")

    # Automatic fallback: retry errored cards with direct browser page visits
    if total_error > 0:
        browser_limit = (limit - total_ok) if limit > 0 else 0
        if limit == 0 or browser_limit > 0:
            console.print(f"\n[bold]Phase 4b: Retrying {total_error} errors via direct browser visit[/bold]\n")
            browser_ok, browser_no_img = await _retry_errors_with_browser(page, limit=browser_limit)
            total_ok += browser_ok
            total_no_image += browser_no_img
            remaining_errors = total_error - browser_ok - browser_no_img
        else:
            remaining_errors = total_error
    else:
        remaining_errors = 0

    console.print(f"\n  Final: [green]{total_ok}[/green] image URLs, [yellow]{total_no_image}[/yellow] no image, [red]{remaining_errors}[/red] remaining errors")
    db.log_event("phase4_complete", f"Found {total_ok} image URLs, {total_no_image} no image, {remaining_errors} remaining errors")


async def _retry_errors_with_browser(page: Page, limit: int = 0) -> tuple[int, int]:
    """
    Retry errored cards by visiting each card's detail page in the browser.
    Slower but more reliable — bypasses rate limits since it's a full page load.
    Returns (ok_count, no_image_count).
    """
    total_ok = 0
    total_no_image = 0

    while True:
        cards = db.get_errored_cards(100)
        if not cards:
            break

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Browser retry", total=len(cards))

            for card in cards:
                title = (card["product_name"] or "")[:35]
                progress.update(task, description=f"Retry: {title}")

                url = card.get("full_url")
                pid = card["product_id"]

                if not url:
                    db.mark_card_no_image(pid)
                    total_no_image += 1
                    progress.advance(task)
                    continue

                ok = await safe_goto(page, url)
                if not ok:
                    progress.advance(task)
                    continue  # Leave as error, try again next run

                image_url = await extract_card_image_browser(page)

                if image_url:
                    if image_url.startswith("/"):
                        image_url = f"{config.BASE_URL}{image_url}"
                    db.update_card_image_url(pid, image_url)
                    total_ok += 1
                else:
                    db.mark_card_no_image(pid)
                    total_no_image += 1

                progress.advance(task)

                if limit > 0 and total_ok >= limit:
                    console.print(f"  Browser retry: [green]{total_ok}[/green] found, [yellow]{total_no_image}[/yellow] no image")
                    return total_ok, total_no_image

                await random_delay()

    console.print(f"  Browser retry: [green]{total_ok}[/green] found, [yellow]{total_no_image}[/yellow] no image")
    return total_ok, total_no_image


def _extract_image_url_from_html(html: str) -> str | None:
    """
    Extract the card image URL from raw HTML (offers page or detail page).

    The image is an <img> tag with src pointing to:
      https://storage.googleapis.com/images.pricecharting.com/{hash}/{size}.jpg

    Available sizes: 60, 120, 240, 1600.
    We store the 1600px version for best embedding quality.
    """
    match = _IMAGE_RE.search(html)
    if match:
        img_hash = match.group(1)
        return f"https://storage.googleapis.com/images.pricecharting.com/{img_hash}/1600.jpg"
    return None


async def scrape_card_images_browser(page: Page, limit: int = 0):
    """
    LEGACY: Browser-based image scraping — kept as fallback.
    Visit each card's detail page to find the image URL.
    Use scrape_card_images() instead for ~200x faster throughput.
    """
    batch_size = 100
    total = 0

    console.print(f"\n[bold]Phase 4 (legacy browser): Scraping card pages for image URLs[/bold]\n")

    while True:
        cards = db.get_cards_needing_images(batch_size)
        if not cards:
            console.print("[green]No more cards needing image URLs.[/green]")
            break

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Images", total=len(cards))

            for card in cards:
                title = (card["product_name"] or "")[:35]
                progress.update(task, description=f"Img: {title}")

                url = card.get("full_url")
                if not url:
                    db.mark_card_no_image(card["product_id"])
                    progress.advance(task)
                    continue

                ok = await safe_goto(page, url)
                if not ok:
                    db.mark_card_error(card["product_id"], "Page load failed")
                    progress.advance(task)
                    continue

                image_url = await extract_card_image_browser(page)

                if image_url:
                    if image_url.startswith("/"):
                        image_url = f"{config.BASE_URL}{image_url}"
                    db.update_card_image_url(card["product_id"], image_url)
                else:
                    db.mark_card_no_image(card["product_id"])

                progress.advance(task)
                total += 1

                if limit > 0 and total >= limit:
                    return

                await random_delay()

    console.print(f"\n  Scraped image URLs for [green]{total}[/green] cards")
    db.log_event("phase4_browser_complete", f"Scraped {total} card images")


async def extract_card_image_browser(page: Page) -> str | None:
    """Extract the main card image URL from a card detail page (browser-based)."""
    return await page.evaluate("""() => {
        const gcpImg = document.querySelector('img[src*="images.pricecharting.com"]');
        if (gcpImg) {
            const src = gcpImg.getAttribute('src') || '';
            if (src) {
                // Upgrade to 1600px for best embedding quality
                return src.replace(/\\/\\d+(\\.jpg)?$/, '/1600.jpg');
            }
        }

        const selectors = [
            '#product_photo img',
            '.product-image img',
            '.game-image img',
            '#main_photo',
            'img.product-image',
            'img.js-show-dialog',
        ];

        for (const sel of selectors) {
            const el = document.querySelector(sel);
            if (el) {
                const src = el.getAttribute('src') || el.getAttribute('data-src');
                if (src && !src.includes('logo') && !src.includes('app-store')
                    && !src.includes('play-store')) return src;
            }
        }

        const ogImg = document.querySelector('meta[property="og:image"]');
        if (ogImg) {
            const content = ogImg.getAttribute('content');
            if (content && !content.includes('logo')) return content;
        }

        const imgs = [...document.querySelectorAll('img')];
        const excludePatterns = [
            'logo', 'icon', 'sprite', 'ad-', 'avatar', 'flag',
            'app-store', 'play-store', 'centering-calculator',
            'apple-app', 'android-play', 'psa'
        ];
        const candidates = imgs.filter(img => {
            const src = (img.src || img.getAttribute('data-src') || '');
            const w = img.naturalWidth || img.width || 0;
            return src && w > 80 &&
                !excludePatterns.some(p => src.toLowerCase().includes(p));
        });

        candidates.sort((a, b) => {
            const wa = a.naturalWidth || a.width || 0;
            const wb = b.naturalWidth || b.width || 0;
            return wb - wa;
        });

        return candidates.length > 0
            ? (candidates[0].src || candidates[0].getAttribute('data-src'))
            : null;
    }""")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Download images
# ═══════════════════════════════════════════════════════════════════════════

async def download_images(limit: int = 0):
    """Download all card images. No browser needed — pure HTTP."""
    console.print(f"\n[bold]Phase 5: Downloading card images[/bold]\n")

    # Get total count for progress bar
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cards WHERE status='image_found' AND image_url IS NOT NULL")
    all_pending = cur.fetchone()[0]
    cur.close()
    conn.close()
    total_to_download = min(all_pending, limit) if limit > 0 else all_pending

    if total_to_download == 0:
        console.print("  [yellow]No images to download.[/yellow]")
        return

    console.print(f"  Found [cyan]{total_to_download:,}[/cyan] images to download "
                  f"([dim]{config.IMAGE_CONCURRENT_DOWNLOADS} concurrent[/dim])\n")

    downloaded = 0
    skipped = 0  # already cached locally
    failed = 0
    sem = asyncio.Semaphore(config.IMAGE_CONCURRENT_DOWNLOADS)

    # Shared state for live display
    progress_state = {"current_file": "", "current_set": ""}

    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=config.IMAGE_DOWNLOAD_TIMEOUT),
        headers={"User-Agent": random.choice(config.USER_AGENTS)},
        connector=connector,
        trust_env=True,
    ) as session:

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("[dim]|[/dim]"),
            TimeRemainingColumn(),
            console=console,
        )
        task = progress.add_task("Downloading", total=total_to_download)

        def make_status_table():
            tbl = Table.grid(padding=(0, 2))
            tbl.add_row(
                f"  [green]✓ {downloaded:,}[/green] downloaded",
                f"[yellow]⊘ {skipped:,}[/yellow] cached",
                f"[red]✗ {failed:,}[/red] failed",
            )
            tbl.add_row(
                f"  [dim]Set: {progress_state['current_set'][:50]}[/dim]",
                f"[dim]File: {progress_state['current_file'][:50]}[/dim]",
                "",
            )
            return tbl

        with Live(progress, console=console, refresh_per_second=8) as live:
            def refresh_display():
                from rich.console import Group
                live.update(Group(progress, make_status_table()))

            while True:
                cards = db.get_cards_needing_download(200)
                if not cards:
                    break

                tasks = []
                processed = downloaded + skipped + failed
                for card in cards:
                    if limit > 0 and processed + len(tasks) >= limit:
                        break
                    tasks.append(download_one_image(
                        session, sem, card, progress, task,
                        progress_state, refresh_display
                    ))

                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if r == "downloaded":
                        downloaded += 1
                    elif r == "cached":
                        skipped += 1
                    else:
                        failed += 1
                refresh_display()

                if limit > 0 and (downloaded + skipped + failed) >= limit:
                    break

    # Final summary
    console.print()
    summary = Table.grid(padding=(0, 2))
    summary.add_row("[bold]Phase 5 Complete[/bold]")
    summary.add_row(
        f"  [green]✓ {downloaded:,}[/green] downloaded",
        f"[yellow]⊘ {skipped:,}[/yellow] already cached",
        f"[red]✗ {failed:,}[/red] failed",
    )
    total = downloaded + skipped + failed
    summary.add_row(f"  [bold]{total:,}[/bold] total processed")
    console.print(Panel(summary, border_style="green" if failed == 0 else "yellow"))
    db.log_event("phase5_complete", f"Downloaded {downloaded}, cached {skipped}, failed {failed}")


async def download_one_image(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    card: dict,
    progress: Progress,
    task,
    state: dict,
    refresh_fn,
) -> str:
    """Returns 'downloaded', 'cached', or 'failed'."""
    async with sem:
        try:
            set_dir = os.path.join(config.IMAGE_DIR, card["set_slug"])
            os.makedirs(set_dir, exist_ok=True)

            safe_name = re.sub(r'[^\w\-.]', '_', card.get("card_url_slug", str(card["product_id"])))
            ext = _get_ext(card["image_url"])
            filepath = os.path.join(set_dir, f"{safe_name}{ext}")

            state["current_set"] = card.get("set_slug", "")
            state["current_file"] = f"{safe_name}{ext}"
            refresh_fn()

            if os.path.exists(filepath):
                db.mark_card_downloaded(card["product_id"], filepath)
                progress.advance(task)
                return "cached"

            async with session.get(card["image_url"]) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(data)
                    db.mark_card_downloaded(card["product_id"], filepath)
                    progress.advance(task)
                    return "downloaded"
                else:
                    db.mark_card_error(card["product_id"], f"HTTP {resp.status}")
                    progress.advance(task)
                    return "failed"
        except Exception as e:
            db.mark_card_error(card["product_id"], str(e)[:200])
            progress.advance(task)
            return "failed"


def _get_ext(url: str) -> str:
    _, ext = os.path.splitext(urlparse(url).path)
    return ext.lower() if ext.lower() in (".jpg", ".jpeg", ".png", ".webp", ".gif") else ".jpg"


# ═══════════════════════════════════════════════════════════════════════════
# Full pipeline
# ═══════════════════════════════════════════════════════════════════════════

async def run_full_pipeline(sport: str = None, limit: int = 0):
    db.init_db()
    db.log_event("pipeline_start", f"sport={sport}, limit={limit}")

    async with async_playwright() as pw:
        browser, context = await create_browser(pw)
        page = await new_stealth_page(context)

        try:
            # Login for CSV access
            logged_in = await login(page)
            if not logged_in:
                console.print("[yellow]Continuing without login — CSV downloads may fail[/yellow]")

            # Phase 1: Discover sets
            await discover_sets(page, sport)

            # Phase 2: Download CSVs
            await download_csvs(page, sport)

            # Phase 3: Parse CSVs (no browser)
            parse_csvs()

        finally:
            await context.close()
            await browser.close()

    # Phase 4: Scrape image URLs via lightweight HTTP (no browser needed!)
    # Uses /offers?product={id} endpoint: ~38 KB vs ~289 KB, 20x concurrent
    await scrape_card_images(limit=limit)

    # Phase 5: Download images (no browser)
    await download_images(limit)

    stats = db.get_stats()
    console.print("\n[bold]═══ Final Stats ═══[/bold]")
    for k, v in stats.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")
    db.log_event("pipeline_complete", json.dumps(stats))
