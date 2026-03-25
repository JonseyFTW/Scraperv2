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
from urllib.parse import urljoin, urlparse
from pathlib import Path

import aiohttp
import aiofiles
from playwright.async_api import async_playwright, Page, Browser, BrowserContext
try:
    from playwright_stealth import stealth_async  # v1.x
except ImportError:
    from playwright_stealth import Stealth  # v2.x
    _stealth_instance = Stealth()
    async def stealth_async(page):
        await _stealth_instance.apply_stealth_async(page)
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import config
import database as db

console = Console()


# ═══════════════════════════════════════════════════════════════════════════
# Browser helpers
# ═══════════════════════════════════════════════════════════════════════════

async def create_browser(playwright) -> tuple[Browser, BrowserContext]:
    # Find the installed chromium executable (version may differ from pip package)
    chromium_path = None
    import pathlib
    pw_cache = pathlib.Path.home() / ".cache" / "ms-playwright"
    for d in sorted(pw_cache.glob("chromium-*"), reverse=True):
        candidate = d / "chrome-linux" / "chrome"
        if candidate.exists():
            chromium_path = str(candidate)
            break

    launch_kwargs = dict(
        headless=config.HEADLESS,
        args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--ignore-certificate-errors"],
    )
    if chromium_path:
        launch_kwargs["executable_path"] = chromium_path

    # Configure proxy if environment proxy is set
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    if proxy_url:
        # Parse proxy URL to extract username/password if present
        from urllib.parse import urlparse as _urlparse
        parsed = _urlparse(proxy_url)
        proxy_config = {"server": f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"}
        if parsed.username:
            proxy_config["username"] = parsed.username
        if parsed.password:
            proxy_config["password"] = parsed.password
        launch_kwargs["proxy"] = proxy_config

    browser = await playwright.chromium.launch(**launch_kwargs)
    context = await browser.new_context(
        user_agent=random.choice(config.USER_AGENTS),
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="America/Chicago",
        accept_downloads=True,  # Required for CSV downloads
        ignore_https_errors=True,
    )
    return browser, context


async def new_stealth_page(context: BrowserContext) -> Page:
    page = await context.new_page()
    await stealth_async(page)
    return page


async def safe_goto(page: Page, url: str, wait_for_cf: bool = False) -> bool:
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = await page.goto(url, timeout=config.PAGE_LOAD_TIMEOUT, wait_until="domcontentloaded")
            if wait_for_cf:
                await asyncio.sleep(config.CLOUDFLARE_WAIT)

            content = await page.content()
            if "challenge-platform" in content or "Just a moment" in content:
                console.print("  [yellow]Cloudflare challenge, waiting...[/yellow]")
                await asyncio.sleep(config.CLOUDFLARE_WAIT)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                except:
                    pass

            if resp and resp.status in (200, 301, 302):
                return True
            elif resp and resp.status == 403:
                console.print(f"  [red]403 on attempt {attempt+1}[/red]")
                await asyncio.sleep(config.RETRY_DELAY * (attempt + 1))
            else:
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

async def scrape_card_images(page: Page, limit: int = 0):
    """
    Visit each card's detail page to find the image URL.
    Groups by set so we maintain session continuity.
    """
    batch_size = 100
    total = 0

    console.print(f"\n[bold]Phase 4: Scraping card pages for image URLs[/bold]\n")

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

                image_url = await extract_card_image(page)

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
    db.log_event("phase4_complete", f"Scraped {total} card images")


async def extract_card_image(page: Page) -> str | None:
    """Extract the main card image URL from a card detail page."""
    return await page.evaluate("""() => {
        // The main card photo on SportsCardsPro is hosted on
        // storage.googleapis.com/images.pricecharting.com and typically has
        // class 'js-show-dialog' or an alt text containing the card name.
        const gcpImg = document.querySelector('img[src*="images.pricecharting.com"]');
        if (gcpImg) {
            const src = gcpImg.getAttribute('src') || '';
            if (src) return src;
        }

        // Primary selectors for the main card photo
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

        // Try og:image meta tag
        const ogImg = document.querySelector('meta[property="og:image"]');
        if (ogImg) {
            const content = ogImg.getAttribute('content');
            if (content && !content.includes('logo')) return content;
        }

        // Fallback: find the largest non-UI image, excluding known non-card images
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

        // Sort by size descending, take the first
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

    total = 0
    sem = asyncio.Semaphore(config.IMAGE_CONCURRENT_DOWNLOADS)

    # Use proxy if set in environment
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    connector = aiohttp.TCPConnector(ssl=False)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=config.IMAGE_DOWNLOAD_TIMEOUT),
        headers={"User-Agent": random.choice(config.USER_AGENTS)},
        connector=connector,
        trust_env=True,
    ) as session:

        while True:
            cards = db.get_cards_needing_download(200)
            if not cards:
                break

            tasks = []
            for card in cards:
                if limit > 0 and total + len(tasks) >= limit:
                    break
                tasks.append(download_one_image(session, sem, card))

            results = await asyncio.gather(*tasks, return_exceptions=True)
            total += sum(1 for r in results if r is True)

            if limit > 0 and total >= limit:
                break

    console.print(f"\n  Downloaded [green]{total}[/green] images")
    db.log_event("phase5_complete", f"Downloaded {total} images")


async def download_one_image(session: aiohttp.ClientSession, sem: asyncio.Semaphore, card: dict) -> bool:
    async with sem:
        try:
            set_dir = os.path.join(config.IMAGE_DIR, card["set_slug"])
            os.makedirs(set_dir, exist_ok=True)

            safe_name = re.sub(r'[^\w\-.]', '_', card.get("card_url_slug", str(card["product_id"])))
            ext = _get_ext(card["image_url"])
            filepath = os.path.join(set_dir, f"{safe_name}{ext}")

            if os.path.exists(filepath):
                db.mark_card_downloaded(card["product_id"], filepath)
                return True

            async with session.get(card["image_url"]) as resp:
                if resp.status == 200:
                    data = await resp.read()
                    async with aiofiles.open(filepath, "wb") as f:
                        await f.write(data)
                    db.mark_card_downloaded(card["product_id"], filepath)
                    return True
                else:
                    db.mark_card_error(card["product_id"], f"HTTP {resp.status}")
                    return False
        except Exception as e:
            db.mark_card_error(card["product_id"], str(e)[:200])
            return False


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

            # Phase 4: Scrape card pages for images
            await scrape_card_images(page, limit)

        finally:
            await context.close()
            await browser.close()

    # Phase 5: Download images (no browser)
    await download_images(limit)

    stats = db.get_stats()
    console.print("\n[bold]═══ Final Stats ═══[/bold]")
    for k, v in stats.items():
        console.print(f"  {k}: [cyan]{v}[/cyan]")
    db.log_event("pipeline_complete", json.dumps(stats))
