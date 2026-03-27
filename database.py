"""
SportsCardPro Scraper v2 - Database Layer (PostgreSQL)
"""
import os
import socket

import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from config import DATABASE_URL

# Worker identity — uses hostname so each LXC container is distinct
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())


def get_connection(retries=3):
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
            conn.autocommit = False
            return conn
        except psycopg2.OperationalError:
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                raise


def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sets (
            slug        TEXT PRIMARY KEY,
            name        TEXT,
            sport       TEXT,
            url         TEXT NOT NULL,
            csv_status  TEXT DEFAULT 'pending',
            img_status  TEXT DEFAULT 'pending',
            csv_path    TEXT,
            card_count  INTEGER DEFAULT 0,
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS cards (
            id              SERIAL PRIMARY KEY,
            product_id      TEXT,
            set_slug        TEXT NOT NULL,
            product_name    TEXT,
            console_name    TEXT,
            card_url_slug   TEXT,
            full_url        TEXT,
            image_url       TEXT,
            image_path      TEXT,
            loose_price     REAL,
            cib_price       REAL,
            new_price       REAL,
            graded_price    TEXT,
            status          TEXT DEFAULT 'pending',
            error_msg       TEXT,
            FOREIGN KEY (set_slug) REFERENCES sets(slug)
        );

        CREATE TABLE IF NOT EXISTS scrape_log (
            id          SERIAL PRIMARY KEY,
            timestamp   TEXT NOT NULL,
            event       TEXT NOT NULL,
            details     TEXT
        );
    """)
    # Add worker_id column if missing (tracks which container processed each card)
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE cards ADD COLUMN worker_id TEXT;
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$;
    """)

    # Create indexes (IF NOT EXISTS supported in PG 9.5+)
    cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_cards_pid ON cards(product_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_set ON cards(set_slug)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_status ON cards(status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_cards_worker ON cards(worker_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sets_csv ON sets(csv_status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sets_img ON sets(img_status)")
    conn.commit()
    cur.close()
    conn.close()


def log_event(event: str, details: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scrape_log (timestamp, event, details) VALUES (%s, %s, %s)",
        (datetime.now(timezone.utc).isoformat(), event, details)
    )
    conn.commit()
    cur.close()
    conn.close()


# ── Set operations ────────────────────────────────────────────────────────

def upsert_set(slug: str, name: str, sport: str, url: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO sets (slug, name, sport, url)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(slug) DO UPDATE SET name=EXCLUDED.name, sport=EXCLUDED.sport, url=EXCLUDED.url
    """, (slug, name, sport, url))
    conn.commit()
    cur.close()
    conn.close()


def bulk_upsert_sets(sets_data: list[tuple]):
    """Each tuple: (slug, name, sport, url)"""
    conn = get_connection()
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO sets (slug, name, sport, url)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(slug) DO UPDATE SET name=EXCLUDED.name
    """, sets_data)
    conn.commit()
    cur.close()
    conn.close()


def get_sets_needing_csv(sport: str = None) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if sport:
        cur.execute(
            "SELECT * FROM sets WHERE csv_status='pending' AND sport=%s ORDER BY slug", (sport,)
        )
    else:
        cur.execute(
            "SELECT * FROM sets WHERE csv_status='pending' ORDER BY slug"
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_sets_needing_parse() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM sets WHERE csv_status='downloaded' ORDER BY slug")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def mark_set_csv_downloaded(slug: str, csv_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sets SET csv_status='downloaded', csv_path=%s, updated_at=%s WHERE slug=%s
    """, (csv_path, datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    cur.close()
    conn.close()


def mark_set_csv_parsed(slug: str, card_count: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sets SET csv_status='parsed', card_count=%s, updated_at=%s WHERE slug=%s
    """, (card_count, datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    cur.close()
    conn.close()


def mark_set_csv_error(slug: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE sets SET csv_status='error' WHERE slug=%s", (slug,))
    conn.commit()
    cur.close()
    conn.close()


def mark_set_images_scraped(slug: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sets SET img_status='scraped', updated_at=%s WHERE slug=%s
    """, (datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    cur.close()
    conn.close()


# ── Card operations ───────────────────────────────────────────────────────

def bulk_insert_cards(cards: list[dict]):
    """Insert cards from CSV parse. Each dict has CSV column data."""
    conn = get_connection()
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO cards
            (product_id, set_slug, product_name, console_name,
             card_url_slug, full_url, loose_price, cib_price, new_price)
        VALUES
            (%(product_id)s, %(set_slug)s, %(product_name)s, %(console_name)s,
             %(card_url_slug)s, %(full_url)s, %(loose_price)s, %(cib_price)s, %(new_price)s)
        ON CONFLICT (product_id) DO NOTHING
    """, cards)
    conn.commit()
    cur.close()
    conn.close()


def get_cards_needing_images(limit: int = 500) -> list[dict]:
    """Atomically claim a batch of pending cards for image scraping.
    Uses FOR UPDATE SKIP LOCKED so multiple workers never get the same rows."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        UPDATE cards SET status='processing', worker_id=%s
        WHERE id IN (
            SELECT id FROM cards
            WHERE status='pending'
            ORDER BY set_slug DESC, id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *
    """, (WORKER_ID, limit))
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def count_pending_images() -> int:
    """Count how many cards still need image scraping (pending status)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM cards WHERE status = 'pending'")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def get_errored_cards(limit: int = 500) -> list[dict]:
    """Claim errored cards for retry. Uses SKIP LOCKED for concurrency."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        UPDATE cards SET status='processing', worker_id=%s
        WHERE id IN (
            SELECT id FROM cards
            WHERE status='error'
            ORDER BY set_slug DESC, id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *
    """, (WORKER_ID, limit))
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def get_cards_needing_download(limit: int = 500) -> list[dict]:
    """Claim cards for image download. Uses SKIP LOCKED for concurrency."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        UPDATE cards SET status='downloading', worker_id=%s
        WHERE id IN (
            SELECT id FROM cards
            WHERE status='image_found' AND image_url IS NOT NULL
            ORDER BY id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *
    """, (WORKER_ID, limit))
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def update_card_image_url(product_id: str, image_url: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards SET image_url=%s, status='image_found' WHERE product_id=%s
    """, (image_url, product_id))
    conn.commit()
    cur.close()
    conn.close()


def mark_card_downloaded(product_id: str, image_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards SET image_path=%s, status='downloaded' WHERE product_id=%s
    """, (image_path, product_id))
    conn.commit()
    cur.close()
    conn.close()


def mark_card_no_image(product_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE cards SET status='no_image' WHERE product_id=%s", (product_id,))
    conn.commit()
    cur.close()
    conn.close()


def mark_card_error(product_id: str, msg: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE cards SET status='error', error_msg=%s WHERE product_id=%s", (msg, product_id))
    conn.commit()
    cur.close()
    conn.close()


# ── Multi-source retry ────────────────────────────────────────────────────

def get_cards_for_multi_source(limit: int = 500) -> list[dict]:
    """Claim cards with no_image or error status for multi-source retry.
    These are cards where SportsCardPro didn't have an image or failed."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        UPDATE cards SET status='processing', worker_id=%s
        WHERE id IN (
            SELECT id FROM cards
            WHERE status IN ('no_image', 'error')
            ORDER BY set_slug DESC, id
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        )
        RETURNING *
    """, (WORKER_ID, limit))
    rows = cur.fetchall()
    conn.commit()
    cur.close()
    conn.close()
    return [dict(r) for r in rows]


def count_multi_source_candidates() -> int:
    """Count cards eligible for multi-source retry (no_image + error)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM cards WHERE status IN ('no_image', 'error')")
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def reset_multi_source_cards() -> int:
    """Reset all cards found via multi-source back to 'no_image' so Phase 4 can retry.
    Also clears image_url and image_path for these cards."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards SET status='no_image', image_url=NULL, image_path=NULL, error_msg=NULL
        WHERE error_msg LIKE 'found_via:%'
    """)
    count = cur.rowcount
    # Also reset any that were downloaded from multi-source
    cur.execute("""
        UPDATE cards SET status='no_image', image_url=NULL, image_path=NULL, error_msg=NULL
        WHERE status IN ('image_found', 'downloaded') AND error_msg LIKE 'found_via:%'
    """)
    conn.commit()
    cur.close()
    conn.close()
    return count


def update_card_image_source(product_id: str, image_url: str, source: str):
    """Update a card with an image URL found via multi-source search."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards SET image_url=%s, status='image_found',
        error_msg=%s WHERE product_id=%s
    """, (image_url, f"found_via:{source}", product_id))
    conn.commit()
    cur.close()
    conn.close()


# ── Stats ─────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    s = {}

    cur.execute("SELECT COUNT(*) AS c FROM sets")
    s["total_sets"] = cur.fetchone()["c"]
    for st in ("pending", "downloaded", "parsed", "error"):
        cur.execute("SELECT COUNT(*) AS c FROM sets WHERE csv_status=%s", (st,))
        s[f"sets_csv_{st}"] = cur.fetchone()["c"]

    cur.execute("SELECT COUNT(*) AS c FROM cards")
    s["total_cards"] = cur.fetchone()["c"]
    for st in ("pending", "processing", "image_found", "downloading", "downloaded", "no_image", "error"):
        cur.execute("SELECT COUNT(*) AS c FROM cards WHERE status=%s", (st,))
        s[f"cards_{st}"] = cur.fetchone()["c"]

    # Per sport
    cur.execute("SELECT DISTINCT sport FROM sets ORDER BY sport")
    sports = cur.fetchall()
    for sp in sports:
        cur.execute(
            "SELECT COUNT(*) AS c FROM cards WHERE set_slug IN (SELECT slug FROM sets WHERE sport=%s)",
            (sp["sport"],)
        )
        s[f"cards_{sp['sport']}"] = cur.fetchone()["c"]

    cur.close()
    conn.close()
    return s


def get_worker_stats() -> list[dict]:
    """Get card counts per worker/container."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("""
        SELECT
            COALESCE(worker_id, 'unassigned') AS worker,
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'processing' THEN 1 ELSE 0 END) AS processing,
            SUM(CASE WHEN status = 'image_found' THEN 1 ELSE 0 END) AS image_found,
            SUM(CASE WHEN status = 'downloading' THEN 1 ELSE 0 END) AS downloading,
            SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) AS downloaded,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
            SUM(CASE WHEN status = 'no_image' THEN 1 ELSE 0 END) AS no_image
        FROM cards
        WHERE worker_id IS NOT NULL
        GROUP BY worker_id
        ORDER BY total DESC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def reset_errors():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE sets SET csv_status='pending' WHERE csv_status='error'")
    cur.execute("UPDATE cards SET status='pending', error_msg=NULL WHERE status='error'")
    # Also reset any stuck processing/downloading cards (from crashed workers)
    cur.execute("UPDATE cards SET status='pending' WHERE status='processing'")
    cur.execute("UPDATE cards SET status='image_found' WHERE status='downloading'")
    conn.commit()
    cur.close()
    conn.close()


def reset_no_image():
    """Reset 'no_image' cards back to pending so they can be retried."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT COUNT(*) AS c FROM cards WHERE status='no_image'")
    count = cur.fetchone()["c"]
    cur.execute("UPDATE cards SET status='pending' WHERE status='no_image'")
    conn.commit()
    cur.close()
    conn.close()
    return count


def get_image_failure_stats() -> dict:
    """Get breakdown of cards without images by error reason."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    stats = {}
    cur.execute("""
        SELECT error_msg, COUNT(*) AS c FROM cards
        WHERE status='error' AND error_msg IS NOT NULL
        GROUP BY error_msg ORDER BY c DESC
    """)
    for r in cur.fetchall():
        stats[f"error: {r['error_msg']}"] = r["c"]
    cur.execute("SELECT COUNT(*) AS c FROM cards WHERE status='no_image'")
    no_img = cur.fetchone()["c"]
    if no_img:
        stats["no_image (confirmed)"] = no_img
    cur.close()
    conn.close()
    return stats
