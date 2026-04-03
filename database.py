"""
SportsCardPro Scraper v2 - Database Layer (PostgreSQL)
"""
import os
import socket
import threading

import psycopg2
import psycopg2.extras
import psycopg2.pool
from datetime import datetime, timezone
from config import DATABASE_URL

# Worker identity — uses hostname so each LXC container is distinct
WORKER_ID = os.environ.get("WORKER_ID", socket.gethostname())

# ── Connection pool ──────────────────────────────────────────────────────
# ThreadedConnectionPool is thread-safe; min 2 connections, max 10.
_pool = None
_pool_lock = threading.Lock()


def _get_pool():
    global _pool
    if _pool is None or _pool.closed:
        with _pool_lock:
            if _pool is None or _pool.closed:
                _pool = psycopg2.pool.ThreadedConnectionPool(
                    minconn=2,
                    maxconn=10,
                    dsn=DATABASE_URL,
                    connect_timeout=10,
                )
    return _pool


def _conn_is_alive(conn):
    """Check if a pooled connection is still usable."""
    try:
        if conn.closed:
            return False
        # Reset any failed transaction state, ping, then end the transaction
        conn.rollback()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.rollback()  # End transaction so autocommit can be set after
        return True
    except Exception:
        return False


def get_connection(retries=3):
    """Get a healthy connection from the pool, replacing dead ones."""
    for attempt in range(retries):
        try:
            pool = _get_pool()
            conn = pool.getconn()
            if _conn_is_alive(conn):
                conn.autocommit = False
                return conn
            # Connection is dead — discard it and try again
            pool.putconn(conn, close=True)
            continue
        except (psycopg2.OperationalError, psycopg2.pool.PoolError):
            if attempt < retries - 1:
                import time
                time.sleep(2 ** attempt)
            else:
                raise
    # All retries exhausted (e.g. dead connections each time) — direct connect
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=10)
    conn.autocommit = False
    return conn


def put_connection(conn):
    """Return a connection to the pool, discarding it if broken."""
    try:
        if conn.closed:
            return
        pool = _get_pool()
        pool.putconn(conn)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass


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

    # ── Pokemon TCG tables ───────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pokemon_sets (
            id              TEXT PRIMARY KEY,
            name            TEXT NOT NULL,
            series          TEXT,
            total           INTEGER DEFAULT 0,
            release_date    TEXT,
            images_symbol   TEXT,
            images_logo     TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS pokemon_cards (
            id              TEXT PRIMARY KEY,   -- e.g. "base1-1"
            name            TEXT NOT NULL,
            local_id        TEXT,               -- card number within set
            set_id          TEXT REFERENCES pokemon_sets(id),
            set_name        TEXT,
            category        TEXT,               -- Pokemon, Trainer, Energy
            image_url       TEXT,               -- TCGdex base image URL
            image_path      TEXT,               -- local file path after download
            status          TEXT DEFAULT 'pending',  -- pending | downloaded | error
            error_msg       TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_pokemon_cards_status ON pokemon_cards(status);
        CREATE INDEX IF NOT EXISTS idx_pokemon_cards_set ON pokemon_cards(set_id);
    """)

    # ── TCGPlayer Pokemon tables ────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tcgplayer_sets (
            group_id        INTEGER PRIMARY KEY,   -- TCGPlayer groupId
            name            TEXT NOT NULL,
            abbreviation    TEXT,
            is_supplemental BOOLEAN DEFAULT FALSE,
            published_on    TEXT,
            modified_on     TEXT,
            card_count      INTEGER DEFAULT 0,
            created_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS tcgplayer_cards (
            product_id      INTEGER PRIMARY KEY,   -- TCGPlayer productId
            name            TEXT NOT NULL,
            clean_name      TEXT,
            group_id        INTEGER REFERENCES tcgplayer_sets(group_id),
            group_name      TEXT,
            image_url       TEXT,                   -- CDN image URL
            image_path      TEXT,                   -- local file path after download
            product_url     TEXT,                   -- tcgplayer.com product page
            card_number     TEXT,
            rarity          TEXT,
            card_type       TEXT,                   -- Pokemon, Trainer, Energy
            ext_data        JSONB,                  -- full extendedData from API
            status          TEXT DEFAULT 'pending', -- pending | downloaded | error | skipped
            error_msg       TEXT,
            created_at      TIMESTAMPTZ DEFAULT NOW(),
            updated_at      TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_tcgplayer_cards_status ON tcgplayer_cards(status);
        CREATE INDEX IF NOT EXISTS idx_tcgplayer_cards_group ON tcgplayer_cards(group_id);
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
    put_connection(conn)


def log_event(event: str, details: str = None):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO scrape_log (timestamp, event, details) VALUES (%s, %s, %s)",
        (datetime.now(timezone.utc).isoformat(), event, details)
    )
    conn.commit()
    cur.close()
    put_connection(conn)


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
    put_connection(conn)


def bulk_upsert_sets(sets_data: list[tuple]):
    """Each tuple: (slug, name, sport, url)"""
    conn = get_connection()
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO sets (slug, name, sport, url)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT(slug) DO UPDATE SET name=EXCLUDED.name, sport=EXCLUDED.sport
    """, sets_data)
    conn.commit()
    cur.close()
    put_connection(conn)


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
    put_connection(conn)
    return [dict(r) for r in rows]


def get_sets_needing_parse(sport: str = None) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if sport:
        cur.execute("SELECT * FROM sets WHERE csv_status='downloaded' AND sport=%s ORDER BY slug", (sport,))
    else:
        cur.execute("SELECT * FROM sets WHERE csv_status='downloaded' ORDER BY slug")
    rows = cur.fetchall()
    cur.close()
    put_connection(conn)
    return [dict(r) for r in rows]


def mark_set_csv_downloaded(slug: str, csv_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sets SET csv_status='downloaded', csv_path=%s, updated_at=%s WHERE slug=%s
    """, (csv_path, datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    cur.close()
    put_connection(conn)


def mark_set_csv_parsed(slug: str, card_count: int):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sets SET csv_status='parsed', card_count=%s, updated_at=%s WHERE slug=%s
    """, (card_count, datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    cur.close()
    put_connection(conn)


def mark_set_csv_error(slug: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE sets SET csv_status='error' WHERE slug=%s", (slug,))
    conn.commit()
    cur.close()
    put_connection(conn)


def mark_set_images_scraped(slug: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE sets SET img_status='scraped', updated_at=%s WHERE slug=%s
    """, (datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    cur.close()
    put_connection(conn)


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
    put_connection(conn)


def peek_cards_needing_images(limit: int = 10, sport: str = None) -> list[dict]:
    """Read a sample of pending cards WITHOUT claiming them (no status change)."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if sport:
        cur.execute("""
            SELECT c.* FROM cards c
            JOIN sets s ON s.slug = c.set_slug
            WHERE c.status='pending' AND s.sport = %s
            ORDER BY c.set_slug DESC, c.id LIMIT %s
        """, (sport, limit))
    else:
        cur.execute("""
            SELECT * FROM cards WHERE status='pending'
            ORDER BY set_slug DESC, id LIMIT %s
        """, (limit,))
    rows = cur.fetchall()
    cur.close()
    put_connection(conn)
    return [dict(r) for r in rows]


def get_cards_needing_images(limit: int = 500, sport: str = None) -> list[dict]:
    """Atomically claim a batch of pending cards for image scraping.
    Uses FOR UPDATE SKIP LOCKED so multiple workers never get the same rows."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if sport:
        cur.execute("""
            UPDATE cards SET status='processing', worker_id=%s
            WHERE id IN (
                SELECT c.id FROM cards c
                JOIN sets s ON s.slug = c.set_slug
                WHERE c.status='pending' AND s.sport = %s
                ORDER BY c.set_slug DESC, c.id
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
        """, (WORKER_ID, sport, limit))
    else:
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
    put_connection(conn)
    return [dict(r) for r in rows]


def count_pending_images() -> int:
    """Count how many cards still need image scraping (pending status)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM cards WHERE status = 'pending'")
    count = cur.fetchone()[0]
    cur.close()
    put_connection(conn)
    return count


def get_errored_cards(limit: int = 500, sport: str = None) -> list[dict]:
    """Claim errored cards for retry. Uses SKIP LOCKED for concurrency."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if sport:
        cur.execute("""
            UPDATE cards SET status='processing', worker_id=%s
            WHERE id IN (
                SELECT c.id FROM cards c
                JOIN sets s ON s.slug = c.set_slug
                WHERE c.status='error' AND s.sport = %s
                ORDER BY c.set_slug DESC, c.id
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
        """, (WORKER_ID, sport, limit))
    else:
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
    put_connection(conn)
    return [dict(r) for r in rows]


def get_cards_needing_download(limit: int = 500, sport: str = None) -> list[dict]:
    """Claim cards for image download. Uses SKIP LOCKED for concurrency."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    if sport:
        cur.execute("""
            UPDATE cards SET status='downloading', worker_id=%s
            WHERE id IN (
                SELECT c.id FROM cards c
                JOIN sets s ON s.slug = c.set_slug
                WHERE c.status='image_found' AND c.image_url IS NOT NULL AND s.sport = %s
                ORDER BY c.id
                LIMIT %s
                FOR UPDATE SKIP LOCKED
            )
            RETURNING *
        """, (WORKER_ID, sport, limit))
    else:
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
    put_connection(conn)
    return [dict(r) for r in rows]


def update_card_image_url(product_id: str, image_url: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards SET image_url=%s, status='image_found' WHERE product_id=%s
    """, (image_url, product_id))
    conn.commit()
    cur.close()
    put_connection(conn)


def mark_card_downloaded(product_id: str, image_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cards SET image_path=%s, status='downloaded' WHERE product_id=%s
    """, (image_path, product_id))
    conn.commit()
    cur.close()
    put_connection(conn)


def mark_card_no_image(product_id: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE cards SET status='no_image' WHERE product_id=%s", (product_id,))
    conn.commit()
    cur.close()
    put_connection(conn)


def mark_card_error(product_id: str, msg: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE cards SET status='error', error_msg=%s WHERE product_id=%s", (msg, product_id))
    conn.commit()
    cur.close()
    put_connection(conn)


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
    put_connection(conn)
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
    put_connection(conn)
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
    put_connection(conn)


def reset_no_image():
    """Reset 'no_image' cards back to pending so they can be retried."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT COUNT(*) AS c FROM cards WHERE status='no_image'")
    count = cur.fetchone()["c"]
    cur.execute("UPDATE cards SET status='pending' WHERE status='no_image'")
    conn.commit()
    cur.close()
    put_connection(conn)
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
    put_connection(conn)
    return stats


# ── Pokemon TCG operations ───────────────────────────────────────────────

def upsert_pokemon_set(set_data: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO pokemon_sets (id, name, series, total, release_date, images_symbol, images_logo)
        VALUES (%(id)s, %(name)s, %(series)s, %(total)s, %(release_date)s, %(images_symbol)s, %(images_logo)s)
        ON CONFLICT(id) DO UPDATE SET
            name = EXCLUDED.name, series = EXCLUDED.series, total = EXCLUDED.total,
            release_date = EXCLUDED.release_date, images_symbol = EXCLUDED.images_symbol,
            images_logo = EXCLUDED.images_logo
    """, set_data)
    conn.commit()
    cur.close()
    put_connection(conn)


def upsert_pokemon_cards_bulk(cards: list[dict]):
    conn = get_connection()
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO pokemon_cards (id, name, local_id, set_id, set_name, category, image_url, status)
        VALUES (%(id)s, %(name)s, %(local_id)s, %(set_id)s, %(set_name)s, %(category)s, %(image_url)s, %(status)s)
        ON CONFLICT(id) DO UPDATE SET
            name = EXCLUDED.name, local_id = EXCLUDED.local_id, set_name = EXCLUDED.set_name,
            category = EXCLUDED.category, image_url = EXCLUDED.image_url,
            updated_at = NOW()
    """, cards)
    conn.commit()
    cur.close()
    put_connection(conn)


def pokemon_mark_downloaded(card_id: str, image_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE pokemon_cards SET status = 'downloaded', image_path = %s, updated_at = NOW()
        WHERE id = %s
    """, (image_path, card_id))
    conn.commit()
    cur.close()
    put_connection(conn)


def pokemon_mark_error(card_id: str, msg: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE pokemon_cards SET status = 'error', error_msg = %s, updated_at = NOW()
        WHERE id = %s
    """, (msg, card_id))
    conn.commit()
    cur.close()
    put_connection(conn)


def get_pokemon_cards_by_status(status: str, limit: int = 0) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql = "SELECT * FROM pokemon_cards WHERE status = %s ORDER BY id"
    if limit > 0:
        sql += f" LIMIT {limit}"
    cur.execute(sql, (status,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    put_connection(conn)
    return rows


def get_pokemon_stats() -> dict:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    s = {}
    cur.execute("SELECT COUNT(*) AS c FROM pokemon_sets")
    s["sets"] = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM pokemon_cards")
    s["total"] = cur.fetchone()["c"]
    for st in ("pending", "downloaded", "error"):
        cur.execute("SELECT COUNT(*) AS c FROM pokemon_cards WHERE status = %s", (st,))
        s[st] = cur.fetchone()["c"]
    cur.close()
    put_connection(conn)
    return s


# ── TCGPlayer Pokemon operations ────────────────────────────────────────

def upsert_tcgplayer_set(set_data: dict):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO tcgplayer_sets (group_id, name, abbreviation, is_supplemental,
                                     published_on, modified_on, card_count)
        VALUES (%(group_id)s, %(name)s, %(abbreviation)s, %(is_supplemental)s,
                %(published_on)s, %(modified_on)s, %(card_count)s)
        ON CONFLICT(group_id) DO UPDATE SET
            name = EXCLUDED.name, abbreviation = EXCLUDED.abbreviation,
            is_supplemental = EXCLUDED.is_supplemental,
            published_on = EXCLUDED.published_on, modified_on = EXCLUDED.modified_on,
            card_count = EXCLUDED.card_count
    """, set_data)
    conn.commit()
    cur.close()
    put_connection(conn)


def upsert_tcgplayer_cards_bulk(cards: list[dict]):
    conn = get_connection()
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, """
        INSERT INTO tcgplayer_cards (product_id, name, clean_name, group_id, group_name,
                                      image_url, product_url, card_number, rarity,
                                      card_type, ext_data, status)
        VALUES (%(product_id)s, %(name)s, %(clean_name)s, %(group_id)s, %(group_name)s,
                %(image_url)s, %(product_url)s, %(card_number)s, %(rarity)s,
                %(card_type)s, %(ext_data)s, %(status)s)
        ON CONFLICT(product_id) DO UPDATE SET
            name = EXCLUDED.name, clean_name = EXCLUDED.clean_name,
            group_name = EXCLUDED.group_name, image_url = EXCLUDED.image_url,
            product_url = EXCLUDED.product_url, card_number = EXCLUDED.card_number,
            rarity = EXCLUDED.rarity, card_type = EXCLUDED.card_type,
            ext_data = EXCLUDED.ext_data, updated_at = NOW()
    """, cards)
    conn.commit()
    cur.close()
    put_connection(conn)


def tcgplayer_mark_downloaded(product_id: int, image_path: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE tcgplayer_cards SET status = 'downloaded', image_path = %s, updated_at = NOW()
        WHERE product_id = %s
    """, (image_path, product_id))
    conn.commit()
    cur.close()
    put_connection(conn)


def tcgplayer_mark_error(product_id: int, msg: str):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE tcgplayer_cards SET status = 'error', error_msg = %s, updated_at = NOW()
        WHERE product_id = %s
    """, (msg, product_id))
    conn.commit()
    cur.close()
    put_connection(conn)


def get_tcgplayer_cards_by_status(status: str, limit: int = 0) -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    sql = "SELECT * FROM tcgplayer_cards WHERE status = %s ORDER BY product_id"
    if limit > 0:
        sql += f" LIMIT {limit}"
    cur.execute(sql, (status,))
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    put_connection(conn)
    return rows


def get_tcgplayer_sets() -> list[dict]:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM tcgplayer_sets ORDER BY name")
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    put_connection(conn)
    return rows


def get_tcgplayer_stats() -> dict:
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    s = {}
    cur.execute("SELECT COUNT(*) AS c FROM tcgplayer_sets")
    s["sets"] = cur.fetchone()["c"]
    cur.execute("SELECT COUNT(*) AS c FROM tcgplayer_cards")
    s["total"] = cur.fetchone()["c"]
    for st in ("pending", "downloaded", "error", "skipped"):
        cur.execute("SELECT COUNT(*) AS c FROM tcgplayer_cards WHERE status = %s", (st,))
        s[st] = cur.fetchone()["c"]
    cur.close()
    put_connection(conn)
    return s
