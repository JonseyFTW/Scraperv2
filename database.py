"""
SportsCardPro Scraper v2 - Database Layer
"""
import sqlite3
import os
from datetime import datetime, timezone
from config import DB_PATH


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sets (
            slug        TEXT PRIMARY KEY,
            name        TEXT,
            sport       TEXT,
            url         TEXT NOT NULL,
            csv_status  TEXT DEFAULT 'pending',   -- pending | downloaded | parsed | error
            img_status  TEXT DEFAULT 'pending',   -- pending | scraped | error
            csv_path    TEXT,
            card_count  INTEGER DEFAULT 0,
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS cards (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id      TEXT,
            set_slug        TEXT NOT NULL,
            product_name    TEXT,
            console_name    TEXT,              -- set name from CSV
            card_url_slug   TEXT,              -- constructed URL slug
            full_url        TEXT,
            image_url       TEXT,
            image_path      TEXT,
            loose_price     REAL,              -- in dollars
            cib_price       REAL,
            new_price       REAL,
            graded_price    TEXT,              -- JSON of grade prices
            status          TEXT DEFAULT 'pending', -- pending | image_found | downloaded | no_image | error
            error_msg       TEXT,
            FOREIGN KEY (set_slug) REFERENCES sets(slug)
        );

        CREATE TABLE IF NOT EXISTS scrape_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            event       TEXT NOT NULL,
            details     TEXT
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_cards_pid ON cards(product_id);
        CREATE INDEX IF NOT EXISTS idx_cards_set ON cards(set_slug);
        CREATE INDEX IF NOT EXISTS idx_cards_status ON cards(status);
        CREATE INDEX IF NOT EXISTS idx_sets_csv ON sets(csv_status);
        CREATE INDEX IF NOT EXISTS idx_sets_img ON sets(img_status);
    """)
    conn.commit()
    conn.close()


def log_event(event: str, details: str = None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO scrape_log (timestamp, event, details) VALUES (?, ?, ?)",
        (datetime.now(timezone.utc).isoformat(), event, details)
    )
    conn.commit()
    conn.close()


# ── Set operations ────────────────────────────────────────────────────────

def upsert_set(slug: str, name: str, sport: str, url: str):
    conn = get_connection()
    conn.execute("""
        INSERT INTO sets (slug, name, sport, url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET name=excluded.name, sport=excluded.sport, url=excluded.url
    """, (slug, name, sport, url))
    conn.commit()
    conn.close()


def bulk_upsert_sets(sets_data: list[tuple]):
    """Each tuple: (slug, name, sport, url)"""
    conn = get_connection()
    conn.executemany("""
        INSERT INTO sets (slug, name, sport, url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(slug) DO UPDATE SET name=excluded.name
    """, sets_data)
    conn.commit()
    conn.close()


def get_sets_needing_csv(sport: str = None) -> list[dict]:
    conn = get_connection()
    if sport:
        rows = conn.execute(
            "SELECT * FROM sets WHERE csv_status='pending' AND sport=? ORDER BY slug", (sport,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM sets WHERE csv_status='pending' ORDER BY slug"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_sets_needing_parse() -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM sets WHERE csv_status='downloaded' ORDER BY slug"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_set_csv_downloaded(slug: str, csv_path: str):
    conn = get_connection()
    conn.execute("""
        UPDATE sets SET csv_status='downloaded', csv_path=?, updated_at=? WHERE slug=?
    """, (csv_path, datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    conn.close()


def mark_set_csv_parsed(slug: str, card_count: int):
    conn = get_connection()
    conn.execute("""
        UPDATE sets SET csv_status='parsed', card_count=?, updated_at=? WHERE slug=?
    """, (card_count, datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    conn.close()


def mark_set_csv_error(slug: str):
    conn = get_connection()
    conn.execute("UPDATE sets SET csv_status='error' WHERE slug=?", (slug,))
    conn.commit()
    conn.close()


def mark_set_images_scraped(slug: str):
    conn = get_connection()
    conn.execute("""
        UPDATE sets SET img_status='scraped', updated_at=? WHERE slug=?
    """, (datetime.now(timezone.utc).isoformat(), slug))
    conn.commit()
    conn.close()


# ── Card operations ───────────────────────────────────────────────────────

def bulk_insert_cards(cards: list[dict]):
    """Insert cards from CSV parse. Each dict has CSV column data."""
    conn = get_connection()
    conn.executemany("""
        INSERT OR IGNORE INTO cards
            (product_id, set_slug, product_name, console_name,
             card_url_slug, full_url, loose_price, cib_price, new_price)
        VALUES
            (:product_id, :set_slug, :product_name, :console_name,
             :card_url_slug, :full_url, :loose_price, :cib_price, :new_price)
    """, cards)
    conn.commit()
    conn.close()


def get_cards_needing_images(limit: int = 500) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM cards WHERE status='pending' ORDER BY set_slug, id LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_cards_needing_download(limit: int = 500) -> list[dict]:
    conn = get_connection()
    rows = conn.execute("""
        SELECT * FROM cards WHERE status='image_found' AND image_url IS NOT NULL
        ORDER BY id LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_card_image_url(product_id: str, image_url: str):
    conn = get_connection()
    conn.execute("""
        UPDATE cards SET image_url=?, status='image_found' WHERE product_id=?
    """, (image_url, product_id))
    conn.commit()
    conn.close()


def mark_card_downloaded(product_id: str, image_path: str):
    conn = get_connection()
    conn.execute("""
        UPDATE cards SET image_path=?, status='downloaded' WHERE product_id=?
    """, (image_path, product_id))
    conn.commit()
    conn.close()


def mark_card_no_image(product_id: str):
    conn = get_connection()
    conn.execute("UPDATE cards SET status='no_image' WHERE product_id=?", (product_id,))
    conn.commit()
    conn.close()


def mark_card_error(product_id: str, msg: str):
    conn = get_connection()
    conn.execute("UPDATE cards SET status='error', error_msg=? WHERE product_id=?", (msg, product_id))
    conn.commit()
    conn.close()


# ── Stats ─────────────────────────────────────────────────────────────────

def get_stats() -> dict:
    conn = get_connection()
    s = {}

    s["total_sets"] = conn.execute("SELECT COUNT(*) c FROM sets").fetchone()["c"]
    for st in ("pending", "downloaded", "parsed", "error"):
        s[f"sets_csv_{st}"] = conn.execute(
            "SELECT COUNT(*) c FROM sets WHERE csv_status=?", (st,)
        ).fetchone()["c"]

    s["total_cards"] = conn.execute("SELECT COUNT(*) c FROM cards").fetchone()["c"]
    for st in ("pending", "image_found", "downloaded", "no_image", "error"):
        s[f"cards_{st}"] = conn.execute(
            "SELECT COUNT(*) c FROM cards WHERE status=?", (st,)
        ).fetchone()["c"]

    # Per sport
    sports = conn.execute("SELECT DISTINCT sport FROM sets ORDER BY sport").fetchall()
    for sp in sports:
        cnt = conn.execute(
            "SELECT COUNT(*) c FROM cards WHERE set_slug IN (SELECT slug FROM sets WHERE sport=?)",
            (sp["sport"],)
        ).fetchone()["c"]
        s[f"cards_{sp['sport']}"] = cnt

    conn.close()
    return s


def reset_errors():
    conn = get_connection()
    conn.execute("UPDATE sets SET csv_status='pending' WHERE csv_status='error'")
    conn.execute("UPDATE cards SET status='pending', error_msg=NULL WHERE status='error'")
    conn.commit()
    conn.close()


def reset_no_image():
    """Reset 'no_image' cards back to pending so they can be retried."""
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) as c FROM cards WHERE status='no_image'").fetchone()["c"]
    conn.execute("UPDATE cards SET status='pending' WHERE status='no_image'")
    conn.commit()
    conn.close()
    return count


def get_image_failure_stats() -> dict:
    """Get breakdown of cards without images by error reason."""
    conn = get_connection()
    stats = {}
    # Error reasons
    rows = conn.execute("""
        SELECT error_msg, COUNT(*) as c FROM cards
        WHERE status='error' AND error_msg IS NOT NULL
        GROUP BY error_msg ORDER BY c DESC
    """).fetchall()
    for r in rows:
        stats[f"error: {r['error_msg']}"] = r["c"]
    # No image count
    no_img = conn.execute("SELECT COUNT(*) as c FROM cards WHERE status='no_image'").fetchone()["c"]
    if no_img:
        stats["no_image (confirmed)"] = no_img
    conn.close()
    return stats
