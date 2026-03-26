#!/usr/bin/env python3
"""
One-time migration: SQLite → PostgreSQL

Copies all sets, cards, and scrape_log from the old SQLite database
into the new PostgreSQL database. Safe to run multiple times (uses
ON CONFLICT to skip existing rows).

Usage:
    python migrate_sqlite_to_pg.py
    python migrate_sqlite_to_pg.py --sqlite-path C:\Scripts\Scraperv2\data\sportscards.db
"""
import argparse
import os
import sqlite3

import psycopg2
import psycopg2.extras
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

import config
import database as db

console = Console()


def migrate(sqlite_path: str):
    console.print(f"\n[bold]Migrating SQLite → PostgreSQL[/bold]\n")
    console.print(f"  Source: [cyan]{sqlite_path}[/cyan]")
    console.print(f"  Target: [cyan]{config.DATABASE_URL.split('@')[1] if '@' in config.DATABASE_URL else config.DATABASE_URL}[/cyan]\n")

    if not os.path.exists(sqlite_path):
        console.print(f"  [red]SQLite file not found: {sqlite_path}[/red]")
        return

    # Ensure PG tables exist
    db.init_db()

    # Connect to SQLite
    sconn = sqlite3.connect(sqlite_path)
    sconn.row_factory = sqlite3.Row

    # Connect to PostgreSQL
    pconn = db.get_connection()
    pcur = pconn.cursor()

    # --- Migrate sets ---
    sets_rows = sconn.execute("SELECT * FROM sets").fetchall()
    console.print(f"  Sets to migrate: [cyan]{len(sets_rows):,}[/cyan]")

    if sets_rows:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
            task = progress.add_task("Migrating sets", total=len(sets_rows))
            batch = []
            for r in sets_rows:
                batch.append((
                    r["slug"], r["name"], r["sport"], r["url"],
                    r["csv_status"], r["img_status"], r["csv_path"],
                    r["card_count"], r["updated_at"],
                ))
                progress.advance(task)

            psycopg2.extras.execute_batch(pcur, """
                INSERT INTO sets (slug, name, sport, url, csv_status, img_status, csv_path, card_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (slug) DO NOTHING
            """, batch)
            pconn.commit()

    # --- Migrate cards ---
    total_cards = sconn.execute("SELECT COUNT(*) as c FROM cards").fetchone()["c"]
    console.print(f"  Cards to migrate: [cyan]{total_cards:,}[/cyan]")

    if total_cards > 0:
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), console=console) as progress:
            task = progress.add_task("Migrating cards", total=total_cards)
            batch_size = 1000
            offset = 0

            while offset < total_cards:
                rows = sconn.execute(
                    "SELECT * FROM cards ORDER BY id LIMIT ? OFFSET ?", (batch_size, offset)
                ).fetchall()
                if not rows:
                    break

                batch = []
                for r in rows:
                    batch.append((
                        r["product_id"], r["set_slug"], r["product_name"],
                        r["console_name"], r["card_url_slug"], r["full_url"],
                        r["image_url"], r["image_path"],
                        r["loose_price"], r["cib_price"], r["new_price"],
                        r["graded_price"], r["status"], r["error_msg"],
                    ))

                psycopg2.extras.execute_batch(pcur, """
                    INSERT INTO cards (product_id, set_slug, product_name, console_name,
                        card_url_slug, full_url, image_url, image_path,
                        loose_price, cib_price, new_price, graded_price, status, error_msg)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (product_id) DO NOTHING
                """, batch)
                pconn.commit()

                progress.advance(task, len(rows))
                offset += batch_size

    # --- Migrate scrape_log ---
    log_rows = sconn.execute("SELECT * FROM scrape_log").fetchall()
    console.print(f"  Log entries to migrate: [cyan]{len(log_rows):,}[/cyan]")

    if log_rows:
        batch = [(r["timestamp"], r["event"], r["details"]) for r in log_rows]
        psycopg2.extras.execute_batch(pcur, """
            INSERT INTO scrape_log (timestamp, event, details)
            VALUES (%s, %s, %s)
        """, batch)
        pconn.commit()

    pcur.close()
    pconn.close()
    sconn.close()

    console.print(f"\n  [green]Migration complete![/green]")
    console.print(f"  Run [bold]python main.py --stats[/bold] to verify.\n")


def main():
    default_path = os.path.join(config.DATA_DIR, "sportscards.db")
    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument("--sqlite-path", type=str, default=default_path,
                        help=f"Path to SQLite database (default: {default_path})")
    args = parser.parse_args()
    migrate(args.sqlite_path)


if __name__ == "__main__":
    main()
