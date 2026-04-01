#!/usr/bin/env python3
"""
Auto-reset error cards back to pending once pending hits zero.
Loops until both pending and errors are zero.

Usage:
    python auto_reset_errors.py
    python auto_reset_errors.py --interval 60   # check every 60 seconds (default: 30)
"""
import argparse
import sys
import time

import database as db


def get_counts():
    """Get current pending and error counts."""
    conn = db.get_connection()
    cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM cards WHERE status IN ('pending', 'error', 'image_found', 'downloading', 'processing') GROUP BY status")
    counts = dict(cur.fetchall())
    cur.close()
    db.put_connection(conn)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Auto-reset errors when pending reaches zero")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds (default: 30)")
    args = parser.parse_args()

    db.init_db()
    reset_count = 0

    print(f"Watching for pending=0 to reset errors (checking every {args.interval}s)...")
    print("Press Ctrl+C to stop.\n")

    while True:
        counts = get_counts()
        pending = counts.get("pending", 0)
        errors = counts.get("error", 0)
        image_found = counts.get("image_found", 0)
        downloading = counts.get("downloading", 0)
        processing = counts.get("processing", 0)

        in_flight = image_found + downloading + processing
        timestamp = time.strftime("%H:%M:%S")

        print(f"[{timestamp}] pending={pending:,}  in_flight={in_flight:,}  errors={errors:,}", end="")

        if pending == 0 and errors == 0 and in_flight == 0:
            print(f"\n\nAll done! Reset errors {reset_count} time(s).")
            break

        if pending == 0 and in_flight == 0 and errors > 0:
            print(f"  → resetting {errors:,} errors to pending")
            db.reset_errors()
            reset_count += 1
        else:
            print()

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
