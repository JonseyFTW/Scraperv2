#!/usr/bin/env python3
"""
Container Activity Monitor
Check which container is actively processing cards
"""
import psycopg2
import psycopg2.extras
import time
import os
from datetime import datetime

# Use your database connection
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:changeme@192.168.1.14:5433/sportscards")

def check_activity():
    """Check current processing activity"""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    
    print("=== Container Processing Activity ===")
    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
    
    # Check currently processing cards
    cur.execute("SELECT COUNT(*) as count FROM cards WHERE status='processing'")
    processing = cur.fetchone()['count']
    print(f"Cards currently being processed: {processing}")
    
    # Check recent completions (cards that moved from processing to image_found)
    cur.execute("""
        SELECT COUNT(*) as count 
        FROM cards 
        WHERE status='image_found' 
        AND id > (SELECT MAX(id) - 1000 FROM cards WHERE status IN ('processing', 'image_found'))
    """)
    recent_found = cur.fetchone()['count']
    print(f"Recent image URLs found: {recent_found}")
    
    # Show some recent activity
    cur.execute("""
        SELECT product_id, status, set_slug
        FROM cards 
        WHERE status IN ('processing', 'image_found') 
        ORDER BY id DESC 
        LIMIT 5
    """)
    recent = cur.fetchall()
    print(f"\nRecent activity:")
    for card in recent:
        print(f"  {card['product_id']} ({card['set_slug']}): {card['status']}")
    
    cur.close()
    conn.close()

def monitor_progress():
    """Monitor progress over time"""
    print("\n=== Monitoring Progress (press Ctrl+C to stop) ===")
    
    last_found_count = None
    
    try:
        while True:
            conn = psycopg2.connect(DATABASE_URL)
            cur = conn.cursor()
            
            # Get current image_found count
            cur.execute("SELECT COUNT(*) FROM cards WHERE status='image_found'")
            current_found = cur.fetchone()[0]
            
            if last_found_count is not None:
                new_found = current_found - last_found_count
                if new_found > 0:
                    print(f"{datetime.now().strftime('%H:%M:%S')} - Found {new_found} new image URLs (total: {current_found})")
                elif new_found == 0:
                    print(f"{datetime.now().strftime('%H:%M:%S')} - No new progress (total: {current_found})")
            
            last_found_count = current_found
            
            cur.close()
            conn.close()
            
            time.sleep(10)  # Check every 10 seconds
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        check_activity()
        monitor_progress()
    else:
        check_activity()