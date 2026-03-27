#!/bin/bash
#
# Check which containers are actively processing cards
# Run this on your Proxmox host

echo "=== Container Activity Monitor ==="
echo "Time: $(date)"
echo ""

# Function to check a specific container
check_container() {
    local CTID=$1
    echo "--- Container $CTID ---"
    
    if ! pct status $CTID >/dev/null 2>&1; then
        echo "Container $CTID not found or not running"
        return
    fi
    
    # Check if container has scraper running
    if pct exec $CTID -- pgrep -f "python.*main" >/dev/null 2>&1; then
        echo "✓ Python scraper process running"
        
        # Get current database stats
        pct exec $CTID -- bash -c "cd /opt/scraperv2 2>/dev/null && source venv/bin/activate 2>/dev/null && python -c \"
import database as db
import psycopg2.extras
try:
    db.init_db()
    conn = db.get_connection()
    cur = conn.cursor()
    
    # Check processing cards
    cur.execute('SELECT COUNT(*) FROM cards WHERE status=\\\"processing\\\"')
    processing = cur.fetchone()[0]
    print(f'Cards being processed: {processing}')
    
    # Check recent progress 
    cur.execute('SELECT COUNT(*) FROM cards WHERE status=\\\"image_found\\\"')
    found = cur.fetchone()[0]
    print(f'Total image URLs found: {found}')
    
    cur.close()
    conn.close()
except Exception as e:
    print(f'Error: {e}')
\"" 2>/dev/null || echo "Could not get database stats"
    else
        echo "✗ No scraper process running"
    fi
    echo ""
}

# Check your containers (adjust container IDs as needed)
echo "Checking your containers..."
echo ""

# Check containers 128, 129, 130 (adjust as needed)
for CTID in 128 129 130; do
    check_container $CTID
done

echo "=== Quick Activity Check ==="
echo "To monitor live progress on container 128:"
echo "pct exec 128 -- bash -c \"cd /opt/scraperv2 && source venv/bin/activate && python check_container_activity.py monitor\""
echo ""
echo "To see current stats:"
echo "pct exec 128 -- scraper --stats"