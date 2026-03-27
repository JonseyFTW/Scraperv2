#!/usr/bin/env bash
#
# Fix v3 dependencies in existing LXC container
# Run this from Proxmox host: bash fix_v3_deps.sh [CTID]
#

set -euo pipefail

CTID="${1:-128}"  # Default to container 128
INSTALL_DIR="/opt/scraperv2"

echo "Fixing v3 dependencies in container $CTID..."

# Install system dependencies for curl_cffi
pct exec "$CTID" -- bash -c "
    apt-get update -qq
    apt-get install -y -qq python3-dev python3-venv libcurl4-openssl-dev build-essential
"

# Install Python dependencies in virtual environment
pct exec "$CTID" -- bash -c "
    cd ${INSTALL_DIR}
    
    # Ensure virtual environment exists
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Verify we're in venv
    echo \"Using Python: \$(which python)\"
    echo \"Using pip: \$(which pip)\"
    
    # Upgrade pip first
    pip install --upgrade pip wheel setuptools
    
    # Install curl_cffi (main dependency for v3)
    echo 'Installing curl_cffi...'
    pip install curl-cffi
    
    # Install optional v3 packages
    echo 'Installing scrapling (optional)...'
    pip install scrapling || echo 'Scrapling failed, will use Playwright fallback'
    
    echo 'Installing redis (optional)...'
    pip install redis || echo 'Redis failed, will use PostgreSQL queue'
    
    # Ensure all requirements are met
    echo 'Installing/updating all requirements...'
    pip install -r requirements.txt
    
    echo 'Verifying v3 dependencies...'
    python -c 'import curl_cffi; print(\"curl_cffi:\", curl_cffi.__version__)' || echo 'curl_cffi: FAILED'
    python -c 'import scrapling; print(\"scrapling: OK\")' || echo 'scrapling: OPTIONAL (OK)'
    python -c 'import redis; print(\"redis: OK\")' || echo 'redis: OPTIONAL (OK)'
    
    echo 'Testing v3 scraper import...'
    python -c 'import scraper_v3; print(\"scraper_v3: OK\")'
"

# Update the helper scripts to use virtual environment
pct exec "$CTID" -- bash -c "
# Update scraper command
cat > /usr/local/bin/scraper << 'SCRIPTEOF'
#!/bin/bash
cd ${INSTALL_DIR}
set -a; source .env 2>/dev/null || true; set +a
source venv/bin/activate
python main_v3.py \"\$@\"
SCRIPTEOF
chmod +x /usr/local/bin/scraper

# Update v2 fallback command
cat > /usr/local/bin/scraper-v2 << 'SCRIPTEOF'
#!/bin/bash
cd ${INSTALL_DIR}
set -a; source .env 2>/dev/null || true; set +a
source venv/bin/activate
python main.py \"\$@\"
SCRIPTEOF
chmod +x /usr/local/bin/scraper-v2

# Add CDN test command
cat > /usr/local/bin/test-cdn << 'SCRIPTEOF'
#!/bin/bash
cd ${INSTALL_DIR}
set -a; source .env 2>/dev/null || true; set +a
source venv/bin/activate
python main_v3.py --cdn-test
SCRIPTEOF
chmod +x /usr/local/bin/test-cdn

# Add queue monitoring (if redis available)
cat > /usr/local/bin/scraper-queue << 'SCRIPTEOF'
#!/bin/bash
cd ${INSTALL_DIR}
set -a; source .env 2>/dev/null || true; set +a
source venv/bin/activate
python task_queue.py stats
SCRIPTEOF
chmod +x /usr/local/bin/scraper-queue
"

echo "Dependencies fixed! Now you can:"
echo "  pct enter $CTID"
echo "  test-cdn      # Test CDN pattern discovery"
echo "  scraper       # Run v3 scraper"
echo "  scraper-v2    # Fallback to v2"