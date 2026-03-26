#!/bin/bash
#
# SportsCardPro Scraper — Proxmox LXC Container Setup
#
# Run this ON your Proxmox host to create a scraper LXC container
# with NordVPN, Python, and all dependencies pre-configured.
#
# Usage:
#   bash setup_scraper_lxc.sh 201 us10299      # CTID 201, NordVPN server us10299
#   bash setup_scraper_lxc.sh 202 uk2547       # CTID 202, different server
#   bash setup_scraper_lxc.sh 203 de1048       # CTID 203, different server
#
# After setup, SSH into the container and run:
#   nordvpn login --token YOUR_NORD_TOKEN
#   nordvpn connect <server>
#   cd /opt/scraperv2 && python main.py --phase 4 --sport football
#
# Prerequisites:
#   - Ubuntu 24.04 LXC template downloaded on Proxmox
#     pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst
#   - NordVPN token (get from: https://my.nordaccount.com/dashboard/nordvpn/manual-configuration/)

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
CTID="${1:?Usage: $0 <CTID> <NORDVPN_SERVER> [NORD_TOKEN]}"
NORD_SERVER="${2:?Usage: $0 <CTID> <NORDVPN_SERVER> [NORD_TOKEN]}"
NORD_TOKEN="${3:-}"

HOSTNAME="scraper-${NORD_SERVER}"
TEMPLATE="local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst"
STORAGE="local-lvm"
MEMORY=4096
SWAP=1024
CORES=2
DISK_SIZE="20"
BRIDGE="vmbr0"
NAMESERVER="1.1.1.1"

# Git repo
REPO_URL="https://github.com/JonseyFTW/Scraperv2.git"
REPO_BRANCH="claude/add-phase5-progress-visuals-hY7L5"
INSTALL_DIR="/opt/scraperv2"

# Database — point to your UGREEN
DB_URL="postgresql://postgres:changeme@192.168.1.14:5433/sportscards"
SCP_EMAIL="${SCP_EMAIL:-}"
SCP_PASSWORD="${SCP_PASSWORD:-}"

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $1"; }
info() { echo -e "${CYAN}[i]${NC} $1"; }
err()  { echo -e "${RED}[!]${NC} $1"; exit 1; }

# ── Preflight checks ─────────────────────────────────────────────────────
command -v pct >/dev/null 2>&1 || err "This script must be run on a Proxmox host"
[[ -f "/var/lib/vz/template/cache/$(basename "$TEMPLATE" | sed 's/.*\///')" ]] || \
    pveam list local | grep -q "ubuntu-24.04" || \
    err "Ubuntu 24.04 template not found. Run:\n  pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst"

pct status "$CTID" >/dev/null 2>&1 && err "Container $CTID already exists"

# ── Create container ──────────────────────────────────────────────────────
log "Creating LXC container ${BOLD}$CTID${NC} (${HOSTNAME})"

pct create "$CTID" "$TEMPLATE" \
    --hostname "$HOSTNAME" \
    --memory "$MEMORY" \
    --swap "$SWAP" \
    --cores "$CORES" \
    --rootfs "${STORAGE}:${DISK_SIZE}" \
    --net0 "name=eth0,bridge=${BRIDGE},ip=dhcp" \
    --nameserver "$NAMESERVER" \
    --unprivileged 0 \
    --features "nesting=1,keyctl=1" \
    --onboot 1 \
    --start 0

# NordVPN needs /dev/net/tun — add to container config
log "Enabling TUN device for VPN"
cat >> "/etc/pve/lxc/${CTID}.conf" <<'EOF'
lxc.cgroup2.devices.allow: c 10:200 rwm
lxc.mount.entry: /dev/net dev/net none bind,create=dir
EOF

# ── Start container ───────────────────────────────────────────────────────
log "Starting container $CTID"
pct start "$CTID"
sleep 5

# Wait for network
log "Waiting for network..."
for i in $(seq 1 30); do
    if pct exec "$CTID" -- ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
pct exec "$CTID" -- ping -c 1 -W 2 1.1.1.1 >/dev/null 2>&1 || err "Container has no network"

# Get container IP
CT_IP=$(pct exec "$CTID" -- hostname -I | awk '{print $1}')
log "Container IP: ${BOLD}${CT_IP}${NC}"

# ── Install system packages ──────────────────────────────────────────────
log "Installing system packages (this takes a few minutes)..."
pct exec "$CTID" -- bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq \
        python3 python3-pip python3-venv \
        git curl wget gnupg2 apt-transport-https \
        libpq-dev build-essential \
        ca-certificates >/dev/null 2>&1
    echo "System packages installed"
'

# ── Install NordVPN ───────────────────────────────────────────────────────
log "Installing NordVPN..."
pct exec "$CTID" -- bash -c '
    sh <(curl -sSf https://downloads.nordcdn.com/apps/linux/install.sh) <<< "y"
    # Wait for nordvpnd to be ready
    sleep 3
    systemctl enable --now nordvpnd 2>/dev/null || true
    sleep 2
    echo "NordVPN installed"
'

# ── Configure NordVPN ─────────────────────────────────────────────────────
if [[ -n "$NORD_TOKEN" ]]; then
    log "Logging into NordVPN with token..."
    pct exec "$CTID" -- nordvpn login --token "$NORD_TOKEN"
    sleep 2
fi

log "Configuring NordVPN settings..."
pct exec "$CTID" -- bash -c "
    nordvpn set technology nordlynx
    nordvpn set firewall off
    nordvpn set killswitch off
    nordvpn set autoconnect on ${NORD_SERVER}
"

# Connect if logged in
if [[ -n "$NORD_TOKEN" ]]; then
    log "Connecting to NordVPN server: ${BOLD}${NORD_SERVER}${NC}"
    pct exec "$CTID" -- nordvpn connect "$NORD_SERVER"
    sleep 5
    pct exec "$CTID" -- nordvpn status
fi

# ── Clone repo and install Python deps ────────────────────────────────────
log "Setting up scraper..."
pct exec "$CTID" -- bash -c "
    git clone --branch ${REPO_BRANCH} ${REPO_URL} ${INSTALL_DIR}
    cd ${INSTALL_DIR}
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
    playwright install chromium
    playwright install-deps chromium
    echo 'Scraper installed'
"

# ── Write environment file ────────────────────────────────────────────────
log "Writing environment config..."
pct exec "$CTID" -- bash -c "cat > ${INSTALL_DIR}/.env << 'ENVEOF'
DATABASE_URL=${DB_URL}
SCP_EMAIL=${SCP_EMAIL}
SCP_PASSWORD=${SCP_PASSWORD}
ENVEOF"

# Write a helper script to source env and run
pct exec "$CTID" -- bash -c "cat > /usr/local/bin/scraper << 'SCRIPTEOF'
#!/bin/bash
cd ${INSTALL_DIR}
set -a
source .env
set +a
source venv/bin/activate
python main.py \"\$@\"
SCRIPTEOF
chmod +x /usr/local/bin/scraper"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}  Container $CTID ($HOSTNAME) is ready!${NC}"
echo -e "${GREEN}${BOLD}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}VPN Server:${NC}  $NORD_SERVER"
echo -e "  ${CYAN}IP Address:${NC}  $CT_IP"
echo -e "  ${CYAN}Database:${NC}    $DB_URL"
echo ""

if [[ -z "$NORD_TOKEN" ]]; then
    echo -e "  ${RED}⚠  NordVPN not logged in yet. Run:${NC}"
    echo -e "     pct exec $CTID -- nordvpn login --token YOUR_TOKEN"
    echo -e "     pct exec $CTID -- nordvpn connect $NORD_SERVER"
    echo ""
fi

echo -e "  ${BOLD}Quick start:${NC}"
echo -e "     pct enter $CTID"
echo -e "     scraper --stats"
echo -e "     scraper --phase 4 --sport football --limit 500"
echo ""
echo -e "  ${BOLD}Or run from Proxmox host:${NC}"
echo -e "     pct exec $CTID -- scraper --phase 4 --sport football"
echo ""
