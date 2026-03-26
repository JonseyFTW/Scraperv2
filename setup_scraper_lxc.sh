#!/usr/bin/env bash
#
# SportsCardPro Scraper — Proxmox LXC Setup Wizard
#
# Interactive setup that walks you through creating a scraper
# container with NordVPN, Python, and all dependencies.
#
# Install:
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/JonseyFTW/Scraperv2/main/setup_scraper_lxc.sh)"
#
# Or locally:
#   bash setup_scraper_lxc.sh

set -euo pipefail

# ── Colors & helpers ──────────────────────────────────────────────────────
YW="\033[33m"
GN="\033[1;92m"
CL="\033[m"
RD="\033[01;31m"
BL="\033[36m"
CM="${GN}✓${CL}"
CROSS="${RD}✗${CL}"

APP="SportsCardPro Scraper"
REPO_URL="https://github.com/JonseyFTW/Scraperv2.git"
REPO_BRANCH="claude/add-phase5-progress-visuals-hY7L5"
INSTALL_DIR="/opt/scraperv2"

header_info() {
    clear
    cat <<"EOF"
   ___              _      ___              _
  / __|_ __  ___ _ _| |_ __/ __|__ _ _ _ __| |
  \__ \ '_ \/ _ \ '_|  _(_) (__/ _` | '_/ _` |
  |___/ .__/\___/_|  \__(_)\___\__,_|_| \__,_|
      |_|  Pro Scraper v2 — LXC Setup Wizard

EOF
}

msg_ok()   { echo -e " ${CM}  ${GN}${1}${CL}"; }
msg_info() { echo -e " ...  ${YW}${1}${CL}"; }
msg_error(){ echo -e " ${CROSS}  ${RD}${1}${CL}"; }

# ── Preflight ─────────────────────────────────────────────────────────────
header_info
if ! command -v pct &>/dev/null; then
    msg_error "This script must be run on a Proxmox VE host."
    exit 1
fi

if ! command -v whiptail &>/dev/null; then
    msg_error "whiptail not found. Install it: apt install whiptail"
    exit 1
fi

# Check for Ubuntu template
TEMPLATE_FILE=$(pveam list local 2>/dev/null | grep -o "local:vztmpl/ubuntu-24.04[^ ]*" | head -1 || true)
if [[ -z "$TEMPLATE_FILE" ]]; then
    if whiptail --title "Template Missing" --yesno \
        "Ubuntu 24.04 LXC template not found.\n\nDownload it now? (This may take a minute)" 10 60; then
        msg_info "Downloading Ubuntu 24.04 template..."
        pveam download local ubuntu-24.04-standard_24.04-2_amd64.tar.zst
        TEMPLATE_FILE="local:vztmpl/ubuntu-24.04-standard_24.04-2_amd64.tar.zst"
        msg_ok "Template downloaded"
    else
        msg_error "Cannot continue without template"
        exit 1
    fi
fi

# ── Step 1: Container ID ─────────────────────────────────────────────────
header_info

# Find next available CTID
NEXT_ID=$(pvesh get /cluster/nextid 2>/dev/null || echo "200")

CTID=$(whiptail --title "$APP — Container ID" \
    --inputbox "Enter the Container ID (CTID) for the new LXC:\n\nNext available: ${NEXT_ID}" \
    10 60 "$NEXT_ID" 3>&1 1>&2 2>&3) || exit 1

# Validate CTID not in use
if pct status "$CTID" &>/dev/null; then
    msg_error "Container $CTID already exists!"
    exit 1
fi

# ── Step 2: Resources ────────────────────────────────────────────────────
header_info

CORES=$(whiptail --title "$APP — CPU Cores" \
    --inputbox "Number of CPU cores:" \
    8 60 "2" 3>&1 1>&2 2>&3) || exit 1

MEMORY=$(whiptail --title "$APP — Memory" \
    --inputbox "Memory in MB:" \
    8 60 "4096" 3>&1 1>&2 2>&3) || exit 1

DISK_SIZE=$(whiptail --title "$APP — Disk Size" \
    --inputbox "Root disk size in GB:" \
    8 60 "20" 3>&1 1>&2 2>&3) || exit 1

# ── Step 3: Storage selection ────────────────────────────────────────────
header_info

# Get available storages
STORAGE_LIST=$(pvesm status -content rootdir 2>/dev/null | awk 'NR>1 {print $1, $1}' || echo "local-lvm local-lvm")
STORAGE=$(whiptail --title "$APP — Storage" \
    --menu "Select storage for the container:" \
    14 60 5 $STORAGE_LIST 3>&1 1>&2 2>&3) || exit 1

# ── Step 4: Network bridge ───────────────────────────────────────────────
header_info

BRIDGE_LIST=$(ip link show type bridge 2>/dev/null | grep -oP '(?<=: )\w+' | awk '{print $1, $1}' || echo "vmbr0 vmbr0")
BRIDGE=$(whiptail --title "$APP — Network Bridge" \
    --menu "Select network bridge:" \
    14 60 5 $BRIDGE_LIST 3>&1 1>&2 2>&3) || exit 1

# ── Step 5: NordVPN ──────────────────────────────────────────────────────
header_info

NORD_TOKEN=$(whiptail --title "$APP — NordVPN Token" \
    --inputbox "Enter your NordVPN access token:\n\n(Get it from: https://my.nordaccount.com/dashboard/nordvpn/manual-configuration/)\n\nLeave blank to configure later." \
    14 70 "" 3>&1 1>&2 2>&3) || exit 1

NORD_SERVER=$(whiptail --title "$APP — NordVPN Server" \
    --menu "Select a VPN region (or type a specific server like us10299):" \
    20 60 10 \
    "us"        "United States (auto)" \
    "uk"        "United Kingdom (auto)" \
    "ca"        "Canada (auto)" \
    "de"        "Germany (auto)" \
    "nl"        "Netherlands (auto)" \
    "se"        "Sweden (auto)" \
    "ch"        "Switzerland (auto)" \
    "au"        "Australia (auto)" \
    "jp"        "Japan (auto)" \
    "custom"    "Enter a specific server..." \
    3>&1 1>&2 2>&3) || exit 1

if [[ "$NORD_SERVER" == "custom" ]]; then
    NORD_SERVER=$(whiptail --title "$APP — Custom Server" \
        --inputbox "Enter NordVPN server name:\n\n(e.g. us10299, uk2547, de1048)" \
        10 60 "" 3>&1 1>&2 2>&3) || exit 1
fi

# ── Step 6: Shared storage (images + CSVs) ───────────────────────────────
header_info

NAS_SHARE=$(whiptail --title "$APP — Shared Image Storage" \
    --inputbox "NFS/SMB share path on your NAS for images + CSVs:\n\n(This is where ALL containers save images so your GPU\nmachine can access them for embedding generation)\n\nNFS example: 192.168.1.14:/volume1/Data/scraper" \
    14 70 "192.168.1.14:/volume1/Data/scraper" 3>&1 1>&2 2>&3) || exit 1

SHARE_MOUNT="/mnt/scraper-data"

# ── Step 7: Database ─────────────────────────────────────────────────────
header_info

DB_URL=$(whiptail --title "$APP — PostgreSQL" \
    --inputbox "PostgreSQL connection string:\n\n(Your UGREEN NAS)" \
    10 70 "postgresql://postgres:changeme@192.168.1.14:5433/sportscards" 3>&1 1>&2 2>&3) || exit 1

# ── Step 8: SportsCardsPro credentials ───────────────────────────────────
header_info

SCP_EMAIL=$(whiptail --title "$APP — SportsCardsPro Login" \
    --inputbox "SportsCardsPro email:" \
    8 60 "" 3>&1 1>&2 2>&3) || exit 1

SCP_PASSWORD=$(whiptail --title "$APP — SportsCardsPro Login" \
    --passwordbox "SportsCardsPro password:" \
    8 60 "" 3>&1 1>&2 2>&3) || exit 1

# ── Confirm ──────────────────────────────────────────────────────────────
header_info

HOSTNAME="scraper-${NORD_SERVER}"

whiptail --title "$APP — Confirm Setup" --yesno \
"Ready to create the scraper container:\n
  Container ID:   $CTID
  Hostname:       $HOSTNAME
  Cores / RAM:    $CORES / ${MEMORY}MB
  Disk:           ${DISK_SIZE}GB on $STORAGE
  Bridge:         $BRIDGE
  VPN Server:     $NORD_SERVER
  Shared Storage: $NAS_SHARE → $SHARE_MOUNT
  Database:       $(echo "$DB_URL" | sed 's|://[^@]*@|://***@|')
  SCP Account:    $SCP_EMAIL

Proceed?" 22 68 || exit 1

# ══════════════════════════════════════════════════════════════════════════
#  INSTALLATION
# ══════════════════════════════════════════════════════════════════════════
header_info

# ── Create container ──────────────────────────────────────────────────────
msg_info "Creating LXC container $CTID ($HOSTNAME)"

pct create "$CTID" "$TEMPLATE_FILE" \
    --hostname "$HOSTNAME" \
    --memory "$MEMORY" \
    --swap 1024 \
    --cores "$CORES" \
    --rootfs "${STORAGE}:${DISK_SIZE}" \
    --net0 "name=eth0,bridge=${BRIDGE},ip=dhcp" \
    --nameserver "1.1.1.1" \
    --unprivileged 0 \
    --features "nesting=1,keyctl=1" \
    --onboot 1 \
    --start 0

msg_ok "Container $CTID created"

# ── Enable TUN for VPN ───────────────────────────────────────────────────
msg_info "Enabling TUN device for VPN"
cat >> "/etc/pve/lxc/${CTID}.conf" <<'TUNEOF'
lxc.cgroup2.devices.allow: c 10:200 rwm
lxc.mount.entry: /dev/net dev/net none bind,create=dir
TUNEOF
msg_ok "TUN device enabled"

# ── Start container ───────────────────────────────────────────────────────
msg_info "Starting container"
pct start "$CTID"
sleep 5

msg_info "Waiting for network"
for i in $(seq 1 30); do
    pct exec "$CTID" -- ping -c 1 -W 2 1.1.1.1 &>/dev/null && break
    sleep 2
done
pct exec "$CTID" -- ping -c 1 -W 2 1.1.1.1 &>/dev/null || { msg_error "No network"; exit 1; }

CT_IP=$(pct exec "$CTID" -- hostname -I | awk '{print $1}')
msg_ok "Network ready (IP: $CT_IP)"

# ── Mount shared NAS storage ─────────────────────────────────────────────
msg_info "Mounting shared storage ($NAS_SHARE)"
pct exec "$CTID" -- bash -c "
    apt-get install -y -qq nfs-common &>/dev/null
    mkdir -p ${SHARE_MOUNT}
    echo '${NAS_SHARE} ${SHARE_MOUNT} nfs defaults,_netdev 0 0' >> /etc/fstab
    mount ${SHARE_MOUNT}
"
if pct exec "$CTID" -- mountpoint -q "$SHARE_MOUNT" 2>/dev/null; then
    msg_ok "Shared storage mounted at $SHARE_MOUNT"
else
    msg_error "NFS mount failed — trying SMB/CIFS instead"
    pct exec "$CTID" -- bash -c "
        apt-get install -y -qq cifs-utils &>/dev/null
        # Remove failed NFS entry
        sed -i '\|${NAS_SHARE}|d' /etc/fstab
        # Try SMB mount (guest/no password)
        echo '//${NAS_SHARE#*:} ${SHARE_MOUNT} cifs guest,_netdev,uid=0,gid=0 0 0' >> /etc/fstab
        mount ${SHARE_MOUNT}
    " 2>/dev/null
    if pct exec "$CTID" -- mountpoint -q "$SHARE_MOUNT" 2>/dev/null; then
        msg_ok "Shared storage mounted via SMB at $SHARE_MOUNT"
    else
        msg_error "Could not mount shared storage — you may need to mount manually"
    fi
fi

# ── Install system packages ──────────────────────────────────────────────
msg_info "Installing system packages (this takes a few minutes)"
pct exec "$CTID" -- bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq &>/dev/null
    apt-get install -y -qq \
        python3 python3-pip python3-venv \
        git curl wget gnupg2 apt-transport-https \
        libpq-dev build-essential \
        ca-certificates &>/dev/null
'
msg_ok "System packages installed"

# ── Install NordVPN ───────────────────────────────────────────────────────
msg_info "Installing NordVPN"
pct exec "$CTID" -- bash -c '
    sh <(curl -sSf https://downloads.nordcdn.com/apps/linux/install.sh) <<< "y" &>/dev/null
    sleep 3
    systemctl enable --now nordvpnd &>/dev/null || true
    sleep 2
'
msg_ok "NordVPN installed"

# ── Configure NordVPN ─────────────────────────────────────────────────────
msg_info "Configuring NordVPN (server: $NORD_SERVER)"

pct exec "$CTID" -- bash -c "
    nordvpn set technology nordlynx 2>/dev/null
    nordvpn set firewall off 2>/dev/null
    nordvpn set killswitch off 2>/dev/null
    nordvpn set autoconnect on ${NORD_SERVER} 2>/dev/null
" || true

if [[ -n "$NORD_TOKEN" ]]; then
    pct exec "$CTID" -- nordvpn login --token "$NORD_TOKEN" 2>/dev/null || true
    sleep 2
    pct exec "$CTID" -- nordvpn connect "$NORD_SERVER" 2>/dev/null || true
    sleep 3
    msg_ok "NordVPN connected"
else
    msg_ok "NordVPN installed (login required — see instructions below)"
fi

# ── Clone repo and install Python deps ────────────────────────────────────
msg_info "Installing scraper and Python dependencies"
pct exec "$CTID" -- bash -c "
    git clone --branch ${REPO_BRANCH} ${REPO_URL} ${INSTALL_DIR} &>/dev/null
    cd ${INSTALL_DIR}
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt 2>/dev/null
    playwright install chromium &>/dev/null
    playwright install-deps chromium &>/dev/null
"
msg_ok "Scraper installed"

# ── Write environment file ────────────────────────────────────────────────
msg_info "Writing configuration"
pct exec "$CTID" -- bash -c "cat > ${INSTALL_DIR}/.env << 'ENVEOF'
DATABASE_URL=${DB_URL}
SCP_DATA_DIR=${SHARE_MOUNT}
SCP_EMAIL=${SCP_EMAIL}
SCP_PASSWORD=${SCP_PASSWORD}
ENVEOF"

# Helper command with update support — write to temp file then push
# (avoids bash history expansion issues with #!/bin/bash inside pct exec)
cat > /tmp/scraper_helper_${CTID}.sh << HELPEREOF
#!/bin/bash
INSTALL_DIR="${INSTALL_DIR}"
cd "\$INSTALL_DIR"
set -a; source .env; set +a
source venv/bin/activate

if [[ "\${1:-}" == "update" ]]; then
    echo "Pulling latest changes..."
    git pull origin ${REPO_BRANCH}
    echo "Updating dependencies..."
    pip install -q -r requirements.txt
    echo "Done! Scraper is up to date."
    exit 0
fi

if [[ "\${1:-}" == "vpn" ]]; then
    nordvpn status
    exit 0
fi

python main.py "\$@"
HELPEREOF

pct push "$CTID" /tmp/scraper_helper_${CTID}.sh /usr/local/bin/scraper
pct exec "$CTID" -- chmod +x /usr/local/bin/scraper
pct exec "$CTID" -- ln -sf /usr/local/bin/scraper /usr/bin/scraper
rm -f /tmp/scraper_helper_${CTID}.sh

msg_ok "Configuration written"

# ── Done! ─────────────────────────────────────────────────────────────────
VPN_STATUS=""
if [[ -n "$NORD_TOKEN" ]]; then
    VPN_STATUS=$(pct exec "$CTID" -- nordvpn status 2>/dev/null | grep -E "Status|Server|IP" | head -3 || echo "  Connected")
fi

echo ""
echo -e " ${GN}══════════════════════════════════════════════════════${CL}"
echo -e " ${GN}  ${APP} — Container $CTID Ready!${CL}"
echo -e " ${GN}══════════════════════════════════════════════════════${CL}"
echo ""
echo -e "   ${BL}Container:${CL}   $CTID ($HOSTNAME)"
echo -e "   ${BL}IP Address:${CL}  $CT_IP"
echo -e "   ${BL}VPN Server:${CL}  $NORD_SERVER"
echo -e "   ${BL}Images:${CL}      $NAS_SHARE → $SHARE_MOUNT"
if [[ -n "$VPN_STATUS" ]]; then
    echo -e "   ${BL}VPN Status:${CL}"
    echo "$VPN_STATUS" | sed 's/^/                 /'
fi
echo ""

if [[ -z "$NORD_TOKEN" ]]; then
    echo -e "   ${YW}⚠  NordVPN login still needed:${CL}"
    echo -e "      pct exec $CTID -- nordvpn login --token YOUR_TOKEN"
    echo -e "      pct exec $CTID -- nordvpn connect $NORD_SERVER"
    echo ""
fi

echo -e "   ${GN}Quick start:${CL}"
echo -e "      pct enter $CTID"
echo -e "      scraper --stats"
echo -e "      scraper --phase 4 --sport football"
echo ""
echo -e "   ${GN}Run from Proxmox host:${CL}"
echo -e "      pct exec $CTID -- scraper --phase 4 --sport football"
echo ""
