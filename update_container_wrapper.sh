#!/usr/bin/env bash
#
# Update the /usr/local/bin/scraper wrapper on existing LXC containers.
# Run from the Proxmox host:
#   bash update_container_wrapper.sh 128
#   bash update_container_wrapper.sh 128 129 130   # multiple containers
#   bash update_container_wrapper.sh all            # all running scraper containers
#

set -euo pipefail

GN="\033[1;92m"
YW="\033[33m"
RD="\033[01;31m"
CL="\033[m"

msg_ok()   { echo -e " \033[1;92m✓\033[m  ${GN}${1}${CL}"; }
msg_info() { echo -e " ...  ${YW}${1}${CL}"; }
msg_error(){ echo -e " \033[01;31m✗\033[m  ${RD}${1}${CL}"; }

if ! command -v pct &>/dev/null; then
    msg_error "This script must be run on a Proxmox VE host."
    exit 1
fi

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <CTID> [CTID2 ...] | all"
    echo ""
    echo "Updates the scraper wrapper script on existing containers."
    echo "Use 'all' to update all running containers with 'scraper' in the name."
    exit 1
fi

# Resolve container list
CTIDS=()
if [[ "$1" == "all" ]]; then
    while IFS= read -r line; do
        ctid=$(echo "$line" | awk '{print $1}')
        status=$(echo "$line" | awk '{print $2}')
        name=$(echo "$line" | awk '{print $3}')
        if [[ "$status" == "running" ]] && echo "$name" | grep -qi "scraper"; then
            CTIDS+=("$ctid")
        fi
    done < <(pct list 2>/dev/null | tail -n +2)

    if [[ ${#CTIDS[@]} -eq 0 ]]; then
        msg_error "No running scraper containers found."
        exit 1
    fi
    echo "Found ${#CTIDS[@]} scraper container(s): ${CTIDS[*]}"
else
    CTIDS=("$@")
fi

for CTID in "${CTIDS[@]}"; do
    # Check container exists and is running
    STATUS=$(pct status "$CTID" 2>/dev/null | awk '{print $2}' || true)
    if [[ -z "$STATUS" ]]; then
        msg_error "Container $CTID does not exist, skipping."
        continue
    fi
    if [[ "$STATUS" != "running" ]]; then
        msg_error "Container $CTID is $STATUS, skipping (must be running)."
        continue
    fi

    HOSTNAME=$(pct exec "$CTID" -- hostname 2>/dev/null || echo "unknown")
    msg_info "Updating container $CTID ($HOSTNAME)"

    # Detect install directory
    INSTALL_DIR=""
    for dir in /opt/scraperv3 /opt/scraperv2 /opt/scraper; do
        if pct exec "$CTID" -- test -d "$dir" 2>/dev/null; then
            INSTALL_DIR="$dir"
            break
        fi
    done

    if [[ -z "$INSTALL_DIR" ]]; then
        msg_error "  No scraper install found in $CTID, skipping."
        continue
    fi

    msg_info "  Install dir: $INSTALL_DIR"

    # Write the updated wrapper
    pct exec "$CTID" -- bash -c "cat > /usr/local/bin/scraper << 'WRAPPEREOF'
#!/bin/bash
INSTALL_DIR=\"$INSTALL_DIR\"
cd \"\$INSTALL_DIR\"
set -a; source .env; set +a
source venv/bin/activate

case \"\${1:-}\" in
    update)
        echo \"Pulling latest changes...\"
        git pull origin main
        echo \"Updating dependencies...\"
        pip install -q -r requirements.txt
        pip install -q curl-cffi>=0.7.0 scrapling>=0.2.0 redis>=5.0.0 2>/dev/null || true
        if python -c \"import camoufox\" 2>/dev/null; then
            echo \"Fetching Camoufox browser...\"
            python -m camoufox fetch 2>/dev/null || true
        fi
        echo \"Done! Scraper is up to date.\"
        ;;
    vpn)
        nordvpn status
        ;;
    cdn-test)
        python main_v3.py --cdn-test
        ;;
    queue)
        python task_queue.py stats
        ;;
    workers)
        shift
        python lxc_stats.py \"\$@\"
        ;;
    reset-no-image)
        python main_v3.py --reset-no-image
        ;;
    v2)
        shift
        python main.py \"\$@\"
        ;;
    *)
        python main_v3.py \"\$@\"
        ;;
esac
WRAPPEREOF
chmod +x /usr/local/bin/scraper"

    msg_ok "  Container $CTID ($HOSTNAME) updated"
done

echo ""
msg_ok "All done! Test with: pct exec <CTID> -- scraper update"
