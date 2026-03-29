#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Build and push the DINOv2 RunPod serverless handler to GHCR
#
# Usage:
#   ./build_and_push.sh              # Build + push :latest
#   ./build_and_push.sh v2           # Build + push :v2 and :latest
#
# Prerequisites:
#   - Docker (or Podman) installed
#   - Logged in to GHCR:  docker login ghcr.io -u YOUR_GITHUB_USERNAME
#     (use a Personal Access Token with write:packages scope as the password)
#
# After pushing:
#   1. Go to RunPod dashboard → Serverless → your endpoint
#   2. Click "Edit Endpoint"
#   3. Update the image to the new tag (or re-pull :latest)
#   4. Save — RunPod will deploy the new image on next cold start
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

IMAGE="ghcr.io/jonseyftw/dinov2-cardscanner"
TAG="${1:-latest}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================="
echo "  DINOv2 RunPod Handler — Build & Push"
echo "============================================="
echo "  Image:  ${IMAGE}:${TAG}"
echo "  Context: ${SCRIPT_DIR}"
echo ""

# Build
echo "[1/3] Building Docker image..."
docker build -t "${IMAGE}:${TAG}" "${SCRIPT_DIR}"

# Also tag as :latest if a version tag was given
if [ "${TAG}" != "latest" ]; then
    docker tag "${IMAGE}:${TAG}" "${IMAGE}:latest"
    echo "  Also tagged as :latest"
fi

# Push
echo ""
echo "[2/3] Pushing to GHCR..."
docker push "${IMAGE}:${TAG}"
if [ "${TAG}" != "latest" ]; then
    docker push "${IMAGE}:latest"
fi

echo ""
echo "[3/3] Done!"
echo ""
echo "Next steps:"
echo "  1. Go to RunPod dashboard → Serverless → endpoint m5et95n0vtnnmv"
echo "  2. Edit Endpoint → update image tag if needed"
echo "  3. New workers will use the updated image on next cold start"
echo "  4. To force immediate update: scale to 0 workers, then back up"
echo ""
