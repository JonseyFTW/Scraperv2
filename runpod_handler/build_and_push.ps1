# ──────────────────────────────────────────────────────────────────────────
# Build and push the DINOv2 RunPod serverless handler to GHCR
#
# Usage (PowerShell):
#   .\build_and_push.ps1              # Build + push :latest
#   .\build_and_push.ps1 -Tag v2      # Build + push :v2 and :latest
#
# Prerequisites:
#   - Docker Desktop installed and running
#   - Logged in to GHCR:
#       docker login ghcr.io -u YOUR_GITHUB_USERNAME
#       (use a Personal Access Token with write:packages scope)
#
# After pushing:
#   1. Go to RunPod dashboard → Serverless → your endpoint
#   2. Click "Edit Endpoint"
#   3. Update the image to the new tag (or re-pull :latest)
#   4. Save — RunPod will deploy the new image on next cold start
# ──────────────────────────────────────────────────────────────────────────
param(
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"

$Image = "ghcr.io/jonseyftw/dinov2-cardscanner"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  DINOv2 RunPod Handler - Build & Push"       -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "  Image:   ${Image}:${Tag}"
Write-Host "  Context: ${ScriptDir}"
Write-Host ""

# Build
Write-Host "[1/3] Building Docker image..." -ForegroundColor Yellow
docker build -t "${Image}:${Tag}" $ScriptDir
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

# Also tag as :latest if a version tag was given
if ($Tag -ne "latest") {
    docker tag "${Image}:${Tag}" "${Image}:latest"
    Write-Host "  Also tagged as :latest"
}

# Push
Write-Host ""
Write-Host "[2/3] Pushing to GHCR..." -ForegroundColor Yellow
docker push "${Image}:${Tag}"
if ($LASTEXITCODE -ne 0) { throw "Docker push failed" }

if ($Tag -ne "latest") {
    docker push "${Image}:latest"
    if ($LASTEXITCODE -ne 0) { throw "Docker push (latest) failed" }
}

Write-Host ""
Write-Host "[3/3] Done!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Go to RunPod dashboard -> Serverless -> endpoint m5et95n0vtnnmv"
Write-Host "  2. Edit Endpoint -> update image tag if needed"
Write-Host "  3. New workers will use the updated image on next cold start"
Write-Host "  4. To force immediate update: scale to 0 workers, then back up"
Write-Host ""
