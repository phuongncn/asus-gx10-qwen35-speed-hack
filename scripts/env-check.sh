#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

step "Environment Check"

if ! command -v docker &>/dev/null; then
    error "Docker not installed. Install at: https://docs.docker.com/engine/install/"
    exit 1
fi
info "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"

if ! docker info &>/dev/null; then
    error "Docker daemon not running. Try: sudo systemctl start docker"
    exit 1
fi

# Check GPU
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | xargs)
[ -n "$GPU_NAME" ] && info "GPU: $GPU_NAME" || warn "Could not detect GPU"

# ─── Docker image check ──────────────────────────────────────────
_HAS_V2=false;  _HAS_SM121=false
docker image inspect vllm-qwen35-v2:latest  >/dev/null 2>&1 && _HAS_V2=true
docker image inspect vllm-sm121:latest      >/dev/null 2>&1 && _HAS_SM121=true

if $_HAS_V2 && $_HAS_SM121; then
    info "Docker images: vllm-qwen35-v2 ✓  vllm-sm121 ✓"
elif $_HAS_V2; then
    info "Docker images: vllm-qwen35-v2 ✓  vllm-sm121 ✗"
elif $_HAS_SM121; then
    info "Docker images: vllm-sm121 ✓  vllm-qwen35-v2 ✗"
else
    warn "Docker images not built yet — see option 1 in the menu below"
fi
