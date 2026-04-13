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
GPU_MEM_RAW=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '[:space:]')
if [[ "$GPU_MEM_RAW" =~ ^[0-9]+$ ]]; then
    info "GPU memory: ${GPU_MEM_RAW} MiB ($(( GPU_MEM_RAW / 1024 )) GB)"
    if [ "$GPU_MEM_RAW" -lt 100000 ]; then
        warn "GPU RAM < 100GB — 122B model may not load"
    fi
else
    warn "Could not read GPU memory info (${GPU_MEM_RAW:-empty})"
fi
