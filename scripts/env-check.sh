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

# ─── Docker image check ──────────────────────────────────────────
_HAS_V2=false;  _HAS_SM121=false
docker image inspect vllm-qwen35-v2:latest  >/dev/null 2>&1 && _HAS_V2=true
docker image inspect vllm-sm121:latest      >/dev/null 2>&1 && _HAS_SM121=true

if $_HAS_V2 && $_HAS_SM121; then
    info "Docker images: vllm-qwen35-v2 ✓  vllm-sm121 ✓"
elif $_HAS_V2; then
    info  "Docker images: vllm-qwen35-v2 ✓"
    warn  "vllm-sm121 not found — FP8-native models will use vllm/vllm-openai:latest (may fail on GB10)"
    warn  "Rebuild: option 1 → install 122B, or option 6 to rebuild images"
elif $_HAS_SM121; then
    info  "Docker images: vllm-sm121 ✓"
    warn  "vllm-qwen35-v2 not found — Hybrid INT4+FP8 models will use vllm-sm121 (no autoround patch)"
    warn  "Rebuild: option 6 to rebuild images"
else
    warn  "Neither vllm-qwen35-v2 nor vllm-sm121 found!"
    warn  "Will fall back to vllm/vllm-openai:latest — this WILL FAIL on GB10 (no SM121 support)"
    warn  "Fix: run option 1 (First-time setup) or option 6 (Rebuild Docker image)"
fi
