#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

step "Build Checkpoint"
echo "  1. Hybrid INT4+FP8   — merge INT4 AutoRound + FP8 weights (max speed)"
echo "  2. Native FP8 + MTP  — download FP8 directly, inject MTP (full quality)"
echo ""
read -p "Select (1-2): " BUILD_MODE

# ── Setup venv ───────────────────────────────────────────────────
cd "$REPO_DIR"
if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -U pip
pip install -q torch numpy safetensors huggingface_hub

HF_CACHE_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# ════════════════════════════════════════════════════════════════
# MODE 1 — Hybrid INT4+FP8
# ════════════════════════════════════════════════════════════════
if [ "$BUILD_MODE" = "1" ]; then

    echo ""
    echo "  Enter a local path (e.g. /home/user/models/my-int4) or HF repo ID."
    echo ""

    # ── INT4 source ──────────────────────────────────────────────────
    read -p "INT4 source (local path or HF repo) [Intel/Qwen3.5-35B-A3B-int4-AutoRound]: " INT4_SRC
    INT4_SRC=${INT4_SRC:-Intel/Qwen3.5-35B-A3B-int4-AutoRound}

    # ── FP8 source ───────────────────────────────────────────────────
    read -p "FP8 HF repo [Qwen/Qwen3.5-35B-A3B-FP8]: " FP8_REPO
    FP8_REPO=${FP8_REPO:-Qwen/Qwen3.5-35B-A3B-FP8}

    # ── Output ───────────────────────────────────────────────────────
    read -p "Output name in ~/models/ [qwen35-35b-hybrid-int4fp8]: " OUT_NAME
    OUT_NAME=${OUT_NAME:-qwen35-35b-hybrid-int4fp8}
    HYBRID_OUT="$HOME/models/$OUT_NAME"

    info "INT4 source : $INT4_SRC"
    info "FP8  source : $FP8_REPO"
    info "Output      : $HYBRID_OUT"

    # ── Check if output already exists ───────────────────────────────
    SKIP_BUILD=false
    if [ -f "$HYBRID_OUT/model.safetensors.index.json" ]; then
        warn "$OUT_NAME already exists"
        read -p "Rebuild (delete and rebuild)? (y/N): " DO_REBUILD
        if [[ "$DO_REBUILD" =~ ^[Yy]$ ]]; then
            rm -rf "$HYBRID_OUT"
        else
            info "Skipping build — keeping $OUT_NAME"
            SKIP_BUILD=true
        fi
    fi

    # ── Resolve INT4 dir ─────────────────────────────────────────────
    INTEL_DIR=""
    if ! $SKIP_BUILD; then
        if [[ "$INT4_SRC" == /* ]]; then
            # Local path
            [ -d "$INT4_SRC" ] || { error "Local path not found: $INT4_SRC"; exit 1; }
            INTEL_DIR="$INT4_SRC"
            info "Using local INT4 dir: $INTEL_DIR"
        else
            # HF repo
            info "Downloading INT4 AutoRound: $INT4_SRC..."
            hf download "$INT4_SRC"
            INTEL_DIR=$(hf download "$INT4_SRC" --quiet)
            [ -d "$INTEL_DIR" ] || { error "INT4 dir not found after download"; exit 1; }
            info "INT4 dir: $INTEL_DIR"
        fi

        info "Building hybrid checkpoint (may take 20-60 minutes)..."
        python "$REPO_DIR/patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py" \
            --gptq-dir "$INTEL_DIR" \
            --fp8-repo "$FP8_REPO" \
            --output "$HYBRID_OUT" \
            --force
        info "Hybrid checkpoint done: $HYBRID_OUT"
    else
        # Resolve INTEL_DIR for MTP step even when skipping build
        if [[ "$INT4_SRC" == /* ]]; then
            INTEL_DIR="$INT4_SRC"
        else
            _int4_slug=$(echo "$INT4_SRC" | sed 's|/|--|g' | sed 's/^/models--/')
            INTEL_DIR=$(find "$HF_CACHE_HUB/$_int4_slug/snapshots" \
                -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || echo "")
        fi
    fi

    # ── MTP speculative weights ───────────────────────────────────────
    echo ""
    read -p "Add MTP speculative decoding weights? (Y/n): " DO_MTP
    DO_MTP=${DO_MTP:-Y}

    if [[ "$DO_MTP" =~ ^[Yy]$ ]]; then
        if [ -f "$HYBRID_OUT/model_extra_tensors.safetensors" ] \
           && grep -q '"mtp\.' "$HYBRID_OUT/model.safetensors.index.json" 2>/dev/null; then
            info "MTP weights already present — skipping"
        elif [ -n "$INTEL_DIR" ] && [ -d "$INTEL_DIR" ] \
           && [ -f "$INTEL_DIR/model_extra_tensors.safetensors" ]; then
            info "Adding MTP weights from INT4 source..."
            python "$REPO_DIR/patches/02-mtp-speculative/add-mtp-weights.py" \
                --source "$INTEL_DIR" \
                --target "$HYBRID_OUT"
            info "MTP done"
        else
            warn "INT4 source has no model_extra_tensors — extracting from FP8 source..."
            python "$SCRIPT_DIR/add-mtp-from-fp8.py" \
                --fp8-repo "$FP8_REPO" \
                --target "$HYBRID_OUT"
            info "MTP done (from FP8 source)"
        fi
    else
        info "Skipping MTP"
    fi

    echo ""
    info "Done! Hybrid checkpoint at: $HYBRID_OUT"

# ════════════════════════════════════════════════════════════════
# MODE 2 — Native FP8 + MTP
# ════════════════════════════════════════════════════════════════
elif [ "$BUILD_MODE" = "2" ]; then

    echo ""
    read -p "FP8 HF repo [Qwen/Qwen3.5-35B-A3B-FP8]: " FP8_NATIVE_REPO
    FP8_NATIVE_REPO=${FP8_NATIVE_REPO:-Qwen/Qwen3.5-35B-A3B-FP8}

    read -p "Output name in ~/models/ [qwen35-35b-fp8-mtp]: " OUT_NAME_FP8
    OUT_NAME_FP8=${OUT_NAME_FP8:-qwen35-35b-fp8-mtp}
    FP8_NATIVE_OUT="$HOME/models/$OUT_NAME_FP8"

    info "FP8 source : $FP8_NATIVE_REPO"
    info "Output     : $FP8_NATIVE_OUT"

    SKIP_FP8_BUILD=false
    if [ -f "$FP8_NATIVE_OUT/model.safetensors.index.json" ]; then
        warn "$OUT_NAME_FP8 already exists"
        read -p "Rebuild (delete and rebuild)? (y/N): " DO_REBUILD_FP8
        if [[ "$DO_REBUILD_FP8" =~ ^[Yy]$ ]]; then
            rm -rf "$FP8_NATIVE_OUT"
        else
            info "Skipping download — keeping $OUT_NAME_FP8"
            SKIP_FP8_BUILD=true
        fi
    fi

    if ! $SKIP_FP8_BUILD; then
        info "Downloading FP8 model (~35GB, takes a while)..."
        mkdir -p "$FP8_NATIVE_OUT"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$FP8_NATIVE_REPO', local_dir='$FP8_NATIVE_OUT', local_dir_use_symlinks=False)
print('Download complete')
"
        [ -f "$FP8_NATIVE_OUT/model.safetensors.index.json" ] \
            || { error "FP8 download failed — index file not found"; exit 1; }
        info "FP8 model downloaded: $FP8_NATIVE_OUT"
    fi

    # ── MTP ──────────────────────────────────────────────────────────
    info "Checking MTP speculative decoding weights..."
    _mtp_count=$(python3 -c "
import json, sys
with open('$FP8_NATIVE_OUT/model.safetensors.index.json') as f:
    idx = json.load(f)
mtp = [k for k in idx['weight_map'] if 'mtp' in k.lower()]
print(len(mtp))
" 2>/dev/null || echo "0")

    if [ "$_mtp_count" -gt 0 ] 2>/dev/null; then
        info "MTP weights already embedded ($_mtp_count tensors) — ready for speculative decoding"
    elif [ -f "$FP8_NATIVE_OUT/model_extra_tensors.safetensors" ] \
       && [ -f "$REPO_DIR/patches/02-mtp-speculative/add-mtp-weights.py" ]; then
        info "Adding MTP weights from model_extra_tensors..."
        python "$REPO_DIR/patches/02-mtp-speculative/add-mtp-weights.py" \
            --source "$FP8_NATIVE_OUT" \
            --target "$FP8_NATIVE_OUT"
        info "MTP done"
    else
        warn "MTP tensors not found — speculative decoding may not be available"
    fi

    echo ""
    info "Done! FP8+MTP checkpoint at: $FP8_NATIVE_OUT"

else
    error "Invalid selection"
    exit 1
fi

info "Run vllm.sh → option 2 to start server"
