#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

step "Setup"

# ── Clone repo if not exists ─────────────────────────────────────
if [ -d "$REPO_DIR" ]; then
    read -p "Pull latest repo update? (y/N): " DO_PULL
    [[ "$DO_PULL" =~ ^[Yy]$ ]] && git -C "$REPO_DIR" pull
else
    info "Cloning repo..."
    git clone "$REPO_URL" "$REPO_DIR"
fi

# ── Headless suggestion ─────────────────────────────────────────
CURRENT_TARGET=$(systemctl get-default 2>/dev/null || echo "unknown")
if [ "$CURRENT_TARGET" != "multi-user.target" ]; then
    read -p "Switch to headless mode (saves RAM)? (Y/n): " DO_HEADLESS
    DO_HEADLESS=${DO_HEADLESS:-Y}
    if [[ "$DO_HEADLESS" =~ ^[Yy]$ ]]; then
        sudo systemctl set-default multi-user.target
        info "Headless mode set. Reboot required to take full effect."
    fi
fi

# ── HuggingFace token ────────────────────────────────────────────
if [ -z "${HF_TOKEN:-}" ]; then
    read -p "HF_TOKEN (press enter to skip if already logged in): " HF_TOKEN
fi
[ -n "${HF_TOKEN:-}" ] && export HF_TOKEN

# ── Check installation status ────────────────────────────────────
HYBRID_122B="$HOME/models/qwen35-122b-hybrid-int4fp8"
HAVE_122B=false; HAVE_DOCKER=false
[ -f "$HYBRID_122B/model.safetensors.index.json" ] && HAVE_122B=true
docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1 && HAVE_DOCKER=true

# ── Model selection menu ─────────────────────────────────────────
echo ""
echo "=== Select model to install ==="
echo ""
L122="1. Qwen3.5-122B-A10B Hybrid  (~51 tok/s  | ~75GB download + ~71GB output)"
$HAVE_122B && $HAVE_DOCKER && L122="1. Qwen3.5-122B-A10B Hybrid  (~51 tok/s)  ✓ already fully installed"
echo "  $L122"
echo "  2. Qwen3.5-35B-A3B Hybrid   (~112 tok/s | best speed, INT4+FP8 merged)"
echo "  3. Qwen3.5-35B-A3B FP8+MTP  (better quality, no INT4, ~35GB download)"
echo "  4. Custom model              (enter INT4 AutoRound + FP8 repo of any)"
echo "  5. Both (1+2)"
echo ""
read -p "Select (1-5): " INSTALL_CHOICE

case "$INSTALL_CHOICE" in
    1) DO_122B=true;  DO_HYBRID=false; DO_FP8_NATIVE=false ;;
    2) DO_122B=false; DO_HYBRID=true;  DO_FP8_NATIVE=false; HYBRID_PRESET="35b" ;;
    3) DO_122B=false; DO_HYBRID=false; DO_FP8_NATIVE=true ;;
    4) DO_122B=false; DO_HYBRID=true;  DO_FP8_NATIVE=false; HYBRID_PRESET="custom" ;;
    5) DO_122B=true;  DO_HYBRID=true;  DO_FP8_NATIVE=false; HYBRID_PRESET="35b" ;;
    *) error "Invalid selection"; exit 1 ;;
esac

read -p "Drop system memory cache before install? (Recommended to free RAM) [Y/n]: " _DO_DROP
_DO_DROP=${_DO_DROP:-Y}
if [[ "$_DO_DROP" =~ ^[Yy]$ ]]; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && info "Memory cache flushed" || warn "Could not flush cache (continuing anyway)"
fi

# ════════════════════════════════════════════════════════════════
# 122B — use original install.sh (idempotent, skips already done steps)
# ════════════════════════════════════════════════════════════════
if ${DO_122B:-false}; then
    if $HAVE_122B && $HAVE_DOCKER; then
        info "122B already fully installed — skipping"
    else
        info "Starting 122B install (build Docker + download model ~100GB)..."
        warn "Takes 30-90 minutes depending on network and processing speed"
        cd "$REPO_DIR" && ./install.sh --no-launch
        info "122B installation complete!"
    fi
fi

# ════════════════════════════════════════════════════════════════
# HYBRID BUILD — shared for 35B preset and custom model
# ════════════════════════════════════════════════════════════════
if ${DO_HYBRID:-false}; then

    # ── Setup venv ───────────────────────────────────────────────
    cd "$REPO_DIR"
    if [ ! -d .venv ]; then python3 -m venv .venv; fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -q -U pip
    pip install -q torch numpy safetensors huggingface_hub

    HF_CACHE_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"

    # ── Ask build info ────────────────────────────────────────────
    echo ""
    if [ "$HYBRID_PRESET" = "35b" ]; then
        step "Install Hybrid: Qwen3.5-35B-A3B"
        DEF_INT4="Intel/Qwen3.5-35B-A3B-int4-AutoRound"
        DEF_FP8="Qwen/Qwen3.5-35B-A3B-FP8"
        DEF_OUT="qwen35-35b-hybrid-int4fp8"
        DEF_MTP="y"
    else
        step "Install Hybrid: Custom Model"
        echo "  Example INT4: Intel/Qwen3.5-27B-int4-AutoRound"
        echo "                Intel/gemma-4-26B-A4B-it-int4-mixed-AutoRound"
        echo "  Example FP8 : Qwen/Qwen3.5-27B-FP8"
        echo ""
        DEF_INT4=""
        DEF_FP8=""
        DEF_OUT="custom-hybrid-int4fp8"
        DEF_MTP="n"
    fi

    read -p "INT4 AutoRound HF repo [${DEF_INT4:-required}]: " INT4_REPO
    INT4_REPO=${INT4_REPO:-$DEF_INT4}
    [ -z "$INT4_REPO" ] && { error "INT4 repo cannot be empty"; exit 1; }

    read -p "FP8 HF repo [${DEF_FP8:-required}]: " FP8_REPO
    FP8_REPO=${FP8_REPO:-$DEF_FP8}
    [ -z "$FP8_REPO" ] && { error "FP8 repo cannot be empty"; exit 1; }

    echo ""
    echo "  Output folder name in ~/models/"
    read -p "Output name [$DEF_OUT]: " OUT_NAME
    OUT_NAME=${OUT_NAME:-$DEF_OUT}
    HYBRID_OUT="$HOME/models/$OUT_NAME"

    info "INT4 source : $INT4_REPO"
    info "FP8  source : $FP8_REPO"
    info "Output      : $HYBRID_OUT"

    # ── Check if output already exists ─────────────────────────────
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

    # ── Download INT4 and build ────────────────────────────────────
    INTEL_DIR=""
    if ! $SKIP_BUILD; then
        info "Downloading INT4 AutoRound: $INT4_REPO..."
        INTEL_DIR=$(hf download "$INT4_REPO" --quiet)
        [ -d "$INTEL_DIR" ] || { error "INT4 dir not found after download"; exit 1; }
        info "INT4 dir: $INTEL_DIR"

        info "Building hybrid checkpoint (may take 20-60 minutes)..."
        python "$REPO_DIR/patches/01-hybrid-int4-fp8/build-hybrid-checkpoint.py" \
            --gptq-dir "$INTEL_DIR" \
            --fp8-repo "$FP8_REPO" \
            --output "$HYBRID_OUT" \
            --force
        info "Hybrid checkpoint done: $HYBRID_OUT"
    else
        # Find INT4 cache for MTP
        _int4_slug=$(echo "$INT4_REPO" | sed 's|/|--|g' | sed 's/^/models--/')
        INTEL_DIR=$(find "$HF_CACHE_HUB/$_int4_slug/snapshots" \
            -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1 || echo "")
    fi

    # ── MTP speculative weights (only supports Qwen3.5) ───────────
    echo ""
    if [ "$HYBRID_PRESET" = "custom" ]; then
        warn "MTP speculative decoding: add-mtp-weights.py script is only tested with Qwen3.5."
        warn "Other models may encounter errors or see no effect."
    fi
    read -p "Add MTP speculative decoding weights? (${DEF_MTP}/$([ "$DEF_MTP" = "y" ] && echo N || echo Y)): " DO_MTP
    DO_MTP=${DO_MTP:-$DEF_MTP}

    if [[ "$DO_MTP" =~ ^[Yy]$ ]]; then
        if [ -f "$HYBRID_OUT/model_extra_tensors.safetensors" ] \
           && grep -q '"mtp\.' "$HYBRID_OUT/model.safetensors.index.json" 2>/dev/null; then
            info "MTP weights already present — skipping"
        elif [ -n "$INTEL_DIR" ] && [ -d "$INTEL_DIR" ] \
           && [ -f "$INTEL_DIR/model_extra_tensors.safetensors" ]; then
            # Old way: Intel model has separate file (35B, 122B)
            info "Adding MTP weights from Intel INT4 source..."
            python "$REPO_DIR/patches/02-mtp-speculative/add-mtp-weights.py" \
                --source "$INTEL_DIR" \
                --target "$HYBRID_OUT"
            info "MTP done"
        else
            # New way: extract MTP from FP8 source (27B, custom model)
            warn "Intel INT4 has no model_extra_tensors — extracting from FP8 source..."
            python "$SCRIPT_DIR/add-mtp-from-fp8.py" \
                --fp8-repo "$FP8_REPO" \
                --target "$HYBRID_OUT"
            info "MTP done (from FP8 source)"
        fi
    else
        info "Skipping MTP"
    fi

    # ── Docker image ─────────────────────────────────────────────
    if $HAVE_DOCKER || docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1; then
        info "Docker image vllm-qwen35-v2 already exists — skipping build"
    else
        warn "Docker image not found. You need to install 122B first to build the image."
        warn "Run option 1 → select 1 (122B) first."
    fi

    info "$OUT_NAME ready! Run option 2 → select model to start."
fi

# ════════════════════════════════════════════════════════════════
# FP8 NATIVE + MTP — download FP8 directly, no INT4 needed
# ════════════════════════════════════════════════════════════════
if ${DO_FP8_NATIVE:-false}; then
    step "Install FP8 native + MTP: Qwen3.5-35B-A3B"

    cd "$REPO_DIR"
    if [ ! -d .venv ]; then python3 -m venv .venv; fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    pip install -q -U pip
    pip install -q torch numpy safetensors huggingface_hub

    DEF_FP8_NATIVE="Qwen/Qwen3.5-35B-A3B-FP8"
    DEF_OUT_FP8="qwen35-35b-fp8-mtp"

    echo ""
    read -p "FP8 HF repo [$DEF_FP8_NATIVE]: " FP8_NATIVE_REPO
    FP8_NATIVE_REPO=${FP8_NATIVE_REPO:-$DEF_FP8_NATIVE}

    read -p "Output name in ~/models/ [$DEF_OUT_FP8]: " OUT_NAME_FP8
    OUT_NAME_FP8=${OUT_NAME_FP8:-$DEF_OUT_FP8}
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
            info "Skipping build — keeping $OUT_NAME_FP8"
            SKIP_FP8_BUILD=true
        fi
    fi

    if ! $SKIP_FP8_BUILD; then
        info "Downloading FP8 model to local dir (~35GB, takes a while)..."
        mkdir -p "$FP8_NATIVE_OUT"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$FP8_NATIVE_REPO', local_dir='$FP8_NATIVE_OUT', local_dir_use_symlinks=False)
print('Download complete')
"
        [ -f "$FP8_NATIVE_OUT/model.safetensors.index.json" ] \
            || { error "FP8 download failed — index file not found"; exit 1; }
        info "FP8 model downloaded: $FP8_NATIVE_OUT"

        info "Checking MTP speculative decoding weights..."
        _mtp_count=$(python3 -c "
import json, sys
with open('$FP8_NATIVE_OUT/model.safetensors.index.json') as f:
    idx = json.load(f)
mtp = [k for k in idx['weight_map'] if 'mtp' in k.lower()]
print(len(mtp))
" 2>/dev/null || echo "0")
        if [ "$_mtp_count" -gt 0 ] 2>/dev/null; then
            info "MTP weights already embedded in FP8 model ($_mtp_count tensors) — ready for speculative decoding"
        elif [ -f "$REPO_DIR/patches/02-mtp-speculative/add-mtp-weights.py" ] \
           && [ -f "$FP8_NATIVE_OUT/model_extra_tensors.safetensors" ]; then
            python "$REPO_DIR/patches/02-mtp-speculative/add-mtp-weights.py" \
                --source "$FP8_NATIVE_OUT" \
                --target "$FP8_NATIVE_OUT"
            info "MTP done"
        else
            warn "MTP tensors not found in index — speculative decoding may not be available"
        fi
    fi

    info "$OUT_NAME_FP8 ready! Run option 2 → select model to start."
fi

info "Installation complete!"
