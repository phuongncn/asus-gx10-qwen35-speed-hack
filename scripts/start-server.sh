#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# ─── Find model ────────────────────────────────────────────────
step "Scanning models"

HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

# Scan all models in ~/models/
# - 122B: name contains "122b"
# - Others: folders with model.safetensors.index.json + name contains "hybrid|int4fp8|fp8"
LOCAL_HYBRID_SMALL_LIST=()   # 35B, 27B, custom, FP8-native, ...
LOCAL_HYBRID_122B=""
for d in "$HOME/models"/*/; do
    [ -f "${d}model.safetensors.index.json" ] || continue
    dname=$(basename "$d")
    if echo "$dname" | grep -qi "122b"; then
        LOCAL_HYBRID_122B="${d%/}"
    elif echo "$dname" | grep -qi "hybrid\|int4fp8\|fp8"; then
        LOCAL_HYBRID_SMALL_LIST+=("${d%/}")
    fi
done
LOCAL_HYBRID_SMALL=""
[ "${#LOCAL_HYBRID_SMALL_LIST[@]}" -gt 0 ] && LOCAL_HYBRID_SMALL="${LOCAL_HYBRID_SMALL_LIST[0]}"
# Compat alias for code below
LOCAL_HYBRID_35B_LIST=("${LOCAL_HYBRID_SMALL_LIST[@]}")
LOCAL_HYBRID_35B="$LOCAL_HYBRID_SMALL"

# Scan HF cache
HF_HAS_35B_AR=false;   HF_HAS_35B_FP8=false;  HF_HAS_122B_AR=false
[ -d "$HF_CACHE/models--Intel--Qwen3.5-35B-A3B-int4-AutoRound" ]  && HF_HAS_35B_AR=true
[ -d "$HF_CACHE/models--Qwen--Qwen3.5-35B-A3B-FP8" ]              && HF_HAS_35B_FP8=true
[ -d "$HF_CACHE/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound" ] && HF_HAS_122B_AR=true

# Build labels
if [ "${#LOCAL_HYBRID_SMALL_LIST[@]}" -eq 0 ]; then
    L_SMALL="Hybrid (small)             — not found, run option 1 to build"
else
    _SMALL_NAMES=""
    for _p in "${LOCAL_HYBRID_SMALL_LIST[@]}"; do
        _SMALL_NAMES+="$(basename $_p), "
    done
    _SMALL_NAMES="${_SMALL_NAMES%, }"
    L_SMALL="Hybrid (small)             ✓ ${_SMALL_NAMES}"
fi

L122H="Qwen3.5-122B-A10B Hybrid v2  [~51 tok/s]  ★ — needs build"
[ -n "$LOCAL_HYBRID_122B" ] && L122H="Qwen3.5-122B-A10B Hybrid v2  [~51 tok/s]  ★ ✓ local: $(basename $LOCAL_HYBRID_122B)"

# Menu
echo ""
echo "=== Available models ==="
echo ""
echo "  ── Hybrid small (35B / 27B / custom) ───────────────────────"
echo "  1. $L_SMALL"
echo ""
echo "  ── 122B-A10B (higher quality) ──────────────────────────────"
echo "  2. $L122H"
echo ""
echo "  3. Enter model ID / path manually"
echo ""
read -p "Select model (1-3): " MODEL_CHOICE

case "$MODEL_CHOICE" in
    1)
        # 35B Hybrid — allow selection if multiple versions found
        if [ "${#LOCAL_HYBRID_35B_LIST[@]}" -gt 1 ]; then
            echo ""
            echo "=== Select 35B Hybrid version ==="
            for i in "${!LOCAL_HYBRID_35B_LIST[@]}"; do
                echo "  $((i+1)). $(basename ${LOCAL_HYBRID_35B_LIST[$i]})"
            done
            echo ""
            read -p "Select (1-${#LOCAL_HYBRID_35B_LIST[@]}): " V35
            V35=$((${V35:-1} - 1))
            MODEL_ID="${LOCAL_HYBRID_35B_LIST[$V35]:-${LOCAL_HYBRID_35B_LIST[0]}}"
            info "Using: $MODEL_ID"
        elif [ -n "$LOCAL_HYBRID_35B" ]; then
            MODEL_ID="$LOCAL_HYBRID_35B"
            info "Using local hybrid: $MODEL_ID"
        elif $HF_HAS_35B_AR; then
            AR35_DIR=$(find "$HF_CACHE/models--Intel--Qwen3.5-35B-A3B-int4-AutoRound/snapshots" \
                -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
            MODEL_ID="$AR35_DIR"
            warn "No hybrid 35B found — using pure AutoRound (~75 tok/s)"
            warn "Build hybrid: option 1 → select 35B"
        else
            MODEL_ID="Intel/Qwen3.5-35B-A3B-int4-AutoRound"
            warn "Will download Intel/Qwen3.5-35B-A3B-int4-AutoRound (~18GB)"
        fi
        # Detect if selected model is FP8-native (no INT4 merge)
        if echo "$MODEL_ID" | grep -qi "fp8" && ! echo "$MODEL_ID" | grep -qi "int4\|hybrid"; then
            USE_HYBRID=false; QUANT=""; MTP_TOKENS=2; MODEL_SIZE="35B"
            info "Detected FP8-native model — using fp8 quantization mode"
        else
            USE_HYBRID=true; QUANT="autoround"; MTP_TOKENS=2; MODEL_SIZE="35B"
        fi
        ;;
    2)
        # 122B Hybrid v2
        if [ -n "$LOCAL_HYBRID_122B" ]; then
            MODEL_ID="$LOCAL_HYBRID_122B"
            info "Using local hybrid: $MODEL_ID"
        elif $HF_HAS_122B_AR; then
            AR122_DIR=$(find "$HF_CACHE/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots" \
                -maxdepth 1 -mindepth 1 -type d 2>/dev/null | head -1)
            MODEL_ID="$AR122_DIR"
            warn "No hybrid found — using pure AutoRound (~38 tok/s). Run install.sh to build hybrid."
        else
            MODEL_ID="Intel/Qwen3.5-122B-A10B-int4-AutoRound"
            warn "Run option 1 (Install) first to auto-download and build the hybrid."
        fi
        USE_HYBRID=true; QUANT="autoround"; MTP_TOKENS=2; MODEL_SIZE="122B"
        ;;
    3)
        read -p "Enter HF model ID or local path: " MODEL_ID
        USE_HYBRID=false; QUANT=""; MTP_TOKENS=0; MODEL_SIZE="custom"
        ;;
    *)
        error "Invalid selection"; exit 1
        ;;
esac

info "Model: $MODEL_ID"

# ─── Configuration ─────────────────────────────────────────────
step "Server configuration"
echo "(Press Enter to use default values)"
echo ""

# Defaults based on model size
if [ "${MODEL_SIZE:-}" = "122B" ] && [ "${USE_HYBRID:-false}" = true ]; then
    DEF_CTX=262144; DEF_GPU_UTIL=0.90; USE_FLASHINFER=true
else
    DEF_CTX=131072; DEF_GPU_UTIL=0.92; USE_FLASHINFER=false
fi

read -p "Port [8000]: " PORT
PORT=${PORT:-8000}

read -p "Context length [$DEF_CTX]: " CTX_LEN
CTX_LEN=${CTX_LEN:-$DEF_CTX}

read -p "Max model len [$DEF_CTX]: " MAX_MODEL_LEN
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$DEF_CTX}

read -p "GPU memory utilization [$DEF_GPU_UTIL]: " GPU_MEM_UTIL
GPU_MEM_UTIL=${GPU_MEM_UTIL:-$DEF_GPU_UTIL}

# MTP
if [ "$MTP_TOKENS" -gt 0 ] 2>/dev/null; then
    read -p "MTP speculative tokens [$MTP_TOKENS]: " MTP_INPUT
    MTP_TOKENS=${MTP_INPUT:-$MTP_TOKENS}
else
    read -p "MTP speculative tokens [0]: " MTP_INPUT
    MTP_TOKENS=${MTP_INPUT:-0}
fi

read -p "Thinking mode (yes/no) [yes]: " THINKING
THINKING=${THINKING:-yes}

read -p "Vision encoder (yes/no) [no — saves RAM]: " USE_VISION
USE_VISION=${USE_VISION:-no}

read -p "Tensor parallel size [1]: " TP_SIZE
TP_SIZE=${TP_SIZE:-1}

echo ""
echo -e "${BOLD}=== Configuration ==="
echo "  Model        : $MODEL_ID"
echo "  Port         : $PORT"
echo "  Context      : $CTX_LEN"
echo "  GPU mem util : $GPU_MEM_UTIL"
echo "  MTP tokens   : $MTP_TOKENS"
echo "  Thinking     : $THINKING"
echo "  Vision       : $USE_VISION"
echo "  TP size      : $TP_SIZE"
[ "$USE_HYBRID" = true ] && echo "  Quant mode   : ${QUANT:-autoround}"
echo -e "${NC}"
read -p "Continue? (Y/n): " CONFIRM
CONFIRM=${CONFIRM:-Y}
[[ ! "$CONFIRM" =~ ^[Yy]$ ]] && exit 0

# ─── Flush cache before start ──────────────────────────────────
step "Preparing system"
read -p "Drop system memory cache before starting? (Recommended to free RAM for the model) [Y/n]: " _DO_DROP
_DO_DROP=${_DO_DROP:-Y}
if [[ "$_DO_DROP" =~ ^[Yy]$ ]]; then
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' && info "Memory cache flushed" || warn "Could not flush cache (continuing anyway)"
else
    info "Skipping cache flush"
fi

# Stop and remove old container (whether running or dead)
if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
    warn "Removing old container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME"
fi

# ─── Start Docker ──────────────────────────────────────────────
step "Starting Docker"

# Build vLLM args
VLLM_ARGS="--model $MODEL_ID"
VLLM_ARGS="$VLLM_ARGS --port $PORT"
VLLM_ARGS="$VLLM_ARGS --host 0.0.0.0"
VLLM_ARGS="$VLLM_ARGS --max-model-len $MAX_MODEL_LEN"
VLLM_ARGS="$VLLM_ARGS --gpu-memory-utilization $GPU_MEM_UTIL"
VLLM_ARGS="$VLLM_ARGS --tensor-parallel-size $TP_SIZE"
VLLM_ARGS="$VLLM_ARGS --reasoning-parser qwen3"
VLLM_ARGS="$VLLM_ARGS --enable-auto-tool-choice"
VLLM_ARGS="$VLLM_ARGS --tool-call-parser qwen3_coder"

[ -n "$QUANT" ] && VLLM_ARGS="$VLLM_ARGS --quantization $QUANT"
[ "${USE_FLASHINFER:-false}" = true ] && VLLM_ARGS="$VLLM_ARGS --attention-backend FLASHINFER"

# MTP
if [ "$MTP_TOKENS" -gt 0 ] 2>/dev/null; then
    VLLM_ARGS="$VLLM_ARGS --speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$MTP_TOKENS}'"
fi

# Vision
if [ "$USE_VISION" = "no" ] || [ "$USE_VISION" = "n" ]; then
    VLLM_ARGS="$VLLM_ARGS --language-model-only"
fi

# HF cache mount
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

# Select Docker image: use vllm-qwen35-v2 if available (has hybrid FP8 patch)
# Image selection:
#   vllm-qwen35-v2  → hybrid INT4+FP8 (has autoround dispatch patch), SM121
#   vllm-sm121      → FP8 native or any model needing SM121 (Blackwell/GB10)
#   vllm/vllm-openai:latest → fallback, compiled for SM8.x/9.0 (NOT for GB10)
# Both vllm-qwen35-v2 and vllm-sm121 use `vllm serve` CLI entrypoint
DOCKER_IMAGE="vllm/vllm-openai:latest"
VLLM_CMD_PREFIX=""
if [ "${USE_HYBRID:-false}" = true ] && [ "${QUANT:-}" = "autoround" ] && docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1; then
    DOCKER_IMAGE="vllm-qwen35-v2"
    VLLM_CMD_PREFIX="serve"
    VLLM_ARGS="$VLLM_ARGS --load-format fastsafetensors"
    info "Using image vllm-qwen35-v2 (with hybrid FP8 dispatch patch, fastsafetensors)"
elif docker image inspect vllm-sm121:latest >/dev/null 2>&1; then
    DOCKER_IMAGE="vllm-sm121"
    VLLM_CMD_PREFIX="vllm serve"
    VLLM_ARGS="$VLLM_ARGS --load-format fastsafetensors"
    info "Using image vllm-sm121 (SM121/Blackwell native, fastsafetensors)"
else
    warn "vllm-sm121 not found — falling back to vllm/vllm-openai:latest (may fail on GB10)"
fi

# Mount local model path if MODEL_ID is a local path (starts with /)
LOCAL_MODEL_MOUNT=""
MODEL_ID_DOCKER="$MODEL_ID"
if [[ "$MODEL_ID" == /* ]]; then
    MODEL_PARENT=$(dirname "$MODEL_ID")
    MODEL_BASENAME=$(basename "$MODEL_ID")
    LOCAL_MODEL_MOUNT="-v ${MODEL_PARENT}:/local_models"
    MODEL_ID_DOCKER="/local_models/${MODEL_BASENAME}"
    VLLM_ARGS="--model $MODEL_ID_DOCKER ${VLLM_ARGS#--model $MODEL_ID}"
fi

DOCKER_CMD="docker run -d \
    --name $CONTAINER_NAME \
    --gpus all \
    --net=host \
    --ipc=host \
    --shm-size=16g \
    -v $HF_CACHE_DIR:/root/.cache/huggingface \
    ${LOCAL_MODEL_MOUNT} \
    -e HF_TOKEN=${HF_TOKEN:-} \
    -e VLLM_UF_EAGER_ALLREDUCE=1 \
    $DOCKER_IMAGE \
    $VLLM_CMD_PREFIX $VLLM_ARGS"

echo ""
echo "Docker command:"
echo "$DOCKER_CMD"
echo ""

eval "$DOCKER_CMD"
info "Container started: $CONTAINER_NAME"

# ─── Poll health ─────────────────────────────────────────────
step "Waiting for model to load"
warn "First time: ~13 min | Subsequent: ~5-7 min"
warn "Ctrl+C to exit (container keeps running in background)"
echo ""

_HEALTH_URL="http://127.0.0.1:$PORT/health"
ELAPSED=0
INTERVAL=5
while true; do
    if curl -sf "$_HEALTH_URL" &>/dev/null; then
        echo ""
        info "Server ready! Endpoint: http://localhost:$PORT/v1"
        break
    fi
    # Check if container is still alive
    if ! docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        echo ""
        error "Container crashed. Check logs:"
        docker logs "$CONTAINER_NAME" --tail 30 2>/dev/null || true
        exit 1
    fi
    ELAPSED=$((ELAPSED + INTERVAL))
    MINS=$((ELAPSED / 60))
    SECS=$((ELAPSED % 60))
    printf "\r  [%02d:%02d] Loading..." "$MINS" "$SECS"
    sleep $INTERVAL
done

echo ""
echo -e "${BOLD}${GREEN}✓ Ready!${NC}"
echo ""
echo "  API  : http://localhost:$PORT/v1"
echo "  Chat : curl http://localhost:$PORT/v1/chat/completions \\"
echo "           -H 'Content-Type: application/json' \\"
echo "           -d '{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":100}'"
echo ""
echo "  Benchmark: bash scripts/benchmark.sh $PORT"
echo "  Logs     : docker logs $CONTAINER_NAME --tail 50 -f"
