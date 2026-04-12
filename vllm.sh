#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  vllm.sh — Qwen3.5-122B-A10B/35B Hybrid INT4+FP8 on ASUS GX10
#  Pipeline v2: ~51/~112 tok/s | MTP-2 | FlashInfer 0.6.7
#  Reference: github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
# ═══════════════════════════════════════════════════════════════════

set -e

# ─── Default paths ───────────────────────────────────────────────
MODELS_DIR="$HOME/models/vllm"
REPO_DIR="$HOME/GX10-QWEN3.5-SPEED-HACK"
REPO_URL="https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4.git"
CONTAINER_NAME="vllm-qwen35"
HEALTH_URL="http://127.0.0.1:8000/health"

# ─── Colors ──────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }
step()  { echo -e "\n${BOLD}${CYAN}═══ $1 ═══${NC}"; }

# ─── Environment check ───────────────────────────────────────────
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

# ─── Main menu ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=== vLLM Manager for ASUS GX10 ===${NC}"
echo "  1. First-time setup (clone repo + build Docker + download model)"
echo "  2. Select model and start server"
echo "  3. Stop server"
echo "  4. View logs"
echo "  5. Run benchmark"
echo "  6. Rebuild Docker image (--no-cache)"
echo ""
read -p "Select (1-6): " MENU_CHOICE

# ═══════════════════════════════════════════════════════════════════
# OPTION 1: FIRST-TIME SETUP
# ═══════════════════════════════════════════════════════════════════
if [ "$MENU_CHOICE" = "1" ]; then
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
    echo "  2. Qwen3.5-35B-A3B Hybrid   (~112 tok/s | ~18GB int4 + FP8 download)"
    echo "  3. Custom model              (enter INT4 AutoRound + FP8 repo of any)"
    echo "  4. Both (1+2)"
    echo ""
    read -p "Select (1-4): " INSTALL_CHOICE

    case "$INSTALL_CHOICE" in
        1) DO_122B=true;  DO_HYBRID=false ;;
        2) DO_122B=false; DO_HYBRID=true;  HYBRID_PRESET="35b" ;;
        3) DO_122B=false; DO_HYBRID=true;  HYBRID_PRESET="custom" ;;
        4) DO_122B=true;  DO_HYBRID=true;  HYBRID_PRESET="35b" ;;
        *) error "Invalid selection"; exit 1 ;;
    esac

    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

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
            hf download "$INT4_REPO"
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
            _int4_slug=$(echo "$INT4_REPO" | tr '/' '--' | sed 's/^/models--/')
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
                python "$REPO_DIR/patches/02-mtp-speculative/add-mtp-from-fp8.py" \
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

    info "Installation complete!"
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# OPTION 6: REBUILD
# ═══════════════════════════════════════════════════════════════════
if [ "$MENU_CHOICE" = "6" ]; then
    step "Rebuild Docker image"
    if [ ! -d "$REPO_DIR" ]; then
        error "Not installed. Run option 1 first."
        exit 1
    fi
    cd "$REPO_DIR"
    warn "Rebuilding with --no-cache, takes ~60-90 minutes..."
    ./install.sh --no-cache
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# OPTION 3: STOP SERVER
# ═══════════════════════════════════════════════════════════════════
if [ "$MENU_CHOICE" = "3" ]; then
    step "Stop server"
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker stop "$CONTAINER_NAME" && docker rm "$CONTAINER_NAME"
        info "Container stopped: $CONTAINER_NAME"
    else
        warn "Container $CONTAINER_NAME is not running"
    fi
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# OPTION 4: VIEW LOGS
# ═══════════════════════════════════════════════════════════════════
if [ "$MENU_CHOICE" = "4" ]; then
    step "Logs"
    if docker ps -q -f name="$CONTAINER_NAME" | grep -q .; then
        docker logs "$CONTAINER_NAME" --tail 50 -f
    else
        warn "Container not running. Showing last logs:"
        docker logs "$CONTAINER_NAME" --tail 50 2>/dev/null || echo "No logs available"
    fi
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# OPTION 5: BENCHMARK
# ═══════════════════════════════════════════════════════════════════
if [ "$MENU_CHOICE" = "5" ]; then
    step "Benchmark"

    if ! curl -sf "$HEALTH_URL" &>/dev/null; then
        error "Server not running or not ready at $HEALTH_URL"
        exit 1
    fi

    # Auto-detect model name from API
    BENCH_MODEL=$(curl -sf http://localhost:8000/v1/models \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")
    if [ -z "$BENCH_MODEL" ]; then
        error "Could not retrieve model name from /v1/models"
        exit 1
    fi
    info "Model: $BENCH_MODEL"

    API="http://localhost:8000/v1/chat/completions"
    TMPDIR_BENCH=$(mktemp -d)

    # Benchmark a single request, write result to file
    _bench_one() {
        local label="$1" prompt="$2" max_tokens="$3" outfile="$4"
        local start end elapsed completion_tokens tps
        start=$(date +%s%3N)
        local resp
        resp=$(curl -sf "$API" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$BENCH_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":$max_tokens,\"temperature\":0.0}" 2>/dev/null)
        end=$(date +%s%3N)
        elapsed=$(( end - start ))
        completion_tokens=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['usage']['completion_tokens'])" 2>/dev/null || echo "0")
        if [ -z "$completion_tokens" ] || [ "$completion_tokens" = "0" ]; then
            echo "FAIL $label" > "$outfile"
        else
            tps=$(echo "scale=1; $completion_tokens * 1000 / $elapsed" | bc 2>/dev/null || echo "N/A")
            echo "OK $label $completion_tokens $elapsed $tps" > "$outfile"
        fi
    }

    _print_result() {
        local outfile="$1"
        local res; res=$(cat "$outfile")
        local status; status=$(echo "$res" | cut -d' ' -f1)
        if [ "$status" = "OK" ]; then
            local label tokens elapsed tps
            label=$(echo "$res" | cut -d' ' -f2)
            tokens=$(echo "$res" | cut -d' ' -f3)
            elapsed=$(echo "$res" | cut -d' ' -f4)
            tps=$(echo "$res" | cut -d' ' -f5)
            local secs; secs=$(echo "scale=2; $elapsed / 1000" | bc)
            printf "  [%-10s] %5s tokens in %6ss = ${GREEN}%s tok/s${NC}\n" "$label" "$tokens" "$secs" "$tps"
        else
            local label; label=$(echo "$res" | cut -d' ' -f2)
            printf "  [%-10s] ${RED}FAILED${NC}\n" "$label"
        fi
    }

    echo ""
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  Benchmark: $(basename "$BENCH_MODEL")  —  $(date '+%Y-%m-%d %H:%M')"
    echo "╚══════════════════════════════════════════════════════╝"

    # Warm-up
    echo ""
    echo -n "  Warm-up... "
    curl -sf "$API" -H "Content-Type: application/json" \
        -d "{\"model\":\"$BENCH_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":16,\"temperature\":0}" > /dev/null
    echo "done"

    # ── Sequential benchmark (single request) ────────────────────
    echo ""
    echo "── Sequential (1 request) ──────────────────────────────"
    for RUN in 1 2; do
        echo "  Run $RUN/2:"
        for task in \
            "Q&A|What are the main differences between TCP and UDP? Be concise.|256" \
            "Code|Write a Python binary search function with type hints and docstring.|512" \
            "JSON|Generate a JSON array of 10 fictional employees: name,age,dept,salary,skills. Output ONLY valid JSON.|1024" \
            "Math|What is 7823 * 4519? Show only the answer.|32" \
            "LongCode|Write a complete Python red-black tree with insert, delete, search, in-order traversal.|2048"
        do
            label=$(echo "$task" | cut -d'|' -f1)
            prompt=$(echo "$task" | cut -d'|' -f2)
            maxtok=$(echo "$task" | cut -d'|' -f3)
            outf="$TMPDIR_BENCH/seq_${RUN}_${label}"
            _bench_one "$label" "$prompt" "$maxtok" "$outf"
            _print_result "$outf"
        done
        echo ""
    done

    # ── Concurrent benchmark (N requests in parallel) ───────────────
    echo "── Concurrent (4 parallel requests) ───────────────────────────"
    echo "  Sending 4 requests simultaneously, measuring total throughput..."
    echo ""

    PROMPT_CONC="Write a complete Python implementation of a REST API using FastAPI with full CRUD operations, authentication middleware, and database models."
    MAXTOK_CONC=1024

    _conc_start=$(date +%s%3N)
    for i in 1 2 3 4; do
        outf="$TMPDIR_BENCH/conc_$i"
        _bench_one "req$i" "$PROMPT_CONC" "$MAXTOK_CONC" "$outf" &
    done
    wait
    _conc_end=$(date +%s%3N)

    _conc_elapsed=$(( _conc_end - _conc_start ))
    _conc_total_tokens=0
    _conc_ok=0
    for i in 1 2 3 4; do
        outf="$TMPDIR_BENCH/conc_$i"
        res=$(cat "$outf")
        if [ "$(echo "$res" | cut -d' ' -f1)" = "OK" ]; then
            tok=$(echo "$res" | cut -d' ' -f3)
            elapsed_i=$(echo "$res" | cut -d' ' -f4)
            tps_i=$(echo "$res" | cut -d' ' -f5)
            _conc_total_tokens=$(( _conc_total_tokens + tok ))
            _conc_ok=$(( _conc_ok + 1 ))
            printf "  [req%-2s] %5s tokens = %s tok/s (end-to-end)\n" "$i" "$tok" "$tps_i"
        else
            printf "  [req%-2s] FAILED\n" "$i"
        fi
    done

    if [ "$_conc_ok" -gt 0 ]; then
        _conc_wall=$(echo "scale=2; $_conc_elapsed / 1000" | bc)
        _conc_total_tps=$(echo "scale=1; $_conc_total_tokens * 1000 / $_conc_elapsed" | bc 2>/dev/null || echo "N/A")
        echo ""
        printf "  Total: %s tokens in %ss\n" "$_conc_total_tokens" "$_conc_wall"
        printf "  ${GREEN}Total throughput: %s tok/s${NC} (%s requests completed)\n" "$_conc_total_tps" "$_conc_ok"
    fi

    rm -rf "$TMPDIR_BENCH"
    echo ""
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════
# OPTION 2: SELECT MODEL AND START
# ═══════════════════════════════════════════════════════════════════
if [ "$MENU_CHOICE" = "2" ]; then

    # ─── Find model ────────────────────────────────────────────────
    step "Scanning models"

    HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"

    # Scan all hybrid models in ~/models/
    # - 122B: name contains "122b"
    # - Others: all folders with model.safetensors.index.json + name contains "hybrid|int4fp8"
    LOCAL_HYBRID_SMALL_LIST=()   # 35B, 27B, custom, ...
    LOCAL_HYBRID_122B=""
    for d in "$HOME/models"/*/; do
        [ -f "${d}model.safetensors.index.json" ] || continue
        dname=$(basename "$d")
        if echo "$dname" | grep -qi "122b"; then
            LOCAL_HYBRID_122B="${d%/}"
        elif echo "$dname" | grep -qi "hybrid\|int4fp8"; then
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
            USE_HYBRID=true; QUANT="autoround"; MTP_TOKENS=2; MODEL_SIZE="35B"
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
    [ "$USE_HYBRID" = true ] && echo "  Hybrid FP8   : $HYBRID_FP8"
    echo -e "${NC}"
    read -p "Continue? (Y/n): " CONFIRM
    CONFIRM=${CONFIRM:-Y}
    [[ ! "$CONFIRM" =~ ^[Yy]$ ]] && exit 0

    # ─── Flush cache before start ──────────────────────────────────
    step "Preparing system"
    warn "Flushing memory cache..."
    sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    info "Cache flushed"

    # Stop and remove old container (whether running or dead)
    if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
        warn "Removing old container..."
        docker stop "$CONTAINER_NAME" 2>/dev/null || true
        docker rm "$CONTAINER_NAME"
    fi

    # ─── Fallback: Run Docker manually ────────────────────────────
    # ─── Start Docker ──────────────────────────────────────────────
    step "Starting Docker"
    if true; then

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
        VLLM_ARGS="$VLLM_ARGS --load-format fastsafetensors"

        [ -n "$QUANT" ] && VLLM_ARGS="$VLLM_ARGS --quantization $QUANT"
        [ "${USE_FLASHINFER:-false}" = true ] && VLLM_ARGS="$VLLM_ARGS --attention-backend FLASHINFER"

        # MTP
        if [ "$MTP_TOKENS" -gt 0 ] 2>/dev/null; then
            VLLM_ARGS="$VLLM_ARGS --speculative-config '{\"method\":\"qwen3_next_mtp\",\"num_speculative_tokens\":$MTP_TOKENS}'"
        fi

        # Vision
        [ "$USE_VISION" = "no" ] || [ "$USE_VISION" = "n" ] && \
            VLLM_ARGS="$VLLM_ARGS --language-model-only"

        # HF cache mount
        HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"

        # Select Docker image: use vllm-qwen35-v2 if available (has hybrid FP8 patch)
        # vllm-qwen35-v2 uses `vllm` CLI entrypoint → needs subcommand "serve"
        # vllm/vllm-openai:latest uses python entrypoint directly → no need
        DOCKER_IMAGE="vllm/vllm-openai:latest"
        VLLM_CMD_PREFIX=""
        if [ "${USE_HYBRID:-false}" = true ] && docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1; then
            DOCKER_IMAGE="vllm-qwen35-v2"
            VLLM_CMD_PREFIX="serve"
            info "Using image vllm-qwen35-v2 (with hybrid FP8 dispatch patch)"
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
    fi

    # ─── Poll health ─────────────────────────────────────────────
    step "Waiting for model to load"
    warn "First time: ~13 min | Subsequent: ~5-7 min"
    warn "Ctrl+C to exit (container keeps running in background)"
    echo ""

    ELAPSED=0
    INTERVAL=5
    while true; do
        if curl -sf "$HEALTH_URL" &>/dev/null; then
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
    echo "  Benchmark: $0 → select option 5"
    echo "  Logs     : $0 → select option 4"

fi
