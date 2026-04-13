#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
#  vllm.sh — Qwen3.5-122B-A10B/35B Hybrid INT4+FP8 on ASUS GX10
#  Pipeline v2: ~51-55 tok/s | MTP-2 | FlashInfer 0.6.7
#  Reference: github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4
# ═══════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Bootstrap: if scripts/ missing, guide user ──────────────────
if [ ! -d "$SCRIPT_DIR/scripts" ]; then
    echo "scripts/ not found — please clone the full repo."
    echo "  git clone https://github.com/phuongncn/asus-gx10-qwen35-speed-hack"
    exit 1
fi

# shellcheck disable=SC1091
source "$SCRIPT_DIR/scripts/common.sh"

# ─── Environment check ───────────────────────────────────────────
bash "$SCRIPT_DIR/scripts/env-check.sh"

# ─── Check image status for menu display ─────────────────────────
_MENU_HAS_V2=false; _MENU_HAS_SM121=false
docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1 && _MENU_HAS_V2=true
docker image inspect vllm-sm121:latest     >/dev/null 2>&1 && _MENU_HAS_SM121=true

# ─── Main menu ───────────────────────────────────────────────────
echo ""
echo -e "${BOLD}=== vLLM Manager for ASUS GX10 ===${NC}"
echo ""
if ! $_MENU_HAS_V2 && ! $_MENU_HAS_SM121; then
    echo -e "  ${YELLOW}Getting started? Run option 1 first to set up the server (one-time, ~60-90 min).${NC}"
    echo ""
fi
echo "  1. First-time setup  → [122B / 35B Hybrid / 35B FP8+MTP / Custom / Both]"
echo "  2. Select model and start server"
echo "  3. Stop server"
echo "  4. View logs"
echo "  5. Run benchmark"
echo "  6. Rebuild Docker image (--no-cache)"
echo "  7. Build checkpoint    → [Hybrid INT4+FP8 / Native FP8+MTP]"
echo ""
read -p "Select (1-7): " MENU_CHOICE

case "$MENU_CHOICE" in
    1) bash "$SCRIPT_DIR/scripts/install.sh" ;;
    2) bash "$SCRIPT_DIR/scripts/start-server.sh" ;;
    3)
        step "Stop server"
        if docker ps -aq -f name="^${CONTAINER_NAME}$" | grep -q .; then
            docker stop "$CONTAINER_NAME" 2>/dev/null || true
            docker rm "$CONTAINER_NAME"
            info "Container stopped: $CONTAINER_NAME"
        else
            warn "Container $CONTAINER_NAME is not running"
        fi
        ;;
    4)
        step "Logs"
        if docker ps -q -f name="^${CONTAINER_NAME}$" | grep -q .; then
            docker logs "$CONTAINER_NAME" --tail 50 -f
        else
            warn "Container not running. Showing last logs:"
            docker logs "$CONTAINER_NAME" --tail 50 2>/dev/null || echo "No logs available"
        fi
        ;;
    5)
        _PORT=$(docker inspect "$CONTAINER_NAME" --format '{{join .Args " "}}' 2>/dev/null \
            | grep -oP '(?<=--port )\d+' | head -1)
        _PORT=${_PORT:-8000}
        bash "$SCRIPT_DIR/scripts/benchmark.sh" "$_PORT"
        ;;
    6)
        step "Rebuild Docker image"
        if [ ! -d "$REPO_DIR" ]; then
            error "Not installed. Run option 1 first."
            exit 1
        fi
        warn "This will rebuild the vllm-qwen35-v2 Docker image from scratch."
        warn "Takes 60-90 minutes. The existing image will be replaced."
        echo ""
        warn "⚠  WARNING: install.sh --no-cache DELETES existing images BEFORE rebuilding."
        warn "   If you Ctrl+C mid-build, both vllm-sm121 and vllm-qwen35-v2 will be gone"
        warn "   and the server will fall back to vllm/vllm-openai:latest (fails on GB10)."
        warn "   Do NOT interrupt this process once started."
        echo ""
        read -p "Continue? (y/N): " _CONFIRM_REBUILD
        [[ ! "$_CONFIRM_REBUILD" =~ ^[Yy]$ ]] && { info "Cancelled."; exit 0; }
        cd "$REPO_DIR"
        warn "Rebuilding with --no-cache... Do NOT Ctrl+C!"
        ./install.sh --no-cache
        ;;
    7) bash "$SCRIPT_DIR/scripts/build-hybrid.sh" ;;
    *) error "Invalid selection"; exit 1 ;;
esac
