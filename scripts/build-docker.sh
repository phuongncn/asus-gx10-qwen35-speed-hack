#!/bin/bash
# ───────────────────────────────────────────────────────────────────
#  build-docker.sh — build vllm-sm121 and vllm-qwen35-v2 images
#  Does NOT download any model files. Runs Steps 3 & 4 only.
#  Called by:
#    vllm.sh option 6 (Rebuild Docker)
#    install.sh FP8 native block (when Docker images are missing)
# ───────────────────────────────────────────────────────────────────
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

NO_CACHE=false
for arg in "$@"; do
    case "$arg" in
        --no-cache) NO_CACHE=true ;;
        *) error "Unknown flag: $arg"; exit 1 ;;
    esac
done

if [ ! -d "$REPO_DIR" ]; then
    error "Repo not found at $REPO_DIR. Run option 1 (First-time setup) first."
    exit 1
fi

SPARK_VLLM_DIR="$REPO_DIR/spark-vllm-docker"
SPARK_VLLM_PIN="49d6d9fefd7cd05e63af8b28e4b514e9d30d249f"

# ── --no-cache: nuke existing images and BuildKit cache ──────────────
if $NO_CACHE; then
    warn "--no-cache: removing existing Docker images and pruning BuildKit cache..."
    warn "Do NOT Ctrl+C once this starts — both images will be deleted before rebuild."
    docker rmi -f vllm-qwen35-v2:latest 2>/dev/null || true
    docker rmi -f vllm-sm121:latest     2>/dev/null || true
    docker builder prune -af >/dev/null 2>&1 || true
    info "Stale layers removed — rebuilding from scratch"
fi

# ── Step 3: build vllm-sm121 ─────────────────────────────────────────
if docker image inspect vllm-sm121:latest >/dev/null 2>&1; then
    info "vllm-sm121:latest already exists — skipping (pass --no-cache to force rebuild)"
else
    step "Build vllm-sm121 (SM121/Blackwell base image)"
    warn "First build: ~30-60 min (compiles vLLM, FlashInfer, NCCL for SM121)"

    # Clone or refresh upstream spark-vllm-docker
    if [ ! -d "${SPARK_VLLM_DIR}/.git" ]; then
        info "Cloning eugr/spark-vllm-docker into ${SPARK_VLLM_DIR}..."
        git clone https://github.com/eugr/spark-vllm-docker.git "${SPARK_VLLM_DIR}"
    else
        info "spark-vllm-docker already cloned — refreshing..."
        git -C "${SPARK_VLLM_DIR}" fetch --quiet origin
    fi

    # Pin to the exact commit used for our reference image
    git -C "${SPARK_VLLM_DIR}" -c advice.detachedHead=false checkout --force "${SPARK_VLLM_PIN}"

    # Strip upstream "TEMPORARY PATCH" RUN blocks (PR 35568, PR 38919) that
    # were force-pushed after our 2026-04-04 build and no longer apply to v0.19.0
    sed -i '/# TEMPORARY PATCH for broken FP8 kernels/,/&& rm pr35568.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"
    sed -i '/# TEMPORARY PATCH for broken compilation/,/&& rm pr38919.diff/d' \
        "${SPARK_VLLM_DIR}/Dockerfile"

    if grep -qE 'pr35568|pr38919' "${SPARK_VLLM_DIR}/Dockerfile"; then
        error "sed didn't strip the PR blocks cleanly — upstream Dockerfile may have changed shape."
        exit 1
    fi

    # Suppress CUTLASS×CUDA13 deprecation warning spam during nvcc compilation
    if ! grep -q 'NVCC_APPEND_FLAGS' "${SPARK_VLLM_DIR}/Dockerfile"; then
        sed -i '/^ENV TORCH_CUDA_ARCH_LIST=/a ENV NVCC_APPEND_FLAGS="-Xcompiler=-Wno-deprecated-declarations -diag-suppress=20012 -diag-suppress=20013 -diag-suppress=20014 -diag-suppress=20015"' \
            "${SPARK_VLLM_DIR}/Dockerfile"
    fi

    (
        cd "${SPARK_VLLM_DIR}"
        ./build-and-copy.sh -t vllm-sm121 --vllm-ref v0.19.0 --tf5 2>&1
    )

    docker image inspect vllm-sm121:latest >/dev/null 2>&1 \
        || { error "vllm-sm121:latest not found after build — something failed silently."; exit 1; }
    info "vllm-sm121 built successfully"
fi

# ── Step 4: build vllm-qwen35-v2 ──────────────────────────────────────
if docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1; then
    info "vllm-qwen35-v2:latest already exists — skipping (pass --no-cache to force rebuild)"
else
    step "Build vllm-qwen35-v2 (hybrid INT4+FP8 dispatch image)"
    warn "Thin layer on top of vllm-sm121: ~5 min"

    cd "$REPO_DIR"
    docker build -t vllm-qwen35-v2 -f docker/Dockerfile.v2 .

    docker image inspect vllm-qwen35-v2:latest >/dev/null 2>&1 \
        || { error "vllm-qwen35-v2:latest not found after build."; exit 1; }
    info "vllm-qwen35-v2 built successfully"
fi

echo ""
info "Docker images ready:"
docker images vllm-sm121     --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' 2>/dev/null | grep -v '^$' || true
docker images vllm-qwen35-v2 --format '   {{.Repository}}:{{.Tag}}   {{.Size}}' 2>/dev/null | grep -v '^$' || true
