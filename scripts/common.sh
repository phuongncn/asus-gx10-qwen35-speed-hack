# ─── Shared vars ─────────────────────────────────────────────────
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
