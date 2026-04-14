#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

# Standalone usage: ./scripts/benchmark.sh [port]
PORT="${1:-8000}"
HEALTH_URL="http://127.0.0.1:$PORT/health"
API="http://localhost:$PORT/v1/chat/completions"

step "Benchmark"

if ! curl -sf "$HEALTH_URL" &>/dev/null; then
    error "Server not running or not ready at $HEALTH_URL"
    exit 1
fi

# Auto-detect model name from API
BENCH_MODEL=$(curl -sf "http://localhost:$PORT/v1/models" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null || echo "")
if [ -z "$BENCH_MODEL" ]; then
    error "Could not retrieve model name from /v1/models"
    exit 1
fi
info "Model: $BENCH_MODEL"

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
CONCURRENCY="${CONCURRENCY:-4}"
echo "── Concurrent ($CONCURRENCY parallel requests) ───────────────────────────"
echo "  Sending $CONCURRENCY requests simultaneously, measuring total throughput..."
echo ""

PROMPT_CONC="Write a complete Python implementation of a REST API using FastAPI with full CRUD operations, authentication middleware, and database models."
MAXTOK_CONC=1024

_conc_start=$(date +%s%3N)
for i in $(seq 1 "$CONCURRENCY"); do
    outf="$TMPDIR_BENCH/conc_$i"
    _bench_one "req$i" "$PROMPT_CONC" "$MAXTOK_CONC" "$outf" &
done
wait
_conc_end=$(date +%s%3N)

_conc_elapsed=$(( _conc_end - _conc_start ))
_conc_total_tokens=0
_conc_ok=0
for i in $(seq 1 "$CONCURRENCY"); do
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
