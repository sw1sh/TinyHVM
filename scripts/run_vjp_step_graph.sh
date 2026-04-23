#!/bin/bash
# Clear per-step graphs and rerun the vjp_sum_of_square test with
# step-graph tracing enabled.  Produces BOTH granularities in separate
# sibling folders:
#   graphs/grad_rules/vjp_sum_of_square/steps_fire/   — one frame per
#       IC interaction rule fire (fine-grained)
#   graphs/grad_rules/vjp_sum_of_square/steps_reduce/ — one frame per
#       top-level reduction boundary (coarse, HVM4-style)
#
# Usage:
#   ./scripts/run_vjp_step_graph.sh          # run both, list files
#   ./scripts/run_vjp_step_graph.sh trace    # stream THVM_INTERACT_TRACE
#   ./scripts/run_vjp_step_graph.sh png      # render PNGs too
#   ./scripts/run_vjp_step_graph.sh fire     # only the fire granularity
#   ./scripts/run_vjp_step_graph.sh reduce   # only the reduce granularity

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BASE="graphs/grad_rules/vjp_sum_of_square"
# Output dirs are siblings of BASE (not children), because the test's
# setup_graph_dir does `rm -rf BASE && mkdir -p BASE` at session start,
# which would nuke any sibling output folder placed inside BASE.
FIRE_DIR="graphs/vjp_sum_of_square_steps_fire"
REDUCE_DIR="graphs/vjp_sum_of_square_steps_reduce"
DEFAULT_DIR="${BASE}/steps"

run_granularity() {
    local gran="$1"      # "fire" or "reduce"
    local target_dir="$2"
    rm -rf "$target_dir"
    mkdir -p "$target_dir"

    # Session writes to the default ${BASE}/steps — move the output after.
    rm -rf "$DEFAULT_DIR"
    mkdir -p "$DEFAULT_DIR"

    export THVM_FOCUS_VJP=1
    export THVM_STEP_GRAPH=1
    export THVM_STEP_GRAPH_NO_PNG=1
    export THVM_STEP_GRAPH_GRANULARITY="$gran"
    [ -n "${TRACE:-}" ] && export THVM_INTERACT_TRACE=1

    echo "=== granularity=$gran ==="
    ./bin/test_grad_rules 2>&1 \
        | grep -Ev "^heap_dot:|^THVM step graph|^THVM coarse" \
        | tail -20

    # Move artifacts into the granularity-specific folder.  The session
    # may have blown away $target_dir via its mkdir -p $dir pattern, so
    # ensure it exists before moving.
    mkdir -p "$target_dir"
    for f in "$DEFAULT_DIR"/step_*.dot "$DEFAULT_DIR"/step_*.png; do
        [ -f "$f" ] || continue
        mv "$f" "$target_dir"/
    done

    unset THVM_STEP_GRAPH_GRANULARITY THVM_INTERACT_TRACE
}

render_pngs() {
    local dir="$1"
    unset THVM_STEP_GRAPH_NO_PNG
    for f in "$dir"/step_*.dot; do
        [ -f "$f" ] || continue
        dot -Tpng -Gdpi=150 "$f" -o "${f%.dot}.png" 2>/dev/null
    done
}

MODE="${1:-plain}"
case "$MODE" in
    trace)
        TRACE=1 run_granularity fire "$FIRE_DIR"
        TRACE=1 run_granularity reduce "$REDUCE_DIR"
        ;;
    png)
        run_granularity fire "$FIRE_DIR"
        run_granularity reduce "$REDUCE_DIR"
        render_pngs "$FIRE_DIR"
        render_pngs "$REDUCE_DIR"
        echo "--- fire/ ---"
        ls "$FIRE_DIR"
        echo "--- reduce/ ---"
        ls "$REDUCE_DIR"
        ;;
    fire)
        run_granularity fire "$FIRE_DIR"
        echo "--- fire/ ---"
        ls "$FIRE_DIR"
        ;;
    reduce)
        run_granularity reduce "$REDUCE_DIR"
        echo "--- reduce/ ---"
        ls "$REDUCE_DIR"
        ;;
    *)
        run_granularity fire "$FIRE_DIR"
        run_granularity reduce "$REDUCE_DIR"
        echo "--- fire/ ---"
        ls "$FIRE_DIR"
        echo "--- reduce/ ---"
        ls "$REDUCE_DIR"
        ;;
esac
