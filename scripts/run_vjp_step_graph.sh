#!/bin/bash
# Clear per-step graphs and rerun the vjp_sum_of_square test with step-graph
# tracing enabled.  Prints the test status + generated step filenames.
#
# Usage:
#   ./scripts/run_vjp_step_graph.sh          # run test, list files
#   ./scripts/run_vjp_step_graph.sh trace    # also stream THVM_INTERACT_TRACE
#   ./scripts/run_vjp_step_graph.sh png      # render PNGs after run

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

STEPS_DIR="graphs/grad_rules/vjp_sum_of_square/steps"
rm -f "$STEPS_DIR"/*.dot "$STEPS_DIR"/*.png 2>/dev/null || true

export THVM_FOCUS_VJP=1
export THVM_STEP_GRAPH=1
export THVM_STEP_GRAPH_NO_PNG=1

MODE="${1:-plain}"
case "$MODE" in
    trace)
        export THVM_INTERACT_TRACE=1
        ./bin/test_grad_rules 2>&1 | grep -E "interact|FAIL|PASS|failures" | tail -60
        ;;
    png)
        ./bin/test_grad_rules 2>&1 | tail -3
        echo "--- generated files ---"
        ls "$STEPS_DIR"
        unset THVM_STEP_GRAPH_NO_PNG
        for f in "$STEPS_DIR"/step_*.dot; do
            dot -Tpng -Gdpi=150 "$f" -o "${f%.dot}.png" 2>/dev/null
        done
        echo "--- rendered PNGs ---"
        ls "$STEPS_DIR" | grep png
        ;;
    *)
        ./bin/test_grad_rules 2>&1 | tail -3
        echo "--- generated files ---"
        ls "$STEPS_DIR"
        ;;
esac
