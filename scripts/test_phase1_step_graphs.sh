#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Build test binary
TEST="${1:-test/test_loop_assign_simple.m}"
BIN="bin/test_step_graph"
mkdir -p bin
clang -O0 -g \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -o "$BIN" "$TEST" 2>&1 | grep -v warning || true

# Run for multiple step counts
FAIL=0
for n in 0 1 2; do
  echo "=== n=$n ==="
  rm -rf thvm_steps
  THVM_TRAIN_STEPS=$n THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_NO_PNG=1 \
    timeout 30 "$BIN" 2>&1 | tail -3
  python3 scripts/check_step_graphs.py thvm_steps || FAIL=1
done

if [ "$FAIL" -eq 0 ]; then
  echo "ALL PASS"
else
  echo "SOME FAILURES"
  exit 1
fi
