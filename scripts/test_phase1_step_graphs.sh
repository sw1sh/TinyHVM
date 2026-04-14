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

# Clean stale loop graph dumps so each run is unambiguous.
rm -rf graphs/loop_assign_simple/n*_steps
rm -rf graphs/loop_assign_simple/n*_steps_vals
rm -rf graphs/loop_assign_simple/n*_steps_fuse_vals
rm -f graphs/loop_assign_simple/n*_pre_reduce.dot \
      graphs/loop_assign_simple/n*_post_reduce.dot \
      graphs/loop_assign_simple/n*_pre_reduce.png \
      graphs/loop_assign_simple/n*_post_reduce.png \
      graphs/loop_assign_simple/n*_post_sched.dot \
      graphs/loop_assign_simple/n*_post_sched.png \
      graphs/loop_assign_simple/n*_post_dispatch.dot \
      graphs/loop_assign_simple/n*_post_dispatch.png

# Full-eval step graph checks (includes FUSE/dispatch interactions in sequence).
FAIL=0
for n in 0 1 2 3; do
  echo "=== n=$n ==="
  STEP_DIR="graphs/loop_assign_simple/n${n}_steps"
  FUSE_DIR="graphs/loop_assign_simple/n${n}_steps_fuse_vals"
  rm -rf "$STEP_DIR"
  rm -rf "$FUSE_DIR"
  THVM_TRAIN_STEPS=$n THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_DIR="$STEP_DIR" \
    timeout 60 "$BIN" 2>&1 | tail -3
  if ! python3 scripts/check_step_graphs.py "$STEP_DIR"; then
    FAIL=1
  fi
  THVM_TRAIN_STEPS=$n THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_FUSE=1 \
    THVM_STEP_GRAPH_TENSOR_VALUES=1 THVM_STEP_GRAPH_DIR="$FUSE_DIR" \
    timeout 60 "$BIN" 2>&1 | tail -3
  if ! python3 scripts/check_step_graphs.py "$FUSE_DIR"; then
    FAIL=1
  fi
done

if [ "$FAIL" -eq 0 ]; then
  echo "ALL PASS"
else
  echo "SOME FAILURES"
  exit 1
fi
