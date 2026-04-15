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
      graphs/loop_assign_simple/n*_post_dispatch.png \
      graphs/loop_assign_simple/n*_steps.log \
      graphs/loop_assign_simple/n*_steps_fuse_vals.log

# Full-eval step graph checks (includes FUSE/dispatch interactions in sequence).
FAIL=0
for n in 0 1 2 3; do
  echo "=== n=$n ==="
  STEP_DIR="graphs/loop_assign_simple/n${n}_steps"
  FUSE_DIR="graphs/loop_assign_simple/n${n}_steps_fuse_vals"
  STEP_LOG="graphs/loop_assign_simple/n${n}_steps.log"
  FUSE_LOG="graphs/loop_assign_simple/n${n}_steps_fuse_vals.log"
  rm -rf "$STEP_DIR"
  rm -rf "$FUSE_DIR"
  rm -f "$STEP_LOG" "$FUSE_LOG"
  THVM_TRAIN_STEPS=$n THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_DIR="$STEP_DIR" \
    timeout 60 "$BIN" >"$STEP_LOG" 2>&1
  if ! python3 scripts/check_step_graphs.py "$STEP_DIR"; then
    FAIL=1
  fi
  if [ "$n" -ge 2 ]; then
    THVM_TRAIN_STEPS=$n THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_FUSE=1 \
      THVM_STEP_GRAPH_TENSOR_VALUES=1 THVM_STEP_GRAPH_DIR="$FUSE_DIR" \
      THVM_SCHED_DIAG=1 timeout 60 "$BIN" >"$FUSE_LOG" 2>&1
  else
    THVM_TRAIN_STEPS=$n THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_FUSE=1 \
      THVM_STEP_GRAPH_TENSOR_VALUES=1 THVM_STEP_GRAPH_DIR="$FUSE_DIR" \
      timeout 60 "$BIN" >"$FUSE_LOG" 2>&1
  fi
  if ! python3 scripts/check_step_graphs.py "$FUSE_DIR"; then
    FAIL=1
  fi
  if [ "$n" -eq 2 ]; then
    if ! python3 scripts/check_loop_fused_trace.py "$FUSE_DIR" --diag-log "$FUSE_LOG"; then
      FAIL=1
    fi
  else
    if ! python3 scripts/check_loop_fused_trace.py "$FUSE_DIR"; then
      FAIL=1
    fi
  fi
done

if ! bash scripts/test_kernel_epoch_redispatch.sh; then
  FAIL=1
fi

if [ "$FAIL" -eq 0 ]; then
  echo "ALL PASS"
else
  echo "SOME FAILURES"
  exit 1
fi
