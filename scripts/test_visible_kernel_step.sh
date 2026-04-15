#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/bin"
BIN="$BIN_DIR/test_fuse_kernel_visible"
OUT_DIR="$ROOT/graphs/fuse_kernel_visible/steps_fuse_vals"
LOWER_DIR="${OUT_DIR}_lower"
LOG="/tmp/test_fuse_kernel_visible.log"
BACKEND="${THVM_TEST_BACKEND:-metal}"

mkdir -p "$BIN_DIR"
rm -rf "$OUT_DIR"
rm -rf "$LOWER_DIR"
rm -f "$LOG"

clang -O0 -g -I"$ROOT/src" -x objective-c \
  "$ROOT/test/test_fuse_kernel_visible.m" \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -lm -o "$BIN"

THVM_STEP_GRAPH=1 \
THVM_STEP_GRAPH_FUSE=1 \
THVM_STEP_GRAPH_TENSOR_VALUES=1 \
THVM_LOWER_DIAG=1 \
THVM_LOWER_GRAPH=1 \
THVM_TEST_BACKEND="$BACKEND" \
THVM_STEP_GRAPH_DIR="$OUT_DIR" \
  "$BIN" >"$LOG" 2>&1

python3 "$ROOT/scripts/check_step_graphs.py" "$OUT_DIR"
python3 "$ROOT/scripts/check_visible_kernel_steps.py" "$OUT_DIR"
python3 "$ROOT/scripts/check_private_lowering.py" "$LOG" "$OUT_DIR"
python3 "$ROOT/scripts/check_lower_graph_trace.py" "$LOWER_DIR"

printf 'result summary: %s\n' "$(python3 - <<'PY'
from pathlib import Path
text = Path('/tmp/test_fuse_kernel_visible.log').read_text(encoding='utf-8', errors='replace').strip().splitlines()
print(text[-1] if text else 'no output')
PY
)"
