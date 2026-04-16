#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/bin"
BIN="$BIN_DIR/test_fuse_kernel_visible"
BASE_DIR="$ROOT/graphs/fuse_kernel_visible"
OUT_DIR="$BASE_DIR/steps_fuse_vals"
LOWER_DIR="${OUT_DIR}_lower"
SETTLED_LOWER_DIR="$LOWER_DIR/settled"
PASS_DUMP="$BASE_DIR/thvm_global_passes.txt"
LOG="/tmp/test_fuse_kernel_visible.log"
BACKEND="${THVM_TEST_BACKEND:-metal}"

mkdir -p "$BIN_DIR"
mkdir -p "$BASE_DIR"
rm -rf "$OUT_DIR"
rm -rf "$LOWER_DIR"
rm -f "$BASE_DIR"/thvm_*.dot
rm -f "$PASS_DUMP"
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
if compgen -G "$LOWER_DIR/step_*.dot" > /dev/null; then
  echo "expected settled lower dumps under $SETTLED_LOWER_DIR, found top-level step dumps in $LOWER_DIR" >&2
  exit 1
fi
python3 "$ROOT/scripts/check_lower_graph_trace.py" "$SETTLED_LOWER_DIR"

THVM_GRAPH=1 \
THVM_GRAPH_TENSOR_VALUES=1 \
THVM_TEST_BACKEND="$BACKEND" \
THVM_GRAPH_DIR="$BASE_DIR" \
  "$BIN" >>"$LOG" 2>&1

for path in \
  "$BASE_DIR/thvm_0_pre_reduce.dot" \
  "$BASE_DIR/thvm_1_post_reduce.dot" \
  "$BASE_DIR/thvm_2_post_dispatch.dot" \
  "$BASE_DIR/thvm_3_post_passes.dot" \
  "$BASE_DIR/thvm_4_post_exec.dot" \
  "$PASS_DUMP"
do
  test -f "$path"
done

python3 - "$PASS_DUMP" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
required = [
    "registered_passes=",
    "input=",
    "output=",
]
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit(f"missing pass dump fields: {missing}")
print("pass dump summary:", text.strip().splitlines()[0] if text.strip() else "empty")
PY

printf 'result summary: %s\n' "$(python3 - <<'PY'
from pathlib import Path
text = Path('/tmp/test_fuse_kernel_visible.log').read_text(encoding='utf-8', errors='replace').strip().splitlines()
print(text[-1] if text else 'no output')
PY
)"
