#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TEST="${1:-test/test_loop_assign_simple.m}"
BIN="bin/test_step_graph"
GLOBAL_N="${THVM_GLOBAL_TRAIN_STEPS:-1}"
GLOBAL_DIR="graphs/loop_assign_simple/global_eval"
GLOBAL_LOG="$GLOBAL_DIR/run.log"
PASS_DUMP="$GLOBAL_DIR/thvm_global_passes.txt"

bash "$ROOT/scripts/test_phase1_step_graphs.sh" "$TEST"

mkdir -p "$GLOBAL_DIR"
rm -rf "$GLOBAL_DIR"
mkdir -p "$GLOBAL_DIR"
rm -f "$GLOBAL_LOG"

THVM_TRAIN_STEPS="$GLOBAL_N" \
THVM_ALLOW_NON_TEN_RESULT=1 \
THVM_GRAPH=1 \
THVM_GRAPH_TENSOR_VALUES=1 \
THVM_GRAPH_DIR="$GLOBAL_DIR" \
timeout 60 "$BIN" >"$GLOBAL_LOG" 2>&1

for path in \
  "$GLOBAL_DIR/thvm_0_pre_reduce.dot" \
  "$GLOBAL_DIR/thvm_1_post_reduce.dot" \
  "$GLOBAL_DIR/thvm_2_post_dispatch.dot" \
  "$GLOBAL_DIR/thvm_3_post_passes.dot" \
  "$GLOBAL_DIR/thvm_4_post_exec.dot" \
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

python3 - "$GLOBAL_LOG" <<'PY'
from pathlib import Path
import sys

lines = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").strip().splitlines()
tail = lines[-3:] if len(lines) >= 3 else lines
for line in tail:
    print(line)
PY
