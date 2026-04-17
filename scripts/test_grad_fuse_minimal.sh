#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/bin"
BIN="$BIN_DIR/test_grad_fuse_minimal"
OUT_DIR="$ROOT/graphs/grad_fuse_minimal"
GIF="$OUT_DIR/step_trace.gif"
LOG="$OUT_DIR/run.log"
BACKEND="${THVM_TEST_BACKEND:-cpu}"

mkdir -p "$BIN_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

clang -O0 -g -I"$ROOT/src" -x objective-c \
  "$ROOT/test/test_grad_fuse_minimal.m" \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -lm -o "$BIN"

THVM_STEP_GRAPH=1 \
THVM_STEP_GRAPH_TENSOR_VALUES=1 \
THVM_STEP_GRAPH_DIR="$OUT_DIR" \
THVM_TEST_BACKEND="$BACKEND" \
  "$BIN" >"$LOG" 2>&1

THVM_GRAPH=1 \
THVM_GRAPH_TENSOR_VALUES=1 \
THVM_GRAPH_DIR="$OUT_DIR" \
THVM_LOWER_GRAPH=1 \
THVM_LOWER_GRAPH_DIR="$OUT_DIR" \
THVM_TEST_BACKEND="$BACKEND" \
  "$BIN" >>"$LOG" 2>&1

python3 - "$OUT_DIR" <<'PY'
from pathlib import Path
import re
import sys

out_dir = Path(sys.argv[1])
steps = sorted(out_dir.glob("step_*.dot"))
if len(steps) < 2:
    raise SystemExit("expected at least two step graphs")

def tags_of(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return set(re.findall(r'label="([A-Z]+)', text))

def fuse_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count('label="FUSE')

def app_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count('label="APP')

def kernel_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count('label="KERNEL')

def ctr_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count('label="CTR')

def assign_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    return text.count('label="ASSIGN')

step0, step1 = steps[0], steps[1]
tags0 = tags_of(step0)
tags1 = tags_of(step1)
if not {"GRAD", "FUSE"} <= tags0:
    raise SystemExit(f"{step0.name} must contain both GRAD and FUSE")
if app_count(step0) != 1:
    raise SystemExit(f"{step0.name} must contain the keep-driver APP")
if "APP" not in tags0:
    raise SystemExit(f"{step0.name} must show the keep-driver APP")
if "GRAD" not in tags1:
    raise SystemExit(f"{step1.name} must keep GRAD live")
final_step = next((step for step in reversed(steps) if "state_final" in step.name), steps[-1])
if kernel_count(final_step) != 2:
    raise SystemExit(f"{final_step.name} must end with exactly two kernels")
if ctr_count(final_step) != 1:
    raise SystemExit(f"{final_step.name} must end with a single CTR collector")
if app_count(final_step) != 0:
    raise SystemExit(f"{final_step.name} must not contain APP")
if assign_count(final_step) != 0:
    raise SystemExit(f"{final_step.name} must not contain ASSIGN")
print("initial live frame:", step0.name)
print("post-step live frame:", step1.name)
print("final frame:", final_step.name)
PY

for path in \
  "$OUT_DIR/thvm_0_pre_reduce.dot" \
  "$OUT_DIR/thvm_1_post_reduce.dot" \
  "$OUT_DIR/thvm_2_post_dispatch.dot" \
  "$OUT_DIR/thvm_3_post_passes.dot" \
  "$OUT_DIR/thvm_4_post_exec.dot" \
  "$OUT_DIR/thvm_global_passes.txt"
do
  test -f "$path"
done

if ! compgen -G "$OUT_DIR/kernel_*_kid0.dot" > /dev/null; then
  echo "expected kid0 lowering graph in $OUT_DIR" >&2
  exit 1
fi

if ! compgen -G "$OUT_DIR/kernel_*_kid1.dot" > /dev/null; then
  echo "expected kid1 lowering graph in $OUT_DIR" >&2
  exit 1
fi

if command -v dot >/dev/null 2>&1; then
  for f in "$OUT_DIR"/thvm_*.dot "$OUT_DIR"/kernel_*.dot; do
    [ -f "$f" ] || continue
    dot -Tpng -Gdpi=150 "$f" -o "${f%.dot}.png"
  done
fi

python3 "$ROOT/scripts/render_step_gif.py" "$OUT_DIR" "$GIF"

printf 'trace dir: %s\n' "$OUT_DIR"
printf 'gif: %s\n' "$GIF"
