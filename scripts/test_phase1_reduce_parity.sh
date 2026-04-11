#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${1:-bin/test_tiny_single_param}"
SRC="${2:-test/test_tiny_single_param.m}"

mkdir -p "$(dirname "$BIN")"
clang -O2 -Wall -Wextra -std=c11 \
  -DDEVICE='"metal"' \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -Isrc -o "$BIN" "$SRC"

STEP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/thvm_step_parity.XXXXXX")"
GRAPH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/thvm_graph_parity.XXXXXX")"
trap 'rm -rf "$STEP_DIR" "$GRAPH_DIR"' EXIT

THVM_STEP_GRAPH=1 THVM_STEP_GRAPH_NO_PNG=1 THVM_STEP_GRAPH_DIR="$STEP_DIR" "$BIN"
python3 scripts/check_step_graphs.py "$STEP_DIR"

THVM_GRAPH=1 THVM_STEP_GRAPH_NO_PNG=1 THVM_GRAPH_DIR="$GRAPH_DIR" "$BIN"

STEP_FINAL="$(ls -1 "$STEP_DIR"/step_*_state_final.dot | tail -n 1)"
POST_REDUCE="$GRAPH_DIR/thvm_1_post_reduce.dot"

python3 - "$STEP_FINAL" "$POST_REDUCE" <<'PY'
import difflib
import sys

step_final, post_reduce = sys.argv[1], sys.argv[2]

def sanitize(path: str) -> list[str]:
    keep = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith("// PREV_INTERACTION:"):
                continue
            if stripped.startswith("// NEXT_INTERACTION:"):
                continue
            keep.append(line)
    return keep

a = sanitize(step_final)
b = sanitize(post_reduce)
if a != b:
    sys.stdout.writelines(
        difflib.unified_diff(a, b, fromfile=step_final, tofile=post_reduce)
    )
    raise SystemExit(1)

print(f"PASS: {post_reduce} matches {step_final} (ignoring step metadata)")
PY
