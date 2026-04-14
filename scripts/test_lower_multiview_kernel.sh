#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/bin"
BIN="$BIN_DIR/test_lower_multiview_kernel"
LOG="/tmp/test_lower_multiview_kernel.log"

mkdir -p "$BIN_DIR"
rm -f "$LOG"

clang -O0 -g -I"$ROOT/src" -x objective-c \
  "$ROOT/test/test_lower_multiview_kernel.m" \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -lm -o "$BIN"

THVM_LOWER_DIAG=1 "$BIN" >"$LOG" 2>&1

python3 "$ROOT/scripts/check_private_lowering.py" "$LOG"

printf 'result summary: %s\n' "$(python3 - <<'PY'
from pathlib import Path
text = Path('/tmp/test_lower_multiview_kernel.log').read_text(encoding='utf-8', errors='replace').strip().splitlines()
for line in reversed(text):
    if line.startswith('multiview_total = '):
        print(line)
        break
else:
    print('no result output')
PY
)"
