#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN_DIR="$ROOT/bin"
BIN="$BIN_DIR/test_kernel_epoch_redispatch"
LOG="/tmp/test_kernel_epoch_redispatch.log"

mkdir -p "$BIN_DIR"
rm -f "$LOG"

clang -O0 -g -I"$ROOT/src" -x objective-c \
  "$ROOT/test/test_kernel_epoch_redispatch.m" \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -lm -o "$BIN"

THVM_SCHED_DIAG=1 THVM_LOWER_DIAG=1 "$BIN" >"$LOG" 2>&1

python3 "$ROOT/scripts/check_private_lowering.py" "$LOG"
python3 "$ROOT/scripts/check_kernel_epoch_redispatch.py" "$LOG"

printf 'result summary: %s\n' "$(python3 - <<'PY'
from pathlib import Path
text = Path('/tmp/test_kernel_epoch_redispatch.log').read_text(encoding='utf-8', errors='replace').strip().splitlines()
for line in reversed(text):
    if line.startswith('result = ') or line.startswith('final_w = '):
        print(line)
        break
else:
    print('no result output')
PY
)"
