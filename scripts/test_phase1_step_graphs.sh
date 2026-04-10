#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="${1:-bin/test_tiny}"
mkdir -p bin
clang -O2 -Wall -Wextra -std=c11 \
  -DDEVICE='"metal"' \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -Isrc -o "$BIN" test/test_tiny.m

THVM_STEP_GRAPH=1 "$BIN"
python3 scripts/check_step_graphs.py thvm_steps
