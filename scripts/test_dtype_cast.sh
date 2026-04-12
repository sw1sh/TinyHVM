#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="bin/test_dtype_cast"
LOG="$(mktemp "${TMPDIR:-/tmp}/thvm_dtype_cast.out.XXXXXX")"
trap 'rm -f "$LOG"' EXIT

mkdir -p bin
clang -O2 -Wall -Wextra -std=c11 \
  -DDEVICE='"metal"' \
  -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
  -Isrc -o "$BIN" test/test_dtype_cast.m >"$LOG" 2>&1 || {
    cat "$LOG"
    exit 1
  }

"$BIN" >"$LOG" 2>&1 || {
  cat "$LOG"
  exit 1
}

grep -F "u32=7" "$LOG" >/dev/null
grep -F "cast_i32=[1,-2,3]" "$LOG" >/dev/null
grep -F "cast_f32=[3.0,-4.0]" "$LOG" >/dev/null
grep -F "sched_add_i32=[7,-16]" "$LOG" >/dev/null
grep -F "sched_sum_i32=[6]" "$LOG" >/dev/null
grep -F "sched_cast_add_i32=[2,-1,4]" "$LOG" >/dev/null
grep -F "where_i32=[1,20,3]" "$LOG" >/dev/null
grep -F "argmax_i32=[1,0]" "$LOG" >/dev/null
grep -F "acc_i32=100.0" "$LOG" >/dev/null
grep -F "seed=f16:1.0" "$LOG" >/dev/null

echo "PASS: typed CAST/readback smoke"
