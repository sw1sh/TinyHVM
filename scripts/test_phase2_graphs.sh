#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

build_bin() {
  local bin="$1"
  local src="$2"
  local log="$3"
  mkdir -p "$(dirname "$bin")"
  if ! clang -O2 -Wall -Wextra -std=c11 \
    -DDEVICE='"metal"' \
    -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate \
    -Isrc -o "$bin" "$src" >"$log" 2>&1; then
    cat "$log"
    return 1
  fi
}

run_case() {
  local log="$1"
  shift
  if ! "$@" >"$log" 2>&1; then
    cat "$log"
    return 1
  fi
}

SP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/thvm_phase2_sp.XXXXXX")"
LB_DIR="$(mktemp -d "${TMPDIR:-/tmp}/thvm_phase2_lb.XXXXXX")"
SPK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/thvm_phase2_spk.XXXXXX")"
LBK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/thvm_phase2_lbk.XXXXXX")"
BUILD_LOG="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_build.out.XXXXXX")"
PARITY_SP_LOG="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_parity_sp.out.XXXXXX")"
PARITY_LB_LOG="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_parity_lb.out.XXXXXX")"
SP_OUT="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_sp.out.XXXXXX")"
LB_OUT="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_lb.out.XXXXXX")"
SPK_OUT="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_spk.out.XXXXXX")"
LBK_OUT="$(mktemp "${TMPDIR:-/tmp}/thvm_phase2_lbk.out.XXXXXX")"
trap 'rm -rf "$SP_DIR" "$LB_DIR" "$SPK_DIR" "$LBK_DIR" "$BUILD_LOG" "$PARITY_SP_LOG" "$PARITY_LB_LOG" "$SP_OUT" "$LB_OUT" "$SPK_OUT" "$LBK_OUT"' EXIT

build_bin bin/test_tiny_single_param      test/test_tiny_single_param.m      "$BUILD_LOG"
build_bin bin/test_tiny_linear_bias       test/test_tiny_linear_bias.m       "$BUILD_LOG"
build_bin bin/test_tiny_single_param_keep test/test_tiny_single_param_keep.m "$BUILD_LOG"
build_bin bin/test_tiny_linear_bias_keep  test/test_tiny_linear_bias_keep.m  "$BUILD_LOG"

if ! ./scripts/test_phase1_reduce_parity.sh bin/test_tiny_single_param test/test_tiny_single_param.m >"$PARITY_SP_LOG" 2>&1; then
  cat "$PARITY_SP_LOG"
  exit 1
fi
if ! ./scripts/test_phase1_reduce_parity.sh bin/test_tiny_linear_bias test/test_tiny_linear_bias.m >"$PARITY_LB_LOG" 2>&1; then
  cat "$PARITY_LB_LOG"
  exit 1
fi

run_case "$SP_OUT"  env THVM_GRAPH=1 THVM_GRAPH_DIR="$SP_DIR"  ./bin/test_tiny_single_param
run_case "$LB_OUT"  env THVM_GRAPH=1 THVM_GRAPH_DIR="$LB_DIR"  ./bin/test_tiny_linear_bias
run_case "$SPK_OUT" env THVM_GRAPH=1 THVM_GRAPH_DIR="$SPK_DIR" ./bin/test_tiny_single_param_keep
run_case "$LBK_OUT" env THVM_GRAPH=1 THVM_GRAPH_DIR="$LBK_DIR" ./bin/test_tiny_linear_bias_keep

python3 scripts/check_phase2_graphs.py "$SP_DIR"  --expect-log 1 --expect-kernels 2
python3 scripts/check_phase2_graphs.py "$LB_DIR"  --expect-log 1 --expect-kernels 2
python3 scripts/check_phase2_graphs.py "$SPK_DIR" --allow-ctr --expect-ctr 1 --expect-kernels 1
python3 scripts/check_phase2_graphs.py "$LBK_DIR" --allow-ctr --expect-ctr 1 --expect-kernels 5

grep -F "grad_w = [5.00, 7.00, 9.00]" "$SPK_OUT" >/dev/null
grep -F "grad_w = [5.00, 0.00, 5.00, 0.00, 7.00, 0.00, 7.00, 0.00, 9.00, 0.00, 9.00, 0.00]" "$LBK_OUT" >/dev/null
grep -F "grad_b = [2.00, 0.00, 2.00, 0.00]" "$LBK_OUT" >/dev/null

grep -F "PASS:" "$PARITY_SP_LOG"
grep -F "PASS:" "$PARITY_LB_LOG"
echo "PASS: phase-2 plain and keep graph checks"
