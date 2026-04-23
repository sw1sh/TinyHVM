#!/bin/bash
# Incremental build of bin/test_grad_rules on macOS.  Kept separate from
# the full ./build.sh (which rebuilds every test binary) so iteration on
# the grad/step-graph code path doesn't re-link unrelated tests.
#
# Usage: ./scripts/build_grad_rules.sh
#
# Exits non-zero on compile failure; prints first error line.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CFLAGS="-O2 -Wall -Wextra -std=c11"
DEFS='-DDEVICE="metal"'
FRAMEWORKS="-framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate"

# Compile; filter output to show only errors + the first few warnings.
clang $CFLAGS $DEFS $FRAMEWORKS -Isrc -o bin/test_grad_rules test/test_grad_rules.m 2>&1 \
    | grep -E "error|^test/" | head -30 || true

# Verify artifact.
if [ ! -x bin/test_grad_rules ]; then
    echo "build_grad_rules.sh: bin/test_grad_rules not produced" >&2
    exit 1
fi
