#!/bin/bash
# TinyHVM Build Script
# Compiles the LibraryLink bridge and Metal shaders, installs into paclet.
#
# Usage:
#   ./build.sh           # Build only
#   ./build.sh --clean   # Clean first
#   ./build.sh --install # Build + install paclet via wolframscript

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TINYHVM_SRC="${SCRIPT_DIR}/../src"
CSOURCE_DIR="${SCRIPT_DIR}/CSource"
LIB_DEST="${SCRIPT_DIR}/LibraryResources/MacOSX-ARM64"

# Wolfram include path
WOLFRAM_APP="/Applications/Wolfram 15.0.app"
WOLFRAM_INCLUDE="${WOLFRAM_APP}/Contents/SystemFiles/IncludeFiles/C"

CC=clang
CFLAGS="-O2 -Wall -Wextra -std=c11 -fPIC -DDEVICE='\"metal\"'"
FRAMEWORKS="-framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate"

# Parse args
CLEAN=false
INSTALL=false
for arg in "$@"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --install) INSTALL=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

if $CLEAN; then
    echo "=== Cleaning ==="
    rm -f "${LIB_DEST}/TinyHVM.dylib"
    rm -f "${LIB_DEST}/shaders.metallib"
fi

mkdir -p "${LIB_DEST}"

# Step 1: Compile Metal shaders
echo "=== Compiling Metal shaders ==="
SHADER_SRC="${TINYHVM_SRC}/shaders.metal"
if [ -f "$SHADER_SRC" ]; then
    xcrun -sdk macosx metal -c "$SHADER_SRC" -o /tmp/thvm_shaders.air
    xcrun -sdk macosx metallib /tmp/thvm_shaders.air -o "${LIB_DEST}/shaders.metallib"
    rm -f /tmp/thvm_shaders.air
    echo "  Compiled shaders.metallib"
else
    echo "  WARNING: ${SHADER_SRC} not found, Metal backend may fail"
fi

# Step 2: Compile LibraryLink bridge
echo "=== Compiling TinyHVM.dylib ==="
$CC $CFLAGS \
    -I"${WOLFRAM_INCLUDE}" \
    -I"${TINYHVM_SRC}" \
    $FRAMEWORKS \
    -dynamiclib \
    -o "${LIB_DEST}/TinyHVM.dylib" \
    "${CSOURCE_DIR}/tinyhvmlink.m"

codesign --force --sign - "${LIB_DEST}/TinyHVM.dylib"

echo "  Built TinyHVM.dylib ($(du -h "${LIB_DEST}/TinyHVM.dylib" | cut -f1))"

# Step 3: Optionally install paclet
if $INSTALL; then
    echo "=== Installing paclet ==="
    wolframscript -c "
        archive = CreatePacletArchive[\"${SCRIPT_DIR}\"];
        Print[\"Archive: \", archive];
        Quiet[PacletUninstall[\"TinyHVM\"]];
        p = PacletInstall[archive, ForceVersionInstall -> True];
        Print[\"Installed: \", p];
        Print[\"Location: \", p[\"Location\"]];
        Needs[\"TinyHVM\`\"];
        Print[\"FindLibrary: \", FindLibrary[\"TinyHVM\"]];
    "
fi

echo "=== Done ==="
