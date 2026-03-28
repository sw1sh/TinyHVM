#!/bin/bash
# TinyHVM Build Script
# Compiles test binaries and the Wolfram Language paclet bridge.
#
# Usage:
#   ./build.sh              # Build test binaries to bin/
#   ./build.sh --paclet     # Build WL paclet bridge (macOS only)
#   ./build.sh --all        # Build everything
#   ./build.sh --clean      # Clean build artifacts

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
SRC="${ROOT}/src"
BIN="${ROOT}/bin"

# ── Detect platform ─────────────────────────────────────────────────────

OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Darwin)
        CC=clang
        CFLAGS="-O2 -Wall -Wextra -std=c11"
        FRAMEWORKS="-framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate"
        DEVICE_FLAG='-DDEVICE="metal"'
        DYLIB_EXT=".dylib"
        case "$ARCH" in
            arm64) PLATFORM="MacOSX-ARM64" ;;
            *)     PLATFORM="MacOSX-x86-64" ;;
        esac
        ;;
    Linux)
        CC="${CC:-gcc}"
        CFLAGS="-O2 -Wall -Wextra -std=c11"
        FRAMEWORKS="-lm -lpthread"
        DEVICE_FLAG='-DDEVICE="cpu"'
        DYLIB_EXT=".so"
        case "$ARCH" in
            x86_64)  PLATFORM="Linux-x86-64" ;;
            aarch64) PLATFORM="Linux-ARM" ;;
            *)       PLATFORM="Linux-${ARCH}" ;;
        esac
        ;;
    *)
        echo "Unsupported OS: $OS"
        exit 1
        ;;
esac

# ── Parse arguments ─────────────────────────────────────────────────────

BUILD_TESTS=false
BUILD_PACLET=false
BUILD_PYTHON=false
CLEAN=false

if [ $# -eq 0 ]; then
    BUILD_TESTS=true
fi

for arg in "$@"; do
    case "$arg" in
        --clean)  CLEAN=true ;;
        --paclet) BUILD_PACLET=true ;;
        --python) BUILD_PYTHON=true ;;
        --all)    BUILD_TESTS=true; BUILD_PACLET=true; BUILD_PYTHON=true ;;
        --help|-h)
            echo "Usage: ./build.sh [--clean] [--paclet] [--python] [--all]"
            echo "  (no args)   Build test binaries to bin/"
            echo "  --paclet    Build WL paclet bridge (macOS Metal only)"
            echo "  --python    Build Python shared library (libthvm)"
            echo "  --all       Build everything"
            echo "  --clean     Remove build artifacts"
            exit 0
            ;;
        *) BUILD_TESTS=true ;;
    esac
done

# ── Clean ────────────────────────────────────────────────────────────────

if $CLEAN; then
    echo "=== Cleaning ==="
    rm -rf "${BIN}"
    rm -f shaders.metallib
    rm -rf "${ROOT}/wl/LibraryResources/${PLATFORM}"
    rm -f "${ROOT}/py/tinyhvm/libthvm${DYLIB_EXT}"
    echo "  Done"
    exit 0
fi

# ── Build test binaries ─────────────────────────────────────────────────

if $BUILD_TESTS; then
    mkdir -p "${BIN}"

    # Metal shaders (macOS only)
    if [ "$OS" = "Darwin" ] && [ ! -f shaders.metallib ]; then
        echo "=== Compiling Metal shaders ==="
        xcrun -sdk macosx metal -c "${SRC}/shaders.metal" -o /tmp/thvm_shaders.air
        xcrun -sdk macosx metallib /tmp/thvm_shaders.air -o shaders.metallib
        rm -f /tmp/thvm_shaders.air
    fi

    echo "=== Building test binaries (${OS}/${ARCH}) ==="

    for test_src in test/test_*.m; do
        [ -f "$test_src" ] || continue
        name="$(basename "$test_src" .m)"
        echo "  ${name}"
        $CC $CFLAGS $DEVICE_FLAG $FRAMEWORKS -I"${SRC}" -o "${BIN}/${name}" "$test_src"
    done

    echo "=== Done → bin/"
fi

# ── Build WL paclet bridge ──────────────────────────────────────────────

if $BUILD_PACLET; then
    if [ "$OS" != "Darwin" ]; then
        echo "ERROR: Paclet bridge requires macOS (Metal GPU backend)"
        exit 1
    fi

    PACLET="${ROOT}/wl"
    LIB_DEST="${PACLET}/LibraryResources/${PLATFORM}"
    CSOURCE="${PACLET}/CSource"

    # Find Wolfram installation
    WOLFRAM_APP=""
    for v in 15.0 14.1 14.0 13.3 13.2 13.1 13.0; do
        candidate="/Applications/Wolfram ${v}.app"
        if [ -d "$candidate" ]; then
            WOLFRAM_APP="$candidate"
            break
        fi
    done
    if [ -z "$WOLFRAM_APP" ]; then
        echo "ERROR: No Wolfram application found in /Applications/"
        exit 1
    fi
    WOLFRAM_INCLUDE="${WOLFRAM_APP}/Contents/SystemFiles/IncludeFiles/C"
    echo "=== Using ${WOLFRAM_APP} ==="

    mkdir -p "${LIB_DEST}"

    # Metal shaders
    echo "=== Compiling Metal shaders ==="
    if [ -f "${SRC}/shaders.metal" ]; then
        xcrun -sdk macosx metal -c "${SRC}/shaders.metal" -o /tmp/thvm_shaders.air
        xcrun -sdk macosx metallib /tmp/thvm_shaders.air -o "${LIB_DEST}/shaders.metallib"
        rm -f /tmp/thvm_shaders.air
    fi

    # LibraryLink bridge
    echo "=== Compiling TinyHVM${DYLIB_EXT} ==="
    $CC $CFLAGS \
        -I"${WOLFRAM_INCLUDE}" \
        -I"${SRC}" \
        $FRAMEWORKS \
        -fPIC -dynamiclib \
        -DDEVICE='"metal"' \
        -o "${LIB_DEST}/TinyHVM${DYLIB_EXT}" \
        "${CSOURCE}/tinyhvmlink.m"

    codesign --force --sign - "${LIB_DEST}/TinyHVM${DYLIB_EXT}"
    echo "  Built TinyHVM${DYLIB_EXT} ($(du -h "${LIB_DEST}/TinyHVM${DYLIB_EXT}" | cut -f1))"

    echo "=== Paclet ready at ${PACLET}/ ==="
fi

# ── Build Python shared library ───────────────────────────────────────

if $BUILD_PYTHON; then
    PY_CSRC="${ROOT}/py/csrc"
    PY_LIB="${ROOT}/py/tinyhvm"

    # Metal shaders (macOS only)
    if [ "$OS" = "Darwin" ] && [ ! -f shaders.metallib ]; then
        echo "=== Compiling Metal shaders ==="
        xcrun -sdk macosx metal -c "${SRC}/shaders.metal" -o /tmp/thvm_shaders.air
        xcrun -sdk macosx metallib /tmp/thvm_shaders.air -o shaders.metallib
        rm -f /tmp/thvm_shaders.air
    fi

    echo "=== Compiling libthvm${DYLIB_EXT} (Python FFI) ==="
    $CC $CFLAGS \
        -I"${SRC}" \
        $FRAMEWORKS \
        -fPIC -dynamiclib \
        -fvisibility=hidden \
        -DDEVICE='"metal"' \
        -o "${PY_LIB}/libthvm${DYLIB_EXT}" \
        "${PY_CSRC}/libthvm.m"

    codesign --force --sign - "${PY_LIB}/libthvm${DYLIB_EXT}"

    # Copy metallib next to dylib so the runtime can find it
    if [ -f shaders.metallib ]; then
        cp shaders.metallib "${PY_LIB}/shaders.metallib"
    fi

    echo "  Built libthvm${DYLIB_EXT} ($(du -h "${PY_LIB}/libthvm${DYLIB_EXT}" | cut -f1))"
    echo "=== Python library ready at ${PY_LIB}/ ==="
fi
