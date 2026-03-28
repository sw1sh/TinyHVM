# TinyHVM

Lazy interaction-net tensor engine with Metal GPU acceleration.

## Repository Layout

```
src/              C engine (single-TU build via tinyhvm.c)
test/             Test programs (.m = Objective-C with Metal)
bin/              Compiled test binaries (gitignored)
TinyHVM/          Wolfram Language paclet
  Kernel/         WL source (TinyHVM.wl, Layers.wl, Visualization.wl)
  CSource/        LibraryLink bridge (tinyhvmlink.m)
  LibraryResources/  Compiled dylib per platform
Notebooks/        Evaluated .nb files (gitignored)
```

## Building

### Test binaries

Requires macOS with Xcode command-line tools (Metal GPU backend).

```bash
./build.sh              # builds all test/test_*.m → bin/
make test_metal         # build + run test_term with Metal
make test_train_metal   # build + run test_train with Metal
```

Binaries go to `bin/`. The build compiles Metal shaders to `shaders.metallib` on first run.

### Wolfram Language paclet

Requires macOS + Wolfram Language 13.0+.

```bash
./build.sh --paclet     # compiles TinyHVM.dylib + shaders into TinyHVM/LibraryResources/
```

Then in Wolfram:
```wolfram
PacletDirectoryLoad["path/to/TinyHVM"];
Get["TinyHVM`"]
TInit["metal"]
```

### Everything

```bash
./build.sh --all
```

## Architecture

- **Single translation unit**: `src/tinyhvm.c` includes all `.c`/`.m` files
- **Interaction net**: HVM4-style bit-packed terms (u64 with TAG/EXT/VAL fields)
- **Lazy evaluation**: all ops return term handles; `thvm_reduce()` triggers the inet reducer
- **Metal GPU**: JIT kernel codegen via `src/backend/metal/codegen.m`
- **Autograd**: gradient as inet interaction rules, not a tape
- **Fusion**: declarative pattern matching in `src/fuse/`

## Key Files

- `src/tinyhvm.h` — public API, term layout, UOp codes
- `src/interact/_.c` — interaction rules (the core reducer)
- `src/grad/_.c` — gradient interaction rules
- `src/fuse/_.c` — operation fusion
- `src/backend/metal/codegen.m` — Metal kernel JIT
- `test/test_cnn_small.m` — MNIST CNN training test
