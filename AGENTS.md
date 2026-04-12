# TinyHVM

Agent instructions for TinyHVM. This file consolidates the previous
`AGENTS.md` project notes and `CLAUDE.md` working guidelines. Bias toward
correctness, simple solutions, minimal diffs, and explicit verification.

## Project Overview

Lazy interaction-net tensor engine with Metal GPU acceleration.

## Repository Layout

```text
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
./build.sh              # builds all test/test_*.m -> bin/
make test_metal         # build + run test_term with Metal
make test_train_metal   # build + run test_train with Metal
```

Binaries go to `bin/`. The build compiles Metal shaders to
`shaders.metallib` on first run.

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

- **Single translation unit**: `src/tinyhvm.c` includes all `.c` and `.m` files
- **Interaction net**: HVM4-style bit-packed terms (`u64` with `TAG`/`EXT`/`VAL` fields)
- **Lazy evaluation**: all ops return term handles; `thvm_reduce()` triggers the inet reducer
- **Metal GPU**: JIT kernel codegen via `src/backend/metal/codegen.m`
- **Autograd**: gradient is encoded as inet interaction rules, not a tape
- **Fusion**: declarative pattern matching in `src/fuse/`

## Key Files

- `src/tinyhvm.h` - public API, term layout, UOp codes
- `src/interact/_.c` - interaction rules (the core reducer)
- `src/grad/_.c` - gradient interaction rules
- `src/fuse/_.c` - operation fusion
- `src/backend/metal/codegen.m` - Metal kernel JIT
- `test/test_cnn_small.m` - MNIST CNN training test

## Working Guidelines

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

# 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

# 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

# 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

# 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.