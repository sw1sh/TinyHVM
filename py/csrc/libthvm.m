// libthvm.m — Standalone shared library for TinyHVM (Python FFI target)
//
// Single-TU build: #includes the entire TinyHVM runtime.
// Objective-C (.m) because the Metal backend requires Foundation + Metal frameworks.
//
// Build:
//   clang -O2 -std=c11 -fPIC -dynamiclib \
//     -framework Metal -framework MetalPerformanceShaders \
//     -framework Foundation -framework Accelerate \
//     -I../../src -o libthvm.dylib libthvm.m

#ifdef __APPLE__
#import <Foundation/Foundation.h>
#endif

#include "tinyhvm.c"
#include "backend/cpu/_.c"
#ifdef __APPLE__
#include "backend/metal/_.m"
#endif
#include "nn/_.c"

// ── Python-friendly C API ────────────────────────────────────────────────────
// All functions use simple C types (no WL/LibraryLink dependencies).
// Prefix: thvm_ (matches tinyhvm.h) + py_ for Python-specific helpers.

// Metallib path override (must be set before py_init)
static char g_metallib_path[1024] = {0};

__attribute__((visibility("default")))
void py_set_metallib_path(const char *path) {
    if (path) strncpy(g_metallib_path, path, sizeof(g_metallib_path) - 1);
}

// Resolve shaders.metallib relative to this dylib
#include <dlfcn.h>
static void _resolve_metallib_path(void) {
    if (g_metallib_path[0]) return;  // already set by user
    Dl_info info;
    if (dladdr((void *)_resolve_metallib_path, &info) && info.dli_fname) {
        // dylib is at e.g. .../tinyhvm/libthvm.dylib
        // shaders.metallib is next to it
        const char *slash = strrchr(info.dli_fname, '/');
        if (slash) {
            size_t dirlen = (size_t)(slash - info.dli_fname);
            snprintf(g_metallib_path, sizeof(g_metallib_path),
                     "%.*s/shaders.metallib", (int)dirlen, info.dli_fname);
        }
    }
}

__attribute__((visibility("default")))
TinyHVM *py_init(const char *device) {
    _resolve_metallib_path();
    if (g_metallib_path[0]) setenv("THVM_METALLIB", g_metallib_path, 1);
    return thvm_init(device);
}

__attribute__((visibility("default")))
void py_free(TinyHVM *ctx) { thvm_free(ctx); }

__attribute__((visibility("default")))
void py_reset(TinyHVM *ctx, u32 keep) { thvm_reset(ctx, keep); }

// Tensor creation: data is float32 array, dims/rank describe shape
__attribute__((visibility("default")))
u64 py_tensor(TinyHVM *ctx, const float *data, const u32 *dims, u32 rank) {
    Shape s = shape_of(dims, rank);
    return (u64)thvm_tensor(ctx, data, s);
}

// Tensor from scalar
__attribute__((visibility("default")))
u64 py_scalar(TinyHVM *ctx, float val) {
    return (u64)thvm_tensor(ctx, &val, SHAPE(1));
}

// Binary/unary op
__attribute__((visibility("default")))
u64 py_op(TinyHVM *ctx, u32 uop, u64 a, u64 b) {
    return (u64)thvm_op(ctx, uop, (Term)a, (Term)b);
}

// Reduce (trigger computation)
__attribute__((visibility("default")))
u64 py_reduce(TinyHVM *ctx, u64 t) {
    return (u64)thvm_reduce(ctx, (Term)t);
}

// Realize in-place
__attribute__((visibility("default")))
void py_realize(TinyHVM *ctx, u64 t) {
    thvm_realize(ctx, (Term)t);
}

// Read tensor data to host (returns pointer to float array — caller must NOT free)
__attribute__((visibility("default")))
const float *py_to_host(TinyHVM *ctx, u64 t) {
    return thvm_to_host(ctx, (Term)t);
}

// Get view for any term (TAG_TEN or TAG_TOP via ShapeTracker)
static const View *_term_view(TinyHVM *ctx, Term term) {
    u32 tag = term_tag(term);
    if (tag == TAG_TEN) {
        return &ctx->tensors[(u32)term_val(term)].view;
    } else if (tag == TAG_TOP) {
        return st_get(term_val(term));
    }
    return NULL;
}

// Get tensor shape
__attribute__((visibility("default")))
u32 py_ndim(TinyHVM *ctx, u64 t) {
    const View *v = _term_view(ctx, (Term)t);
    return v ? v->shape.rank : 0;
}

__attribute__((visibility("default")))
void py_shape(TinyHVM *ctx, u64 t, u32 *out_dims, u32 *out_rank) {
    const View *v = _term_view(ctx, (Term)t);
    if (!v) { *out_rank = 0; return; }
    *out_rank = v->shape.rank;
    for (u32 i = 0; i < v->shape.rank; i++) out_dims[i] = v->shape.dims[i];
}

__attribute__((visibility("default")))
u32 py_numel(TinyHVM *ctx, u64 t) {
    const View *v = _term_view(ctx, (Term)t);
    return v ? v->numel : 0;
}

// Movement ops
__attribute__((visibility("default")))
u64 py_reshape(TinyHVM *ctx, u64 t, const u32 *dims, u32 rank) {
    return (u64)thvm_reshape(ctx, (Term)t, shape_of(dims, rank));
}

__attribute__((visibility("default")))
u64 py_expand(TinyHVM *ctx, u64 t, const u32 *dims, u32 rank) {
    return (u64)thvm_expand(ctx, (Term)t, shape_of(dims, rank));
}

__attribute__((visibility("default")))
u64 py_permute(TinyHVM *ctx, u64 t, const u32 *axes, u32 rank) {
    return (u64)thvm_permute(ctx, (Term)t, axes, rank);
}

__attribute__((visibility("default")))
u64 py_pad(TinyHVM *ctx, u64 t, const u32 *pairs, u32 ndim) {
    return (u64)thvm_pad(ctx, (Term)t, pairs, ndim);
}

__attribute__((visibility("default")))
u64 py_shrink(TinyHVM *ctx, u64 t, const u32 *pairs, u32 ndim) {
    return (u64)thvm_shrink(ctx, (Term)t, pairs, ndim);
}

// Autograd
__attribute__((visibility("default")))
void py_set_requires_grad(TinyHVM *ctx, u64 t) {
    thvm_set_requires_grad(ctx, (Term)t);
}

__attribute__((visibility("default")))
u64 py_grad(TinyHVM *ctx, u64 y, u64 x) {
    return (u64)thvm_grad(ctx, (Term)y, (Term)x);
}

// Where (ternary select)
__attribute__((visibility("default")))
u64 py_where(TinyHVM *ctx, u64 cond, u64 then_t, u64 else_t) {
    return (u64)thvm_where(ctx, (Term)cond, (Term)then_t, (Term)else_t);
}

// Assign (in-place update)
__attribute__((visibility("default")))
u64 py_assign(TinyHVM *ctx, u64 dst, u64 src) {
    return (u64)thvm_assign(ctx, (Term)dst, (Term)src);
}

// Conv2d
__attribute__((visibility("default")))
u64 py_conv2d(TinyHVM *ctx, u64 x, u64 w, u64 bias,
              u32 groups, const u32 *stride, const u32 *padding) {
    return (u64)thvm_conv2d(ctx, (Term)x, (Term)w, (Term)bias,
                            groups, stride, padding);
}

// MaxPool2d
__attribute__((visibility("default")))
u64 py_maxpool2d(TinyHVM *ctx, u64 x, const u32 *kernel, const u32 *stride) {
    return (u64)thvm_maxpool2d(ctx, (Term)x, kernel, stride);
}

// ERA term (used as placeholder for unary ops)
__attribute__((visibility("default")))
u64 py_era(void) {
    return (u64)term_new(TAG_ERA, 0, 0);
}

// Term introspection
__attribute__((visibility("default")))
u32 py_term_tag(u64 t) { return term_tag((Term)t); }

__attribute__((visibility("default")))
u32 py_term_ext(u64 t) { return term_ext((Term)t); }

// Profiling
__attribute__((visibility("default")))
void py_profile_report(TinyHVM *ctx) { thvm_profile_report(ctx); }

__attribute__((visibility("default")))
void py_profile_reset(TinyHVM *ctx) { thvm_profile_reset(ctx); }
