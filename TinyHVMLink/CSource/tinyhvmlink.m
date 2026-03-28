// tinyhvmlink.m — Wolfram LibraryLink bridge for TinyHVM
//
// Single-TU build: #includes the entire TinyHVM runtime (same pattern as test/*.m).
// Objective-C (.m) because the Metal backend requires Foundation + Metal frameworks.
//
// Global state:
//   g_ctx      — singleton TinyHVM context
//   g_terms[]  — WL id → Term (u64) mapping
//   g_na_funcs — NumericArray library functions

#import <Foundation/Foundation.h>
#include "WolframLibrary.h"
#include "WolframNumericArrayLibrary.h"

// ── Include the entire TinyHVM runtime ─────────────────────────────────────
#include "tinyhvm.c"
#include "backend/cpu/_.c"
#ifdef __APPLE__
#include "backend/metal/_.m"
#endif
#include "nn/_.c"

// ── Global state ───────────────────────────────────────────────────────────

static TinyHVM *g_ctx = NULL;

#define TERM_MAP_INIT_CAP 4096
static Term *g_terms = NULL;
static u32   g_term_cap = 0;
static u32   g_term_count = 0;

static WolframNumericArrayLibrary_Functions g_na_funcs = NULL;

// Optional: path override for shaders.metallib (set before thvmInit)
static char g_metallib_path[1024] = {0};

// ── Helpers ────────────────────────────────────────────────────────────────

static void ensure_term_cap(mint id) {
    if (id < 0) return;
    u32 need = (u32)id + 1;
    if (need <= g_term_cap) return;
    u32 new_cap = g_term_cap ? g_term_cap : TERM_MAP_INIT_CAP;
    while (new_cap < need) new_cap *= 2;
    g_terms = (Term *)realloc(g_terms, new_cap * sizeof(Term));
    memset(g_terms + g_term_cap, 0, (new_cap - g_term_cap) * sizeof(Term));
    g_term_cap = new_cap;
}

static inline Term get_term(mint id) {
    if (id <= 0 || (u32)id >= g_term_cap) return term_era();
    return g_terms[id];
}

static inline void set_term(mint id, Term t) {
    ensure_term_cap(id);
    g_terms[id] = t;
}

// ── WolframLibrary lifecycle ───────────────────────────────────────────────

EXTERN_C DLLEXPORT mint WolframLibrary_getVersion(void) {
    return WolframLibraryVersion;
}

EXTERN_C DLLEXPORT int WolframLibrary_initialize(WolframLibraryData libData) {
    g_na_funcs = libData->numericarrayLibraryFunctions;
    return LIBRARY_NO_ERROR;
}

EXTERN_C DLLEXPORT void WolframLibrary_uninitialize(WolframLibraryData libData) {
    (void)libData;
    if (g_ctx) { thvm_free(g_ctx); g_ctx = NULL; }
    if (g_terms) { free(g_terms); g_terms = NULL; }
    g_term_cap = 0;
    g_term_count = 0;
}

// ============================================================
// Context lifecycle
// ============================================================

// thvmSetMetallibPath[path_String] — set shader path before init
EXTERN_C DLLEXPORT int thvmSetMetallibPath(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    const char *path = MArgument_getUTF8String(args[0]);
    strncpy(g_metallib_path, path, sizeof(g_metallib_path) - 1);
    g_metallib_path[sizeof(g_metallib_path) - 1] = '\0';
    return LIBRARY_NO_ERROR;
}

// thvmInit["cpu"|"metal"] → Integer (1=success, 0=fail)
EXTERN_C DLLEXPORT int thvmInit(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc;
    if (g_ctx) { thvm_free(g_ctx); g_ctx = NULL; }

    const char *device = MArgument_getUTF8String(args[0]);

#ifdef __APPLE__
    // If metallib path was set, chdir to its directory so metal_init finds it
    if (g_metallib_path[0] && strcmp(device, "metal") == 0) {
        // Copy the metallib file to "shaders.metallib" in a findable location
        // by setting the working directory temporarily
        char saved_cwd[1024];
        getcwd(saved_cwd, sizeof(saved_cwd));

        // Extract directory from path
        char dir[1024];
        strncpy(dir, g_metallib_path, sizeof(dir));
        char *last_slash = strrchr(dir, '/');
        if (last_slash) {
            *last_slash = '\0';
            chdir(dir);
        }

        g_ctx = thvm_init(thvm_device(device));

        // Restore working directory
        chdir(saved_cwd);
    } else
#endif
    {
        g_ctx = thvm_init(thvm_device(device));
    }

    // Init term map
    if (g_terms) { free(g_terms); g_terms = NULL; }
    g_term_cap = 0;
    g_term_count = 0;
    ensure_term_cap(TERM_MAP_INIT_CAP);

    MArgument_setInteger(res, g_ctx ? 1 : 0);
    return LIBRARY_NO_ERROR;
}

// thvmFree[] → Void
EXTERN_C DLLEXPORT int thvmFree(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args; (void)res;
    if (g_ctx) { thvm_free(g_ctx); g_ctx = NULL; }
    if (g_terms) { free(g_terms); g_terms = NULL; }
    g_term_cap = 0;
    g_term_count = 0;
    return LIBRARY_NO_ERROR;
}

// thvmReset[keep] → Void
EXTERN_C DLLEXPORT int thvmReset(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;
    mint keep = MArgument_getInteger(args[0]);
    thvm_reset(g_ctx, (u32)keep);
    return LIBRARY_NO_ERROR;
}

// ============================================================
// Tensor creation & data transfer
// ============================================================

// thvmTensor[outId, NumericArray, shape:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmTensor(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    MNumericArray na = MArgument_getMNumericArray(args[1]);
    MTensor shape_mt = MArgument_getMTensor(args[2]);

    // Parse shape
    mint shape_len = 0;
    mint const *shape_data = NULL;
    {
        mint *sd = libData->MTensor_getIntegerData(shape_mt);
        shape_len = libData->MTensor_getFlattenedLength(shape_mt);
        shape_data = sd;
    }

    Shape s = {.rank = (u32)shape_len};
    for (mint i = 0; i < shape_len; i++)
        s.dims[i] = (u32)shape_data[i];

    // Get float data
    float *fdata = (float *)g_na_funcs->MNumericArray_getData(na);
    if (!fdata) return LIBRARY_FUNCTION_ERROR;

    Term t = thvm_tensor(g_ctx, fdata, s);
    set_term(out_id, t);

    return LIBRARY_NO_ERROR;
}

// thvmToHost[termId] → NumericArray (Real32)
EXTERN_C DLLEXPORT int thvmToHost(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint term_id = MArgument_getInteger(args[0]);
    Term t = get_term(term_id);

    // Reduce if not already a tensor
    if (term_tag(t) != TAG_TEN) {
        t = thvm_reduce(g_ctx, t);
        set_term(term_id, t);
    }
    if (term_tag(t) != TAG_TEN) return LIBRARY_FUNCTION_ERROR;

    f32 *data = thvm_to_host(g_ctx, t);
    if (!data) return LIBRARY_FUNCTION_ERROR;

    u32 tid = (u32)term_val(t);
    TensorMeta *m = &g_ctx->tensors[tid];

    // Build NumericArray with shape
    mint dims[MAX_DIM];
    for (u32 i = 0; i < m->view.shape.rank; i++)
        dims[i] = (mint)m->view.shape.dims[i];

    MNumericArray out;
    int err = g_na_funcs->MNumericArray_new(
        MNumericArray_Type_Real32,
        (mint)m->view.shape.rank, dims, &out);
    if (err) return LIBRARY_FUNCTION_ERROR;

    void *out_data = g_na_funcs->MNumericArray_getData(out);
    memcpy(out_data, data, (size_t)m->view.numel * sizeof(f32));

    MArgument_setMNumericArray(res, out);
    return LIBRARY_NO_ERROR;
}

// thvmDimensions[termId] → {Integer, 1}
EXTERN_C DLLEXPORT int thvmDimensions(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint term_id = MArgument_getInteger(args[0]);
    Term t = get_term(term_id);

    // For TAG_TEN, read directly from tensor metadata
    if (term_tag(t) == TAG_TEN) {
        u32 tid = (u32)term_val(t);
        TensorMeta *m = &g_ctx->tensors[tid];
        MTensor out;
        mint rank_dim = (mint)m->view.shape.rank;
        libData->MTensor_new(MType_Integer, 1, &rank_dim, &out);
        mint *od = libData->MTensor_getIntegerData(out);
        for (u32 i = 0; i < m->view.shape.rank; i++)
            od[i] = (mint)m->view.shape.dims[i];
        MArgument_setMTensor(res, out);
        return LIBRARY_NO_ERROR;
    }

    // For TAG_TOP, look up ShapeTracker
    if (term_tag(t) == TAG_TOP) {
        u64 loc = term_val(t);
        const View *v = st_get(loc);
        if (v) {
            MTensor out;
            mint rank_dim = (mint)v->shape.rank;
            libData->MTensor_new(MType_Integer, 1, &rank_dim, &out);
            mint *od = libData->MTensor_getIntegerData(out);
            for (u32 i = 0; i < v->shape.rank; i++)
                od[i] = (mint)v->shape.dims[i];
            MArgument_setMTensor(res, out);
            return LIBRARY_NO_ERROR;
        }
    }

    return LIBRARY_FUNCTION_ERROR;
}

// thvmTermTag[termId] → Integer (tag code)
EXTERN_C DLLEXPORT int thvmTermTag(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc;
    mint term_id = MArgument_getInteger(args[0]);
    Term t = get_term(term_id);
    MArgument_setInteger(res, (mint)term_tag(t));
    return LIBRARY_NO_ERROR;
}

// thvmTermExt[termId] → Integer (EXT field — UOp code for TAG_TOP, dtype for TAG_TEN)
EXTERN_C DLLEXPORT int thvmTermExt(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc;
    mint term_id = MArgument_getInteger(args[0]);
    Term t = get_term(term_id);
    MArgument_setInteger(res, (mint)term_ext(t));
    return LIBRARY_NO_ERROR;
}

// thvmTensorCount[] → Integer
EXTERN_C DLLEXPORT int thvmTensorCount(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args;
    if (!g_ctx) { MArgument_setInteger(res, 0); return LIBRARY_NO_ERROR; }
    MArgument_setInteger(res, (mint)g_ctx->tensor_count);
    return LIBRARY_NO_ERROR;
}

// ============================================================
// Core operations (TOp dispatch)
// ============================================================

// thvmOp[outId, uop, aId, bId] → Void
// bId == 0 means ERA (for unary ops)
EXTERN_C DLLEXPORT int thvmOp(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint uop    = MArgument_getInteger(args[1]);
    mint a_id   = MArgument_getInteger(args[2]);
    mint b_id   = MArgument_getInteger(args[3]);

    Term a = get_term(a_id);
    Term b = (b_id == 0) ? term_era() : get_term(b_id);

    Term result = thvm_op(g_ctx, (u32)uop, a, b);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Reduction
// ============================================================

// thvmReduce[termId, outId] → Void
EXTERN_C DLLEXPORT int thvmReduce(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint term_id = MArgument_getInteger(args[0]);
    mint out_id  = MArgument_getInteger(args[1]);

    Term t = get_term(term_id);
    Term result = thvm_reduce(g_ctx, t);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmRealize[termId] → Void (reduce in-place, update stored term)
EXTERN_C DLLEXPORT int thvmRealize(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint term_id = MArgument_getInteger(args[0]);
    Term t = get_term(term_id);
    Term result = thvm_reduce(g_ctx, t);
    set_term(term_id, result);

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Movement ops
// ============================================================

// thvmReshape[outId, termId, shape:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmReshape(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);
    MTensor shape_mt = MArgument_getMTensor(args[2]);

    mint *sd = libData->MTensor_getIntegerData(shape_mt);
    mint slen = libData->MTensor_getFlattenedLength(shape_mt);

    Shape s = {.rank = (u32)slen};
    for (mint i = 0; i < slen; i++) s.dims[i] = (u32)sd[i];

    Term result = thvm_reshape(g_ctx, get_term(term_id), s);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmExpand[outId, termId, shape:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmExpand(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);
    MTensor shape_mt = MArgument_getMTensor(args[2]);

    mint *sd = libData->MTensor_getIntegerData(shape_mt);
    mint slen = libData->MTensor_getFlattenedLength(shape_mt);

    Shape s = {.rank = (u32)slen};
    for (mint i = 0; i < slen; i++) s.dims[i] = (u32)sd[i];

    Term result = thvm_expand(g_ctx, get_term(term_id), s);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmPermute[outId, termId, axes:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmPermute(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);
    MTensor axes_mt = MArgument_getMTensor(args[2]);

    mint *ad = libData->MTensor_getIntegerData(axes_mt);
    mint alen = libData->MTensor_getFlattenedLength(axes_mt);

    u32 axes[MAX_DIM];
    for (mint i = 0; i < alen; i++) axes[i] = (u32)ad[i];

    Term result = thvm_permute(g_ctx, get_term(term_id), axes, (u32)alen);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmPad[outId, termId, pairs:{Integer,1}] → Void
// pairs = {lo0, hi0, lo1, hi1, ...} flattened
EXTERN_C DLLEXPORT int thvmPad(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);
    MTensor pairs_mt = MArgument_getMTensor(args[2]);

    mint *pd = libData->MTensor_getIntegerData(pairs_mt);
    mint plen = libData->MTensor_getFlattenedLength(pairs_mt);

    u32 pairs[MAX_DIM * 2];
    for (mint i = 0; i < plen; i++) pairs[i] = (u32)pd[i];

    Term result = thvm_pad(g_ctx, get_term(term_id), pairs, (u32)(plen / 2));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmShrink[outId, termId, pairs:{Integer,1}] → Void
// pairs = {lo0, hi0, lo1, hi1, ...} flattened
EXTERN_C DLLEXPORT int thvmShrink(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);
    MTensor pairs_mt = MArgument_getMTensor(args[2]);

    mint *pd = libData->MTensor_getIntegerData(pairs_mt);
    mint plen = libData->MTensor_getFlattenedLength(pairs_mt);

    u32 pairs[MAX_DIM * 2];
    for (mint i = 0; i < plen; i++) pairs[i] = (u32)pd[i];

    Term result = thvm_shrink(g_ctx, get_term(term_id), pairs, (u32)(plen / 2));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmSumAxes[outId, termId, axes:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmSumAxes(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);
    MTensor axes_mt = MArgument_getMTensor(args[2]);

    mint *ad = libData->MTensor_getIntegerData(axes_mt);
    mint alen = libData->MTensor_getFlattenedLength(axes_mt);

    u32 axes[MAX_DIM];
    for (mint i = 0; i < alen; i++) axes[i] = (u32)ad[i];

    Term result = thvm_sum_axes(g_ctx, get_term(term_id), axes, (u32)alen);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Interaction net primitives
// ============================================================

// thvmLam[lamOutId, varOutId, bodyId] → Void
EXTERN_C DLLEXPORT int thvmLam(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint lam_id  = MArgument_getInteger(args[0]);
    mint var_id  = MArgument_getInteger(args[1]);
    mint body_id = MArgument_getInteger(args[2]);

    Term var_out;
    Term lam = thvm_lam(g_ctx, &var_out, get_term(body_id));
    set_term(lam_id, lam);
    set_term(var_id, var_out);

    return LIBRARY_NO_ERROR;
}

// thvmApp[outId, funId, argId] → Void
EXTERN_C DLLEXPORT int thvmApp(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint fun_id = MArgument_getInteger(args[1]);
    mint arg_id = MArgument_getInteger(args[2]);

    Term result = thvm_app(g_ctx, get_term(fun_id), get_term(arg_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmSup[outId, aId, bId] → Void
EXTERN_C DLLEXPORT int thvmSup(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint a_id   = MArgument_getInteger(args[1]);
    mint b_id   = MArgument_getInteger(args[2]);

    Term result = thvm_sup(g_ctx, get_term(a_id), get_term(b_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmDup[zId, out0Id, out1Id] → Void
EXTERN_C DLLEXPORT int thvmDup(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint z_id    = MArgument_getInteger(args[0]);
    mint out0_id = MArgument_getInteger(args[1]);
    mint out1_id = MArgument_getInteger(args[2]);

    Term out0, out1;
    thvm_dup(g_ctx, get_term(z_id), &out0, &out1);
    set_term(out0_id, out0);
    set_term(out1_id, out1);

    return LIBRARY_NO_ERROR;
}

// thvmDefine[bodyId] → Integer (name)
EXTERN_C DLLEXPORT int thvmDefine(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint body_id = MArgument_getInteger(args[0]);
    u32 name = thvm_define(g_ctx, get_term(body_id));
    MArgument_setInteger(res, (mint)name);

    return LIBRARY_NO_ERROR;
}

// thvmRef[outId, name] → Void
EXTERN_C DLLEXPORT int thvmRef(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint name   = MArgument_getInteger(args[1]);

    Term result = thvm_ref(g_ctx, (u32)name);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Advanced ops
// ============================================================

// thvmWhere[outId, condId, thenId, elseId] → Void
EXTERN_C DLLEXPORT int thvmWhere(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint cond_id = MArgument_getInteger(args[1]);
    mint then_id = MArgument_getInteger(args[2]);
    mint else_id = MArgument_getInteger(args[3]);

    Term result = thvm_where(g_ctx, get_term(cond_id), get_term(then_id), get_term(else_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmAssign[outId, dstId, srcId] → Void
EXTERN_C DLLEXPORT int thvmAssign(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint dst_id = MArgument_getInteger(args[1]);
    mint src_id = MArgument_getInteger(args[2]);

    Term result = thvm_assign(g_ctx, get_term(dst_id), get_term(src_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmIfz[outId, counterId, zeroCaseId, succLamId] → Void
EXTERN_C DLLEXPORT int thvmIfz(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id      = MArgument_getInteger(args[0]);
    mint counter_id  = MArgument_getInteger(args[1]);
    mint zero_id     = MArgument_getInteger(args[2]);
    mint succ_id     = MArgument_getInteger(args[3]);

    Term result = thvm_ifz(g_ctx, get_term(counter_id), get_term(zero_id), get_term(succ_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmLogPrint[outId, termId] → Void
EXTERN_C DLLEXPORT int thvmLogPrint(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint term_id = MArgument_getInteger(args[1]);

    Term result = thvm_log_print(g_ctx, get_term(term_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Autograd
// ============================================================

// thvmSetRequiresGrad[termId] → Void
EXTERN_C DLLEXPORT int thvmSetRequiresGrad(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint term_id = MArgument_getInteger(args[0]);
    Term t = get_term(term_id);
    thvm_set_requires_grad(g_ctx, t);

    return LIBRARY_NO_ERROR;
}

// thvmGrad[outId, yId, xId] → Void
EXTERN_C DLLEXPORT int thvmGrad(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint y_id   = MArgument_getInteger(args[1]);
    mint x_id   = MArgument_getInteger(args[2]);

    Term result = thvm_grad(g_ctx, get_term(y_id), get_term(x_id));
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmGradMulti[outId, lossId, paramIds:{Integer,1}, slotIds:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmGradMulti(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint loss_id = MArgument_getInteger(args[1]);
    MTensor param_mt = MArgument_getMTensor(args[2]);
    MTensor slot_mt  = MArgument_getMTensor(args[3]);

    mint *param_ids = libData->MTensor_getIntegerData(param_mt);
    mint *slot_ids  = libData->MTensor_getIntegerData(slot_mt);
    mint n = libData->MTensor_getFlattenedLength(param_mt);

    Term params[64], slots[64];
    for (mint i = 0; i < n && i < 64; i++) {
        params[i] = get_term(param_ids[i]);
        slots[i]  = get_term(slot_ids[i]);
    }

    Term result = thvm_grad_multi(g_ctx, get_term(loss_id), params, slots, (u32)n);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmBackward[lossId, paramIds:{Integer,1}, gradOutIds:{Integer,1}] → Void
// Uses thvm_grad per-param (thvm_backward is declared but not implemented in TinyHVM).
EXTERN_C DLLEXPORT int thvmBackward(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint loss_id = MArgument_getInteger(args[0]);
    MTensor param_mt = MArgument_getMTensor(args[1]);
    MTensor grad_mt  = MArgument_getMTensor(args[2]);

    mint *param_ids = libData->MTensor_getIntegerData(param_mt);
    mint *grad_ids  = libData->MTensor_getIntegerData(grad_mt);
    mint n = libData->MTensor_getFlattenedLength(param_mt);

    Term loss = get_term(loss_id);
    for (mint i = 0; i < n && i < 64; i++) {
        Term grad = thvm_grad(g_ctx, loss, get_term(param_ids[i]));
        set_term(grad_ids[i], grad);
    }

    return LIBRARY_NO_ERROR;
}

// ============================================================
// CNN layers
// ============================================================

// thvmConv2d[outId, xId, wId, bId, groups, stride:{Integer,1}, padding:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmConv2d(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id  = MArgument_getInteger(args[0]);
    mint x_id    = MArgument_getInteger(args[1]);
    mint w_id    = MArgument_getInteger(args[2]);
    mint b_id    = MArgument_getInteger(args[3]);
    mint groups  = MArgument_getInteger(args[4]);
    MTensor stride_mt  = MArgument_getMTensor(args[5]);
    MTensor padding_mt = MArgument_getMTensor(args[6]);

    mint *sd = libData->MTensor_getIntegerData(stride_mt);
    mint *pd = libData->MTensor_getIntegerData(padding_mt);

    u32 stride[2]  = {(u32)sd[0], (u32)sd[1]};
    // padding is {top, bottom, left, right} = 4 elements
    mint plen = libData->MTensor_getFlattenedLength(padding_mt);
    u32 padding[4] = {0};
    for (mint i = 0; i < plen && i < 4; i++) padding[i] = (u32)pd[i];

    Term bias = (b_id == 0) ? term_era() : get_term(b_id);
    Term result = thvm_conv2d(g_ctx, get_term(x_id), get_term(w_id), bias,
                              (u32)groups, stride, padding);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmMaxPool2d[outId, xId, kernel:{Integer,1}, stride:{Integer,1}] → Void
EXTERN_C DLLEXPORT int thvmMaxPool2d(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id = MArgument_getInteger(args[0]);
    mint x_id   = MArgument_getInteger(args[1]);
    MTensor kernel_mt = MArgument_getMTensor(args[2]);
    MTensor stride_mt = MArgument_getMTensor(args[3]);

    mint *kd = libData->MTensor_getIntegerData(kernel_mt);
    mint *sd = libData->MTensor_getIntegerData(stride_mt);

    u32 kernel[2] = {(u32)kd[0], (u32)kd[1]};
    u32 stride[2] = {(u32)sd[0], (u32)sd[1]};

    Term result = thvm_maxpool2d(g_ctx, get_term(x_id), kernel, stride);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// thvmPool[outId, xId, kernel:{Integer,1}, stride:{Integer,1}, nSpatial] → Void
EXTERN_C DLLEXPORT int thvmPool(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint out_id    = MArgument_getInteger(args[0]);
    mint x_id      = MArgument_getInteger(args[1]);
    MTensor kern_mt  = MArgument_getMTensor(args[2]);
    MTensor stride_mt = MArgument_getMTensor(args[3]);
    mint n_spatial = MArgument_getInteger(args[4]);

    mint *kd = libData->MTensor_getIntegerData(kern_mt);
    mint *sd = libData->MTensor_getIntegerData(stride_mt);

    u32 kernel[MAX_DIM], stride[MAX_DIM];
    for (mint i = 0; i < n_spatial; i++) {
        kernel[i] = (u32)kd[i];
        stride[i] = (u32)sd[i];
    }

    Term result = thvm_pool(g_ctx, get_term(x_id), kernel, stride, (u32)n_spatial);
    set_term(out_id, result);

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Debug / profiling
// ============================================================

// thvmPrintTerm[termId] → Void
EXTERN_C DLLEXPORT int thvmPrintTerm(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint term_id = MArgument_getInteger(args[0]);
    thvm_print_term(g_ctx, get_term(term_id));

    return LIBRARY_NO_ERROR;
}

// thvmProfileReport[] → Void
EXTERN_C DLLEXPORT int thvmProfileReport(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;
    thvm_profile_report(g_ctx);
    return LIBRARY_NO_ERROR;
}

// thvmProfileReset[] → Void
EXTERN_C DLLEXPORT int thvmProfileReset(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;
    thvm_profile_reset(g_ctx);
    return LIBRARY_NO_ERROR;
}

// thvmHeapPos[] → Integer
EXTERN_C DLLEXPORT int thvmHeapPos(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args;
    if (!g_ctx) { MArgument_setInteger(res, 0); return LIBRARY_NO_ERROR; }
    MArgument_setInteger(res, (mint)g_ctx->heap_pos);
    return LIBRARY_NO_ERROR;
}

// thvmInteractionCount[] → Integer
EXTERN_C DLLEXPORT int thvmInteractionCount(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args;
    if (!g_ctx) { MArgument_setInteger(res, 0); return LIBRARY_NO_ERROR; }
    MArgument_setInteger(res, (mint)g_ctx->itrs);
    return LIBRARY_NO_ERROR;
}

// ============================================================
// Inet helpers for recursive training programs
// ============================================================

// thvmAllocDef[] → Integer (pre-allocate a definition slot)
EXTERN_C DLLEXPORT int thvmAllocDef(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;
    u32 name = g_ctx->def_count++;
    g_ctx->defs[name] = term_era();
    MArgument_setInteger(res, (mint)name);
    return LIBRARY_NO_ERROR;
}

// thvmSetDef[name, bodyId] → Void
EXTERN_C DLLEXPORT int thvmSetDef(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;
    mint name = MArgument_getInteger(args[0]);
    mint body_id = MArgument_getInteger(args[1]);
    g_ctx->defs[name] = get_term(body_id);
    return LIBRARY_NO_ERROR;
}

// thvmLamOpen[lamOutId, varOutId] → Void
// Creates a lambda with placeholder body (ERA). Use thvmLamSetBody to set body later.
EXTERN_C DLLEXPORT int thvmLamOpen(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint lam_id = MArgument_getInteger(args[0]);
    mint var_id = MArgument_getInteger(args[1]);

    u64 loc = heap_alloc(g_ctx, 2);
    Term var = term_new(TAG_VAR, 0, loc);
    heap_set(g_ctx, loc, term_set_sub(var));
    heap_set(g_ctx, loc + 1, term_era());
    Term lam = term_new(TAG_LAM, 0, loc);

    set_term(lam_id, lam);
    set_term(var_id, var);

    return LIBRARY_NO_ERROR;
}

// thvmLamSetBody[lamId, bodyId] → Void
EXTERN_C DLLEXPORT int thvmLamSetBody(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)res;
    if (!g_ctx) return LIBRARY_FUNCTION_ERROR;

    mint lam_id = MArgument_getInteger(args[0]);
    mint body_id = MArgument_getInteger(args[1]);

    Term lam = get_term(lam_id);
    u64 loc = term_val(lam);
    heap_set(g_ctx, loc + 1, get_term(body_id));

    return LIBRARY_NO_ERROR;
}

// ============================================================
// Extended profiling
// ============================================================

// thvmProfileEnable[] → Void (force-enable profiling without env var)
EXTERN_C DLLEXPORT int thvmProfileEnable(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args; (void)res;
    thvm_prof_global.enabled = 1;
    return LIBRARY_NO_ERROR;
}

// thvmProfileData[] → NumericArray (Real64)
// Layout: [uop_ns(29), uop_cnt(29), uop_tensors(29), phase_ns(5),
//          buf_bytes_alloc, buf_alloc_cnt, buf_bytes_peak, buf_bytes_current,
//          tensor_peak, tensor_created, tensor_freed, heap_peak, heap_at_reset,
//          cpu_read_bytes, cpu_read_cnt, cpu_write_bytes, cpu_write_cnt]
#define PROF_EXPORT_N UOP_COUNT
#define PROF_DATA_LEN (PROF_EXPORT_N * 3 + PHASE_COUNT + 13)
EXTERN_C DLLEXPORT int thvmProfileData(
    WolframLibraryData libData, mint argc, MArgument *args, MArgument res)
{
    (void)libData; (void)argc; (void)args;
    mint dims[1] = {PROF_DATA_LEN};
    MNumericArray out;
    int err = g_na_funcs->MNumericArray_new(MNumericArray_Type_Real64, 1, dims, &out);
    if (err) return LIBRARY_FUNCTION_ERROR;

    double *d = (double *)g_na_funcs->MNumericArray_getData(out);
    ThvmProfile *p = &thvm_prof_global;
    int idx = 0;
    for (int i = 0; i < PROF_EXPORT_N; i++) d[idx++] = (double)p->uop_ns[i];
    for (int i = 0; i < PROF_EXPORT_N; i++) d[idx++] = (double)p->uop_cnt[i];
    for (int i = 0; i < PROF_EXPORT_N; i++) d[idx++] = (double)p->uop_tensors[i];
    for (int i = 0; i < PHASE_COUNT; i++) d[idx++] = (double)p->phase_ns[i];
    d[idx++] = (double)p->buf_bytes_alloc;
    d[idx++] = (double)p->buf_alloc_cnt;
    d[idx++] = (double)p->buf_bytes_peak;
    d[idx++] = (double)p->buf_bytes_current;
    d[idx++] = (double)p->tensor_peak;
    d[idx++] = (double)p->tensor_created;
    d[idx++] = (double)p->tensor_freed;
    d[idx++] = (double)p->heap_peak;
    d[idx++] = (double)p->heap_at_reset;
    d[idx++] = (double)p->cpu_read_bytes;
    d[idx++] = (double)p->cpu_read_cnt;
    d[idx++] = (double)p->cpu_write_bytes;
    d[idx++] = (double)p->cpu_write_cnt;

    MArgument_setMNumericArray(res, out);
    return LIBRARY_NO_ERROR;
}
