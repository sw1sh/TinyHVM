// test_jit_tiny.m — Absolute minimal JIT: W = W - lr * grad(loss, W)
// loss = sum(X @ W), SGD. No softmax, no BN, no one-hot.
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    srand(42);
    TinyHVM *ctx = thvm_init("metal");
    u32 BS = 4, D = 3, K = 2;

    // Weight W [D, K]
    f32 wdata_init[] = {1, 2, 3, 4, 5, 6};
    Term w = thvm_tensor(ctx, wdata_init, SHAPE(D, K));
    thvm_set_requires_grad(ctx, w);

    // Fixed input buffer X [BS, D]
    f32 xdata[] = {1,0,0, 0,1,0, 0,0,1, 1,1,1};
    Term x = thvm_tensor(ctx, xdata, SHAPE(BS, D));
    thvm_set_requires_grad(ctx, x);

    // LR
    f32 lr_val = 0.1f;
    Term lr = thvm_tensor(ctx, &lr_val, SHAPE(1));

    // Grad accumulator
    f32 gz[6] = {0};
    Term gw = thvm_tensor(ctx, gz, SHAPE(D, K));

    printf("tensor_count=%u, metal_pool.count=%u\n", ctx->tensor_count, metal_pool.count);

    // Build graph: loss = sum(X @ W)
    Term mm = thvm_op(ctx, UOP_MM, x, w);  // [BS, K]
    Term loss = thvm_sum_axes(ctx, mm, (u32[]){0, 1}, 2);  // scalar
    Term grad_term = thvm_grad_multi(ctx, loss, &w, &gw, 1);
    Term sgd = thvm_assign(ctx, w,
        thvm_op(ctx, UOP_SUB, w, thvm_op(ctx, UOP_MUL, lr, gw)));
    Term train_step = thvm_app(ctx, grad_term, sgd);

    u32 n_persistent = ctx->tensor_count;
    printf("after graph: tensor_count=%u, metal_pool.count=%u\n", n_persistent, metal_pool.count);

    // Read initial W
    u32 w_bufid = ctx->tensors[(u32)term_val(w)].buf_id;
    f32 *wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("W init: [%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);

    // Zero grad
    u32 gw_bufid = ctx->tensors[(u32)term_val(gw)].buf_id;
    memset(metal_pool.bufs[gw_bufid].contents, 0, 6 * sizeof(f32));

    // Capture
    jit_begin_capture(n_persistent);
    thvm_reduce(ctx, train_step);
    jit_end_capture();

    jit_flush();
    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    f32 *gp = (f32 *)metal_pool.bufs[gw_bufid].contents;
    printf("W after capture: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);
    printf("grad after capture: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", gp[0],gp[1],gp[2],gp[3],gp[4],gp[5]);

    // Dump slot table
    printf("\n--- JIT slots ---\n");
    for (u32 i = 0; i < jit.n_slots; i++) {
        printf("  slot %2u: buf_id=%2u alloc=%llu %s\n",
               i, jit.slots[i].buf_id, jit.slots[i].alloc_size,
               jit.slots[i].persistent ? "PERSISTENT" : "ephemeral");
    }
    printf("\n--- JIT commands ---\n");
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        if (cmd->is_blit) {
            printf("  cmd %2u: BLIT dst_slot=%u(buf=%u) src_slot=%u(buf=%u) size=%llu\n",
                   ci, cmd->blit_dst_slot, jit.slots[cmd->blit_dst_slot].buf_id,
                   cmd->blit_src_slot, jit.slots[cmd->blit_src_slot].buf_id,
                   cmd->blit_size);
        } else if (cmd->is_mps) {
            printf("  cmd %2u: MPS dst=%u a=%u b=%u M=%u K=%u N=%u\n",
                   ci, cmd->mps_dst_slot, cmd->mps_a_slot, cmd->mps_b_slot,
                   cmd->mps_m, cmd->mps_k, cmd->mps_n);
        } else {
            printf("  cmd %2u: COMPUTE bufs=[", ci);
            for (u32 i = 0; i < cmd->n_bufs; i++)
                printf("%u(b%u)%s", cmd->buf_slots[i], jit.slots[cmd->buf_slots[i]].buf_id,
                       i < cmd->n_bufs - 1 ? "," : "");
            printf("] grid=%u,%u,%u\n", cmd->grid[0], cmd->grid[1], cmd->grid[2]);
        }
    }

    // Key buffers
    printf("\nKey buf_ids: W=%u gw=%u x=%u lr=%u\n",
           ctx->tensors[(u32)term_val(w)].buf_id,
           ctx->tensors[(u32)term_val(gw)].buf_id,
           ctx->tensors[(u32)term_val(x)].buf_id,
           ctx->tensors[(u32)term_val(lr)].buf_id);

    // Check pipeline state pointers against known pipelines
    printf("\nPipeline states:\n");
    printf("  Known: zero_fill=%p fast_sub=%p fast_mul=%p fast_add=%p\n",
           (void*)pipe_zero_fill, (void*)fpipe_sub, (void*)fpipe_mul, (void*)fpipe_add);
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        if (!cmd->is_blit && !cmd->is_mps) {
            const char *name = "unknown";
            if (cmd->pipe == pipe_zero_fill) name = "ZERO_FILL!";
            else if (cmd->pipe == fpipe_sub) name = "fast_sub";
            else if (cmd->pipe == fpipe_mul) name = "fast_mul";
            else if (cmd->pipe == fpipe_add) name = "fast_add";
            else if (cmd->pipe == pipe_reduce_sum) name = "reduce_sum";
            else if (cmd->pipe == pipe_reduce_sum_par) name = "reduce_sum_par";
            printf("  cmd %u: pipe=%p (%s)\n", ci, (void*)cmd->pipe, name);
        }
    }

    // Dump params for ALL commands
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        if (cmd->is_blit || cmd->is_mps) continue;
        printf("\ncmd %u: n_bufs=%u n_params=%u grid=%u\n", ci, cmd->n_bufs, cmd->n_params, cmd->grid[0]);
        u32 poff = 0;
        for (u32 pi = 0; pi < cmd->n_params; pi++) {
            u32 sz = cmd->param_sizes[pi];
            printf("  param%u: %u bytes", pi, sz);
            if (sz == sizeof(ViewParams)) {
                ViewParams *vp = (ViewParams *)(cmd->params + poff);
                printf(" VP: shape=[%u", vp->shape[0]);
                for (u32 d = 1; d < vp->rank; d++) printf(",%u", vp->shape[d]);
                printf("] strides=[%d", vp->strides[0]);
                for (u32 d = 1; d < vp->rank; d++) printf(",%d", vp->strides[d]);
                printf("] numel=%u rank=%u", vp->numel, vp->rank);
            } else if (sz == 4) {
                printf(" u32=%u", *(u32*)(cmd->params + poff));
            }
            printf("\n");
            poff += sz;
        }
        if (cmd->n_params == 0) printf("  (no params)\n");
    }

    // Reset
    extern u32 total_dispatches; total_dispatches = 0;
    thvm_reset(ctx, n_persistent);

    // Verify W survives reset
    jit_flush();
    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("W after reset: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);

    // Check buffer identity
    printf("\nMTLBuffer pointers:\n");
    for (u32 i = 0; i < 15; i++) {
        if (metal_pool.bufs[i])
            printf("  buf %u: %p (size=%llu)\n", i, (void*)metal_pool.bufs[i], metal_pool.sizes[i]);
    }

    // Zero grad and replay
    memset(metal_pool.bufs[gw_bufid].contents, 0, 6 * sizeof(f32));
    extern int jit_debug_replay;
    jit_debug_replay = 1;

    // Print ephemeral buffer pointers after alloc (before replay commands)
    printf("Ephemeral MTLBuffer pointers after alloc:\n");
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
        u32 bid = jit.slots[i].buf_id;
        printf("  slot %u (buf %u): %p (size=%llu)\n", i, bid, (void*)metal_pool.bufs[bid], metal_pool.sizes[bid]);
    }
    // Check for aliasing with persistent buffers
    printf("Checking for MTLBuffer aliasing with persistent bufs...\n");
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
        u32 ebid = jit.slots[i].buf_id;
        for (u32 j = 1; j < jit.persistent_count; j++) {
            u32 pbid = jit.slots[j].buf_id;
            if (metal_pool.bufs[ebid] && metal_pool.bufs[pbid] &&
                metal_pool.bufs[ebid] == metal_pool.bufs[pbid]) {
                printf("  !!! ALIAS: eph slot %u (buf %u) == pers slot %u (buf %u) at %p\n",
                       i, ebid, j, pbid, (void*)metal_pool.bufs[ebid]);
            }
        }
    }

    printf("\nEph slots before replay:\n");
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++)
        printf("  slot %u: buf_id=%u alloc=%llu\n", i, jit.slots[i].buf_id, jit.slots[i].alloc_size);

    jit_replay();
    jit_flush();

    printf("\nEph buffer contents after replay:\n");
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
        u32 bid = jit.slots[i].buf_id;
        u32 nf = (u32)(jit.slots[i].alloc_size / 4);
        f32 *bp = (f32 *)metal_pool.bufs[bid].contents;
        printf("  slot %u (buf %u, %llu B): [", i, bid, jit.slots[i].alloc_size);
        for (u32 j = 0; j < nf && j < 8; j++) printf("%.4f%s", bp[j], j<nf-1?",":"");
        printf("]\n");
    }

    // Also check persistent buffers
    printf("\nPers buffer contents after replay:\n");
    for (u32 i = 0; i < jit.persistent_count; i++) {
        u32 bid = jit.slots[i].buf_id;
        if (!metal_pool.bufs[bid]) continue;
        u32 nf = (u32)(jit.slots[i].alloc_size / 4);
        if (nf == 0 || nf > 16) continue;
        f32 *bp = (f32 *)metal_pool.bufs[bid].contents;
        printf("  slot %u (buf %u, %llu B): [", i, bid, jit.slots[i].alloc_size);
        for (u32 j = 0; j < nf && j < 8; j++) printf("%.4f%s", bp[j], j<nf-1?",":"");
        printf("]\n");
    }

    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    gp = (f32 *)metal_pool.bufs[gw_bufid].contents;
    printf("\nW after replay1: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);
    printf("grad after replay1: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", gp[0],gp[1],gp[2],gp[3],gp[4],gp[5]);

    // Replay 2
    memset(metal_pool.bufs[gw_bufid].contents, 0, 6 * sizeof(f32));
    jit_replay();
    jit_flush();

    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    gp = (f32 *)metal_pool.bufs[gw_bufid].contents;
    printf("W after replay2: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);
    printf("grad after replay2: [%.4f,%.4f,%.4f,%.4f,%.4f,%.4f]\n", gp[0],gp[1],gp[2],gp[3],gp[4],gp[5]);

    thvm_free(ctx);
    return 0;
}
