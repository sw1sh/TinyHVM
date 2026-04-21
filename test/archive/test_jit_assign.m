// test_jit_assign.m — Ultra-minimal JIT: just ASSIGN(W, W - 0.2)
#include "../src/tinyhvm.c"
#include "../src/backend/cpu/_.c"
#ifdef __APPLE__
  #include "../src/backend/metal/_.m"
#endif
#include "../src/nn/_.c"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    TinyHVM *ctx = thvm_init("metal");

    // W = [1, 2, 3, 4, 5, 6]
    f32 wdata[] = {1, 2, 3, 4, 5, 6};
    Term w = thvm_tensor(ctx, wdata, SHAPE(6));

    // delta = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    f32 ddata[] = {0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
    Term delta = thvm_tensor(ctx, ddata, SHAPE(6));

    // Build: ASSIGN(W, W - delta)
    Term step = thvm_assign(ctx, w, thvm_op(ctx, UOP_SUB, w, delta));

    u32 n_persistent = ctx->tensor_count;
    printf("n_persistent=%u, metal_pool.count=%u\n", n_persistent, metal_pool.count);

    u32 w_bufid = ctx->tensors[(u32)term_val(w)].buf_id;
    f32 *wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("W init: [%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);

    // Capture
    jit_begin_capture(n_persistent);
    thvm_reduce(ctx, step);
    jit_end_capture();

    jit_flush();
    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("W after capture: [%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);

    // Dump JIT
    printf("\n--- JIT slots ---\n");
    for (u32 i = 0; i < jit.n_slots; i++)
        printf("  slot %2u: buf_id=%2u alloc=%llu %s\n",
               i, jit.slots[i].buf_id, jit.slots[i].alloc_size,
               jit.slots[i].persistent ? "PERS" : "eph");
    printf("\n--- JIT commands ---\n");
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        if (cmd->is_blit) {
            printf("  cmd %u: BLIT dst_slot=%u(b%u) src_slot=%u(b%u) size=%llu\n",
                   ci, cmd->blit_dst_slot, jit.slots[cmd->blit_dst_slot].buf_id,
                   cmd->blit_src_slot, jit.slots[cmd->blit_src_slot].buf_id,
                   cmd->blit_size);
        } else {
            printf("  cmd %u: COMPUTE bufs=[", ci);
            for (u32 i = 0; i < cmd->n_bufs; i++)
                printf("s%u(b%u)%s", cmd->buf_slots[i], jit.slots[cmd->buf_slots[i]].buf_id,
                       i < cmd->n_bufs - 1 ? "," : "");
            printf("] grid=%u,%u,%u\n", cmd->grid[0], cmd->grid[1], cmd->grid[2]);
        }
    }

    // Reset
    extern u32 total_dispatches; total_dispatches = 0;
    thvm_reset(ctx, n_persistent);

    // Replay 1
    jit_replay();
    jit_flush();
    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("\nW after replay1: [%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);

    // Replay 2
    jit_replay();
    jit_flush();
    wp = (f32 *)metal_pool.bufs[w_bufid].contents;
    printf("W after replay2: [%.2f,%.2f,%.2f,%.2f,%.2f,%.2f]\n", wp[0],wp[1],wp[2],wp[3],wp[4],wp[5]);

    thvm_free(ctx);
    return 0;
}
