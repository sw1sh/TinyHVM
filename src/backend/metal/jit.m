// metal/jit.m — JIT capture/replay for training loops
// Records GPU dispatch sequence on step 0, replays for steps 1+.
// JITState, JITCmd, JITSlot defined in init.m.

static u32 jit_slot_for_buf(u32 buf_id) {
    for (u32 i = 0; i < jit.n_slots; i++)
        if (jit.slots[i].buf_id == buf_id) return i;
    assert(jit.n_slots < JIT_MAX_SLOTS);
    u32 s = jit.n_slots++;
    jit.slots[s].buf_id = buf_id;
    jit.slots[s].alloc_size = metal_pool.sizes[buf_id];
    jit.slots[s].persistent = 0;
    return s;
}

void jit_begin_capture(u32 n_persistent_bufs) {
    memset(&jit, 0, sizeof(jit));
    jit.state = JIT_CAPTURE;
    jit.persistent_count = n_persistent_bufs;
    for (u32 i = 0; i < n_persistent_bufs && i < JIT_MAX_SLOTS; i++) {
        jit.slots[i].buf_id = i;
        jit.slots[i].alloc_size = metal_pool.sizes[i];
        jit.slots[i].persistent = 1;
    }
    jit.n_slots = n_persistent_bufs;
}

static void jit_record_dispatch_1d(id<MTLComputePipelineState> pipe,
                                     id<MTLBuffer> *bufs, u32 n_bufs,
                                     const void **params, u64 *param_sizes, u32 n_params,
                                     u32 gw, u32 gh, u32 gd, u32 tw, u32 th, u32 td) {
    if (jit.n_cmds >= JIT_MAX_CMDS) return;
    JITCmd *cmd = &jit.cmds[jit.n_cmds++];
    memset(cmd, 0, sizeof(*cmd));
    cmd->pipe = pipe;
    cmd->n_bufs = n_bufs;
    cmd->is_mps = 0;
    for (u32 i = 0; i < n_bufs; i++) {
        for (u32 b = 0; b < metal_pool.count; b++) {
            if (metal_pool.bufs[b] == bufs[i]) {
                cmd->buf_slots[i] = jit_slot_for_buf(b);
                break;
            }
        }
    }
    u32 offset = 0;
    cmd->n_params = n_params;
    for (u32 i = 0; i < n_params && params; i++) {
        u32 sz = (u32)param_sizes[i];
        assert(offset + sz <= JIT_MAX_PARAMS);
        memcpy(cmd->params + offset, params[i], sz);
        cmd->param_sizes[i] = sz;
        offset += sz;
    }
    cmd->grid[0] = gw; cmd->grid[1] = gh; cmd->grid[2] = gd;
    cmd->tg[0] = tw; cmd->tg[1] = th; cmd->tg[2] = td;
}

static void jit_record_mps(u32 dst_buf, u32 a_buf, u32 b_buf, u32 M, u32 K, u32 N,
                            BOOL trans_a, BOOL trans_b,
                            u32 phys_a_rows, u32 phys_a_cols, u32 phys_b_rows, u32 phys_b_cols) {
    if (jit.n_cmds >= JIT_MAX_CMDS) return;
    JITCmd *cmd = &jit.cmds[jit.n_cmds++];
    memset(cmd, 0, sizeof(*cmd));
    cmd->is_mps = 1;
    cmd->mps_m = M; cmd->mps_k = K; cmd->mps_n = N;
    cmd->mps_dst_slot = jit_slot_for_buf(dst_buf);
    cmd->mps_a_slot = jit_slot_for_buf(a_buf);
    cmd->mps_b_slot = jit_slot_for_buf(b_buf);
    cmd->mps_trans_a = trans_a; cmd->mps_trans_b = trans_b;
    cmd->mps_phys_a_rows = phys_a_rows; cmd->mps_phys_a_cols = phys_a_cols;
    cmd->mps_phys_b_rows = phys_b_rows; cmd->mps_phys_b_cols = phys_b_cols;
}

void jit_end_capture(void) {
    jit.state = JIT_OFF;
    fprintf(stderr, "JIT captured: %u cmds, %u slots (%u persistent)\n",
            jit.n_cmds, jit.n_slots, jit.persistent_count);
}

void jit_replay(void) {
    // Allocate ephemeral buffers (reuse from pool)
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
        jit.slots[i].buf_id = metal_buf_alloc(jit.slots[i].alloc_size);
    }

    // Encode all commands
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cmd = &jit.cmds[ci];
        if (cmd->is_blit) {
            if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
            if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];
            u32 dst = jit.slots[cmd->blit_dst_slot].buf_id;
            u32 src = jit.slots[cmd->blit_src_slot].buf_id;
            id<MTLBlitCommandEncoder> blit = [batch_cmd blitCommandEncoder];
            [blit copyFromBuffer:metal_pool.bufs[src] sourceOffset:0
                        toBuffer:metal_pool.bufs[dst] destinationOffset:0
                            size:cmd->blit_size];
            [blit endEncoding];
            batch_dirty = 1;
        } else if (cmd->is_mps) {
            if (batch_encoder) { [batch_encoder endEncoding]; batch_encoder = nil; }
            if (!batch_cmd) batch_cmd = [mtl_queue commandBuffer];
            u32 dst = jit.slots[cmd->mps_dst_slot].buf_id;
            u32 a = jit.slots[cmd->mps_a_slot].buf_id;
            u32 b = jit.slots[cmd->mps_b_slot].buf_id;
            MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:cmd->mps_phys_a_rows columns:cmd->mps_phys_a_cols rowBytes:cmd->mps_phys_a_cols*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:cmd->mps_phys_b_rows columns:cmd->mps_phys_b_cols rowBytes:cmd->mps_phys_b_cols*4 dataType:MPSDataTypeFloat32];
            MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:cmd->mps_m columns:cmd->mps_n rowBytes:cmd->mps_n*4 dataType:MPSDataTypeFloat32];
            MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[a] descriptor:dA];
            MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[b] descriptor:dB];
            MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:metal_pool.bufs[dst] descriptor:dC];
            MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:mtl_dev transposeLeft:cmd->mps_trans_a transposeRight:cmd->mps_trans_b resultRows:cmd->mps_m resultColumns:cmd->mps_n interiorColumns:cmd->mps_k alpha:1.0 beta:0.0];
            [mm encodeToCommandBuffer:batch_cmd leftMatrix:mA rightMatrix:mB resultMatrix:mC];
            batch_dirty = 1;
        } else {
            id<MTLComputeCommandEncoder> enc = get_encoder();
            [enc setComputePipelineState:cmd->pipe];
            for (u32 i = 0; i < cmd->n_bufs; i++)
                [enc setBuffer:metal_pool.bufs[jit.slots[cmd->buf_slots[i]].buf_id] offset:0 atIndex:i];
            u32 poff = 0;
            for (u32 i = 0; i < cmd->n_params; i++) {
                [enc setBytes:cmd->params + poff length:cmd->param_sizes[i] atIndex:cmd->n_bufs + i];
                poff += cmd->param_sizes[i];
            }
            [enc dispatchThreads:MTLSizeMake(cmd->grid[0], cmd->grid[1], cmd->grid[2])
               threadsPerThreadgroup:MTLSizeMake(cmd->tg[0], cmd->tg[1], cmd->tg[2])];
            batch_dirty = 1;
        }
    }
    total_dispatches += jit.n_cmds;

    // Free ephemeral buffers (return to pool)
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
        metal_buf_free(jit.slots[i].buf_id);
    }
}
