// metal/jit.m — JIT capture/replay for training loops
// Records GPU dispatch sequence on step 0, replays for steps 1+.
// JITState, JITCmd, JITSlot defined in init.m.

int jit_debug_replay = 0;
double jit_cap_sums[2048];

static u32 jit_slot_for_buf(u32 buf_id) {
    for (u32 i = 0; i < jit.n_slots; i++)
        if (jit.slots[i].buf_id == buf_id) return i;
    assert(jit.n_slots < JIT_MAX_SLOTS);
    u32 s = jit.n_slots++;
    jit.slots[s].buf_id = buf_id;
    jit.slots[s].alloc_size = metal_pool.sizes[buf_id];
    jit.slots[s].persistent = 0;
    if (s == 188 || (metal_pool.sizes[buf_id] == 32 && jit.state == JIT_CAPTURE))
        fprintf(stderr, "  [JIT] slot %u for buf_id=%u (%lluB) at cmd %u\n", s, buf_id, metal_pool.sizes[buf_id], jit.n_cmds);
    return s;
}

void jit_begin_capture(u32 n_persistent_hint) {
    (void)n_persistent_hint;
    memset(&jit, 0, sizeof(jit));
    jit.state = JIT_CAPTURE;
    // Use actual buffer count — tensor_count differs due to view sharing
    u32 n_bufs = metal_pool.count;
    jit.persistent_count = n_bufs;
    for (u32 i = 0; i < n_bufs && i < JIT_MAX_SLOTS; i++) {
        jit.slots[i].buf_id = i;
        jit.slots[i].alloc_size = metal_pool.sizes[i];
        jit.slots[i].persistent = 1;
    }
    jit.n_slots = n_bufs;
}

// Record a dispatch using buf_ids directly (avoids MTLBuffer pointer aliasing)
static void jit_record_dispatch_ids(id<MTLComputePipelineState> pipe,
                                      u32 *buf_ids, u32 n_bufs,
                                      const void **params, u64 *param_sizes, u32 n_params,
                                      u32 gw, u32 gh, u32 gd, u32 tw, u32 th, u32 td) {
    if (jit.n_cmds >= JIT_MAX_CMDS) return;
    JITCmd *cmd = &jit.cmds[jit.n_cmds++];
    memset(cmd, 0, sizeof(*cmd));
    cmd->pipe = pipe;
    cmd->n_bufs = n_bufs > 24 ? 24 : n_bufs;
    cmd->is_mps = 0;
    for (u32 i = 0; i < cmd->n_bufs; i++) {
        cmd->buf_slots[i] = jit_slot_for_buf(buf_ids[i]);
        if (buf_ids[i] == 266)
            fprintf(stderr, "  [REC] cmd %u buf[%u]=266 → slot %u (n_bufs=%u)\n",
                    jit.n_cmds-1, i, cmd->buf_slots[i], cmd->n_bufs);
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

// Legacy: record using MTLBuffer pointers (searches pool for buf_ids)
static void jit_record_dispatch_1d(id<MTLComputePipelineState> pipe,
                                     id<MTLBuffer> *bufs, u32 n_bufs,
                                     const void **params, u64 *param_sizes, u32 n_params,
                                     u32 gw, u32 gh, u32 gd, u32 tw, u32 th, u32 td) {
    // Convert MTLBuffer pointers to buf_ids by searching from END (most recent first)
    u32 ids[24];
    u32 nb = n_bufs > 24 ? 24 : n_bufs;
    for (u32 i = 0; i < nb; i++) {
        ids[i] = 0;
        for (u32 b = metal_pool.count; b > 0; b--) {
            if (metal_pool.bufs[b-1] == bufs[i]) { ids[i] = b-1; break; }
        }
    }
    jit_record_dispatch_ids(pipe, ids, n_bufs, params, param_sizes, n_params,
                            gw, gh, gd, tw, th, td);
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
    // Save constant data from small ephemeral buffers (CPU-written scalars
    // created during backward that no GPU command produces).
    if (batch_dirty) metal_flush(); // ensure all GPU writes complete
    // Save output sums for each command (for capture vs replay comparison)
    if (batch_dirty) metal_flush();
    // Build set of slots written by GPU commands
    u8 slot_written[JIT_MAX_SLOTS] = {0};
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cm = &jit.cmds[ci];
        if (cm->is_blit) { slot_written[cm->blit_dst_slot] = 1; }
        else if (cm->is_mps) { slot_written[cm->mps_dst_slot] = 1; }
        else if (cm->n_bufs > 0) {
            // Only mark buffer(0) = primary output as written.
            // Side outputs at buffer(1+) are also written but we can't distinguish
            // them from inputs. Constants are never at buffer(0), so this is safe.
            slot_written[cm->buf_slots[0]] = 1;
        }
    }

    // Per-command capture sums were recorded inline (jit_debug_replay==5 in codegen.m).

    // Map buf_ids to slots for constants saved during capture (by metal_buf_write).
    // Also check: how many ephemeral slots SHOULD have constants?
    u32 n_valid = 0, n_eph_small = 0;
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        if (jit.slots[s].alloc_size <= JIT_CONST_MAX_BYTES) n_eph_small++;
    }
    for (u32 ci = 0; ci < jit.n_consts; ci++) {
        u32 bid = jit.consts[ci].slot; // temporarily stores buf_id
        for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
            if (jit.slots[s].buf_id == bid) {
                jit.consts[n_valid] = jit.consts[ci];
                jit.consts[n_valid].slot = s;
                n_valid++;
                break;
            }
        }
    }
    jit.n_consts = n_valid;
    // Count: how many small eph slots SHOULD have been saved (buf_id was in a buf_write)?
    u32 n_should_save = 0;
    for (u32 s = jit.persistent_count; s < jit.n_slots; s++) {
        if (jit.slots[s].alloc_size > JIT_CONST_MAX_BYTES) continue;
        u32 bid = jit.slots[s].buf_id;
        // Check if this buf_id had a buf_write
        int was_written = 0;
        for (u32 ci = 0; ci < n_valid; ci++) {
            if (jit.consts[ci].slot == s) { was_written = 1; break; }
        }
        if (!was_written) {
            // Check if buf_id appears in any command's constants list
            // (buf_ids that had buf_write but weren't mapped)
            // Actually, let me just count
            n_should_save++;
        }
    }
    fprintf(stderr, "  %u eph small slots, %u saved, %u small WITHOUT saved data\n",
            n_eph_small, n_valid, n_should_save);
    // Check for ephemeral buffers > 32B that aren't written by any cmd
    u32 cmd_outputs[2048] = {0}; // track which slots are written by commands
    for (u32 ci = 0; ci < jit.n_cmds; ci++) {
        JITCmd *cm = &jit.cmds[ci];
        if (cm->is_blit) cmd_outputs[cm->blit_dst_slot] = 1;
        else if (cm->is_mps) cmd_outputs[cm->mps_dst_slot] = 1;
        else if (cm->n_bufs > 0) {
            // Mark ALL buffers as potentially written (primary + side outputs)
            // We can't distinguish side outputs from inputs, so mark them all
            // and then subtract known inputs later
            for (u32 bi = 0; bi < cm->n_bufs; bi++)
                cmd_outputs[cm->buf_slots[bi]] = 1;
        }
    }
    u32 n_unwritten = 0;
    for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
        if (!cmd_outputs[i] && jit.slots[i].alloc_size > JIT_CONST_MAX_BYTES) {
            n_unwritten++;
            if (n_unwritten <= 5)
                fprintf(stderr, "  WARN: eph slot %u (buf %u, %llu B) not written by any cmd\n",
                        i, jit.slots[i].buf_id, jit.slots[i].alloc_size);
        }
    }
    if (n_unwritten > 5) fprintf(stderr, "  ... %u more unwritten eph slots\n", n_unwritten - 5);

    // Print constant summary
    for (u32 ci = 0; ci < jit.n_consts && ci < 10; ci++) {
        JITConst *c = &jit.consts[ci];
        f32 v0 = 0; memcpy(&v0, c->data, 4);
        fprintf(stderr, "  const[%u]: slot=%u size=%u val=[%.6f", ci, c->slot, c->size, v0);
        if (c->size > 4) { f32 v1; memcpy(&v1, c->data+4, 4); fprintf(stderr, ",%.6f", v1); }
        fprintf(stderr, "]\n");
    }
    if (jit.n_consts > 10) fprintf(stderr, "  ... %u more\n", jit.n_consts - 10);
    fprintf(stderr, "JIT captured: %u cmds, %u slots (%u persistent), %u consts, %u unwritten\n",
            jit.n_cmds, jit.n_slots, jit.persistent_count, jit.n_consts, n_unwritten);
}

// Flush previous replay's GPU work — call BEFORE overwriting shared buffers.
void jit_flush(void) {
    if (batch_dirty) metal_flush();
}

// (jit_debug_replay and jit_cap_sums declared at top of file)

void jit_replay(void) {
    // Allocate ephemeral buffers ONCE — reuse across replays.
    if (!jit.ephemeral_ready) {
        for (u32 i = jit.persistent_count; i < jit.n_slots; i++) {
            jit.slots[i].buf_id = metal_buf_alloc(jit.slots[i].alloc_size);
        }
        jit.ephemeral_ready = 1;
    }
    // Restore constant data (CPU-written values not produced by GPU commands)
    for (u32 ci = 0; ci < jit.n_consts; ci++) {
        u32 bid = jit.slots[jit.consts[ci].slot].buf_id;
        memcpy(metal_pool.bufs[bid].contents, jit.consts[ci].data, jit.consts[ci].size);
    }
    // (nuclear restore removed — not the issue)

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
        // Debug: capture vs replay comparison
        static double jit_cap_abssum[2048];
        if (jit_debug_replay == 3) {
            // Phase 3: save capture output sums (called during jit_replay while capturing)
            metal_flush();
            u32 obid = 0; u64 osz = 0;
            if (cmd->is_blit) { obid = jit.slots[cmd->blit_dst_slot].buf_id; osz = cmd->blit_size; }
            else if (cmd->is_mps) { obid = jit.slots[cmd->mps_dst_slot].buf_id; osz = (u64)cmd->mps_m*cmd->mps_n*4; }
            else if (cmd->n_bufs > 0) { obid = jit.slots[cmd->buf_slots[0]].buf_id; osz = metal_pool.sizes[obid]; }
            if (obid && osz > 0 && metal_pool.bufs[obid]) {
                f32 *od = (f32*)metal_pool.bufs[obid].contents;
                u32 nf = (u32)(osz/4);
                // (using file-scope jit_cap_abssum)
                double s = 0;
                for (u32 j = 0; j < nf && j < 100000; j++) s += fabs(od[j]);
                jit_cap_abssum[ci] = s;
            }
        }
        if (jit_debug_replay && cmd->is_blit) {
            u32 dst = jit.slots[cmd->blit_dst_slot].buf_id;
            u32 src = jit.slots[cmd->blit_src_slot].buf_id;
            if (dst < jit.persistent_count || ci == 233) {
                metal_flush();
                f32 *sd = (f32*)metal_pool.bufs[src].contents;
                fprintf(stderr, "  BLIT cmd %u: dst_slot=%u→b%u(%s) src_slot=%u→b%u [%.4f,%.4f] size=%llu\n",
                        ci, cmd->blit_dst_slot, dst, dst<jit.persistent_count?"PERS":"eph",
                        cmd->blit_src_slot, src, sd[0], sd[1], cmd->blit_size);
            }
        }
        if (jit_debug_replay == 4) {
            // Phase 4: compare replay vs capture output sums
            metal_flush();
            u32 obid = 0; u64 osz = 0;
            if (cmd->is_blit) { obid = jit.slots[cmd->blit_dst_slot].buf_id; osz = cmd->blit_size; }
            else if (cmd->is_mps) { obid = jit.slots[cmd->mps_dst_slot].buf_id; osz = (u64)cmd->mps_m*cmd->mps_n*4; }
            else if (cmd->n_bufs > 0) { obid = jit.slots[cmd->buf_slots[0]].buf_id; osz = metal_pool.sizes[obid]; }
            if (obid && osz > 0 && metal_pool.bufs[obid]) {
                // (file-scope jit_cap_sums)
                f32 *od = (f32*)metal_pool.bufs[obid].contents;
                u32 nf = (u32)(osz/4);
                double s = 0;
                for (u32 j = 0; j < nf && j < 100000; j++) s += fabs(od[j]);
                double cap = jit_cap_sums[ci];
                double ratio = (cap > 1e-10) ? s / cap : 0;
                if (ci < 5) {
                    f32 *od2 = (f32*)metal_pool.bufs[obid].contents;
                    u32 nf2 = (u32)(osz/4);
                    fprintf(stderr, "  [REP] cmd %u: sum=%.4f [%.8f,%.8f,%.8f]\n",
                            ci, s, od2[0], nf2>1?od2[1]:0, nf2>2?od2[2]:0);
                }
                if (ci < 20 || ratio > 2.0 || ratio < 0.5) {
                    fprintf(stderr, "DIVERGE cmd %u: cap=%.4f rep=%.4f ratio=%.1f n_bufs=%u\n",
                            ci, cap, s, ratio, cmd->n_bufs);
                }
            }
        }
        // Original debug code
        if (jit_debug_replay == 1) {
            metal_flush();
            u32 obid = 0; u64 osz = 0;
            if (cmd->is_blit) { obid = jit.slots[cmd->blit_dst_slot].buf_id; osz = cmd->blit_size; }
            else if (cmd->is_mps) { obid = jit.slots[cmd->mps_dst_slot].buf_id; osz = (u64)cmd->mps_m*cmd->mps_n*4; }
            else if (cmd->n_bufs > 0) { obid = jit.slots[cmd->buf_slots[0]].buf_id; osz = metal_pool.sizes[obid]; }
            if (ci < 40 && (!obid || osz == 0)) {
                fprintf(stderr, "  cmd %2u → (no output) %s\n", ci,
                        cmd->is_blit?"BLIT":(cmd->is_mps?"MPS":"COMP"));
            }
            // Check b273's slot info
            if (ci == 0) {
                // Find which slot maps to b273
                for (u32 s = 0; s < jit.n_slots; s++) {
                    if (jit.slots[s].buf_id == 273) {
                        fprintf(stderr, "  b273 → slot %u (alloc=%llu %s)\n",
                                s, jit.slots[s].alloc_size,
                                jit.slots[s].persistent ? "PERS" : "eph");
                    }
                }
                // Also check how many slots are persistent vs ephemeral
                fprintf(stderr, "  persistent_count=%u n_slots=%u\n", jit.persistent_count, jit.n_slots);
            }
            // Trace cmd 40-42 buffer slots
            if (ci >= 40 && ci <= 42 && !cmd->is_blit && !cmd->is_mps) {
                fprintf(stderr, "  REPLAY cmd %u: slots=[", ci);
                for (u32 i = 0; i < cmd->n_bufs; i++) {
                    fprintf(stderr, "s%u→b%u%s", cmd->buf_slots[i],
                            jit.slots[cmd->buf_slots[i]].buf_id,
                            i<cmd->n_bufs-1?",":"");
                }
                fprintf(stderr, "]\n");
            }
            if (0 && (ci == 3 || ci == 33) && !cmd->is_blit && !cmd->is_mps) {
                // Check for MTLBuffer aliasing
                if (ci == 33) {
                    u32 ob = jit.slots[cmd->buf_slots[0]].buf_id;
                    id<MTLBuffer> obuf = metal_pool.bufs[ob];
                    fprintf(stderr, "  CMD 33 output: b%u MTLBuffer=%p\n", ob, (__bridge void*)obuf);
                    for (u32 i = 0; i < jit.persistent_count; i++) {
                        if (metal_pool.bufs[i] == obuf)
                            fprintf(stderr, "  !!! ALIAS: output b%u == persistent b%u\n", ob, i);
                    }
                    // Check all 5 bufs for aliasing
                    for (u32 i = 0; i < cmd->n_bufs; i++) {
                        u32 bi = jit.slots[cmd->buf_slots[i]].buf_id;
                        for (u32 j = i+1; j < cmd->n_bufs; j++) {
                            u32 bj = jit.slots[cmd->buf_slots[j]].buf_id;
                            if (metal_pool.bufs[bi] == metal_pool.bufs[bj])
                                fprintf(stderr, "  !!! CMD33 SELF-ALIAS: buf[%u]=b%u == buf[%u]=b%u\n", i,bi,j,bj);
                        }
                    }
                }
                fprintf(stderr, "  CMD %u detail: n_bufs=%u grid=%u bufs=[", ci, cmd->n_bufs, cmd->grid[0]);
                for (u32 i = 0; i < cmd->n_bufs; i++) {
                    u32 bid2 = jit.slots[cmd->buf_slots[i]].buf_id;
                    u64 isz = metal_pool.sizes[bid2];
                    f32 *id2 = (f32*)metal_pool.bufs[bid2].contents;
                    int izero = 1;
                    for (u32 j = 0; j < isz/4 && j < 256; j++) if(id2[j]!=0.0f)izero=0;
                    fprintf(stderr, "b%u(%lluB%s%s)%s", bid2, isz,
                            jit.slots[cmd->buf_slots[i]].persistent?" P":" E",
                            izero?" Z":"",
                            i<cmd->n_bufs-1?",":"");
                }
                fprintf(stderr, "]\n");
            }
            if (obid && osz > 0 && metal_pool.bufs[obid]) {
                f32 *od = (f32*)metal_pool.bufs[obid].contents;
                u32 nf = (u32)(osz/4);
                int all_zero = 1, has_nan = 0;
                for (u32 j = 0; j < nf && j < 1024; j++) {
                    if (od[j] != 0.0f) all_zero = 0;
                    if (isnan(od[j])||isinf(od[j])) has_nan = 1;
                }
                if (ci < 45 || (all_zero && nf > 1)) {
                    fprintf(stderr, "  cmd %2u → b%u (%llu B) %s%s\n", ci, obid, osz,
                            cmd->is_blit?"BLIT":(cmd->is_mps?"MPS":"COMP"),
                            (all_zero && nf > 1)?" [ZERO]":"");
                }
                if (all_zero && nf > 1) {
                    fprintf(stderr, "ZERO cmd %u: out_b=%u (%llu B) n_bufs=%u %s grid=%u",
                            ci, obid, osz, cmd->n_bufs,
                            cmd->is_blit?"BLIT":(cmd->is_mps?"MPS":"COMP"),
                            cmd->is_blit?0:(cmd->is_mps?cmd->mps_m:cmd->grid[0]));
                    // Check inputs for zero too
                    if (!cmd->is_blit && !cmd->is_mps) {
                        fprintf(stderr, " inputs:");
                        for (u32 i = 1; i < cmd->n_bufs; i++) {
                            u32 ib = jit.slots[cmd->buf_slots[i]].buf_id;
                            f32 *id2 = (f32*)metal_pool.bufs[ib].contents;
                            u64 isz = metal_pool.sizes[ib];
                            int izero = 1;
                            for (u32 j = 0; j < isz/4 && j < 256; j++) if(id2[j]!=0)izero=0;
                            fprintf(stderr, " b%u%s", ib, izero?"(Z)":"");
                        }
                    }
                    fprintf(stderr, "\n");
                }
                if (has_nan) {
                    fprintf(stderr, "NaN! cmd %u: out_b=%u n_bufs=%u\n", ci, obid, cmd->n_bufs);
                }
            }
        }
    }
    total_dispatches += jit.n_cmds;
    // Ephemeral buffers kept alive — freed by jit_end_capture or program exit.
}
