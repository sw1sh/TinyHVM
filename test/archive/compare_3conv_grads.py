#!/usr/bin/env python3
"""3-conv CNN gradient comparison: TinyHVM vs numpy reference.

TinyHVM's thvm_maxpool2d uses two sequential RMAX operations:
  1. RMAX over KW (last dim of [B,C,OH,OW,KH,KW]) -> row max per kernel row
  2. RMAX over KH (last dim of [B,C,OH,OW,KH])     -> overall max

The backward of each RMAX scatters gradient to ALL positions that equal the
maximum (tie-broadcast), not just the first position (tie-argmax).  This numpy
reference replicates that exact semantics.
"""
import sys
import numpy as np

BS = 64
C1, C2, C3 = 8, 16, 32
flat_f = C3 * 5 * 5  # 800

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def conv_fwd(x, w, b, pad=0):
    """Forward conv2d, returns (output, col_cache, padded_x) for backward."""
    B, Cin, H, W = x.shape
    Cout, _, KH, KW = w.shape
    if pad > 0:
        x = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    H2, W2 = x.shape[2], x.shape[3]
    OH, OW = H2 - KH + 1, W2 - KW + 1
    col = np.zeros((B, Cin, KH, KW, OH, OW), dtype=np.float32)
    for kh in range(KH):
        for kw in range(KW):
            col[:, :, kh, kw, :, :] = x[:, :, kh:kh+OH, kw:kw+OW]
    w_f = w.reshape(Cout, Cin * KH * KW)
    col_f = col.reshape(B, Cin * KH * KW, OH * OW)
    out = np.einsum('ck,bkp->bcp', w_f, col_f).reshape(B, Cout, OH, OW)
    out += b.reshape(1, Cout, 1, 1)
    return out, col, x  # x here is the padded version

def conv_bwd(dy, x_padded, w, col, pad=0):
    """Backward conv2d.  x_padded is the (already-padded) input used in fwd."""
    B, _Cin_pad, Hp, Wp = x_padded.shape
    Cout, Cin, KH, KW = w.shape
    OH, OW = Hp - KH + 1, Wp - KW + 1
    db = dy.sum(axis=(0, 2, 3))
    dy_f = dy.reshape(B, Cout, OH * OW)
    col_f = col.reshape(B, Cin * KH * KW, OH * OW)
    w_f = w.reshape(Cout, Cin * KH * KW)
    dw = np.einsum('bcp,bkp->ck', dy_f, col_f).reshape(w.shape)
    dcol_f = np.einsum('ck,bcp->bkp', w_f, dy_f).reshape(B, Cin, KH, KW, OH, OW)
    dx_padded = np.zeros_like(x_padded)
    for kh in range(KH):
        for kw in range(KW):
            dx_padded[:, :, kh:kh+OH, kw:kw+OW] += dcol_f[:, :, kh, kw, :, :]
    if pad > 0:
        dx = dx_padded[:, :, pad:-pad, pad:-pad]
    else:
        dx = dx_padded
    return dw, db, dx


def maxpool_fwd(x, kh=2, kw=2, sh=2, sw=2):
    """Forward max-pool matching TinyHVM's two-step RMAX approach.

    TinyHVM expands input to [B,C,OH,OW,KH,KW] then applies:
      r1 = RMAX over KW  ->  [B,C,OH,OW,KH]
      r2 = RMAX over KH  ->  [B,C,OH,OW]

    Returns (out, cache) where cache stores intermediate values for backward.
    """
    B, C, H, W = x.shape
    OH = (H - kh) // sh + 1
    OW = (W - kw) // sw + 1

    # Build window tensor [B, C, OH, OW, KH, KW]
    windows = np.zeros((B, C, OH, OW, kh, kw), dtype=np.float32)
    for ki in range(kh):
        for kj in range(kw):
            windows[:, :, :, :, ki, kj] = x[:, :,
                np.arange(OH)[:, None] * sh + ki,
                np.arange(OW)[None, :] * sw + kj]

    # Step 1: RMAX over KW (last dim)  ->  [B,C,OH,OW,KH,1]
    row_max = windows.max(axis=-1, keepdims=True)

    # Step 2: RMAX over KH (now last dim after squeeze would be KH)
    rows_maxval = row_max.squeeze(-1)               # [B,C,OH,OW,KH]
    col_max = rows_maxval.max(axis=-1, keepdims=True)  # [B,C,OH,OW,1]
    out = col_max.squeeze(-1)                        # [B,C,OH,OW]

    # Store for backward
    cache = (windows, row_max, rows_maxval, col_max, kh, kw, sh, sw, H, W)
    return out, cache


def maxpool_bwd(dy, cache):
    """Backward through TinyHVM's two-step RMAX max-pool.

    TinyHVM's UOP_RMAX backward:
        mask = (a >= max_bc)   i.e. (a == max_bc)  — all positions equal to max
        da = gy * mask

    There is NO division by count — all equal-max positions each receive the
    full upstream gradient.  This can produce gradient > 1x the upstream when
    there are ties (e.g. all-zero ReLU output windows).
    """
    windows, row_max, rows_maxval, col_max, kh, kw, sh, sw, H, W = cache
    B, C, OH, OW = dy.shape

    # Step 2 backward: RMAX over KH (last dim of rows_maxval [B,C,OH,OW,KH])
    # mask_kh[...,ki] = 1 if rows_maxval[...,ki] == col_max (the max over KH)
    mask_kh = (rows_maxval == col_max).astype(np.float32)   # (B,C,OH,OW,KH)
    d_rows = mask_kh * dy[:, :, :, :, np.newaxis]           # (B,C,OH,OW,KH)

    # Step 1 backward: RMAX over KW (last dim of windows [B,C,OH,OW,KH,KW])
    # mask_kw[...,ki,kj] = 1 if windows[...,ki,kj] == row_max[...,ki,0]
    mask_kw = (windows == row_max).astype(np.float32)        # (B,C,OH,OW,KH,KW)
    d_windows = mask_kw * d_rows[:, :, :, :, :, np.newaxis]  # (B,C,OH,OW,KH,KW)

    # Scatter back to input shape (B,C,H,W)
    dx = np.zeros((B, C, H, W), dtype=np.float32)
    for ki in range(kh):
        for kj in range(kw):
            ri = np.arange(OH)[:, None] * sh + ki  # (OH,OW)
            ci = np.arange(OW)[None, :] * sw + kj
            np.add.at(dx, (slice(None), slice(None), ri, ci),
                      d_windows[:, :, :, :, ki, kj])
    return dx


# ---------------------------------------------------------------------------
# Load weights and data
# ---------------------------------------------------------------------------

cw1 = np.fromfile('/tmp/gc3_cw1.bin', dtype=np.float32).reshape(C1, 1, 3, 3)
cb1 = np.fromfile('/tmp/gc3_cb1.bin', dtype=np.float32)
cw2 = np.fromfile('/tmp/gc3_cw2.bin', dtype=np.float32).reshape(C2, C1, 3, 3)
cb2 = np.fromfile('/tmp/gc3_cb2.bin', dtype=np.float32)
cw3 = np.fromfile('/tmp/gc3_cw3.bin', dtype=np.float32).reshape(C3, C2, 3, 3)
cb3 = np.fromfile('/tmp/gc3_cb3.bin', dtype=np.float32)
lw  = np.fromfile('/tmp/gc3_lw.bin',  dtype=np.float32).reshape(flat_f, 10)
lb  = np.fromfile('/tmp/gc3_lb.bin',  dtype=np.float32)

X = np.fromfile('/tmp/gc3_X.bin', dtype=np.float32).reshape(BS, 1, 28, 28)
Y = np.fromfile('/tmp/gc3_Y.bin', dtype=np.float32).reshape(BS, 10)

# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

# Conv1(1,8,3,p=0) -> ReLU -> MaxPool(2,2)  [BS,8,13,13]
h1p, c1, x1_pad = conv_fwd(X, cw1, cb1, pad=0)
h1 = np.maximum(h1p, 0.0)
h1p2, minfo1 = maxpool_fwd(h1)

# Conv2(8,16,3,p=0) -> ReLU -> MaxPool(2,2) [BS,16,5,5]
h2p, c2, x2_pad = conv_fwd(h1p2, cw2, cb2, pad=0)
h2 = np.maximum(h2p, 0.0)
h2p2, minfo2 = maxpool_fwd(h2)

# Conv3(16,32,3,p=1) -> ReLU  [BS,32,5,5]
h3p, c3, x3_pad = conv_fwd(h2p2, cw3, cb3, pad=1)
h3 = np.maximum(h3p, 0.0)

# Flatten -> Linear
flat = h3.reshape(BS, flat_f)
logits = flat @ lw + lb

# Cross-entropy loss (numerically stable)
mx = logits.max(1, keepdims=True)
sh_logits = logits - mx
e = np.exp(sh_logits)
es = e.sum(1, keepdims=True)
pr = e / es
cl = np.maximum(pr, 1e-7)
lp = np.log(cl)
loss = -(Y * lp).sum() / BS
print(f"numpy loss={loss:.6f}")

# ---------------------------------------------------------------------------
# Backward pass
# ---------------------------------------------------------------------------

dl = (pr - Y) / BS                  # [BS,10]

d_lw = flat.T @ dl                  # [flat_f, 10]
d_lb = dl.sum(0)                    # [10]
d_flat = dl @ lw.T                  # [BS, flat_f]

dh3 = d_flat.reshape(BS, C3, 5, 5)
dh3p = dh3 * (h3p > 0).astype(np.float32)

d_cw3, d_cb3, dh2p2 = conv_bwd(dh3p, x3_pad, cw3, c3, pad=1)

dh2 = maxpool_bwd(dh2p2, minfo2)
dh2p_pre = dh2 * (h2p > 0).astype(np.float32)

d_cw2, d_cb2, dh1p2 = conv_bwd(dh2p_pre, x2_pad, cw2, c2, pad=0)

dh1 = maxpool_bwd(dh1p2, minfo1)
dh1p_pre = dh1 * (h1p > 0).astype(np.float32)

d_cw1, d_cb1, _ = conv_bwd(dh1p_pre, x1_pad, cw1, c1, pad=0)

# ---------------------------------------------------------------------------
# Compare against TinyHVM dumped gradients
# ---------------------------------------------------------------------------

names  = ['g_cw1', 'g_cb1', 'g_cw2', 'g_cb2', 'g_cw3', 'g_cb3', 'g_lw', 'g_lb']
refs   = [d_cw1, d_cb1, d_cw2, d_cb2, d_cw3, d_cb3, d_lw, d_lb]
shapes = [cw1.shape, cb1.shape, cw2.shape, cb2.shape, cw3.shape, cb3.shape, lw.shape, lb.shape]
sizes  = [C1*9, C1, C2*C1*9, C2, C3*C2*9, C3, flat_f*10, 10]

ok = True
THRESHOLD = 1e-5
for n, sz, r, sh in zip(names, sizes, refs, shapes):
    g = np.fromfile(f'/tmp/gc3_{n}.bin', dtype=np.float32).reshape(sh)
    diff = np.abs(g - r)
    mx_d = float(diff.max())
    mn_d = float(diff.mean())
    status = "OK" if mx_d < THRESHOLD else "FAIL"
    if mx_d >= THRESHOLD:
        ok = False
    print(f"{n}: max_abs_diff={mx_d:.3e}  mean_abs_diff={mn_d:.3e}  {status}")

print()
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
