#!/usr/bin/env python3
"""2-layer CNN + CE loss grad check"""
import numpy as np
BS,C1,C2=4,4,8; flat_f=C2*24*24

def conv_fwd(x,w,b):
    B,Cin,H,W=x.shape;Cout,_,KH,KW=w.shape;OH,OW=H-KH+1,W-KW+1
    col=np.zeros((B,Cin,KH,KW,OH,OW),dtype=np.float32)
    for kh in range(KH):
        for kw in range(KW): col[:,:,kh,kw,:,:]=x[:,:,kh:kh+OH,kw:kw+OW]
    w_f=w.reshape(Cout,Cin*KH*KW);col_f=col.reshape(B,Cin*KH*KW,OH*OW)
    out=np.einsum('ck,bkp->bcp',w_f,col_f).reshape(B,Cout,OH,OW)+b.reshape(1,Cout,1,1)
    return out,col
def conv_bwd(dy,x,w,col):
    B,Cin,H,W=x.shape;Cout,_,KH,KW=w.shape;OH,OW=H-KH+1,W-KW+1
    db=dy.sum(axis=(0,2,3));dy_f=dy.reshape(B,Cout,OH*OW);col_f=col.reshape(B,Cin*KH*KW,OH*OW)
    w_f=w.reshape(Cout,Cin*KH*KW)
    dw=np.einsum('bcp,bkp->ck',dy_f,col_f).reshape(w.shape)
    dcol_f=np.einsum('ck,bcp->bkp',w_f,dy_f).reshape(B,Cin,KH,KW,OH,OW)
    dx=np.zeros_like(x)
    for kh in range(KH):
        for kw in range(KW): dx[:,:,kh:kh+OH,kw:kw+OW]+=dcol_f[:,:,kh,kw,:,:]
    return dw,db,dx

cw1=np.fromfile('/tmp/c2ce_cw1.bin',dtype=np.float32).reshape(C1,1,3,3)
cb1=np.zeros(C1,dtype=np.float32)
cw2=np.fromfile('/tmp/c2ce_cw2.bin',dtype=np.float32).reshape(C2,C1,3,3)
cb2=np.zeros(C2,dtype=np.float32)
lw=np.fromfile('/tmp/c2ce_lw.bin',dtype=np.float32).reshape(flat_f,10)
lb=np.zeros(10,dtype=np.float32)
X=np.fromfile('/tmp/c2ce_X.bin',dtype=np.float32).reshape(BS,1,28,28)
Y=np.fromfile('/tmp/c2ce_Y.bin',dtype=np.float32).reshape(BS,10)

h1p,c1=conv_fwd(X,cw1,cb1);h1=np.maximum(h1p,0)
h2p,c2=conv_fwd(h1,cw2,cb2);h2=np.maximum(h2p,0)
flat=h2.reshape(BS,flat_f);logits=flat@lw+lb
mx=logits.max(1,keepdims=True);sh=logits-mx;e=np.exp(sh);es=e.sum(1,keepdims=True)
pr=e/es;cl=np.maximum(pr,1e-7);lp=np.log(cl);loss=-(Y*lp).sum()/BS
print(f"numpy loss={loss:.6f}")

dl=(pr-Y)/BS;d_lw=flat.T@dl;d_lb=dl.sum(0);df=dl@lw.T
dh2=df.reshape(BS,C2,24,24);dh2p=dh2*(h2p>0).astype(np.float32)
d_cw2,d_cb2,dh1=conv_bwd(dh2p,h1,cw2,c2)
dh1p=dh1*(h1p>0).astype(np.float32)
d_cw1,d_cb1,_=conv_bwd(dh1p,X,cw1,c1)

names=['g_cw1','g_cb1','g_cw2','g_cb2','g_lw','g_lb']
sizes=[C1*9,C1,C2*C1*9,C2,flat_f*10,10]
refs=[d_cw1,d_cb1,d_cw2,d_cb2,d_lw,d_lb]
shapes=[cw1.shape,cb1.shape,cw2.shape,cb2.shape,lw.shape,lb.shape]
ok=True
for n,sz,r,sh in zip(names,sizes,refs,shapes):
    g=np.fromfile(f'/tmp/c2ce_{n}.bin',dtype=np.float32).reshape(sh)
    d=np.abs(g-r);mx_d=d.max();mn_d=d.mean()
    s="OK" if mx_d<1e-3 else "FAIL"
    if mx_d>1e-3: ok=False
    print(f"{n}: max={mx_d:.2e} mean={mn_d:.2e} {s}")
print(f"\n{'PASS' if ok else 'FAIL'}")
