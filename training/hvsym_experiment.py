import sys, glob, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, r"C:\Libraries\znedi3-pypi\training")
import nnedi3_torch as N
import hvsym_train as H
from hvsym_format import XDIM, YDIM, NNS
from PIL import Image

si, ni, et, q = 3, 1, "abs", 1          # abs n3 nns32 q1 -- collapsed to ~linear (0.0300)
xd, yd = XDIM[si], YDIM[si]; nns = NNS[ni]; ng = nns//4; nm1 = xd-1
X48, Y = H.gather_bank(si)

r2 = np.random.default_rng(7); vw, vt = [], []
for f in [ff for ff in sorted(glob.glob(r"E:\train2017\*.jpg"))][300::1700][:20]:
    F = np.asarray(Image.open(f).convert("L")).astype(np.float32)/255.0; S = F[0::2]; Hf, Wd = S.shape
    if Hf < 2*yd+2 or Wd < 2*xd+2: continue
    g = r2.integers(yd, Hf-yd, 2000); x = r2.integers(xd, Wd-xd, 2000)
    vw.append(N.gather_windows(S, si, g, x)); vt.append(F[2*g-1, x])
h48 = torch.tensor(np.concatenate(vw).astype(np.float32)); hy = torch.tensor(np.concatenate(vt).astype(np.float32))
h47 = H.drop_col(h48, xd, yd)
col = nm1//2
lin = (0.5*(h47[:, (yd//2-1)*nm1+col] + h47[:, (yd//2)*nm1+col]) - hy).abs().mean().item()
ship = N.load_net(H.w, si, ni, et, q, dtype=torch.float32)
for p in ship.parameters(): p.requires_grad_(False)
print(f"linear~{lin:.4f}  shipped={(ship(h48)-hy).abs().mean().item():.4f}", flush=True)

d = H.w.predictors[(et, si, ni)]; qs = f"q{q}"
S = d[f"softmax_{qs}"].astype(np.float32); E = d[f"elliott_{qs}"].astype(np.float32); SB = d[f"softmax_bias_{qs}"].astype(np.float32); EB = d[f"elliott_bias_{qs}"].astype(np.float32)
drop = lambda a: a.reshape(a.shape[0], yd, xd)[:, :, :nm1].reshape(a.shape[0], -1)
with torch.no_grad():
    Xs = X48[:200000]; mu = Xs.mean(1, keepdim=True); v = ((Xs*Xs).mean(1, keepdim=True)-mu*mu).clamp_min(0)
    inv = torch.where(v < N.FLT_EPSILON, torch.zeros_like(v), 1/v.clamp_min(N.FLT_EPSILON).sqrt())
    a = ((Xs-mu)*inv) @ torch.tensor(S).t() + torch.tensor(SB); a = a - a.mean(1, keepdim=True)
    sel = np.argsort(-torch.softmax(a.clamp(-80, 80), 1).sum(0).numpy())[:ng]

def train_one(alpha, epochs, lr, seed):
    net = N.NNEDI3Net(si, ni, symmetric="hv", xdim=nm1)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        net.Ws.copy_(torch.tensor(drop(S[sel]))); net.bs.copy_(torch.tensor(SB[sel]))
        net.We.copy_(torch.tensor(drop(E[sel]))); net.be.copy_(torch.tensor(EB[sel]))
        if seed > 0:
            net.Ws.add_(torch.randn(net.Ws.shape, generator=gen)*0.02); net.We.add_(torch.randn(net.We.shape, generator=gen)*0.02)
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9); M = X48.shape[0]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs*(M//512), eta_min=lr*0.1)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        perm = torch.from_numpy(rng.permutation(M))
        for i in range(0, M-512, 512):
            idx = perm[i:i+512]; xb = X48[idx]
            with torch.no_grad(): tb = ship(xb)
            opt.zero_grad(); pred = net(H.drop_col(xb, xd, yd))
            (alpha*(pred-tb).abs().mean() + (1-alpha)*(pred-Y[idx]).abs().mean()).backward(); opt.step(); sched.step()
    with torch.no_grad(): return (net(h47)-hy).abs().mean().item()

for alpha, ep, lr, seeds, tag in [(0.7,12,3e-3,1,"baseline a0.7 e12"), (0.9,16,3e-3,1,"a0.9 e16"),
                                  (1.0,16,3e-3,1,"a1.0 e16 (pure distill)"), (0.9,16,3e-3,3,"a0.9 e16 x3seed-best"),
                                  (0.85,16,1e-3,1,"a0.85 e16 lr1e-3")]:
    t0 = time.time(); best = min(train_one(alpha, ep, lr, s) for s in range(seeds))
    print(f"{tag:26} held-out={best:.4f}  ({time.time()-t0:.0f}s)", flush=True)
