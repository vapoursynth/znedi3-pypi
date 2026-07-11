"""Train nnedi3_weights_hvsym.bin -- (N-1)-wide h+v-symmetric generators for every config, by
distillation from the shipped weights (same recipe as the vmirror v-only run: 0.7*distill +
0.3*truth, top-usage init, batch 512, 12 epochs to the plateau).

The C++ folded predictor gathers an (N-1)-wide window, normalises over N-1, and materialises the
Klein-4 orbit about the centre column. For an ODD width, NNEDI3Net(symmetric="hv", xdim=N-1) is the
exact training-time equivalent -- so we train that, feed it the 48-wide corpus window minus its last
column, distil against the shipped 48-wide predictor, and write out net.Ws/We/bs/be as generators.
"""
import sys, os, glob, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, r"C:\Libraries\znedi3-pypi\training")
import nnedi3_torch as N
from hvsym_format import write_hvsym, XDIM, YDIM, NNS
from PIL import Image

SHIPPED = r"C:\Libraries\znedi3-pypi\nnedi3_weights.bin"
OUT     = r"C:\Libraries\znedi3-pypi\nnedi3_weights_hvsym.bin"
NIMG, MPERIMG, EPOCHS, BATCH, LR, ALPHA = 100, 20000, 12, 512, 3e-3, 0.7
NS_ONLY = os.environ.get("HV_NS_ONLY")            # e.g. "3" to train only nsize 3 (for the self-test)
w = N.NNEDI3Weights.read(SHIPPED)
imgs = sorted(glob.glob(r"E:\train2017\*.jpg"))[:NIMG]


def selftest():
    """NNEDI3Net(symmetric='hv', xdim=odd).forward == znedi3 reference math on its materialised experts."""
    net = N.NNEDI3Net(3, 4, symmetric="hv", xdim=47)
    with torch.no_grad():
        for p in net.parameters(): p.normal_(0, 0.1)
    ex = net.export()                              # materialise the 256 experts (47-wide)
    d = { "softmax_q1": ex["softmax"], "elliott_q1": ex["elliott"], "softmax_bias_q1": ex["softmax_bias"], "elliott_bias_q1": ex["elliott_bias"] }
    rng = np.random.default_rng(0); win = rng.random((2000, 47*6)).astype(np.float32)
    ref = N.reference_predict(d, win, 1)
    with torch.no_grad():
        got = net(torch.tensor(win)).numpy()
    err = float(np.abs(ref - got).max())
    print(f"[selftest] 47-wide hv forward vs reference math: max|diff| = {err:.2e}  {'OK' if err < 1e-4 else 'FAIL'}", flush=True)
    assert err < 1e-4


def is_copy(si, ni):
    a = w.predictors[("abs", si, ni)]; m = w.predictors[("mse", si, ni)]
    return all(np.array_equal(a[k], m[k]) for k in a)


def gather_bank(si):
    xd, yd = XDIM[si], YDIM[si]; rng = np.random.default_rng(1000+si); W48, T = [], []
    for f in imgs:
        F = np.asarray(Image.open(f).convert("L")).astype(np.float32)/255.0
        S = F[0::2]; Hf, Wd = S.shape
        if Hf < 2*yd+2 or Wd < 2*xd+2: continue
        g = rng.integers(yd, Hf-yd, MPERIMG); x = rng.integers(xd, Wd-xd, MPERIMG)
        W48.append(N.gather_windows(S, si, g, x)); T.append(F[2*g-1, x])
    X = torch.tensor(np.concatenate(W48).astype(np.float32)); Y = torch.tensor(np.concatenate(T).astype(np.float32))
    return X, Y


def drop_col(x48, xd, yd):
    return x48.view(-1, yd, xd)[:, :, :xd-1].reshape(x48.shape[0], -1)      # 48-wide window -> 47-wide (drop last col)


def train_config(si, ni, et, q, X48, Y, held):
    nns = NNS[ni]; ng = nns // 4; xd = XDIM[si]; yd = YDIM[si]; nm1 = xd - 1
    ship = N.load_net(w, si, ni, et, q, dtype=torch.float32)
    for p in ship.parameters(): p.requires_grad_(False)
    net = N.NNEDI3Net(si, ni, symmetric="hv", xdim=nm1)
    d = w.predictors[(et, si, ni)]; qs = f"q{q}"
    S = d[f"softmax_{qs}"].astype(np.float32); E = d[f"elliott_{qs}"].astype(np.float32)
    SB = d[f"softmax_bias_{qs}"].astype(np.float32); EB = d[f"elliott_bias_{qs}"].astype(np.float32)
    drop = lambda a: a.reshape(a.shape[0], yd, xd)[:, :, :nm1].reshape(a.shape[0], -1)
    with torch.no_grad():
        Xs = X48[:200000]
        mu = Xs.mean(1, keepdim=True); v = ((Xs*Xs).mean(1, keepdim=True)-mu*mu).clamp_min(0)
        inv = torch.where(v < N.FLT_EPSILON, torch.zeros_like(v), 1/v.clamp_min(N.FLT_EPSILON).sqrt())
        a = ((Xs-mu)*inv) @ torch.tensor(S).t() + torch.tensor(SB); a = a - a.mean(1, keepdim=True)
        sel = np.argsort(-torch.softmax(a.clamp(-80, 80), 1).sum(0).numpy())[:ng]
        net.Ws.copy_(torch.tensor(drop(S[sel]))); net.bs.copy_(torch.tensor(SB[sel]))
        net.We.copy_(torch.tensor(drop(E[sel]))); net.be.copy_(torch.tensor(EB[sel]))

    opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9)
    M = X48.shape[0]; sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS*(M//BATCH), eta_min=LR*0.1)
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        net.train(); perm = torch.from_numpy(rng.permutation(M))
        for i in range(0, M-BATCH, BATCH):
            idx = perm[i:i+BATCH]; xb48 = X48[idx]
            with torch.no_grad(): tb = ship(xb48)
            opt.zero_grad(); pred = net(drop_col(xb48, xd, yd))
            (ALPHA*(pred-tb).abs().mean() + (1-ALPHA)*(pred-Y[idx]).abs().mean()).backward()
            opt.step(); sched.step()
    with torch.no_grad():
        h48, hy = held
        syl = (net(drop_col(h48, xd, yd))-hy).abs().mean().item()
    return (net.Ws.detach().numpy().copy(), net.We.detach().numpy().copy(),
            net.bs.detach().numpy().copy(), net.be.detach().numpy().copy(), syl)


MARGIN = 0.0008                          # a config is KEPT only if both quals beat linear by this much
CKDIR = r"E:\train2017\hvsym_gen"        # per-config checkpoints (resumable; survives a stop)


def linear_hl(h48, hy, xd, yd):
    nm1 = xd - 1; col = nm1 // 2; h47 = drop_col(h48, xd, yd)
    return (0.5*(h47[:, (yd//2-1)*nm1+col] + h47[:, (yd//2)*nm1+col]) - hy).abs().mean().item()


def main():
    selftest()
    os.makedirs(CKDIR, exist_ok=True)
    used = set(Path(l).name for l in open(r"E:\train2017\trainingset2_list.txt").read().splitlines() if l.strip()) if os.path.exists(r"E:\train2017\trainingset2_list.txt") else set()
    heldpaths = [f for f in sorted(glob.glob(r"E:\train2017\*.jpg")) if Path(f).name not in used][300::1700][:20]
    GEN = {}; kept = dropped = 0; t_all = time.time()
    ns_range = [int(NS_ONLY)] if NS_ONLY else range(7)
    for si in ns_range:
        xd, yd = XDIM[si], YDIM[si]
        r2 = np.random.default_rng(7); vw, vt = [], []
        for f in heldpaths:
            F = np.asarray(Image.open(f).convert("L")).astype(np.float32)/255.0; S = F[0::2]; Hf, Wd = S.shape
            if Hf < 2*yd+2 or Wd < 2*xd+2: continue
            g = r2.integers(yd, Hf-yd, 2000); x = r2.integers(xd, Wd-xd, 2000)
            vw.append(N.gather_windows(S, si, g, x)); vt.append(F[2*g-1, x])
        h48 = torch.tensor(np.concatenate(vw).astype(np.float32)); hy = torch.tensor(np.concatenate(vt).astype(np.float32))
        held = (h48, hy); lin = linear_hl(h48, hy, xd, yd)
        need = any(not os.path.exists(os.path.join(CKDIR, f"{et}_{ni}_{si}.npz"))
                   for ni in range(5) for et in ("abs", "mse") if not (et == "mse" and is_copy(si, ni)))
        X48 = Y = None
        if need:
            X48, Y = gather_bank(si)
        print(f"== nsize {si} ({xd}x{yd}) linear={lin:.4f} ==", flush=True)
        for ni in range(5):
            for et_i, et in enumerate(("abs", "mse")):
                if et == "mse" and is_copy(si, ni):
                    for q in (1, 2): GEN[(1, ni, si, q)] = GEN[(0, ni, si, q)]
                    continue
                ck = os.path.join(CKDIR, f"{et}_{ni}_{si}.npz")
                if os.path.exists(ck):
                    z = np.load(ck)
                    g1 = (z["sg1"], z["eg1"], z["sb1"], z["eb1"]); g2 = (z["sg2"], z["eg2"], z["sb2"], z["eb2"])
                    h1 = float(z["h1"]); h2 = float(z["h2"])
                else:
                    *a1, h1 = train_config(si, ni, et, 1, X48, Y, held)
                    *a2, h2 = train_config(si, ni, et, 2, X48, Y, held)
                    g1, g2 = tuple(a1), tuple(a2)
                    np.savez(ck, sg1=g1[0], eg1=g1[1], sb1=g1[2], eb1=g1[3], sg2=g2[0], eg2=g2[1], sb2=g2[2], eb2=g2[3], h1=h1, h2=h2)
                good = (h1 < lin - MARGIN) and (h2 < lin - MARGIN)
                if good:
                    GEN[(et_i, ni, si, 1)] = g1; GEN[(et_i, ni, si, 2)] = g2; kept += 1
                else:
                    zg = tuple(np.zeros_like(a) for a in g1); GEN[(et_i, ni, si, 1)] = zg; GEN[(et_i, ni, si, 2)] = zg; dropped += 1
                print(f"  {et} n{si} nns{NNS[ni]}: q1={h1:.4f} q2={h2:.4f} lin={lin:.4f} -> {'KEEP' if good else 'drop(fallback)'}", flush=True)
        if X48 is not None: del X48, Y

    if not NS_ONLY:
        write_hvsym(OUT, lambda et, ni, si, q, ng, Dm1: GEN[(et, ni, si, q)])
        print(f"DONE wrote {OUT}  kept={kept} dropped={dropped}  ({(time.time()-t_all)/60:.1f} min)", flush=True)
    else:
        print(f"NS_ONLY={NS_ONLY}  kept={kept} dropped={dropped}  ({(time.time()-t_all)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
