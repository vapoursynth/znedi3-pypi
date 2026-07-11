"""Disk-backed streaming trainer for the symmetric flagship at tritical scale (~60M patterns),
for machines that can't hold the corpus in RAM. Extract windows as uint8 to a memmap on a big
drive, then train the symmetric 48x6/256 net over shuffled super-batches with cosine-annealed LR,
checkpointing a znedi3-loadable .bin every epoch.

Env overrides (for a quick smoke run): TS_TARGET, TS_EPOCHS, TS_MAXIMG, TS_VAL.
"""
import sys, os, glob, time, numpy as np, torch
sys.path.insert(0, r"C:\Libraries\znedi3-pypi\training")
import nnedi3_torch as N
from PIL import Image

IMAGES  = r"E:\train2017\trainingset2\*.jpg"
WORK    = os.environ.get("TS_WORK", r"E:\train2017\_ts2_work")
BASE    = r"C:\Libraries\znedi3-pypi\nnedi3_weights.bin"
OUT_BIN = r"E:\train2017\trainingset2_sym_flagship.bin"
OUT_PT  = r"E:\train2017\trainingset2_sym_flagship.pt"
NSIZE, NNS_IDX, ETYPE, QUAL = 3, 4, "abs", 1
XDIM, YDIM = N.XDIM[NSIZE], N.YDIM[NSIZE]; D = XDIM * YDIM

TARGET   = int(os.environ.get("TS_TARGET", 64_000_000))
EPOCHS   = int(os.environ.get("TS_EPOCHS", 20))
MAXIMG   = int(os.environ.get("TS_MAXIMG", 0)) or None
VAL_N    = int(os.environ.get("TS_VAL", 200))
PER_IMG  = 1100
BATCH    = 4096
SUPER    = 2_000_000
LR_MAX   = 1e-2
os.makedirs(WORK, exist_ok=True)
rng = np.random.default_rng(0)


def load_u8(p):
    return np.asarray(Image.open(p).convert("L"), dtype=np.uint8)


def extract(frame, per_parity, rng):
    """Random windows (uint8 [k,D]) + targets (uint8 [k]) in znedi3 flagship geometry, both parities."""
    H, W = frame.shape; ws, ts = [], []
    for parity in (0, 1):
        S = frame[parity::2]; Hf, Wd = S.shape
        if Hf < YDIM + 2 or Wd < XDIM + 4:
            continue
        g = rng.integers(YDIM // 2, Hf - (YDIM - YDIM // 2), size=per_parity)
        x = rng.integers(XDIM // 2 - 1, Wd - (XDIM - (XDIM // 2 - 1)), size=per_parity)
        tr = 2 * g + parity - 1
        m = (tr >= 0) & (tr < H); g, x, tr = g[m], x[m], tr[m]
        ws.append(N.gather_windows(S, NSIZE, g, x)); ts.append(frame[tr, x])
    if not ws:
        return None, None
    return np.concatenate(ws).astype(np.uint8), np.concatenate(ts).astype(np.uint8)


files = sorted(glob.glob(IMAGES)); rng.shuffle(files)
val_files = files[:VAL_N]; train_files = files[VAL_N:]
if MAXIMG:
    train_files = train_files[:MAXIMG]
print(f"{len(train_files)} train images, {len(val_files)} val images", flush=True)

# ---- Phase 1: extract to memmap (skip if a matching one already exists) ----
win_path, tgt_path, meta_path = (os.path.join(WORK, f) for f in ("win.u8", "tgt.u8", "meta.txt"))
cap = TARGET + 4_000_000
if os.path.exists(meta_path):
    Ntot = int(open(meta_path).read().strip())
    print(f"reusing existing extraction: {Ntot:,} patterns", flush=True)
else:
    mw = np.memmap(win_path, dtype=np.uint8, mode="w+", shape=(cap, D))
    mt = np.memmap(tgt_path, dtype=np.uint8, mode="w+", shape=(cap,))
    n = 0; t0 = time.time()
    for i, f in enumerate(train_files):
        if n >= TARGET:
            break
        try:
            w, t = extract(load_u8(f), PER_IMG // 2, rng)
        except Exception:
            continue
        if w is None:
            continue
        k = min(len(w), cap - n)
        mw[n:n + k] = w[:k]; mt[n:n + k] = t[:k]; n += k
        if i % 3000 == 0:
            print(f"  extract: {n:,} patterns / {i} images  ({time.time()-t0:.0f}s)", flush=True)
    mw.flush(); mt.flush(); Ntot = n
    open(meta_path, "w").write(str(Ntot))
    print(f"extracted {Ntot:,} patterns ({Ntot*D/1e9:.1f} GB)  ({time.time()-t0:.0f}s)", flush=True)

mw = np.memmap(win_path, dtype=np.uint8, mode="r", shape=(cap, D))
mt = np.memmap(tgt_path, dtype=np.uint8, mode="r", shape=(cap,))

vw, vt = [], []
for f in val_files:
    try:
        w, t = extract(load_u8(f), 400, rng)
    except Exception:
        continue
    if w is not None:
        vw.append(w); vt.append(t)
Xv = torch.from_numpy(np.concatenate(vw).astype(np.float32) / 255.0)
Yv = torch.from_numpy(np.concatenate(vt).astype(np.float32) / 255.0)
print(f"val patterns: {len(Xv):,}", flush=True)

# ---- Phase 2: train the symmetric flagship ----
net = N.NNEDI3Net(NSIZE, NNS_IDX, symmetric=True)
print(f"symmetric flagship free params: {sum(p.numel() for p in net.parameters()):,} | torch threads {torch.get_num_threads()}", flush=True)
loss_fn = torch.nn.L1Loss() if ETYPE == "abs" else torch.nn.MSELoss()
opt = torch.optim.SGD(net.parameters(), lr=LR_MAX, momentum=0.9)
spe = Ntot // BATCH
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS * spe, eta_min=LR_MAX * 0.005)


def evaluate():
    net.eval()
    with torch.no_grad():
        return torch.cat([net(Xv[i:i+65536]) for i in range(0, len(Xv), 65536)]).sub_(Yv).abs().mean().item()


# baseline: 2-tap linear on the val set
_c = XDIM // 2 - 1   # the target's own column within the window (cols run x-23..x+24, so x is index 23)
lin = float(torch.abs(0.5 * (Xv[:, (YDIM//2-1)*XDIM + _c] + Xv[:, (YDIM//2)*XDIM + _c]) - Yv).mean())
print(f"(val 2-tap-linear baseline ~ {lin:.5f})  starting train...", flush=True)
best, seen = 1e9, 0
for ep in range(EPOCHS):
    net.train(); t0 = time.time(); sie = 0; rl = 0.0
    starts = list(range(0, Ntot, SUPER)); rng.shuffle(starts)
    for s in starts:
        e = min(s + SUPER, Ntot)
        Xu = np.asarray(mw[s:e]); Yu = np.asarray(mt[s:e])
        perm = rng.permutation(len(Xu))
        for i in range(0, len(Xu) - BATCH + 1, BATCH):
            idx = perm[i:i+BATCH]
            xb = torch.from_numpy(Xu[idx].astype(np.float32) / 255.0)
            yb = torch.from_numpy(Yu[idx].astype(np.float32) / 255.0)
            opt.zero_grad(); l = loss_fn(net(xb), yb); l.backward(); opt.step(); sched.step()
            seen += BATCH; sie += 1; rl += l.item()
            if sie % 500 == 0:   # within-epoch progress
                print(f"    ep{ep} step {sie}/{spe}  train {ETYPE}={rl/500:.5f}  lr={opt.param_groups[0]['lr']:.2e}"
                      f"  ({sie*BATCH/max(time.time()-t0,1e-9)/1e6:.1f}M/s)", flush=True)
                rl = 0.0
    vl = evaluate(); best = min(best, vl)
    torch.save(net.state_dict(), OUT_PT)
    N.write_weights(OUT_BIN, BASE, {(ETYPE, NSIZE, NNS_IDX, QUAL): net})
    print(f"epoch {ep:2d}  val L1={vl:.5f}  best={best:.5f}  lr={opt.param_groups[0]['lr']:.2e}  seen={seen/1e6:.0f}M  ({time.time()-t0:.0f}s)", flush=True)
print(f"DONE. best val L1={best:.5f}  vs linear {lin:.5f}.  wrote {OUT_BIN}", flush=True)
