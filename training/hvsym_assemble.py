"""Assemble nnedi3_weights_hvsym.bin from the per-config checkpoints, keeping a config only where
the h+v fold preserved the shipped quality (gap = hvsym - shipped < GAP for both quals). Dropped
configs are written as zero generators -> the C++ loader skips them -> the filter falls back to the
dense predictor (never worse than stock). Re-runnable with a different GAP without retraining."""
import sys, os, glob, numpy as np, torch
from pathlib import Path
sys.path.insert(0, r"C:\Libraries\znedi3-pypi\training")
import nnedi3_torch as N
import hvsym_train as H
from hvsym_format import write_hvsym, XDIM, YDIM, NNS

CKDIR, OUT, w = H.CKDIR, H.OUT, H.w
GAP = float(os.environ.get("HV_GAP", 0.0010))

used = set(Path(l).name for l in open(r"E:\train2017\trainingset2_list.txt").read().splitlines() if l.strip()) if os.path.exists(r"E:\train2017\trainingset2_list.txt") else set()
heldpaths = [f for f in sorted(glob.glob(r"E:\train2017\*.jpg")) if Path(f).name not in used][300::1700][:20]
from PIL import Image


def held_for(si):
    xd, yd = XDIM[si], YDIM[si]; r2 = np.random.default_rng(7); vw, vt = [], []
    for f in heldpaths:
        F = np.asarray(Image.open(f).convert("L")).astype(np.float32)/255.0; S = F[0::2]; Hf, Wd = S.shape
        if Hf < 2*yd+2 or Wd < 2*xd+2: continue
        g = r2.integers(yd, Hf-yd, 2000); x = r2.integers(xd, Wd-xd, 2000)
        vw.append(N.gather_windows(S, si, g, x)); vt.append(F[2*g-1, x])
    return torch.tensor(np.concatenate(vw).astype(np.float32)), torch.tensor(np.concatenate(vt).astype(np.float32))


GEN = {}; kept = dropped = 0; rows = []
for si in range(7):
    h48, hy = held_for(si)
    for ni in range(5):
        for et_i, et in enumerate(("abs", "mse")):
            if et == "mse" and H.is_copy(si, ni):
                for q in (1, 2): GEN[(1, ni, si, q)] = GEN[(0, ni, si, q)]
                if GEN[(1, ni, si, 1)][0].any(): rows.append((et, si, NNS[ni], "copy-keep"))
                continue
            ck = os.path.join(CKDIR, f"{et}_{ni}_{si}.npz")
            if not os.path.exists(ck):
                print(f"MISSING {ck}"); sys.exit(1)
            z = np.load(ck)
            g1 = (z["sg1"], z["eg1"], z["sb1"], z["eb1"]); g2 = (z["sg2"], z["eg2"], z["sb2"], z["eb2"])
            h1, h2 = float(z["h1"]), float(z["h2"])
            with torch.no_grad():
                s1 = (N.load_net(w, si, ni, et, 1, dtype=torch.float32)(h48)-hy).abs().mean().item()
                s2 = (N.load_net(w, si, ni, et, 2, dtype=torch.float32)(h48)-hy).abs().mean().item()
            good = (h1 - s1 < GAP) and (h2 - s2 < GAP)
            if good:
                GEN[(et_i, ni, si, 1)] = g1; GEN[(et_i, ni, si, 2)] = g2; kept += 1
            else:
                zg = tuple(np.zeros_like(a) for a in g1); GEN[(et_i, ni, si, 1)] = zg; GEN[(et_i, ni, si, 2)] = zg; dropped += 1
            rows.append((et, si, NNS[ni], f"q1 {h1:.4f}/{s1:.4f}(+{h1-s1:.4f})  q2 {h2:.4f}/{s2:.4f}(+{h2-s2:.4f})  {'KEEP' if good else 'drop'}"))

print(f"GAP<{GAP}:  kept {kept}  dropped {dropped}  (of {kept+dropped} distinct configs; mse-copies follow abs)")
for et, si, nns, info in rows:
    print(f"  {et} n{si} {XDIM[si]}x{YDIM[si]} nns{nns:<3}  {info}")
write_hvsym(OUT, lambda et, ni, si, q, ng, Dm1: GEN[(et, ni, si, q)])
print(f"wrote {OUT}")
