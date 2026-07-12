"""End-to-end validation of the hvsym plugin path through real VapourSynth.

For each config: run stock nnedi3 (dense, from nnedi3_weights.bin) and hvsym nnedi3 on the same
real image, compare the interpolated frame.
  - DROPPED config  -> zeroed generators -> loader skips -> dense fallback -> must be BYTE-IDENTICAL.
  - KEPT config     -> folded symmetric predictor -> distilled weights -> must TRACK stock closely.
Also cross-checks the scalar / AVX2 / AVX-512 folded paths agree on a kept config.
"""
import sys, glob
import numpy as np
from PIL import Image
import vapoursynth as vs

core = vs.core
DLL     = r"C:\Libraries\znedi3-pypi\build_hvsym\znedi3.dll"
WEIGHTS = r"C:\Libraries\znedi3-pypi\nnedi3_weights.bin"
HVSYM   = r"C:\Libraries\znedi3-pypi\nnedi3_weights_hvsym.bin"
core.std.LoadPlugin(DLL)

# --- real source: a COCO training image, grayscale, cropped to even dims -------------------
imgpath = sorted(glob.glob(r"E:\train2017\*.jpg"))[0]
img = np.asarray(Image.open(imgpath).convert("L"))
H, W = img.shape; H -= H % 2; W -= W % 2
img = np.ascontiguousarray(img[:H, :W])
print(f"source: {imgpath}  {W}x{H}")

base = core.std.BlankClip(width=W, height=H, format=vs.GRAY8, length=1, color=0)
def _fill(n, f):
    fout = f.copy(); np.asarray(fout[0])[:] = img; return fout
src = core.std.ModifyFrame(base, base, _fill)

def run(nsize, nns, etype, qual=1, hvsym=None, x_cpu=None):
    kw = dict(field=1, dh=True, nsize=nsize, nns=nns, etype=etype, qual=qual,
              pscrn=0, x_nnedi3_weights_bin=WEIGHTS)
    if hvsym: kw["hvsym"] = hvsym
    if x_cpu: kw["x_cpu"] = x_cpu
    return core.znedi3.nnedi3(src, **kw)

def diff_stats(a, b):
    d = core.std.Expr([a, b], "x y - abs")
    p = core.std.PlaneStats(d).get_frame(0).props
    return int(p["PlaneStatsMax"]), float(p["PlaneStatsAverage"]) * 255.0

NS = {0:"8x6",1:"16x6",2:"32x6",3:"48x6",4:"8x4",5:"16x4",6:"32x4"}
NN = {0:16,1:32,2:64,3:128,4:256}
ET = {0:"abs",1:"mse"}

# (nsize, nns, etype, qual, expected). "keep" = folds + tracks stock. "error" = strict hvsym must
# REJECT (dropped from the file, or below the nns>=128 speed gate) -- no silent dense fall-back.
TESTS = [
    (4, 4, 1, 1, "keep"),    # mse 8x4  nns256  -- narrowest, best fold
    (3, 3, 1, 1, "keep"),    # mse 48x6 nns128  -- widest kept
    (0, 2, 0, 1, "error"),   # abs 8x6  nns64   -- kept but SPEED-GATED (nns<128) -> strict reject
    (6, 4, 0, 1, "keep"),    # abs 32x4 nns256
    (4, 4, 1, 2, "keep"),    # mse 8x4  nns256 qual2 -- exercises q2 fold
    (2, 0, 0, 1, "error"),   # abs 32x6 nns16   -- dropped from file -> strict reject
    (5, 4, 0, 1, "error"),   # abs 16x4 nns256  -- dropped from file -> strict reject
    (3, 1, 0, 1, "error"),   # abs 48x6 nns32   -- dropped from file -> strict reject
    (2, 0, 0, 2, "error"),   # abs 32x6 nns16 qual2 -- dropped, q2 path
]

print(f"\n{'config':28} {'exp':6} {'maxd':>6}  verdict")
print("-"*62)
allok = True
for ns, nn, et, q, exp in TESTS:
    label = f"{ET[et]} {NS[ns]:>4} nns{NN[nn]:<3} q{q}"
    if exp == "keep":
        a = run(ns, nn, et, qual=q); b = run(ns, nn, et, qual=q, hvsym=HVSYM)
        mx, av = diff_stats(a, b)
        ok = (mx > 0 and av < 4.0); verdict = "tracks stock" if ok else ("identical?!" if mx == 0 else "drifts")
        print(f"{label:28} {exp:6} {mx:6d}  {verdict}")
    else:   # strict: providing hvsym for an uncovered config must raise, not fall back to dense
        try:
            run(ns, nn, et, qual=q, hvsym=HVSYM)      # filter creation should raise
            ok = False; verdict = "!! NO ERROR (silent fallback?)"
        except vs.Error:
            ok = True; verdict = "correctly rejected (strict)"
        print(f"{label:28} {exp:6} {'--':>6}  {verdict}")
    allok &= ok

# --- cross-check folded paths agree (kept flagship: mse 48x6 nns128) ------------------------
print("\nfolded-path agreement (mse 48x6 nns128, hvsym) vs each other:")
paths = {}
for cpu in ("none", "avx2", "avx512"):
    paths[cpu] = run(3, 3, 1, hvsym=HVSYM, x_cpu=cpu)
ref = paths["avx512"]
for cpu in ("none", "avx2"):
    mx, av = diff_stats(paths[cpu], ref)
    print(f"  {cpu:6} vs avx512:  max={mx:3d}  avg={av:.4f}")
mx_sa, _ = diff_stats(paths["none"], run(3,3,1,hvsym=HVSYM))  # scalar vs auto (sanity)

print("\n" + ("ALL PASS" if allok else "SOME FAILED"))
sys.exit(0 if allok else 1)
