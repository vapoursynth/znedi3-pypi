"""
nnedi3_torch.py
===============

An *exact* PyTorch reference for the nnedi3 predictor network, plus a from-scratch
trainer. Two goals:

  (a) VALIDATE against znedi3 using the shipped `nnedi3_weights.bin`. The forward pass
      here is derived line-by-line from znedi3's `kernel.cpp` (`PredictorC` / `wae5`) and
      the file layout from `weights.cpp` (`read_nnedi3_weights`). A self-test loads the
      real .bin and checks this module against an independent NumPy port of znedi3's exact
      scalar algorithm (including its mean-fold optimisation).

  (b) RETRAIN independently with online / SGD gradient descent, reproducing tritical's
      procedure (random init, per-pattern updates, both abs and mse losses, two seeds for
      qual=2), then serialise the result back into the .bin layout.

Exactness notes
---------------
* The math is the *clean form*: normalise each window to zero-mean/unit-std, dot with the
  RAW weights, softmax-gate elliott experts, rescale by 5*sigma and re-add the mean. This is
  algebraically identical to znedi3's inference (which folds the mean into the weights as a
  load-time optimisation - see `subtract_mean` in weights.cpp); the self-test confirms it.
* znedi3 runs in float32 and (in SIMD paths) uses an approximate exp. This module defaults to
  float64. Expect agreement with the C scalar path (`cpu=none`, `slow_exp=1`) to ~1e-4, and
  ~1e-11 against the NumPy fold-port in double precision.
* Only the PREDICTOR is implemented (the ~150k-param network of the training discussion).
  Prescreener weights are parsed for correct byte accounting but their forward pass is not
  reproduced here.

Requires: numpy, torch.
"""

from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Constants (from znedi3/weights.h and znedi3.h) -- do not change; they define the file.
# --------------------------------------------------------------------------------------
XDIM = [8, 16, 32, 48, 8, 16, 32]     # NNEDI3_XDIM, indexed by nsize 0..6
YDIM = [6, 6, 6, 6, 4, 4, 4]          # NNEDI3_YDIM
NNS = [16, 32, 64, 128, 256]          # NNEDI3_NNS, indexed by nns-enum 0..4
ETYPES = ["abs", "mse"]               # file stores abs set, then mse set
WEIGHTS_SIZE_BYTES = 13_574_928       # NNEDI3_WEIGHTS_SIZE
FLT_EPSILON = 1.1920929e-7            # std::numeric_limits<float>::epsilon()
NSIZE_NAMES = ["8x6", "16x6", "32x6", "48x6", "8x4", "16x4", "32x4"]


# --------------------------------------------------------------------------------------
# Weight file parsing  (mirrors read_nnedi3_weights in weights.cpp, in order)
# --------------------------------------------------------------------------------------
class NNEDI3Weights:
    """Parsed nnedi3_weights.bin. `predictors[(etype, nsize, nns_idx)]` -> dict of arrays."""

    def __init__(self):
        self.prescreener_old = {}
        self.prescreener_new = []          # 3 variants (forward not implemented here)
        self.predictors = {}               # (etype:str, nsize:int, nns_idx:int) -> dict

    @staticmethod
    def read(path: str) -> "NNEDI3Weights":
        raw = np.fromfile(path, dtype="<f4")
        if raw.nbytes != WEIGHTS_SIZE_BYTES:
            raise ValueError(f"expected {WEIGHTS_SIZE_BYTES} bytes, got {raw.nbytes}")
        w = NNEDI3Weights()
        off = 0

        def take(n):
            nonlocal off
            chunk = raw[off:off + n]
            off += n
            return chunk

        # --- old prescreener ---
        w.prescreener_old = dict(
            kernel_l0=take(4 * 48).reshape(4, 48).copy(),
            bias_l0=take(4).copy(),
            kernel_l1=take(4 * 4).reshape(4, 4).copy(),
            bias_l1=take(4).copy(),
            kernel_l2=take(4 * 8).reshape(4, 8).copy(),
            bias_l2=take(4).copy(),
        )

        # --- new prescreeners x3 (stored SIMD-shuffled; unshuffle to row-major) ---
        for _ in range(3):
            k0_shuf = take(4 * 64)
            b0 = take(4).copy()
            k1_shuf = take(4 * 4)
            b1 = take(4).copy()
            k0 = np.empty((4, 64), np.float32)
            k1 = np.empty((4, 4), np.float32)
            for nrn in range(4):
                for k in range(64):
                    k0[nrn, k] = k0_shuf[(k // 8) * 32 + nrn * 8 + k % 8]
                for k in range(4):
                    k1[nrn, k] = k1_shuf[k * 4 + nrn]
            w.prescreener_new.append(dict(kernel_l0=k0, bias_l0=b0, kernel_l1=k1, bias_l1=b1))

        # --- predictors: for etype in {abs,mse}: for nns: for nsize: q1 then q2 ---
        for etype in ETYPES:
            for nns_idx, nns in enumerate(NNS):
                for nsize in range(7):
                    fs = XDIM[nsize] * YDIM[nsize]
                    d = {}
                    d["softmax_q1"] = take(nns * fs).reshape(nns, fs).copy()
                    d["elliott_q1"] = take(nns * fs).reshape(nns, fs).copy()
                    d["softmax_bias_q1"] = take(nns).copy()
                    d["elliott_bias_q1"] = take(nns).copy()
                    d["softmax_q2"] = take(nns * fs).reshape(nns, fs).copy()
                    d["elliott_q2"] = take(nns * fs).reshape(nns, fs).copy()
                    d["softmax_bias_q2"] = take(nns).copy()
                    d["elliott_bias_q2"] = take(nns).copy()
                    w.predictors[(etype, nsize, nns_idx)] = d

        assert off == raw.size, f"layout mismatch: consumed {off} of {raw.size}"
        return w


# --------------------------------------------------------------------------------------
# The trainable network (one predictor network = softmax bank + elliott bank).
# qual=2 at inference is just the average of two of these (two seeds).
# --------------------------------------------------------------------------------------
class NNEDI3Net(nn.Module):
    """One nnedi3 predictor network. forward(windows[N, xdim*ydim]) -> prediction[N].

    Window pixels must be in row-major (row 0..ydim-1, col 0..xdim-1) order, matching the
    weight filter layout and `gather_windows` below.
    """

    def __init__(self, nsize: int, nns_idx: int, symmetric=False, init_std: float = 0.05, xdim=None, ydim=None):
        super().__init__()
        self.nsize = nsize
        self.nns_idx = nns_idx
        # xdim/ydim default to the nsize preset, but can be overridden -- notably to xdim = N-1 (odd)
        # for the target-centred (N-1)-wide net the folded C++ predictor consumes. With an ODD xdim
        # the plain Klein-4 "hv" mode is EXACT (the centre column IS the target, its own fixed point),
        # so symmetric="hv" + xdim=N-1 is the training-time equivalent of the C++ folded model.
        self.xdim = XDIM[nsize] if xdim is None else xdim
        self.ydim = YDIM[nsize] if ydim is None else ydim
        self.D = self.xdim * self.ydim
        self.nns = NNS[nns_idx]
        # Symmetric weight tying by flip-orbit sharing. Two modes:
        #   "hv" / True : full Klein-4 (self + h-flip + v-flip + 180 deg), nns/4 generators. Gives
        #                 EXACT h AND v mirror invariance -- BUT with an even xdim the target column
        #                 sits off-centre (index xdim/2-1, i.e. 23 of 48: 23 left, 24 right), so the
        #                 horizontal flip maps the target to a 1-px-shifted pixel. That half-wrong
        #                 tie forces every generator to serve a shifted task and empirically caps
        #                 accuracy *below* linear. Retained for experiments; not recommended.
        #   "v"         : vertical Z2 (self + v-flip), nns/2 generators. The target sits at the
        #                 vertical centre of the ydim rows, so the vertical flip is an EXACT task
        #                 symmetry -- invariance for free with no accuracy cost. This is the one to use.
        # softmax_exp + wae5 are permutation-invariant over experts, so flipping the input permutes
        # the orbit and leaves the output exactly invariant.
        # "hv23" = Klein-4 about the TARGET column (not the window centre): drops the extra right
        # column so the remaining odd width is target-centred, making the h-flip exact too. See _orbit.
        mode = {False: None, True: "hv", "hv": "hv", "v": "v", "hv23": "hv23"}.get(symmetric, "bad")
        if mode == "bad":
            raise ValueError(f"symmetric must be False/True/'hv'/'v'/'hv23', got {symmetric!r}")
        self.sym_mode = mode
        self.symmetric = mode is not None
        m = self.nns
        if mode in ("hv", "hv23"):
            if self.nns % 4 != 0:
                raise ValueError("hv/hv23 symmetric mode needs nns divisible by 4")
            m = self.nns // 4
        elif mode == "v":
            if self.nns % 2 != 0:
                raise ValueError("v symmetric mode needs nns divisible by 2")
            m = self.nns // 2
        self.n_free = m
        # softmax (gate) bank and elliott (value) bank (free generators if symmetric, else full)
        self.Ws = nn.Parameter(torch.randn(m, self.D) * init_std)
        self.bs = nn.Parameter(torch.zeros(m))
        self.We = nn.Parameter(torch.randn(m, self.D) * init_std)
        self.be = nn.Parameter(torch.zeros(m))
        if mode == "hv":   # buffers for the Klein-4 symmetry-adapted fast forward
            self.register_buffer("_orbit_idx", self._make_orbit_idx(), persistent=False)
            self.register_buffer("_Hc", torch.tensor(   # Klein-4 character table over [e,h,v,hv]
                [[1., 1., 1., 1.], [1., -1., 1., -1.], [1., 1., -1., -1.], [1., -1., -1., 1.]]),
                persistent=False)

    def _make_orbit_idx(self):
        """Flat indices of the 4 positions {e, h, v, hv} of each of the D/4 window flip-orbits."""
        ydim, xdim = self.ydim, self.xdim
        rows = []
        for r in range(ydim // 2):
            for c in range(xdim // 2):
                R, C = ydim - 1 - r, xdim - 1 - c
                rows.append([r * xdim + c, r * xdim + C, R * xdim + c, R * xdim + C])
        return torch.tensor(rows, dtype=torch.long)   # [M, 4], M = D/4

    def _orbit(self, Wg, bg):
        """Expand generators into the full nns experts via flip-orbits (identity if not symmetric).
        hv order: [generators, h-flips, v-flips, 180°]; v order: [generators, v-flips].
        All orbit members share the generator's bias."""
        if not self.symmetric:
            return Wg, bg
        w = Wg.view(-1, self.ydim, self.xdim)
        if self.sym_mode == "v":
            full = torch.cat([w, torch.flip(w, [1])], dim=0)          # self + vertical flip (Z2)
            return full.reshape(-1, self.D), bg.repeat(2)
        if self.sym_mode == "hv23":
            # Klein-4 about the TARGET column (index xdim//2-1), not the window centre. The last
            # column (index xdim-1, the extra one the +1 gather bias puts on the right) is pinned to
            # 0 so the remaining xdim-1 columns are odd and centred on the target -> the h-flip
            # (c -> xdim-2-c) becomes an EXACT task symmetry. Materialises to the standard xdim-wide
            # experts (dropped column = 0), so it still writes/loads as an ordinary znedi3 .bin.
            mask = w.new_ones(self.xdim); mask[-1] = 0.0
            wm = w * mask                                             # zero the dropped right column
            act = self.xdim - 1
            wh = torch.cat([wm[:, :, :act].flip(-1), torch.zeros_like(wm[:, :, act:])], dim=2)
            full = torch.cat([wm, wh, wm.flip(1), wh.flip(1)], dim=0)  # self, h, v, hv (about target)
            return full.reshape(-1, self.D), bg.repeat(4)
        full = torch.cat([w, torch.flip(w, [2]), torch.flip(w, [1]), torch.flip(w, [1, 2])], dim=0)
        return full.reshape(-1, self.D), bg.repeat(4)

    def _activations_fast(self, xp):
        """Compute the nns softmax+elliott pre-activations with ~4x fewer multiplies, exploiting
        the flip-orbit structure. Each generator's whole orbit-of-4 is computed from 4 partial
        dot products over the D/4 orbit reps (D mults instead of 4D), via the Klein-4 transform."""
        N = xp.shape[0]
        idx, Hc = self._orbit_idx, self._Hc.to(xp.dtype)       # [M,4], [4,4]
        Xh = xp[:, idx] @ Hc.t()                               # [N,M,4]  gather orbit values + transform
        out = []
        for W, b in ((self.Ws, self.bs), (self.We, self.be)):
            Wh = W[:, idx] @ Hc.t()                            # [ngen,M,4]  (adds; weights transform)
            # P[n,k,r] = sum_o Xh[n,o,r] Wh[k,o,r]  -> 4 batched GEMMs of [N,M]x[M,ngen] = N*D*ngen mults
            P = torch.bmm(Xh.permute(2, 0, 1), Wh.permute(2, 1, 0))         # [4,N,ngen]
            a = 0.25 * (P.permute(1, 2, 0) @ Hc)               # [N,ngen,4]  sign-combine to the 4 flips
            out.append(a.permute(0, 2, 1).reshape(N, self.nns) + b.repeat(4))
        return out[0], out[1]

    def forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Same result as forward() but via the reduced-multiply path (Klein-4 nets only;
        other modes fall back to the standard forward)."""
        if self.sym_mode != "hv":
            return self.forward(x)
        mu = x.mean(dim=1, keepdim=True)
        var = ((x * x).mean(dim=1, keepdim=True) - mu * mu).clamp_min(0.0)
        flat = var < FLT_EPSILON
        std = torch.sqrt(var)
        inv_std = torch.where(flat, torch.zeros_like(std), 1.0 / torch.sqrt(var.clamp_min(FLT_EPSILON)))
        a, e = self._activations_fast((x - mu) * inv_std)
        a = a - a.mean(dim=1, keepdim=True)
        gate = torch.exp(a.clamp(-80.0, 80.0)); val = e / (1.0 + e.abs())
        wsum = gate.sum(1); vsum = (gate * val).sum(1)
        W = torch.where(wsum > 1e-10, vsum / wsum.clamp_min(1e-30), torch.zeros_like(wsum))
        return mu.squeeze(1) + 5.0 * std.squeeze(1) * W

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, D]. Per-window mean/std normalisation (kernel.cpp gather_input).
        mu = x.mean(dim=1, keepdim=True)                       # [N,1]
        var = (x * x).mean(dim=1, keepdim=True) - mu * mu      # [N,1]
        var = var.clamp_min(0.0)
        flat = var < FLT_EPSILON
        std = torch.sqrt(var)                                  # 0 where flat
        inv_std = torch.where(flat, torch.zeros_like(std),
                              1.0 / torch.sqrt(var.clamp_min(FLT_EPSILON)))
        xp = (x - mu) * inv_std                                # [N,D], 0 where flat

        Ws, bs = self._orbit(self.Ws, self.bs)                 # [nns,D],[nns] (materialise if symmetric)
        We, be = self._orbit(self.We, self.be)
        a = xp @ Ws.t() + bs                                   # [N,nns] softmax pre-act
        e = xp @ We.t() + be                                   # [N,nns] elliott pre-act
        # Subtract the per-window MEAN logit before the clamp. This is not just a stability
        # trick: znedi3's load-time mean-fold (weights.cpp subtract_mean) shifts every softmax
        # logit by exactly inv_std*(x . mean_filter) + mean_bias, which equals mean_j(logit_j).
        # So mean-subtraction reproduces znedi3's folded logits bit-for-bit, ±80 clamp included.
        a = a - a.mean(dim=1, keepdim=True)
        gate = torch.exp(a.clamp(-80.0, 80.0))                 # softmax_exp
        val = e / (1.0 + e.abs())                              # elliott
        wsum = gate.sum(dim=1)                                 # [N]
        vsum = (gate * val).sum(dim=1)                         # [N]
        # wae5: mstd[3] += (5*vsum/wsum)*std + mu  (or just mu if wsum tiny)
        W = torch.where(wsum > 1e-10, vsum / wsum.clamp_min(1e-30), torch.zeros_like(wsum))
        return mu.squeeze(1) + 5.0 * std.squeeze(1) * W        # [N]

    # ---- load real weights from a parsed .bin ----
    @torch.no_grad()
    def load_from(self, d: dict, quality: int):
        if self.symmetric:
            raise RuntimeError("load_from is for non-symmetric nets; shipped weights aren't flip-closed")
        q = "q1" if quality == 1 else "q2"
        self.Ws.copy_(torch.as_tensor(d[f"softmax_{q}"], dtype=self.Ws.dtype))
        self.bs.copy_(torch.as_tensor(d[f"softmax_bias_{q}"], dtype=self.bs.dtype))
        self.We.copy_(torch.as_tensor(d[f"elliott_{q}"], dtype=self.We.dtype))
        self.be.copy_(torch.as_tensor(d[f"elliott_bias_{q}"], dtype=self.be.dtype))
        return self

    # ---- export weights as numpy (full nns experts, bin order; materialises the orbit if symmetric) ----
    @torch.no_grad()
    def export(self):
        Ws, bs = self._orbit(self.Ws, self.bs)
        We, be = self._orbit(self.We, self.be)
        return dict(
            softmax=Ws.detach().cpu().numpy().astype("<f4"),
            softmax_bias=bs.detach().cpu().numpy().astype("<f4"),
            elliott=We.detach().cpu().numpy().astype("<f4"),
            elliott_bias=be.detach().cpu().numpy().astype("<f4"),
        )


def predict_qual2(net_q1: NNEDI3Net, net_q2: NNEDI3Net, x: torch.Tensor) -> torch.Tensor:
    """qual=2 inference: average two seed networks (kernel.cpp: /2 of the accumulated resids)."""
    return 0.5 * (net_q1(x) + net_q2(x))


def load_net(weights: NNEDI3Weights, nsize: int, nns_idx: int, etype: str, quality: int,
             dtype=torch.float64) -> NNEDI3Net:
    net = NNEDI3Net(nsize, nns_idx).to(dtype)
    net.load_from(weights.predictors[(etype, nsize, nns_idx)], quality)
    return net


# --------------------------------------------------------------------------------------
# Independent NumPy port of znedi3's EXACT scalar algorithm (fold form) -- the reference
# the module is validated against. Mirrors subtract_mean() + PredictorC::process().
# --------------------------------------------------------------------------------------
def _fold(Sq, Eq, Sb):
    """subtract_mean() from weights.cpp: fold the per-window mean into the weights."""
    nns = Sq.shape[0]
    s_mean = Sq.mean(1)
    e_mean = Eq.mean(1)
    mean_filter = (Sq - s_mean[:, None]).mean(0)          # (1/nns) sum_nn (Sq[nn]-s_mean[nn])
    mean_bias = Sb.mean()
    Sf = Sq - s_mean[:, None] - mean_filter[None, :]
    Ef = Eq - e_mean[:, None]
    Sbf = Sb - mean_bias
    return Sf, Ef, Sbf


def reference_predict(d: dict, windows: np.ndarray, qual: int) -> np.ndarray:
    """znedi3's exact predictor math (fold form), float64. windows: [N, D] raw pixels."""
    x = windows.astype(np.float64)
    n = x.shape[1]
    mu = x.mean(1)
    var = np.clip((x * x).mean(1) - mu * mu, 0.0, None)
    flat = var < FLT_EPSILON
    std = np.sqrt(var)
    inv = np.where(flat, 0.0, 1.0 / np.sqrt(np.clip(var, FLT_EPSILON, None)))
    acc = np.zeros(x.shape[0])
    for q in ("q1", "q2")[:qual]:
        Sq = d[f"softmax_{q}"].astype(np.float64)
        Eq = d[f"elliott_{q}"].astype(np.float64)
        Sb = d[f"softmax_bias_{q}"].astype(np.float64)
        Eb = d[f"elliott_bias_{q}"].astype(np.float64)
        Sf, Ef, Sbf = _fold(Sq, Eq, Sb)
        a = (x @ Sf.T) * inv[:, None] + Sbf[None, :]      # scale=inv_std applied to dot
        e = (x @ Ef.T) * inv[:, None] + Eb[None, :]
        gate = np.exp(np.clip(a, -80.0, 80.0))
        val = e / (1.0 + np.abs(e))
        wsum = gate.sum(1)
        vsum = (gate * val).sum(1)
        acc += np.where(wsum > 1e-10, 5.0 * vsum / np.where(wsum > 1e-10, wsum, 1.0) * std + mu, mu)
    return acc / qual


# --------------------------------------------------------------------------------------
# Validation: module (clean form) vs znedi3 algorithm (fold form) on the real .bin.
# --------------------------------------------------------------------------------------
def _smooth_field(rng, H, W, peak):
    """An image-like (spatially-correlated) field, so gathered windows are in-distribution."""
    f = rng.standard_normal((H, W))
    for _ in range(8):  # a few box blurs -> smooth, like real pixel neighbourhoods
        f = (f + np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)) / 5.0
    f -= f.min()
    f /= (f.max() + 1e-12)
    return (f * peak).astype(np.float32)


def validate(bin_path: str, configs=None, n=4096, seed=0, peak=1.0):
    weights = NNEDI3Weights.read(bin_path)
    rng = np.random.default_rng(seed)
    if configs is None:
        configs = [(3, 4, "abs"), (3, 4, "mse"), (0, 0, "abs"), (6, 2, "mse")]  # flagship + small
    H = W = 256
    field = _smooth_field(rng, H, W, peak)
    print(f"{'config':<22}{'qual':<6}{'maxdiff':<12}{'meandiff'}")
    ok = True
    for nsize, nns_idx, etype in configs:
        xdim, ydim = XDIM[nsize], YDIM[nsize]
        ys = rng.integers(ydim, H - ydim, size=n)
        xs = rng.integers(xdim, W - xdim, size=n)
        windows = gather_windows(field, nsize, ys, xs).astype(np.float32)
        d = weights.predictors[(etype, nsize, nns_idx)]
        for qual in (1, 2):
            ref = reference_predict(d, windows, qual)
            if qual == 1:
                net = load_net(weights, nsize, nns_idx, etype, 1)
                got = net(torch.as_tensor(windows, dtype=torch.float64)).detach().numpy()
            else:
                n1 = load_net(weights, nsize, nns_idx, etype, 1)
                n2 = load_net(weights, nsize, nns_idx, etype, 2)
                got = predict_qual2(n1, n2, torch.as_tensor(windows, dtype=torch.float64)).detach().numpy()
            dmax = np.max(np.abs(got - ref))
            dmean = np.mean(np.abs(got - ref))
            tag = f"{NSIZE_NAMES[nsize]}/{NNS[nns_idx]}/{etype}"
            print(f"{tag:<22}{qual:<6}{dmax:<12.3e}{dmean:.3e}")
            ok = ok and dmax < 1e-6
    print("PASS" if ok else "FAIL (>1e-6 — investigate)")
    return ok


# --------------------------------------------------------------------------------------
# Window gathering (matches PredictFilter::process + PredictorC geometry).
# For output line y, znedi3 uses source rows y-3..y+2 sliced to ydim; columns are centred
# with a +1 bias:   window[r, c] = src[y - ydim//2 + r, x - xdim//2 + 1 + c].
# --------------------------------------------------------------------------------------
def gather_windows(src: np.ndarray, nsize: int, ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    xdim, ydim = XDIM[nsize], YDIM[nsize]
    ys = np.asarray(ys)[:, None, None]
    xs = np.asarray(xs)[:, None, None]
    rr = ys - ydim // 2 + np.arange(ydim)[None, :, None]        # [N,ydim,1]
    cc = xs - xdim // 2 + 1 + np.arange(xdim)[None, None, :]    # [N,1,xdim]
    win = src[rr, cc]                                           # [N,ydim,xdim]
    return win.reshape(win.shape[0], ydim * xdim)


def make_training_pairs(frame: np.ndarray, nsize: int, parity: int = 0, stride: int = 1):
    """Turn ONE progressive frame into (windows[N,D], targets[N]) in znedi3 geometry.

    Field-split: source field S = frame[parity::2]; the pixel interpolated in the gap
    between S[g-1] and S[g] is the true frame pixel at row 2g+parity-1. The window is the
    vertically-symmetric xdim x ydim block of S straddling that gap.

    Geometry CONFIRMED vs znedi3 (empirically): the target is the vertical centre of the window,
    which is exactly what znedi3 predicts for BOTH field parities. vsznedi3 maps source-field
    line k -> output line 2k+P and predicted line i -> 2i+(1-P), and the pad_load_t/pad_load_b
    split (znedi3_impl.cpp:537) supplies the per-parity line shift, so both parities feed a
    symmetric window around the target gap. Verified with the shipped weights as an oracle:
    predicting the shift=0 target gives ~0.008 mean abs error vs ~0.18 for a +/-1 field-line
    shift. Train on both parities for field symmetry (same function, more data).
    """
    xdim, ydim = XDIM[nsize], YDIM[nsize]
    S = frame[parity::2]                                        # source field
    Hf, W = S.shape
    g0, g1 = ydim // 2, Hf - (ydim - ydim // 2)                 # valid gap range in S
    x0, x1 = xdim // 2 - 1, W - (xdim - (xdim // 2 - 1))        # valid col range
    gs = np.arange(g0, g1, stride)
    xs = np.arange(x0, x1, stride)
    gg, xx = np.meshgrid(gs, xs, indexing="ij")
    gg, xx = gg.ravel(), xx.ravel()
    tgt_rows = 2 * gg + parity - 1
    valid = (tgt_rows >= 0) & (tgt_rows < frame.shape[0])
    gg, xx, tgt_rows = gg[valid], xx[valid], tgt_rows[valid]
    windows = gather_windows(S, nsize, gg, xx).astype(np.float32)
    targets = frame[tgt_rows, xx].astype(np.float32)
    return windows, targets


# --------------------------------------------------------------------------------------
# Training (online / SGD). tritical: random init, per-pattern updates, run to a minimum
# with "progressive stops" (plateau-triggered LR reduction), abs or mse loss.
# --------------------------------------------------------------------------------------
def train(net: NNEDI3Net,
          windows: np.ndarray, targets: np.ndarray,
          etype: str = "abs",
          epochs: int = 20,
          batch_size: int = 1,          # 1 == online GD (tritical); raise for GPU speed
          lr: float = 1e-3,
          momentum: float = 0.9,
          device: str = "cpu",
          val_frac: float = 0.05,
          seed: int = 0,
          verbose: bool = True):
    """Online/SGD training of ONE network. Returns the trained net. Run twice (different
    seeds) to produce the qual=2 pair; run per etype ('abs'->L1, 'mse'->MSE)."""
    torch.manual_seed(seed)
    net = net.to(device).to(torch.float32)
    X = torch.as_tensor(windows, dtype=torch.float32)
    Y = torch.as_tensor(targets, dtype=torch.float32)
    nval = max(1, int(len(X) * val_frac))
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed))
    vidx, tidx = perm[:nval], perm[nval:]
    Xt, Yt, Xv, Yv = X[tidx], Y[tidx], X[vidx], Y[vidx]

    loss_fn = nn.L1Loss() if etype == "abs" else nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # "progressive stops": drop LR when validation error plateaus.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    def evaluate(Xe, Ye):
        net.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(Xe), 65536):
                out.append(net(Xe[i:i + 65536].to(device)).cpu())
            pred = torch.cat(out)
            l1 = (pred - Ye).abs().mean().item()
            return l1

    for ep in range(epochs):
        net.train()
        order = torch.randperm(len(Xt), generator=torch.Generator().manual_seed(seed + ep))
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb, yb = Xt[idx].to(device), Yt[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
        vl1 = evaluate(Xv, Yv)
        sched.step(vl1)
        if verbose:
            print(f"epoch {ep:3d}  val L1={vl1:.5f}  lr={opt.param_groups[0]['lr']:.2e}")
    return net


# --------------------------------------------------------------------------------------
# Serialisation: assemble a full nnedi3_weights.bin from trained nets (inverse of read()).
# `nets[(etype, nsize, nns_idx, quality)] = NNEDI3Net`. Missing entries are copied from a
# base .bin (so you can retrain a subset). Prescreeners are copied from the base file.
# --------------------------------------------------------------------------------------
def write_weights(out_path: str, base_bin: str, nets: dict):
    base = NNEDI3Weights.read(base_bin)
    raw = np.fromfile(base_bin, dtype="<f4").copy()
    # locate predictor section start: after old + 3 new prescreeners
    off = 252 + 3 * 280
    for etype in ETYPES:
        for nns_idx, nns in enumerate(NNS):
            for nsize in range(7):
                fs = XDIM[nsize] * YDIM[nsize]
                for quality in (1, 2):
                    key = (etype, nsize, nns_idx, quality)
                    if key in nets:
                        ex = nets[key].export()
                        raw[off:off + nns * fs] = ex["softmax"].reshape(-1)
                        raw[off + nns * fs:off + 2 * nns * fs] = ex["elliott"].reshape(-1)
                        raw[off + 2 * nns * fs:off + 2 * nns * fs + nns] = ex["softmax_bias"]
                        raw[off + 2 * nns * fs + nns:off + 2 * nns * fs + 2 * nns] = ex["elliott_bias"]
                    off += 2 * nns * fs + 2 * nns
    assert off == raw.size, (off, raw.size)
    raw.astype("<f4").tofile(out_path)
    print(f"wrote {out_path} ({raw.nbytes} bytes)")


# --------------------------------------------------------------------------------------
# Real training mode. Pick a network with the SAME knobs nnedi3 uses to select it
# (nsize / nns / etype / qual), point at a corpus, train, and optionally merge into a .bin.
# --------------------------------------------------------------------------------------
def _resolve_nsize(s):
    s = str(s).lower()
    if s in NSIZE_NAMES:
        return NSIZE_NAMES.index(s)                 # "48x6"
    v = int(s)
    if 0 <= v <= 6:
        return v                                    # nnedi3 nsize enum
    raise SystemExit(f"bad --nsize {s!r}: use 0..6 or one of {NSIZE_NAMES}")


def _resolve_nns(s):
    v = int(s)
    if v in NNS:
        return NNS.index(v)                         # actual neuron count 16..256
    if 0 <= v <= 4:
        return v                                    # nnedi3 nns enum
    raise SystemExit(f"bad --nns {s!r}: use 0..4 or one of {NNS}")


def _resolve_etype(s):
    s = str(s).lower()
    if s in ("abs", "mse"):
        return s
    if s in ("0", "1"):
        return "abs" if s == "0" else "mse"
    raise SystemExit(f"bad --etype {s!r}: use abs|mse or 0|1")


def load_gray_frames(pattern):
    """Load images/.npy as grayscale (luma) float32 in [0,1]. Progressive frames only."""
    import glob
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no files matched {pattern!r}")
    frames = []
    for pth in paths:
        if pth.lower().endswith(".npy"):
            a = np.load(pth)
            if a.ndim == 3:
                a = a[..., 0]
        else:
            from PIL import Image
            a = np.asarray(Image.open(pth).convert("L"))
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255.0
        elif a.dtype == np.uint16:
            a = a.astype(np.float32) / 65535.0
        else:
            a = a.astype(np.float32)
        frames.append(a)
    return frames


def build_corpus(frames, nsize, stride=2, limit=None, seed=0):
    """Gather (windows, targets) from progressive frames, BOTH field parities. Per-frame
    subsampling keeps the total near `limit` without ever materialising more than that."""
    rng = np.random.default_rng(seed)
    per = max(1, limit // (len(frames) * 2)) if limit else None
    Ws, Ts = [], []
    for f in frames:
        if f.ndim != 2:
            raise SystemExit("frames must be 2-D (grayscale)")
        for parity in (0, 1):
            w, t = make_training_pairs(f, nsize, parity=parity, stride=stride)
            if per and len(w) > per:
                idx = rng.choice(len(w), size=per, replace=False)
                w, t = w[idx], t[idx]
            Ws.append(w); Ts.append(t)
    return np.concatenate(Ws), np.concatenate(Ts)


def _default_base_bin():
    cand = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnedi3_weights.bin")
    return cand if os.path.exists(cand) else None


def train_cli(args):
    nsize = _resolve_nsize(args.nsize)
    nns_idx = _resolve_nns(args.nns)
    etype = _resolve_etype(args.etype)
    print(f"network: nsize={NSIZE_NAMES[nsize]} ({XDIM[nsize]}x{YDIM[nsize]}, D={XDIM[nsize]*YDIM[nsize]}), "
          f"nns={NNS[nns_idx]}, etype={etype}, qual={args.qual}, symmetric={args.symmetric}")

    if args.synthetic > 0:
        rng = np.random.default_rng(args.seed)
        frames = [_smooth_field(rng, 256, 256, 1.0) for _ in range(args.synthetic)]
        print(f"corpus: {len(frames)} synthetic smooth frames")
    elif args.images:
        frames = load_gray_frames(args.images)
        print(f"corpus: {len(frames)} frames from {args.images!r}")
    else:
        raise SystemExit("provide --images <glob> or --synthetic <N>")

    W, T = build_corpus(frames, nsize, stride=args.stride, limit=args.limit, seed=args.seed)
    print(f"patterns: {len(W):,} of dim {W.shape[1]}  (~{W.nbytes / 1e9:.2f} GB in RAM)")

    nets = {}
    for quality in range(1, args.qual + 1):
        seed = args.seed + (quality - 1) * 100003              # distinct seed => the qual=2 ensemble
        online = " = online GD" if args.batch == 1 else ""
        print(f"\n=== training q{quality}  (seed {seed}, {etype} loss, batch {args.batch}{online}) ===")
        net = NNEDI3Net(nsize, nns_idx, symmetric=args.symmetric)
        if quality == 1:
            tot = sum(p.numel() for p in net.parameters())
            print(f"   free params: {tot:,}{'  (flip-orbit tied — 1/4 of full)' if args.symmetric else ''}")
        train(net, W, T, etype=etype, epochs=args.epochs, batch_size=args.batch,
              lr=args.lr, momentum=args.momentum, device=args.device, seed=seed)
        nets[(etype, nsize, nns_idx, quality)] = net.to("cpu").to(torch.float32)
        if args.save_pt:
            path = args.save_pt if args.qual == 1 else args.save_pt.replace(".pt", f".q{quality}.pt")
            torch.save(net.state_dict(), path)
            print(f"saved {path}")

    if args.out:
        base = args.base or _default_base_bin()
        if not base or not os.path.exists(base):
            raise SystemExit("--out needs --base <nnedi3_weights.bin> (full file to merge the trained net into)")
        write_weights(args.out, base, nets)
    else:
        print("\n(no --out given; trained net kept in memory only. Pass --out + --base to write a .bin.)")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="nnedi3 predictor: validate / train (PyTorch reference)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("validate", help="check the forward pass vs znedi3 on a .bin")
    pv.add_argument("bin", help="path to nnedi3_weights.bin")

    sub.add_parser("demo-train", help="tiny synthetic smoke test of the training loop")

    pt = sub.add_parser("train", help="train a predictor network (real mode)")
    g = pt.add_argument_group("network selection (same knobs as nnedi3)")
    g.add_argument("--nsize", required=True, help="0..6 or WxH: 8x6 16x6 32x6 48x6 8x4 16x4 32x4")
    g.add_argument("--nns", required=True, help="0..4 or actual count 16/32/64/128/256")
    g.add_argument("--etype", default="abs", help="abs|mse (or 0|1). Both are needed for a full file.")
    g.add_argument("--qual", type=int, default=1, choices=(1, 2),
                   help="1 = train q1; 2 = also train q2 (2nd seed) for the qual=2 ensemble")
    g.add_argument("--symmetric", action="store_true",
                   help="bake in exact h+v mirror invariance via flip-orbit tying (1/4 the free params; nns must be /4)")
    d = pt.add_argument_group("training data")
    d.add_argument("--images", help='glob of progressive frames (png/jpg/.../.npy), e.g. "corpus/*.png"')
    d.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="use N generated smooth frames instead of --images (pipeline test)")
    d.add_argument("--stride", type=int, default=2, help="pixel-position subsample step per frame (default 2)")
    d.add_argument("--limit", type=int, default=1_000_000, help="cap on total training patterns held in RAM")
    o = pt.add_argument_group("optimization")
    o.add_argument("--epochs", type=int, default=20)
    o.add_argument("--batch", type=int, default=1, help="1 = online GD (tritical); larger is faster, esp. on GPU")
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--momentum", type=float, default=0.9)
    o.add_argument("--device", default="cpu", help="cpu | cuda")
    o.add_argument("--seed", type=int, default=0)
    ow = pt.add_argument_group("output")
    ow.add_argument("--out", help="write a full .bin with the trained net(s) merged into --base")
    ow.add_argument("--base", help="base .bin to merge into (default: nnedi3_weights.bin next to this script)")
    ow.add_argument("--save-pt", dest="save_pt", help="also save raw trained net(s) as .pt")

    args = ap.parse_args()
    if args.cmd == "validate":
        validate(args.bin)
    elif args.cmd == "demo-train":
        rng = np.random.default_rng(0)
        frames = [_smooth_field(rng, 192, 192, 1.0) for _ in range(4)]
        W, T = build_corpus(frames, 4, stride=2, seed=0)       # 8x4 / nns=16
        print(f"{len(W)} training patterns, D={W.shape[1]}")
        net = NNEDI3Net(4, 0)
        train(net, W, T, etype="abs", epochs=8, batch_size=64, lr=5e-3)
    elif args.cmd == "train":
        train_cli(args)
