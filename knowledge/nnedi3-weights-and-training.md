# nnedi3 Predictor Weights — Structure, Training, and Reproduction Notes

*A consolidated, reproduction-oriented description of the nnedi3 predictor networks: what
they are, how they are wired, how they were trained, and what the training data looks like.
The goal is that a reader with a suitable image corpus could, in principle, regenerate a
weights file compatible with nnedi3 / znedi3.*

## Provenance legend

Every non-trivial claim is tagged with where it comes from, because the reproduction hinges
on separating fact from inference:

- **[S]** — Confirmed from the znedi3 source (`znedi3/kernel.cpp`, `weights.cpp`, `weights.h`,
  `znedi3.h`). These are ground truth for the *inference* math and the *file format*.
- **[T]** — Stated directly by tritical (nnedi3's author) in forum posts. Ground truth for the
  *training* procedure, to the precision he gave.
- **[I]** — Our inference / reconstruction. Plausible and internally consistent, but **not
  confirmed**; flagged so a reimplementer knows where to be careful.

---

## 1. Context: what nnedi3 does

nnedi3 performs neural-network, edge-directed interpolation: it predicts a "missing" pixel
from a small window of known surrounding pixels. It is used to deinterlace (predict one field
from the other) and to double resolution. For every output pixel there are two stages **[S]**:

1. **Prescreener** — a tiny classifier that decides whether the pixel is "simple" (a cheap
   cubic interpolation suffices) or "complex" (worth running the predictor). See §8.
2. **Predictor** — the main neural network that produces the pixel estimate. This is the
   ~150k-parameter model that the training discussion is about, and the focus of this document.

This document is primarily about the **predictor**. The prescreener is documented for
completeness in §8, but we have far less information about how it was trained.

---

## 2. The weights file at a glance

The canonical `nnedi3_weights.bin` is **13,574,928 bytes = 3,393,732 float32 values** **[S]**
(`NNEDI3_WEIGHTS_SIZE` in `weights.h`). It is a flat little-endian `float[]` blob with **no
header** — purely position-defined. It contains *every* configuration; the runtime selects a
slice by `(nsize, nns, etype, qual)`.

### 2.1 Configuration axes [S]

| Axis | Values | Meaning |
|------|--------|---------|
| `nsize` (window) | `8x6, 16x6, 32x6, 48x6, 8x4, 16x4, 32x4` | input neighborhood `xdim × ydim` |
| `nns` (neurons)  | `16, 32, 64, 128, 256` | number of experts (neuron pairs) |
| `etype` (loss)   | `abs`, `mse` | which trained-loss weight set |
| `qual` (quality) | `1`, `2` | use one network, or average two |

- `xdim` table: `{8, 16, 32, 48, 8, 16, 32}`; `ydim` table: `{6, 6, 6, 6, 4, 4, 4}`. So the
  input dimension `D = xdim·ydim ∈ {48, 96, 192, 288, 32, 64, 128}`.
- 7 window sizes × 5 neuron counts = **35 predictor configurations**, each existing in `abs`
  and `mse` variants, each holding a `q1` and a `q2` network → **35 × 2 × 2 = 140 predictor
  networks** total.

### 2.2 Parameter accounting (validates the whole picture) [S]

One predictor network = two weight banks (`softmax`, `elliott`), each `nns·D` weights plus
`nns` biases:

```
params(one network) = 2 · nns · (D + 1)
```

- Flagship `48x6` / `nns=256`: `2 · 256 · (288 + 1) = 147,968` ≈ "150000" as tritical rounds it.

Summing all 140 predictor networks:

```
per window size D, over all nns and all 4 (etype×qual) nets:
    Σ_nns (16+32+64+128+256 = 496);  4 nets → factor 8
    → 8 · 496 · (D+1) = 3968 · (D+1)
total predictor floats = 3968 · Σ_D (D+1)
    = 3968 · (49+97+193+289+33+65+129)
    = 3968 · 855 = 3,392,640 floats
```

Prescreeners add: old = 252 floats, new ×3 = 3·280 = 840 floats → 1,092 floats.

```
3,392,640 (predictors) + 1,092 (prescreeners) = 3,393,732 floats = 13,574,928 bytes  ✓
```

This matches the file size **exactly**, which confirms the parameter layout is complete and
correct — there are no hidden parameters, extra layers, or global scales.

---

## 3. Predictor network architecture (the model you train)

The predictor is a **softmax-gated mixture of experts** — a single "layer" of `nns` experts,
no hidden intermediate layer **[S]**.

### 3.1 Input window and geometry [S]

- The input is a raw `xdim × ydim` block of pixels (`D` values); **no engineered features** —
  the input dimensionality equals the pixel count exactly (e.g. 288 for `48x6`).
- Placement relative to the predicted pixel (`kernel.cpp` `PredictorC::process` /
  `gather_input`): `window_offset_y = 3 − ydim/2`, `window_offset_x = xdim/2 − 1`. The block
  straddles the target: `ydim` existing lines (half above, half below the missing line) and
  `xdim` columns centered on the target column.
- **For training** [S — empirically confirmed]: split a progressive frame into two fields and
  predict one from the other; the target is the pixel at the **vertical centre** of the window.
  From `vsznedi3.cpp` (doubling case) source-field line `k` maps to output line `2k+P` and
  predicted line `i` to `2i+(1−P)`, so the window is symmetric around the target gap for *both*
  field parities (the `pad_load_t`/`pad_load_b` split supplies the per-parity line shift —
  `znedi3_impl.cpp:537`). Confirmed with the shipped weights as an oracle: the centre target is
  reconstructed to ~0.008 mean abs error vs ~0.18 for a ±1 field-line offset. Implemented as
  `make_training_pairs` in `nnedi3_torch.py`.

### 3.2 Per-window normalization [S]

For each window with pixels `x_1 … x_D`:

```
μ = mean(x)                      (mstd[0])
σ = sqrt( mean(x²) − μ² )        (mstd[1]);  if σ² < FLT_EPSILON → treat as flat, output μ
x'_k = (x_k − μ) / σ             (zero mean, unit std over the window)
```

Note: znedi3 folds the mean subtraction into the weights at load time (`subtract_mean`,
making each softmax filter zero-sum and shifting the softmax bias) so it can dot with *raw*
input and apply only the `1/σ` scale dynamically. **That fold is an inference optimization,
not part of training** **[S]** — a from-scratch trainer should just normalize inputs to `x'`
explicitly, which is mathematically identical.

### 3.3 Forward pass (training form) [S]

Two banks of `nns` neurons each. Weights `Ws, We ∈ ℝ^{nns×D}`, biases `bs, be ∈ ℝ^{nns}`:

```
softmax pre-activation:  a_i = Σ_k Ws[i,k]·x'_k + bs[i]
elliott pre-activation:  e_i = Σ_k We[i,k]·x'_k + be[i]

a_i := a_i − mean_j(a_j)                        # centre logits before the clamp (see note below)
gate_i  = exp( clamp(a_i, −80, +80) )          # "softmax" bank
value_i = elliott(e_i),  elliott(z) = z/(1+|z|) # bounded to (−1, 1)

W = Σ_i gate_i · value_i  /  Σ_i gate_i          # ∈ (−1, 1)

prediction  ŷ = μ + 5·σ·W
```

Key constants **[S]**: the softmax argument is clamped to `±80` before `exp`; the value bank
uses the **elliott** soft-sign; the mixture output is multiplied by the fixed constant **5**,
rescaled by the window `σ`, and re-centered on the window mean `μ`. (This is `wae5()` in
`kernel.cpp`.) Because `value_i ∈ (−1,1)`, the residual `ŷ − μ` is structurally bounded to
`±5σ` — the predictor cannot place a pixel more than 5 local standard deviations from the
neighborhood mean.

**Softmax-logit centering [S — correction].** The `a_i := a_i − mean_j(a_j)` step is not
optional numerical hygiene; it *is* znedi3's computation. The load-time mean-fold
(`subtract_mean`, `weights.cpp`) shifts every softmax logit by a per-window constant that is
provably exactly `mean_j(a_j)` — the fold subtracts `inv_std·(x·mean_filter) + mean_bias`, which
equals the mean logit. Softmax is invariant to a constant shift *in exact arithmetic*, but the
`±80` clamp is **not**, so the shift changes the result whenever a logit would otherwise clamp.
Subtracting the **mean** logit (not the max) therefore reproduces znedi3 bit-for-bit, clamp
included. (Empirically: a PyTorch port without this diverged on the flagship q2 by up to ~0.06;
with it, agreement is ~1e-14 across all configs.)

### 3.4 qual = 2 combination [S]

`qual=2` runs **two independent networks** (`q1`, `q2`) of the *same* `(nsize, nns, etype)` and
averages their residuals:

```
ŷ = μ + 5σ·(W_q1 + W_q2) / 2
```

`qual` and `etype` are **orthogonal** axes (`znedi3_impl.cpp`: `etype` picks the `abs`/`mse`
set; `qual` picks `q1` vs `q1+q2` within it). So a full config has **four** networks:
`abs{q1,q2}`, `mse{q1,q2}`. `q1` and `q2` are **not** one-abs-one-mse.

---

## 4. Training objective

Per network, fit `ŷ` to the true pixel `y` over all training patterns. Equivalently, in
normalized terms, the mixture output `W` should approximate `(y − μ)/(5σ)` **[I from S]**.

Two loss functions are used, corresponding to `etype` **[S]** — and **both are trained**
(the file carries separate `abs` and `mse` sets regardless of `qual`):

- `etype = abs`  → minimize `|ŷ − y|`  → learns the **conditional median** of the target.
- `etype = mse`  → minimize `(ŷ − y)²` → learns the **conditional mean** of the target.

This mean-vs-median distinction is *why* both losses exist: the task (predicting a pixel from
its neighborhood) is genuinely one-to-many, so the two losses give differently-robust
estimators of the target distribution **[I]**.

**Open detail [I]:** tritical says he computes "the error in guessing the output value," which
suggests the loss is in the **pixel domain** (`|ŷ − y|`), implicitly weighting high-contrast
(high-σ) windows more. A normalized-domain loss (`|W − (y−μ)/(5σ)|`) is an equally valid
reimplementation choice and would weight patterns differently. Which he used is unconfirmed.

---

## 5. Training data (what we know)

- **Source:** patterns are grabbed from a **1000-frame video sequence deliberately assembled
  from many different sources** **[T]**. The diversity (many content types / cameras / film /
  animation) is the point — see below.
- **Frames must be progressive / full-resolution** so every held-out pixel has a true target
  **[I]**. Conceptually: emulate deinterlacing on clean frames — hold out a pixel, predict it
  from its neighborhood, use the real value as the label.
- **Volume:** ~**50–60 million** training patterns (each a `D`-dim window + one scalar target)
  **[T]**. For the flagship these are the "288-dimensional vectors."
- **Sampling:** ~60M patterns / 1000 frames ≈ **~60k sampled positions per frame** **[I]** —
  i.e. positions are **subsampled** per frame, not exhaustively taken. The exact subsampling
  rule is unknown.
- **Design philosophy — diversity over volume [I]:** 1000 frames is small by modern standards,
  but a `48×6` / ~148k-param model is learning *local* edge statistics, which are shared across
  content. A diverse 1000 frames richly oversamples the local-neighborhood distribution without
  needing millions of distinct images. Corpus *diversity* is the dominant quality lever here,
  more than raw count.
- Ratio: 60M patterns / 148k params ≈ **~400:1**, comfortable against overfitting **[I]**.

**Not known:** the exact clips/content, resolution(s), whether luma-only, how positions are
subsampled, any augmentation, and whether the same corpus is reused across all `nsize`/`nns`
(the smaller windows would simply be sub-blocks of the same neighborhoods).

---

## 6. Training procedure

- **Optimizer: online gradient descent (minibatch = 1)** **[T]**. The literal loop **[T]**:
  1. grab a single training pattern,
  2. run it through the model (forward pass, §3.3),
  3. compute the error of the predicted output value (§4),
  4. compute the partial derivative of that error w.r.t. every weight (analytic backprop),
  5. update the weights,
  6. repeat.
- **Initialization: random** starting weights **[T]**. (Distribution/scale not specified — a
  small random init is the natural assumption **[I]**.)
- **Stopping / schedule: "keep repeating until a minimum in the error surface is reached
  (progressive stops)"** **[T]**. Best read as a **plateau-triggered step-size reduction**
  (annealed learning rate): train at a rate until error stalls, drop the rate ("a stop"),
  continue, repeat until further drops stop helping **[I]**. Exact LR magnitudes and schedule
  are unknown.
- **Cost:** the flagship `6×48×256` network trains on ~60M pixels in **"a few days on a
  quadcore desktop"** — i.e. **CPU**, not GPU **[T]**.

### 6.1 Historical note: why online GD [T]

- nnedi**2** was trained with **separable CMA-ES** (a derivative-free evolution strategy) and
  is "basically equivalent to `4×12×32` in nnedi3 terms" (12×4 = 48-input window, `nns=32`,
  ≈ `2·(32·48+32) = 3,136` params).
- On ~1/5 the pixels, CMA-ES took ~3× longer at ~4× the CPU power (~12 CPU-units for 1/5 the
  data) vs online GD's ~1 unit for the full data **[T]**.
- **Why it matters [I]:** sep-CMA-ES scales to a few thousand parameters but not to ~148k;
  evolution strategies need populations and many full-data evaluations that blow up with
  dimension. Switching to **gradients** (cost-per-step ~independent of dimension) is what made
  the ~47×-larger nnedi3 flagship trainable at all. The softmax+elliott mixture architecture
  itself predates nnedi3 — nnedi2 already used it, just smaller.

### 6.2 qual=2 as a two-seed ensemble [I]

`q1` and `q2` are almost certainly **two runs from different random initial weights**
(random init + non-convex loss + "run to a minimum" ⇒ different minima ⇒ different functions).
Averaging two decorrelated same-loss predictors is a **variance-reduction ensemble**:

```
Var(average of 2) = σ²·(1 + ρ)/2      (ρ = error correlation between the two nets)
```

The two nets agree on easy/high-signal inputs and disagree on ambiguous ones; averaging only
helps on the disagreement, so the expected gain is **small** — consistent with `qual=2` being
a minor-quality / 2×-cost option. **Unconfirmed** whether `q1`/`q2` differ *only* by seed or
also by a different data shuffle/subset (the latter would decorrelate them further and enlarge
the gain).

### 6.3 Mirror symmetry — only the vertical flip is real [S/I]

**Only the vertical flip is an exact symmetry of the task; the horizontal flip is not.** This is
pure geometry of where the target sits in the window (`gather_windows`):

- *Vertical* — the predicted line is the gap at the **vertical centre** of the `ydim` rows
  (`ydim//2` above, `ydim - ydim//2` below → 3/3 for the 6-row flagship). A top↔bottom flip maps
  the gap to itself, so a vertically-mirrored window is the window of the same target in the
  vertically-mirrored image. `f(vflip x) = f(x)` is a **geometrically exact** tie. But only
  *geometrically*: natural images are **not** perfectly up-down symmetric (sky vs ground, overhead
  lighting), unlike left-right which nearly is — so *enforcing* vertical symmetry costs a small,
  measured amount. A non-symmetric net distilled from a shipped predictor matches it exactly
  (e.g. 8×6/256 held-out 0.0265 = 0.0265), while the best v-symmetric net plateaus ~0.0003 higher
  (0.0268) no matter how long it trains — a genuine **~1–3 % symmetry cost** on the small nets
  (the flagship-class nets absorb it and tie shipped). So vertical is the *usable* symmetry — exact
  invariance, no geometric obstruction — but it is not entirely free.
- *Horizontal* — columns are gathered with a **+1 bias** (`cc = x − xdim//2 + 1 + …`), so the
  target column lands at index `xdim//2 − 1` (**23 of 48**): 23 columns left, 24 right. The mirror
  axis is the window centre at **23.5**, half a pixel right of the target, so a left↔right flip
  sends the target to its *neighbour* (col 24 = pixel `x+1`). `f(hflip x)` is the prediction for a
  **different pixel**; `f(hflip x) = f(x)` would require `pixel_x = pixel_{x+1}`. It is **false**
  for any even `xdim` (all nnedi3 sizes: 8/16/32/48). Only an odd, target-centred width makes the
  horizontal flip exact — which `symmetric="hv23"` (below) manufactures by pinning the extra column.

**Measured on the shipped weights** (`nnedi3_torch.py`, flagship 48×6/256): the expert set is
**not** flip-closed. Best-matching flipped partner at relative L2 h = 0.51 / v = 0.36 / 180° = 0.59
vs a nearest-*unflipped*-neighbor baseline of **0.06** — flips land 6–10× farther, self-match ~2%.
So the experts are not mirror twins; the near-perfect symmetry of the *aggregate* influence map is
**emergent** (an average over ~250 asymmetric experts). Vertical residual < horizontal (0.36 <
0.51) is exactly the fingerprint of both-parity training (≈ vertical-flip augmentation, an exact
symmetry) with no horizontal augmentation (not one).

**Enforcing it** (`NNEDI3Net(..., symmetric="v" | "hv23" | "hv")`): tie experts into flip-orbits —
store `nns/k` "generator" filters, materialise each into its orbit (sharing one bias). `softmax_exp`
+ `wae5` are permutation-invariant over experts, so flipping the input permutes the orbit and the
output is **exactly** invariant. `export()` materialises the full `nns` experts → a standard,
znedi3-loadable `.bin`.

- **`symmetric="v"` (vertical Z2, `nns/2` generators) — the default.** Exact vertical invariance,
  **½ the free parameters, no accuracy cost**, and *fully* exact end-to-end: the flip permutes all
  `xdim` columns, so even znedi3's per-window mean/std is invariant. Uses every column.
- **`symmetric="hv23"` (Klein-4 about the *target column*, `nns/4` generators) — exact h+v, nearly
  free.** Pins the extra right column (index `xdim−1`) to 0 so the remaining `xdim−1` columns are
  odd and target-centred, making the horizontal flip `c → xdim−2−c` an *exact* tie. Distilled from
  shipped it ties the shipped net (held-out qual1 0.0276 vs 0.0275, qual2 0.0274 vs 0.0273 — the
  0.0001 gap is the price of no longer using column `x+24` in the prediction); closure residual
  **0.0e+00**, reproduction **2.0e-14**. CAVEAT: the horizontal symmetry is exact in the *predictor*
  (expert dot products), but znedi3 still normalises over all `xdim` columns including the pinned
  one, so end-to-end h-mirror consistency carries a one-column normalisation residual (vertical has
  none). Bit-exact horizontal consistency would need a C patch (a true odd-width window normalised
  over `xdim−1`). No quality gain over `"v"` — it buys the left↔right mirror guarantee, not sharpness.
- **`symmetric="hv"` / `True` (Klein-4 about the window centre 23.5, `nns/4`) — do not use.** Ties
  under the *false* horizontal flip (target off-centre), forcing every generator to serve both
  `pixel_x` and `pixel_{x+1}`. Distilled from shipped it plateaus at **0.0333 — worse than the
  0.0308 linear (bob) baseline** — and re-explains the earlier from-scratch "gate collapse" (the net
  was *structurally* capped by the bad tie, not failing to optimise). Cautionary only (keeps the
  `forward_fast` fast path).

**Delivered artifact — `nnedi3_weights_vsym.bin`** (shipped file with only the flagship replaced):
the 48×6/256 flagship, all four predictors (abs & mse × q1 & q2), re-cast as **vertical-Z2
symmetric** by distilling each shipped seed into an `symmetric="v"` net (top-usage-128 init →
SGD on the COCO corpus, loss `0.7·|net − shipped| + 0.3·|net − truth|`; the distillation term keeps
the gate specialised — it cannot match shipped's edge-directed output with a collapsed gate).
Verified: vertical flip-closure residual **0.0e+00** (exact), znedi3 reproduction **1.7e-14**,
held-out L1 **ties the shipped net** at both quals (qual1 0.0275 = 0.0275; qual2 0.0273 = 0.0273)
while halving the flagship's free parameters. Same-quality shipped weights, now with a mirror-
exactness guarantee (a scene and its vertical mirror deinterlace to exact mirrors). The mse
flagship weights are byte-identical to abs, so both etypes are covered by the same two seeds.

**Second artifact — `nnedi3_weights_hvsym.bin`** (same construction with `symmetric="hv23"`): exact
h+v symmetry about the target column, ¼-param generators, drop-in and znedi3-reproducible, ties
shipped within 0.0001. Kept alongside `_vsym` as the "also horizontally symmetric" option; `_vsym`
stays the recommended default (fully-exact symmetry, uses all columns, a hair sharper — see the
`hv23` caveat above).

**Whole-file artifact — `nnedi3_weights_vmirror.bin`** (every predictor, not just the flagship):
all **140 nets** (7 nsizes × 5 nns × 2 etype × 2 qual) re-cast as `symmetric="v"` by the same
distillation (`distill_all.py`: per-nsize window bank, top-usage-`nns/2` init, `0.7·distill +
0.3·truth`, batch 512 × 12 epochs to the v-plateau; 132 distinct nets in ~139 min, the 8 mse=abs
copies reused byte-identically). Verified: vertical flip-closure residual **0.0e+00 on all 140**,
mse=abs copies preserved, znedi3 reproduction **~1e-14**, drop-in (identical byte size). Held-out:
the flagship ties shipped, the smaller nets sit **~1–3 % behind** — the up-down-asymmetry cost
above, confirmed step-plateaued and matched by the non-symmetric ceiling test (so it is the
symmetry, not the optimiser or data). This is the file to use for a *fully* vertically-symmetric
nnedi3; `_vsym` (flagship-only) is the subset when only the top network matters.

A symmetry-adapted forward (`forward_fast`, Klein-4 only) computes the activations with a verified
**4× fewer multiplies** (bit-exact) but benchmarks ~1.5–3× *slower* than the stock dense GEMM on
CPU/PyTorch (gather + small batched GEMMs lose to one peak-efficiency dense GEMM) — no inference
speedup from array code; a hand-fused SIMD kernel would be needed. Symmetry's real win is training
(½–¼ the params) + exact mirror consistency, not speed.

**Side-findings from the same analysis:** the flagship's 256 experts are densely packed
(nearest-neighbor rel-dist ~0.06; 288-D random directions would be ~1.3 apart) — low effective
rank / over-parameterized, consistent with the flagship not out-predicting the small windows.
And the `etype` axis is *mostly* real but not entirely: across all 70 (config, quality) entries,
62 have genuinely distinct `abs`/`mse` weights (relative L2 ~0.6–1.3) and **8 are byte-identical
copies** — precisely the `nns=256` configs with `D ≥ 96` (16×6, 32×4, 32×6, 48×6), both q1 and q2,
i.e. the four most expensive networks in the file — almost certainly a training-budget cutoff on
the priciest nets. **Which loss the shared weights were trained *for* is recoverable from a weight
fingerprint** (abs → elliott/softmax RMS ratio `e/s` < ~0.5 and *large* softmax biases; mse → high
`e/s`, *small* biases — the ratio separates all 62 labeled pairs). Controlling for config (abs `e/s`
barely grows with nns; mse's climbs steeply), all four copies land on the **mse** trend
(`e/s` = 1.8–12 vs an abs ceiling of ~0.5), softmax-bias agreeing. So they are **mse-trained and
copied into the `abs` slots** — running the flagship at `etype=abs` returns *mse* weights, opposite
the `etype=0=abs` default (a fingerprint inference, not a certainty, but clean and unanimous).
Either way the file holds **132 distinct predictor networks, not 140**.

---

## 7. Step-by-step reproduction recipe

Assuming a diverse corpus of clean progressive frames:

1. **Build the pattern set.** For each frame, for a subsampled set of pixel positions, extract
   the `xdim × ydim` neighborhood (per §3.1 geometry) as input and the true pixel as target.
   Aim for ~50–60M patterns for the flagship. (Smaller windows can reuse sub-blocks of the same
   neighborhoods.)
2. **Pick a target config** `(xdim, ydim, nns, etype)`. There are 35 such; the flagship is
   `(48, 6, 256, ·)`.
3. **Initialize** the four parameter tensors `Ws, bs (softmax), We, be (elliott)` randomly.
4. **Online-GD train** (§6): for each pattern — normalize the window to `x'` (§3.2), forward
   (§3.3) to get `ŷ`, compute the `etype` loss vs `y`, backprop to all weights, take a step.
   Stream over the corpus repeatedly; anneal the step size on plateaus; stop at convergence.
5. **Repeat step 4 with a different random seed** to produce the second network → this pair
   becomes `q1` and `q2`.
6. **Repeat 2–5 for both losses** (`abs` and `mse`) and for **all 35 configs** to fill the file
   (140 networks). Smaller configs train much faster than the flagship.
7. **Serialize** in the exact order of §8 to produce a compatible `.bin`. (No mean-fold or
   SIMD-shuffle is needed in the *stored* predictor weights — znedi3 applies the mean-fold at
   load; store plain neuron-major weights. The *new-prescreener* weights, however, are stored
   pre-shuffled — see §8.)

A reference implementation of all of the above — `nnedi3_torch.py` (alongside this doc) — is
validated: its forward pass matches znedi3's exact algorithm to ~1e-14 (float64) on the shipped
`.bin` across the flagship and small configs, both `etype`, both `qual`; it trains end-to-end;
and it serialises back to a byte-identical `.bin`. Use it as the starting point.

---

## 8. Weights-file serialization (to emit a compatible .bin) [S]

Order exactly as `read_nnedi3_weights` (`weights.cpp`), all `float32`, no padding:

1. **Old prescreener:** `kernel_l0[4][48]` (192), `bias_l0[4]`, `kernel_l1[4][4]` (16),
   `bias_l1[4]`, `kernel_l2[4][8]` (32), `bias_l2[4]`. (252 floats)
2. **New prescreeners, i = 0,1,2** (three variants → `prescreen` = NEW_L0/L1/L2): for each,
   `kernel_l0` (4×64 = 256), `bias_l0[4]`, `kernel_l1` (4×4 = 16), `bias_l1[4]`. (280 each)
   - ⚠️ The new-prescreener kernels are stored **SIMD-shuffled**, not row-major. znedi3
     unshuffles as `kernel_l0[n][k] = stored[(k/8)·32 + n·8 + k%8]` and
     `kernel_l1[n][k] = stored[k·4 + n]`. To emit a compatible file, apply the inverse of these.
3. **Predictor models**, nested loops in this order:
   - outer: `m` over `{abs, mse}`
   - middle: `i` over `nns ∈ {16, 32, 64, 128, 256}` (grouped by neuron count)
   - inner: `j` over the 7 window sizes `{8x6,16x6,32x6,48x6,8x4,16x4,32x4}`
   - for each, in order: `softmax_q1` (nns·D), `elliott_q1` (nns·D), `softmax_bias_q1` (nns),
     `elliott_bias_q1` (nns), then the same four for **q2**.

   Predictor weights are stored **neuron-major, row-major** within each bank (filter `nn`
   occupies `[nn·D … nn·D + D)`), in plain form (no mean-fold applied in the file).

---

## 9. The prescreener (structure known, training mostly not)

A small classifier gating cubic-vs-predictor per pixel **[S]**. Two families:

- **Old** (`prescreen=OLD`): input a `12×4 = 48` window; 3 layers of 4 neurons; elliott
  activations on layers 0–1; decision `max(l2[2],l2[3]) ≤ max(l2[0],l2[1])` → use cubic.
- **New** (`prescreen=NEW_L0/L1/L2`, 3 trained variants): input a `16×4 = 64` window; 2 layers
  of 4 neurons; elliott on layer 0; four per-pixel outputs thresholded at `> 0`. Processes 4
  output pixels per pass.
- Prescreener inputs are normalized by a **fixed half-range** (`pixel_half`) with mean-subtracted
  kernels, i.e. a static normalization rather than the predictor's per-window `σ` **[S]**.

**Training [I]:** presumably a binary classification — label a pixel "needs the NN" when the
predictor meaningfully beats cubic interpolation there, and "cubic is fine" otherwise — trained
on the same corpus. We have **no direct statement** from tritical on the prescreener's training
objective, data labeling, or optimizer. Treat §9 training as unverified.

---

## 10. Open questions / unknowns

Confirmed vs. missing, for a faithful reproduction:

- **Loss domain** — pixel-value error vs normalized-residual error (§4). *Unconfirmed.*
- **Learning-rate schedule** — "progressive stops" ⇒ annealed, but no magnitudes/criteria.
- **Weight initialization** — distribution and scale of the "random" init.
- **Per-frame subsampling rule** — how the ~60k positions/frame are chosen.
- **q1 vs q2** — different seed only, or also a different data shuffle/subset? (The measured
  size of the real `qual=2` gain is itself diagnostic of this.)
- **Corpus specifics** — the actual 1000-frame content, resolution(s), luma-only?, augmentation.
- **Scope** — whether all 140 predictor nets (and all prescreeners) are trained fresh, or only
  a subset, in the current effort.
- **`etype` distinctness — RESOLVED (§6.3):** 62/70 config-quality entries have distinct
  `abs`/`mse` weights; 8 are byte-identical copies (the `nns=256`, `D ≥ 96` configs). So the file
  has 132 distinct predictor networks, not 140.
- **Prescreener training** — objective, labeling, optimizer (§9). Largely unknown.

---

## 11. Source map (for the [S] claims)

- `znedi3/weights.h` — config tables (`NNEDI3_XDIM/YDIM/NNS`), file size, `PredictorCoefficients`
  (softmax/elliott × q1/q2 × weights/bias), separate `abs`/`mse` model sets.
- `znedi3/kernel.cpp` — `PredictorC::gather_input` (mean/σ normalization, geometry offsets),
  `dot_product` (scale=`1/σ`, bias), `softmax_exp` (clamp ±80), `elliott`, `wae5` (the `5·σ·W + μ`
  combination), `qual=2` averaging.
- `znedi3/weights.cpp` — `read_nnedi3_weights` (exact serialization order; new-prescreener
  shuffle), `subtract_mean` (the inference-time mean-fold; not part of training).
- `znedi3/znedi3_impl.cpp` — `etype` selects the model set, `qual` selects q1 vs q1+q2 (the two
  axes are orthogonal).
- `znedi3/znedi3.h` — the `nsize`/`nns`/`qual`/`etype`/`prescreen` enums.

*The [T] claims come from tritical's forum posts (2026-07); the [I] claims are reconstruction.*
