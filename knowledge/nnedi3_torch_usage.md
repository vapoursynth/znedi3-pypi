# `nnedi3_torch.py` — usage

An exact PyTorch reference for the nnedi3 **predictor** network, with a from-scratch trainer.
Two uses: **validate** the forward pass against znedi3 on the shipped `nnedi3_weights.bin`, and
**train** new weights that drop straight back into a znedi3-compatible `.bin`.

For the theory (network structure, training procedure, file layout), see
[`nnedi3-weights-and-training.md`](nnedi3-weights-and-training.md). This file is just how to run it.

## Requirements

- Python 3.x with **numpy** and **torch**. **PIL** (Pillow) is needed only for `--images`
  loading (`.npy` frames work without it).
- Tested with `C:\vapoursynth-portable\python.exe` (numpy 2.4.6, torch 2.12.0+cpu; PIL/imageio/cv2 present).
- `nnedi3_weights.bin` (the standard 13,574,928-byte file) for `validate` and as a `--base` for training.

> On Windows, a bare `python` in some shells hits the Store stub. Use a real interpreter, e.g.
> `py`, or the full path: `& 'C:\vapoursynth-portable\python.exe' nnedi3_torch.py ...`

## Commands

```
python nnedi3_torch.py validate <nnedi3_weights.bin>   # forward pass vs znedi3 (should print PASS)
python nnedi3_torch.py demo-train                       # tiny synthetic smoke test of the trainer
python nnedi3_torch.py train  [options...]              # real training (see below)
```

### `validate`

Loads the `.bin`, and for a spread of configs (flagship 48x6/256 + small, both `abs`/`mse`,
both `qual`) checks this module against an independent NumPy port of znedi3's exact algorithm.
Prints per-config `maxdiff`/`meandiff` and `PASS` if all agree to < 1e-6 (typically ~1e-14).

```
python nnedi3_torch.py validate nnedi3_weights.bin
```

### `train`

Pick the network with the **same knobs nnedi3 uses to select it**, point at a corpus, train,
and (optionally) write a full `.bin`.

| Option | Default | Meaning |
|---|---|---|
| **network selection** | | |
| `--nsize` | *(required)* | `0..6` or `WxH`: `8x6 16x6 32x6 48x6 8x4 16x4 32x4` |
| `--nns` | *(required)* | `0..4` (enum) or the actual count `16/32/64/128/256` |
| `--etype` | `abs` | `abs`\|`mse` (or `0`\|`1`). **Both** are needed for a complete file. |
| `--qual` | `1` | `1` trains q1; `2` also trains q2 (second seed) — the qual=2 ensemble |
| `--symmetric` | off | bake in exact h+v mirror invariance (flip-orbit tying) — ¼ the free params, same capacity, exports to a normal `.bin`; `nns` must be divisible by 4 |
| **training data** | | |
| `--images` | — | glob of progressive frames (`png/jpg/…` or `.npy`), e.g. `"corpus/*.png"` |
| `--synthetic` | `0` | use `N` generated smooth frames instead of `--images` (pipeline test) |
| `--stride` | `2` | pixel-position subsample step per frame |
| `--limit` | `1000000` | cap on total training patterns held in RAM (per-frame subsampled) |
| **optimization** | | |
| `--epochs` | `20` | |
| `--batch` | `1` | `1` = online GD (tritical's method); larger = faster, esp. on GPU |
| `--lr` | `1e-3` | SGD learning rate (annealed on plateau = "progressive stops") |
| `--momentum` | `0.9` | |
| `--device` | `cpu` | `cpu` \| `cuda` |
| `--seed` | `0` | q2 automatically uses a different seed |
| **output** | | |
| `--out` | — | write a full `.bin` with the trained net(s) merged into `--base` |
| `--base` | *(next to script)* | base `.bin` to merge into; other configs copied through unchanged |
| `--save-pt` | — | also save raw trained net(s) as `.pt` |

Notes:
- The predictor is scale-equivariant, so frames are loaded as grayscale/luma in `[0,1]`; the
  learned weights are independent of that choice.
- `--qual 2` trains two independent networks (different seeds) on the same corpus; pass a
  different `--seed` and/or subsample differently if you want data-bagged diversity too.
- `--symmetric` ties the experts into horizontal+vertical flip-orbits, so the network is
  *exactly* invariant to mirroring the window. It cuts the free parameters 4× at the same expert
  capacity and materialises to a standard `.bin` on export — the shipped nnedi3 weights are *not*
  trained this way, so it's a genuine change, not a reformat.
- Only the **predictor** is trained here (the prescreener is not).

## Examples

Quick pipeline test (no corpus, no writing) — a few epochs on synthetic frames:
```
python nnedi3_torch.py train --nsize 8x4 --nns 16 --synthetic 4 --epochs 8 --batch 64
```

Train the flagship, both quality nets, from a real corpus, into a usable file:
```
python nnedi3_torch.py train \
    --nsize 48x6 --nns 256 --etype abs --qual 2 \
    --images "corpus/*.png" --limit 3000000 \
    --batch 1 \
    --out my_weights.bin --base nnedi3_weights.bin
```

Faster (GPU, minibatch) run of a mid-size config:
```
python nnedi3_torch.py train --nsize 32x6 --nns 128 --etype mse \
    --images "corpus/*.png" --batch 4096 --device cuda \
    --out my_weights.bin --base nnedi3_weights.bin
```

A **complete** replacement file requires all 35 `(nsize, nns)` configs × `{abs, mse}` × `{q1, q2}`
= 140 networks. Run `train` per config (each `--out` merges into the previous `--base`), or drive
the library API (below) in a loop. Smaller configs train much faster than the flagship.

## GPU training (`nnedi3_torch_gpu.py`)

`nnedi3_torch.py`'s `--device cuda` works but is slow: it keeps the corpus in host RAM and copies
every batch to the device, so with `--batch 1` (online GD) it's millions of tiny transfers — often
slower than CPU. **`nnedi3_torch_gpu.py`** is a GPU-first trainer that fixes this. It *imports the
validated core* from `nnedi3_torch` (same model, window geometry, and `.bin` writer — none of the
bit-exact math is forked); only the training loop is new:

- **Corpus resident on the device**, indexed and shuffled *on-device* — no per-batch host↔device copy.
- **uint8-packed by default** (4× less VRAM; lossless for 8-bit-sourced frames). A 24 GB GPU then
  holds ~80M×288 patterns — the full flagship — in VRAM. `--fp32-corpus` keeps float32 for non-8-bit sources.
- **Mini-batch SGD** (large `--batch`). This is what makes the GPU fast, and the one deliberate
  deviation from tritical's `batch=1` online GD; lower `--batch` to move back toward online behaviour.

Trained nets still merge into a standard, znedi3-loadable `.bin` (`nnedi3_torch.py validate` passes).

### Commands

```
python nnedi3_torch_gpu.py train        [options]   # corpus in RAM (bounded by --limit), resident on the GPU
python nnedi3_torch_gpu.py train-memmap [options]   # stream train_streaming.py's uint8 memmap to the GPU (flagship scale)
python nnedi3_torch_gpu.py demo                      # tiny synthetic smoke test
```

`train` takes the same network-selection (`--nsize/--nns/--etype/--qual/--symmetric`), data
(`--images/--synthetic/--stride/--limit`) and output (`--out/--base/--save-pt`) options as
`nnedi3_torch.py train`, with GPU-tuned defaults:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `auto` (cuda→mps→cpu) \| `cuda` \| `mps` \| `dml` \| `cpu` |
| `--batch` | `8192` | mini-batch size — large = full GPU utilisation |
| `--lr` | `1e-2` | SGD LR (plateau-annealed); higher than the CPU default to suit big batches |
| `--limit` | `4000000` | patterns held in RAM before packing onto the device |
| `--fp32-corpus` | off | keep the resident corpus in float32 (default is uint8) |

`train-memmap` adds `--work <dir>` (the `win.u8`/`tgt.u8`/`meta.txt` produced by `train_streaming.py`'s
extraction phase) and `--super` (super-batch shipped to the device at once, default 2,000,000); it
uses cosine-annealed LR. Use it when the corpus is too big to hold in RAM even as uint8.

### Examples

Flagship on an NVIDIA GPU, both quality nets, into a usable file:
```
python nnedi3_torch_gpu.py train --nsize 48x6 --nns 256 --etype abs --qual 2 \
    --images "corpus/*.png" --limit 40000000 --batch 16384 --device cuda \
    --out my_weights.bin --base nnedi3_weights.bin
```

Flagship at full ~60M scale from a pre-extracted memmap:
```
python nnedi3_torch_gpu.py train-memmap --nsize 48x6 --nns 256 --symmetric \
    --work E:\train\_ts_work --batch 16384 --device cuda \
    --out flagship.bin --base nnedi3_weights.bin
```

### GPU backends — NVIDIA, **AMD**, Apple, Intel

The `--device` string maps to a PyTorch backend:

| Hardware / OS | How to get it | `--device` |
|---|---|---|
| **NVIDIA** | a CUDA build of torch | `cuda` / `auto` |
| **AMD, Linux** | a **ROCm** build of torch — reports as CUDA, so the script runs unchanged | `cuda` / `auto` |
| **AMD, Windows** | **DirectML**: `pip install torch-directml` (any DirectX-12 GPU) | `dml` |
| **AMD, Windows via WSL2** | ROCm inside WSL2 (supported cards) | `cuda` / `auto` |
| **Apple Silicon** | Metal (MPS), built into mainstream torch | `mps` / `auto` |
| **Intel Arc** | DirectML (Windows) or IPEX/XPU (Linux) | `dml` |

So **yes, AMD GPUs work — two ways:**

- **ROCm (Linux, or Windows→WSL2)** is the real-performance path and needs *no code change*: AMD's
  PyTorch masquerades as CUDA (`torch.cuda.is_available()` is `True`, `device="cuda"` runs on the
  Radeon), so `--device cuda`/`auto` just works. Needs a ROCm-supported card (CDNA MI-series; RDNA3
  like RX 7900 XT/XTX are supported — older/consumer cards are hit-or-miss) and a ROCm torch wheel.
- **DirectML (Windows)** runs on essentially any DX-12 AMD (or Intel) GPU with no Linux: install
  `torch-directml` and pass `--device dml`. Easy, but **more experimental** — DirectML's op coverage
  is partial, so a specific op (an advanced-index gather, a device-side `randperm`, etc.) can be
  missing; if you hit that, try a smaller config or fall back to `--device cpu`. Expect it slower
  than native ROCm/CUDA but still a real win over CPU on the larger nets.
- **ZLUDA** (a CUDA-on-AMD shim) can sometimes run the `cuda` path on Radeons, but it's unofficial
  and its status is uncertain — not recommended.

> The environment here ships **CPU-only** torch (`2.12.0+cpu`), so no GPU is usable until you install
> a GPU build *into that interpreter*: a CUDA wheel (NVIDIA), a ROCm wheel (AMD/Linux), or
> `torch-directml` (AMD/Intel on Windows). The script needs no change — `--device auto`/`cuda`/`dml`
> picks it up. `--device auto` on a CPU-only install simply runs on the CPU (still resident + big-batch,
> just not fast).

## Library API

Everything the CLI does is importable:

```python
import numpy as np, torch, nnedi3_torch as N

# --- load & inspect weights ---
w = N.NNEDI3Weights.read("nnedi3_weights.bin")
d = w.predictors[("abs", 3, 4)]          # (etype, nsize=48x6, nns_idx=4 -> 256): softmax/elliott q1/q2

# --- exact forward pass (bit-identical to znedi3) ---
net = N.load_net(w, nsize=3, nns_idx=4, etype="abs", quality=1)   # NNEDI3Net (float64)
pred = net(torch.as_tensor(windows, dtype=torch.float64))         # windows: [N, xdim*ydim] row-major
# qual=2:
n1, n2 = N.load_net(w,3,4,"abs",1), N.load_net(w,3,4,"abs",2)
pred2 = N.predict_qual2(n1, n2, x)

# --- build training data from a progressive frame (znedi3 geometry, both parities) ---
Wp, Tp = N.make_training_pairs(frame2d, nsize=3, parity=0, stride=2)     # windows, targets
frames = N.load_gray_frames("corpus/*.png")
X, Y   = N.build_corpus(frames, nsize=3, stride=2, limit=3_000_000, seed=0)

# --- train one network (run twice w/ different seeds for the qual=2 pair) ---
net = N.NNEDI3Net(nsize=3, nns_idx=4)
N.train(net, X, Y, etype="abs", epochs=20, batch_size=1, lr=1e-3, device="cpu", seed=0)

# --- serialize into a znedi3-loadable .bin (untouched configs copied from base) ---
N.write_weights("my_weights.bin", base_bin="nnedi3_weights.bin",
                nets={("abs", 3, 4, 1): net})     # key: (etype, nsize, nns_idx, quality)
```

Index conventions: `nsize` `0..6` → `XDIM/YDIM` tables; `nns_idx` `0..4` → `NNS=[16,32,64,128,256]`;
`etype` `"abs"|"mse"`; `quality` `1|2`.

## Caveats

- **Precision vs the real znedi3 binary.** The ~1e-14 match is against znedi3's *scalar*
  algorithm in float64. The shipping SIMD paths use float32 and an approximate `exp`, so a live
  `vsznedi3` comparison agrees to ~1e-4, not bit-exact — expected, not a bug.
- **RAM / scale.** Patterns are held in memory bounded by `--limit`; the full ~60M-pattern
  flagship run would want streaming. Set `--limit`/`--stride` to fit your machine, or train the
  cheaper configs first. The pieces to stream (`build_corpus`, per-image `make_training_pairs`)
  are in the library.
- **Geometry.** `make_training_pairs` matches znedi3's inference geometry for both field
  parities (empirically confirmed against the shipped weights) — trained weights need no vertical
  shift.
