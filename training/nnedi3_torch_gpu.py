"""
nnedi3_torch_gpu.py
===================

GPU-accelerated trainer for the nnedi3 predictor network -- a drop-in alternative
to `nnedi3_torch.py train`.

It reuses the *validated* pieces of nnedi3_torch unchanged (the NNEDI3Net forward
pass, the znedi3 window geometry, corpus building, and the byte-valid .bin
serialisation), so anything trained here still drops straight into znedi3. Only
the training loop is new.

Why a separate loop? nnedi3_torch.train() was written for CPU online GD: it keeps
the corpus in host RAM and, every step, fancy-indexes on the CPU and copies the
batch to the device. With batch=1 (tritical's online GD) that is millions of tiny
host->device transfers, so `--device cuda` there is actually *slower* than CPU.

This trainer instead:

  * Keeps the whole corpus RESIDENT ON THE DEVICE and does all indexing/shuffling
    on-device -- zero per-batch host<->device traffic.
  * Packs the corpus to uint8 by default (4x less VRAM; lossless for 8-bit-sourced
    frames, which is what load_gray_frames produces) and converts to float per
    batch on the device. A 24 GB GPU then holds ~80M x 288 patterns -- the full
    "flagship" scale -- entirely in VRAM.
  * Uses mini-batch SGD (large batch). This is REQUIRED for GPU utilisation and is
    where the speed-up comes from; it is a standard, well-behaved deviation from
    strict batch=1 online GD. Lower --batch to move back toward online behaviour
    (at the cost of speed).
  * Auto-selects the device (cuda -> mps -> cpu) and reports throughput.

Two entry points:
    train         build a corpus from images/frames (bounded by --limit, held in
                  RAM) and train it resident on the GPU. The direct GPU analogue
                  of `nnedi3_torch.py train`.
    train-memmap  stream the uint8 memmap produced by train_streaming.py's
                  extraction phase to the GPU in super-batches, for corpora too
                  big to hold in RAM even as uint8 (only extraction touches disk).

Everything else (loss/optimizer/plateau schedule, qual=2 two-seed ensemble) matches
nnedi3_torch. Requires: numpy, torch (a CUDA build to actually use a GPU).
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # find sibling nnedi3_torch.py
import nnedi3_torch as N   # validated core -- do not fork the math, import it


# --------------------------------------------------------------------------------------
# Device selection
# --------------------------------------------------------------------------------------
def pick_device(spec: str = "auto"):
    """cuda (NVIDIA, or AMD ROCm which also reports as 'cuda') | mps (Apple) |
    dml (DirectML: AMD/Intel on Windows, needs `pip install torch-directml`) | cpu | auto."""
    spec = (spec or "auto").lower()
    if spec in ("dml", "directml"):
        import torch_directml                       # Windows AMD/Intel via DirectX 12
        return torch_directml.device()
    if spec == "auto":
        if torch.cuda.is_available():               # NVIDIA CUDA *or* AMD ROCm
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def _pack(windows, targets, device, as_uint8):
    """Move the corpus resident to `device`. uint8 packing is lossless for 8-bit sources
    (windows/targets are floats in [0,1] that came from uint8/255); it quantises otherwise."""
    if as_uint8:
        Xu = torch.from_numpy(np.rint(np.clip(windows, 0.0, 1.0) * 255.0).astype(np.uint8))
        Yu = torch.from_numpy(np.rint(np.clip(targets, 0.0, 1.0) * 255.0).astype(np.uint8))
        return Xu.to(device), Yu.to(device), True
    Xf = torch.as_tensor(np.ascontiguousarray(windows), dtype=torch.float32)
    Yf = torch.as_tensor(np.ascontiguousarray(targets), dtype=torch.float32)
    return Xf.to(device), Yf.to(device), False


# --------------------------------------------------------------------------------------
# Resident-on-device training (corpus fits in VRAM, packed as uint8 by default)
# --------------------------------------------------------------------------------------
def train_gpu(net, windows: np.ndarray, targets: np.ndarray, *,
              etype: str = "abs", epochs: int = 20, batch_size: int = 8192,
              lr: float = 1e-2, momentum: float = 0.9, device="auto",
              val_frac: float = 0.02, seed: int = 0, corpus_uint8: bool = True,
              verbose: bool = True):
    """Train ONE network with the corpus resident on `device`. Returns the trained net on CPU.
    Run twice with different seeds for the qual=2 ensemble; run per etype (abs->L1, mse->MSE)."""
    dev = pick_device(device) if isinstance(device, str) else device
    torch.manual_seed(seed)
    net = net.to(dev).float()
    g = torch.Generator().manual_seed(seed)                 # CPU generator -> deterministic splits

    n_all = len(windows)
    X, Y, is_u8 = _pack(windows, targets, dev, corpus_uint8)
    to_f = (lambda t: t.float().div_(255.0)) if is_u8 else (lambda t: t)

    nval = max(1, int(n_all * val_frac))
    perm = torch.randperm(n_all, generator=g)
    vidx = perm[:nval].to(dev)
    tidx = perm[nval:].to(dev)
    n_tr = tidx.numel()

    loss_fn = torch.nn.L1Loss() if etype == "abs" else torch.nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)

    @torch.no_grad()
    def evaluate():
        net.eval()
        tot, cnt = 0.0, 0
        for i in range(0, vidx.numel(), 65536):
            b = vidx[i:i + 65536]
            tot += (net(to_f(X[b])) - to_f(Y[b])).abs().sum().item()
            cnt += b.numel()
        return tot / max(cnt, 1)

    if verbose:
        gib = (X.element_size() * X.nelement() + Y.element_size() * Y.nelement()) / 2**30
        print(f"device={dev}  corpus={'uint8' if is_u8 else 'float32'} {gib:.2f} GiB resident  "
              f"train={n_tr:,} val={nval:,}  batch={batch_size}", flush=True)

    for ep in range(epochs):
        net.train()
        t0 = time.time()
        order = tidx[torch.randperm(n_tr, generator=g).to(dev)]
        for i in range(0, n_tr - batch_size + 1, batch_size):
            b = order[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(to_f(X[b])), to_f(Y[b]))
            loss.backward()
            opt.step()
        vl = evaluate()
        sched.step(vl)
        if verbose:
            dt = time.time() - t0
            print(f"epoch {ep:3d}  val L1={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{n_tr / dt / 1e6:.1f}M pat/s  ({dt:.1f}s)", flush=True)
    return net.to("cpu").float()


# --------------------------------------------------------------------------------------
# Streaming from a uint8 memmap (flagship scale; extraction done by train_streaming.py)
# --------------------------------------------------------------------------------------
def train_memmap(net, work_dir: str, D: int, *, etype: str = "abs", epochs: int = 20,
                 batch_size: int = 8192, super_batch: int = 2_000_000, lr: float = 1e-2,
                 momentum: float = 0.9, device="auto", val_n: int = 200_000, seed: int = 0,
                 verbose: bool = True):
    """Train from win.u8 [cap,D] / tgt.u8 [cap] / meta.txt (Ntot) as written by train_streaming.py.
    Super-batches are shipped to the device once and mini-batched on-device (the only host traffic)."""
    dev = pick_device(device) if isinstance(device, str) else device
    torch.manual_seed(seed)
    net = net.to(dev).float()
    rng = np.random.default_rng(seed)

    win_path = os.path.join(work_dir, "win.u8")
    tgt_path = os.path.join(work_dir, "tgt.u8")
    ntot = int(open(os.path.join(work_dir, "meta.txt")).read().strip())
    cap = os.path.getsize(win_path) // D                    # cap not stored; recover from file size
    mw = np.memmap(win_path, dtype=np.uint8, mode="r", shape=(cap, D))
    mt = np.memmap(tgt_path, dtype=np.uint8, mode="r", shape=(cap,))
    val_n = min(val_n, ntot // 5)
    n_tr = ntot - val_n
    # fixed held-out tail on the device
    Xv = torch.from_numpy(np.asarray(mw[n_tr:ntot])).to(dev)
    Yv = torch.from_numpy(np.asarray(mt[n_tr:ntot])).to(dev)

    loss_fn = torch.nn.L1Loss() if etype == "abs" else torch.nn.MSELoss()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    steps = max(1, (n_tr // batch_size)) * epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.005)

    @torch.no_grad()
    def evaluate():
        net.eval()
        tot, cnt = 0.0, 0
        for i in range(0, Xv.shape[0], 65536):
            xb = Xv[i:i + 65536].float().div_(255.0)
            yb = Yv[i:i + 65536].float().div_(255.0)
            tot += (net(xb) - yb).abs().sum().item(); cnt += xb.shape[0]
        return tot / max(cnt, 1)

    if verbose:
        print(f"device={dev}  memmap {ntot:,} patterns (D={D})  train={n_tr:,} val={Xv.shape[0]:,}  "
              f"batch={batch_size} super={super_batch:,}", flush=True)

    for ep in range(epochs):
        net.train()
        t0 = time.time()
        starts = list(range(0, n_tr, super_batch))
        rng.shuffle(starts)
        for s in starts:
            e = min(s + super_batch, n_tr)
            Xu = torch.from_numpy(np.asarray(mw[s:e])).to(dev)     # one host->device copy per super-batch
            Yu = torch.from_numpy(np.asarray(mt[s:e])).to(dev)
            perm = torch.randperm(Xu.shape[0]).to(dev)         # CPU-gen -> portable across backends
            for i in range(0, Xu.shape[0] - batch_size + 1, batch_size):
                b = perm[i:i + batch_size]
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(net(Xu[b].float().div_(255.0)), Yu[b].float().div_(255.0))
                loss.backward()
                opt.step()
                sched.step()
        vl = evaluate()
        if verbose:
            dt = time.time() - t0
            print(f"epoch {ep:3d}  val L1={vl:.5f}  lr={opt.param_groups[0]['lr']:.2e}  ({dt:.1f}s)", flush=True)
    return net.to("cpu").float()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def _frames_from_args(args):
    if args.synthetic > 0:
        rng = np.random.default_rng(args.seed)
        print(f"corpus: {args.synthetic} synthetic smooth frames")
        return [N._smooth_field(rng, 256, 256, 1.0) for _ in range(args.synthetic)]
    if args.images:
        frames = N.load_gray_frames(args.images)
        print(f"corpus: {len(frames)} frames from {args.images!r}")
        return frames
    raise SystemExit("provide --images <glob> or --synthetic <N>")


def cli_train(args):
    nsize = N._resolve_nsize(args.nsize)
    nns_idx = N._resolve_nns(args.nns)
    etype = N._resolve_etype(args.etype)
    print(f"network: nsize={N.NSIZE_NAMES[nsize]} ({N.XDIM[nsize]}x{N.YDIM[nsize]}, "
          f"D={N.XDIM[nsize]*N.YDIM[nsize]}), nns={N.NNS[nns_idx]}, etype={etype}, "
          f"qual={args.qual}, symmetric={args.symmetric}")

    frames = _frames_from_args(args)
    W, T = N.build_corpus(frames, nsize, stride=args.stride, limit=args.limit, seed=args.seed)
    print(f"patterns: {len(W):,} of dim {W.shape[1]}  (~{W.nbytes/1e9:.2f} GB float32 in RAM)")

    nets = {}
    for quality in range(1, args.qual + 1):
        seed = args.seed + (quality - 1) * 100003
        print(f"\n=== training q{quality}  (seed {seed}, {etype} loss, batch {args.batch}) ===")
        net = N.NNEDI3Net(nsize, nns_idx, symmetric=args.symmetric)
        if quality == 1:
            tot = sum(p.numel() for p in net.parameters())
            print(f"   free params: {tot:,}{'  (flip-orbit tied)' if args.symmetric else ''}")
        net = train_gpu(net, W, T, etype=etype, epochs=args.epochs, batch_size=args.batch,
                        lr=args.lr, momentum=args.momentum, device=args.device, seed=seed,
                        corpus_uint8=not args.fp32_corpus)
        nets[(etype, nsize, nns_idx, quality)] = net
        if args.save_pt:
            path = args.save_pt if args.qual == 1 else args.save_pt.replace(".pt", f".q{quality}.pt")
            torch.save(net.state_dict(), path); print(f"saved {path}")

    if args.out:
        base = args.base or N._default_base_bin()
        if not base or not os.path.exists(base):
            raise SystemExit("--out needs --base <nnedi3_weights.bin>")
        N.write_weights(args.out, base, nets)
    else:
        print("\n(no --out; net kept in memory only. Pass --out + --base to write a .bin.)")


def cli_train_memmap(args):
    nsize = N._resolve_nsize(args.nsize)
    nns_idx = N._resolve_nns(args.nns)
    etype = N._resolve_etype(args.etype)
    D = N.XDIM[nsize] * N.YDIM[nsize]
    print(f"network: nsize={N.NSIZE_NAMES[nsize]} (D={D}), nns={N.NNS[nns_idx]}, etype={etype}, "
          f"qual={args.qual}, symmetric={args.symmetric}  <- memmap {args.work}")
    nets = {}
    for quality in range(1, args.qual + 1):
        seed = args.seed + (quality - 1) * 100003
        print(f"\n=== training q{quality}  (seed {seed}) ===")
        net = N.NNEDI3Net(nsize, nns_idx, symmetric=args.symmetric)
        net = train_memmap(net, args.work, D, etype=etype, epochs=args.epochs,
                           batch_size=args.batch, super_batch=args.super, lr=args.lr,
                           momentum=args.momentum, device=args.device, seed=seed)
        nets[(etype, nsize, nns_idx, quality)] = net
        if args.save_pt:
            path = args.save_pt if args.qual == 1 else args.save_pt.replace(".pt", f".q{quality}.pt")
            torch.save(net.state_dict(), path); print(f"saved {path}")
    if args.out:
        base = args.base or N._default_base_bin()
        if not base or not os.path.exists(base):
            raise SystemExit("--out needs --base <nnedi3_weights.bin>")
        N.write_weights(args.out, base, nets)


def cli_demo(args):
    dev = pick_device(args.device)
    print(f"demo smoke test on {dev}")
    rng = np.random.default_rng(0)
    frames = [N._smooth_field(rng, 192, 192, 1.0) for _ in range(4)]
    W, T = N.build_corpus(frames, 4, stride=2, seed=0)          # 8x4 / nns=16
    print(f"{len(W):,} patterns, D={W.shape[1]}")
    net = N.NNEDI3Net(4, 0)
    train_gpu(net, W, T, etype="abs", epochs=5, batch_size=4096, lr=1e-2, device=args.device)


def _add_common(p):
    g = p.add_argument_group("network selection (same knobs as nnedi3)")
    g.add_argument("--nsize", required=True, help="0..6 or WxH: 8x6 16x6 32x6 48x6 8x4 16x4 32x4")
    g.add_argument("--nns", required=True, help="0..4 or actual count 16/32/64/128/256")
    g.add_argument("--etype", default="abs", help="abs|mse (or 0|1)")
    g.add_argument("--qual", type=int, default=1, choices=(1, 2))
    g.add_argument("--symmetric", action="store_true", help="flip-orbit tying (nns must be /4)")
    o = p.add_argument_group("optimization")
    o.add_argument("--epochs", type=int, default=20)
    o.add_argument("--batch", type=int, default=8192, help="mini-batch (large = fast on GPU)")
    o.add_argument("--lr", type=float, default=1e-2)
    o.add_argument("--momentum", type=float, default=0.9)
    o.add_argument("--device", default="auto", help="auto | cuda (NVIDIA/AMD-ROCm) | mps | dml (Win AMD/Intel) | cpu")
    o.add_argument("--seed", type=int, default=0)
    w = p.add_argument_group("output")
    w.add_argument("--out", help="write a full .bin with the trained net(s) merged into --base")
    w.add_argument("--base", help="base .bin to merge into (default: nnedi3_weights.bin next to this script)")
    w.add_argument("--save-pt", dest="save_pt", help="also save raw trained net(s) as .pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="nnedi3 predictor: GPU-accelerated trainer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train", help="build a corpus (RAM, bounded by --limit) and train it resident on the GPU")
    _add_common(pt)
    d = pt.add_argument_group("training data")
    d.add_argument("--images", help='glob of progressive frames, e.g. "corpus/*.png"')
    d.add_argument("--synthetic", type=int, default=0, metavar="N", help="use N generated smooth frames")
    d.add_argument("--stride", type=int, default=2, help="pixel-position subsample step per frame")
    d.add_argument("--limit", type=int, default=4_000_000, help="cap on total patterns held in RAM")
    d.add_argument("--fp32-corpus", dest="fp32_corpus", action="store_true",
                   help="keep the resident corpus in float32 (default: uint8, 4x less VRAM, lossless for 8-bit)")

    pm = sub.add_parser("train-memmap", help="stream train_streaming.py's uint8 memmap to the GPU (flagship scale)")
    _add_common(pm)
    m = pm.add_argument_group("memmap source")
    m.add_argument("--work", required=True, help="dir with win.u8 / tgt.u8 / meta.txt (train_streaming WORK)")
    m.add_argument("--super", type=int, default=2_000_000, help="super-batch shipped to the device at once")

    pdemo = sub.add_parser("demo", help="tiny synthetic smoke test of the GPU training loop")
    pdemo.add_argument("--device", default="auto")

    args = ap.parse_args()
    if args.cmd == "train":
        cli_train(args)
    elif args.cmd == "train-memmap":
        cli_train_memmap(args)
    elif args.cmd == "demo":
        cli_demo(args)
