"""nnedi3_weights_hvsym.bin  --  h+v-symmetric, (N-1)xM generator weight file.

Layout (mirrors the original nnedi3 predictor grouping, but stores only the nns/4 flip-orbit
GENERATORS at the target-centred odd width N-1, and NO prescreener -- the prescreener is still
loaded from the original nnedi3_weights.bin):

    bytes 0..7   : magic  b"HVSYMB1\0"
    then float32, little-endian, in nested order
        for etype in (abs, mse):                      # 2
          for nns_i in 0..4:      NNS = 16,32,64,128,256
            for size_i in 0..6:   XDIM = 8,16,32,48,8,16,32   YDIM = 6,6,6,6,4,4,4
              ng   = NNS[nns_i] // 4
              Dm1  = (XDIM[size_i] - 1) * YDIM[size_i]        # (N-1) x M window
              for q in (1, 2):
                softmax_gen[ng, Dm1]   # generator = the identity-flip representative, row-major [M][N-1]
                elliott_gen[ng, Dm1]
                softmax_bias[ng]       # one bias per orbit (shared by all 4 flips)
                elliott_bias[ng]

The C++ loader folds each generator into 4 Klein-4 channels at load time. Materialising the full
nns experts is: expert[f*ng+g] = flip_f(generator[g]) with the shared bias, f in {e,h,v,hv}.
"""
import numpy as np

XDIM = [8, 16, 32, 48, 8, 16, 32]
YDIM = [6, 6, 6, 6, 4, 4, 4]
NNS  = [16, 32, 64, 128, 256]
MAGIC = b"HVSYMB1\0"


def write_hvsym(path, gen_fn):
    """gen_fn(etype, nns_i, size_i, qual, ng, Dm1) -> (softmax_gen[ng,Dm1], elliott_gen[ng,Dm1],
    softmax_bias[ng], elliott_bias[ng]).  All returned as float arrays."""
    parts = []
    for et in range(2):
        for ni in range(5):
            for si in range(7):
                ng = NNS[ni] // 4
                Dm1 = (XDIM[si] - 1) * YDIM[si]
                for q in (1, 2):
                    sg, eg, sb, eb = gen_fn(et, ni, si, q, ng, Dm1)
                    parts += [np.asarray(sg, np.float32).reshape(-1), np.asarray(eg, np.float32).reshape(-1),
                              np.asarray(sb, np.float32).reshape(-1), np.asarray(eb, np.float32).reshape(-1)]
    with open(path, "wb") as f:
        f.write(MAGIC)
        np.concatenate(parts).astype("<f4").tofile(f)
    return 8 + sum(p.size for p in parts) * 4


def read_hvsym(path):
    """Read nnedi3_weights_hvsym.bin -> GEN[(etype_i, nns_i, size_i, qual)] = (sg[ng,Dm1], eg[ng,Dm1], sb[ng], eb[ng])."""
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError("bad magic")
        data = np.fromfile(f, dtype="<f4")
    GEN = {}; off = 0
    def take(n):
        nonlocal off; a = data[off:off+n]; off += n; return a
    for et in range(2):
        for ni in range(5):
            for si in range(7):
                ng = NNS[ni] // 4; Dm1 = (XDIM[si] - 1) * YDIM[si]
                for q in (1, 2):
                    sg = take(ng*Dm1).reshape(ng, Dm1); eg = take(ng*Dm1).reshape(ng, Dm1)
                    sb = take(ng); eb = take(ng)
                    GEN[(et, ni, si, q)] = (sg.copy(), eg.copy(), sb.copy(), eb.copy())
    assert off == data.size, (off, data.size)
    return GEN


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else r"E:\train2017\hvsym_test.bin"
    rng = np.random.default_rng(0)
    def rand_gen(et, ni, si, q, ng, Dm1):
        return (rng.standard_normal((ng, Dm1)).astype("f4") * 0.1,
                rng.standard_normal((ng, Dm1)).astype("f4") * 0.1,
                rng.standard_normal(ng).astype("f4") * 0.1,
                rng.standard_normal(ng).astype("f4") * 0.1)
    n = write_hvsym(out, rand_gen)
    print(f"wrote {out} ({n} bytes)")
