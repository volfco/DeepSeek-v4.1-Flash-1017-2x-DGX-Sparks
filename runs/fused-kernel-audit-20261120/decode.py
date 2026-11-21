"""Compare 0127 + pinned RMSNorm, the first repaired 0197, and fused candidates."""
import argparse
from pathlib import Path

import numpy as np
import probe as p


def check(rows, d, layout, timing, specs):
    n = rows * 2
    stride = d + int(layout == "unaligned")
    slots = p.rng.permutation(n)[:rows].astype("i8")
    if layout == "cached":
        pair = np.tile(np.setdiff1d(np.arange(n), slots), (2, 1)).T.copy()
    elif layout in ("cross", "invalid", "stress"):
        pair = np.column_stack((np.roll(slots, 1), np.roll(slots, -1)))
        if layout == "invalid":
            pair[::5, 1] = -1
            pair[::7, 0] = n
    else:
        pair = np.column_stack((slots, np.setdiff1d(np.arange(n), slots)))
        pair[1::2] = pair[1::2, ::-1]
    inputs = [p.rng.normal(size=(rows, stride)).astype("f4") for _ in range(2)]
    rings = [p.rng.normal(size=(n, d)).astype("f4") for _ in range(2)]
    if layout == "stress":
        # Extreme logits including subnormals, NaNs and both infinities.
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff], "u4")
        for a in (inputs[1], rings[1]):
            a[:, ::7] = np.resize(bits.view("f4"), a[:, ::7].shape)
    x, gate = map(p.Buffer, inputs)
    slot, gi = p.Buffer(slots), p.Buffer(pair)
    w = p.Buffer(p.bf(d))
    states, chains = [], []
    for label, mod, block in specs:
        rk, rg = map(p.Buffer, rings)
        flag = p.Buffer(np.zeros(1, "u4"))
        mid, dst = p.out((rows, d), "u2"), p.out((rows, d + 16), "u2")
        prefix = (x, p.L(stride), gate, p.L(stride), slot, rk, rg, gi, p.L(n))

        def chain(mod=mod, block=block, prefix=prefix, dst=dst, mid=mid, flag=flag):
            if block == 0:
                p.native(mod, "compress2x", rows, 256, *prefix, mid, p.L(d), rows, d, flag)
                p.aot(f"norm{d}", rows, mid, d, w, dst, d + 16, p.eps)
            else:
                p.native(mod, "compress2x_norm", rows, block, *prefix, w, dst, p.L(d + 16), rows, d, p.eps, flag)

        chain()
        chains.append(chain)
        states.append([dst, rk, rg, flag])
        if block:
            p.compare(label, states[-1], states[0])
    if timing:
        p.time_chains(f"{layout} R={rows} D={d}", chains, ", ".join(s[0] for s in specs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--previous", type=Path, help="optional cubin from the first repaired 0197")
    p.opts = parser.parse_args()
    p.cu("cuInit", 0)
    ctx = p.P()
    p.cu("cuDevicePrimaryCtxRetain", p.ref(ctx), 0)
    p.cu("cuCtxSetCurrent", ctx)
    p.cu("cuStreamCreate", p.ref(p.stream), 1)
    for d in (128, 512):
        specs = [("0127+kit", p.opts.previous or p.opts.candidate, 0)]
        if p.opts.previous:
            specs.append(("prior0197", p.opts.previous, 128))
        specs.append(("0197", p.opts.candidate, d))
        for rows in (1, 2, 7, 16, 32, 48):
            p.run(check, rows, d, "mixed", True, specs)
        for layout in ("unaligned", "cached", "cross", "invalid", "stress"):
            for rows in (1, 7, 33, 65):
                p.run(check, rows, d, layout, layout in ("cached", "cross") and rows == 7, specs)
    print("PASS", flush=True)
