"""Byte comparisons and interleaved CUDA graph timings. Run with the cubins built by build.sh."""

import argparse
import ctypes as ct
from pathlib import Path
from statistics import median

import numpy as np

I, L, P, F = ct.c_int, ct.c_int64, ct.c_void_p, ct.c_float
ref = ct.byref
cuda = ct.CDLL("libcuda.so.1")
stream = P()
modules, kernels, allocations = {}, {}, []
rng = np.random.default_rng(197)
eps = F(1e-20)
kit = {
    "norm128": ("0cca97", "_rmsnorm", 16),
    "norm512": ("ee9e02", "_rmsnorm", 16),
    "rope128": ("e8bd0a", "_rope_heads", 0),
    "rope512": ("a22128", "_rope_heads", 0),
    "rope32": ("5ed6b4", "_rope_heads", 0),
    "store128": ("345438", "_fp4_store", 512),
    "store512": ("9cd357", "_fp4_store", 2048),
    "qd": ("4cbef7", "_fp4qd_p2", 0),
}


def cu(name, *args):
    err = getattr(cuda, name)(*args)
    if err:
        msg = ct.c_char_p()
        cuda.cuGetErrorString(err, ref(msg))
        raise RuntimeError(f"{name}: {err} {msg.value}")


class Buffer:
    def __init__(self, data):
        data = np.ascontiguousarray(data)
        self.shape, self.dtype = data.shape, data.dtype
        self.addr = P()
        cu("cuMemAlloc_v2", ref(self.addr), ct.c_size_t(data.nbytes))
        cu("cuMemcpyHtoD_v2", self.addr, P(data.ctypes.data), ct.c_size_t(data.nbytes))
        allocations.append(self.addr)

    def read(self):
        host = np.empty(self.shape, self.dtype)
        cu("cuMemcpyDtoH_v2", P(host.ctypes.data), self.addr, ct.c_size_t(host.nbytes))
        return host.view("u1").ravel()


def out(shape, dtype):
    return Buffer(np.full(shape, 0x55, dtype))


def launch(module, name, grid, threads, shared, *args):
    if module not in modules:
        mod = P()
        cu("cuModuleLoad", ref(mod), str(module).encode())
        modules[module] = mod
    if (module, name) not in kernels:
        fn = P()
        cu("cuModuleGetFunction", ref(fn), modules[module], name.encode())
        kernels[module, name] = fn
    pack = [a.addr if isinstance(a, Buffer) else I(a) if isinstance(a, int) else a for a in args]
    argv = (P * len(pack))(*(ct.addressof(x) for x in pack))
    gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
    bx, by = threads if isinstance(threads, tuple) else (threads, 1)
    cu("cuLaunchKernel", kernels[module, name], gx, gy, 1, bx, by, 1, shared, stream, argv, None)


def aot(key, grid, *args):
    prefix, name, smem = kit[key]
    (path,) = opts.kit.glob(prefix + "*.cubin")
    launch(path, name, grid, 128, smem, *args, P(), P())


def native(module, name, grid, block, *args, shared=0):
    launch(module, "tf_ds_" + name + "_kernel", grid, block, shared, *args, 0)


def compare(label, actual, expected, allow=False):
    cu("cuStreamSynchronize", stream)
    total = 0
    for i, (buf, gold) in enumerate(zip(actual, expected, strict=True)):
        mismatch = np.flatnonzero(buf.read() != gold.read())
        if mismatch.size and not allow:
            raise AssertionError(
                f"{label} buffer {i}: {len(mismatch)} different bytes, first at {mismatch[0]}"
            )
        total += len(mismatch)
    return total


def time_chains(label, chains, names="reference, old, new"):
    batch = 32
    graphs = []
    for chain in chains:
        chain()
        cu("cuStreamSynchronize", stream)
        cu("cuStreamBeginCapture_v2", stream, 0)
        for _ in range(batch):
            chain()
        graph, exe = P(), P()
        cu("cuStreamEndCapture", stream, ref(graph))
        cu("cuGraphInstantiateWithFlags", ref(exe), graph, ct.c_uint64(0))
        cu("cuGraphDestroy", graph)
        cu("cuGraphLaunch", exe, stream)
        cu("cuStreamSynchronize", stream)
        graphs.append(exe)
    start, stop = P(), P()
    for ev in (start, stop):
        cu("cuEventCreate", ref(ev), 0)
    samples = [[] for _ in graphs]
    for rep in range(9):
        for offset in range(len(graphs)):
            i = (rep + offset) % len(graphs)
            cu("cuEventRecord", start, stream)
            cu("cuGraphLaunch", graphs[i], stream)
            cu("cuEventRecord", stop, stream)
            cu("cuEventSynchronize", stop)
            ms = F()
            cu("cuEventElapsedTime", ref(ms), start, stop)
            samples[i].append(ms.value * 1000 / batch)
    for graph in graphs:
        cu("cuGraphExecDestroy", graph)
    for ev in (start, stop):
        cu("cuEventDestroy_v2", ev)
    print(label, f"graph us [{names}]", [round(median(s), 3) for s in samples], flush=True)


def bf(shape):
    return (rng.normal(size=shape).astype("f4").view("u4") >> 16).astype("u2")


def norms(d):
    rows = 4096
    host_x = bf((rows, d))
    x, w = Buffer(host_x), Buffer(bf(d))
    ys = [out((rows, d), "u2") for _ in range(3)]
    aot(f"norm{d}", rows, x, d, w, ys[0], d, eps)
    for mod, dst in zip(torch_modules[1:], ys[1:]):
        launch(mod, "audit_norm", rows, 128, 0, x, w, dst, d, eps)
    old = compare("old norm", ys[1:2], ys[:1], True)
    compare("repaired norm", ys[2:], ys[:1])
    witness = np.flatnonzero(ys[1].read() != ys[0].read())
    launch(torch_modules[1], "audit_root", rows, 128, 0, x, w, ys[1], d, eps)
    root = compare("root-only diagnostic", ys[1:2], ys[:1], True)
    print(f"norm D={d} R={rows}: old={old} differing bytes, root-only={root}, new=0", flush=True)
    if not old:
        raise AssertionError("seed did not expose the old defect")

    # Replay a differing norm row through the actual decode chain. Two equal partners with zero logits
    # produce exactly that bf16 row at the compressor boundary, independent of the norm implementation.
    row = host_x[witness[0] // (2 * d)]
    kv = Buffer((row.astype("u4") << 16).view("f4"))
    gate = Buffer(np.zeros(d, "f4"))
    slot, gi = Buffer(np.array([0], "i8")), Buffer(np.array([0, 0], "i8"))
    for i, mod in enumerate(torch_modules):
        rk, rg = out(d, "f4"), out(d, "f4")
        prefix = (kv, L(d), gate, L(d), slot, rk, rg, gi, L(1))
        if i:
            native(mod, "compress2x_norm", 1, d if i == 2 else 128, *prefix, w, ys[i], L(d), 1, d, eps, P())
        else:
            mid = out(d, "u2")
            native(mod, "compress2x", 1, 256, *prefix, mid, L(d), 1, d, P())
            aot(f"norm{d}", 1, mid, d, w, ys[i], d, eps)
    cu("cuStreamSynchronize", stream)
    a, b, c = (y.read()[: 2 * d] for y in ys)
    if np.array_equal(a, b) or not np.array_equal(a, c):
        raise AssertionError("decode compressor witness did not reproduce and resolve")
    print(
        f"decode compressor witness D={d}: old={np.count_nonzero(a != b)} differing bytes, new=0", flush=True
    )


def compressor(rows, d, prompt):
    width = d if prompt else d + 4
    count = 2 * rows
    x, s = [Buffer(rng.normal(size=(count, width)).astype("f4")) for _ in range(2)]
    w = Buffer(bf(d))
    slot = Buffer(rng.permutation(count)[:rows].astype("i8"))
    gi = rng.integers(count, size=(rows, 2), dtype="i8")
    pair = Buffer(gi)
    initial = [rng.normal(size=(count, d)).astype("f4") for _ in range(2)]
    chains, states = [], []
    for i, mod in enumerate(torch_modules):
        rk, rg = map(Buffer, initial)
        invalid = Buffer(np.zeros(1, "u4"))
        mid, dst = out((rows, d), "u2"), out((rows, d + 16), "u2")
        lanes = min(d // 4, 512 // min(1 << (rows.bit_length() - 1), 16))
        height = min(512 // lanes, 64)
        factor = F(np.float32(rows) / np.float32(rows * d))
        prefix = (x, L(width), s, L(width), slot, rk, rg, pair, L(count))

        def chain(i=i, mod=mod, mid=mid, dst=dst, invalid=invalid, prefix=prefix):
            if prompt:
                if i:
                    target = 512 if rows <= 32 else 256 if rows <= 128 else 128
                    block = (lanes, max(1, min(d, target) // lanes)) if i == 2 else lanes
                    native(mod, "compress2_norm", rows, block, x, s, w, dst, L(d + 16), rows, d, factor, eps)
                else:
                    native(mod, "compress2", (rows * d + 255) // 256, 256, x, s, mid, rows, d)
                    native(
                        mod,
                        "rms_norm",
                        (rows + height - 1) // height,
                        (lanes, height),
                        mid,
                        L(d),
                        w,
                        dst,
                        L(d + 16),
                        rows,
                        d,
                        factor,
                        eps,
                        shared=lanes * height * 4 if lanes > 32 else 0,
                    )
            elif i:
                native(mod, "compress2x_norm", rows, d if i == 2 else 128, *prefix, w, dst, L(d + 16), rows, d, eps, invalid)
            else:
                native(mod, "compress2x", rows, 256, *prefix, mid, L(d), rows, d, invalid)
                aot(f"norm{d}", rows, mid, d, w, dst, d + 16, eps)

        chain()
        chains.append(chain)
        states.append([dst, rk, rg, invalid])
    compare("compressor", states[2], states[0])
    if prompt:
        time_chains(f"compress prompt R={rows} D={d}", chains)
        return
    rk, rg = map(Buffer, initial)
    gathered = [out((2 * rows, d), "f4") for _ in range(2)]
    mid, dst = out((rows, d), "u2"), out((rows, d + 16), "u2")
    flag = Buffer(np.zeros(1, "u4"))

    def default_chain():
        for source, ring in ((x, rk), (s, rg)):
            native(
                ops_modules[0],
                "scatter_rows",
                rows,
                256,
                source,
                L(width * 4),
                slot,
                ring,
                L(d * 4),
                L(d * 4),
                rows,
            )
        cu("cuMemsetD8Async", flag.addr, 0, L(4), stream)
        for ring, temp in zip((rk, rg), gathered):
            launch(
                opts.binaries / "movement.cubin",
                "tf_gather_rows_kernel",
                2 * rows,
                256,
                0,
                ring,
                temp,
                pair,
                L(2 * rows),
                L(count),
                L(d * 4),
                flag,
                0,
            )
        native(torch_modules[0], "compress2", (rows * d + 255) // 256, 256, *gathered, mid, rows, d)
        aot(f"norm{d}", rows, mid, d, w, dst, d + 16, eps)

    default_chain()
    compare("production scatter/gather compressor", states[2], [dst, rk, rg, flag])
    time_chains(f"compress decode R={rows} D={d}", [default_chain, *chains], "unfused, 0127, old, new")


def selection(rows, nb, k, mask_on, distribution="ties"):
    width, ss, ms = 8 * nb - 3, 8 * nb, nb + 7
    scores = rng.integers(-10, 10, (rows, ss)).astype("f4")
    if distribution == "random":
        scores = rng.normal(size=(rows, ss)).astype("f4")
    elif distribution == "flat":
        scores.fill(-np.inf)
    scores[:, 8:16], scores[:, 24:32], scores[:, 32:40], scores[:, 40:48] = np.nan, -np.inf, -0.0, np.inf
    x = Buffer(scores)
    vis = Buffer(np.linspace(0, width, rows, dtype="i8"))
    every = Buffer(np.full(rows, 1 << 40, dtype="i8"))
    chains, states = [], []
    for i, mod in enumerate(ops_modules):
        mx, ix, mask, blocks = (
            out((rows, nb), "u4"),
            out((rows, k), "i8"),
            out((rows, ms), "u1"),
            out((rows, k), "i4"),
        )
        m, c = (mask, P()) if mask_on else (P(), blocks)

        def chain(i=i, mod=mod, mx=mx, ix=ix, m=m, c=c):
            if i:
                native(
                    mod,
                    "idx_fused",
                    rows,
                    1024,
                    x,
                    L(ss),
                    width,
                    8,
                    vis,
                    every,
                    mx,
                    L(nb),
                    ix,
                    k,
                    m,
                    L(ms),
                    c,
                    shared=nb * 4 if i == 2 else 0,
                )
            else:
                native(
                    mod, "block_max", ((nb + 255) // 256, rows), 256, x, L(ss), width, 8, vis, mx, L(nb), nb
                )
                native(mod, "topk_indices", rows, 1024, mx, L(nb), nb, k, every, ix)
                native(mod, "pool_pick", rows, 1024, mx, L(nb), nb, ix, k, m, L(ms), c)

        chain()
        chains.append(chain)
        states.append([mx, ix, mask, blocks])
    for state in states[1:]:
        compare("index pool", state, states[0])
    time_chains(f"pool R={rows} nb={nb} k={k} mask={mask_on} {distribution}", chains)


def rotary(rows, path, special=False):
    key = path.endswith("key")
    prompt = path.startswith("prompt")
    query = path == "query"
    heads = 32 if query else 1
    d = 128 if key or query else 512
    src = bf((rows, heads, d))
    if special:
        src[..., :8] = [0x7FC1, 0x7F80, 0xFF80, 0x8000, 0, 1, 0x7F7F, 0x8001]
        src[..., 32:64] = 0x7FC1
    weights = Buffer(bf(d))
    pos = Buffer(rng.permutation(rows).astype("i8"))
    dest = Buffer(rng.permutation(rows).astype("i8"))
    angle = rng.uniform(-np.pi, np.pi, (rows, 32))
    cos, sin = Buffer(np.cos(angle).astype("f4")), Buffer(np.sin(angle).astype("f4"))
    chains, states = [], []
    for i, mod in enumerate(torch_modules):
        x, rotated, dequant = Buffer(src), out(src.shape, "u2"), out(src.shape, "u2")
        codes, scales = out((rows, d // 2), "u1"), out((rows, d // (32 if key else 16)), "u1")

        def chain(i=i, mod=mod, x=x, rot=rotated, dq=dequant, codes=codes, scales=scales):
            if path == "prompt latent" and i < 2:
                native(ops_modules[0], "copy_rows16", (1, rows), 256, x, L(d // 8), rot, L(d // 8), L(d // 8))
            if i:
                if key and prompt:
                    native(
                        mod,
                        "fused_prompt_idx_key",
                        rows,
                        32,
                        x,
                        L(d),
                        weights,
                        eps,
                        F(1 / d),
                        cos,
                        sin,
                        pos,
                        rot,
                        codes,
                        scales,
                        dest,
                        d,
                        64,
                    )
                elif key:
                    native(
                        mod,
                        "fused_idx_key",
                        rows,
                        128,
                        x,
                        L(d),
                        weights,
                        cos,
                        sin,
                        pos,
                        rot,
                        codes,
                        scales,
                        dest,
                        d,
                        64,
                        eps,
                    )
                elif query:
                    native(mod, "fused_rope_qd", rows * heads, 128, x, dq, cos, sin, pos, heads, d, 64)
                elif prompt:
                    src_args = (x, rot) if i == 2 else (rot,)
                    native(
                        mod,
                        "fused_prompt_latent",
                        rows,
                        256 if i == 2 else 128,
                        *src_args,
                        cos,
                        sin,
                        pos,
                        codes,
                        scales,
                        dest,
                        d,
                        64,
                    )
                else:
                    native(
                        mod,
                        "fused_rope_store",
                        rows,
                        256 if i == 2 else 128,
                        x,
                        cos,
                        sin,
                        pos,
                        codes,
                        scales,
                        dest,
                        d,
                        64,
                    )
                return
            if key:
                if prompt:
                    native(
                        mod,
                        "rms_norm",
                        (rows + 15) // 16,
                        (32, 16),
                        x,
                        L(d),
                        weights,
                        rot,
                        L(d),
                        rows,
                        d,
                        F(1 / d),
                        eps,
                    )
                else:
                    aot("norm128", rows, x, d, weights, rot, d, eps)
            target = rot if key or prompt else x
            if prompt:
                native(
                    mod,
                    "rope",
                    (rows * 32 + 255) // 256,
                    256,
                    target,
                    L(d),
                    d - 64,
                    cos,
                    sin,
                    pos,
                    rows,
                    32,
                    0,
                )
            else:
                aot("rope32" if query else f"rope{d}", (rows, 2 if query else 1), target, cos, sin, pos)
            if query:
                nblk = rows * heads * d // 32
                aot("qd", (nblk + 31) // 32, target, dq, nblk)
            else:
                aot(f"store{d}", rows, target, d, codes, scales, dest)

        chain()
        chains.append(chain)
        states.append([x, rotated, dequant, codes, scales])
    compare(path, states[2], states[0])
    if special:
        print(f"{path} nonfinite/denormal/zero bytes match", flush=True)
    else:
        time_chains(f"rope {path} R={rows} D={d}", chains)


def latent_values():
    d = 512
    values = np.arange(65536, dtype="u2")
    blocks = np.repeat(values[:, None], 16, axis=1)
    blocks[:, 1::2] = rng.permutation(values)[:, None]
    source = blocks.reshape(-1, d)
    rows = len(source)
    x, rot = Buffer(source), Buffer(source)
    cos = Buffer(np.ones((1, 32), "f4"))
    sin = Buffer(np.zeros((1, 32), "f4"))
    pos = Buffer(np.zeros(rows, "i8"))
    dst = Buffer(np.arange(rows, dtype="i8"))
    codes, scales = out((rows, d // 2), "u1"), out((rows, d // 16), "u1")
    expected = [out((rows, d // 2), "u1"), out((rows, d // 16), "u1")]
    aot("rope512", rows, x, cos, sin, pos)
    aot("store512", rows, x, d, *expected, dst)
    native(torch_modules[2], "fused_rope_store", rows, 256, rot, cos, sin, pos, codes, scales, dst, d, 64)
    compare("all bf16 encodings through packed latent", [rot, codes, scales], [x, *expected])
    print("all 65,536 bf16 input encodings through the latent store match the kit", flush=True)


def norm_extremes(d):
    # Exercise both ends of the reciprocal's normal range, and rows of infinities/NaNs.
    samples = np.arange(65536, dtype="u2")
    host = np.broadcast_to(samples[:, None], (len(samples), d))
    w = Buffer(np.full(d, 0x3F80, "u2"))
    for first in range(0, len(samples), 2048):
        chunk = host[first : first + 2048].copy()
        chunk[:, ::2] ^= 0x8000
        rows = len(chunk)
        x, y, z = Buffer(chunk), out(chunk.shape, "u2"), out(chunk.shape, "u2")
        # Deterministic model epsilon. The final store is compared bitwise including NaN encodings.
        aot(f"norm{d}", rows, x, d, w, y, d, eps)
        launch(torch_modules[2], "audit_norm", rows, 128, 0, x, w, z, d, eps)
        compare("norm extreme exponents", [z], [y])
        for addr in allocations[-3:]:
            cu("cuMemFree_v2", addr)
        del allocations[-3:]
    print(f"norm D={d}: all bf16 exponents and mantissas in constant-magnitude rows match", flush=True)


def scale_midpoints():
    # Independent exhaustive check of the latent-store division elimination. Every possible bf16
    # input against all positive, finite E4M3 scale bytes emitted by the kit.
    x = (np.arange(65536, dtype="u4") << 16).view("f4")
    midpoint = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype="f4")
    odd = [False, True, False, True, False, True, False]

    def magnitude(a, thresholds):
        return sum((a >= t if up else a > t).astype("u1") for t, up in zip(thresholds, odd))

    with np.errstate(over="ignore", invalid="ignore"):
        for byte in range(1, 127):
            e, m = divmod(byte, 8)
            sc = np.float32(m * 2**-9 if not e else (1 + m / 8) * 2 ** (e - 7))
            t = np.fmin(np.fmax(x / sc, -6), 6)
            expected = magnitude(np.abs(t), midpoint)
            expected |= np.where((t < 0) & (expected != 0), 8, 0).astype("u1")
            actual = magnitude(np.abs(x), midpoint * sc)
            actual |= np.where((x < 0) & (actual != 0), 8, 0).astype("u1")
            actual[np.isnan(x)] = 15
            np.testing.assert_array_equal(actual, expected)
    print("8,257,536 bf16/E4M3 input-scale pairs match the RN division and tie rules", flush=True)


def run(fn, *args):
    try:
        fn(*args)
    finally:
        for addr in allocations:
            cu("cuMemFree_v2", addr)
        allocations.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kit", type=Path, help="pinned kit/aot/cubins")
    parser.add_argument("binaries", type=Path, help="directory containing new/old .cubin files")
    opts = parser.parse_args()
    torch_modules = [opts.binaries / f"{v}-torch.cubin" for v in ("old", "old", "new")]
    ops_modules = [opts.binaries / f"{v}-ops.cubin" for v in ("old", "old", "new")]
    cu("cuInit", 0)
    ctx = P()
    cu("cuDevicePrimaryCtxRetain", ref(ctx), 0)
    cu("cuCtxSetCurrent", ctx)
    cu("cuStreamCreate", ref(stream), 1)
    for dim in (128, 512):
        run(norms, dim)
    for r in (1, 7, 32):
        run(compressor, r, 512, False)
    for r, d in ((1, 512), (8, 512), (128, 512), (1024, 512), (1, 2048)):
        run(compressor, r, d, True)
    for rows, nb, k, mask in [
        (7, 8, 8, False),
        (7, 2049, 2048, False),
        (7, 4096, 2048, False),
        (32, 8192, 2048, False),
        (128, 4096, 2048, True),
    ]:
        run(selection, rows, nb, k, mask)
    run(selection, 7, 4096, 2048, False, "random")
    run(selection, 7, 8192, 2048, True, "flat")
    run(selection, 3, 513, 17, False, "random")
    for path in ("key", "latent", "query", "prompt key", "prompt latent"):
        run(rotary, 7 if not path.startswith("prompt") else 128, path)
        run(rotary, 7, path, True)
    run(latent_values)
    for dim in (128, 512):
        run(norm_extremes, dim)
    scale_midpoints()
    print("PASS", flush=True)
