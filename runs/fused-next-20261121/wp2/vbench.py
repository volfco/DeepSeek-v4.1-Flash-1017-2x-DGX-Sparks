#!/usr/bin/env python3
"""WP2 candidate bench for the 0197 decode compressor (tf_ds_compress2x_norm_kernel).

Arms: the engine's current fused kernel (bin/new-torch.cubin), the unfused kit chain it replaces (compress2x + the
pinned kit `_rmsnorm`), and any candidate kernels in bin/variants.cubin by name. Byte parity of every live output
(output row, both ring buffers, the invalid flag) against the kit chain, then interleaved CUDA-graph timings with the
empty-kernel floor at the same launch shape.

Usage (on the head, under the GPU lock):
  python3 vbench.py --arms v_cur,v_128x4 --rows 1,7 --d 512 --json out.json
"""

import argparse
import ctypes as ct
import fcntl
import json
import os
import sys
import urllib.request
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
opts = None
results = []
failures = []

KIT = {
    "norm128": ("0cca97", "_rmsnorm", 16),
    "norm512": ("ee9e02", "_rmsnorm", 16),
}
ENGINE = {"old": ("old", "tf_ds_compress2x_norm_kernel", "d", None),
          "new128": ("new", "tf_ds_compress2x_norm128_kernel", 128, 128),
          "new512": ("new", "tf_ds_compress2x_norm512_kernel", 512, 512)}
VARIANTS = {
    # name: (threads)
    "v_cur": "d", "v_128x4": 128, "v_128x4v": 128, "v_128x4_r1": 128, "v_128x4v_r1": 128,
    "v_128x4v_1e": 128, "v_128x4v_r1_1e": 128,
    "v_cur_r1": "d", "v_cur_1e": "d", "v_cur_r1_1e": "d",
    "v_512only": 512, "v_128only": 128, "v_128only_2k": 128, "v_final128": 128, "v_final512x4": 128,
    "v_final128_1e": 128, "v_final128_nopair": 128, "v_final128g": 128, "v_256x2": 256,
    "v_win1": 128, "v_win4": (128, 4), "v_win7": (128, 7), "v_win7n": (128, 7),
}
ABLATION = {"v_final128_nopair"}


def cu(name, *args):
    err = getattr(cuda, name)(*args)
    if err:
        msg = ct.c_char_p()
        cuda.cuGetErrorString(err, ref(msg))
        raise RuntimeError(f"{name}: {err} {msg.value}")


def argptr(x):
    if isinstance(x, ct._Pointer):
        return ct.addressof(x)
    return ct.addressof(x)


class Buffer:
    def __init__(self, data):
        global allocated
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

    def readarr(self):
        host = np.empty(self.shape, self.dtype)
        cu("cuMemcpyDtoH_v2", P(host.ctypes.data), self.addr, ct.c_size_t(host.nbytes))
        return host


def out(shape, dtype):
    return Buffer(np.full(shape, 0x55, dtype))


def module(path):
    if path not in modules:
        mod = P()
        cu("cuModuleLoad", ref(mod), str(path).encode())
        modules[path] = mod
    return modules[path]


def function(path, name):
    key = (path, name)
    if key not in kernels:
        fn = P()
        cu("cuModuleGetFunction", ref(fn), module(path), name.encode())
        kernels[key] = fn
    return kernels[key]


def launch(path, name, grid, threads, shared, *args):
    fn = function(path, name)
    pack = []
    for a in args:
        if isinstance(a, Buffer):
            pack.append(a.addr)
        elif isinstance(a, int):
            pack.append(I(a))
        elif isinstance(a, np.integer):
            pack.append(I(int(a)))
        else:
            pack.append(a)
    argv = (P * len(pack))(*(argptr(x) for x in pack))
    gx, gy = grid if isinstance(grid, tuple) else (grid, 1)
    bx, by = threads if isinstance(threads, tuple) else (threads, 1)
    cu("cuLaunchKernel", fn, gx, gy, 1, bx, by, 1, shared, stream, argv, None)


def aot(key, grid, *args):
    prefix, name, smem = KIT[key]
    (path,) = opts.kit.glob(prefix + "*.cubin")
    launch(path, name, grid, 128, smem, *args, P(), P())


def compare(label, actual, expected):
    cu("cuStreamSynchronize", stream)
    total = 0
    for i, (buf, gold) in enumerate(zip(actual, expected, strict=True)):
        mismatch = np.flatnonzero(buf.read() != gold.read())
        if mismatch.size:
            failures.append(f"{label} buffer {i}: {len(mismatch)} bytes differ (first {mismatch[0]})")
            total += len(mismatch)
    return total


def time_graphs(label, chains, names, floor, batch, reps):
    def build(chain):
        chain()
        cu("cuStreamSynchronize", stream)
        cu("cuStreamBeginCapture_v2", stream, 0)
        for _ in range(batch):
            chain()
        g, e = P(), P()
        cu("cuStreamEndCapture", stream, ref(g))
        cu("cuGraphInstantiateWithFlags", ref(e), g, ct.c_uint64(0))
        cu("cuGraphDestroy", g)
        return e

    exes = [build(c) for c in chains]
    if floor is not None:
        exes.append(build(floor))
        names = list(names) + ["floor"]
    for e in exes:
        cu("cuGraphLaunch", e, stream)
        cu("cuStreamSynchronize", stream)
    start, stop = P(), P()
    for ev in (start, stop):
        cu("cuEventCreate", ref(ev), 0)
    samples = None
    for attempt in range(opts.tries):
        c0 = None
        if not opts.no_guard:
            wait_health(opts.wait)
            c0 = served_counters()
        samples = [[] for _ in exes]
        for rep in range(reps):
            for off in range(len(exes)):
                i = (rep + off) % len(exes)
                cu("cuEventRecord", start, stream)
                cu("cuGraphLaunch", exes[i], stream)
                cu("cuEventRecord", stop, stream)
                cu("cuEventSynchronize", stop)
                ms = F()
                cu("cuEventElapsedTime", ref(ms), start, stop)
                samples[i].append(ms.value * 1000.0 / batch)
        if opts.no_guard or served_counters() == c0:
            break
        print(f"{label}: production served during the timing block, retry {attempt + 1}/{opts.tries}", flush=True)
    for e in exes:
        cu("cuGraphExecDestroy", e)
    for ev in (start, stop):
        cu("cuEventDestroy_v2", ev)
    stats = {}
    for name, s in zip(names, samples):
        a = np.asarray(s)
        stats[name] = {"median": round(float(median(s)), 3),
                       "iqr": round(float(np.percentile(a, 75) - np.percentile(a, 25)), 3), "n": len(s)}
    print(f"{label}: " + " ".join(f"{k}={v['median']:.3f}[{v['iqr']:.3f}]" for k, v in stats.items()), flush=True)
    return stats


def bf(shape):
    return (rng.normal(size=shape).astype("f4").view("u4") >> 16).astype("u2")


def case_decode(rows, d, layout="mixed", timing=True, special=False):
    n = rows * 2
    stride = d + int(layout == "unaligned")
    label = f"decode R={rows} D={d} {layout}{' special' if special else ''}"
    slots = rng.permutation(n)[:rows].astype("i8")
    if layout == "cached":
        pair = np.tile(np.setdiff1d(np.arange(n), slots), (2, 1)).T.copy()
    elif layout == "window":
        # the served speculative shape: seven-row windows with four distinct ordered source pairs (rows j and j+4
        # share one), so a deduping kernel has something to dedupe.
        pair = np.zeros((rows, 2), "i8")
        for base in range(0, rows, 7):
            k = min(7, rows - base)
            for j in range(k):
                a = j % 4
                b = (a + 1) % 4 if k > 4 else (a + 1) % k
                pair[base + j] = (slots[base + a], slots[base + b])
    elif layout in ("cross", "invalid", "stress"):
        pair = np.column_stack((np.roll(slots, 1), np.roll(slots, -1)))
        if layout == "invalid":
            pair[::5, 1] = -1
            pair[::7, 0] = n
    else:
        pair = np.column_stack((slots, np.setdiff1d(np.arange(n), slots)))
        pair[1::2] = pair[1::2, ::-1]
    inputs = [rng.normal(size=(rows, stride)).astype("f4") for _ in range(2)]
    rings = [rng.normal(size=(n, d)).astype("f4") for _ in range(2)]
    if special or layout == "stress":
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x00800000], "u4")
        for a in (inputs[1], rings[1]):
            a[:, ::7] = np.resize(bits.view("f4"), a[:, ::7].shape)
        inputs[0][:, ::5] = np.resize(bits.view("f4"), inputs[0][:, ::5].shape)
    x, gate = map(Buffer, inputs)
    slot, gi = Buffer(slots), Buffer(pair)
    w = Buffer(bf(d))
    chains, states, names = [], [], []
    for i, arm in enumerate(["ref", "old"] + opts.arms):
        if arm in ENGINE and ENGINE[arm][3] is not None and ENGINE[arm][3] != d:
            continue
        rk, rg = map(Buffer, rings)
        flag = Buffer(np.zeros(1, "u4"))
        mid, dst = out((rows, d), "u2"), out((rows, d + 16), "u2")
        prefix = (x, L(stride), gate, L(stride), slot, rk, rg, gi, L(n))

        def chain(arm=arm, prefix=prefix, dst=dst, mid=mid, flag=flag):
            if arm == "ref":
                launch(opts.new_torch, "tf_ds_compress2x_kernel", rows, 256, 0, *prefix, mid, L(d), rows, d, flag, 0)
                aot(f"norm{d}", rows, mid, d, w, dst, d + 16, eps)
            elif arm in ENGINE:
                ver, name, thr, _w = ENGINE[arm]
                launch(opts.old_torch if ver == "old" else opts.new_torch, name, rows,
                       d if thr == "d" else thr, 0, *prefix, w, dst, L(d + 16), rows, d, eps, flag, 0)
            else:
                thr = VARIANTS[arm]
                launch(opts.variants, arm, (rows + thr[1] - 1) // thr[1] if isinstance(thr, tuple) else rows,
                       thr if isinstance(thr, tuple) else (d if thr == "d" else thr), 0, *prefix, w, dst, L(d + 16),
                       rows, d, eps, flag, 0)

        chain()
        chains.append(chain)
        names.append(arm)
        states.append([dst, rk, rg, flag])
    diffs = {}
    for k, arm in enumerate(names):
        if arm == "ref" or arm in ABLATION:
            continue
        diffs[arm] = compare(f"{label} {arm}", states[k], states[0])
        if diffs[arm] and opts.debug:
            a, b = states[k][0].readarr(), states[0][0].readarr()
            rows_bad = np.flatnonzero((a != b).any(axis=1))
            print(f"  {arm}: rows differing {rows_bad[:12].tolist()} of {a.shape[0]}", flush=True)
            if rows_bad.size:
                r = int(rows_bad[0])
                print(f"    row {r} ref {b[r, :6]} got {a[r, :6]}", flush=True)
    row = {"kernel": "0197-compress-decode", "shape": label, "rows": rows, "d": d, "parity": diffs}
    if timing and layout in ("mixed", "window") and not special:
        floor = lambda: launch(opts.floor, "empty_kernel", rows, d, 0)
        row["timing"] = time_graphs(label, chains, names, floor, opts.batch, opts.reps)
    results.append(row)
    return row


def floor_probe():
    sink = Buffer(np.zeros(1, "u4"))
    for grid in (1, 48):
        for threads in (128, 512):
            a = lambda: launch(opts.variants, "v_empty", grid, threads, 0)
            b = lambda: launch(opts.variants, "v_empty2k", grid, threads, 0, sink)
            st = time_graphs(f"floor probe grid={grid} block={threads}", [a, b], ["empty", "empty2k"], None,
                             opts.batch, opts.reps)
            results.append({"kernel": "floor-probe", "shape": f"grid={grid} block={threads}", "timing": st})


def pair_probe():
    vals = np.array([0.0, -0.0, 1.0, -1.0, 3.402823466e38, -3.402823466e38, float("inf"), float("-inf"),
                     float("nan"), 1e-38, -1e-38, 100.0, -100.0, 0.5, -0.5, 2.0], "f4")
    table = Buffer(vals)
    n = len(vals)
    total = n ** 4
    res = out((total,), "u4")
    launch(opts.variants, "v_pair_probe", (total + 255) // 256, 256, 0, table, n, res)
    cu("cuStreamSynchronize", stream)
    arr = res.readarr()
    bad = int(arr.sum())
    print(f"pair probe: {total} combinations, {bad} differing bf16 outputs", flush=True)
    results.append({"kernel": "pair-probe", "shape": f"{total} combos", "parity": {"one_exp_vs_ref": bad}})
    if bad:
        idx = np.flatnonzero(arr)[:8]
        for i in idx:
            a = i // (n ** 3), (i // (n ** 2)) % n, (i // n) % n, i % n
            print("  differs at", a, [float(vals[j]) for j in a], flush=True)


def pair_probe3(n=1 << 22):
    def draw():
        a = rng.normal(size=n).astype("f4")
        bits = np.array([0, 0x80000000, 1, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f7fffff, 0x00800000,
                         0x80000001, 0xff7fffff], "u4")
        a[::97] = np.resize(bits.view("f4"), a[::97].shape)
        a[::1009] = np.float32(-3.402823466e38)
        return a
    cols = [Buffer(draw()) for _ in range(4)]
    res = out((n,), "u4")
    launch(opts.variants, "v_pair_probe3", (n + 255) // 256, 256, 0, *cols, n, res)
    cu("cuStreamSynchronize", stream)
    bad = int(res.readarr().sum())
    print(f"pair probe3: {n} random+special rows, {bad} differing bf16 outputs", flush=True)
    results.append({"kernel": "pair-probe3", "shape": f"{n} rows", "parity": {"one_exp_vs_ref": bad}})


def mem_available_gib():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1048576.0
    return 0.0


def served_counters():
    h = health_now()
    return {k: h.get(k) for k in ("prompt_tokens_total", "generation_tokens_total")}


def health_now():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            return json.load(r)
    except Exception:  # noqa: BLE001
        return {}


def wait_health(timeout=1800):
    import time
    deadline = time.time() + timeout if timeout else None
    while True:
        h = health_now()
        conns = h.get("live", {}).get("connections")
        if conns == 0:
            print(f"health: connections={conns}", flush=True)
            return h
        print(f"production serving ({conns} connections), waiting", flush=True)
        if deadline is not None and time.time() > deadline:
            sys.exit("production stayed busy, refusing")
        time.sleep(20)


def main():
    global opts
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", type=Path, default=Path.home() / "tf-bench/fused-next/wp2/bin")
    p.add_argument("--kit", type=Path, default=Path("/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins"))
    p.add_argument("--arms", default="v_cur")
    p.add_argument("--rows", default="1,7,32,48")
    p.add_argument("--d", default="128,512")
    p.add_argument("--layouts", default="mixed,unaligned,cached,cross,invalid")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--timing-only", action="store_true")
    p.add_argument("--pair-probe", action="store_true")
    p.add_argument("--floor-probe", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--pair-probe3", action="store_true")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--wait", type=int, default=1800)
    p.add_argument("--tries", type=int, default=20)
    opts = p.parse_args()
    opts.arms = [a for a in opts.arms.split(",") if a]
    opts.new_torch = opts.bin / "new-torch.cubin"
    opts.old_torch = opts.bin / "old-torch.cubin"
    opts.variants = opts.bin / "variants.cubin"
    opts.floor = opts.bin / "floor.cubin"
    for f in (opts.new_torch, opts.old_torch, opts.variants, opts.floor):
        if not f.exists():
            sys.exit(f"missing {f}")
    lock = None
    if not opts.no_guard:
        avail = mem_available_gib()
        if avail < 3.0:
            sys.exit(f"MemAvailable {avail:.2f} GiB < 3 GiB, refusing")
        print(f"MemAvailable {avail:.2f} GiB", flush=True)
        lock = open(os.path.expanduser("~/tf-bench/gpu.lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        h0 = wait_health(opts.wait)
        print(f"health before: connections={h0.get('connections')} "
              f"{served_counters()}", flush=True)
    cu("cuInit", 0)
    ctx = P()
    cu("cuDevicePrimaryCtxRetain", ref(ctx), 0)
    cu("cuCtxSetCurrent", ctx)
    cu("cuStreamCreate", ref(stream), 1)
    try:
        if opts.pair_probe:
            pair_probe()
        if opts.pair_probe3:
            pair_probe3()
        if opts.floor_probe:
            floor_probe()
        for rows in [int(v) for v in opts.rows.split(",") if v]:
            for d in [int(v) for v in opts.d.split(",") if v]:
                for layout in opts.layouts.split(","):
                    if not layout:
                        continue
                    special = layout == "stress"
                    timing = not opts.parity_only and layout in ("mixed", "window")
                    case_decode(rows, d, layout, timing, special)
        for layout in ("stress",):
            case_decode(7, 512, layout, False, True)
            case_decode(33, 512, "unaligned", False, False)
            case_decode(65, 512, "cached", False, False)
            case_decode(65, 512, "cross", False, False)
            case_decode(65, 512, "invalid", False, False)
            case_decode(1, 128, "invalid", False, False)
    finally:
        cu("cuStreamSynchronize", stream)
        for addr in allocations:
            cu("cuMemFree_v2", addr)
    if opts.json:
        opts.json.write_text(json.dumps({"results": results, "failures": failures}, indent=1))
    print(f"failures: {len(failures)}", flush=True)
    for f in failures[:20]:
        print(" ", f, flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
