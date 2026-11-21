#!/usr/bin/env python3
"""WP3 candidate bench for the 0198 rope-store family (decode key/latent/query, prompt key/latent).

Arms: the kit chain each fused kernel replaces (AOT `_rmsnorm`/`_rope_heads`/`_fp4_store`/`_fp4qd_p2`), the engine
kernel from the WP2-tip cubin ("old"), the engine kernel from the current tree ("new"), and any candidate in
bin/variants.cubin by name. Byte parity of every live output against the kit chain, then interleaved CUDA-graph
timings with an empty-kernel floor at each arm's own launch shape.

Usage (on the head, under the GPU lock):
  python3 vbench.py --kind query --arms v_q_cur,v_q_h8 --rows 1,7,32,48 --json out.json
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
rng = np.random.default_rng(198)
opts = None
results = []
failures = []

KIT = {
    "norm128": ("0cca97", "_rmsnorm", 16),
    "rope32": ("5ed6b4", "_rope_heads", 0),
    "rope128": ("e8bd0a", "_rope_heads", 0),
    "rope512": ("a22128", "_rope_heads", 0),
    "store128": ("345438", "_fp4_store", 512),
    "store512": ("9cd357", "_fp4_store", 2048),
    "qd": ("4cbef7", "_fp4qd_p2", 0),
}
# per-arm launch geometry: (cubin, kernel, heads-per-block divisor, threads)
VARIANTS = {
    "v_q_cur": ("v_q_cur", 1, 128),
    "v_q_h2": ("v_q_h2", 2, 256),
    "v_q_h4": ("v_q_h4", 4, 512),
    "v_q_h8": ("v_q_h8", 8, 1024),
    "v_q_row": ("v_q_row", "row", 128),
    "v_q_row4": ("v_q_row4", "row", 512),
    "v_q_recip": ("v_q_recip", 1, 128),
    "v_q_bits": ("v_q_bits", 1, 128),
    "v_q_w4r": ("v_q_w4r", 4, 128),
    "v_q_w4b": ("v_q_w4b", 4, 128),
    "v_q_w8b": ("v_q_w8b", 8, 256),
    "v_q_p1": ("v_q_p1", 1, 32),
    "v_q_p2": ("v_q_p2", 2, 64),
    "v_q_p4": ("v_q_p4", 4, 128),
    "v_q_p8": ("v_q_p8", 8, 256),
    "v_q_p4f": ("v_q_p4f", 4, 128),
    "v_q_p8i": ("v_q_p8i", 8, 256),
    "v_q_p4i": ("v_q_p4i", 4, 128),
    "v_q_bitsi": ("v_q_bitsi", 1, 128),
    "v_q_p8f": ("v_q_p8f", 8, 256),
    "v_q_s1": ("v_q_s1", "seg", 32),
    "v_q_p8x": ("v_q_p8x", 8, 256),
    "v_q_p8xf": ("v_q_p8xf", 8, 256),
    "v_q_p4x": ("v_q_p4x", 4, 128),
    "v_q_p8y": ("v_q_p8y", 8, 256),
    "v_q_p8yf": ("v_q_p8yf", 8, 256),
    "v_q_p16": ("v_q_p16", 16, 512),
    "v_q_p2b": ("v_q_p2b", 2, 64),
    "v_q_bitsi2": ("v_q_bitsi2", 1, 128),
}


def cu(name, *args):
    err = getattr(cuda, name)(*args)
    if err:
        msg = ct.c_char_p()
        cuda.cuGetErrorString(err, ref(msg))
        raise RuntimeError(f"{name}: {err} {msg.value}")


def argptr(x):
    return ct.addressof(x)


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
        got, want = buf.read(), gold.read()
        mismatch = np.flatnonzero(got != want)
        if mismatch.size:
            failures.append(f"{label} buffer {i}: {len(mismatch)} bytes differ (first {mismatch[0]})")
            total += len(mismatch)
            if opts.debug:
                k = int(mismatch[0]) & ~1
                g = np.frombuffer(got[k:k + 32].tobytes(), dtype="u2")
                w = np.frombuffer(want[k:k + 32].tobytes(), dtype="u2")
                print(f"  {label}: {mismatch.size} bytes differ, first byte {int(mismatch[0])} "
                      f"ref={w[:8].tolist()} got={g[:8].tolist()}", flush=True)
    return total


def time_graphs(label, chains, names, floors, batch, reps):
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
    names = list(names)
    for i, fl in enumerate(floors):
        if fl is not None:
            exes.append(build(fl))
            names.append(f"floor_{names[i]}")
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


def case_query(rows, special=False, timing=True):
    """The decode indexer query: kit `_rope_heads` (RD=64, H=32) + `_fp4qd_p2` against the fused kernels."""
    heads, hd = 32, 128
    label = f"query R={rows} H={heads} HD={hd}{' special' if special else ''}"
    src = bf((rows, heads, hd))
    if special:
        src[..., :8] = [0x7FC1, 0x7F80, 0xFF80, 0x8000, 0, 1, 0x7F7F, 0x8001]
        src[..., 32:64] = 0x7FC1
        src[..., 64:72] = [0x0001, 0x8001, 0x7F7F, 0xFBFF, 0x0400, 0x7C00, 0xFC00, 0x3C00]
    pos = Buffer(rng.permutation(rows).astype("i8"))
    angle = rng.uniform(-np.pi, np.pi, (rows, 32))
    cos, sin = Buffer(np.cos(angle).astype("f4")), Buffer(np.sin(angle).astype("f4"))
    if opts.debug:
        print(f"  {label}: src[0,0,:8]={src[0, 0, :8].tolist()} pos={pos.readarr()[:4].tolist()} "
              f"cos={cos.readarr()[0, :4].tolist()}", flush=True)
    chains, states, names, floors = [], [], [], []
    for arm in ["ref", "old", "new"] + opts.arms:
        x = Buffer(src)
        dq = out((rows, heads, hd), "u2")

        def chain(arm=arm, x=x, dq=dq):
            if arm == "ref":
                aot("rope32", (rows, 2), x, cos, sin, pos)
                nblk = rows * heads * hd // 32
                aot("qd", (nblk + 31) // 32, x, dq, nblk)
            elif arm == "old":
                launch(opts.old_torch, "tf_ds_fused_rope_qd_kernel", rows * heads, 128, 0, x, dq, cos, sin, pos, heads,
                       hd, 64, 0)
            elif arm == "new":
                # this tree's dispatch: a warp a head from decode's largest window upward
                if rows >= 32:
                    launch(opts.new_torch, "tf_ds_fused_rope_qd_pair_kernel", (rows, heads // 8), 256, 0, x, dq, cos,
                           sin, pos, heads, hd, 64, 0)
                else:
                    launch(opts.new_torch, "tf_ds_fused_rope_qd_kernel", (rows, heads), 128, 0, x, dq, cos, sin, pos,
                           heads, hd, 64, 0)
            else:
                name, hpb, thr = VARIANTS[arm]
                if hpb == "row":
                    grid = rows
                elif hpb == 1:
                    grid = (rows, heads)          # one element a thread: a block a head
                elif hpb == "seg":
                    grid = (rows, heads * hd // 64)
                else:
                    grid = (rows, heads // hpb)   # hpb heads a block, no division in the kernel
                if name.endswith("x") or name.endswith("y"):
                    launch(opts.variants, name, grid, thr, 0, x, dq, cos, sin, pos, heads, 64, 0)
                else:
                    launch(opts.variants, name, grid, thr, 0, x, dq, cos, sin, pos, heads, hd, 64, 0)

        chain()
        chains.append(chain)
        names.append(arm)
        states.append([dq])
        if arm == "old":
            floors.append(lambda: launch(opts.floor, "empty_kernel", rows * heads, 128, 0))
        elif arm == "new":
            if rows >= 32:
                floors.append(lambda: launch(opts.floor, "empty_kernel", (rows, heads // 8), 256, 0))
            else:
                floors.append(lambda: launch(opts.floor, "empty_kernel", (rows, heads), 128, 0))
        elif arm == "ref":
            floors.append(None)
        else:
            _, hpb, thr = VARIANTS[arm]
            if hpb == "row":
                grid = rows
            elif hpb == 1:
                grid = (rows, heads)
            elif hpb == "seg":
                grid = (rows, heads * hd // 64)
            else:
                grid = (rows, heads // hpb)
            floors.append(lambda g=grid, t=thr: launch(opts.floor, "empty_kernel", g, t, 0))
    diffs = {}
    for k, arm in enumerate(names):
        if arm == "ref":
            continue
        diffs[arm] = compare(f"{label} {arm}", states[k], states[0])
    row = {"kernel": "0198-query", "shape": label, "rows": rows, "parity": diffs}
    if timing and not special:
        row["timing"] = time_graphs(label, chains, names, floors, opts.batch, opts.reps)
    results.append(row)
    return row


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
    p.add_argument("--bin", type=Path, default=Path.home() / "tf-bench/fused-next/wp3/bin")
    p.add_argument("--kit", type=Path, default=Path("/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins"))
    p.add_argument("--kind", default="query")
    p.add_argument("--variants", type=Path, default=None, help="candidate cubin (default bin/variants.cubin)")
    p.add_argument("--arms", default="")
    p.add_argument("--rows", default="1,7,32,48")
    p.add_argument("--json", type=Path, default=None)
    p.add_argument("--parity-only", action="store_true")
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--batch", type=int, default=100)
    p.add_argument("--no-guard", action="store_true")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--wait", type=int, default=1800)
    p.add_argument("--tries", type=int, default=20)
    opts = p.parse_args()
    opts.arms = [a for a in opts.arms.split(",") if a]
    opts.new_torch = opts.bin / "new-torch.cubin"
    opts.old_torch = opts.bin / "old-torch.cubin"
    opts.variants = opts.variants or (opts.bin / "variants.cubin")
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
        print(f"health before: connections={h0.get('connections')} {served_counters()}", flush=True)
    cu("cuInit", 0)
    ctx = P()
    cu("cuDevicePrimaryCtxRetain", ref(ctx), 0)
    cu("cuCtxSetCurrent", ctx)
    cu("cuStreamCreate", ref(stream), 1)
    try:
        rows = [int(v) for v in opts.rows.split(",") if v]
        if opts.kind == "query":
            for r in rows:
                case_query(r, False, not opts.parity_only)
            case_query(7, True, False)
            case_query(33, False, False)
            case_query(1, True, False)
        elif opts.kind == "fp4":
            table = np.arange(65536, dtype="u2")
            tb = Buffer(table)
            res = Buffer(np.zeros(1, "u4"))
            launch(opts.variants, "v_fp4_probe", (65536 * 255 + 255) // 256, 256, 0, tb, 65536, res)
            cu("cuStreamSynchronize", stream)
            bad = int(res.readarr()[0])
            print(f"fp4 probe: {65536 * 255} (bf16, E8M0 byte) pairs, {bad} differing codes/outputs", flush=True)
            results.append({"kernel": "fp4-probe", "shape": f"{65536 * 255} pairs",
                            "parity": {"intrinsic_vs_engine": bad}})
            if bad:
                failures.append(f"fp4 probe: {bad} pairs differ")
                pairs, count = out((128,), "u4"), Buffer(np.zeros(1, "u4"))
                launch(opts.variants, "v_fp4_diff2", (65536 * 255 + 255) // 256, 256, 0, tb, 65536, pairs, count)
                cu("cuStreamSynchronize", stream)
                cnt = int(count.readarr()[0])
                print(f"  output-byte differences: {cnt}", flush=True)
                for q in pairs.readarr()[:10]:
                    q = int(q)
                    vi, byte, oc, oi = q >> 24, (q >> 16) & 0xFF, (q >> 8) & 0xFFFF, q & 0xFFFF
                    v = np.array([table[vi]], "u2").view("f2")[0]
                    print(f"    bits {int(table[vi]):#06x} v={float(v):.6g} byte={byte} "
                          f"engine={oc:#06x} intrinsic={oi:#06x}", flush=True)
                pairs, count = out((128,), "u4"), Buffer(np.zeros(1, "u4"))
                launch(opts.variants, "v_fp4_diff", (65536 * 255 + 255) // 256, 256, 0, tb, 65536, pairs, count)
                cu("cuStreamSynchronize", stream)
                cnt = int(count.readarr()[0])
                print(f"  code differences: {cnt}", flush=True)
                for p in pairs.readarr()[:10]:
                    vi, byte = int(p) >> 8, int(p) & 0xFF
                    v = np.array([table[vi]], "u2").view("f2")[0]
                    print(f"    bits {int(table[vi]):#06x} v={float(v):.6g} byte={byte}", flush=True)
                for byte in (127, 120):
                    cur, intr = out((65536,), "u4"), out((65536,), "u4")
                    launch(opts.variants, "v_fp4_dump", 256, 256, 0, tb, 65536, byte, cur, intr)
                    cu("cuStreamSynchronize", stream)
                    a, b = cur.readarr(), intr.readarr()
                    badv = np.flatnonzero(a != b)
                    print(f"  byte {byte}: {badv.size} inputs differ", flush=True)
                    for k in badv[:10]:
                        v = np.array([table[k]], "u2").view("f2")[0]
                        print(f"    bits {int(table[k]):#06x} v={float(v):.6g} engine={int(a[k])} "
                              f"intrinsic={int(b[k])}", flush=True)
        else:
            sys.exit(f"unknown --kind {opts.kind}")
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
