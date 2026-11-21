#!/usr/bin/env python3
"""Device-to-device copy bandwidth on this GPU, for the fused kernels' memory rooflines.

Times floor.cubin's copy_kernel through CUDA graphs at a few sizes (interleaved, warmed, median + IQR) and reports
GB/s moved (read + write). Run on the head beside the bench:

  python3 bw.py ~/tf-bench/fused-next/bin [--reps 20] [--batch 50]
"""

import argparse
import ctypes as ct
import fcntl
import json
import os
import time
import urllib.request
from statistics import median

import numpy as np

P, I = ct.c_void_p, ct.c_int
ref = ct.byref
cuda = ct.CDLL("libcuda.so.1")
stream = P()
opts = None


def cu(name, *args):
    err = getattr(cuda, name)(*args)
    if err:
        msg = ct.c_char_p()
        cuda.cuGetErrorString(err, ref(msg))
        raise RuntimeError(f"{name}: {err} {msg.value}")


def mem_available_gib():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1048576.0
    return 0.0


def health_idle():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r:
            return json.load(r).get("live", {}).get("connections") == 0
    except Exception:  # noqa: BLE001
        return True


def launch(fn, grid, block, *args):
    pack = [a if isinstance(a, ct.c_void_p) else I(a) for a in args]
    argv = (P * len(pack))(*(ct.addressof(x) for x in pack))
    cu("cuLaunchKernel", fn, grid, 1, 1, block, 1, 1, 0, stream, argv, None)


def time_graph(chain, batch, reps, label, moved):
    chain()
    cu("cuStreamSynchronize", stream)
    cu("cuStreamBeginCapture_v2", stream, 0)
    for _ in range(batch):
        chain()
    g, e = P(), P()
    cu("cuStreamEndCapture", stream, ref(g))
    cu("cuGraphInstantiateWithFlags", ref(e), g, ct.c_uint64(0))
    cu("cuGraphDestroy", g)
    cu("cuGraphLaunch", e, stream)
    cu("cuStreamSynchronize", stream)
    start, stop = P(), P()
    for ev in (start, stop):
        cu("cuEventCreate", ref(ev), 0)
    samples = []
    for _ in range(reps):
        cu("cuEventRecord", start, stream)
        cu("cuGraphLaunch", e, stream)
        cu("cuEventRecord", stop, stream)
        cu("cuEventSynchronize", stop)
        ms = ct.c_float()
        cu("cuEventElapsedTime", ref(ms), start, stop)
        samples.append(ms.value * 1000.0 / batch)
    cu("cuGraphExecDestroy", e)
    for ev in (start, stop):
        cu("cuEventDestroy_v2", ev)
    a = np.asarray(samples)
    med = float(median(samples))
    gbs = moved / (med * 1e-6) / 1e9
    print(f"{label}: {med:.2f} us [{np.percentile(a, 75) - np.percentile(a, 25):.2f}] "
          f"{moved / 1e6:.1f} MB moved -> {gbs:.1f} GB/s", flush=True)
    return {"label": label, "us": med, "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)), "GBps": gbs}


def main():
    global opts
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bin", type=str, help="directory with floor.cubin")
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=50)
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--no-guard", action="store_true")
    opts = parser.parse_args()

    if not opts.no_guard:
        avail = mem_available_gib()
        if avail < 3.0:
            raise SystemExit(f"MemAvailable {avail:.2f} GiB < 3 GiB, refusing")
        print(f"MemAvailable {avail:.2f} GiB", flush=True)
        lock = open(os.path.expanduser("~/tf-bench/gpu.lock"), "w")
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not health_idle():
            raise SystemExit("production is serving, retry later")

    cu("cuInit", 0)
    ctx = P()
    cu("cuDevicePrimaryCtxRetain", ref(ctx), 0)
    cu("cuCtxSetCurrent", ctx)
    cu("cuStreamCreate", ref(stream), 1)
    mod = P()
    cu("cuModuleLoad", ref(mod), os.path.join(opts.bin, "floor.cubin").encode())
    fn = P()
    cu("cuModuleGetFunction", ref(fn), mod, b"copy_kernel")

    rows = []
    # (MiB per side, grid blocks, block threads): grid-stride saturates with ~4 blocks/SM on 10 SM x 2? use 2048.
    for mib in (1, 8, 64, 256):
        n = mib * 1024 * 1024 // 16
        src, dst = P(), P()
        cu("cuMemAlloc_v2", ref(src), ct.c_size_t(mib * 1024 * 1024))
        cu("cuMemAlloc_v2", ref(dst), ct.c_size_t(mib * 1024 * 1024))
        cu("cuMemsetD8Async", dst, 0, ct.c_size_t(mib * 1024 * 1024), stream)
        chain = lambda src=src, dst=dst, n=n: launch(fn, 2048, 256, src, dst, n)
        rows.append(time_graph(chain, opts.batch, opts.reps, f"copy {mib} MiB", 2 * mib * 1024 * 1024))
        cu("cuMemFree_v2", src)
        cu("cuMemFree_v2", dst)
    cache = max(r["GBps"] for r in rows if "1 MiB" in r["label"] or "8 MiB" in r["label"])
    dram = max(r["GBps"] for r in rows if "64 MiB" in r["label"] or "256 MiB" in r["label"])
    print(f"cache-resident copy peak {cache:.1f} GB/s; DRAM copy peak {dram:.1f} GB/s", flush=True)
    if opts.json:
        open(opts.json, "w").write(json.dumps({"rows": rows, "cache_GBps": cache, "dram_GBps": dram,
                                               "peak_GBps": max(cache, dram)}, indent=1))
        print(f"wrote {opts.json}")


if __name__ == "__main__":
    main()
