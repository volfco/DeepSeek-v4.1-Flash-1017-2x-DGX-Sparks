#!/usr/bin/env python3
"""Derive the kit _score_keys key/tmax encoding from its own bytes (read-only use of the pinned cubin)."""
import ctypes as ct, json, os, sys, fcntl, urllib.request
from pathlib import Path
import numpy as np

I, L, P, F = ct.c_int, ct.c_int64, ct.c_void_p, ct.c_float
U64 = ct.c_ulonglong
ref = ct.byref
cuda = ct.CDLL("libcuda.so.1")
stream = P()

def cu(name, *a):
    e = getattr(cuda, name)(*a)
    if e:
        m = ct.c_char_p(); cuda.cuGetErrorString(e, ref(m)); raise RuntimeError(f"{name}: {e} {m.value}")

def mem(data):
    a = P(); cu("cuMemAlloc_v2", ref(a), ct.c_size_t(data.nbytes))
    cu("cuMemcpyHtoD_v2", a, P(data.ctypes.data), ct.c_size_t(data.nbytes)); return a

def read(a, shape, dtype):
    h = np.empty(shape, dtype); cu("cuMemcpyDtoH_v2", P(h.ctypes.data), a, ct.c_size_t(h.nbytes)); return h

def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as r: return json.load(r)
    except Exception: return {}

avail = int([l for l in open("/proc/meminfo") if l.startswith("MemAvailable")][0].split()[1])/1048576
print(f"MemAvailable {avail:.2f} GiB")
assert avail >= 3.0

KIT = Path("/nfs1/tensorfold-dsv41/kit-abe07c7d/kit/aot/cubins")
cubin = next(KIT.glob("df85d0*.cubin"))
lock = open(os.path.expanduser("~/tf-bench/gpu.lock"), "w"); fcntl.flock(lock, fcntl.LOCK_EX)
while health().get("live", {}).get("connections") != 0:
    print("production busy, waiting"); import time; time.sleep(20)
print("health: connections=0")

cu("cuInit", 0); ctx = P(); cu("cuDevicePrimaryCtxRetain", ref(ctx), 0); cu("cuCtxSetCurrent", ctx)
cu("cuStreamCreate", ref(stream), 1)
mod = P(); cu("cuModuleLoad", ref(mod), str(cubin).encode())
fn = P(); cu("cuModuleGetFunction", ref(fn), mod, b"_score_keys")

def run(scores):
    rows, w = scores.shape
    nt = (w + 63)//64
    x = mem(scores.astype("f4"))
    keys = mem(np.zeros((rows, w), "u8"))
    tmax = mem(np.zeros((rows, nt), "u8"))
    gx, gy = rows, (w + 1023)//1024
    a0, a1, a2, a3, a4 = x, keys, ct.c_int(w), tmax, ct.c_int(nt)
    n0, n1 = P(), P()
    pack = [a0, a1, a2, a3, a4, n0, n1]
    argv = (P*len(pack))(*(ct.addressof(t) for t in pack))
    cu("cuLaunchKernel", fn, gx, gy, 1, 128, 1, 1, 4096, stream, argv, None)
    cu("cuStreamSynchronize", stream)
    return read(keys, (rows, w), "u8").view("i8"), read(tmax, (rows, nt), "u8").view("i8")

def ordered(bits):
    b = bits.astype(np.uint32).copy()
    b = np.where(b == 0x80000000, 0, b)
    sb = b.view(np.int32)
    return np.where(sb < 0, sb ^ np.int32(0x7fffffff), sb).astype(np.int64)

def cand(bits, j, xor):
    o = ordered(bits).astype(np.uint64)
    k = (o << np.uint64(32)) | (np.uint64(0xFFFFFFFF) - j.astype(np.uint64))
    return (k ^ np.uint64(xor)).view(np.int64)

# a controlled row: mixed signs, ties, +-0, +-inf, NaN, all distinct columns
vals = np.array([0x00000000, 0x80000000, 0x3f800000, 0xbf800000, 0x40000000, 0xc0000000,
                 0x7f800000, 0xff800000, 0x7fc00000, 0x3dcccccd, 0xbdcccccd, 0x00000001,
                 0x80000001, 0x7f7fffff, 0xff7fffff, 0x41200000], "u4")
for w in (64, 68, 4096+4, 16388):
    row = np.tile(vals, (w + len(vals) - 1)//len(vals))[:w]
    keys, tmax = run(row[None, :].view("f4"))
    bits = row.astype(np.uint32)
    j = np.arange(w, dtype=np.int64)
    k_no = cand(bits, j, 0)
    k_xor = cand(bits, j, 0x8000000000000000)
    print(f"width={w}: keys match no-xor={int((keys[0]==k_no).sum())}/{w} xor={int((keys[0]==k_xor).sum())}/{w}")
    for name, k in (("no-xor", k_no), ("xor", k_xor)):
        if (keys[0] == k).all():
            print(f"  keys == {name} encoding")
            break
    else:
        print("  neither; first mismatch", hex(int(keys[0][0])), hex(int(k_no[0])), hex(int(k_xor[0])))
    # tmax: signed max of the matched key per 64-tile (out-of-range = INT64_MIN)
    kk = keys[0].astype(np.int64)
    nt = (w + 63)//64
    t_signed = np.array([max(kk[t*64:min(w,(t+1)*64)]) for t in range(nt)], dtype=np.int64)
    t_unsigned = np.array([max(kk[t*64:min(w,(t+1)*64)].view(np.uint64)) for t in range(nt)], dtype=np.uint64).view(np.int64)
    print(f"  tmax == signed tile max: {int((tmax[0]==t_signed).sum())}/{nt};  == unsigned: {int((tmax[0]==t_unsigned).sum())}/{nt}")
    if (tmax[0] != t_signed).any():
        bad = np.flatnonzero(tmax[0] != t_signed)[:3]
        for t in bad:
            print(f"   tile {t}: kit={hex(int(tmax[0][t])&0xffffffffffffffff)} signed={hex(int(t_signed[t])&0xffffffffffffffff)} unsigned={hex(int(t_unsigned[t])&0xffffffffffffffff)} keys={[hex(int(v)&0xffffffffffffffff) for v in kk[t*64:(t+1)*64][:4]]}")
    print("  first keys:", [hex(int(v) & 0xffffffffffffffff) for v in keys[0][:4]])
    print("  first tmax:", [hex(int(v) & 0xffffffffffffffff) for v in tmax[0][:2]])
