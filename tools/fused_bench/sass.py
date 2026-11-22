#!/usr/bin/env python3
"""Per-kernel SASS evidence: opcode histogram (LDG/STG by width, LDS/STS, SHFL, BAR, ATOMS, MUFU, BRA, total),
ptxas register/stack/shared/local usage, and theoretical occupancy at the served block shape, old against new.

Usage: python3 sass.py <bin directory> [--json out.json]
  (bin holds {old,new}-{ops,torch}.cubin, from build.sh)
"""

import argparse
import ctypes as ct
import json
import re
import subprocess
from pathlib import Path

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"

# served launch block (x, y) per fused kernel, from zig/src/families/dsv41/exact.zig and ops.zig
BLOCKS = {
    "tf_ds_topk_indices_kernel": (1024, 1),
    "tf_ds_topk_indices_cut_kernel": (1024, 1),
    "tf_ds_cand_fast_kernel": (256, 1),
    "tf_ds_cand_fast8_kernel": (256, 1),
    "tf_ds_idx_fused_kernel": (1024, 1),
    "tf_ds_compress2_norm_kernel": (32, 8),    # d=512, R=128: rmsLanes=32, groups=8
    "tf_ds_compress2x_norm128_kernel": (128, 1),  # d=128, one channel a thread
    "tf_ds_compress2x_norm512_kernel": (512, 1),  # d=512
    "tf_ds_fused_idx_key_kernel": (128, 1),
    "tf_ds_fused_rope_store_kernel": (256, 1),
    "tf_ds_fused_rope_qd_kernel": (128, 1),
    "tf_ds_fused_rope_qd_pair_kernel": (256, 1),
    "tf_ds_fused_prompt_idx_key_kernel": (32, 2),
    "tf_ds_fused_prompt_latent_kernel": (256, 1),
    "tf_ds_prune_six_kernel": (64, 1),
}
WANT = list(BLOCKS)
OPS = {"tf_ds_idx_fused_kernel", "tf_ds_prune_six_kernel"}

PATTERNS = {
    "LDG": r"\bLDG\b",
    "LDG.128": r"\bLDG\.E\.128\b",
    "LDG.64": r"\bLDG\.E\.64\b",
    "LDG.U16": r"\bLDG\.E\.U16\b|\bLDG\.E\.S16\b",
    "STG": r"\bSTG\b",
    "STG.128": r"\bSTG\.E\.128\b",
    "STG.64": r"\bSTG\.E\.64\b",
    "LDS": r"\bLDS\b",
    "STS": r"\bSTS\b",
    "SHFL": r"\bSHFL\b",
    "BAR": r"\bBAR\.SYNC\b",
    "ATOM": r"\bATOM\b|\bATOMS\b|\bRED\b",
    "MUFU": r"\bMUFU\b",
    "REDUX": r"\bREDUX\b",
    "BRA": r"\bBRA\b",
    "FFMA": r"\bFFMA\b",
    "FMUL": r"\bFMUL\b",
    "FADD": r"\bFADD\b",
}


def sections(text):
    out, name, body = {}, None, []
    for line in text.splitlines():
        m = re.search(r"Function : (\S+)", line)
        if m:
            if name:
                out[name] = body
            name, body = m.group(1), []
            continue
        if name:
            m = re.search(r"/\*[0-9a-f]+\*/\s+(@!?\w+\s+)?(\w[\w.]*)", line)
            if m:
                body.append(m.group(2))
    if name:
        out[name] = body
    return out


def usage(text):
    out, name = {}, None
    for line in text.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m and name:
            out[name] = dict(zip(("REG", "STACK", "SHARED", "LOCAL"), map(int, m.groups())))
    return out


def device_limits():
    """(max threads/SM, regs/SM, smem/SM, warps/SM) from the driver, so occupancy is this GPU's."""
    cuda = ct.CDLL("libcuda.so.1")
    cuda.cuInit(0)
    ctx = ct.c_void_p()
    cuda.cuDevicePrimaryCtxRetain(ct.byref(ctx), 0)
    cuda.cuCtxSetCurrent(ctx)
    names = {
        0x27: "max_threads_per_sm",       # MAX_THREADS_PER_MULTIPROCESSOR
        0x52: "regs_per_sm",              # MAX_REGISTERS_PER_MULTIPROCESSOR
        0x51: "smem_per_sm",              # MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        0x53: "smem_per_block_optin",
        0x10: "sm_count",
        0x0a: "warp_size",
    }
    out = {}
    for code, key in names.items():
        v = ct.c_int()
        cuda.cuDeviceGetAttribute(ct.byref(v), code, 0)
        out[key] = v.value
    return out


def occupancy(regs, smem_static, block, limits):
    bx, by = block
    threads = bx * by
    if threads == 0:
        return None
    lim = limits
    by_threads = lim["max_threads_per_sm"] // threads
    by_regs = lim["regs_per_sm"] // max(1, regs * threads) if regs else 64
    smem_block = smem_static  # dynamic shared is launch-time; static only here
    by_smem = lim["smem_per_sm"] // smem_block if smem_block else 64
    blocks = max(0, min(by_threads, by_regs, by_smem))
    warps = blocks * threads // lim["warp_size"]
    return {"blocks_per_sm": blocks, "warps_per_sm": warps,
            "theoretical_pct": round(100.0 * warps / (lim["max_threads_per_sm"] // lim["warp_size"]), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bin", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    limits = device_limits()
    print(f"device limits: {limits}", flush=True)
    report = {}
    for version in ("old", "new"):
        print(f"===== {version}", flush=True)
        dumps = {}
        for cubin in ("ops", "torch"):
            path = args.bin / f"{version}-{cubin}.cubin"
            sass = subprocess.run([CUOBJDUMP, "-sass", str(path)], capture_output=True, text=True).stdout
            use = subprocess.run([CUOBJDUMP, "-res-usage", str(path)], capture_output=True, text=True).stdout
            dumps.update(sections(sass))
            dumps.setdefault("_usage", {}).update(usage(use))
        vrep = {}
        header = f"{'kernel':<38} {'insn':>6} " + " ".join(f"{k:>7}" for k in PATTERNS)
        print(header, flush=True)
        for name in WANT:
            body = dumps.get(name)
            if body is None:
                print(f"{name:<38} (not in this pair)", flush=True)
                continue
            counts = {k: sum(1 for ins in body if re.search(p, ins)) for k, p in PATTERNS.items()}
            use = dumps.get("_usage", {}).get(name, {})
            occ = occupancy(use.get("REG", 0), use.get("SHARED", 0), BLOCKS[name], limits)
            print(f"{name.replace('tf_ds_', ''):<38} {len(body):>6} " + " ".join(f"{counts[k]:>7}" for k in PATTERNS)
                  + f"  REG={use.get('REG')} STACK={use.get('STACK')} SHARED={use.get('SHARED')} "
                    f"LOCAL={use.get('LOCAL')} block={BLOCKS[name]} occ={occ}", flush=True)
            vrep[name] = {"insn": len(body), "counts": counts, "usage": use, "block": BLOCKS[name],
                          "occupancy": occ}
        report[version] = vrep
    if args.json:
        args.json.write_text(json.dumps({"limits": limits, "kernels": report}, indent=1))
        print(f"wrote {args.json}", flush=True)


if __name__ == "__main__":
    main()
