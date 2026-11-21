#!/usr/bin/env python3
"""SASS evidence for 0202, 0210 and 0212: per-kernel instruction counts (barriers, shared loads/stores, shuffles,
global loads/stores) and ptxas' register/spill/shared usage, the previous tree against the 0200-0212 tree.
Usage: python3 sass.py <bin directory>  (the cubins build2.sh made)."""

import re
import subprocess
import sys
from pathlib import Path

CUOBJDUMP = "/usr/local/cuda/bin/cuobjdump"
WANT = (
    "tf_ds_fused_prompt_idx_key_kernel",
    "tf_ds_fused_idx_key_kernel",
    "tf_ds_fused_rope_store_kernel",
    "tf_ds_fused_prompt_latent_kernel",
    "tf_ds_fused_rope_qd_kernel",
    "tf_ds_compress2_norm_kernel",
    "tf_ds_idx_fused_kernel",
)
OPS = {"tf_ds_idx_fused_kernel"}
PATTERNS = {
    "BAR": r"\bBAR\.SYNC\b",
    "LDS": r"\bLDS\b",
    "STS": r"\bSTS\b",
    "SHFL": r"\bSHFL\b",
    "LDG": r"\bLDG\b",
    "STG": r"\bSTG\b",
    "MUFU": r"\bMUFU\b",
    "IADD3": r"\bIADD3\b",
}


def sections(text):
    """Split cuobjdump -sass output into {kernel name: [instructions]}."""
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
    """ptxas resource usage per function: {name: {REG: n, STACK: n, SHARED: n, LOCAL: n}}."""
    out, name = {}, None
    for line in text.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            name = m.group(1).removeprefix("tf_ds_").removesuffix("_kernel")
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m and name:
            out.setdefault(name, []).append(dict(zip(("REG", "STACK", "SHARED", "LOCAL"), map(int, m.groups()))))
    return out


def main(bin_dir):
    bin_dir = Path(bin_dir)
    for version in ("old", "new"):
        print(f"===== {version}")
        dumps = {}
        for cubin in ("ops", "torch"):
            path = bin_dir / f"{version}-{cubin}.cubin"
            sass = subprocess.run([CUOBJDUMP, "-sass", str(path)], capture_output=True, text=True).stdout
            use = subprocess.run([CUOBJDUMP, "-res-usage", str(path)], capture_output=True, text=True).stdout
            dumps.update(sections(sass))
            for name, entries in usage(use).items():
                dumps.setdefault(name, [])
                dumps.setdefault("_usage", {})[name] = entries
        print(f"{'kernel':<38} " + " ".join(f"{k:>6}" for k in PATTERNS))
        for name in WANT:
            body = dumps.get(name)
            if body is None:
                continue
            counts = {k: sum(1 for ins in body if re.search(p, ins)) for k, p in PATTERNS.items()}
            print(f"{name.replace('tf_ds_', ''):<38} " + " ".join(f"{counts[k]:>6}" for k in PATTERNS))
        print("-- ptxas usage (REG STACK SHARED LOCAL) per function, one line per variant")
        for name in WANT:
            short = name.removeprefix("tf_ds_").removesuffix("_kernel")
            for entry in dumps.get("_usage", {}).get(short, []):
                print(f"  {short:<36} " + " ".join(f"{k}={v}" for k, v in entry.items()))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "bin")
