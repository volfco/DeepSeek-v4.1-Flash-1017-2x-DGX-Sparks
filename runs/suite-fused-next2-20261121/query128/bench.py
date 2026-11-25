#!/usr/bin/env python3
"""Compare the shipped fused query kernel with the deployed kit's prompt fallback."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools/fused_bench"))
import bench as cuda

ROWS, HEADS, HD, RD = 128, 32, 128, 64
SHAPE = (ROWS, HEADS, HD)
ARG_COUNTS = {"tf_ds_fused_rope_qd_pair_kernel": 9, "_rope_heads": 6, "_fp4qd_p2": 5}
FUSED_SYMBOL = "tf_ds_fused_rope_qd_pair_kernel"
ARMS = ("query_pair", "kit")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def embedded_fatbin(binary, name="fatbin_dsv41_torch"):
    symbols = subprocess.check_output(["nm", "-S", str(binary)], text=True)
    symbol, = (line for line in symbols.splitlines() if f'"{name}"' in line)
    address, size = (int(field, 16) for field in symbol.split()[:2])
    segments = subprocess.check_output(["readelf", "-lW", str(binary)], text=True)
    for line in segments.splitlines():
        fields = line.split()
        if not fields or fields[0] != "LOAD":
            continue
        offset, virtual, _, length = (int(field, 16) for field in fields[1:5])
        if virtual <= address and address + size <= virtual + length:
            with binary.open("rb") as f:
                f.seek(offset + address - virtual)
                return f.read(size)
    raise RuntimeError(f"embedded fatbin lies outside file-backed segments in {binary}")


def variant(manifest, name, constants, alignment):
    spec, = (
        k for k in manifest["kernels"]
        if k["fn"] == name
        and {key: value["int"] for key, value in k["consts"].items()} == constants
        and {p["name"]: p["div16"] for p in k["params"]} == alignment
    )
    return spec


def ready(previous=None):
    health = cuda.health_now()
    counters = cuda.health_counters(health)
    if not cuda.check_idle_health(health) or health["live"]["waiting"] != 0:
        raise RuntimeError("production must be idle with valid health for the benchmark")
    if previous is not None and counters != previous:
        raise RuntimeError("production counters changed during the benchmark")
    return counters


def graph(chain, slots):
    start, stop = cuda.P(), cuda.P()
    for event in (start, stop):
        cuda.cu("cuEventCreate", cuda.ref(event), 0)
    cuda.cu("cuStreamBeginCapture_v2", cuda.stream, 0)
    # CU_EVENT_RECORD_EXTERNAL creates timestamp nodes inside the captured graph.
    cuda.cu("cuEventRecordWithFlags", start, cuda.stream, 1)
    for x, out in slots:
        chain(x, out)
    cuda.cu("cuEventRecordWithFlags", stop, cuda.stream, 1)
    g, exe = cuda.P(), cuda.P()
    cuda.cu("cuStreamEndCapture", cuda.stream, cuda.ref(g))
    cuda.cu("cuGraphInstantiateWithFlags", cuda.ref(exe), g, cuda.U64(0))
    cuda.cu("cuGraphDestroy", g)
    return exe, start, stop


def summarize(pairs):
    a, b = np.array(pairs).T
    d = a - b
    rng = np.random.default_rng(20261121)
    # Resample complete pairs so the interval preserves local clock and load variation.
    bootstrap = np.median(rng.choice(d, (10000, len(d))), axis=1)

    def summary(x):
        return {
            "median": float(np.median(x)),
            "iqr": (np.percentile(x, 75) - np.percentile(x, 25)).item(),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    return {
        "query_pair_us": summary(a),
        "kit_us": summary(b),
        "paired_difference_us": summary(d),
        "paired_difference_bootstrap_95_us": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "split_faster_pairs": int(np.count_nonzero(d > 0)),
    }


def measure(chains, pristine, src, counters, batch, reps, reuse=False):
    slots = [(cuda.Buffer(src), cuda.out(src.shape, "u2")) for _ in range(1 if reuse else batch)]
    graph_slots = slots * batch if reuse else slots
    exes = {arm: graph(chain, graph_slots) for arm, chain in chains.items()}

    def replay(arm):
        # The kit's RoPE overwrites X. Both graphs use identical input and output addresses.
        # Restore all slots on the same stream before entering the measured graph.
        ready(counters)
        for x, _ in slots:
            cuda.cu("cuMemcpyDtoDAsync_v2", x.addr, pristine.addr, cuda.L(src.nbytes), cuda.stream)
        exe, start, stop = exes[arm]
        cuda.cu("cuGraphLaunch", exe, cuda.stream)
        cuda.cu("cuStreamSynchronize", cuda.stream)
        elapsed = cuda.F()
        cuda.cu("cuEventElapsedTime", cuda.ref(elapsed), start, stop)
        return elapsed.value * 1000 / batch

    try:
        for _ in range(4):
            for arm in ARMS:
                replay(arm)
        pairs = []
        for i in range(reps):
            order = ARMS if i % 2 == 0 else ARMS[::-1]
            pair = {arm: replay(arm) for arm in order}
            pairs.append([pair[arm] for arm in ARMS])
        replay("query_pair")
        expected = [out.read() for _, out in slots]
        replay("kit")
        diffs = sum(int(np.count_nonzero(out.read() != gold)) for (_, out), gold in zip(slots, expected))
        if diffs:
            raise RuntimeError(f"captured graph output mismatch: {diffs} bytes")
        ready(counters)
        return {
            "graph_batch": batch, "input_slots": len(slots), "n_pairs": reps, "arm_order": list(ARMS),
            "angles": "identity" if reuse else "seeded",
            "paired_us": pairs, "graph_output_mismatched_bytes": diffs, **summarize(pairs),
        }
    finally:
        for exe, start, stop in exes.values():
            cuda.cu("cuGraphExecDestroy", exe)
            for event in (start, stop):
                cuda.cu("cuEventDestroy_v2", event)
        for x, out in slots:
            for buffer in (x, out):
                cuda.cu("cuMemFree_v2", buffer.addr)
                cuda.allocations.remove(buffer.addr)


def run(opts, fatbin, kit_specs):
    available = cuda.mem_available_gib()
    if available < 3:
        raise RuntimeError(f"MemAvailable {available:.2f} GiB < 3 GiB")
    counters = ready()
    cuda.cu("cuInit", 0)
    ctx = cuda.P()
    cuda.cu("cuDevicePrimaryCtxRetain", cuda.ref(ctx), 0)
    cuda.cu("cuCtxSetCurrent", ctx)
    cuda.cu("cuStreamCreate", cuda.ref(cuda.stream), 1)
    rng = np.random.default_rng(197)
    values = rng.standard_normal(SHAPE, dtype=np.float32).view("u4")
    src = ((values + 0x7fff + ((values >> 16) & 1)) >> 16).astype("u2")
    angles = rng.uniform(-np.pi, np.pi, (1024 + ROWS, RD // 2))
    x = cuda.Buffer(src)
    out = cuda.out(SHAPE, "u2")
    pristine = cuda.Buffer(src)
    cos, sin = [cuda.Buffer(trig(angles).astype("f4")) for trig in (np.cos, np.sin)]
    pos = cuda.Buffer(np.arange(1024, 1024 + ROWS, dtype="i8"))
    nblk = src.size // 32
    assert all(buffer.addr.value % 16 == 0 for buffer in (x, out, cos, sin, pos))

    def fused(x, out):
        cuda.native(fatbin, "fused_rope_qd_pair", (ROWS, HEADS // 8), 256, x, out, cos, sin, pos, HEADS, HD, RD)

    def aot(spec, grid, *args):
        path = opts.aot / "cubins" / (spec["hash"] + ".cubin")
        cuda.launch(path, spec["name"], grid, spec["num_warps"] * 32, spec["shared"], *args,
                    cuda.P(), cuda.P(), pdl=spec["pdl"])

    def kit(x, out):
        aot(kit_specs[0], (ROWS, HEADS // 16), x, cos, sin, pos)
        aot(kit_specs[1], nblk // 32, x, out, nblk)

    loaded = [(fatbin, FUSED_SYMBOL)] + [
        (opts.aot / "cubins" / (s["hash"] + ".cubin"), s["name"]) for s in kit_specs
    ]
    for path, symbol in loaded:
        expected = ARG_COUNTS[symbol]
        actual = cuda.param_count(path, symbol)
        if actual != expected:
            raise RuntimeError(f"{symbol}: expected {expected} arguments, got {actual}")

    chains = {"query_pair": fused, "kit": kit}
    try:
        parity = {}
        edge = src.copy()
        edge[..., :8] = [0x7FC1, 0x7F80, 0xFF80, 0x8000, 0, 1, 0x7F7F, 0x8001]
        for name, data in (("seeded", src), ("special", edge)):
            ready(counters)
            trial = cuda.Buffer(data)
            fused(trial, out)
            expected = out.read()
            kit(trial, out)
            actual = out.read()
            diffs = int(np.count_nonzero(actual != expected))
            if diffs:
                raise RuntimeError(f"{name} output mismatch: {diffs} bytes")
            parity[name] = {"mismatched_bytes": diffs, "output_bytes": actual.size,
                            "input_sha256": digest(data.tobytes()), "output_sha256": digest(actual)}
        batches = (8, 4, 2, 1) if opts.reverse_batches else (1, 2, 4, 8)
        timings = [measure(chains, pristine, src, counters, batch, opts.reps) for batch in batches]
        # Identity rotation keeps the shared BF16 input constant across the hot graph's kit invocations.
        cos, sin = [cuda.Buffer(np.full(angles.shape, value, dtype="f4")) for value in (1, 0)]
        timings.append(measure(chains, pristine, src, counters, 8, opts.reps, reuse=True))
        return {
            "shape": {"rows": ROWS, "heads": HEADS, "hd": HD, "rd": RD},
            "geometry": {
                FUSED_SYMBOL: {"grid": [ROWS, HEADS // 8], "block": 256, "pdl": False},
                "_rope_heads": {"grid": [ROWS, HEADS // 16], "block": 128, "pdl": True},
                "_fp4qd_p2": {"grid": [nblk // 32, 1], "block": 128, "pdl": True},
            },
            "initial_memory_available_gib": available, "initial_health_counters": counters,
            "final_health_counters": ready(counters), "parity": parity, "timings": timings,
        }
    finally:
        cuda.free_all()
        cuda.cu("cuStreamDestroy_v2", cuda.stream)
        cuda.cu("cuDevicePrimaryCtxRelease_v2", 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--reps", default=80, type=int)
    parser.add_argument("--reverse-batches", action="store_true")
    opts = parser.parse_args()
    native = Path("/opt/tensorfold/native/bin/tensorfold-native")
    lanes = Path("/opt/tensorfold/bin/tf-dsv41-lanes")
    fatbin = embedded_fatbin(native)
    if fatbin != embedded_fatbin(lanes):
        raise RuntimeError("native and lanes share different query fatbins")
    config = json.loads(opts.config.read_text())["text_config"]
    actual = tuple(config[k] for k in ("index_n_heads", "index_head_dim", "qk_rope_head_dim"))
    if actual != (HEADS, HD, RD):
        raise RuntimeError(f"unexpected serving geometry: {actual}")
    manifest = json.loads((opts.aot / "aot.json").read_text())
    rope = variant(manifest, "_rope_heads", {"H": HEADS, "HD": HD, "RD": RD, "HB": 16, "INV": 0, "PDL": 1},
                   dict.fromkeys(("X", "COS", "SIN", "POS"), True))
    qd = variant(manifest, "_fp4qd_p2", {"BLKS": 32, "PDL": 1}, dict.fromkeys(("X", "OUT", "nblk"), True))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "dsv41_torch.fatbin"
        path.write_bytes(fatbin)
        result = run(opts, path, (rope, qd))
    result.update({
        "utc": datetime.now(timezone.utc).isoformat(),
        "image_id": opts.image_id,
        "native_sha256": digest(native.read_bytes()),
        "lanes_sha256": digest(lanes.read_bytes()),
        "embedded_fatbin_sha256": digest(fatbin),
        "model_config_sha256": digest(opts.config.read_bytes()),
        "kit": [
            {"variant_hash": spec["hash"], "file_sha256": digest(
                (opts.aot / "cubins" / (spec["hash"] + ".cubin")).read_bytes()), "manifest": spec}
            for spec in (rope, qd)
        ],
    })
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
