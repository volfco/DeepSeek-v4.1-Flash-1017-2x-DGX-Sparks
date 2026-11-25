"""0266 vs its shipped baseline on identical GB10 ratio-2 decode windows."""
import json
import runpy
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
helpers = runpy.run_path(str(ROOT / "runs/suite-fused-next2-20261121/query128/bench.py"))
cuda = helpers["cuda"]
ready = helpers["ready"]


def measure(module, d, windows, counters):
    rng = np.random.default_rng(266)
    raw = 64
    slots = np.concatenate([
        index * raw + (np.arange(length) + 62 + index) % raw
        for index, length in enumerate(windows)
    ]).astype("i8")
    rows = len(slots)
    ring_rows = len(windows) * raw
    keys = np.column_stack((slots & ~1, slots | 1))
    x, gates = [cuda.Buffer(rng.standard_normal((rows, d), dtype="f4")) for _ in range(2)]
    slot, pairs = cuda.Buffer(slots), cuda.Buffer(keys)
    weights = cuda.Buffer(np.full(d, 0x3f80, dtype="u2"))
    initial = [rng.standard_normal((ring_rows, d), dtype="f4") for _ in range(2)]
    graphs, outputs = [], []
    batch = 32
    # A second baseline on independent buffers measures the same-kernel timing spread.
    for suffix in ("", "_pair", ""):
        name = f"compress2x_norm{d}{suffix}"
        if cuda.param_count(module, f"tf_ds_{name}_kernel") != 17:
            raise RuntimeError(f"unexpected {name} signature")
        rk, rg = map(cuda.Buffer, initial)
        out = cuda.out((rows, d + 16), "u2")
        flag = cuda.Buffer(np.zeros(1, dtype="u4"))

        def chain(*unused):
            cuda.native(module, name, rows, d, x, cuda.L(d), gates, cuda.L(d), slot,
                        rk, rg, pairs, cuda.L(ring_rows), weights, out, cuda.L(d + 16),
                        rows, d, cuda.F(1e-20), flag)

        chain()
        outputs.append([b.read() for b in (rk, rg, out, flag)])
        graphs.append(helpers["graph"](chain, [(None, None)] * batch))
    if any(not np.array_equal(a, b) for state in outputs[1:]
           for a, b in zip(outputs[0], state, strict=True)):
        raise RuntimeError(f"ring, output or flag mismatch for {windows=} {d=}")
    timings = []
    for rep in range(26):
        ready(counters)
        sample = [0.] * len(graphs)
        for offset in range(len(graphs)):
            arm = (rep + offset) % len(graphs)
            exe, start, stop = graphs[arm]
            cuda.cu("cuGraphLaunch", exe, cuda.stream)
            cuda.cu("cuStreamSynchronize", cuda.stream)
            elapsed = cuda.F()
            cuda.cu("cuEventElapsedTime", cuda.ref(elapsed), start, stop)
            sample[arm] = elapsed.value * 1000 / batch
        if rep >= 2:
            timings.append(sample)
    ready(counters)
    for exe, start, stop in graphs:
        cuda.cu("cuGraphExecDestroy", exe)
        for event in (start, stop):
            cuda.cu("cuEventDestroy_v2", event)
    cuda.free_all()
    times = np.asarray(timings)
    return dict(d=d, windows=windows, unique_pairs=len(np.unique(keys, axis=0)),
                mismatched_bytes=0, arms=["baseline", "pair_once", "baseline_repeat"], samples_us=timings,
                median_us=np.median(times, axis=0).tolist(),
                paired_delta_us=float(np.median(times[:, 1] - times[:, 0])),
                baseline_repeat_delta_us=float(np.median(times[:, 2] - times[:, 0])))


def main():
    counters = ready()
    binary = Path("/opt/tensorfold/bin/tf-dsv41-lanes")
    fatbin = helpers["embedded_fatbin"](binary)
    cuda.cu("cuInit", 0)
    ctx = cuda.P()
    cuda.cu("cuDevicePrimaryCtxRetain", cuda.ref(ctx), 0)
    cuda.cu("cuCtxSetCurrent", ctx)
    cuda.cu("cuStreamCreate", cuda.ref(cuda.stream), 1)
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "torch.fatbin"
        path.write_bytes(fatbin)
        print(json.dumps({"fatbin_sha256": helpers["digest"](fatbin)}), flush=True)
        for windows in ([1], [5], [6], [7], [5, 6, 7, 5]):
            for d in (128, 512):
                ready(counters)
                print(json.dumps(measure(path, d, windows, counters)), flush=True)


if __name__ == "__main__":
    main()
