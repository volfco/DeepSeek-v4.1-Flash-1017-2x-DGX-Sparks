#!/usr/bin/env python3
"""Time the shipped 0267 fusion against the kit's bias + score on the TP2 geometry."""
import json
import runpy
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HELPERS = runpy.run_path(str(ROOT / "runs/suite-fused-next2-20261121/query128/bench.py"))
CUDA = HELPERS["cuda"]
READY = HELPERS["ready"]
FUSED = "tf_ds_markov_fused_kernel"
COLS, RANK, BLOCK, STRIDE = 64640, 256, 6, 7
TILES = (COLS + 1023) // 1024
ARMS = ("split", "fused", "bias", "score")


def kit_launch(aot, spec, grid, args):
    path = aot / "cubins" / (spec["hash"] + ".cubin")
    name = spec["name"]
    return lambda: CUDA.launch(path, name, grid, 32 * spec["num_warps"], spec["shared"],
                               *args, CUDA.P(), CUDA.P(), pdl=spec["pdl"])


def measure(aot, bias_spec, score_spec, native, weights, rows, misses, counters):
    rng = np.random.default_rng(267)
    tokens = np.zeros((rows, STRIDE), dtype="i8")
    tokens[:, 0] = np.arange(rows)
    tok = CUDA.Buffer(tokens)
    slot = CUDA.Buffer(np.array([-1] * misses + list(range(rows - misses)), dtype="i4"))
    cache = CUDA.Buffer(rng.standard_normal((rows, COLS), dtype="f4").astype("f2"))
    lg = CUDA.Buffer(rng.standard_normal((rows, BLOCK, COLS), dtype="f4"))
    head, emb = weights
    stage = CUDA.out((rows, COLS), "f2")
    split_value, fused_value = [CUDA.out((rows, TILES), "f4") for _ in range(2)]
    split_index, fused_index = [CUDA.out((rows, TILES), "i4") for _ in range(2)]
    segment_stride, row_stride = rows * BLOCK * COLS, BLOCK * COLS
    bias = kit_launch(aot, bias_spec, (COLS + 255) // 256,
                      (tok, STRIDE, slot, emb, head, stage, COLS, rows, COLS, 0))
    score = kit_launch(aot, score_spec, TILES,
                       (lg, segment_stride, row_stride, tok, STRIDE, slot, cache, COLS, stage, COLS,
                        split_value, split_index, rows, COLS, 0, TILES))

    def split():
        bias()
        score()

    def fused():
        CUDA.launch(native, FUSED, TILES, 256, 0, lg, CUDA.U64(segment_stride), CUDA.U64(row_stride),
                    tok, CUDA.U64(STRIDE), slot, cache, CUDA.U64(COLS), emb, head, RANK,
                    fused_value, fused_index, rows, COLS, 0, TILES, 0)

    split()
    fused()
    mismatch = sum(int(np.count_nonzero(a.read() != b.read()))
                   for a, b in [(split_value, fused_value), (split_index, fused_index)])
    if mismatch:
        raise RuntimeError(f"Markov tile parity failed for {rows=} {misses=}: {mismatch} bytes")
    batch = 16
    graphs = [HELPERS["graph"](lambda *_: chain(), [(None, None)] * batch)
              for chain in (split, fused, bias, score)]
    samples = []
    for rep in range(26):
        READY(counters)
        times = [0.] * len(graphs)
        for arm in (*range(rep % 4, 4), *range(0, rep % 4)):
            exe, start, stop = graphs[arm]
            CUDA.cu("cuGraphLaunch", exe, CUDA.stream)
            CUDA.cu("cuStreamSynchronize", CUDA.stream)
            elapsed = CUDA.F()
            CUDA.cu("cuEventElapsedTime", CUDA.ref(elapsed), start, stop)
            times[arm] = elapsed.value * 1000 / batch
        READY(counters)
        if rep >= 2:
            samples.append(times)
    for exe, start, stop in graphs:
        CUDA.cu("cuGraphExecDestroy", exe)
        for event in (start, stop):
            CUDA.cu("cuEventDestroy_v2", event)
    times = np.asarray(samples)
    return dict(rows=rows, misses=misses, mismatched_bytes=mismatch, arms=ARMS, samples_us=samples,
                median_us=dict(zip(ARMS, np.median(times, axis=0).tolist())),
                paired_fused_minus_split_us=float(np.median(times[:, 1] - times[:, 0])))


def attributes(path, name):
    fn = CUDA.function(path, name)
    values = {}
    for attribute, code in (("static_shared_bytes", 1), ("local_bytes_per_thread", 3), ("registers_per_thread", 4)):
        result = CUDA.I()
        CUDA.cu("cuFuncGetAttribute", CUDA.ref(result), code, fn)
        values[attribute] = result.value
    return values


def main():
    counters = READY()
    aot = Path("/artifacts/kit-abe07c7d/kit/aot")
    manifest = json.loads((aot / "aot.json").read_text())
    bias, score = (
        next(k for k in manifest["kernels"] if k["fn"] == name and k["consts"] == constants
             and all(p["div16"] for p in k["params"] if p["type"].startswith("*")))
        for name, constants in (
            ("_bias", {key: {"int": value} for key, value in
                       {"BN": 64, "SUB": 4, "RANK": RANK, "NP": 16, "PDL": 1}.items()}),
            ("_score", {"BC": {"int": 1024}, "PDL": {"int": 1}}),
        )
    )
    CUDA.cu("cuInit", 0)
    ctx = CUDA.P()
    CUDA.cu("cuDevicePrimaryCtxRetain", CUDA.ref(ctx), 0)
    CUDA.cu("cuCtxSetCurrent", ctx)
    CUDA.cu("cuStreamCreate", CUDA.ref(CUDA.stream), 1)
    with tempfile.TemporaryDirectory() as scratch:
        binary = Path("/opt/tensorfold/bin/tf-dsv41-lanes")
        fatbin = HELPERS["embedded_fatbin"](binary, "fatbin_dsv41_ops")
        native = Path(scratch) / "ops.fatbin"
        native.write_bytes(fatbin)
        for path, symbol, argc in ((native, FUSED, 18),
                                   (aot / "cubins" / (bias["hash"] + ".cubin"), "_bias", 12),
                                   (aot / "cubins" / (score["hash"] + ".cubin"), "_score", 18)):
            if CUDA.param_count(path, symbol) != argc:
                raise RuntimeError(f"unexpected {symbol} signature")
        print(json.dumps(dict(fatbin_sha256=HELPERS["digest"](fatbin), bias_hash=bias["hash"],
                              score_hash=score["hash"], kit_pdl=True, native_pdl=False,
                              fused_attributes=attributes(native, FUSED))), flush=True)
        rng = np.random.default_rng(26)
        head = rng.integers(-16, 17, size=(COLS, RANK), dtype="i2").astype("f2") / np.float16(32)
        emb = rng.integers(-16, 17, size=(4, RANK), dtype="i2").astype("f4") / 32
        for rows in (1, 2, 4):
            for misses in range(rows + 1):
                READY(counters)
                weights = (CUDA.Buffer(head), CUDA.Buffer((emb.view("u4") >> 16).astype("u2")))
                print(json.dumps(measure(aot, bias, score, native, weights, rows, misses, counters)), flush=True)
                CUDA.free_all()


if __name__ == "__main__":
    main()
