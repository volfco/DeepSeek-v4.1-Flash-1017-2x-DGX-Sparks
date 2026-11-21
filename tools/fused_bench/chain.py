#!/usr/bin/env python3
"""WP4 chain harness: one decode layer's fused kernels back to back in one CUDA graph, with and without PDL.

Extends tools/fused_bench (bench.py is imported, not copied): the launch plumbing, buffers, the byte comparator and
the guarded CUDA-graph timing are bench.py's; this file only builds the layer chain and the canaries.

Modes:
  chain   old/new cubins x pdl on/off, each one graph of `batch` replays of the whole decode-layer chain in serving
          order (0197 compressor -> 0198 index key -> 0198 latent -> 0198 query -> 0195 indexer -> 0199 prune), plus
          the same number of empty kernel nodes (plain and with the PDL attribute) as the launch floor.
  canary  per fused kernel: a delayed producer writes the kernel's input, then the kernel runs as its PDL dependent.
          A pre-wait read sees the stale bytes and the output differs; the reference has no delay.
  parity  graph vs non-graph and old vs new byte parity of the whole chain (every live output compared).

Run on the head, under ~/tf-bench/gpu.lock (bench.py's guard does that), one process at a time:
  python3 chain.py --mode chain --shape "R=7 nb=8192 k=2048"
"""
import argparse
import ctypes as ct
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench as b  # noqa: E402

L, P, F = b.L, b.P, b.F
rng = b.rng

SHAPES = [  # (rows, nb, k) at the served decode row counts, the small pool and the 16K-65K window
    (1, 8, 8), (7, 8, 8), (32, 8, 8), (48, 8, 8),
    (7, 4096, 2048), (32, 8192, 2048), (48, 8192, 2048),
]


def parse_shape(text):
    fields = dict(kv.split("=") for kv in text.split())
    return int(fields["R"]), int(fields["nb"]), int(fields["k"])


# ---------------------------------------------------------------------------------------------------------------------
# one decode layer's fused launches, in the serving order round.zig runs them

class Layer:
    def __init__(self, rows, nb, k):
        self.rows, self.nb, self.k = rows, nb, k
        d = 512
        idim = 128
        heads = 32
        width = 8 * nb - 3
        n = 2 * rows
        self.scores = b.Buffer(rng.normal(size=(rows, 8 * nb)).astype("f4"))
        self.vis = b.Buffer(np.full(rows, width, dtype="i8"))
        self.every = b.Buffer(np.full(rows, 1 << 40, dtype="i8"))
        self.mask = b.out((rows, nb + 7), "u1")
        # 0197 decode compressor
        self.cx = b.Buffer(rng.normal(size=(rows, d)).astype("f4"))
        self.cgate = b.Buffer(rng.normal(size=(rows, d)).astype("f4"))
        self.slot = b.Buffer(rng.permutation(n)[:rows].astype("i8"))
        self.gi = b.Buffer(np.column_stack((np.arange(rows), np.arange(rows) + rows)).astype("i8"))
        self.rk = b.Buffer(rng.normal(size=(n, d)).astype("f4"))
        self.rg = b.Buffer(rng.normal(size=(n, d)).astype("f4"))
        self.flag = b.Buffer(np.zeros(1, "u4"))
        self.cw = b.Buffer(b.bf(d))
        self.lat2 = b.out((rows, d + 16), "u2")
        # 0198 index key, latent, query
        self.kx = b.Buffer(b.bf((rows, idim)))
        self.kw = b.Buffer(b.bf(idim))
        self.ikp = b.out((rows, idim), "u2")
        self.ik = b.Buffer(b.bf((rows, idim)))
        self.cos = b.Buffer(rng.uniform(-1, 1, (rows, 32)).astype("f4"))
        self.sin = b.Buffer(rng.uniform(-1, 1, (rows, 32)).astype("f4"))
        self.pos = b.Buffer(rng.permutation(rows).astype("i8"))
        self.dest = b.Buffer(rng.permutation(rows).astype("i8"))
        self.icodes = b.out((rows, idim // 2), "u1")
        self.iscales = b.out((rows, idim // 32), "u1")
        self.lat = b.Buffer(b.bf((rows, d)))
        self.ccodes = b.out((rows, d // 2), "u1")
        self.cscales = b.out((rows, d // 16), "u1")
        self.qx = b.Buffer(b.bf((rows, heads, idim)))
        self.qdq = b.out((rows, heads, idim), "u2")
        # 0199 prune
        self.pick = b.Buffer(rng.integers(0, 512, (rows, 7), dtype="i4"))
        self.wts = b.Buffer((rng.random((rows, 7)) + 0.01).astype("f4"))
        self.stats = b.Buffer(np.zeros(6, "u8"))
        self.prune = b.TfDsPrune(0.85, 3, 1, 1, 6, 7, rows, 0, 0)
        self.shared = nb * 4
        self.bmax = b.out((rows, nb), "f4")
        self.blocks = b.out((rows, k), "i4")
        # every buffer a node reads or writes in place: restored before each arm, so the arms start from the same state
        self.pristine = {}
        for name in ("scores", "vis", "every", "cx", "cgate", "slot", "gi", "rk", "rg", "flag", "kx", "kw", "ik",
                     "cos", "sin", "pos", "dest", "lat", "qx", "pick", "wts", "stats"):
            buf = getattr(self, name)
            data = np.array(buf.read())
            self.pristine[name] = (b.Buffer(data), int(data.nbytes))

    def reset(self):
        for name, (src, n) in self.pristine.items():
            b.cu("cuMemcpyDtoDAsync_v2", getattr(self, name).addr, src.addr, ct.c_size_t(n), b.stream)
        b.cu("cuStreamSynchronize", b.stream)

    def plan(self, mod, ops, pdl):
        """The layer's fused launches in serving order: (label, grid, block, shared, callable). The indexer takes the
        small-row form the engine dispatches at rows <= 7 with nbk >= 6144 (block_max + idx_pick), else the one-launch
        fused kernel."""
        s, r = self, self.rows
        steps = [
            ("0197-compress", r, 512, 0,
             lambda: b.native(mod, "compress2x_norm512", r, 512, s.cx, L(512), s.cgate, L(512), s.slot, s.rk, s.rg,
                              s.gi, L(2 * r), s.cw, s.lat2, L(528), r, 512, b.eps, s.flag, pdl=pdl)),
            ("0198-key", r, 128, 0,
             lambda: b.native(mod, "fused_idx_key", r, 128, s.kx, L(128), s.kw, s.cos, s.sin, s.pos, s.icodes,
                              s.iscales, s.dest, 128, 64, b.eps, pdl=pdl)),
            ("0198-latent", r, 256, 0,
             lambda: b.native(mod, "fused_rope_store", r, 256, s.lat, s.cos, s.sin, s.pos, s.ccodes, s.cscales,
                              s.dest, 512, 64, pdl=pdl)),
        ]
        if r >= 32:
            steps.append(("0198-query", (r, 4), 256, 0,
                          lambda: b.native(mod, "fused_rope_qd_pair", (r, 4), 256, s.qx, s.qdq, s.cos, s.sin, s.pos,
                                           32, 128, 64, pdl=pdl)))
        else:
            steps.append(("0198-query", (r, 32), 128, 0,
                          lambda: b.native(mod, "fused_rope_qd", (r, 32), 128, s.qx, s.qdq, s.cos, s.sin, s.pos, 32,
                                           128, 64, pdl=pdl)))
        if r <= 7 and self.nb >= 6144:
            steps.append(("0195-block-max", ((self.nb + 255) // 256, r), 256, 0,
                          lambda: b.native(ops, "block_max", ((self.nb + 255) // 256, r), 256, s.scores, L(8 * self.nb),
                                           8 * self.nb - 3, 8, s.vis, s.bmax, L(self.nb), self.nb, pdl=pdl)))
            steps.append(("0195-idx-pick", r, 1024, self.shared,
                          lambda: b.native(ops, "idx_pick", r, 1024, s.bmax, L(self.nb), self.nb, s.k, s.every,
                                           s.mask, shared=self.shared, pdl=pdl)))
        else:
            steps.append(("0195-idx", r, 1024, self.shared,
                          lambda: b.native(ops, "idx_fused", r, 1024, s.scores, L(8 * self.nb), 8 * self.nb - 3, 8,
                                           s.vis, s.every, s.k, s.mask, L(self.nb + 7), P(), shared=self.shared,
                                           pdl=pdl)))
        steps.append(("0199-prune", 1, 32 if r <= 32 else 64, 0,
                      lambda: b.native(ops, "prune_six", 1, 32 if r <= 32 else 64, s.pick, s.wts, s.prune, s.stats,
                                       P(), pdl=pdl)))
        return steps


def arm(plan):
    def run():
        for step in plan:
            step[4]()
    return run


def empty_arm(plan, pdl):
    def run():
        for label, grid, block, shared, _ in plan:
            b.launch(b.opts.floor, "empty_kernel", grid, block, shared, pdl=pdl)
    return run


# ---------------------------------------------------------------------------------------------------------------------
# modes

def mode_chain(rows, nb, k):
    layer = Layer(rows, nb, k)
    label = f"chain R={rows} nb={nb} k={k}"
    plan_new = layer.plan(b.opts.new_torch, b.opts.new_ops, 0)
    chains = [arm(layer.plan(b.opts.old_torch, b.opts.old_ops, 0)), arm(layer.plan(b.opts.old_torch, b.opts.old_ops, 1)),
              arm(layer.plan(b.opts.new_torch, b.opts.new_ops, 0)), arm(layer.plan(b.opts.new_torch, b.opts.new_ops, 1)),
              empty_arm(plan_new, 0), empty_arm(plan_new, 1)]
    names = ["old", "oldpdl", "new", "newpdl", "empty", "emptypdl"]
    for c in chains:
        c()
    b.cu("cuStreamSynchronize", b.stream)
    timing = b.time_graphs(label, chains, names, None, b.opts.batch, b.opts.reps)
    return {"kernel": "wp4-chain", "shape": label, "rows": rows, "nb": nb, "k": k, "timing": timing}


def mode_parity(rows, nb, k):
    """Graph and non-graph byte parity of the chain, old vs new, pdl on and off: every live output of every node."""
    layer = Layer(rows, nb, k)
    label = f"chain-parity R={rows} nb={nb} k={k}"
    live = [layer.lat2, layer.flag, layer.icodes, layer.iscales, layer.ccodes, layer.cscales, layer.qdq, layer.mask,
            layer.pick, layer.wts, layer.stats, layer.rk, layer.rg]
    rows_out = []
    for graph in (False, True):
        states = {}
        for tag, mod, pdl in (("old", (b.opts.old_torch, b.opts.old_ops), 0),
                              ("oldpdl", (b.opts.old_torch, b.opts.old_ops), 1),
                              ("new", (b.opts.new_torch, b.opts.new_ops), 0),
                              ("newpdl", (b.opts.new_torch, b.opts.new_ops), 1)):
            layer.reset()
            nodes = layer.plan(mod[0], mod[1], pdl)
            if graph:
                b.cu("cuStreamBeginCapture_v2", b.stream, 0)
                arm(nodes)()
                g, e = P(), P()
                b.cu("cuStreamEndCapture", b.stream, ct.byref(g))
                b.cu("cuGraphInstantiateWithFlags", ct.byref(e), g, ct.c_uint64(0))
                b.cu("cuGraphDestroy", g)
                b.cu("cuGraphLaunch", e, b.stream)
            else:
                arm(nodes)()
            b.cu("cuStreamSynchronize", b.stream)
            states[tag] = [np.array(x.read()) for x in live]
            if graph:
                b.cu("cuGraphExecDestroy", e)
        diffs = {tag: int(sum(np.count_nonzero(a != s) for a, s in zip(states[tag], states["old"])))
                 for tag in states if tag != "old"}
        diffs["graph_vs_stream"] = int(sum(np.count_nonzero(a != s)
                                           for a, s in zip(states["newpdl"], states["new"])))
        for tag, n in diffs.items():
            if n:
                b.failures.append(f"{label} {'graph' if graph else 'stream'} {tag}: {n} bytes differ")
        rows_out.append({"graph": graph, "diffs": diffs})
        print(f"{label} {'graph' if graph else 'stream'}: {diffs}", flush=True)
    return {"kernel": "wp4-chain-parity", "shape": label, "rows": rows, "nb": nb, "parity": rows_out}


def canary_kernels(layer):
    """(label, launch(input-override, pdl), the buffer the delayed producer fills, that buffer's live shape)."""
    s = layer
    r = s.rows
    return [
        ("0197-compress", lambda pdl: b.native(b.opts.new_torch, "compress2x_norm512", r, 512, s.cx, L(512),
                                               s.cgate, L(512), s.slot, s.rk, s.rg, s.gi, L(2 * r), s.cw,
                                               s.lat2, L(528), r, 512, b.eps, s.flag, pdl=pdl),
         [s.cx, s.cgate], s.lat2),
        ("0198-key", lambda pdl: b.native(b.opts.new_torch, "fused_idx_key", r, 128, s.kx, L(128), s.kw, s.cos,
                                          s.sin, s.pos, s.icodes, s.iscales, s.dest, 128, 64, b.eps, pdl=pdl),
         [s.kx], s.icodes),
        ("0198-latent", lambda pdl: b.native(b.opts.new_torch, "fused_rope_store", r, 256, s.lat, s.cos, s.sin,
                                             s.pos, s.ccodes, s.cscales, s.dest, 512, 64, pdl=pdl),
         [s.lat], s.ccodes),
        ("0198-query", lambda pdl: b.native(b.opts.new_torch, "fused_rope_qd_pair", (r, 4), 256, s.qx, s.qdq, s.cos,
                                            s.sin, s.pos, 32, 128, 64, pdl=pdl) if r >= 32 else
         b.native(b.opts.new_torch, "fused_rope_qd", (r, 32), 128, s.qx, s.qdq, s.cos, s.sin, s.pos, 32, 128, 64,
                  pdl=pdl), [s.qx], s.qdq),
        ("0195-idx", lambda pdl: b.native(b.opts.new_ops, "idx_fused", r, 1024, s.scores, L(8 * s.nb), 8 * s.nb - 3,
                                          8, s.vis, s.every, s.k, P(), 0, s.blocks, shared=s.shared, pdl=pdl),
         [s.scores, s.vis], s.blocks),
        ("0199-prune", lambda pdl: b.native(b.opts.new_ops, "prune_six", 1, 32 if r <= 32 else 64, s.pick, s.wts,
                                            s.prune, s.stats, P(), pdl=pdl), [s.pick, s.wts], s.pick),
    ]


def mode_canary(rows, nb, k, delay_us):
    """Each fused kernel as the PDL dependent of delayed producers that land its real inputs.

    Every input starts as its real bytes' complement, so any read before griddepcontrol.wait returns different data.
    reference: the kernel on the real inputs, no producer. canary: complement -> a delayed copy of the real bytes into
    every input -> the kernel (pdl on and off); the live output must equal the reference. race control: the same
    producers on a side stream with no dependency, so the kernel does read pre-landing bytes: the output must differ,
    which proves the canary is sensitive for this kernel.
    """
    layer = Layer(rows, nb, k)
    label = f"canary R={rows} nb={nb} k={k} delay={delay_us}us"
    cycles = delay_us * 1000
    out = []
    for name, launch, inputs, live in canary_kernels(layer):
        pristine = [np.array(x.read()) for x in inputs]
        sizes = [p.nbytes for p in pristine]
        realb = [b.Buffer(p) for p in pristine]
        other = [b.Buffer(p ^ 0xFF) for p in pristine]

        def restore(arm):
            for x, src, n in zip(inputs, realb if arm == "real" else other, sizes):
                b.cu("cuMemcpyDtoDAsync_v2", x.addr, src.addr, ct.c_size_t(n), b.stream)
            b.cu("cuStreamSynchronize", b.stream)

        def produce(on_stream):
            for x, src, n in zip(inputs, realb, sizes):
                b.launch(b.opts.floor, "delay_copy", 64, 256, 0, src, x, L(n // 16), b.U64(cycles),
                         on_stream=on_stream)

        for pdl in (0, 1):
            restore("real")
            launch(pdl)
            b.cu("cuStreamSynchronize", b.stream)
            ref = np.array(live.read())
            restore("other")
            produce(b.stream)
            launch(pdl)
            b.cu("cuStreamSynchronize", b.stream)
            canary = int(np.count_nonzero(np.array(live.read()) != ref))
            restore("other")
            produce(b.side)
            launch(pdl)
            b.cu("cuStreamSynchronize", b.stream)
            b.cu("cuStreamSynchronize", b.side)
            raced = int(np.count_nonzero(np.array(live.read()) != ref))
            if canary:
                b.failures.append(f"{label} {name} pdl={pdl}: {canary} bytes differ (a read before the wait)")
            if not raced:
                b.failures.append(f"{label} {name} pdl={pdl}: race control insensitive (0 bytes differ)")
            print(f"{label} {name} pdl={pdl}: canary_diff={canary} race_control_diff={raced}", flush=True)
            out.append({"kernel": name, "pdl": pdl, "canary_diff": canary, "race_control_diff": raced})
    return {"kernel": "wp4-canary", "shape": label, "rows": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old", type=Path, required=True, help="directory with old-*.cubin")
    parser.add_argument("--new", type=Path, required=True, help="directory with new-*.cubin")
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--mode", default="chain", choices=("chain", "parity", "canary"))
    parser.add_argument("--shape", default=None, help='one shape: "R=7 nb=8192 k=2048"')
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--delay-us", type=int, default=200)
    parser.add_argument("--no-guard", action="store_true")
    parser.add_argument("--wait", type=int, default=1800)
    parser.add_argument("--tries", type=int, default=20)
    opts = parser.parse_args()
    b.opts = opts
    opts.old_ops, opts.old_torch = opts.old / "old-ops.cubin", opts.old / "old-torch.cubin"
    opts.new_ops, opts.new_torch = opts.new / "new-ops.cubin", opts.new / "new-torch.cubin"
    opts.floor = opts.new / "floor.cubin"
    opts.kernel = "all"

    lock = None
    if not opts.no_guard:
        avail = b.mem_available_gib()
        if avail < 3.0:
            sys.exit(f"MemAvailable {avail:.2f} GiB < 3 GiB, refusing")
        print(f"MemAvailable {avail:.2f} GiB", flush=True)
        lock = open(Path.home() / "tf-bench/gpu.lock", "w")
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX)
        h0 = b.wait_health(opts.wait)
        print(f"health before: {h0.get('live')}", flush=True)
        counters0 = {k: h0.get(k) for k in ("prompt_tokens_total", "generation_tokens_total")}

    b.cu("cuInit", 0)
    ctx = P()
    b.cu("cuDevicePrimaryCtxRetain", ct.byref(ctx), 0)
    b.cu("cuCtxSetCurrent", ctx)
    b.cu("cuStreamCreate", ct.byref(b.stream), 1)
    b.side = P()
    b.cu("cuStreamCreate", ct.byref(b.side), 1)
    b.DO_PARITY = True
    b.DO_TIMING = opts.mode == "chain"

    shapes = [parse_shape(opts.shape)] if opts.shape else SHAPES
    rows_out = []
    try:
        for rows, nb, k in shapes:
            if opts.mode == "chain":
                rows_out.append(mode_chain(rows, nb, k))
            elif opts.mode == "parity":
                rows_out.append(mode_parity(rows, nb, k))
            else:
                rows_out.append(mode_canary(rows, nb, k, opts.delay_us))
            b.free_all()
    finally:
        b.free_all()
        if lock is not None:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_UN)
    if not opts.no_guard:
        h1 = b.health_now()
        print(f"health after: {h1.get('live')} retries={b.timing_retries}", flush=True)
    if opts.json:
        opts.json.write_text(json.dumps({"rows": rows_out, "failures": b.failures}, indent=1, default=str))
    if b.failures:
        for f in b.failures[:20]:
            print("  " + f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
