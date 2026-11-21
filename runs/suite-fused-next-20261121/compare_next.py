"""Tabulate suite-fused-20261121: production (base, blocks s01+s04) against production + fused 0193-0199 (fused,
blocks s02+s03), and both against the last suite run (runs/suite-20261119: s02 = production's switches incl. the lead
reader, s01 = the same without it, production-after = the probe on the restored pair). Run from the recipe root:

  python3 -I runs/suite-fused-20261121/compare.py

Prefill: uncached prompt tokens over the engine's prefill seconds (done line), median over every clean repetition in
an arm. Decode: aggregate tok/s per repetition (same prompts in every block) and summed decode s over summed rounds.
Probe: the engine's decode rate, exact tokens a round, reply hashes.
"""
import glob
import json
import os
import statistics as st

S = "runs/suite-fused-next-20261121"
P = "runs/suite-fused-20261121"


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


blocks = {}
for f in sorted(glob.glob(f"{S}/blocks/s[0-9][0-9].json")):
    d = json.load(open(f))
    blocks[d["id"]] = d
arms = {"base": [d for d in blocks.values() if d["arm"] == "base"], "fused": [d for d in blocks.values() if d["arm"] in ("next", "next_pdl")]}
prev_lead, prev_base = load(f"{P}/blocks/s02.json"), load(f"{P}/blocks/s01.json")
prev_after = load(f"{P}/blocks/production-after.probe.json")

print("blocks:", {b: f"{d['arm']} status={d.get('status')} cells={len(d['cells'])}/{len(d['cells_planned'])}" for b, d in blocks.items()})


def clean(rep):
    c = [a for a in rep if a.get("clean")]
    return c[-1] if c else None


def reps(d, cell, key=None):
    v = d["cells"].get(cell)
    if v is None:
        return []
    if key is not None:
        v = v.get(key, []) if isinstance(v, dict) else []
    return [clean(r) for r in v if clean(r)]


def prefill(ds, cell, size):
    out = []
    for d in ds:
        for a in reps(d, cell, size):
            for r in a["results"]:
                dl = d.get("done_lines", {}).get(r["id"])
                if dl and float(dl["prefill_s"]) > 0:
                    out.append((int(dl["prompt"]) - int(dl["cached"])) / float(dl["prefill_s"]))
    return out


def decode(ds, cell, key=None):
    return [a["aggregate_tok_s"] for d in ds for a in reps(d, cell, key) if "aggregate_tok_s" in a]


def round_ms(ds, cell, key=None):
    sec = rnd = tok = 0.0
    for d in ds:
        for a in reps(d, cell, key):
            for r in a["results"]:
                dl = d.get("done_lines", {}).get(r["id"])
                if dl and float(dl.get("rounds", 0)) > 0:
                    sec += float(dl["decode_s"]); rnd += float(dl["rounds"]); tok += float(dl["tokens"])
    return (1000 * sec / rnd, tok / rnd) if rnd else (float("nan"), float("nan"))


def shas(d, cell, key=None):
    return [d.get("done_lines", {}).get(r["id"], {}).get("sha") for a in reps(d, cell, key) for r in a["results"]]


def med(x):
    return st.median(x) if x else float("nan")


def pct(a, b):
    return 100 * (a / b - 1) if b and a == a and b == b else float("nan")


B, F = arms["base"], arms["fused"]
L = [prev_lead] if prev_lead else []
print("\n== cold prefill tok/s (median over clean reps; [per-block medians]); Nov 19 = production's switches, lead on")
print(f"{'cell':22s} {'base':>8s} {'fused':>8s} {'fused/base':>10s} {'Nov 19':>8s} {'base/Nov19':>10s}  per-block base | fused")
for cell, size in (("warm", "2000"), ("warm", "8000"), ("warm", "32000"), ("warm", "100000"),
                   ("prefill_div", "2000"), ("prefill_div", "8000"), ("prefill_div", "32000"), ("prefill_div", "100000"),
                   ("prefill_code", "8000"), ("prefill_p125", "8000"), ("deep300", "300000")):
    if cell == "deep300":
        b = [x for d in B for x in prefill([{"cells": {"d": d["cells"].get("deep300", {})}, "done_lines": d.get("done_lines", {})}], "d", "300000")]
        f = [x for d in F for x in prefill([{"cells": {"d": d["cells"].get("deep300", {})}, "done_lines": d.get("done_lines", {})}], "d", "300000")]
    else:
        b, f = prefill(B, cell, size), prefill(F, cell, size)
    l = prefill(L, cell, size) if cell != "deep300" else []
    pb = " ".join(f"{med(prefill([d], cell, size)):.0f}" for d in B) if cell != "deep300" else ""
    pf = " ".join(f"{med(prefill([d], cell, size)):.0f}" for d in F) if cell != "deep300" else ""
    print(f"{cell + ' ' + size:22s} {med(b):8.1f} {med(f):8.1f} {pct(med(f), med(b)):+9.2f}% {med(l):8.1f} {pct(med(b), med(l)):+9.2f}%  {pb} | {pf}")

print("\n== decode: aggregate tok/s (median of reps), ms a round, tokens a round; replies equal base")
for cell, key in (("dec1", None), ("dec4", None), ("dec8", None), ("longctx", "lc1"), ("longctx", "lc4"), ("think", "t1"), ("think", "t4"),
                  ("deep300", "longctx")):
    if cell == "deep300":
        sub = lambda ds: [{"cells": d["cells"].get("deep300", {}), "done_lines": d.get("done_lines", {})} for d in ds]  # noqa: E731
        b = [a["aggregate_tok_s"] for d in sub(B) for k in ("lc1", "lc4") for a in reps({"cells": d["cells"].get("longctx", {}), "done_lines": d["done_lines"]}, k) if "aggregate_tok_s" in a]
        print(f"deep300 longctx      base {[round(x, 1) for x in b]}  fused {[round(a['aggregate_tok_s'], 1) for d in sub(F) for k in ('lc1', 'lc4') for a in reps({'cells': d['cells'].get('longctx', {}), 'done_lines': d['done_lines']}, k) if 'aggregate_tok_s' in a]}")
        continue
    b, f = decode(B, cell, key), decode(F, cell, key)
    mb, mf = round_ms(B, cell, key), round_ms(F, cell, key)
    ref = prev_lead if cell in ("dec1", "dec4") else prev_base
    l = decode([ref], cell, key) if ref else []
    ml = round_ms([ref], cell, key) if ref else (float("nan"),) * 2
    eq = [x == y for d in F for e in B for x, y in zip(shas(d, cell, key), shas(e, cell, key)) if x and y]
    name = cell + (" " + key if key else "")
    print(f"{name:12s} base {med(b):7.1f} fused {med(f):7.1f} ({pct(med(f), med(b)):+.2f}%)  ms/round {mb[0]:.2f} -> {mf[0]:.2f} ({pct(mf[0], mb[0]):+.2f}%)"
          f"  tok/round {mb[1]:.2f} -> {mf[1]:.2f}  | Nov 19 {med(l):7.1f} ({pct(med(b), med(l)):+.2f}% base vs Nov 19) {ml[0]:.2f} ms"
          f"  | replies equal {sum(eq)}/{len(eq)}")


def rate(r):
    return float(r["tok_per_s"])


print("\n== probe (solo rows: engine decode tok/s, exact tokens a round; * = reply hash differs from base s01)")
rows = {d["id"]: d["cells"].get("probe", {}).get("rows", []) for d in blocks.values()}
order = [b for b in sorted(rows) if rows[b]]
ref_rows = rows.get("s01", [])
hdr = f"{'row':18s}" + "".join(f" {b + ' ' + blocks[b]['arm']:>16s}" for b in order) + f" {'Nov19 after':>12s}"
print(hdr)
tot = {b: [0.0, 0.0] for b in order + ["after"]}
for i, r0 in enumerate(ref_rows):
    line = f"{r0['label']:18s}"
    for b in order:
        r = rows[b][i]
        star = "" if str(r["sha"]) == str(r0["sha"]) else "*"
        if "rounds" in r:
            line += f" {rate(r):7.1f} {r['exact_tokens_per_round']:5.2f}{star:1s} "
            tot[b][0] += float(r["completion_tokens"]); tot[b][1] += float(r["completion_tokens"]) / rate(r)
        else:
            line += f" {float(r['agg_tok_s_wall']):7.1f} {float(r['tokens_per_stream_round']):5.2f}{star:1s} "
    if prev_after and i < len(prev_after):
        r = prev_after[i]
        line += f" {rate(r) if 'rounds' in r else float(r['agg_tok_s_wall']):11.1f}{'' if str(r['sha']) == str(r0['sha']) else '*'}"
        if "rounds" in r:
            tot["after"][0] += float(r["completion_tokens"]); tot["after"][1] += float(r["completion_tokens"]) / rate(r)
    print(line)
line = f"{'solo total tok/s':18s}"
for b in order + ["after"]:
    if tot[b][1]:
        line += f" {b}:{tot[b][0] / tot[b][1]:.1f}"
print(line)
sb = [tot[b][0] / tot[b][1] for b in order if blocks[b]["arm"] == "base" and tot[b][1]]
sf = [tot[b][0] / tot[b][1] for b in order if blocks[b]["arm"] in ("next","next_pdl") and tot[b][1]]
if sb and sf:
    print(f"solo total: base {st.mean(sb):.1f} fused {st.mean(sf):.1f} ({pct(st.mean(sf), st.mean(sb)):+.2f}%); block spread base {min(sb):.1f}-{max(sb):.1f}, fused {min(sf):.1f}-{max(sf):.1f}")

print("\n== correctness")
for b, d in blocks.items():
    c = d["cells"]
    ex = c.get("exact:runs/combo3/blocks/a01.exact.json") or {k: v for k, v in c.items() if k.startswith("exact")}
    exs = json.dumps(ex)[-220:] if ex else "-"
    si = c.get("sampled_ident")
    print(f"{b} {d['arm']:5s} validate rc={c.get('validate', {}).get('rc')}  exact: {exs}")
    if si is not None:
        print(f"      sampled_ident: {json.dumps(si)[:220]}")

print("\n== block notes")
for b, d in blocks.items():
    s = d.get("samples", {})
    h, w = s.get("head", {}), s.get("worker", {})
    print(f"{b} {d['arm']:5s} image {d.get('image')} status {d.get('status')} arm_proof {d.get('arm_proof_problems')} "
          f"clock max {h.get('clock_max')}/{w.get('clock_max')} temp max {h.get('temp_max')}/{w.get('temp_max')} "
          f"mem low {h.get('mem_low_kb', 0) / 1048576:.1f}/{w.get('mem_low_kb', 0) / 1048576:.1f} GiB")
