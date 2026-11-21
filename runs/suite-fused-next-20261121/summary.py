"""Tabulate the shortened suite: one block per arm against block s01 (base). Run from the recipe root:

  python3 -I runs/suite-fused-20261121/summary.py

Prefill rates are uncached prompt tokens over the engine's own prefill seconds (its done line). Probe rates are the
engine's decode rate; tokens a round are exact counts. "ctl" is production's deployed probe from November 18 on the
rows it shares with this probe: the spread between it and s01 is the only control spread this run has.
"""
import glob
import json
import os
import statistics as st

S = "runs/suite-fused-20261121"
REF = json.load(open("runs/draft-block-20261118/deploy/probe.json"))
MAP = [0, 1, 4, 5, 6, 10, 11]
docs = {}
for f in sorted(glob.glob(f"{S}/blocks/s[0-9][0-9].json")):
    d = json.load(open(f))
    if d.get("status") == "ok":
        docs[d["arm"]] = d
print("blocks:", {a: d["id"] for a, d in docs.items()})
base = docs["base"]


def clean(rep):
    c = [a for a in rep if a.get("clean")]
    return c[-1] if c else None


def prefill(d, cell, size):
    out = []
    for rep in d["cells"].get(cell, {}).get(size, []):
        a = clean(rep)
        if not a:
            continue
        for r in a["results"]:
            dl = d["done_lines"].get(r["id"])
            if dl and float(dl["prefill_s"]) > 0:
                out.append((int(dl["prompt"]) - int(dl["cached"])) / float(dl["prefill_s"]))
    return out


def decode(d, cell):
    return [clean(rep)["aggregate_tok_s"] for rep in d["cells"].get(cell, []) if clean(rep)]


def rounds_ms(d, cell):
    """Summed decode seconds over summed rounds for a decode cell's streams: holds whatever the reply lengths."""
    sec = rnd = tok = 0.0
    for rep in d["cells"].get(cell, []):
        a = clean(rep)
        for r in (a["results"] if a else []):
            dl = d["done_lines"].get(r["id"])
            if dl:
                sec += float(dl["decode_s"]); rnd += float(dl["rounds"]); tok += float(dl["tokens"])
    return (1000 * sec / rnd, tok / rnd) if rnd else (float("nan"), float("nan"))


if "lead" in docs:
    lead = docs["lead"]
    print("\n== lead reader against base: cold prefill tok/s on natural text (different passages per block), median [reps]")
    for cell, size in (("warm", "2000"), ("warm", "8000"), ("warm", "32000"), ("warm", "100000"), ("prefill_div", "2000"), ("prefill_div", "8000"),
                       ("prefill_div", "32000"), ("prefill_div", "100000"), ("prefill_code", "8000")):
        b, l = prefill(base, cell, size), prefill(lead, cell, size)
        if b and l:
            print(f"{cell:13s} {size:>6s}: base {st.median(b):7.1f} {[round(x) for x in b]}  lead {st.median(l):7.1f} {[round(x) for x in l]}  {100 * (st.median(l) / st.median(b) - 1):+.1f}%")
    print("== decode (the same prompts in both blocks), aggregate tok/s per repetition")
    for cell in ("dec1", "dec4", "dec8"):
        b, l = decode(base, cell), decode(lead, cell)
        if b and l:
            pairs = [y / x for x, y in zip(b, l)]
            mb, ml = rounds_ms(base, cell), rounds_ms(lead, cell)
            print(f"{cell}: base {[round(x, 1) for x in b]}  lead {[round(x, 1) for x in l]}  paired {100 * (st.median(pairs) - 1):+.1f}%  round time {mb[0]:.2f} -> {ml[0]:.2f} ms ({100 * (ml[0] / mb[0] - 1):+.1f}%)")
    same = [base["done_lines"][r["id"]]["sha"] == lead["done_lines"][q["id"]]["sha"]
            for cell in ("dec1", "dec4") for rb, rl in zip(base["cells"][cell], lead["cells"].get(cell, []))
            for r, q in zip(clean(rb)["results"], clean(rl)["results"]) if r["id"] in base["done_lines"] and q["id"] in lead["done_lines"]]
    print(f"decode replies equal to base's: {sum(same)} of {len(same)}")

arms = [a for a in ("kept3", "b7", "kept2", "lead") if a in docs and "probe" in docs[a]["cells"]]
brow = base["cells"]["probe"]["rows"]
ctlb = json.load(open(f"{S}/blocks/production-after.probe.json")) if os.path.exists(f"{S}/blocks/production-after.probe.json") else None


def rate(r):
    """The engine's own decode rate (0.1 tok/s resolution; its decode seconds are rounded to 0.1 s)."""
    return float(r["tok_per_s"])


def msr(r):
    return 1000 * float(r["completion_tokens"]) / (rate(r) * float(r["rounds"]))


print("\n== probe, solo rows: tokens a round, ms a round, tok/s  (* = reply differs from base)")
names = ["base"] + arms + (["after"] if ctlb else [])
tot = {a: [0.0, 0.0] for a in names + ["ctl"]}
for i, r0 in enumerate(brow):
    if "rounds" not in r0:
        continue
    ctl = REF[MAP[i]] if i < len(MAP) else None
    line = f"{r0['label']:15s} base {r0['exact_tokens_per_round']:.3f} {msr(r0):.2f}ms {rate(r0):.1f}"
    if ctl:
        line += f" (Nov 18 production {rate(ctl):.1f}, base is {100 * (rate(r0) / rate(ctl) - 1):+.1f}%)"
    for a in names:
        r = r0 if a == "base" else ctlb[i] if a == "after" else docs[a]["cells"]["probe"]["rows"][i]
        tot[a][0] += float(r["completion_tokens"]); tot[a][1] += float(r["completion_tokens"]) / rate(r)
        if a == "base":
            continue
        star = "" if r["sha"] == r0["sha"] else "*"
        line += f" | {a} {r['exact_tokens_per_round']:.3f} ({100 * (r['exact_tokens_per_round'] / r0['exact_tokens_per_round'] - 1):+.1f}%) {msr(r):.2f}ms ({100 * (msr(r) / msr(r0) - 1):+.1f}%) {rate(r):.1f} ({100 * (rate(r) / rate(r0) - 1):+.1f}%){star}"
    print(line)
line = f"{'all solo rows':15s} base {tot['base'][0] / tot['base'][1]:.1f} tok/s"
for a in names[1:]:
    line += f" | {a} {tot[a][0] / tot[a][1]:.1f} ({100 * (tot[a][0] / tot[a][1] / (tot['base'][0] / tot['base'][1]) - 1):+.1f}%)"
print(line + "   (total tokens over total decode seconds)")
print("== probe, bursts: aggregate tok/s, tokens per stream per round, drafts accepted")
for i, r0 in enumerate(brow):
    if "width" not in r0:
        continue
    ctl = REF[MAP[i]]
    line = f"{r0['label']:18s} base {float(r0['agg_tok_s_wall']):.1f} {float(r0['tokens_per_stream_round']):.2f} {r0['accept_pct']}% (Nov 18 production {float(ctl['agg_tok_s_wall']):.1f})"
    for a in names[1:]:
        r = ctlb[i] if a == "after" else docs[a]["cells"]["probe"]["rows"][i]
        star = "" if str(r["sha"]) == str(r0["sha"]) else "*"
        line += f" | {a} {float(r['agg_tok_s_wall']):.1f} ({100 * (float(r['agg_tok_s_wall']) / float(r0['agg_tok_s_wall']) - 1):+.1f}%) {float(r['tokens_per_stream_round']):.2f} {r['accept_pct']}%{star}"
    print(line)
print("\n== block notes")
for a, d in docs.items():
    s = d["samples"]
    print(f"{d['id']} {a:6s} cells {list(d['cells'])}  validate rc {d['cells'].get('validate', {}).get('rc')}  mem low H/W "
          f"{s['head'].get('mem_low_kb', 0) / 1048576:.1f}/{s['worker'].get('mem_low_kb', 0) / 1048576:.1f} GiB  clock max {s['head'].get('clock_max')}/{s['worker'].get('clock_max')}")
