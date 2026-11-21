#!/usr/bin/env python3
"""Summaries and ABAB comparisons for runs/overnight/blocks/*.json (scripts/bench_overnight.py's output).

    python3 tools/overnight_report.py [runs/overnight]        # per-block table + per-candidate comparison (markdown)

Only clean attempts count (idle start, request delta == own, nothing running at the end, cached_tokens == 0 for
prefill); a block whose clock samples exceeded 2405 MHz on either Spark is marked invalid and excluded.
Band for a comparison = max(per-rep spread of the candidate and both adjacent bases, |base_L - base_R| / mean).
"""
import glob
import json
import os
import statistics
import sys

D = sys.argv[1] if len(sys.argv) > 1 else "runs/overnight"
PRIMARY = ["pd.8000", "pd.32000", "pd.100000", "dec1", "dec8", "ttft.8000"]


def clean_first(atts):
    for a in atts:
        if a.get("clean"):
            return a
    return None


def metrics(doc):
    """{metric: [per-rep values]} from clean attempts; plus counts."""
    m, cnt = {}, {}
    cells = doc.get("cells", {})

    def add(k, v):   # keeps None placeholders so rep index i is the same prompt in every block (paired decode)
        m.setdefault(k, []).append(v)

    def count(k, atts):
        c = cnt.setdefault(k, [0, 0, 0])  # reps, clean reps, attempts
        c[0] += 1
        c[2] += len(atts)
        c[1] += any(a.get("clean") for a in atts)

    for key, pre in (("prefill_div", "pd"), ("prefill_quick", "pd"), ("prefill_code", "pc"), ("prefill_p125", "pp")):
        for size, reps in (cells.get(key) or {}).items():
            if not isinstance(reps, list):
                continue
            for atts in reps:
                count(f"{pre}.{size}", atts)
                a = clean_first(atts)
                if a:
                    r = a["results"][0]
                    add(f"{pre}.{size}", r.get("prefill_tok_s"))
                    if pre == "pd":
                        add(f"ttft.{size}", r.get("ttft_s"))
                        add(f"tok.{size}", r.get("prompt_tokens"))
    for size, reps in (cells.get("warm") or {}).items():
        if isinstance(reps, list):
            for atts in reps:
                if atts and atts[0].get("results") and "error" not in atts[0]["results"][0]:
                    r = atts[0]["results"][0]
                    m.setdefault(f"cold.{size}", []).append(r.get("prefill_tok_s"))
                    m.setdefault(f"cold_clean.{size}", []).append(atts[0].get("clean"))
    for key, lab in (("dec1", "dec1"), ("dec4", "dec4"), ("dec8", "dec8")):
        reps = cells.get(key)
        if isinstance(reps, list):
            for atts in reps:
                count(lab, atts)
                a = clean_first(atts)
                if not a:
                    add(lab, None)
                if a:
                    if lab == "dec1":
                        add(lab, a["results"][0].get("decode_tok_s"))
                        add("dec1.ct", a["results"][0].get("completion_tokens"))
                    else:
                        add(lab, a.get("aggregate_tok_s"))
                        add(lab + ".per_stream", statistics.mean([r["decode_tok_s"] for r in a["results"] if r.get("decode_tok_s")]))
    lc = cells.get("longctx") or {}
    for lab in ("lc1", "lc4"):
        for atts in lc.get(lab, []) if isinstance(lc, dict) else []:
            count(lab, atts)
            a = clean_first(atts)
            if a:
                add(lab, a["results"][0].get("decode_tok_s") if lab == "lc1" else a.get("aggregate_tok_s"))
                add(lab + ".cached", a["results"][0].get("cached_tokens"))
    th = cells.get("think") or {}
    for lab in ("t1", "t4"):
        for atts in th.get(lab, []) if isinstance(th, dict) else []:
            count(lab, atts)
            a = clean_first(atts)
            if a:
                add(lab, a["results"][0].get("decode_tok_s") if lab == "t1" else a.get("aggregate_tok_s"))
    return m, cnt


def evidence_status(doc):
    """Check the gates the block actually requested, including old blocks labelled ok on liveness alone."""
    if doc.get("status") != "ok":
        return doc.get("status")
    cells = doc["cells"]
    for name in doc.get("cells_planned", cells):
        result = cells.get(name, {})
        if name == "validate" and result.get("rc") != 0:
            return "gate-failed"
        if name == "exact" or name.startswith("exact:"):
            if result.get("record_rc") != 0 or (":" in name and result.get("compare_rc") != 0):
                return "gate-failed"
        if name == "sampled_ident":
            passes = result.get("passes", [])
            seeds = {"9100", "9101", "9102", "9103"}
            if (len(passes) != 2 or passes[0] != passes[1] or set(passes[0]) != seeds
                    or any(not sha or sha.startswith("error") for sha in passes[0].values())):
                return "gate-failed"
    if (doc.get("bailed") or doc.get("clock_abort")
            or any(name not in cells or "exception" in cells[name] for name in doc.get("cells_planned", cells))):
        return "incomplete"
    return arm_status(doc)


def arm_status(doc):
    if doc.get("arm_proof_problems"):
        return "arm-mismatch"
    if doc.get("flags", {}).get("TF_DS_T15_PLACE_BOUND") == "1":
        modes = set()
        for rank in ("rank0", "rank1"):
            env = dict(x.split("=", 1) for x in doc.get("inspect", {}).get(rank, {}).get("env", []))
            prune = env.get("TF_DS_T13_EXPERT_PRUNE", "")
            cap = env.get("TF_DS_T13_EXPERT_CAP", "")
            if (env.get("TF_DS_T15_PLACE_BOUND") != "1" or cap in ("", "0")
                    or prune in ("", "0") or prune.startswith("stats")):
                return "inactive-arm"
            modes.add((prune, cap))
        if len(modes) != 1:
            return "arm-mismatch"
    return "ok"


def block_summary(doc):
    m, cnt = metrics(doc)
    s = doc.get("samples", {})
    h, w = s.get("head", {}), s.get("worker", {})
    return {"id": doc["id"], "arm": doc["arm"], "status": evidence_status(doc), "clock_valid": doc.get("clock_valid"),
            "clock": (h.get("clock_max"), w.get("clock_max")), "loaded_clock": (h.get("clock_loaded_median"), w.get("clock_loaded_median")),
            "temp": (h.get("temp_min"), h.get("temp_max"), w.get("temp_min"), w.get("temp_max")),
            "mem_low_gib": (round(h.get("mem_low_kb", 0) / 1048576, 2), round(w.get("mem_low_kb", 0) / 1048576, 2)),
            "mem_first_last_head_gib": (round(h.get("mem_first_kb", 0) / 1048576, 2), round(h.get("mem_last_kb", 0) / 1048576, 2)),
            "m": m, "cnt": cnt, "proof": doc.get("arm_proof_problems"), "banners": doc.get("banners"),
            "restart": doc.get("restart"), "bailed": doc.get("bailed"), "engram": eng_delta(doc), "t": doc.get("t_start")}


def eng_delta(doc):
    b, a = doc.get("engram_before", {}), doc.get("engram_after", {})
    return {k: round(a[k] - b.get(k, 0), 2) for k in a}


def med(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return statistics.median(xs) if xs else None


def spread(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if len(xs) < 2:
        return None
    return (max(xs) - min(xs)) / statistics.median(xs)


PAIRED = ("dec1", "dec4", "dec8", "lc1", "lc4", "t1", "t4")


def paired(xc, xb):
    return [a / b for a, b in zip(xc, xb) if isinstance(a, (int, float)) and isinstance(b, (int, float)) and b]


def compare(c, L, R, metric, lower_better=False):
    xc, xl, xr = c["m"].get(metric), L["m"].get(metric), R["m"].get(metric)
    if metric in PAIRED and xc and xl and xr:
        pl, pr, pb = paired(xc, xl), paired(xc, xr), paired(xr, xl)
        if pl and pr:
            dl, dr = med(pl) - 1, med(pr) - 1
            sp = [spread(x) for x in (pl, pr)]
            bb = abs(med(pb) - 1) if pb else 0
            band = max([x for x in sp if x is not None] + [bb])
            verdict = "WIN" if min(dl, dr) > band else ("LOSS" if max(dl, dr) < -band else "within")
            return {"cand": med(xc), "L": med(xl), "R": med(xr), "dL": dl, "dR": dr, "band": band, "bb": bb,
                    "verdict": verdict + " (paired)", "n": (len(pl), len(pr), len(pb))}
    mc, ml, mr = med(xc or []), med(xl or []), med(xr or [])
    if None in (mc, ml, mr):
        return None
    sps = [spread(x or []) for x in (xc, xl, xr)]
    bb = abs(ml - mr) / ((ml + mr) / 2)
    band = max([x for x in sps if x is not None] + [bb])
    dl, dr = (mc - ml) / ml, (mc - mr) / mr
    if lower_better:
        dl, dr = -dl, -dr
    verdict = "WIN" if min(dl, dr) > band else ("LOSS" if max(dl, dr) < -band else "within")
    return {"cand": mc, "L": ml, "R": mr, "dL": dl, "dR": dr, "band": band, "bb": bb, "verdict": verdict,
            "n": (len(xc or []), len(xl or []), len(xr or []))}


def main():
    docs = []
    for f in sorted(glob.glob(os.path.join(D, "blocks", "*.json"))):
        try:
            d = json.load(open(f))
            if isinstance(d, dict) and "cells" in d:   # skip the validate/exact records beside the blocks
                docs.append(d)
        except ValueError:
            continue
    docs.sort(key=lambda d: d.get("t_start", 0))
    S = [block_summary(d) for d in docs]
    keys = ["pd.2000", "pd.8000", "pd.32000", "pd.100000", "ttft.8000", "pc.8000", "pp.8000", "dec1", "dec4", "dec8", "lc1", "lc4", "t1", "t4",
            "cold.8000", "cold.100000"]
    print("| block | arm | status | clock max H/W | temp H/W | mem low H/W GiB | " + " | ".join(keys) + " | clean reps/reps (attempts) |")
    print("|" + "---|" * (7 + len(keys)))
    for s in S:
        cells = []
        for k in keys:
            v = s["m"].get(k)
            cells.append("-" if med(v or []) is None else f"{med(v):.1f}" + (f" ±{spread(v) * 100:.1f}%" if spread(v) is not None else "") + f" (n{len([x for x in v if x is not None])})")
        tot = [sum(c[i] for c in s["cnt"].values()) for i in range(3)]
        print(f"| {s['id']} | {s['arm']} | {s['status']}{' BAIL:' + s['bailed'] if s['bailed'] else ''} | {s['clock'][0]}/{s['clock'][1]}{'' if s['clock_valid'] else ' INVALID'} | "
              f"{s['temp'][1]}/{s['temp'][3]} | {s['mem_low_gib'][0]}/{s['mem_low_gib'][1]} | " + " | ".join(cells) + f" | {tot[1]}/{tot[0]} ({tot[2]}) |")
    print()
    ignore_clock = os.environ.get("REPORT_IGNORE_CLOCK") == "1"   # campaigns run without the 2400 MHz lock
    valid = [s for s in S if (s["clock_valid"] or ignore_clock) and s["status"] in ("ok",) and not s["id"].startswith("s") and not s["bailed"]]
    for i, s in enumerate(valid):
        if s["arm"] == "base":
            continue
        L = next((x for x in reversed(valid[:i]) if x["arm"] == "base"), None)
        R = next((x for x in valid[i + 1:] if x["arm"] == "base"), None)
        if not L or not R:
            print(f"### {s['id']} {s['arm']}: missing adjacent base (L={L and L['id']} R={R and R['id']})\n")
            continue
        print(f"### {s['id']} {s['arm']} vs {L['id']} / {R['id']}")
        print("| metric | cand | base L | base R | d vs L | d vs R | band | verdict | n |\n|---|---|---|---|---|---|---|---|---|")
        for k in ["pd.2000", "pd.8000", "pd.32000", "pd.100000", "ttft.8000", "pc.8000", "pp.8000", "dec1", "dec4", "dec8", "lc1", "lc4", "t1", "t4"]:
            c = compare(s, L, R, k, lower_better=k.startswith("ttft"))
            if c:
                print(f"| {k}{' (primary)' if k in PRIMARY else ''} | {c['cand']:.1f} | {c['L']:.1f} | {c['R']:.1f} | {c['dL'] * 100:+.1f}% | {c['dR'] * 100:+.1f}% | "
                      f"{c['band'] * 100:.1f}% | {c['verdict']} | {c['n']} |")
        print(f"\nmem low H/W GiB: cand {s['mem_low_gib']} vs L {L['mem_low_gib']} R {R['mem_low_gib']}; engram delta cand {s['engram']} L {L['engram']}; "
              f"graphs destroyed (r0/r1) cand {[b.get('graphs_destroyed_max') for b in (s['banners'] or {}).values()]}\n")


if __name__ == "__main__":
    main()
