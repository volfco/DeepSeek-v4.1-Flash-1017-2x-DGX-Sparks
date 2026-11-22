#!/usr/bin/env python3
"""Rates table for suite-fused-next2-20261121 (run from the recipe root):

  python3 -I runs/suite-fused-next2-20261121/table2.py

Sections: (1) fixed_replay identity and rates per block with deltas against s01 - the "compare output token hashes,
prefill tokens/s and decode ms/round" of the 0254-0258 pair gate; (2) dec8_replay_audit compare invocations for the
staggered dec8 waves (the admission-order-fixed 8-stream comparison); (3) standard cell rates (prefill tok/s,
dec/think aggregate tok/s and pooled ms/round from the done lines) for orientation. Hash gates live in gate.py.
"""
import glob
import json
import os
import subprocess
import sys

S = os.environ.get("GATE_SUITE", os.path.dirname(os.path.abspath(__file__)))
AUDIT = "runs/pre-bench-20261121/dec8-audit/dec8_replay_audit.py"


def load():
    out = {}
    for f in sorted(glob.glob(f"{S}/blocks/s[0-9][0-9].json")):
        d = json.load(open(f))
        out[os.path.basename(f)[:-5]] = d
    return out


def bucket(pt):
    n = (int(pt) + 7) // 8
    b = 1
    while b < n:
        b *= 2
    return b


def fixed_rows(doc):
    done = doc.get("done_lines", {})
    out = {}
    for name, cell in sorted((doc.get("cells", {}).get("fixed_replay") or {}).items()):
        if not isinstance(cell, list):
            continue
        atts = [a for a in cell if a.get("clean")]
        if not atts:
            continue
        r = atts[-1]["results"][0]
        dl = done.get(r.get("id")) or {}
        ms = (round(1000 * (r["t_last"] - r["t_first"]) / int(dl["rounds"]), 2)
              if dl.get("rounds") and r.get("t_first") is not None else None)
        out[name] = {"pt": r.get("prompt_tokens"), "bucket": bucket(r.get("prompt_tokens") or 0),
                     "text_sha": r.get("text_sha"), "sha": dl.get("sha"), "rounds": dl.get("rounds"),
                     "prefill_tok_s": r.get("prefill_tok_s"), "ttft_s": r.get("ttft_s"), "ms_per_round": ms,
                     "decode_tok_s": r.get("decode_tok_s"), "cached": r.get("cached_tokens")}
    return out


def pooled_ms_round(doc, cells):
    ids = [r["id"] for c in cells for rep in doc.get("cells", {}).get(c) or [] if isinstance(rep, list)
           for a in rep if isinstance(a, dict) and a.get("clean") for r in a["results"] if r.get("id")]
    dl = doc.get("done_lines", {})
    rounds = sum(int(dl[i].get("rounds", 0)) for i in ids if i in dl)
    decs = sum(float(dl[i].get("decode_s", 0)) for i in ids if i in dl)
    return round(1000 * decs / rounds, 2) if rounds else None


def agg_tok_s(doc, cell):
    vals = [a.get("aggregate_tok_s") for rep in doc.get("cells", {}).get(cell) or [] if isinstance(rep, list) for a in rep
            if isinstance(a, dict) and a.get("clean") and a.get("aggregate_tok_s")]
    return round(sum(vals) / len(vals), 2) if vals else None


def main():
    docs = load()
    if not docs:
        sys.exit("no blocks yet")
    base = docs.get("s01")
    print("== fixed_replay (identical input bytes and token IDs across blocks)")
    for bid, doc in docs.items():
        rows = fixed_rows(doc)
        for name, r in rows.items():
            b = fixed_rows(base).get(name, {}) if doc is not base else {}
            d_pf = (f" {100 * (r['prefill_tok_s'] - b['prefill_tok_s']) / b['prefill_tok_s']:+.2f}%"
                    if b.get("prefill_tok_s") and r.get("prefill_tok_s") and doc is not base else "")
            d_ms = (f" {100 * (r['ms_per_round'] - b['ms_per_round']) / b['ms_per_round']:+.2f}%"
                    if b.get("ms_per_round") and r.get("ms_per_round") and doc is not base else "")
            same = "" if doc is base else (" same" if (r["text_sha"], r["pt"], r["sha"]) == (b.get("text_sha"), b.get("pt"), b.get("sha")) else " **DIFFERS**")
            print(f"{bid} {doc['arm']:5s} {name:11s} pt={r['pt']} blk={r['bucket']:<6} "
                  f"prefill={r['prefill_tok_s'] and round(r['prefill_tok_s'], 1)}{d_pf} tok/s "
                  f"ttft={r['ttft_s'] and round(r['ttft_s'], 2)}s rounds={r['rounds']} ms/r={r['ms_per_round']}{d_ms} "
                  f"sha={str(r['sha'])[:10]} cached={r['cached']}{same}")
    print("\n== dec8_staggered (admission order locked; the dec8-dip fix validation)")
    for pair in (("s01", "s02"), ("s04", "s03"), ("s01", "s04"), ("s01", "s05")):
        a = f"{S}/blocks/{pair[0]}.dec8st.json"
        b = f"{S}/blocks/{pair[1]}.dec8st.json"
        if os.path.exists(a) and os.path.exists(b):
            p = subprocess.run(["python3", "-I", AUDIT, "compare", a, b], capture_output=True, text=True, timeout=120)
            print(f"-- {pair[0]} -> {pair[1]}")
            print(p.stdout.strip() or p.stderr[-400:])
    print("\n== standard cells (orientation; exact hashes are gate.py's job)")
    for bid, doc in docs.items():
        pf = {}
        for size in ("2000", "8000", "32000", "100000"):
            vals = [r.get("prefill_tok_s") for rep in (doc.get("cells", {}).get("prefill_div") or {}).get(size) or []
                    for a in rep if a.get("clean") for r in a["results"] if r.get("prefill_tok_s")]
            if vals:
                pf[size] = round(sum(vals) / len(vals), 1)
        dec = {c: agg_tok_s(doc, c) for c in ("dec1", "dec4", "dec8", "t1", "t4")}
        print(f"{bid} {doc['arm']:5s} prefill {pf}  tok/s {dec}  ms/r dec1={pooled_ms_round(doc, ['dec1'])} "
              f"dec4={pooled_ms_round(doc, ['dec4'])} dec8={pooled_ms_round(doc, ['dec8'])}")


if __name__ == "__main__":
    main()
