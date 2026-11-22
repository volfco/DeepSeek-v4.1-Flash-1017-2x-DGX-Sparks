#!/usr/bin/env python3
"""Per-block gate for suite-fused-next2-20261121. Run from the recipe root on a copy of the blocks:

  python3 -I runs/suite-fused-next2-20261121/gate.py [s01 ...]      (no ids = every finished block)

A block passes when its arm proof is clean, validation equals the saved references, every probe reply's hash equals
production's, and the fixed-input replays match s01's input and output hashes. Fixes over suite-fused-next's gate:
the suite path follows this file (or $GATE_SUITE), the lead-reader banner is compared against each block's flags
instead of its arm name, and the "drafting must differ" rule applies only to arms whose flags change drafting
switches - a fused-kernel arm that reproduces base's rounds and accepted drafts is bit-exactness evidence, not a
failure. Narrow probe rows 0-6 are compared with rows 0, 1, 4, 5, 6, 10, 11 of the deployed probe; rows 7-10 (the
four extra code prompts) with block s01's by position and label.
"""
import glob
import json
import os
import sys

S = os.environ.get("GATE_SUITE", os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = json.load(open(os.path.join(ROOT, "runs/draft-block-20261118/deploy/probe.json")))
MAP = [0, 1, 4, 5, 6, 10, 11]
DRAFT_PREFIXES = ("TF_DS_DRAFT", "TF_DS_T26", "TF_DS_SPEC", "TF_DS_T13_EXPERT_CAP", "TF_DS_T13_EXPERT_PRUNE", "TF_DS_FIX_K", "TF_DS_T25")


def attempts(o):
    if isinstance(o, dict):
        if "attempt" in o and "clean" in o:
            yield o
        for v in o.values():
            yield from attempts(v)
    elif isinstance(o, list):
        for v in o:
            yield from attempts(v)


def want_lead(flags):
    """The lead-reader banner the block must show: its TF_DS_ENGRAM_LEAD flag, production's .env default (1)."""
    return str(flags.get("TF_DS_ENGRAM_LEAD", "1")) not in ("", "0")


def changes_drafting(flags):
    """True when the block's flags change drafting/pruning behaviour, so solo rounds may (must) differ from base."""
    return any(k.startswith(DRAFT_PREFIXES) and str(v) not in ("", "0") for k, v in flags.items())


def clean_rows(cell):
    """The last clean attempt's result rows of one fixed_replay name."""
    atts = [a for a in cell if a.get("clean")]
    return atts[-1]["results"] if atts else []


def gate(bid, first):
    p = f"{S}/blocks/{bid}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    arm, cells, bad, notes = d["arm"], d["cells"], [], []
    flags = d.get("flags", {})
    done = d.get("done_lines", {})
    if d.get("status") != "ok":
        bad.append(f"status {d.get('status')}")
    if d.get("arm_proof_problems"):
        bad.append(f"arm proof {d['arm_proof_problems']}")
    v = cells.get("validate")
    if not v or v.get("rc") != 0:
        bad.append(f"validate rc {v and v.get('rc')}")
    else:
        notes.append("validate 0")
    ban = d.get("banners", {})
    lead = [b.get("engram_lead_on") for b in (ban.values() if isinstance(ban, dict) else ban)]
    if lead != [want_lead(flags)] * 2:
        bad.append(f"lead reader banners {lead}, expected {[want_lead(flags)] * 2} for flags {flags}")
    pr = cells.get("probe")
    if pr is not None:
        rows = pr.get("rows") or []
        if pr.get("rc") != 0 or any(r.get("failed") for r in rows) or len(rows) != 11:
            bad.append(f"probe rc {pr.get('rc')} rows {len(rows)} failed {sum(1 for r in rows if r.get('failed'))}")
        sha_bad, differs = [], []
        frows = first["cells"]["probe"]["rows"] if first and first["cells"].get("probe") else []
        for i, r in enumerate(rows):
            ref = REF[MAP[i]] if i < len(MAP) else (frows[i] if i < len(frows) else None)
            if ref is None or r.get("label") != ref.get("label") or str(r.get("seed", r.get("seed0"))) != str(ref.get("seed", ref.get("seed0"))):
                bad.append(f"probe row {i} {r.get('label')} has no reference")
                continue
            if str(r.get("sha")) != str(ref.get("sha")):
                sha_bad.append(r["label"])
            if "rounds" in r and (r.get("rounds"), r.get("accepted")) != (ref.get("rounds"), ref.get("accepted")):
                differs.append(r["label"])
        if sha_bad:
            bad.append(f"replies differ from production: {sha_bad}")
        if changes_drafting(flags):
            if not differs:
                bad.append("switch NOT APPLIED: every solo row has base's rounds and accepted drafts")
        elif differs:
            bad.append(f"rounds or accepted drafts differ on a non-drafting arm: {differs}")
        notes.append(f"probe {len(rows)} rows, {len(rows) - len(sha_bad)} hashes equal, drafting differs on {len(differs)} solo rows")
    fr = cells.get("fixed_replay")
    if fr is not None:
        ffr = (first or {}).get("cells", {}).get("fixed_replay") or {}
        for name in sorted(fr):
            if not isinstance(fr[name], list):
                bad.append(f"fixed_replay {name}: block cell is {fr[name]!r} (bail/exception), no attempt rows")
                continue
            rows = clean_rows(fr[name])
            if len(rows) != 1:
                bad.append(f"fixed_replay {name}: {len(rows)} clean rows (want 1)")
                continue
            r = rows[0]
            dl = done.get(r.get("id")) or {}
            if not dl.get("sha"):
                bad.append(f"fixed_replay {name}: no done line for {r.get('id')}")
            if r.get("cached_tokens"):
                bad.append(f"fixed_replay {name}: cached_tokens {r.get('cached_tokens')} (not a cold prefill)")
            if first and ffr.get(name) and isinstance(ffr.get(name), list):
                frows = clean_rows(ffr[name])
                f = frows[0] if frows else {}
                fd = (first.get("done_lines") or {}).get(f.get("id")) or {}
                if r.get("text_sha") != f.get("text_sha"):
                    bad.append(f"fixed_replay {name}: text_sha {r.get('text_sha')} != s01 {f.get('text_sha')} (inputs differ)")
                if r.get("prompt_tokens") != f.get("prompt_tokens"):
                    bad.append(f"fixed_replay {name}: prompt_tokens {r.get('prompt_tokens')} != s01 {f.get('prompt_tokens')}")
                if dl.get("sha") != fd.get("sha"):
                    bad.append(f"fixed_replay {name}: token_sha {dl.get('sha')} != s01 {fd.get('sha')}")
            ms = None
            if dl.get("rounds") and r.get("t_first") is not None and r.get("t_last") is not None:
                ms = round(1000 * (r["t_last"] - r["t_first"]) / int(dl["rounds"]), 2)
            notes.append(f"fixed {name}: pt={r.get('prompt_tokens')} prefill={r.get('prefill_tok_s') and round(r['prefill_tok_s'], 1)} tok/s "
                         f"ttft={r.get('ttft_s') and round(r['ttft_s'], 2)}s rounds={dl.get('rounds')} ms/r={ms} sha={str(dl.get('sha'))[:10]}")
    ds = cells.get("dec8_staggered")
    if ds is not None:
        if ds.get("rc") != 0 or not (ds.get("rows") or {}).get("waves"):
            bad.append(f"dec8_staggered rc={ds.get('rc')} waves={len((ds.get('rows') or {}).get('waves') or [])}")
        else:
            shas = [[(s.get("prompt_tokens"), s.get("sha")) for s in w.get("streams", [])] for w in ds["rows"]["waves"]]
            fshas = []
            fds = (first or {}).get("cells", {}).get("dec8_staggered") if first else None
            if fds and (fds.get("rows") or {}).get("waves"):
                fshas = [[(s.get("prompt_tokens"), s.get("sha")) for s in w.get("streams", [])] for w in fds["rows"]["waves"]]
            if fshas and shas != fshas:
                bad.append("dec8_staggered: prompt/sha streams differ from s01")
            notes.append(f"dec8st {len(shas)} waves"
                         + (f", {sum(1 for a, b in zip(shas, fshas) if a == b)}/{len(fshas)} waves match s01" if fshas else ""))
    errs = sorted({c for c, r in cells.items() for a in attempts(r) if a.get("errors")})
    unclean = sum(1 for r in cells.values() for a in attempts(r) if not a.get("clean"))
    if errs or unclean:
        notes.append(f"attempts with errors in {errs}, unclean {unclean}")
    mem = d.get("samples", {})
    low = [round(mem.get(k, {}).get("mem_low_kb", 0) / 1048576, 1) for k in ("head", "worker")]
    print(f"{bid} {arm:6s} {'PASS' if not bad else 'FAIL'}  {'; '.join(notes)}; mem low H/W {low[0]}/{low[1]} GiB"
          + (f"\n    PROBLEMS: {bad}" if bad else ""))
    return not bad


def main():
    ids = sys.argv[1:] or sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{S}/blocks/s[0-9][0-9].json"))
    first = json.load(open(f"{S}/blocks/s01.json")) if os.path.exists(f"{S}/blocks/s01.json") else None
    res = [gate(b, first) for b in ids]
    print(f"{sum(1 for r in res if r)} of {sum(1 for r in res if r is not None)} blocks pass")
    sys.exit(0 if all(r for r in res if r is not None) else 1)


if __name__ == "__main__":
    main()
