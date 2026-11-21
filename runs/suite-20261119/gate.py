"""Per-block gate for the shortened suite. Run from the recipe root on a copy of the blocks:

  python3 -I runs/suite-20261119/gate.py [s01 ...]      (no ids = every finished block)

A block passes when its arm proof is clean, validation equals the saved references and every probe reply's hash
equals production's. Narrow probe rows 0-6 are compared with rows 0, 1, 4, 5, 6, 10, 11 of the deployed probe; rows
7-10 (the four extra code prompts) with block s01's by position and label. The base and lead arms must also match
production's rounds and accepted drafts. A drafting arm must differ from base in rounds on at least one solo row, or
its switch did not take effect.
"""
import glob
import json
import os
import sys

S = "runs/suite-20261119"
REF = json.load(open("runs/draft-block-20261118/deploy/probe.json"))
MAP = [0, 1, 4, 5, 6, 10, 11]
SAME_DRAFTING = {"base", "lead"}


def attempts(o):
    if isinstance(o, dict):
        if "attempt" in o and "clean" in o:
            yield o
        for v in o.values():
            yield from attempts(v)
    elif isinstance(o, list):
        for v in o:
            yield from attempts(v)


def gate(bid, first):
    p = f"{S}/blocks/{bid}.json"
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    arm, cells, bad, notes = d["arm"], d["cells"], [], []
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
    if lead != [arm == "lead"] * 2:
        bad.append(f"lead reader banners {lead}")
    pr = cells.get("probe")
    if pr is not None:
        rows = pr.get("rows") or []
        if pr.get("rc") != 0 or any(r.get("failed") for r in rows) or len(rows) != 11:
            bad.append(f"probe rc {pr.get('rc')} rows {len(rows)} failed {sum(1 for r in rows if r.get('failed'))}")
        sha_bad, differs = [], []
        frows = first["cells"]["probe"]["rows"] if first else []
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
        if arm in SAME_DRAFTING and differs:
            bad.append(f"rounds or accepted drafts differ on a non-drafting arm: {differs}")
        if arm not in SAME_DRAFTING and not differs:
            bad.append("switch NOT APPLIED: every solo row has base's rounds and accepted drafts")
        notes.append(f"probe {len(rows)} rows, {len(rows) - len(sha_bad)} hashes equal, drafting differs on {len(differs)} solo rows")
    errs = sorted({c for c, r in cells.items() for a in attempts(r) if a.get("errors")})
    unclean = sum(1 for r in cells.values() for a in attempts(r) if not a.get("clean"))
    if errs or unclean:
        notes.append(f"attempts with errors in {errs}, unclean {unclean}")
    mem = d.get("samples", {})
    low = [round(mem.get(k, {}).get("mem_low_kb", 0) / 1048576, 1) for k in ("head", "worker")]
    print(f"{bid} {arm:6s} {'PASS' if not bad else 'FAIL'}  {'; '.join(notes)}; mem low H/W {low[0]}/{low[1]} GiB"
          + (f"\n    PROBLEMS: {bad}" if bad else ""))
    return not bad


ids = sys.argv[1:] or sorted(os.path.basename(f)[:-5] for f in glob.glob(f"{S}/blocks/s[0-9][0-9].json"))
first = json.load(open(f"{S}/blocks/s01.json")) if os.path.exists(f"{S}/blocks/s01.json") else None
res = [gate(b, first) for b in ids]
print(f"{sum(1 for r in res if r)} of {sum(1 for r in res if r is not None)} blocks pass")
sys.exit(0 if all(r for r in res if r is not None) else 1)
