"""Draft gate for an arms.sh suite: per-request rounds and accepted/drafted against control.

  python3 runs/k280/gate_drafts.py <OUT dir> --control ctl1,ctl2 --test mc2048a,mc2048b,...

Why: 0281 (TF_DS_MARKOV_CACHE) and 0282 (TF_DS_DRAFT_PREFETCH) cannot move an output token, so the exact panel and the
token shas do not test them. A wrong cached bias row would change DRAFTS, and that shows only as a different number of
rounds or different accepted/drafted counts for the same seeded request. This compares, request by request:
- <arm>.probe.json (suite_probe.py): every solo row's `rounds`, `accepted` ("accepted/drafted") and `sha`, keyed by
  the row's label (which carries the seed). Burst rows (`width`) are listed but not gated: concurrent streams share
  rounds and their round counts vary between identical arms.
- <arm>.json (the round-time workload): each one-stream cell's `per_round.rounds` and its request's `sha`.
A row is gated only if the controls agree with each other on it; a row on which the controls already differ is
reported as "not deterministic" and proves nothing either way.
Exit 0: every gated row of every test arm equals control. Exit 1: a difference, printed.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--control", required=True)
ap.add_argument("--test", required=True)
a = ap.parse_args()
ctl, tst = a.control.split(","), a.test.split(",")


def rows(arm):
    got = {}
    p = os.path.join(a.out, f"{arm}.probe.json")
    if os.path.exists(p):
        for r in json.load(open(p)):
            if "width" in r or not r.get("rounds"):
                continue
            got["probe: " + r["label"]] = (str(r["rounds"]), r.get("accepted"), r.get("sha"))
    p = os.path.join(a.out, f"{arm}.json")
    if os.path.exists(p):
        for c in json.load(open(p)).get("cells", []):
            if c.get("streams") != 1:
                continue
            req = (c.get("requests") or [{}])[0]
            got[f"workload: {c['depth'] // 1024}K x 1"] = (str(c["per_round"].get("rounds")), None, req.get("sha"))
    return got


data = {n: rows(n) for n in ctl + tst}
missing = [n for n in ctl + tst if not data[n]]
if missing:
    sys.exit(f"no probe or workload rows for: {', '.join(missing)}")
keys = sorted(set().union(*[set(data[n]) for n in ctl]))
gated, loose, bad = 0, [], []
for k in keys:
    ref = [data[n].get(k) for n in ctl]
    if any(r is None for r in ref) or any(r != ref[0] for r in ref):
        loose.append((k, ref))
        continue
    gated += 1
    for n in tst:
        got = data[n].get(k)
        if got != ref[0]:
            bad.append((n, k, ref[0], got))
print(f"{gated} rows gated (controls {', '.join(ctl)} agree on rounds, accepted/drafted and sha), {len(loose)} not deterministic")
for k, ref in loose:
    print(f"  not gated: {k}: controls {ref}")
for n, k, want, got in bad:
    print(f"  DIFFERENT {n}: {k}: control (rounds, accepted/drafted, sha) {want}, arm {got}")
if bad:
    sys.exit(1)
print(f"drafts IDENTICAL to control in {', '.join(tst)}" if gated else "nothing gated")
sys.exit(0 if gated else 1)
