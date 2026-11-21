#!/usr/bin/env python3
"""4-stream pass at each reached rung, after depth.py (whose 4-stream gate scaled the 2K rung's extra by the depth
ratio and skipped every 4-stream cell). Rebuilds each rung's exact text by replaying depth.py's cpt learning from the
recorded primes, refreshes the rung with a short thinking-on 1-stream request (a kept state; a max_tokens=1 prime is
not kept), then runs the 4-stream cell if the projected low stays above the floor. Ends with a short 1-stream
bookend. Waits for depth.py to exit first.  python3 -I runs/depth/four.py <depth.py pid>
"""
import os
import sys
import time

sys.dont_write_bytecode = True
sys.argv = sys.argv[:2]
pid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
while pid and os.path.exists(f"/proc/{pid}"):
    time.sleep(5)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import depth as D  # noqa: E402  (loads depth.json as D.DOC)

primes = {c["depth_target"]: c for c in D.DOC["cells"] if c["kind"] == "prime"}
texts = {}
for d in [D.SHORT] + D.LADDER:
    if d not in primes:
        break
    h = D.hay(d)
    texts[d] = h
    r0 = primes[d]["attempts"][-1]["results"][0]
    D.learn(D.SALT + h, r0)
    D.log(f"four: rung {d} rebuilt chars={len(h)} prime prompt={r0.get('prompt_tokens')}")
have4 = {c["depth_target"] for c in D.DOC["cells"] if c["kind"] == "dec4"}
reached = [d for d in D.DOC.get("reached") or [] if d in texts and d != D.SHORT and d not in have4]
D.log(f"four: start pid={os.getpid()} rungs={reached}")
extra4 = None
for d in reached:
    h = texts[d]
    ref = D.cell(f"f{d}-s1-ref", d, [D.SALT + h + D.task(f"f{d}-ref")], 128, True, "ref1")
    if not D.ok(ref):
        D.log(f"four: refresh failed at {d}; stop")
        break
    r = ref["attempts"][-1]["results"][0]
    pt0 = None
    for c in D.DOC["cells"]:
        if c["name"] == f"d{d}-s1-r1":
            pt0 = c["attempts"][-1]["results"][0].get("prompt_tokens")
    D.log(f"four: refresh {d} prompt={r.get('prompt_tokens')} (ladder rep1 {pt0}) cached={r.get('cached_tokens')}")
    lo1 = D.low_of(ref)
    for c in D.DOC["cells"]:  # the 4-stream extra measured by the ladder at other depths
        if c["kind"] == "dec4" and c["depth_target"] != D.SHORT:
            l1 = [D.low_of(x) for x in D.DOC["cells"] if x["kind"] == "dec1" and x["depth_target"] == c["depth_target"]]
            l1 = [x for x in l1 if x is not None]
            if l1 and D.low_of(c) is not None:
                extra4 = max(extra4 or 0.0, min(l1) - D.low_of(c))
    # the 1-stream low at this depth from the ladder, if lower
    for c in D.DOC["cells"]:
        if c["kind"] == "dec1" and c["depth_target"] == d and D.low_of(c) is not None:
            lo1 = min(lo1, D.low_of(c))
    proj = lo1 - (0.8 if extra4 is None else max(extra4, 0.2) * 1.5)
    if proj < D.FLOOR_GIB:
        D.log(f"four: SKIP 4-stream at {d}: projected low {proj:.2f} GiB < {D.FLOOR_GIB} (1-stream low {lo1}, extra4 {extra4})")
        D.DOC.setdefault("skipped4_pass2", []).append(d)
        D.save()
        break
    r4 = D.cell(f"d{d}-s4", d, [D.SALT + h + D.task(f"d{d}-s4-{s}") for s in range(4)], D.DEC_TOKENS, True, "dec4")
    lo4 = D.low_of(r4)
    if lo4 is not None:
        extra4 = max(extra4 or 0.0, lo1 - lo4)
    if not D.ok(r4):
        D.log(f"four: 4-stream failed at {d}; stop")
        break
# final drift check: the short 1-stream cell once more
if D.SHORT in texts:
    for rep in range(2):
        D.cell(f"final-d{D.SHORT}-s1-r{rep}", D.SHORT, [D.SALT + texts[D.SHORT] + D.task(f"end-d{D.SHORT}-s1-r{rep}")],
               D.DEC_TOKENS, True, "end1")
D.DOC["four_finished"] = time.time()
D.save()
D.log("four: end")
