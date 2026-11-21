"""Tabulate the draft-block arms against the five control arms. Run from the recipe root:

  python3 -I runs/draft-block-20261118/summary.py

Single-stream rates are the engine's own (the done line's tok_per_s); burst rates are tokens over wall time.
"""
import json
import statistics as st

D = "runs/draft-block-20261118"
controls = ["a0-off", "a1-off", "c0-off", "c1-off", "d0-off"]
arms = {"always-wide 6": "b6", "always-wide 7": "b7", "refined 6": "d6", "refined 7": "d7"}


def load(arm):
    return json.load(open(f"{D}/{arm}/probe.json"))


C = {a: load(a) for a in controls}
A = {k: load(v) for k, v in arms.items()}
n = len(C["a0-off"])
print("hash mismatches against control:", sum(1 for k in A for i in range(n) if A[k][i]["sha"] != C["a0-off"][i]["sha"]),
      "| among controls:", sum(1 for a in controls for i in range(n) if C[a][i]["sha"] != C["a0-off"][i]["sha"]))
print(f"{'cell':22s} {'ctl tok/round':>13s} {'ctl tok/s, 5 arms':>20s} | " + " | ".join(f"{k:>22s}" for k in A))
for i in range(n):
    r0 = C["a0-off"][i]
    burst = "width" in r0
    rate = (lambda r: r["agg_tok_s_wall"]) if burst else (lambda r: float(r["tok_per_s"]))
    per = (lambda r: r["tokens_per_stream_round"]) if burst else (lambda r: r["exact_tokens_per_round"])
    cs = [rate(C[a][i]) for a in controls]
    m = st.mean(cs)
    name = r0["label"] if burst else f"{r0['label']} s{r0['seed']}"
    cells = [f"{per(A[k][i]):.2f} {rate(A[k][i]):6.1f} ({100 * (rate(A[k][i]) / m - 1):+.1f}%)" for k in A]
    print(f"{name:22s} {per(r0):13.2f} {min(cs):9.1f}-{max(cs):<6.1f} | " + " | ".join(f"{c:>22s}" for c in cells))
for k, rows in list(A.items()) + [("control", C["a0-off"])]:
    g = [r["exact_tokens_per_round"] for r in rows if r["label"] == "tetris greedy"]
    s = [r["exact_tokens_per_round"] for r in rows if r["label"] == "tetris sampled"]
    print(f"{k}: Tetris tokens a round, greedy {g[0]:.2f}, sampled mean {st.mean(s):.2f} [{min(s):.2f}-{max(s):.2f}]")
