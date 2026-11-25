"""k296: per-row probe comparison (tok/s, rounds, tokens/round, sha match) across arms. python3 probe_table.py ctl1 conf_a ..."""
import json, sys
arms = sys.argv[1:]; D = {a: {r["label"]: r for r in json.load(open(f"{a}.probe.json"))} for a in arms}
labels = list(D[arms[0]])
print("| row | " + " | ".join(f"{a} tok/s (rounds, tok/rd)" for a in arms) + " | sha = " + arms[0] + " |")
print("|---|" + "---:|" * len(arms) + "---|")
tot = {a: [0.0, 0.0] for a in arms}
for l in labels:
    cells, same = [], []
    for a in arms:
        r = D[a].get(l)
        if not r: cells.append("-"); continue
        cells.append(f'{r.get("tok_per_s")} ({r.get("rounds")}, {r.get("tokens_per_round")})')
        same.append("y" if r.get("sha") == D[arms[0]][l].get("sha") else "N")
        try: tot[a][0] += float(r["completion_tokens"]); tot[a][1] += float(r["decode_s"])
        except Exception: pass
    print(f"| {l} | " + " | ".join(cells) + " | " + "".join(same) + " |")
print("| **total tokens / total decode_s** | " + " | ".join(f"{t[0]/t[1]:.2f}" if t[1] else "-" for t in tot.values()) + " | |")
