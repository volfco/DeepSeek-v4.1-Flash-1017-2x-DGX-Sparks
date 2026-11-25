"""GPU-clock round-time table for an arms.sh suite (replaces table.py's host metric for scoring).

  python3 runs/overnight-20261122/table_gpu.py <OUT dir> --control a,b --test c,d

Why not table.py: its `forward_ms_per_round` is differenced from a lifetime mean the engine prints to 0.01 ms, so a
~100-round cell is off by up to +-0.6 ms once the probe has run first (continue.md, "Scoring artifacts"). This reads
the same <arm>.json but uses the forward's GPU clock: `per_round.tail` (us a round the stretches' kernels ran).
`per_round.wait` (GPU idle between stretches) is kept out of the metric and listed as host stalls instead: a
one-off 70-100 ms stall in a cell is a real event but not the kernels' time.

- One-stream cells: GPU-clock ms a round (rounds are identical across arms for a greedy or seeded request).
- Multi-stream cells: the round count varies between identical arms, so per-round time is confounded. The table
  gives total forward ms (rounds x GPU ms a round) and wall seconds instead.
- Then, from <arm>.profile.txt (rank 0's lifetime profile line): pass_ms, absorb_ms, logits_ms and forward_ms, the
  only place drafter-pass patches show. markov_cache counters are printed when present.
A test value is flagged only when it lies outside the controls' range; with two controls that is weak evidence, so
read the sign across cells and arms.
"""
import argparse, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--control", required=True)
ap.add_argument("--test", required=True)
a = ap.parse_args()
ctl, tst = a.control.split(","), a.test.split(",")
names = ctl + tst
arms = {n: json.load(open(os.path.join(a.out, f"{n}.json"))) for n in names}


def cell(n, i):
    c = arms[n]["cells"][i]
    pr = c["per_round"]
    tail = (pr.get("tail") or [0, 0])[0]
    wait = (pr.get("wait") or [0, 0])[0]
    gpu = tail / 1000 if tail else None   # None: this image recorded no clock
    return {"rounds": pr["rounds"], "gpu": gpu, "host": pr.get("forward_ms_per_round"), "wait": wait / 1000, "wall": c["wall_s"]}


def row(label, vals, fmt):
    cm = sum(vals[n] for n in ctl) / len(ctl)
    tm = sum(vals[n] for n in tst) / len(tst)
    lo, hi = min(vals[n] for n in ctl), max(vals[n] for n in ctl)
    out = sum(not (lo <= vals[n] <= hi) for n in tst)
    print(f"| {label} | " + " | ".join(fmt % vals[n] for n in names)
          + f" | {100 * (tm - cm) / cm:+.2f}% | {100 * (hi - lo) / cm:.2f}% | {out}/{len(tst)} |")


def head(what):
    print(f"\n| {what} | " + " | ".join(names) + " | test vs control | control spread | test outside control range |")
    print("|---|" + "---:|" * (len(names) + 3))


cells = arms[ctl[0]]["cells"]
have_clock = all(cell(n, i)["gpu"] is not None for n in names for i in range(len(cells)))
metric = "gpu" if have_clock else "host"
if not have_clock:
    print("NOTE: no GPU clock in these arms (older image); falling back to the quantized host metric.")

head(f"One-stream cells: forward ms a round ({'GPU clock' if have_clock else 'host'})")
for i, c in enumerate(cells):
    if c["streams"] != 1:
        continue
    v = {n: cell(n, i) for n in names}
    rounds = {v[n]["rounds"] for n in names}
    tag = "" if len(rounds) == 1 else f" (rounds differ: {sorted(rounds)})"
    row(f"{c['depth'] // 1024}K × 1{tag}", {n: v[n][metric] for n in names}, "%.2f")

head("Multi-stream cells: total forward ms (rounds × ms a round)")
for i, c in enumerate(cells):
    if c["streams"] == 1:
        continue
    v = {n: cell(n, i) for n in names}
    row(f"{c['depth'] // 1024}K × {c['streams']}", {n: v[n]["rounds"] * v[n][metric] for n in names}, "%.0f")
print("\nrounds: " + "; ".join(f"{c['depth'] // 1024}K×{c['streams']} " + "/".join(str(cell(n, i)["rounds"]) for n in names)
                                 for i, c in enumerate(cells) if c["streams"] != 1))

head("Wall seconds a cell (includes the cell's prefill)")
for i, c in enumerate(cells):
    row(f"{c['depth'] // 1024}K × {c['streams']}", {n: cell(n, i)["wall"] for n in names}, "%.3f")

stalls = [(n, f"{c['depth'] // 1024}K×{c['streams']}", cell(n, i)["wait"]) for i, c in enumerate(cells) for n in names if cell(n, i)["wait"] > 0.3]
if stalls:
    print("\nhost stalls (GPU idle between stretches, ms a round; a one-off event in that cell): "
          + ", ".join(f"{n} {w} {x:.2f}" for n, w, x in stalls))

# per-round distributions (0299): per-cell forward/pass/wait as mean/median/p95/p99 from the cell's per-round
# samples (per_round.samples, one sample a scrape interval). Cells saved before the samples existed show the one
# per-round mean they kept (per_round.tail[0] / wait[0] us a round) and "-" for the quantiles.
def quant(s, p):
    s = sorted(s)
    return s[max(0, -(-int(p * 100) * len(s) // 100) - 1)]


def dist_cell(n, i, metric):
    pr = arms[n]["cells"][i]["per_round"]
    s = (pr.get("samples") or {}).get(metric + "_ms") or []
    if s:
        return f"{sum(s) / len(s):.2f}/{quant(s, 0.5):.2f}/{quant(s, 0.95):.2f}/{quant(s, 0.99):.2f}"
    if metric == "forward" and pr.get("tail"):
        return f"{pr['tail'][0] / 1000:.2f}/-/-/-"
    if metric == "wait" and pr.get("wait"):
        return f"{pr['wait'][0] / 1000:.2f}/-/-/-"
    return "-/-/-/-"


print("\nPer-round distributions (ms; mean/median/p95/p99; forward = the GPU clock's stretches a round, wait = GPU "
      "idle between stretches a round, pass = the drafter pass a pass):")
print("| cell | metric | " + " | ".join(names) + " |")
print("|---|---|" + "---:|" * len(names))
for i, c in enumerate(cells):
    for metric in ("forward", "pass", "wait"):
        print(f"| {c['depth'] // 1024}K × {c['streams']} | {metric} | " + " | ".join(dist_cell(n, i, metric) for n in names) + " |")

# rank 0's lifetime profile line: the pass, absorb and pick, which the forward metric cannot see
prof = {}
for n in names:
    p = os.path.join(a.out, f"{n}.profile.txt")
    if not os.path.exists(p):
        continue
    for line in open(p):
        i = line.find('{"rank": 0, ')
        if i < 0:
            continue
        try:
            o = json.loads(line[i:])
        except ValueError:
            continue
        for k in ("profile", "markov_cache", "pass_counts", "pass_phases_ms"):
            if k in o:
                prof.setdefault(n, {})[k] = o[k]
if all("profile" in prof.get(n, {}) for n in names):
    for key, fmt in (("pass_ms", "%.2f"), ("absorb_ms", "%.2f"), ("logits_ms", "%.2f"), ("forward_ms", "%.2f"), ("rows_a_round", "%.2f"), ("streams_a_pass", "%.2f")):
        if key == "pass_ms":
            head("Rank 0 lifetime profile (all of the arm's traffic)")
        row(key, {n: prof[n]["profile"][key] for n in names}, fmt)
    base = 0xAAAAAAAAAAAAAAAA   # 0267's counters started at Zig's undefined fill; later builds start at 0
    if all("markov_cache" in prof[n] for n in names):
        fix = lambda x: x - base if x >= base else x
        print("\nmarkov bias cache miss rate: " + ", ".join(
            f"{n} {100 * fix(prof[n]['markov_cache']['misses']) / max(1, fix(prof[n]['markov_cache']['misses']) + fix(prof[n]['markov_cache']['hits'])):.1f}%" for n in names))
    for k in ("pass_counts", "pass_phases_ms"):
        for n in names:
            if k in prof[n]:
                print(f"{n} {k}: {json.dumps(prof[n][k])}")
for n in names:
    p = os.path.join(a.out, f"{n}.cmp.txt")
    if os.path.exists(p):
        print(f"exact {n}: {open(p).read().strip().splitlines()[-1]}")
