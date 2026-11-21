"""One arm's timing: the Nov 20 short probe (greedy Tetris/code 1024 + 4-stream sampled Tetris 768, twice), then one
~100K-token prompt (greedy, 256 tokens, thinking off) so the candidate-source pool (0195: > 16,384 compressed score
columns) and the prompt-path 0197/0198 kernels run. The long prompt carries a per-arm salt at its start so no prefix
cache (memory or Engram) can serve it; its reply differs per arm, so only prefill_s and ms/round compare.
  python3 runs/fused-audit-bench-20261120/probe.py <out.json> <salt>"""
import json, sys, time, urllib.request, subprocess
from pathlib import Path

sys.dont_write_bytecode = True
source = Path("runs/draft-block-20261118/probe.py")
definitions, _ = source.read_text().rsplit("\nmain()", 1)
p = {}
exec(compile(definitions, str(source), "exec"), p)
sys.path.insert(0, "tools")
import bench_div as bd  # noqa: E402

out, salt = sys.argv[1], sys.argv[2]
rows = []

def save(row):
    rows.append(row)
    Path(out).write_text(json.dumps(rows, indent=1) + "\n")
    print({k: v for k, v in row.items() if k not in ("per_round", "hist")}, flush=True)

plan = [("run", "tetris greedy", p["TETRIS"], 1024), ("run", "code greedy", p["CODE"], 1024),
        ("burst", "tetris sampled x4", 4, 768)] * 2
for kind, label, arg, n in plan:
    row = p["run"](label, arg, False, 1, n) if kind == "run" else p["burst"](label, arg, 810, n)
    save(row)
    if row.get("failed"):
        sys.exit(1)

hay = bd.Corpus("runs/overnight/corpus").text("div", 100_000, "fused-audit-hay", fixed=True)
prompt = f"[{salt}] " + hay + "\n\nSummarize the text above section by section, quoting it."
t0 = time.time()
row = p["run"]("long 100K greedy", prompt, False, 1, 256)
row["client_s"] = round(time.time() - t0, 2)
log = subprocess.run(["docker", "logs", "--since", "10m", "dsv41-1017-exl3-tf"], capture_output=True, text=True)
line = next((l for l in (log.stdout + log.stderr).splitlines() if " done " in l and "prefill_s=" in l and row.get("sha") and row["sha"] in l), "")
f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
row.update(prefill_s=f.get("prefill_s"), prompt_tokens=f.get("prompt_tokens") or f.get("prompt"), cached=f.get("cached"), done_line=line[-400:])
save(row)
