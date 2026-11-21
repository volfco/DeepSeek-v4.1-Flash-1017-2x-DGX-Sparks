"""Interleave L1-resident and DRAM streaming reads outside serving cores, retaining fixed work."""
import contextlib
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here.parent / "engine-followup"))
import analyze
import buckets


@contextlib.contextmanager
def readers(out, label, mode):
    jobs = []
    with contextlib.ExitStack() as resources:
        try:
            for rank, prefix in (("head", []), ("worker", ["ssh", "-T", "-o", "BatchMode=yes", "10.50.0.11"])):
                path = out / f"{label}-{rank}.csv"
                data = resources.enter_context(path.open("x"))
                errors = resources.enter_context(path.with_suffix(".err").open("x"))
                process = subprocess.Popen(prefix+["/tmp/dsv41-cpu-read", mode, "3"], stdout=data, stderr=errors, stdin=subprocess.PIPE)
                jobs.append((path, process))
            deadline = time.monotonic() + 15
            while any(len(path.read_text().splitlines()) < 4 for path, _ in jobs):
                if any(p.poll() is not None for _, p in jobs) or time.monotonic() >= deadline:
                    raise RuntimeError(f"CPU readers failed to warm on both ranks: {label}")
                time.sleep(.1)
            yield
        finally:
            for _, process in jobs:
                process.stdin.close()
            for path, process in jobs:
                try:
                    status = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
                    raise RuntimeError(f"{path}: reader did not exit on pipe closure")
                if status:
                    raise RuntimeError(f"{path}: CPU read control exited {status}")


def score(path, depth):
    by_key = {}
    for line in path.read_text().splitlines():
        c = json.loads(line)
        if c["kind"] != "cell" or c["depth"] != depth or c["cell"] == "chat-prime":
            continue
        samples = analyze.active_segment(c)
        if not c["clean"] or len(samples) < 2:
            raise ValueError("Invalid intervention cell")
        a,b = samples[0], samples[-1]
        duration = b["t"]-a["t"]
        rounds = sum(b[k]-a[k] for k in ("kv_shard_fetch_rounds","kv_shard_owner_rounds"))
        if rounds <= 0 or duration < 2:
            raise ValueError("Intervention has insufficient sustained rounds")
        req = c["requests"]
        identity = [(r["input_sha256"],r["cached"],r["tokens"],r["done"]["sha"]) for r in req]
        rows = by_key.setdefault((len(req),c["rep"]), {})
        rows[c["sharing"]] = dict(ms=duration*1000/rounds, rounds=rounds, identity=identity,
                                  tok_s=(b["generation_tokens_running"]-a["generation_tokens_running"])/duration)
    result = []
    for (width, rep), cells in sorted(by_key.items()):
        a, b, c = (cells[x] for x in ("cache-a","stream","cache-b"))
        matched = a["identity"] == b["identity"] == c["identity"]
        result.append(dict(depth=depth,width=width,rep=rep,matched=matched,arms=cells,
                           above_noise=matched and b["ms"] > max(a["ms"],c["ms"])*1.03))
    return result


def measure(client, corpus, out):
    for prefix in buckets.driver.prefixes:
        mask = subprocess.check_output(prefix+["docker","inspect","-f","{{.HostConfig.CpusetCpus}}",
                                              buckets.driver.container],text=True).strip()
        if not mask or buckets.cpu_numbers(mask) & {0,1,2}:
            raise ValueError(f"Serving and load CPUs overlap or are unbounded: {mask}")
    source = corpus.text("div", 360000, "fixed-boundary-1118", fixed=True)
    for depth, repetitions in ((3072, 1), (300032, 2)):
        if depth > 3072 and not any(r["above_noise"] for r in comparison):
            break
        prompts, context = buckets.fixture(client, source, depth)
        client.cell(depth,"chat-prime",[context+"Prime this history."],[4199],budget=16)
        for arm, mode in (("cache-a","cache"),("stream","stream"),("cache-b","cache")):
            with readers(out, f"{depth}-{arm}", mode):
                for rep in range(repetitions):
                    for width in ((1,4) if rep%2 == 0 else (4,1)):
                        client.cell(depth, "solo" if width==1 else "admission", prompts[:width],
                                    list(range(4200+rep*10,4200+rep*10+width)),
                                    budget=384 if width==1 else 512,rep=rep,sharing=arm,
                                    expected_cached=depth//2048*2048)
        comparison = score(out/"workload.jsonl",depth)
        (out/f"{depth}-comparison.json").write_text(json.dumps(comparison,indent=2)+"\n")
