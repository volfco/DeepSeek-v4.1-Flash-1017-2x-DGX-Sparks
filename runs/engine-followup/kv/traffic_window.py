"""Count only numeric completion fields in the archive between the published T15 and admission-grace campaigns."""
import collections
import gzip
import json
import pathlib

start = "2026-11-18T00:15:02"
end = "2026-11-18T00:47:28"
path = pathlib.Path.home()/".cache/dsv41-tensorfold/logs/20261118-004742-rank0.log.gz"
counts = collections.Counter()
first = last = None
with gzip.open(path,"rt",errors="replace") as f:
    for line in f:
        stamp=line[:19]
        if not (stamp.startswith("2026-") and len(stamp)==19):
            continue
        first=stamp if first is None else min(first,stamp)
        last=stamp if last is None else max(last,stamp)
        if not start <= stamp < end or "done " not in line or "prefill_s=" not in line:
            continue
        d=dict(p.split("=",1) for p in line.split() if "=" in p)
        if {"prompt","cached","tokens","queue_s","prefill_s"} <= d.keys():
            counts["completed"] += 1
            counts["deep_100k"] += int(d["prompt"]) >= 100000
            counts["queue_ge_1s"] += float(d["queue_s"]) >= 1
            counts["prefill_ge_1s"] += float(d["prefill_s"]) >= 1
if first is None or first>start or last<end:
    raise SystemExit(f"Archive source does not cover the whole window ({first} to {last})")
print(json.dumps({"start":start+"Z","end":end+"Z","source":"rank-0 timestamped archive","completed":counts["completed"],"counts":dict(counts)}))
