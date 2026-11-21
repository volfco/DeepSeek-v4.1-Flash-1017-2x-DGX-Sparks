"""Recover the exact immutable input size from the recorded 500K history digest."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parents[1]/"tools"))
import bench_div

source=bench_div.Corpus(str(here.parent/"combo4/corpus")).text("div",550000,"review-chat-history",fixed=True).replace("<","(")
events=[json.loads(l) for l in (here.parent/"engine-followup/runtime01/current-a-cache.jsonl").open()]
salt=events[0]["salt"]
key=next(q["history_sha256"] for q in events if q["kind"]=="history")
prefix=f"[review-chat-{salt}]\nUse these engineering notes in the review.\n".encode()
h=hashlib.sha256(prefix)
size=len(prefix)
for char in source:
    b=char.encode()
    h.update(b)
    size+=len(b)
    if h.hexdigest()==key:break
else:raise ValueError("Cannot reproduce the actual warm history")
samples=list(csv.DictReader((here.parent/"engine-followup/cache.csv").open()))
off=statistics.median(float(r["off_encode"]) for r in samples)
on=statistics.median(float(r[k]) for r in samples for k in ("on_a_encode","on_b_encode"))
print(json.dumps(dict(history_bytes=size,sha256=key,off_encode_median_s=off,on_encode_median_s=on,
                      off_input_MB_s=size/off/1e6,on_input_MB_s=size/on/1e6),indent=2))
