"""Check matched restore preparation and summarize passive clocks inside the two all-warm bursts."""
import csv
import datetime
import json
from pathlib import Path
import statistics

here = Path(__file__).resolve().parent
offset = float((here / "committed-warm-clock-offset.txt").read_text())
crosscheck = json.loads((here.parent / "engineering02/trim/clock.json").read_text())
assert all(abs(offset - v) < .02 for v in crosscheck)
rows = [json.loads(l) for l in (here / "committed-warm01.log").read_text().splitlines() if l.startswith("{")]
out = []
names = ("fill-f0", "fill-f3", "pre-pressure-f3", "fill-shared", "pre-pressure-shared", "fill-f1", "pre-pressure-f1")
reference = None
for arm in ("guard-warm", "plain-warm"):
    chosen = {r["sharing"]: r for r in rows if r["kind"] == "cell" and r.get("arm") == arm}
    peak = chosen["peak"]
    prep = [chosen[name]["requests"][0] for name in names]
    signature = [(q["input_sha256"], q["cached"], q["tokens"], q["done"]["sha"]) for q in prep]
    if reference is None:
        reference = signature
    assert signature == reference
    assert all(q["cached"] == (296960 if name.endswith("f3") else 299008) for name,q in zip(names, prep))
    q = peak["requests"]
    start = min(x["start"] for x in q) + offset + 1
    end = max(x["start"] + x["total"] for x in q) + offset - 1
    clocks = {}
    for rank in ("head", "worker"):
        rec = []
        with (here.parent / f"hardware01/serving-duty/{rank}-nvml.csv").open() as stream:
            for v in csv.reader(stream, skipinitialspace=True):
                timestamp = datetime.datetime.strptime(v[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc)
                if start <= timestamp.timestamp() <= end:
                    rec.append([int(v[1]), int(v[3])])
        assert len(rec) >= 30
        clocks[rank] = {"samples": len(rec), "median_busy_pct": statistics.median(r[0] for r in rec),
                        "median_sm_mhz": statistics.median(r[1] for r in rec),
                        "min_sm_mhz": min(r[1] for r in rec), "max_sm_mhz": max(r[1] for r in rec)}
    out.append(dict(arm=arm, prepare_sum_s=sum(z["total"] for z in prep),
        prepare=[dict(name=name,cached=z["cached"],ttft_s=z["ttft"],total_s=z["total"]) for name,z in zip(names,prep)],
        rows_before=[p["kept_rows"] for p in peak["health_before"]["per_part"]],
        clocks=clocks,start_utc=start,end_utc=end,rounds=[r["done"]["rounds"] for r in q]))
print(json.dumps(out, indent=2))
