"""Upper bounds on exact resends using only lengths and timings in the external-only interval."""
from collections import Counter
from datetime import datetime
import hashlib
import json
import re
from pathlib import Path

here = Path(__file__).resolve().parent
archive = here.parents[1] / "overnight" / "break-1600Z.rank0.log"
rows = []
ledger = Counter()
counters = ("hits", "misses", "evicted", "refused", "failed")
samples = []
opening = None
pattern = re.compile(
    r"(\d+) states, (\d+) MiB and (\d+) MiB spare of (\d+) MiB "
    r"\(hits (\d+), misses (\d+), evicted (\d+), refused (\d+), failed (\d+)\)"
)
for line in archive.open():
    sample = pattern.search(line)
    if sample:
        count, charged_mib, _, budget_mib, *values = map(int, sample.groups())
        stamp = line.split()[0]
        current = {"at": stamp, "states": count, **dict(zip(counters, values))}
        if stamp[:19] < "2026-11-13T07:20:00":
            opening = current
        else:
            if budget_mib != 512 or charged_mib != count * 6684672 // 2**20:
                raise ValueError("Archive ledger used another cache geometry or budget")
            samples.append(current)
            ledger[count] += 1
    if line[:19] < "2026-11-13T07:20:00":
        continue
    if '127.0.0.1 "POST /v1/chat/completions' in line:
        raise ValueError("External-only interval included a local chat request")
    if "[tensorfold] done " not in line:
        continue
    fields = dict(x.split("=", 1) for x in line.split() if "=" in x)
    end = datetime.fromisoformat(line.split()[0].replace("Z", "+00:00")).timestamp()
    rows.append(dict(prompt=int(fields["prompt"]), prefill=float(fields["prefill_s"]),
                     first=end-float(fields["decode_s"]), end=end))
rows.sort(key=lambda r: r["first"])
seen = set()
lengths = Counter(r["prompt"] for r in rows)
result = {"archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(), "counts": []}
for index, row in enumerate(rows):
    row["earlier_same_length"] = row["prompt"] in seen
    row["two_preceding_lengths"] = any(q["prompt"] == row["prompt"] for q in rows[max(0, index-2):index])
    seen.add(row["prompt"])
for cut in (0, 8192, 100000, 250000):
    selected = [r for r in rows if r["prompt"] >= cut]
    potential = [r for r in selected if r["earlier_same_length"]]
    result["counts"].append(dict(minimum_prompt_tokens=cut, completed=len(selected),
        unique_lengths=len({r["prompt"] for r in selected}), prior_equal_length=len(potential),
        any_equal_length_in_window=sum(lengths[r["prompt"]] > 1 for r in selected),
        previous_two_first_token_lengths=sum(r["two_preceding_lengths"] for r in selected),
        sum_prefill_s=round(sum(r["prefill"] for r in selected), 3),
        sum_prefill_s_of_possible_repeats=round(sum(r["prefill"] for r in potential), 3)))
result["ordinary_cache_ledger"] = {"samples": sum(ledger.values()), "min_states": min(ledger),
    "max_states": max(ledger), "above_70_charged_slots": sum(v for k, v in ledger.items() if k > 70),
    "at_least_78_states": sum(v for k, v in ledger.items() if k >= 78), "budget_mib": 512}
def ledger_deltas(first, remaining):
    previous = first
    totals = Counter()
    resets = []
    intervals_with_evictions = evicted_above_70 = evicted_at_least_78 = 0
    for row in remaining:
        reset = any(row[k] < previous[k] for k in counters)
        if reset:
            resets.append(row["at"])
        delta = {k: row[k] - (0 if reset else previous[k]) for k in counters}
        totals.update(delta)
        intervals_with_evictions += delta["evicted"] > 0
        evicted_above_70 += delta["evicted"] if row["states"] > 70 else 0
        evicted_at_least_78 += delta["evicted"] if row["states"] >= 78 else 0
        previous = row
    return dict(added=dict(totals), resets=resets, intervals_with_evictions=intervals_with_evictions,
                evictions_on_intervals_ending_above_70_states=evicted_above_70,
                evictions_on_intervals_ending_at_least_78_states=evicted_at_least_78)

assert opening and samples
result["ordinary_cache_ledger"].update(pre_window_sample=opening, first_window_sample=samples[0],
    final_sample=samples[-1], from_pre_window=ledger_deltas(opening, samples),
    strictly_within_window=ledger_deltas(samples[0], samples[1:]))
result["scope"] = ("A full identical prompt needs equal length. Cohort only. Initial resident state, "
                   "failed or missing requests and other periods are unknown. Approximate ordering "
                   "uses first token; the all-window duplicate count needs no timing assumption. "
                   "No token or content identities were recorded.")
print(json.dumps(result, indent=2))
