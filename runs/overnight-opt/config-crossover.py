"""Separate the pinned short-decode regression, token cache and keyed drafting on the same warm workload."""
import fcntl
import importlib
from pathlib import Path
import sys

here = Path(__file__).resolve().parent
sys.path.insert(0, str(here))
import buckets
import paired


def run(out):
    out.mkdir()
    initial = buckets.driver.inspect()
    if any(r["tag"] != buckets.driver.base or any(r["flags"][k] for k in buckets.driver.trial.candidate) for r in initial):
        raise RuntimeError("Expected original pinned configuration")
    buckets.write_json(out / "initial.json", initial)
    controls = [out / "short-control-a", out / "short-control-return"]
    arms = [
        ("short-control-a", ["--deployed"]),
        ("short-token", ["--switch-profile", "token"]),
        ("short-host", ["--switch-profile", "host"]),
        ("short-eight", ["--keyed", "1"]),
        ("short-unkeyed", ["--keyed", "0"]),
        ("short-control-return", ["--deployed"]),
    ]
    try:
        for label, switches in arms:
            opts = [str(out / label), "--graphs", "--serving-drafts", "--private-spill", "--boundaries",
                    "--depths", "3072", "--widths", "1", "4", "--reps", "2",
                    "--solo-budget", "384", "--burst-budget", "512", "--prefill-panel", "3072", "8192", *switches]
            buckets.run(buckets.options().parse_args(opts), baseline=initial)
        comparisons = {label: list(paired.compare(out / label, controls)) for label, _ in arms[1:-1]}
        comparisons["keyed-off-vs-on"] = list(paired.compare(out / "short-unkeyed", [out / "short-eight"]))
        buckets.write_json(out / "paired.json", comparisons)
        screen_results = importlib.import_module("screen-results")
        candidates = []
        prefill = {}
        for label, switches in arms[1:-1]:
            left,right=(list(screen_results.compare(control/"prefill",out/label/"prefill")) for control in controls)
            paired_prefill=[]
            for a,c in zip(left,right,strict=True):
                if (a["depth"],a["rep"])!=(c["depth"],c["rep"]) or a["base_cut"] or c["base_cut"] or a["arm_cut"]:
                    raise ValueError("Incompatible cold work")
                rates = [float(x["base_prefill_s"])/float(x["arm_prefill_s"])-1 for x in (a,c)]
                paired_prefill.append(dict(depth=a["depth"],rep=a["rep"],seconds_A_B_A=[float(a["base_prefill_s"]),
                         float(a["arm_prefill_s"]),float(c["base_prefill_s"])],gain_vs_best_control=min(rates),
                         output_matched=a["same_output"] and c["same_output"]))
            prefill[label]=paired_prefill
            effects = []
            for depth in (3072,8192):
                rows = [r for r in paired_prefill if r["depth"]==depth]
                if len(rows)==2 and all(r["output_matched"] for r in rows):
                    effects.append(min(r["gain_vs_best_control"] for r in rows))
            for width in (1,4):
                rows=[r for r in comparisons[label] if r["width"]==width]
                if len(rows)==2 and all(r["equal_hashes"]==r["outputs"] and r["same_start_occupancy"] for r in rows):
                    effects.append(min(r["arm_tok_s"]/r["control_max_tok_s"]-1 for r in rows))
            if effects and max(effects)>.03:
                candidates.append((max(effects),label,switches))
        buckets.write_json(out/"prefill.json",prefill)
        if candidates:
            _,label,switches=max(candidates)
            buckets.write_json(out/"expanded.json",dict(candidate=label,gate=max(candidates)[0]))
            for name,flags in (("full-a",["--deployed"]),("full-candidate",switches),("full-return",["--deployed"])):
                opts=[str(out/name),"--graphs","--serving-drafts","--private-spill","--boundaries","--depths",
                      "32000","100000","--widths","1","4","--reps","2","--solo-budget","384","--burst-budget","512",
                      "--prefill-panel","32000","100000",*flags]
                buckets.run(buckets.options().parse_args(opts),baseline=initial)
            buckets.write_json(out/"expanded-pairs.json",list(paired.compare(out/"full-candidate",[out/"full-a",out/"full-return"])))
    finally:
        buckets.restore(out, initial)


if __name__ == "__main__":
    with (buckets.followup / ".pair.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(Path(sys.argv[1]).resolve())
