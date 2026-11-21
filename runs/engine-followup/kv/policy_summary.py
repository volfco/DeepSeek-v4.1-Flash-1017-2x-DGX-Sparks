"""Compare the three held-prefix-wait arms from the head campaign's combined output."""
import argparse
import collections
import json
import pathlib
import sys

root=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(root))
import summarize


def analyze(path):
    rows=summarize.load(path)
    if rows[-1]["kind"]!="complete" or not any(r["kind"]=="restored" for r in rows):
        raise ValueError("Incomplete policy run")
    result=[]
    for arm in [r["arm_name"] for r in rows if r["kind"]=="runtime"]:
        own=[r for r in rows if r["arm"]==arm]
        cells=[r for r in own if r["kind"]=="cell"]
        if any(not r["clean"] for r in cells):
            raise ValueError(arm+" has competing work")
        peak,=[c for c in cells if c["sharing"]=="peak"]
        stats=summarize.burst(peak)
        ref={c["sharing"].removeprefix("pre-pressure-"):c for c in cells if c["sharing"].startswith("pre-pressure-")}
        revisits=[c for c in cells if c["sharing"].startswith("after-pressure-")]
        if len(ref)!=3 or len(revisits)!=3:
            raise ValueError(arm+" missing independent history revisit")
        probe_results=[]
        for c in revisits:
            key=c["sharing"].removeprefix("after-pressure-"); a=ref[key]["requests"][0]; b=c["requests"][0]
            if a["input_sha256"]!=b["input_sha256"]:
                raise ValueError(arm+" revisit input changed")
            counter="disk_hits_committed"
            disk=c["health_after"]["kv_counts"]["cache"][counter]-c["health_before"]["kv_counts"]["cache"][counter]
            probe_results.append({"key":key,"cached":b["cached"],"disk":disk,"output_equal":a["done"]["sha"]==b["done"]["sha"],"restore_s":float(b["done"]["restore_s"]),
                                  "added_ttft_s":round(b["ttft"]-a["ttft"],4)})
        events,=[r["events"] for r in own if r["kind"]=="patient"]
        counts=collections.Counter((e["part"],e["outcome"],e["patient"]) for e in events)
        result.append({"arm":arm,"peak":stats,"revisits":probe_results,
                       "placement_event_counts":[{"part":part,"outcome":out,"patient":patient,"count":v}
                          for (part,out,patient),v in sorted(counts.items())],
                       "resumed_after_ms":[e["elapsed_ms"] for e in events if e["outcome"]=="resumed"]})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("path",type=pathlib.Path)
    print(json.dumps(analyze(p.parse_args().path),indent=2))
