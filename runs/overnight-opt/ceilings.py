"""Price identical metered decode work under the concurrent calibration and independent read roofs."""
import csv
import json
from pathlib import Path
import statistics
import sys

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here.parent/"engine-followup"))
import analyze
from windows import bounds


def rates(path):
    with path.open() as f:
        return [int(size)/(int(end)-int(start)) for _,start,end,size in csv.reader(f)]


def compute():
    directory=here/"serving-meter01"
    refs={r:(rates(here/f"cal-read-{r}/scan.csv")[0],
             statistics.mean(rates(here/f"hardware01/{r}-nvml-scan.csv"))) for r in ("head","worker")}
    start,end=bounds(directory/"meter")
    offset=statistics.mean(json.loads((directory/"clock.json").read_text()))
    active={}
    for c in analyze.cells(directory/"workload.jsonl"):
        if c["cell"]=="chat-prime":continue
        samples=[q for q in analyze.active_segment(c) if start<(q["t"]+offset)*1e9<end]
        a,b=samples[0],samples[-1]
        key=(c["depth"],len(c["requests"]),c["rep"])
        active[key]=(b["generation_tokens_running"]-a["generation_tokens_running"],b["t"]-a["t"])
    with (directory/"windows.csv").open() as f:
        for x in csv.DictReader(f):
            row={k:int(x[k]) for k in ("depth","width","rep")}
            tokens,duration=active[(row["depth"],row["width"],row["rep"])]
            assert abs(duration-float(x["meter_active_s"]))<.001
            row["meter_tokens"]=tokens
            for rank,roofs in refs.items():
                row[f"{rank}_read_GB_per_output"]=float(x[f"{rank}_read_GB_s"])*duration/tokens
                for label,roof in zip(("concurrent_cal","repeat_mean"),roofs):
                    row[f"{rank}_{label}_pct"]=100*float(x[f"{rank}_read_GB_s"])/roof
            for j,label in enumerate(("concurrent_cal","repeat_mean")):
                floor=max(1000*float(x[f"{r}_read_GB_round"])/refs[r][j] for r in refs)
                row[label+"_read_floor_ms"]=floor
                row[label+"_read_only_tok_s"]=tokens/float(x["meter_rounds"])*1000/floor
            yield row


if __name__=="__main__":
    values=list(compute())
    writer=csv.DictWriter(sys.stdout,fieldnames=list(values[0]))
    writer.writeheader()
    writer.writerows(values)
