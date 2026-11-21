"""Compare identical off-grid continuation scores and actual reused positions against independent cold scoring."""
import argparse
import csv
import json
from pathlib import Path
import sys


def read(directory):
    results={}
    for line in (directory/"resume-score.jsonl").read_text().splitlines():
        row=json.loads(line)
        if row["source"] in results:raise ValueError("Repeated source in scoring ledger")
        results[row["source"]]=row
    if len(results)!=4:raise ValueError("Expected four independent scored sources")
    return results


def compare(cold,arm):
    reference=read(cold);candidate=read(arm)
    if reference.keys()!=candidate.keys():raise ValueError("Score sources differ")
    for source,base in reference.items():
        q=candidate[source];a,b=base["score"],q["score"]
        if base["cached"] or base["ids_sha"]!=q["ids_sha"] or base["file_sha"]!=q["file_sha"] or a["tokens"]!=b["tokens"]:
            raise ValueError("Mismatched cold reference or scored continuation")
        yield dict(arm=arm.name,source=source,targets=a["tokens"],cold_cut=base["cached"],arm_cut=q["cached"],
                   cold_mean_nll=a["nll_mean"],arm_mean_nll=b["nll_mean"],
                   delta_nll=b["nll_mean"]-a["nll_mean"],
                   cold_argmax=a["argmax_match"],arm_argmax=b["argmax_match"],
                   identical_scores=a==b)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cold",type=Path);p.add_argument("arms",nargs="+",type=Path)
    args=p.parse_args()
    rows=[r for arm in args.arms for r in compare(args.cold,arm)]
    writer=csv.DictWriter(sys.stdout,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
