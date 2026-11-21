"""Plot measured beyond-L2 read pressure against each board's independently calibrated streaming rate."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

here=Path(__file__).resolve().parent
ranks=("head","worker")
roof=dict(zip(ranks,(240.4069,250.302)))
colors={"head":"#005e77","worker":"#b45a22"}
names={"head":"Head","worker":"Worker"}
def rates(q):
    return {r:next(iter(q["ranks"][r]["pm"].values()))["GB_s"] for r in ranks}
prefills=[rates(json.loads(p.read_text())) for p in (here/"prefill-300k.json",here/"serving-meter01/prefill-300032.json")]
windows=list(csv.DictReader((here/"serving-meter01/windows.csv").open()))
whole={"300K prefill":{r:[p[r] for p in prefills] for r in ranks}}
for depth in ("3072","300032"):
    for width in ("1","4"):
        whole[("3K" if depth=="3072" else "300K")+f" decode · {width} stream"+("s" if width=="4" else "")]={
            r:[float(c[r+"_read_GB_s"]) for c in windows if c["depth"]==depth and c["width"]==width] for r in ranks}
stages={}
for rank in ranks:
    for r in csv.DictReader((here/f"stage-pilot01/{rank}-5us-chat-prime.csv").open()):
        if not r["kind"].startswith("prefill"):continue
        label={"engram":"Engram","attn":"Attention","moe":"Routed MoE","head":"Output head",
               "e_gu":"Expert gate / up","e_down":"Expert down","d_gemm":"Dense GEMM + epilogue",
               "gather_a":"Attention gather","a_idx":"Indexer","mix_attn":"Attention residual mix"}.get(r["stage"])
        if label:stages.setdefault(label,{})[rank]=[float(r["sample_GB_s"])]
ordered={k:stages[k] for k in ("Engram","Indexer","Attention gather","Attention","Dense GEMM + epilogue",
                               "Routed MoE","Attention residual mix","Expert gate / up","Expert down","Output head")}
fig,axes=plt.subplots(1,2,figsize=(10.4,5.6),gridspec_kw={"wspace":.71})
for ax,items,title in zip(axes,(whole,ordered),("Graph-on serving","Cold 9K prefill stages")):
    for i,(label,rr) in enumerate(items.items()):
        for rank,dy in zip(ranks,(-.12,.12)):
            v=[x*100/roof[rank] for x in rr[rank]]
            mark="o" if rank=="head" else "s"
            ax.plot([min(v),max(v)],[i+dy]*2,color=colors[rank],lw=2)
            ax.scatter(statistics.mean(v),i+dy,s=35,marker=mark,color=colors[rank],zorder=3)
    ax.set_yticks(range(len(items)),items.keys(),fontsize=9)
    ax.set_ylim(len(items)-.4,-.6)
    ax.set_xlim(0,104);ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel("Calibrated read rate (%)",fontsize=9)
    ax.axvline(100,color="#444",lw=.85,ls="--");ax.grid(axis="x",color="#dfe5e8",lw=.6)
    ax.set_title(title,loc="left",fontsize=12,weight="bold",pad=13)
    for side in ("top","right","left"):ax.spines[side].set_visible(False)
    ax.tick_params(axis="both",length=0,pad=6)
fig.subplots_adjust(left=.18,right=.97,top=.78,bottom=.2)
fig.suptitle("GPU read pressure on the two GB10 boards",x=.033,y=.97,ha="left",fontsize=17,weight="bold")
fig.text(.033,.89,"GPU read misses beyond L2, relative to 240.4 GB/s on the head and 250.3 GB/s on the worker.",fontsize=10)
fig.legend(handles=[Line2D([],[],marker=m,ls="",color=colors[r],label=names[r]) for r,m in zip(ranks,("o","s"))],
           loc="lower left",bbox_to_anchor=(.026,.065),ncol=2,frameon=False,fontsize=9)
fig.text(.033,.045,"Lines span measured windows. Stage intervals overlap. Writes and CPU traffic are outside this counter.",fontsize=8.5,color="#404955")
fig.savefig(here/"read-pressure.png",dpi=175,facecolor="white")
