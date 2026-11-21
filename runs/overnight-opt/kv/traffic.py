"""Count terminal reasons, lengths and overlap from the fixed, external November 13 window."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics
import sys


def lengths(rows):
    values=sorted(r['tokens'] for r in rows)
    if not values:return {'n':0}
    return dict(n=len(rows),finish=dict(Counter(r['finish'] for r in rows)),tokens_min=values[0],
                tokens_median=statistics.median(values),tokens_p90=statistics.quantiles(values,n=10,method='inclusive')[8] if len(rows)>1 else values[0],
                tokens_max=values[-1],tokens_le_80=sum(v<=80 for v in values),tokens_le_256=sum(v<=256 for v in values))

def measure(path):
    rows=[]
    for l in path.read_text().splitlines():
        if l[:19]<'2026-11-13T07:20:00': continue
        if '127.0.0.1 "POST /v1/chat/completions' in l:raise ValueError('Local benchmark request inside external window')
        if '[tensorfold] done ' not in l:continue
        f=dict(x.split('=',1) for x in l.split() if '=' in x)
        end=datetime.fromisoformat(l.split()[0].replace('Z','+00:00')).timestamp()
        first=end-float(f['decode_s']);arrival=first-float(f['ttft_s'])
        rows.append(dict(first=first,end=end,arrival=arrival,prefill=float(f['prefill_s']),tokens=int(f['tokens']),finish=f['finish'],
                         prompt=int(f['prompt']),cached=int(f['cached'])))
    out={'all':lengths(rows),'margins':[],'pressure_coverage':[]}
    for threshold in (100000,250000,299008):
        eligible=[r for r in rows if r['prompt']>=threshold]
        prior=[[p for p in rows if p['arrival']+.2 < r['arrival'] < p['end']-.2] for r in eligible]
        out['pressure_coverage'].append(dict(threshold=threshold,completed=len(eligible),
            zero_cached=sum(r['cached']==0 for r in eligible),
            other_requests_active_ge3=sum(len(ps)>=3 for ps in prior),
            same_depth_other_requests_active_ge3=sum(sum(p['prompt']>=threshold for p in ps)>=3 for ps in prior)))
    for margin in (0,.2,.5):
        leads=set();followers=set();pairs=[];min_left={}
        for i,f in enumerate(rows):
            for j,h in enumerate(rows):
                a=f['arrival']
                if i==j or h['prefill']<=1 or not h['first']+margin<a<min(h['first']+5,h['end'])-margin:continue
                leads.add(j);followers.add(i);remaining=h['end']-a;pairs.append(remaining)
                min_left[j]=min(min_left.get(j,remaining),remaining)
        out['margins'].append(dict(margin=margin,leads=lengths([rows[j] for j in leads]),followers=len(followers),pairs=len(pairs),
            pairs_finish_within_one_s=sum(sec<=1 for sec in pairs),pairs_finish_within_half_s=sum(sec<=.5 for sec in pairs),
            leads_with_any_arrival_within_one_s=sum(sec<=1 for sec in min_left.values()),
            seconds_from_follower_arrival_to_lead_end_median=statistics.median(pairs) if pairs else None))
    return out

if __name__=='__main__':
    src=Path(sys.argv[1]) if len(sys.argv)>1 else Path('runs/overnight/break-1600Z.rank0.log')
    print(json.dumps(measure(src),indent=2))
