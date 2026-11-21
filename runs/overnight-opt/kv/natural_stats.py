"""Summarize the crossed admission policies, keeping natural stops and imposed limits visible."""
from collections import defaultdict,Counter
import argparse
from pathlib import Path
import json
import statistics
import sys

factors=('ten-fixed-a','short-adaptive','short-fixed','ten-adaptive','ten-fixed-b')
fields=('lead_total','lead_ttft','lead_max_gap','follower_max_ttft','wall','throughput','queue_max','no_target','staled')

def analyze(paths,partial=False):
    rows=[]
    for path in paths:
        run=[json.loads(l) for l in path.read_text().splitlines() if l.startswith('{')]
        if run[-1]['kind']!='complete' or not any(r['kind']=='restored' for r in run):raise ValueError('Incomplete campaign '+str(path))
        rows.extend(run)
    groups=defaultdict(list);reference={};fingerprints={};cells=[];visits=[]
    natural={(r['arm'],r['depth'],r['shape'],r['rep']):r for r in rows if r['kind']=='natural-outcome'}
    calibration={}
    for r in (x for x in rows if x['kind']=='calibration'):
        sig=(r['index'],r['tokens'],r['output_sha'])
        if calibration.setdefault(r['depth'],sig)!=sig:raise ValueError('Unequal natural calibration across policies')
    for row in (r for r in rows if r['kind']=='cell'):
        if not row['clean']:raise ValueError('Contaminated workload')
        arm=next(k for k in factors if row['arm']==k or row['arm'].startswith(k+'-'))
        if row['sharing'].startswith('after-pressure-'):
            q,=row['requests'];b=row['health_before']['kv_counts']['cache'];a=row['health_after']['kv_counts']['cache'];visits.append(dict(arm=row['arm'],history=row['sharing'].removeprefix('after-pressure-'),cached=q['cached'],disk=a['disk_hits_committed']-b['disk_hits_committed'],ttft=q['ttft']));continue
        if row['cell']!='admission':continue
        shape='long' if row['sharing']=='peak' else row['sharing'];depth=row['depth'];rep=row['rep'];req=row['requests'];lead=req[0]
        key=depth,shape,rep;inputs=[(q['input_sha256'],q['cached']) for q in req];output=[(q['tokens'],q['finish'],q['done']['sha']) for q in req]
        reference.setdefault(key,inputs);fingerprints.setdefault(key,output)
        if inputs!=reference[key]:raise ValueError(f'Input or cache mismatch {arm} {key}')
        wall=max(q['start']+q['total'] for q in req)-min(q['start'] for q in req)
        sel=dict(arm=arm,depth=depth,shape=shape,rep=rep,lead_tokens=lead['tokens'],lead_finish=lead['finish'],
                 lead_prefill=float(lead['done']['prefill_s']),lead_total=lead['total'],lead_ttft=lead['ttft'],lead_max_gap=lead.get('max_gap_s'),
                 follower_max_ttft=max(q['ttft'] for q in req[1:]),queue_max=max(float(q['done']['queue_s']) for q in req),
                 wall=wall,throughput=sum(q['tokens'] for q in req)/wall,output_matches=sum(a==b for a,b in zip(output,fingerprints[key])),
                 natural_stops=sum(q['finish']=='stop' for q in req),kept=[h['kept_rows'] for h in row['health_before']['per_part']],
                 no_target=row['health_after']['refused_no_target']-row['health_before']['refused_no_target'],
                 staled=row['health_after']['kv_counts']['staled']-row['health_before']['kv_counts']['staled'])
        groups[depth,shape,arm].append(sel);cells.append(sel)
        if shape!='long':
            outcome=natural[row['arm'],depth,shape,rep]
            sel.update(lead_limit=outcome['lead_limit'],unused_budget=outcome['remaining_at_stop'],
                       followers_before_lead_end=sum(v>0 for v in outcome['follower_arrival_to_lead_end_s']),
                       follower_arrival_to_lead_end_s=outcome['follower_arrival_to_lead_end_s'],
                       follower_arrival_after_first_s=outcome['follower_arrival_after_first_s'],
                       request_queue_s=[float(q['done']['queue_s']) for q in req],
                       generated=[q['tokens'] for q in req],stops=[q['finish'] for q in req])
    averages={}
    for k,vals in groups.items():
        averages[k]={f:statistics.mean(q[f] for q in vals) if all(q[f] is not None for q in vals) else None for f in fields}
    comparisons=[]
    for depth,shape in sorted({k[:2] for k in groups}):
        src={k:averages[depth,shape,k] for k in factors if (depth,shape,k) in averages};fh={}
        if len(src)!=len(factors):
            if not partial:raise ValueError('Incomplete policy crossing')
            comparisons.append(dict(depth=depth,shape=shape,means=src,contrasts={},missing=[k for k in factors if k not in src]))
            continue
        for field in fields:
            if src[factors[0]][field] is None:continue
            incumbent=statistics.mean(src[k][field] for k in ('ten-fixed-a','ten-fixed-b'))
            hf=src['short-fixed'][field]-incumbent;ha=src['short-adaptive'][field]-src['ten-adaptive'][field]
            fh[field]={'held_short_minus_long_fixed':round(hf,4),'held_short_minus_long_adaptive':round(ha,4),
                       'adaptive_minus_fixed_at_ten':round(src['ten-adaptive'][field]-incumbent,4),
                       'adaptive_minus_fixed_at_short':round(src['short-adaptive'][field]-src['short-fixed'][field],4),
                       'difference_of_differences':round(ha-hf,4),'return_minus_opening_control':round(src['ten-fixed-b'][field]-src['ten-fixed-a'][field],4)}
        comparisons.append(dict(depth=depth,shape=shape,means=src,contrasts=fh))
    return dict(cells=cells,calibration=calibration,contrasts=comparisons,revisits=visits,extensions=[r for r in rows if r['kind']=='extensions'],
                held=[r for r in rows if r['kind']=='patient'],qualified=[r for r in rows if r['kind']=='qualified'],
                exact=[r for r in rows if r['kind']=='exact'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('logs',type=Path,nargs='+');p.add_argument('--partial',action='store_true')
    a=p.parse_args();print(json.dumps(analyze(a.logs,a.partial),indent=2))
