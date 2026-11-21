"""Pair hard-task outcomes, actual concurrent windows and applied route counts across settings."""
from collections import defaultdict,Counter
from pathlib import Path
import json
import math
import statistics
import sys

def paired_nll(nll):
    summary=[]
    for arm,data in nll:
        ref=nll[0][1]['items'];items=data['items'];diffs=defaultdict(list)
        if set(ref)!=set(items) or data['meta']['set_sha']!=nll[0][1]['meta']['set_sha']:raise ValueError('NLL reference changed')
        for k,v in items.items():diffs[k.rsplit('-',1)[0]].append(v['nll_mean']-ref[k]['nll_mean'])
        by_source={k:statistics.mean(ds) for k,ds in sorted(diffs.items())}
        means=list(by_source.values());delta=statistics.mean(means);se=statistics.stdev(means)/math.sqrt(len(means))
        if len(means)!=20 or any(len(x)!=2 for x in diffs.values()):raise ValueError('Expected two excerpts in each of 20 paired sources')
        summary.append(dict(arm=arm,items=len(items),sources=len(means),target_tokens=data['overall']['tokens'],
                            absolute_item_mean=data['overall']['nll_mean'],delta=delta,ci95=[delta-2.093*se,delta+2.093*se],
                            identical_excerpt_means=sum(items[k]['nll_mean']==v['nll_mean'] for k,v in ref.items()),
                            paired_source_deltas=by_source))
    return summary

def analyze(root):
    arms=[a for a in ('current-a','off','cap12','current-b') if (root/(a+'.jsonl')).exists()];answers={};groups=[];routing=[];nll=[];confirmation=[];reference=None
    first_requests={};trajectories=[]
    if not arms:raise ValueError('No completed hard-task result')
    for arm in arms:
        raw=[json.loads(l) for l in (root/(arm+'.jsonl')).read_text().splitlines() if l.startswith('{')]
        if raw[-1]['kind']!='complete':raise ValueError('Incomplete '+arm)
        if reference is None:reference=raw[0]
        if raw[0]!=reference:raise ValueError('Task fixtures or sampling changed')
        scores={};batches=defaultdict(list);counters=defaultdict(list)
        for r in raw:
            if r['kind']=='scores':
                for i,s in zip(r['ids'],r['success']):scores[r['family'],r['width'],r['seed'],i]=bool(s)
            elif r['kind']=='batch':batches[r['family'],r['width']].append(r)
            elif r['kind']=='routing':counters[r['family'],r['width']].append(r['delta'])
        answers[arm]=scores
        for k,bs in batches.items():
            inputs=Counter();outputs=Counter();totals=Counter()
            for b in bs:
                turn=b['turn']
                for item,q in zip(b['ids'],b['results']):
                    key=(*k,b['seed'],item,turn)
                    signature=(q['input'],q['token_hash'],q['tokens'],q['reason'])
                    if key not in first_requests and arm!=arms[0]:raise ValueError('Missing reference trajectory')
                    ref=first_requests.setdefault(key,signature)
                    totals[turn]+=1;inputs[turn]+=ref[0]==q['input'];outputs[turn]+=ref==signature
            if inputs[0]!=totals[0]:raise ValueError('Opening task contents or seeds changed between policies')
            trajectories.append(dict(arm=arm,family=k[0],width=k[1],requests_per_turn=totals,
                                     matched_inputs_per_turn=inputs,matched_input_and_output_per_turn=outputs))
            reqs=[q for b in bs for q in b['results']];subset=[v for key,v in scores.items() if key[:2]==k];full=[b['common_decode_s'] for b in bs if b['full_width_rounds']>0]
            groups.append(dict(arm=arm,family=k[0],width=k[1],completed_tasks=len(subset),passed_tasks=sum(subset),requests=len(reqs),
                stops=dict(Counter(q['reason'] for q in reqs)),n_batches=len(bs),batches_with_full_width_rounds=len(full),
                median_all_decode_s=statistics.median(b['common_decode_s'] for b in bs),min_all_decode_s=min(b['common_decode_s'] for b in bs),
                interior_physical_rounds=sum(b['full_width_rounds'] for b in bs),interior_s=sum(b['interior_s'] for b in bs),
                summed_output=sum(q['tokens'] for q in reqs),summed_batch_wall_s=sum(b['wall_s'] for b in bs),
                tok_s=sum(q['tokens'] for q in reqs)/sum(b['wall_s'] for b in bs),
                action_scores={t:sum(q['behavior'] for b in bs if b['turn']==t for q in b['results']) for t in sorted({b['turn'] for b in bs})}))
            cs=counters[k]
            if all(v is None for v in cs):routing.append(dict(arm=arm,family=k[0],width=k[1],applied=False));continue
            total={f:sum(v[f] for v in cs) for f in cs[0]};routing.append(dict(arm=arm,family=k[0],width=k[1],applied=True,counts=total,
                before_per_window=total['before']/total['windows'],after_per_window=total['after']/total['windows'],drops_per_row=total['dropped']/total['rows'],dropped_weight_share=total['dropped_ppm']/1e6/total['rows']))
        primary=root/(arm+'.score.json')
        if primary.exists():nll.append((arm,json.loads(primary.read_text())))
        path=root/(arm+'.confirmation.json')
        if path.exists():confirmation.append((arm,json.loads(path.read_text())))
    paired=[]
    for ref_arm in [a for a in ('current-a','current-b') if a in answers]:
        base=answers[ref_arm]
        for arm,ans in answers.items():
            if set(base)!=set(ans):raise ValueError('Missing paired tasks')
            for fam,w in sorted({k[:2] for k in base}):
                keys=sorted(k for k in base if k[:2]==(fam,w))
                wins=[k[2:] for k in keys if ans[k] and not base[k]]
                losses=[k[2:] for k in keys if base[k] and not ans[k]]
                paired.append(dict(arm=arm,reference=ref_arm,family=fam,width=w,n=len(keys),wins=len(wins),losses=len(losses),
                                   won_seed_and_item=wins,lost_seed_and_item=losses,both_pass=sum(base[k] and ans[k] for k in keys)))
    if (nll and len(nll)!=len(arms)) or len(confirmation)!=len(nll):raise ValueError('Incomplete paired NLL crossing')
    return dict(task_groups=groups,paired_tasks=paired,matched_trajectories=trajectories,route_groups=routing,paired_nll=paired_nll(nll),disjoint_paired_nll=paired_nll(confirmation))

if __name__=='__main__':print(json.dumps(analyze(Path(sys.argv[1])),indent=2))
