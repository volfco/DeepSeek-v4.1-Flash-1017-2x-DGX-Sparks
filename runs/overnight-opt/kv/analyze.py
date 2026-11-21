"""Extract measured service and revisit costs from the completed private-tier pressure campaign."""
from pathlib import Path
import json
import statistics
import sys

base=Path(__file__).resolve().parent
sys.path.insert(0,str(base.parents[1]/'engine-followup/kv'))
import summarize

def measure(path,budget=1024,arms=None):
    rows=summarize.load(path)
    assert rows[-1]['kind']=='complete' and any(x['kind']=='restored' for x in rows)
    result=[];baseline_inputs=None
    for row in (r for r in rows if r['kind']=='cell' and r['sharing']=='peak' and r['requests'][0]['tokens']==budget
                and (arms is None or r['arm'] in arms)):
        arm=row['arm'];peers=[r for r in rows if r.get('arm')==arm];b=summarize.burst(row)
        assert row['clean'] and all(q['tokens']==budget for q in row['requests'])
        signatures=[(q['input_sha256'],q['seed'],q['cached'],q['tokens'],q['finish'],q['done']['sha']) for q in row['requests']]
        if baseline_inputs is None:baseline_inputs=signatures
        elif baseline_inputs!=signatures:raise ValueError('Inputs, cache cuts, outputs or stops changed across arms')
        pre={r['sharing'].removeprefix('pre-pressure-'):r for r in peers if r.get('kind')=='cell' and r['sharing'].startswith('pre-pressure-')}
        visits=[]
        for after in (r for r in peers if r.get('kind')=='cell' and r['sharing'].startswith('after-pressure-')):
            key=after['sharing'].removeprefix('after-pressure-');first=pre[key]['requests'][0];last=after['requests'][0]
            assert after['clean'] and first['input_sha256']==last['input_sha256']
            cache=after['health_after']['kv_counts']['cache'];initial=after['health_before']['kv_counts']['cache']
            visits.append(dict(history=key,sequence=len(visits)+1,prior_cached=first['cached'],cached=last['cached'],
                disk_delta=cache['disk_hits_committed']-initial['disk_hits_committed'],miss_delta=cache['misses_stale']-initial['misses_stale'],
                ttft_before_s=round(first['ttft'],3),ttft_after_s=round(last['ttft'],3),added_ttft_s=round(last['ttft']-first['ttft'],3),
                engine_prefill_s=float(last['done']['prefill_s']),engine_restore_s=float(last['done']['restore_s']),return_wall_s=round(float(last['total']),3),
                tokens=last['tokens'],output_match=first['done']['sha']==last['done']['sha']))
        events=next(r['events'] for r in peers if r['kind']=='patient')
        inv=next(r for r in peers if r['kind']=='stage' and r['stage']=='peak')
        fills=[]
        for fill in (x for x in peers if x.get('kind')=='cell' and x.get('sharing','').startswith('fill-')):
            q,=fill['requests'];prior=fill['health_before']['kv_counts']['cache'];after=fill['health_after']['kv_counts']['cache']
            fills.append(dict(history=fill['sharing'].removeprefix('fill-'),cached=q['cached'],
                              disk=after['disk_hits_committed']-prior['disk_hits_committed'],ttft_s=round(q['ttft'],3)))
        guard=next((r for r in peers if r['kind']=='guard'),{})
        probes=[]
        for note in guard.get('probes',[]):
            fields=dict(z.split('=',1) for z in note.split())
            probes.append(dict(id=int(fields['id']),at=int(fields['at']),result=fields['result'],us=int(fields['us'])))
        delays=[q['us'] for q in probes]
        result.append(dict(arm=arm,kept=b['kept'],inventory=inv.get('inventory',inv.get('groups')),preparation=fills,queue_s=b['queue_s'],queue_sum_s=round(sum(b['queue_s']),3),
            queue_max_s=max(b['queue_s']),ttft_max_s=max(b['ttft_s']),four_way_s=b['common_s'],wall_s=b['wall_s'],tok_s=b['aggregate_tok_s'],
            no_target=b['no_target'],staled=b['staled'],sha=b['sha'],resumed_after_ms=[e['elapsed_ms'] for e in events if e['outcome']=='resumed'],
            guard=guard.get('decisions'),probes=probes,probe_total_ms=round(sum(delays)/1000,3),
            probe_median_ms=round(statistics.median(delays)/1000,3) if delays else None,probe_max_ms=max(delays)/1000 if delays else None,
            first_victim=visits[0],revisits=visits,burst_plus_revisits_s=round(b['wall_s']+sum(v['return_wall_s'] for v in visits),3)))
    assert len(result)==2 and result[0]['kept']==result[1]['kept'] and result[0]['sha']==result[1]['sha']
    if arms:
        a,b=[next(r for r in result if r['arm']==arm) for arm in arms]
    else:
        a=next(r for r in result if r['arm'].split('-')[0]=='control');b=next(r for r in result if r['arm'].split('-')[0]=='short')
    wall=a['wall_s']-b['wall_s'];queue=a['queue_sum_s']-b['queue_sum_s']
    cold=[x['first_victim']['added_ttft_s'] for x in result if x['first_victim']['cached']==0]
    return dict(budget=budget,baseline=a['arm'],candidate=b['arm'],arms=[a,b],quiet=next((r for r in rows if r['kind']=='quiet'),None),wall_saved_s=round(wall,3),
                client_queue_sum_saved_s=round(queue,3),burst_plus_revisits_saved_s=round(a['burst_plus_revisits_s']-b['burst_plus_revisits_s'],3),
                first_revisit_cost_delta_s=round(b['first_victim']['added_ttft_s']-a['first_victim']['added_ttft_s'],3),
                extra_cold_visits_per_burst_to_offset_wall=round(wall/(sum(cold)/len(cold)),4) if cold else None)

if __name__=='__main__':print(json.dumps(measure(Path(sys.argv[1]),int(sys.argv[2]) if len(sys.argv)>2 else 1024,sys.argv[3:] or None),indent=2))
