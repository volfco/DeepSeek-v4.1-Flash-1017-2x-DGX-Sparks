"""Real 1/4/8-stream sampled SQL and four-turn tool evaluation with idle pruning-counter sentinels."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

here=Path(__file__).resolve().parent
sys.path.insert(0,str(here));sys.path.insert(0,str(Path.cwd()/'tools'))
import bench_div,score,plans,sql
server=bench_div.Server('http://127.0.0.1:8000',score.DEFAULT_MODEL)
container='dsv41-1017-exl3-tf'

def emit(kind,**r):print(json.dumps(dict(kind=kind,**r)),flush=True)
def digest(o):return hashlib.sha256(json.dumps(o,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def dockerlog(since):return subprocess.check_output(['docker','logs','--timestamps','--since',str(since),container],text=True,stderr=subprocess.STDOUT,timeout=45)

def sentinel():
    if not bench_div.idle_wait(server,90)[0]:raise RuntimeError('Unexpected pending requests')
    before=server.metrics()
    since=time.time();body={'model':server.model,'messages':[{'role':'user','content':'Reply with OK.'}],'temperature':0,'max_tokens':1,'chat_template_kwargs':{'enable_thinking':False}}
    r=score.post(server.url,'/v1/chat/completions',body,60)
    if 'error' in r:raise RuntimeError(r['error'])
    deadline=time.monotonic()+8
    while time.monotonic()<deadline:
        log=dockerlog(since)
        # LaneHost names every native Stream "request". The HTTP done record carries the unique ID.
        marks=re.findall(r'^([^ ]+) [^\n]*?hard-prune snapshot request (off|windows=\d+ rows=\d+ before=\d+ after=\d+ dropped=\d+ dropped_ppm=\d+)',log,re.M)
        done=re.findall(r'^([^ ]+) [^\n]*?\bdone (\S+) [^\n]*?\btokens=(\d+)',log,re.M)
        if marks and done:
            after=server.metrics()
            expected=len(done)==len(marks)==1 and done[0][1:]==(r['id'],'1')
            idle=after.get('tensorfold:requests_running')==after.get('tensorfold:requests_waiting')==0
            delta=after[bench_div.DONE_KEY]-before[bench_div.DONE_KEY]
            if not expected or delta not in (0,1):
                raise RuntimeError(f'Ambiguous one-token marker window, releases={len(marks)}, done={len(done)}')
            if idle and delta==1:
                (at,values),=marks
                emit('sentinel',request=r['id'],released=at,done=done[0][0],counter=values)
                if values=='off':return None
                return {k:int(v) for k,v in (s.split('=') for s in values.split())}
        time.sleep(.2)
    raise RuntimeError(f'One-token sentinel lacked a clean paired completion and release, releases={len(marks)}, done={len(done)}')

def invoke(history,seed,budget,is_tool,barrier):
    body={'model':server.model,'messages':history,'seed':seed,'temperature':1.0,'top_p':.95,'top_k':20,'max_tokens':budget,'chat_template_kwargs':{'enable_thinking':True}}
    if is_tool:body['tools']=plans.tools
    fp=digest(body);barrier.wait(timeout=20);start=time.monotonic();ans=score.post(server.url,'/v1/chat/completions',body,240);end=time.monotonic()
    if 'error' in ans:raise RuntimeError(ans['error'])
    ch=ans['choices'][0];m=ch['message'];return {'id':ans['id'],'input':fp,'tokens':ans['usage']['completion_tokens'],'reason':ch['finish_reason'],
        'content':m.get('content') or '', 'calls':m.get('tool_calls') or [],'start':start,'end':end,'budget':budget}

def batch(histories,indices,seed,kind,turn,budget):
    n=len(indices);bar=threading.Barrier(n);m0=server.metrics();since=time.time()
    epoch_offset=time.time()-time.monotonic()
    if m0.get('tensorfold:requests_running')!=0 or m0.get('tensorfold:requests_waiting')!=0:raise RuntimeError('Pair busy before group')
    with concurrent.futures.ThreadPoolExecutor(n) as pool:
        fs=[pool.submit(invoke,histories[i],seed+i*11+turn,budget,kind=='tool',bar) for i in indices];trace=[]
        while not all(f.done() for f in fs):
            now=time.monotonic();m=server.metrics();trace.append([now,m.get('tensorfold:requests_running'),m.get('tensorfold:requests_waiting'),m.get('tensorfold:kv_shard_fetch_rounds'),m.get('tensorfold:generation_tokens_running')]);concurrent.futures.wait(fs,timeout=.12)
        rows=[f.result() for f in fs]
    if not bench_div.idle_wait(server,60)[0]:raise RuntimeError('Pair busy after group')
    m1=server.metrics()
    if m1[bench_div.DONE_KEY]-m0[bench_div.DONE_KEY]!=n:raise RuntimeError('Unexpected concurrent external request')
    log=dockerlog(since);starts=[];ends=[]
    for r in rows:
        m=re.search(r'^([^ ]+) [^\n]*?done '+re.escape(r['id'])+r' (.*)',log,re.M)
        if not m:raise RuntimeError('Missing engine done metadata')
        f=dict(x.split('=',1) for x in m[2].split() if '=' in x);end=datetime.datetime.fromisoformat(m[1].replace('Z','+00:00')).timestamp()
        starts.append(max(end-float(f['decode_s']),r['start']+epoch_offset+float(f['ttft_s'])));ends.append(end)
        r.update(token_hash=f['sha'],ttft=float(f['ttft_s']),queue=float(f['queue_s']),engine_tokens=int(f['tokens']),prompt=int(f['prompt']),cached=int(f['cached']))
    if abs((time.time()-time.monotonic())-epoch_offset)>.05:raise RuntimeError('Wall clock shifted during the group')
    # The engine rounds decode duration to 0.1 s. Exclude 0.12 s on both sides of the common interval.
    lo=max(starts)+.12;hi=min(ends)-.12
    interior=[(a,b) for a,b in zip(trace,trace[1:]) if lo<=a[0]+epoch_offset<=b[0]+epoch_offset<=hi
              and a[1:3]==b[1:3]==[n,0] and a[3] is not None and b[3] is not None]
    full_rounds=sum(b[3]-a[3] for a,b in interior)
    group=dict(family=kind,seed=seed,width=n,turn=turn,ids=indices,common_decode_s=max(0,min(ends)-max(starts)),
               full_width_rounds=full_rounds,interior_s=sum(b[0]-a[0] for a,b in interior),edge_margin_s=.12,
               wall_s=max(r['end'] for r in rows)-min(r['start'] for r in rows),trace=trace)
    return rows,group

def tool_step(d,history,item,r,stage):
    calls=r.pop('calls');args=None;name=None;parsed=False;contained=r.pop('content')
    if len(calls)==1:
        fn=calls[0].get('function') or {};name=fn.get('name');arg=fn.get('arguments')
        try:args=json.loads(arg) if isinstance(arg,str) else arg;parsed=isinstance(args,dict) and isinstance(name,str) and bool(name)
        except (TypeError,ValueError):pass
    valid,res=plans.respond(d,stage,name,args) if parsed else (False,{'error':'Exactly one valid tool call was required.'})
    r.update(behavior=valid,stage=stage,action=name,arguments=args)
    if parsed:
        # Tool call IDs are transport generated. Canonicalize them and JSON key order across arms.
        ident=f'call_{item}_{len(history)}';history.append({'role':'assistant','content':contained,'tool_calls':[{'id':ident,'type':'function','function':{'name':name,'arguments':json.dumps(args,sort_keys=True,separators=(',',':'))}}]})
        history.append({'role':'tool','tool_call_id':ident,'content':json.dumps(res,sort_keys=True,separators=(',',':'))})
    else:
        history.extend([{'role':'assistant','content':contained},{'role':'user','content':'Continue with exactly one of the requested tools.'}])
    return valid

def group(family,items,seed,budget):
    fixtures={i:plans.data(i,seed) for i in items};progress={i:0 for i in items};success={i:True for i in items}
    history={i:[{'role':'user','content':('Use SQLite. '+sql.schema+'. Write a single SELECT or WITH query: '+sql.prompts[i] if family=='sql' else plans.prompt(fixtures[i]))}] for i in items}
    for turn in range(1 if family=='sql' else 4):
        rows,meta=batch(history,items,seed,family,turn,budget)
        for i,r in zip(items,rows):
            if family=='sql':r.pop('calls');r['behavior'],r['grade']=sql.check(i,r['content']);success[i]=r['behavior']
            else:
                valid=tool_step(fixtures[i],history[i],i,r,progress[i]);progress[i]+=int(valid);success[i]&=valid
        emit('batch',**meta,results=rows)
    emit('scores',family=family,width=len(items),seed=seed,ids=items,success=[s and (family=='sql' or progress[i]==4) for i,s in success.items()],final_stage=list(progress.values()))

def main(a):
    if min(a.sql_budget,a.tool_budget)<1:
        raise ValueError('Output limits must be positive')
    if any(a.item_count%w for w in a.widths):
        raise ValueError('Each requested width must divide the number of selected tasks per family')
    emit('dataset',sql_hash=digest(sql.prompts),schema_hash=digest(sql.schema),tools_hash=digest(plans.tools),seeds=a.seeds,widths=a.widths,
         fixture_hash=digest([[plans.data(i,seed) for i in range(a.item_count)] for seed in a.seeds]),thinking=True,
         **({'selected_item_ids':list(range(a.item_count))} if a.item_count!=8 else {}),
         **({'sql_limit':a.sql_budget,'tool_limit':a.tool_budget} if (a.sql_budget,a.tool_budget)!=(1536,900) else {}))
    idle=[sentinel() for _ in range(3)]
    if any((s is None)!=(idle[0] is None) for s in idle):raise RuntimeError('Prune mode changed between idle sentinels')
    overhead=None
    if idle[0] is not None:
        control=[{k:b[k]-v for k,v in first.items()} for first,b in zip(idle,idle[1:])]
        if control[0]!=control[1]:raise RuntimeError(f'Idle sentinel routing was not repeatable: {control}')
        overhead=control[0]
    emit('sentinel-control',delta_per_marker=overhead)
    for rep,seed in enumerate(a.seeds):
        for width in (a.widths if rep%2==0 else list(reversed(a.widths))):
            for family in (('sql','tool') if rep%2==0 else ('tool','sql')):
                start=sentinel()
                for at in range(0,a.item_count,width):group(family,list(range(at,at+width)),seed,a.sql_budget if family=='sql' else a.tool_budget)
                end=sentinel()
                if (end is None)!=(start is None):raise RuntimeError('Routing mode changed during group')
                raw=None if start is None else {k:end[k]-v for k,v in start.items()}
                delta=None if raw is None else {k:v-overhead[k] for k,v in raw.items()}
                if delta is not None and not (delta['rows']>0 and delta['windows']>0 and delta['dropped']>0 and delta['before']>=delta['after']):raise RuntimeError(f'Invalid applied route evidence {delta}')
                emit('routing',family=family,width=width,seed=seed,raw=raw,delta=delta)
    emit('complete')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--widths',nargs='+',type=int,default=[1,4,8],choices=[1,4,8]);ap.add_argument('--seeds',nargs='+',type=int,default=[6100,6200])
    ap.add_argument('--item-count',type=int,choices=range(1,9),default=8,help='Use the first N frozen task identities in each family.')
    ap.add_argument('--sql-budget',type=int,default=1536);ap.add_argument('--tool-budget',type=int,default=900)
    main(ap.parse_args())
