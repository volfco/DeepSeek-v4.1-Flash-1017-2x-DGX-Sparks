import json,re,bisect,statistics as st
h=json.load(open('engine_logs.json')); g=json.load(open('gauges.json'))
T0=1764098940e6  # placeholder overwritten below
import datetime as dt
T0=dt.datetime(2026,11,25,19,29,tzinfo=dt.timezone.utc).timestamp()
run=[(float(t),float(v)) for t,v in g['tensorfold:requests_running']]
wai=[(float(t),float(v)) for t,v in g['tensorfold:requests_waiting']]
rt=[t for t,_ in run]
def gmax(ser,a,b):
    ts=[t for t,_ in ser]; i=bisect.bisect_left(ts,a-15); j=bisect.bisect_right(ts,b+15)
    v=[x for _,x in ser[i:j]]; return max(v) if v else None
R=[]
for x in h:
    t=x['_timestamp']/1e6
    if t<T0: continue
    b=x['body'].replace('[tensorfold] ','').strip()
    if not b.startswith('done '): continue
    d=dict(re.findall(r'(\w+)=(\S+)',b)); d['t']=t
    for k in ['prompt','cached','tokens','rounds']: d[k]=int(d[k])
    for k in ['prefill_s','ttft_s','decode_s','tok_per_s','queue_s','restore_s','engram_host_ms','encode_s','render_s']: d[k]=float(d[k])
    a,dr=d['accepted'].split('/'); d['acc']=int(a); d['drafted']=int(dr)
    d['dec_start']=t-d['decode_s']; d['arr']=t-d['decode_s']-d['ttft_s']
    d['run_max']=gmax(run,d['arr'],t); d['wait_max']=gmax(wai,d['arr'],t)
    R.append(d)
json.dump(R,open('reqs.json','w'))
print(len(R))
