import json,statistics as st,bisect,datetime as dt
from collections import defaultdict
E=dt.datetime(2026,11,25,19,35,tzinfo=dt.timezone.utc).timestamp(); T3=dt.datetime(2026,11,26,3,0,tzinfo=dt.timezone.utc).timestamp()
R=[r for r in json.load(open('reqs.json')) if r['t']>=E]
for i,r in enumerate(R): r['i']=i; r['pf0']=r['t']-r['decode_s']-r['ttft_s']+r['queue_s']; r['pf1']=r['t']-r['decode_s']
def ms(r): return 1000*r['decode_s']/(r['rounds']-1)
def line(dpr): return 24.73+3.524*dpr
def union(iv):
    U=[]
    for a,b in sorted(iv):
        if b<=a: continue
        if U and a<=U[-1][1]: U[-1][1]=max(U[-1][1],b)
        else: U.append([a,b])
    return U
def ovl(U,a,b):
    s=0
    for x,y in U:
        if y<=a: continue
        if x>=b: break
        s+=min(b,y)-max(a,x)
    return s/(b-a) if b>a else 0
def q(v,f='{:.1f}'):
    v=sorted(v); n=len(v)
    if not n: return 'n=0'
    if n<=5: return 'n=%d ['%n+', '.join(f.format(x) for x in v)+']'
    return 'n=%d med %s (p10 %s, p90 %s)'%(n,f.format(st.median(v)),f.format(v[n//10]),f.format(v[9*n//10]))
iv=[(r['arr'],r['t']) for r in R]
def inflight(r,a,b): return sum(1 for x,y in iv if x<b and y>a)-1
print('== 1) one stream, EXACT: no other request in flight during decode, decode_s>=3')
for lo,hi,lab in [(0,8192,'<8K'),(8192,65536,'8-64K'),(65536,200000,'64-200K'),(200000,10**9,'>=200K')]:
    x=[r for r in R if r['rounds']>=20 and r['decode_s']>=3 and lo<=r['prompt']<hi and inflight(r,r['t']-r['decode_s'],r['t'])==0]
    print(f'  {lab:8s} dev',q([100*(ms(r)-line(r['drafted']/r['rounds']))/line(r['drafted']/r['rounds']) for r in x],'{:+.1f}'),' ms',q([ms(r) for r in x]))
print('== 2) restores: overlap with OTHER requests\' prompt passes; measured waits from counters in restore_wait.txt')
rs=[r for r in R if r['restore_s']>0 and r['t']>=T3]
o=[]
for r in rs:
    U=union([(x['pf0'],x['pf1']) for x in R if x['i']!=r['i'] and x['pf1']>r['pf0']-30 and x['pf0']<r['pf0']+30])
    o.append(ovl(U,r['pf0'],r['pf0']+r['restore_s']))
print('  saturated restores n=%d restore_s %s; share of restore window covered by another pass %s'%(len(rs),q([r['restore_s'] for r in rs],'{:.2f}'),q(o,'{:.2f}')))
print('== 4) short-tail prefill rate, saturated, by chunk count and Engram ms/token')
def chunks(r): s,e=r['cached'],r['prompt']; return (e-1)//2048-s//2048+1 if e>s else 0
S=[r for r in R if r['t']>=T3 and r['prompt']-r['cached']>=1024 and r['prefill_s']>0]
for c in (1,2):
    x=[r for r in S if chunks(r)==c]
    ept=sorted(r['engram_host_ms']/(r['prompt']-r['cached']) for r in x); cut=[ept[len(ept)//3],ept[2*len(ept)//3]]
    print(f'  {c}-chunk passes: tok/s',q([(r['prompt']-r['cached'])/r['prefill_s'] for r in x],'{:.0f}'),' uncached',q([r['prompt']-r['cached'] for r in x],'{:.0f}'))
    for lab,lo,hi in [('engram low',0,cut[0]),('engram mid',cut[0],cut[1]),('engram high',cut[1],9e9)]:
        y=[r for r in x if lo<=r['engram_host_ms']/(r['prompt']-r['cached'])<hi]
        print(f'     {lab:11s} ms/tok {q([r["engram_host_ms"]/(r["prompt"]-r["cached"]) for r in y],"{:.2f}")}  tok/s {q([(r["prompt"]-r["cached"])/r["prefill_s"] for r in y],"{:.0f}")}')
# unsaturated 1-chunk control: 01-03 window, uncached 1K-2K
x=[r for r in R if r['t']<T3 and 1024<=r['prompt']-r['cached'] and chunks(r)==1 and r['prefill_s']>0]
print('  control pre-03 1-chunk passes: tok/s',q([(r['prompt']-r['cached'])/r['prefill_s'] for r in x],'{:.0f}'),' engram ms/tok',q([r['engram_host_ms']/(r['prompt']-r['cached']) for r in x],'{:.2f}'),' prompt',q([r['prompt'] for r in x],'{:.0f}'))
print('== 5) 2-7 streams: decode round with prompt-pass time removed vs line at total drafted rows')
U=union([(r['pf0'],r['pf1']) for r in R])
g=json.load(open('gauges.json')); run=[(float(t),float(v)) for t,v in g['tensorfold:requests_running']]
def gm(a,b,f): v=[x for t,x in run if a-15<=t<=b+15]; return f(v) if v else None
by=defaultdict(list)
for r in R:
    if r['rounds']<20 or r['decode_s']<3: continue
    a,b=r['t']-r['decode_s'],r['t']; mx,mn=gm(a,b,max),gm(a,b,min)
    if mx is None or mx!=mn or mx<2: continue
    o=ovl(U,a,b); dd=ms(r)*(1-o); n=int(mx)
    by[n].append((dd,line(n*r['drafted']/r['rounds']),o,r['prompt']))
for n in sorted(by):
    x=by[n]; print(f'  {n} streams steady: n={len(x)} prefill overlap med {st.median(v[2] for v in x):.2f}; decode-only ms {st.median(v[0] for v in x):.1f}; line at n x drafted {st.median(v[1] for v in x):.1f}; dev {st.median(100*(v[0]-v[1])/v[1] for v in x):+.1f}%; prompt med {st.median(v[3] for v in x):.0f}')
