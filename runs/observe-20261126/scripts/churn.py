import json,statistics as st,datetime as dt,collections
R=json.load(open('reqs.json'))
def ms(r): return 1000*r['decode_s']/(r['rounds']-1)
# per 10-min bucket: arrivals served, uncached tokens prefilled, prefill seconds, restore seconds, gen tokens by wall, ms/round of 8-stream reqs
B=collections.defaultdict(lambda: dict(n=0,unc=0,pf=0.0,rs=0.0,nres=0,gen=0,ms=[],dpr=[],tpr=[],eng=0.0,shas=collections.Counter()))
for r in R:
    k=int(r['t']//600)*600; b=B[k]
    b['n']+=1; b['unc']+=r['prompt']-r['cached']; b['pf']+=r['prefill_s']; b['rs']+=r['restore_s']; b['nres']+=r['restore_s']>0; b['gen']+=r['tokens']; b['eng']+=r['engram_host_ms']/1000
    b['shas'][r['prompt']]+=1
    if r['run_max']==8 and r['rounds']>=20: b['ms'].append(ms(r)); b['dpr'].append(r['drafted']/r['rounds']); b['tpr'].append(r['tokens']/r['rounds'])
print('bucket  reqs  uncached_tok/s  prefill_duty  restore_s  #rest  gen_tok/s(wall)  8str_ms/round(med,n)  dpr  tpr  engram_s')
for k in sorted(B):
    b=B[k]
    if b['n']<20: continue
    print('%s %5d %8.0f %8.0f%% %8.1f %5d %8.1f   %6s n=%-4d %.2f %.2f %6.1f'%(dt.datetime.fromtimestamp(k,dt.timezone.utc).strftime('%H:%M'),b['n'],b['unc']/600,100*b['pf']/600,b['rs'],b['nres'],b['gen']/600,('%.1f'%st.median(b['ms'])) if b['ms'] else '-',len(b['ms']),st.median(b['dpr']) if b['dpr'] else 0,st.median(b['tpr']) if b['tpr'] else 0,b['eng']))
# saturated window request characteristics
S=[r for r in R if r['t']>=dt.datetime(2026,11,26,3,0,tzinfo=dt.timezone.utc).timestamp()]
print('sat window reqs',len(S),'distinct prompt lengths',len(set(r['prompt'] for r in S)),'top', collections.Counter(r['prompt'] for r in S).most_common(3))
print('uncached per req med',st.median(r['prompt']-r['cached'] for r in S),'reply med',st.median(r['tokens'] for r in S),'ttft med',st.median(r['ttft_s'] for r in S),'queue med',st.median(r['queue_s'] for r in S), 'restore frac', sum(r['restore_s']>0 for r in S)/len(S))
print('effort',collections.Counter(r['effort'] for r in S),'thinking',collections.Counter(r['thinking'] for r in S))
