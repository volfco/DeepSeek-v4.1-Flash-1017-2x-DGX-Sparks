import json,statistics as st,bisect
R=json.load(open('reqs.json')); g=json.load(open('gauges.json'))
run=[(float(t),float(v)) for t,v in g['tensorfold:requests_running']]
def gmin(a,b):
    v=[x for t,x in run if a-15<=t<=b+15]; return min(v) if v else None
def q(v,f='{:.1f}'):
    v=sorted(v); n=len(v)
    if n==0: return 'n=0'
    if n<=5: return 'n=%d ['%n+', '.join(f.format(x) for x in v)+']'
    return 'n=%d med '%n+f.format(st.median(v))+' (p10 '+f.format(v[n//10])+', p90 '+f.format(v[(9*n)//10])+')'
A,B=24.73,3.524
def ms(r): return 1000*r['decode_s']/(r['rounds']-1)
print('== ONE STREAM ALONE (run_max<=1), rounds>=20, decode_s>=1.0')
al=[r for r in R if (r['run_max'] or 0)<=1 and r['rounds']>=20 and r['decode_s']>=1.0]
bins=[(0,8192,'<8K'),(8192,65536,'8-64K'),(65536,200000,'64-200K'),(200000,10**9,'>=200K')]
for lo,hi,lab in bins:
    x=[r for r in al if lo<=r['prompt']<hi]
    dev=[100*(ms(r)-(A+B*r['drafted']/r['rounds']))/(A+B*r['drafted']/r['rounds']) for r in x]
    print(f" {lab:7s} ms/round {q([ms(r) for r in x])}\n          dev vs 2bb052f line {q(dev,'{:+.1f}')}\n          dpr {q([r['drafted']/r['rounds'] for r in x],'{:.2f}')} tpr {q([r['tokens']/r['rounds'] for r in x],'{:.2f}')} tok/s {q([r['tok_per_s'] for r in x])}")
print('== COLD PREFILL ALONE (cached<1024, uncached>=8K)')
for r in sorted([r for r in R if r['cached']<1024 and r['prompt']>=8192],key=lambda r:r['prompt']):
    print('  prompt %7d run_max %s tok/s(prefill) %.0f tok/s(ttft) %.0f queue %.2f engram_ms %.0f'%(r['prompt'],r['run_max'],(r['prompt']-r['cached'])/r['prefill_s'],(r['prompt']-r['cached'])/(r['ttft_s']-r['queue_s']) if r['ttft_s']>r['queue_s'] else 0,r['queue_s'],r['engram_host_ms']))
print('== RESUMED TAILS ALONE (cached>=64K, uncached 4K-8K) prefill tok/s')
x=[r for r in R if (r['run_max'] or 0)<=1 and r['cached']>=65536 and 4096<=r['prompt']-r['cached']<8192]
print(' ',q([(r['prompt']-r['cached'])/r['prefill_s'] for r in x],'{:.0f}'))
x=[r for r in R if (r['run_max'] or 0)<=1 and r['cached']>=65536 and 256<=r['prompt']-r['cached']<2048 and r['restore_s']==0]
print('  TTFT 256-2047 uncached, no restore, alone:',q([r['ttft_s'] for r in x],'{:.2f}'))
print('== 8 STREAMS STEADY (run min=max=8 over request), rounds>=20')
x=[r for r in R if r['run_max']==8 and gmin(r['dec_start'],r['t'])==8 and r['rounds']>=20]
print('  ms/round',q([ms(r) for r in x]),'\n  tok/stream/round',q([r['tokens']/r['rounds'] for r in x],'{:.2f}'),'\n  drafted/stream/round',q([r['drafted']/r['rounds'] for r in x],'{:.2f}'),'\n  accepted',q([100*r['acc']/max(1,r['drafted']) for r in x],'{:.0f}'),'\n  per-stream tok/s',q([r['tok_per_s'] for r in x]),'\n  prompt',q([r['prompt'] for r in x],'{:.0f}'))
for lab,f in [('deep>=16K',lambda r:r['prompt']>=16384),('shallow<16K',lambda r:r['prompt']<16384)]:
    y=[r for r in x if f(r)]; print('  ',lab,'ms/round',q([ms(r) for r in y]))
