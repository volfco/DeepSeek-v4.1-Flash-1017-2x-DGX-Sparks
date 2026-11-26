import json,statistics as st,bisect,datetime as dt
exec(open('overlap.py').read().split("X=[r for r")[0])  # reuse R, ms, U, ov
def lin(xs,ys):
    mx,my=st.mean(xs),st.mean(ys); b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); return my-b*mx,b
X=[r for r in R if r['run_max']==8 and r['rounds']>=20 and r['decode_s']>=3]
xs=[ov(r['t']-r['decode_s'],r['t']) for r in X]; ys=[ms(r) for r in X]
a,b=lin(xs,[1/y for y in ys]); rdec=1/a; f=1+b*rdec
print(f'two-rate fit: 1/ms = {a:.5f} {b:+.5f} x overlap  -> decode-only round {rdec:.1f} ms, decoder share during prefill f={f:.2f} (mixed bench 128K cold: 95.5/251 = 0.38)')
print(f'  mixed-bench f=0.38 at overlap 0.60 would give {86.2/(1-0.6*(1-0.38)):.0f} ms; observed median {st.median(ys):.0f}')
T3=dt.datetime(2026,11,26,3,0,tzinfo=dt.timezone.utc).timestamp(); T1=dt.datetime(2026,11,26,1,0,tzinfo=dt.timezone.utc).timestamp(); T2=dt.datetime(2026,11,26,2,0,tzinfo=dt.timezone.utc).timestamp()
for lab,lo,hi in [('01-02 long ctx',T1,T2),('02-03',T2,T3),('03+ saturated',T3,9e9)]:
    x=[r for r in R if lo<=r['t']<hi]
    unc=[r['prompt']-r['cached'] for r in x]
    epk=[r['engram_host_ms']/max(1,r['prompt']-r['cached']) for r in x if r['prompt']-r['cached']>=256]
    pf=[(r['prompt']-r['cached'])/r['prefill_s'] for r in x if r['prompt']-r['cached']>=1024 and r['prefill_s']>0]
    print(f'{lab}: n={len(x)} uncached med {st.median(unc):.0f}; engram_host_ms/req med {st.median(r["engram_host_ms"] for r in x):.0f}; engram ms per uncached token med {st.median(epk):.3f}; prefill tok/s (>=1K uncached) med {st.median(pf):.0f} p10 {sorted(pf)[len(pf)//10]:.0f}; prefill_s med {st.median(r["prefill_s"] for r in x):.2f}')
# restores
rs=[r for r in R if r['restore_s']>0]
print('restores n',len(rs),'restore_s med %.2f p90 %.2f'%(st.median(r['restore_s'] for r in rs),sorted(r['restore_s'] for r in rs)[9*len(rs)//10]),'cached med',st.median(r['cached'] for r in rs))
for lab,lo,hi in [('pre-03',0,T3),('03+',T3,9e9)]:
    y=[r for r in rs if lo<=r['t']<hi]
    if y: print(' ',lab,'n',len(y),'restore_s med %.2f'%st.median(r['restore_s'] for r in y),'cached med',st.median(r['cached'] for r in y),'s per 1K cached %.3f'%st.median(1000*r['restore_s']/max(1,r['cached']) for r in y))
# same restore size, restore_s vs prefill overlap at the time of restore
y=[r for r in rs if r['t']>=T3]
o=[ov(r['t']-r['decode_s']-r['ttft_s']+r['queue_s']-0.001, r['t']-r['decode_s']-r['ttft_s']+r['queue_s']+r['restore_s']) if r['restore_s']>0 else 0 for r in y]
print('  saturated restores: overlap of restore window with prefill union (includes own) med %.2f'%st.median(o))
# exact concurrency from done intervals for >=200K 'alone' rows w/ +49%
iv=[(r['arr'],r['t']) for r in R]
for r in R:
    if (r['run_max'] or 0)<=1 and r['prompt']>=200000 and r['decode_s']>=1:
        a0,b0=r['t']-r['decode_s'],r['t']
        others=sum(1 for a,b in iv if a<b0 and b>a0)-1
        print('  >=200K prompt %d ms/round %.1f  other requests in flight during decode: %d'%(r['prompt'],ms(r),others))
