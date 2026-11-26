import json,statistics as st,bisect
R=[r for r in json.load(open("reqs.json")) if r["t"]>=1795635300]
def ms(r): return 1000*r['decode_s']/(r['rounds']-1)
# prefill interval of each request: [t-decode-ttft+queue, t-decode]  (restore counted inside)
P=sorted((r['t']-r['decode_s']-r['ttft_s']+r['queue_s'], r['t']-r['decode_s'], r['restore_s']>0, r['chatcmpl'] if 'chatcmpl' in r else id(r)) for r in R)
# merge to union
U=[]
for a,b,_,_ in P:
    if b<=a: continue
    if U and a<=U[-1][1]: U[-1][1]=max(U[-1][1],b)
    else: U.append([a,b])
us=[u[0] for u in U]
def ov(a,b):
    i=max(0,bisect.bisect_right(us,a)-1); s=0
    while i<len(U) and U[i][0]<b:
        s+=max(0,min(b,U[i][1])-max(a,U[i][0])); i+=1
    return s/(b-a)
X=[r for r in R if r['run_max']==8 and r['rounds']>=20 and r['decode_s']>=3]
xs=[ov(r['t']-r['decode_s'],r['t']) for r in X]; ys=[ms(r) for r in X]
n=len(xs); mx=st.mean(xs); my=st.mean(ys)
b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); a=my-b*mx
res=[y-(a+b*x) for x,y in zip(xs,ys)]
r2=1-sum(e*e for e in res)/sum((y-my)**2 for y in ys)
print(f'8-stream reqs n={n}; overlap with any prefill: med {st.median(xs):.2f} (p10 {sorted(xs)[n//10]:.2f} p90 {sorted(xs)[9*n//10]:.2f})')
print(f'ms/round = {a:.1f} + {b:.1f} x overlap ; R2 {r2:.2f}; at overlap=0 -> {a:.1f} ms; at 1 -> {a+b:.1f}')
# implied decode-only round: ms*(1-overlap)
dd=[y*(1-x) for x,y in zip(xs,ys)]
print('implied decode-only ms/round (ms x (1-overlap)): med %.1f p10 %.1f p90 %.1f'%(st.median(dd),sorted(dd)[n//10],sorted(dd)[9*n//10]))
for lo,hi in [(0,.4),(.4,.6),(.6,.8),(.8,1.01)]:
    y=[yy for x,yy in zip(xs,ys) if lo<=x<hi]
    if y: print(f'  overlap {lo:.1f}-{hi:.1f}: n={len(y)} med ms/round {st.median(y):.1f}')
# one-stream >=200K outliers
print('--- one-stream alone >=200K')
for r in R:
    if (r['run_max'] or 0)<=1 and r['rounds']>=20 and r['decode_s']>=1 and r['prompt']>=200000:
        line=24.73+3.524*r['drafted']/r['rounds']
        print('  prompt %d cached %d ms/round %.1f dev %+.0f%% decode_s %.1f rounds %d tok/s %.1f overlap %.2f restore %.2f'%(r['prompt'],r['cached'],ms(r),100*(ms(r)-line)/line,r['decode_s'],r['rounds'],r['tok_per_s'],ov(r['t']-r['decode_s'],r['t']),r['restore_s']))
print('--- one-stream alone AND zero prefill overlap, decode_s>=3 (0.1s quantization <=3.3%)')
for lo,hi,lab in [(0,8192,'<8K'),(8192,65536,'8-64K'),(65536,200000,'64-200K'),(200000,10**9,'>=200K')]:
    x=[r for r in R if (r['run_max'] or 0)<=1 and r['rounds']>=20 and r['decode_s']>=3 and lo<=r['prompt']<hi and ov(r['t']-r['decode_s'],r['t'])==0]
    d=sorted(100*(ms(r)-(24.73+3.524*r['drafted']/r['rounds']))/(24.73+3.524*r['drafted']/r['rounds']) for r in x)
    if d: print(f'  {lab:8s} n={len(d):3d} dev med {st.median(d):+.1f}% (p10 {d[len(d)//10]:+.1f}, p90 {d[9*len(d)//10]:+.1f}); ms/round med {st.median(ms(r) for r in x):.1f}; dpr {st.median(r["drafted"]/r["rounds"] for r in x):.2f}')
