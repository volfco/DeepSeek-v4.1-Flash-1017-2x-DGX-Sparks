"""A deterministic four-turn mock procurement workflow with a cost-minimizing order oracle."""
import hashlib
import itertools
import json
import random


def spec(name,desc,fields):
    return {'type':'function','function':{'name':name,'description':desc,'parameters':{'type':'object','properties':fields,'required':list(fields),'additionalProperties':False}}}
string={'type':'string'};integer={'type':'integer'}
tools=[spec('lookup_order','Fetch the order, destination, deadline and quantities.',{'order_id':string}),
       spec('get_offers','Read stock, prices and transit days for this destination and order ticket.',{'ticket':string,'destination':string}),
       spec('hold_plan','Place a hold for the chosen allocation and complete charge in cents.',{'ticket':string,'allocations':{'type':'array','items':{'type':'object','properties':{'shop':string,'A':integer,'B':integer},'required':['shop','A','B'],'additionalProperties':False}},'price_cents':integer}),
       spec('hold_unavailable','Record that no feasible allocation exists.',{'ticket':string}),
       spec('confirm','Confirm the opaque receipt returned by the preceding hold.',{'receipt':string})]

def data(item,seed):
    rng=random.Random(87231+item*1013+seed);oid=f'R{item+23}';ticket=hashlib.sha256(f'case:{item}:{seed}'.encode()).hexdigest()[:9]
    need=(rng.randrange(2,5),rng.randrange(2,5));due=rng.randrange(2,5)
    shops=[dict(shop=s,stock_A=rng.randrange(1,6),stock_B=rng.randrange(1,6),price_A=rng.randrange(2,10),price_B=rng.randrange(2,10),fee=rng.randrange(1,10),days=rng.randrange(1,5)) for s in ('Oak','Ash','Elm','Fir')]
    if item in (3,7):
        # A plausible low-cost late offer, with the on-time shops deliberately one unit short overall.
        for i,s in enumerate(shops):s['stock_A']=need[0]+1 if i==0 else 0;s['days']=due+1 if i==0 else due
    else:
        shops[0]['days']=due;shops[0]['stock_A']=need[0];shops[0]['stock_B']=need[1]
    return dict(order_id=oid,ticket=ticket,destination=['Lima','Oslo','Cork','Kyoto'][item%4],A=need[0],B=need[1],max_days=due,shops=shops,receipt='H'+hashlib.sha256(('hold'+ticket).encode()).hexdigest()[:10])

def plans(d):
    valid=[s for s in d['shops'] if s['days']<=d['max_days']]
    for k in (1,2):
        for sellers in itertools.combinations(valid,k):
            for a in range(d['A']+1):
                for b in range(d['B']+1):
                    amounts=((a,b),(d['A']-a,d['B']-b)) if k==2 else ((a,b),)
                    if sum(q[0] for q in amounts)!=d['A'] or sum(q[1] for q in amounts)!=d['B']:continue
                    if any(not (a+b>0 and a<=s['stock_A'] and b<=s['stock_B']) for s,(a,b) in zip(sellers,amounts)):continue
                    price=sum(s['fee']+a*s['price_A']+b*s['price_B'] for s,(a,b) in zip(sellers,amounts))
                    yield price,sorted((s['shop'],a,b) for s,(a,b) in zip(sellers,amounts))

def prompt(d):
    return (f"Complete order {d['order_id']} through the tools. First obtain the ticket and demand, then obtain offers. "
        'Pick at most two distinct shops. Each must meet the delivery deadline. Buy exactly the requested quantities of A and B, '
        'using integer units within stock. Each used shop charges its fixed fee once plus per-unit prices. '
        'Minimize the total charge; any tied cheapest feasible plan is acceptable. If none is feasible, hold an unavailable notice. '
        'Use one tool per turn. Confirm the receipt returned by the hold. Do not invent a ticket or receipt.')

def respond(d,stage,name,args):
    if not isinstance(args,dict):return False,{'error':'arguments must be an object'}
    if stage==0 and name=='lookup_order' and args=={'order_id':d['order_id']}:
        return True,{k:d[k] for k in ('ticket','destination','A','B','max_days')}
    if stage==1 and name=='get_offers' and args=={k:d[k] for k in ('ticket','destination')}:return True,{'offers':d['shops']}
    if stage==2:
        possible=list(plans(d));best=min((x[0] for x in possible),default=None)
        if best is None and name=='hold_unavailable' and args=={'ticket':d['ticket']}:return True,{'receipt':d['receipt']}
        if name=='hold_plan' and set(args)=={'ticket','allocations','price_cents'} and args['ticket']==d['ticket'] and type(args['price_cents']) is int and isinstance(args['allocations'],list):
            arr=args['allocations']
            if all(isinstance(s,dict) and set(s)=={'shop','A','B'} and type(s['shop']) is str and type(s['A']) is int and type(s['B']) is int for s in arr):
                got=sorted((s['shop'],s['A'],s['B']) for s in arr)
                if args['price_cents']==best and (best,got) in possible:return True,{'receipt':d['receipt']}
    if stage==3 and name=='confirm' and args=={'receipt':d['receipt']}:return True,{'status':'confirmed'}
    return False,{'error':'This call or its arguments did not satisfy the present stage.'}

if __name__=='__main__':
    for seed in (6100,6200):
        for i in range(8):
            d=data(i,seed);arr=list(plans(d));assert bool(arr)==(i not in (3,7))
            if arr:
                price,rows=min(arr);args={'ticket':d['ticket'],'price_cents':price,'allocations':[dict(shop=s,A=a,B=b) for s,a,b in rows]}
                assert respond(d,2,'hold_plan',args)[0] and not respond(d,2,'hold_plan',dict(args,price_cents=price+1))[0]
            else:assert respond(d,2,'hold_unavailable',{'ticket':d['ticket']})[0]
            assert respond(d,3,'confirm',{'receipt':d['receipt']})[0] and not respond(d,3,'confirm',{'receipt':d['ticket']})[0]
    print('All 16 allocation-oracle and staged-tool checks passed')
