"""Pure forward state machine for the frozen rank80/strong1/full10/5 rule.

No broker imports or submission. Entry fills, never recurring target weights,
define each episode. Closing signals are distinct from execution quotes.
"""
from copy import deepcopy
import math

STRATEGY = 'R80_S1_TOP10_FULL_TP10_SL5'
RULES = dict(strategy_id=STRATEGY, top_k=10, rank_exit=80, strong_rank=1,
             profit=.10, stop=.05, risk_degree=.95, entry_weight=.095,
             profit_cooldown=2, loss_cooldown=3, daily_rebalance=True,
             retained_weight_drift=True, partial_profit_exit=False)


def estimated_fee(value, buying):
    if value <= 0:
        return 0.
    return (max(3.,value*.0003)+15+max(.01,value*.0000487)
            +max(.01,value*.00002)+2*max(.01,value*.00002)
            +(0 if buying else value*.001))


def ranks(scores):
    if not scores or not all(math.isfinite(v) for v in scores.values()):
        raise ValueError('Missing/nonfinite daily scores')
    ordered = sorted(scores,key=lambda c:(-scores[c],c))
    return ordered,{c:i+1 for i,c in enumerate(ordered)}


def bootstrap(positions, nav, signal):
    """Inherited lots use verified broker average cost, marked as migrated.

    Previous fees and holding dates cannot be reconstructed from a position
    snapshot. Do not mislabel this as a new fill or full-history episode.
    """
    episodes = {}
    for code, row in positions.items():
        qty=int(row['qty']); cost=float(row.get('average_cost',0))
        if qty <= 0: continue
        if not math.isfinite(cost) or cost <= 0 or row.get('cost_price_valid') is not True:
            raise ValueError('Verified broker average cost missing: '+code)
        factor=signal['factors'].get(code)
        if factor is None or not math.isfinite(factor) or factor <= 0:
            raise ValueError('Missing inherited price adjustment factor: '+code)
        basis=qty*cost
        episodes[code]=dict(qty=qty,initial_cost=basis,unit_cost=cost,
            budget=min(nav*.02,basis*.05),sold_net=0.,factor=factor,
            pending_reason='',entry_day=None,migrated=True,
            cost_source='broker_average_cost; historical fees/date not reconstructed')
    return dict(version=1,strategy_id=STRATEGY,positions=episodes,cooldown={},
                initialized=False,last_execution_date=None)


def validate_cost_factor(ep, factor):
    """Keep the recorded cost basis until a corporate action is reconciled."""
    recorded=ep.get('factor')
    if any(v is None or not math.isfinite(v) or v<=0 for v in (factor,recorded)):
        raise ValueError('Corporate action factor missing/invalid; needs cost/inventory reconciliation')
    # A change in corporate-action factor requires a reconciled broker inventory
    # and cash event; never reinterpret a split/dividend as a price stop.
    if not math.isclose(factor,recorded,rel_tol=1e-6,abs_tol=1e-12):
        raise ValueError('Corporate action needs cost/inventory reconciliation')


def price_reason(ep, raw_close, factor):
    validate_cost_factor(ep,factor)
    if not math.isfinite(raw_close) or raw_close<=0: return ''
    value=ep['qty']*raw_close
    pnl=ep['sold_net']+value-estimated_fee(value,False)-ep['initial_cost']
    if pnl<=-ep['budget']+1e-7: return 'loss_budget'
    ratio=raw_close/ep['unit_cost']
    if ratio>=1.10-1e-9: return 'profit_full'
    if ratio<=.95+1e-9: return 'hard_stop'
    return ''


def decide(state, signal, previous_scores, positions, eligible, sellable, day):
    state=deepcopy(state)
    actual={c:int(r['qty']) for c,r in positions.items() if r['qty']>0}
    if actual != {c:e['qty'] for c,e in state['positions'].items()}:
        raise ValueError('Inventory differs from durable strategy ledger')
    ordered,rank=ranks(signal['predictions'])
    _,previous=ranks(previous_scores)
    pending={}
    for c,ep in state['positions'].items():
        validate_cost_factor(ep,signal['factors'].get(c))
        reason=ep.get('pending_reason','')
        if not reason:
            close=signal['closes'].get(c)
            if close is not None:
                reason=price_reason(ep,close,signal['factors'][c])
            if not reason and rank.get(c,math.inf)>80:reason='rank_exit'
        if reason:pending[c]=reason
    # One-time transition: shrink inherited Top20 to the best ten surviving
    # holdings. Retained shares are not topped up or reset to synthetic entries.
    if not state['initialized']:
        survivors=sorted(set(actual)-set(pending),key=lambda c:(rank.get(c,math.inf),c))
        for c in survivors[10:]:pending[c]='migration_top10'
    selected=set(actual)-{c for c in pending if c in sellable}
    blocked=set(pending)&selected
    can_buy=lambda c: c in eligible and c not in actual and c not in pending and day>=state['cooldown'].get(c,'0000-00-00')
    bought=[]
    entrants=[c for c in ordered[:1] if previous.get(c,math.inf)>1 and can_buy(c)]
    for c in entrants:
        if len(selected)<10:
            selected.add(c);bought.append(c)
        else:
            candidates=[h for h in selected if h in actual and h in sellable and h not in blocked and h not in pending]
            if candidates:
                worst=max(candidates,key=lambda h:(rank.get(h,math.inf),h))
                if signal['predictions'][c]>signal['predictions'].get(worst,-math.inf):
                    pending[worst]='strong_entry';selected.remove(worst);selected.add(c);bought.append(c)
    for c in ordered:
        if len(selected)>=10:break
        if c not in selected and can_buy(c):selected.add(c);bought.append(c)
    for c,reason in pending.items():state['positions'][c]['pending_reason']=reason
    return dict(state_before=state,reasons=pending,new_codes=bought,
                selected=sorted(selected,key=lambda c:(rank.get(c,math.inf),c)),
                ranking=rank,transition=not state['initialized'])


def apply_fills(decision, orders, actual, day, calendar, factors, nav):
    """Pure/idempotent: rebuild from frozen pre-batch ledger and final receipts."""
    state=deepcopy(decision['state_before']); episodes=state['positions']
    expected={c:e['qty'] for c,e in episodes.items()}
    ids=[str(o['order_id']) for o in orders]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate receipts')
    buys={};transition=decision.get('initial_allocation',False)
    for o in orders:
        qty=int(o['dealt_qty']);price=float(o['dealt_avg_price'])
        if qty==0:continue
        if price<=0 or not math.isfinite(price):raise ValueError('Invalid fill price')
        c=o['code'];buy=o['trd_side']=='BUY'
        expected[c]=expected.get(c,0)+(qty if buy else -qty)
        fee=estimated_fee(qty*price,buy)
        if buy:
            if c in episodes and not transition:raise ValueError('No retained-position top-up')
            if c not in decision['new_codes'] and not (transition and c in decision['selected']):raise ValueError('Unexpected buy')
            b=buys.setdefault(c,dict(qty=0,cost=0.));b['qty']+=qty;b['cost']+=qty*price+fee
        else:
            ep=episodes[c];ep['qty']-=qty;ep['sold_net']+=qty*price-fee
            if ep['qty']<0:raise ValueError('Oversell')
    if {c:q for c,q in expected.items() if q>0} != actual:raise ValueError('Receipt/inventory mismatch')
    for c in list(episodes):
        ep=episodes[c]
        if ep['qty']==0:
            reason=decision['reasons'].get(c,'migration_top10')
            delay=2 if reason=='profit_full' else 3 if reason in ('loss_budget','hard_stop') else 0
            step=calendar.index(day)
            if step+delay>=len(calendar):raise ValueError('Cooldown calendar expired')
            if delay:state['cooldown'][c]=calendar[step+delay]
            del episodes[c]
    if transition:
        for c,ep in episodes.items():
            # Initial resizing preserves the original cost per share for kept
            # units. This starts a documented migrated episode, not a trade PnL.
            ep['initial_cost']=ep['qty']*ep['unit_cost'];ep['sold_net']=0.
            ep['budget']=min(nav*.02,ep['initial_cost']*.05)
    for c,b in buys.items():
        if c in episodes:
            ep=episodes[c];ep['qty']+=b['qty'];ep['initial_cost']+=b['cost']
            ep['unit_cost']=ep['initial_cost']/ep['qty'];ep['budget']=min(nav*.02,ep['initial_cost']*.05)
            ep['cost_source']='migrated_average_cost_plus_confirmed_new_fills_and_estimated_fees'
            continue
        episodes[c]=dict(qty=b['qty'],initial_cost=b['cost'],unit_cost=b['cost']/b['qty'],
            budget=min(nav*.02,b['cost']*.05),sold_net=0.,factor=factors[c],
            pending_reason='',entry_day=day,migrated=False,
            cost_source='confirmed_fill_plus_explicit_estimated_fees')
    state.update(initialized=True,last_execution_date=day)
    return state
