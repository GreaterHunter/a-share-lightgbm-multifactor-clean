"""Isolated ranking-policy variants of the sealed minute account.

Only rank exit and strong-arrival replacement are configurable. The original
account, production strategy, models, and archived results remain unchanged.
"""
from __future__ import annotations

import math

from paper_futu.cn_intraday_account import Account, Holding, fee, buy_quantity


class RankPolicyAccount(Account):
    def __init__(self, friction=0., initial_cash=1e6, *, rank_exit=80, strong_entry=True):
        super().__init__(friction, initial_cash)
        if rank_exit is not None and (not isinstance(rank_exit, int) or rank_exit < 10):
            raise ValueError('Rank exit must be None or an integer >= 10')
        self.rank_exit = rank_exit
        self.strong_entry = bool(strong_entry)

    def step(self,signal,execution,scores,signal_prices,quotes,previous_day_closes):
        if execution<=signal:raise ValueError('Signal must precede execution')
        day=str(execution.date())
        ordered=sorted(scores,key=lambda c:(-scores[c],c))
        ranks={c:i+1 for i,c in enumerate(ordered)}
        if not ordered:return
        for c,h in self.positions.items():
            p=signal_prices.get(c,float('nan'))
            if math.isfinite(p) and p>0:self.marks[c]=p
            if not h.pending:
                if math.isfinite(p) and p>0:
                    liquidation=h.sold_net+h.quantity*p-fee(h.quantity*p,False,self.friction)
                    if liquidation-h.cost<=-h.cost*.05+1e-7:h.pending='stop'
                    elif p>=h.unit_cost*1.10-1e-9:h.pending='profit'
                if (not h.pending and self.rank_exit is not None
                        and ranks.get(c,math.inf)>self.rank_exit):h.pending='rank'
                if h.pending:h.trigger=str(signal)
        incoming=ordered[0]
        eligible=self.cooldown.get(incoming,'')<day
        strong=(self.strong_entry and incoming not in self.positions and self.previous_ranks and
                self.previous_ranks.get(incoming,math.inf)>1 and eligible)
        if strong and len(self.positions)>=10 and not any(h.pending for h in self.positions.values()):
            possible=[c for c,h in self.positions.items() if h.entry_day<day]
            if possible:
                worst=max(possible,key=lambda c:(ranks.get(c,math.inf),c))
                if scores[incoming]>scores.get(worst,-math.inf):
                    self.positions[worst].pending='strong';self.positions[worst].trigger=str(signal)
        sold=set()
        for c in list(self.positions):
            if self.positions[c].pending:
                if self.sell(c,quotes.get(c),previous_day_closes.get(c,float('nan')),execution,day):sold.add(c)
        for c in ordered:
            if len(self.positions)>=10:break
            if c in self.positions or c in sold or self.cooldown.get(c,'')>=day:continue
            q=quotes.get(c)
            if not self.valid(q,previous_day_closes.get(c,float('nan')),True):continue
            nav=self.nav();money=min(nav*.095,max(0,self.cash-nav*.05))
            qty=buy_quantity(c,money,q['open'],q['factor'],self.friction)
            minimum,unit=(200,1) if c.startswith('SH688') else (100,100)
            cap=math.floor(q['volume']*.05/unit)*unit
            raw=min(qty*q['factor'],cap)
            if raw<minimum:continue
            qty=raw/q['factor'];value,cost=self.fill(c,qty,q,True,execution,'entry',signal)
            self.positions[c]=Holding(qty,value+cost,day,str(execution),(value+cost)/qty)
        self.previous_ranks=ranks
