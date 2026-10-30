"""Isolated five-minute account; completed signals, delayed fills, CN T+1.

Adjusted inventory matches the sealed research convention. It is not the raw
share/dividend ledger used in production. Future bar volume caps executions,
never features or ranks. No broker import or shared module mutation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from paper_futu.research_rank_buffer_futu_fees import fee, buy_quantity


@dataclass
class Holding:
    quantity: float
    cost: float
    entry_day: str
    entry_time: str
    unit_cost: float
    sold_net: float=0.
    pending: str=''
    trigger: str=''


class Account:
    def __init__(self, friction=0., initial_cash=1e6):
        self.cash=self.initial_cash=initial_cash
        self.friction=friction
        self.positions={};self.marks={};self.cooldown={};self.previous_ranks={}
        self.orders=[];self.closed=[];self.daily=[];self.holdings=[]
        self.blocked_t1=0;self.unfilled=0
        self.previous_nav=initial_cash;self.day_fees=0.;self.day_notional=0.

    def nav(self):
        return self.cash+sum(h.quantity*self.marks[c] for c,h in self.positions.items())

    @staticmethod
    def valid(quote,previous,buy):
        if quote is None or not all(math.isfinite(float(quote.get(k,float('nan')))) and quote[k]>0 for k in ('open','factor','volume')):
            return False
        if not math.isfinite(float(previous)) or previous<=0:return False
        change=quote['open']/previous-1
        return not ((buy and change>=.095) or (not buy and change<=-.095))

    def fill(self,code,qty,quote,buy,when,reason,signal):
        price=quote['open'];value=qty*price;cost=fee(value,buy,self.friction)
        before=self.cash;self.cash+=(-value-cost if buy else value-cost)
        if self.cash < -1e-6:raise ValueError('Negative cash')
        self.day_fees+=cost;self.day_notional+=value
        self.orders.append(dict(time=str(when),signal_time=str(signal),code=code,
            side='BUY' if buy else 'SELL',quantity=qty,raw_quantity=qty*quote['factor'],
            price=price,raw_price=price/quote['factor'],value=value,fee=cost,
            cash_before=before,cash_after=self.cash,reason=reason))
        self.marks[code]=price
        return value,cost

    def sell(self,code,quote,previous,when,day):
        h=self.positions[code]
        if h.entry_day>=day:self.blocked_t1+=1;return False
        if not self.valid(quote,previous,False):self.unfilled+=1;return False
        minimum,step=(200,1) if code.startswith('SH688') else (100,100)
        cap=math.floor(quote['volume']*.05/step)*step
        desired=h.quantity*quote['factor']
        raw=min(desired,cap)
        if raw<desired-1e-6 and raw<minimum:self.unfilled+=1;return False
        if raw<=0:self.unfilled+=1;return False
        qty=raw/quote['factor'];value,cost=self.fill(code,qty,quote,False,when,h.pending,h.trigger)
        h.quantity-=qty;h.sold_net+=value-cost
        if h.quantity>1e-7:return False
        self.closed.append(dict(code=code,entry_time=h.entry_time,exit_time=str(when),
            entry_day=h.entry_day,exit_day=day,cost=h.cost,pnl=h.sold_net-h.cost,
            return_on_cost=h.sold_net/h.cost-1,reason=h.pending))
        if h.pending=='stop':self.cooldown[code]=day
        del self.positions[code]
        return True

    def step(self,signal,execution,scores,signal_prices,quotes,previous_day_closes):
        """Execution is a later bar's opening, never the signal bar's close.

        The caller leaves a complete five-minute bar between signal and fill.
        No quote is used to decide which stock is strong or which price trigger
        fired. Availability/capacity only decide whether an intent can fill.
        """
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
                if not h.pending and ranks.get(c,math.inf)>80:h.pending='rank'
                if h.pending:h.trigger=str(signal)
        # A strong arrival can replace only an actually held, T+1-eligible lot.
        incoming=ordered[0]
        eligible=self.cooldown.get(incoming,'')<day
        strong=(incoming not in self.positions and self.previous_ranks and
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
        # Cash and remaining slots come from confirmed fills only. Retained
        # holdings drift; no topping-up, including partial pending exits.
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

    def close_day(self,day,official_closes):
        for c,h in self.positions.items():
            p=official_closes.get(c,float('nan'))
            if math.isfinite(p) and p>0:self.marks[c]=p
        nav=self.nav();values={c:h.quantity*self.marks[c] for c,h in self.positions.items()}
        self.daily.append(dict(date=day,nav=nav,cash=self.cash,net_return=nav/self.previous_nav-1,
            fees=self.day_fees,turnover=self.day_notional/self.previous_nav,holdings=len(values),
            cash_ratio=self.cash/nav,max_weight=max(values.values(),default=0)/nav))
        for c,h in self.positions.items():self.holdings.append(dict(date=day,code=c,quantity=h.quantity,
            price=self.marks[c],value=values[c],pending=h.pending))
        self.previous_nav=nav;self.day_fees=self.day_notional=0.

    def reconcile(self):
        cash=self.initial_cash;qty={}
        for o in self.orders:
            if abs(cash-o['cash_before'])>1e-5:raise ValueError('Cash before mismatch')
            sign=1 if o['side']=='BUY' else -1
            cash-=sign*o['value']+o['fee'];qty[o['code']]=qty.get(o['code'],0)+sign*o['quantity']
            if abs(cash-o['cash_after'])>1e-5 or qty[o['code']]<-1e-5:raise ValueError('Ledger mismatch')
        if abs(cash-self.cash)>1e-5:raise ValueError('Ending cash mismatch')
        for c,q in qty.items():
            if abs(q-(self.positions[c].quantity if c in self.positions else 0))>1e-5:raise ValueError('Inventory mismatch')
        open_pnl=sum(h.sold_net+h.quantity*self.marks[c]-h.cost for c,h in self.positions.items())
        closed_pnl=sum(x['pnl'] for x in self.closed)
        if abs(self.nav()-self.initial_cash-open_pnl-closed_pnl)>1e-4:raise ValueError('Profit mismatch')
        return dict(status='passed',orders=len(self.orders),closed=len(self.closed),open=len(self.positions),
            cash_error=abs(cash-self.cash),profit_error=abs(self.nav()-self.initial_cash-open_pnl-closed_pnl),
            t1_blocked_attempts=self.blocked_t1,unfilled_attempts=self.unfilled)
