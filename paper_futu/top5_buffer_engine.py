"""Pure offline Top5 episode ledger. No broker, model fitting, or source mutation.

Signals observed at yesterday's completed close execute at today's ideal open.
Prices and inventory use the provider's adjusted units; order units use factor.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math


EPS = 1e-9


def fee(value, buying, friction=0.):
    return max(5., value * (.0005 if buying else .0015)) + value * friction if value > 0 else 0.


def valid(value):
    return value is not None and math.isfinite(value) and value > 0


def lot(code):
    return (200, 1) if code.startswith('SH688') else (100, 100)


def buy_quantity(code, money, adjusted_price, factor, friction=0.):
    if not valid(adjusted_price) or not valid(factor) or money <= 5:
        return 0.
    minimum, step = lot(code)
    raw_price = adjusted_price / factor
    # Both minimum commission and proportional commission must be affordable.
    raw = math.floor(min((money-5)/(raw_price*(1+friction)),
                         money/(raw_price*(1+.0005+friction))) / step) * step
    return raw / factor if raw >= minimum else 0.


def half_quantity(code, quantity, factor):
    """Round UP to the nearest legal sell amount >= half. Cannot split => full."""
    minimum, step = lot(code)
    raw = quantity * factor
    amount = max(minimum, math.ceil((raw/2-1e-7)/step)*step)
    return quantity if amount >= raw-1e-7 else amount/factor


def liquidation_breakeven(quantity, basis, friction):
    # q*p - max(5,q*p*.0015) - q*p*friction >= basis.
    return max((basis+5)/(1-friction), basis/(1-.0015-friction))/quantity


@dataclass
class Episode:
    episode_id: int
    code: str
    entry_day: str
    entry_step: int
    entry_nav: float
    initial_qty: float
    quantity: float
    initial_cost: float
    budget: float
    unit_cost: float
    sold_net: float = 0.
    paid_fees: float = 0.
    half_mode: str = ''
    half_notional: float = 0.
    half_quantity_filled: float = 0.
    half_price: float = 0.
    high_close: float = 0.
    trailing: float = 0.
    pending_reason: str = ''
    pending_quantity: float = 0.
    pending_full: bool = False
    trigger_day: str = ''
    blocked_days: int = 0
    budget_triggered: bool = False
    last_factor: float = 1.

    def liquidation_pnl(self, price, friction):
        value = self.quantity*price
        return self.sold_net+value-fee(value, False, friction)-self.initial_cost

    def remaining_budget_price(self, friction):
        return liquidation_breakeven(self.quantity, self.initial_cost-self.budget-self.sold_net, friction)

    def profit_trailing(self, close, friction):
        self.high_close = max(self.high_close, close, self.half_price)
        # Preserve the realised first-half profit: remaining cost basis, including
        # initial buy fee pro rata, must itself break even after its exit fee.
        basis = self.unit_cost*self.quantity
        self.trailing = max(self.trailing, liquidation_breakeven(self.quantity, basis, friction),
                            self.high_close*.9)
        return self.trailing


def price_action(ep, close, rank, case, friction):
    """Budget takes priority. Return reason, full/half; no order mutation here."""
    if case == 'A' or not valid(close):
        return None
    if ep.liquidation_pnl(close, friction) <= -ep.budget + 1e-7:
        return 'loss_budget', True
    ratio = close/ep.unit_cost
    if ep.half_mode == 'profit':
        if close <= ep.profit_trailing(close, friction)+EPS:
            return 'profit_trailing', True
        if case != 'E' and close >= ep.half_price*1.24-EPS:
            return 'second_profit', True
        return None
    if ep.half_mode == 'loss':
        return ('profit_after_loss_half', True) if ratio >= 1.24-EPS else None
    if ratio >= 1.24-EPS:
        return ('profit_half', False) if rank <= 5 else ('profit_full', True)
    if case == 'B' and ratio <= .90+EPS:
        return 'hard_stop', True
    if case in ('C', 'D', 'E') and ratio <= .92+EPS:
        if rank <= 5:
            return 'loss_half', False
        if case == 'C':
            return 'early_stop', True
        if ratio <= .90+EPS:
            return 'hard_stop', True
    return None


def ranking(scores):
    ordered = sorted(((c, float(s)) for c, s in scores.items() if math.isfinite(s)),
                     key=lambda x: (-x[1], x[0]))
    return [c for c, _ in ordered], {c: i+1 for i, (c, _) in enumerate(ordered)}


class Account:
    def __init__(self, case, friction=0., initial_cash=1_000_000.):
        assert case in 'ABCDE'
        self.case, self.friction = case, friction
        self.initial_cash = self.cash = initial_cash
        self.positions, self.cooldown = {}, {}
        self.orders, self.events, self.episodes, self.daily, self.holdings = [], [], [], [], []
        self.previous_ranks = {}
        self.marks = {}
        self.counter = 0

    def tradable(self, code, quote, previous, buying):
        if not quote or not valid(quote.get('open')) or not valid(quote.get('factor')):
            return False, 'missing_open_or_factor'
        if quote.get('suspended', False):
            return False, 'suspended'
        if not valid(previous):
            return False, 'missing_previous_close'
        change = quote['open']/previous-1
        if (buying and change >= .095) or (not buying and change <= -.095):
            return False, 'uniform_limit_filter'
        return True, ''

    def intent(self, ep, reason, full, day, factor):
        if ep.pending_full:
            return
        if ep.pending_reason and not full:
            return
        ep.pending_reason, ep.pending_full, ep.trigger_day = reason, full, day
        ep.pending_quantity = ep.quantity if full else half_quantity(ep.code, ep.quantity, factor)
        if ep.pending_quantity >= ep.quantity-1e-8:
            ep.pending_full = True
        ep.budget_triggered |= reason == 'loss_budget'
        self.events.append(dict(date=day, code=ep.code, episode_id=ep.episode_id,
                                event='intent', reason=reason, full=ep.pending_full,
                                requested_quantity=ep.pending_quantity, budget=ep.budget))

    def sell(self, ep, day, step, quote, previous):
        ok, blocked = self.tradable(ep.code, quote, previous, False)
        if step <= ep.entry_step:
            ok, blocked = False, 'T+1'
        if not ok:
            ep.blocked_days += 1
            self.events.append(dict(date=day, code=ep.code, episode_id=ep.episode_id,
                                    event='blocked_exit', reason=blocked))
            return False
        factor, price = quote['factor'], quote['open']
        desired = min(ep.quantity, ep.pending_quantity)
        # Optional externally supplied capacity is for fault-injection tests only.
        cap = quote.get('sell_capacity_raw', math.inf)/factor
        qty = min(desired, cap)
        minimum, unit = lot(ep.code)
        if qty < ep.quantity-1e-7:
            raw = math.floor((qty*factor+1e-6)/unit)*unit
            qty = raw/factor if raw >= minimum else 0.
        if qty <= EPS:
            ep.blocked_days += 1
            return False
        value, before = qty*price, self.cash
        costs = fee(value, False, self.friction)
        self.cash += value-costs
        ep.sold_net += value-costs
        ep.paid_fees += costs
        ep.quantity = max(0., ep.quantity-qty)
        ep.pending_quantity = max(0., ep.pending_quantity-qty)
        reason = ep.pending_reason
        if not ep.pending_full:
            ep.half_mode = 'profit' if reason == 'profit_half' else 'loss'
            ep.half_notional += value
            ep.half_quantity_filled += qty
            ep.half_price = ep.half_notional/ep.half_quantity_filled
            if ep.half_mode == 'profit' and ep.quantity > EPS:
                ep.profit_trailing(ep.half_price, self.friction)
        self.orders.append(dict(date=day, code=ep.code, episode_id=ep.episode_id,
                                direction='SELL', quantity=qty, raw_quantity=qty*factor,
                                price=price, raw_price=price/factor, notional=value, fee=costs,
                                cash_before=before, cash_after=self.cash, reason=reason,
                                trigger_day=ep.trigger_day))
        if ep.quantity <= 1e-7:
            pnl = ep.sold_net-ep.initial_cost
            record = asdict(ep)
            record.update(exit_day=day, holding_days=step-ep.entry_step,
                          pnl=pnl, return_on_cost=pnl/ep.initial_cost, exit_reason=reason,
                          budget_overshoot=max(0., -pnl-ep.budget),
                          overshoot_reason=('blocked_exit_then_open' if ep.blocked_days else 'next_open_gap_and_fees')
                          if pnl < -ep.budget-1e-7 else '',
                          risk_rule_active=self.case != 'A')
            self.episodes.append(record)
            del self.positions[ep.code]
            if reason not in ('rank_exit', 'strong_replacement'):
                self.cooldown[ep.code] = step+3
            return True
        if ep.pending_quantity <= 1e-7:
            self.events.append(dict(date=day, code=ep.code, episode_id=ep.episode_id,
                                    event='half_complete', reason=reason,
                                    remaining_stop=ep.remaining_budget_price(self.friction), budget=ep.budget))
            ep.pending_reason, ep.pending_full = '', False
        return False

    def step(self, day, step, scores, quotes, previous_closes):
        ordered, ranks = ranking(scores)
        if not ordered:
            raise ValueError('Missing daily prediction universe')
        for code, ep in self.positions.items():
            close = previous_closes.get(code, self.marks.get(code))
            if valid(close):
                self.marks[code] = close
            if code not in self.marks:
                raise ValueError('Held stock has no valuation')
        nav_open = self.cash + sum(ep.quantity*(quotes[c]['open'] if quotes.get(c) and valid(quotes[c].get('open'))
                                                else self.marks[c]) for c, ep in self.positions.items())
        cash_floor = nav_open*.05
        n_orders = len(self.orders)
        for i in (2, 5, 15):
            if len(ordered) > i and scores[ordered[i-1]] == scores[ordered[i]]:
                self.events.append(dict(date=day, event='boundary_tie', rank=i))
        # Full exits already pending cannot be reversed by improving scores/prices.
        for code, ep in list(self.positions.items()):
            action = price_action(ep, previous_closes.get(code, self.marks.get(code)),
                                  ranks.get(code, math.inf), self.case, self.friction)
            factor = quotes.get(code, {}).get('factor', ep.last_factor)
            if not valid(factor): factor = ep.last_factor
            if action:
                self.intent(ep, *action, day, factor)
            if ranks.get(code, math.inf) > 15:
                self.intent(ep, 'rank_exit', True, day, factor)
        strong = [c for c in ordered[:2] if c not in self.positions and self.previous_ranks
                  and self.previous_ranks.get(c, math.inf) > 2
                  and self.cooldown.get(c, -1) <= step]
        # Decisions are frozen before fills. Project only exits actually sellable
        # at this open; an unfilled exit never silently frees a position slot.
        projected = set(self.positions)
        for c, ep in self.positions.items():
            if ep.pending_full and self.tradable(c, quotes.get(c), previous_closes.get(c), False)[0]:
                projected.discard(c)
        replacements = 0
        for incoming in strong:
            if not self.tradable(incoming, quotes.get(incoming), previous_closes.get(incoming), True)[0]:
                continue
            if len(projected) < 5:
                projected.add(incoming)
                continue
            candidates = [c for c in projected if c in self.positions and not self.positions[c].pending_reason
                          and self.tradable(c, quotes.get(c), previous_closes.get(c), False)[0]]
            if not candidates:
                continue
            worst = max(candidates, key=lambda c: (ranks.get(c, math.inf), c))
            if scores[incoming] > scores.get(worst, -math.inf) and replacements < 2:
                ep = self.positions[worst]
                self.intent(ep, 'strong_replacement', True, day, quotes[worst]['factor'])
                projected.remove(worst); projected.add(incoming); replacements += 1
        excluded = {c for c, ep in self.positions.items() if ep.pending_reason}
        # Any eligible candidate can fill a failed initial buy/vacancy, in rank order.
        buy_order = list(dict.fromkeys(strong+ordered))
        bought, sold = set(), set()
        for code, ep in list(self.positions.items()):
            if ep.pending_reason:
                before = ep.quantity
                self.sell(ep, day, step, quotes.get(code), previous_closes.get(code))
                if ep.quantity < before: sold.add(code)
        for code in buy_order:
            if len(self.positions) >= 5:
                break
            if code in self.positions or code in excluded or code in sold:
                continue
            if self.cooldown.get(code, -1) > step:
                self.events.append(dict(date=day, code=code, event='cooldown_blocks_entry',
                                        eligible_step=self.cooldown[code]))
                continue
            quote = quotes.get(code)
            if not self.tradable(code, quote, previous_closes.get(code), True)[0]:
                continue
            budget = min(nav_open*.19, max(0., self.cash-cash_floor))
            qty = buy_quantity(code, budget, quote['open'], quote['factor'], self.friction)
            if qty <= 0: continue
            self.counter += 1
            value, before = qty*quote['open'], self.cash
            costs = fee(value, True, self.friction)
            initial_cost = value+costs
            self.cash -= initial_cost
            ep = Episode(self.counter, code, day, step, nav_open, qty, qty, initial_cost,
                         min(nav_open*.02, initial_cost*.10), initial_cost/qty, paid_fees=costs,
                         last_factor=quote['factor'])
            self.positions[code] = ep
            self.marks[code] = quote['open']
            bought.add(code)
            self.orders.append(dict(date=day, code=code, episode_id=ep.episode_id,
                                    direction='BUY', quantity=qty, raw_quantity=qty*quote['factor'],
                                    price=quote['open'], raw_price=quote['open']/quote['factor'],
                                    notional=value, fee=costs, cash_before=before, cash_after=self.cash,
                                    reason='strong_entry' if code in strong else 'vacancy', trigger_day='prior_close'))
        assert not bought.intersection(sold) and self.cash >= -1e-6 and len(self.positions) <= 5
        nav = self.cash
        for code, ep in self.positions.items():
            close = quotes.get(code, {}).get('close')
            factor = quotes.get(code, {}).get('factor')
            if valid(factor): ep.last_factor = factor
            if valid(close): self.marks[code] = close
            mark = self.marks[code]
            nav += ep.quantity*mark
            self.holdings.append(dict(date=day, code=code, episode_id=ep.episode_id,
                                     quantity=ep.quantity, mark=mark, value=ep.quantity*mark,
                                     initial_cost=ep.initial_cost, sold_net=ep.sold_net,
                                     liquidation_pnl=ep.liquidation_pnl(mark, self.friction),
                                     budget=ep.budget, half_mode=ep.half_mode, pending_reason=ep.pending_reason))
        prev_nav = self.daily[-1]['nav'] if self.daily else self.initial_cash
        fills = self.orders[n_orders:]
        self.daily.append(dict(date=day, nav=nav, net_return=nav/prev_nav-1,
                               cash=self.cash, cash_ratio=self.cash/nav, holdings=len(self.positions),
                               fees=sum(o['fee'] for o in fills), traded=sum(o['notional'] for o in fills),
                               turnover=sum(o['notional'] for o in fills)/nav_open,
                               max_weight=max((ep.quantity*self.marks[c]/nav for c, ep in self.positions.items()), default=0.)))
        self.previous_ranks = ranks

    def open_episodes(self, last_day, last_step):
        return [{**asdict(ep), 'valuation_day': last_day, 'holding_days': last_step-ep.entry_step,
                 'pnl': ep.sold_net+ep.quantity*self.marks[c]-ep.initial_cost,
                 'liquidation_pnl': ep.liquidation_pnl(self.marks[c], self.friction)}
                for c, ep in self.positions.items()]

    def reconcile(self):
        buys = sum(o['notional']+o['fee'] for o in self.orders if o['direction']=='BUY')
        sales = sum(o['notional']-o['fee'] for o in self.orders if o['direction']=='SELL')
        cash_error = abs(self.initial_cash-buys+sales-self.cash)
        profit = sum(x['pnl'] for x in self.episodes)+sum(
            ep.sold_net+ep.quantity*self.marks[c]-ep.initial_cost for c, ep in self.positions.items())
        nav_error = abs(self.initial_cash+profit-self.daily[-1]['nav'])
        if max(cash_error, nav_error) > 1e-5:
            raise ValueError('Episode/cash/NAV reconciliation failed')
        for c in {o['code'] for o in self.orders}:
            q = sum(o['quantity']*(1 if o['direction']=='BUY' else -1) for o in self.orders if o['code']==c)
            expected = self.positions[c].quantity if c in self.positions else 0.
            if abs(q-expected)>1e-5: raise ValueError('Position quantity reconciliation failed')
        return dict(cash_error=cash_error, nav_error=nav_error, status='passed')
