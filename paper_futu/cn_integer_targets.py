"""Pure, deterministic integer targets; affordability is always rounded DOWN."""
from __future__ import annotations

import math

WEIGHT_TOLERANCE = .005


def board(code):
    if code.startswith("SH.688"):
        return "star"
    if code.startswith(("SH.600", "SH.601", "SH.603", "SH.605", "SZ.000", "SZ.001", "SZ.002", "SZ.003", "SZ.300", "SZ.301")):
        return "regular"
    raise ValueError("Unsupported A-share order unit: " + code)


def affordable_quantity(code, cash, price):
    if not math.isfinite(cash) or not math.isfinite(price) or price <= 0:
        raise ValueError("Invalid affordability inputs")
    count = max(0, math.floor((cash + 1e-8) / price))
    if board(code) == "star":
        return count if count >= 200 else 0
    return count // 100 * 100


def legal_order_quantity(code, quantity, side, held=0, sellable=None):
    quantity = max(0, int(quantity))
    if side == "SELL":
        quantity = min(quantity, int(held), int(held if sellable is None else sellable))
        if quantity == held:
            return quantity  # a complete liquidation may dispose of residual odd shares
    if board(code) == "star":
        return quantity if quantity >= 200 else 0
    return quantity // 100 * 100


def _reachable(code, held, target, sellable):
    if target == held:
        return True
    side = "BUY" if target > held else "SELL"
    return legal_order_quantity(code, abs(target-held), side, held, sellable) == abs(target-held)


def _nearby(code, held, ideal, sellable):
    """Nearest actionable targets, including inherited corporate-action odd shares."""
    values = {held, 0}
    if board(code) == "regular":
        count = (ideal-held)/100
        values.update(held + k*100 for k in (math.floor(count), math.ceil(count)))
        values.add(held-legal_order_quantity(code, held, "SELL", held, sellable))
    else:
        values.update((math.floor(ideal), math.ceil(ideal), held+200, held-200))
        values.add(held-legal_order_quantity(code, held, "SELL", held, sellable))
    return sorted(q for q in values if q >= 0 and _reachable(code, held, q, sellable))


def _lower(code, held, quantity, sellable):
    if quantity <= 0:
        return None
    step = 1 if board(code) == "star" else 100
    choices = {0, quantity-step, held, held+200, held-200}
    if board(code) == "regular":
        choices.add(held + math.floor((quantity-1-held)/100)*100)
    else:
        choices.add(held-legal_order_quantity(code, held, "SELL", held, sellable))
    choices = [q for q in choices if 0 <= q < quantity and _reachable(code, held, q, sellable)]
    return max(choices) if choices else None


def _fee(value, buying):
    return max(5., value*(.0005 if buying else .0015)) if value > 0 else 0.


def allocate_targets(selected, weights, nav, positions, quotes, cash,
                     risk_degree=.95, tolerance=WEIGHT_TOLERANCE, bought_today=None):
    """Choose nearest lots, then deterministic minimum-error-per-yuan budget reductions.

    positions/quotes are mappings keyed by Futu code. Never changes selection.
    Frozen reference quotes, not future closes, value the target portfolio.
    """
    if not math.isfinite(nav) or nav <= 0 or not math.isfinite(cash) or cash < 0:
        raise ValueError("Invalid account capital")
    if len(selected) != len(set(selected)) or set(selected) != set(weights):
        raise ValueError("Invalid selected names/weights")
    if not 0 < risk_degree <= 1 or tolerance < 0 or abs(sum(weights.values())-risk_degree) > 1e-8:
        raise ValueError("Invalid portfolio budget")
    bought_today = bought_today or {}
    held = {c:int(r["qty"]) for c,r in positions.items() if r["qty"] > 0}
    codes = sorted(set(selected) | set(held))
    prices, sellable, targets, reasons = {}, {}, {}, []
    for code in codes:
        row = quotes[code]
        price = float(row["last_price"])
        if not math.isfinite(price) or price <= 0:
            raise ValueError("Invalid reference quote: " + code)
        prices[code] = price
        h = held.get(code, 0)
        can_sell = min(h-int(bought_today.get(code, 0)), int(positions.get(code, {}).get("can_sell_qty", 0)))
        previous = float(row.get("prev_close_price", 0))
        suspended = bool(row.get("suspension", False))
        if suspended or previous <= 0 or price/previous-1 <= -.095:
            can_sell = 0
        sellable[code] = max(0, can_sell)
        w = weights.get(code, 0.)
        if code not in weights:
            targets[code] = h-legal_order_quantity(code, h, "SELL", h, sellable[code])
            continue
        candidates = _nearby(code, h, w*nav/price, sellable[code])
        candidates = [q for q in candidates if q*price <= (w+tolerance)*nav+1e-6]
        if suspended or previous <= 0 or price/previous-1 >= .095:
            candidates = [q for q in candidates if q <= h]
        if not candidates:
            # An unsellable inherited overweight is not permission to add risk.
            targets[code] = h
            reasons.append(dict(code=code, reason="locked_inherited_position"))
        else:
            targets[code] = min(candidates, key=lambda q:(abs(q*price-w*nav), abs(q-h), q))

    rank = {code:i for i,code in enumerate(selected)}
    def metrics():
        exposure = sum(targets[c]*prices[c] for c in codes)
        post_cash = cash
        for c in codes:
            change = targets[c]-held.get(c,0)
            value = abs(change)*prices[c]
            post_cash += (-value-_fee(value,True)) if change > 0 else (value-_fee(value,False))
        return exposure, post_cash

    while True:
        exposure, post_cash = metrics()
        if exposure <= nav*risk_degree+1e-6 and post_cash >= nav*(1-risk_degree)+10-1e-6:
            break
        choices = []
        for c in selected:
            lower = _lower(c, held.get(c,0), targets[c], sellable[c])
            if lower is None:
                continue
            released = (targets[c]-lower)*prices[c]
            wanted = weights[c]*nav
            penalty = (abs(lower*prices[c]-wanted)-abs(targets[c]*prices[c]-wanted))/released
            choices.append((penalty, 0 if not held.get(c,0) else 1, -rank[c], c, lower))
        if not choices:
            # Keep necessary sells, suppress ALL incremental buys on an infeasible budget.
            for c in codes:
                targets[c] = min(targets[c], held.get(c,0))
            reasons.append(dict(reason="infeasible_locked_budget_no_new_buys"))
            break
        _, _, _, c, lower = min(choices)
        targets[c] = lower
        reasons.append(dict(code=c, reason="portfolio_budget_rounding_reduction"))
    actions = []
    for code in codes:
        h, q = held.get(code,0), targets[code]
        if q != h:
            actions.append(dict(code=code, side="BUY" if q > h else "SELL", qty=abs(q-h),
                                held=h, target=q, weight=weights.get(code,0.), reference_price=prices[code]))
    actions.sort(key=lambda a:(a["side"] != "SELL", rank.get(a["code"],len(rank)),a["code"]))
    exposure, post_cash = metrics()
    return dict(targets=targets, actions=actions, reasons=reasons, reference_prices=prices,
                target_exposure=exposure, expected_post_cash=post_cash,
                weight_tolerance=tolerance, sizing_rule="nearest_legal_lot_v5")
