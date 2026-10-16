"""Audited Qlib opening auction account replay, fees, and holding contributions."""
from __future__ import annotations

from collections import defaultdict
from functools import partial
import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.portfolio_strategy import TieredTopKWeightStrategy
from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy.order_generator import OrderGenerator
from qlib.contrib.evaluate import backtest_daily
from src.compare_models import enrich_daily_report, summarize_report
from paper_futu.final_ablation_core import AS_OF, DATA, REPORT, START, Store, digest, fee, read_json

_EXCHANGE = None


class AuditedOrderGenerator(OrderGenerator):
    """Size on the preceding close, then check the actual order direction."""
    def generate_order_list_from_target_weight_position(
        self, current, trade_exchange, target_weight_position, risk_degree,
        pred_start_time, pred_end_time, trade_start_time, trade_end_time,
    ):
        if target_weight_position is None:
            return []
        budget = risk_degree * current.calculate_value()
        amounts = {}
        for stock, weight in target_weight_position.items():
            price = trade_exchange.get_close(stock, pred_start_time, pred_end_time)
            if price is None or not np.isfinite(price) or price <= 0:
                if not current.check_stock(stock):
                    trade_exchange.record_generation(stock, trade_start_time, Order.BUY,
                                                     "missing_previous_close")
                    continue
                price = current.get_stock_price(stock)
            if price is None or not np.isfinite(price) or price <= 0:
                raise ValueError(f"No causal valuation price for held stock {stock}")
            amounts[stock] = budget * weight / price
        return trade_exchange.generate_order_for_target_amount_position(
            amounts, current.get_stock_amount_dict(), trade_start_time, trade_end_time)


class AuditedExchange(Exchange):
    """Base commission floor plus separate friction, no day's closing-price fallback."""
    def reset_audit(self, friction, top_k=None):
        self.friction = friction
        self.top_k = top_k
        self.journal = []
        self.generation_journal = []
        self.today_bought = defaultdict(float)

    def record_generation(self, stock, day, direction, reason):
        self.generation_journal.append(dict(datetime=pd.Timestamp(day).normalize(),
            instrument=stock, direction=int(direction), reason=reason))

    def check_holding_limit(self, position, day, phase):
        if position is not None and self.top_k is not None:
            names = sorted(position.get_stock_list())
            if len(names) > self.top_k:
                raise ValueError(f"Account has more names than its fixed target K: "
                    f"date={day}, phase={phase}, K={self.top_k}, count={len(names)}, "
                    f"holdings={names}, recent_orders={self.journal[-10:]}")

    def generate_order_for_target_amount_position(self, target_position, current_position,
                                                  start_time, end_time):
        sells, buys = [], []
        # Target insertion order is the deterministic signal rank (head tier first).
        stocks = sorted(set(current_position)-set(target_position)) + list(target_position)
        for stock in stocks:
            held, target = current_position.get(stock, 0.), target_position.get(stock, 0.)
            if held == target:
                continue
            direction = Order.BUY if target > held else Order.SELL
            if not self.is_stock_tradable(stock, start_time, end_time, direction=direction):
                self.record_generation(stock, start_time, direction, "suspended_or_direction_limit")
                continue
            factor = self.get_factor(stock, start_time, end_time)
            amount = abs(self.get_real_deal_amount(held, target, factor))
            if amount <= 0:
                self.record_generation(stock, start_time, direction, "below_round_lot")
                continue
            order = Order(stock, amount, direction, start_time, end_time, factor=factor)
            (buys if direction == Order.BUY else sells).append(order)
        return sells + buys

    def check_stock_suspended(self, stock_id, start_time, end_time):
        if stock_id not in self.quote.get_all_stock():
            return True
        price = self.get_quote_info(stock_id, start_time, end_time, "$open")
        factor = self.get_factor(stock_id, start_time, end_time)
        return price is None or not np.isfinite(price) or price <= 0 or factor is None or not np.isfinite(factor) or factor <= 0

    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        price = float(self.get_quote_info(order.stock_id, order.start_time, order.end_time, "$open"))
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        amount = max(0., order.amount)
        if order.direction == Order.SELL:
            current = position.get_stock_amount(order.stock_id) if position and position.check_stock(order.stock_id) else 0.
            bought = self.today_bought[(pd.Timestamp(order.start_time).normalize(), order.stock_id)]
            amount = min(amount, max(0., current-bought))
            if not np.isclose(amount, current):
                amount = self.round_amount_by_trade_unit(amount, order.factor)
        else:
            if (position is not None and self.top_k is not None
                and not position.check_stock(order.stock_id)
                and len(position.get_stock_list()) >= self.top_k):
                amount = 0.
                order.audit_reason = "no_slot_after_actual_sells"
            amount = self.round_amount_by_trade_unit(amount, order.factor)
            if position:
                cash = position.get_cash()
                # Solve both commission-floor branches before rounding to raw 100-share lots.
                affordable = min(cash / (1 + .0005 + self.friction),
                                 max(0., (cash-5.) / (1+self.friction))) / price
                if amount*price + fee(amount*price, Order.BUY, self.friction) > cash:
                    # Qlib's +0.1 raw-share rounding tolerance must not raise a cash cap.
                    # The unit is in adjusted shares (100/factor), like amount and affordable.
                    unit = self.get_amount_of_trade_unit(factor=order.factor)
                    amount = min(amount, affordable)
                    if unit is not None:
                        amount = np.floor(amount/unit)*unit
                    # Protect the boundary against floating-point fee arithmetic as well.
                    while amount > 0 and amount*price + fee(amount*price, Order.BUY, self.friction) > cash:
                        amount = max(0., amount-unit) if unit is not None else np.nextafter(amount, 0.)
        order.deal_amount = max(0., amount)
        value = order.deal_amount * price
        return price, value, fee(value, int(order.direction), self.friction)

    def deal_order(self, order, trade_account=None, position=None, dealt_order_amount=None):
        live = trade_account.current_position if trade_account is not None else position
        day = pd.Timestamp(order.start_time).normalize()
        self.check_holding_limit(live, day, "before_order")
        names_before = len(live.get_stock_list()) if live is not None else None
        cash_before = live.get_cash() if live is not None else None
        result = super().deal_order(order, trade_account, position,
                                    dealt_order_amount if dealt_order_amount is not None else defaultdict(float))
        value, cost, price = result
        factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        if order.direction == Order.BUY and value > 0:
            self.today_bought[(day, order.stock_id)] += order.deal_amount
        self.journal.append(dict(datetime=day, instrument=order.stock_id, direction=int(order.direction),
            requested_adjusted_quantity=order.amount, adjusted_quantity=order.deal_amount,
            raw_quantity=order.deal_amount*factor if factor is not None else np.nan,
            adjusted_price=price, raw_price=price/factor if factor is not None else np.nan,
            notional=value, fee=cost, friction=self.friction,
            status="filled" if value > 0 else "rejected",
            reason=("filled" if value > 0 else getattr(order, "audit_reason", "limit_t1_lot_or_cash")),
            holdings_before=names_before,
            holdings_after=len(live.get_stock_list()) if live is not None else None,
            cash_before=cash_before, cash_after=live.get_cash() if live is not None else None,
            target_k=self.top_k))
        self.check_holding_limit(live, day, "after_order")
        return result


def select_holdings(score, current, top_k, n_drop, buyable, sellable):
    """Keep missing-score holdings visible; substitute at most n_drop existing names."""
    values = score.dropna().to_dict()
    rank = lambda stock: (-values.get(stock, -np.inf), stock)
    selected = set(current)
    if len(selected) > top_k:
        raise ValueError(f"Account has more names than its fixed target K: "
                         f"K={top_k}, count={len(selected)}, holdings={sorted(selected)}")
    new = sorted(set(buyable)-selected, key=rank)
    while len(selected) < top_k and new:
        selected.add(new.pop(0))
    eligible_old = set(current) & set(sellable)
    changes = 0
    for candidate in new:
        old = sorted(eligible_old, key=rank)
        if not old or changes >= n_drop:
            break
        worst = old[-1]
        if values.get(candidate, -np.inf) <= values.get(worst, -np.inf):
            break
        selected.remove(worst)
        selected.add(candidate)
        eligible_old.remove(worst)
        changes += 1
    return sorted(selected, key=rank)


class AuditedStrategy(TieredTopKWeightStrategy):
    def __init__(self, **kwargs):
        super().__init__(order_generator_cls_or_obj=AuditedOrderGenerator, **kwargs)

    def generate_target_weight_position(self, score, current, trade_start_time, trade_end_time):
        self.trade_exchange.check_holding_limit(current, trade_start_time, "selection")
        if isinstance(score, pd.DataFrame):
            score = score.iloc[:, 0]
        score = score.dropna()
        buyable = [s for s in score.index if self.trade_exchange.is_stock_tradable(
            s, trade_start_time, trade_end_time, direction=Order.BUY)]
        sellable = [s for s in current.get_stock_list() if self.trade_exchange.is_stock_tradable(
            s, trade_start_time, trade_end_time, direction=Order.SELL)]
        selected = select_holdings(score, current.get_stock_list(), self.top_k, self.n_drop, buyable, sellable)
        if not selected:
            return {}
        head = min(self.top_tier_size, len(selected))
        if head == len(selected):
            return {s: 1./len(selected) for s in selected}
        return {s: self.top_tier_weight/head if i < head else (1-self.top_tier_weight)/(len(selected)-head)
                for i, s in enumerate(selected)}


def exchange(codes, calendar):
    global _EXCHANGE
    if _EXCHANGE is None:
        _EXCHANGE = AuditedExchange(freq="day", start_time=calendar[calendar < START][-1],
            end_time=AS_OF, codes=codes, deal_price="open", open_cost=.0005, close_cost=.0015,
            min_cost=5, trade_unit=100, limit_threshold=(
                "($open / Ref($close, 1) - 1) >= 0.095", "($open / Ref($close, 1) - 1) <= -0.095"))
    return _EXCHANGE


def summarize_accounts(report, case_id):
    groups = [("all", report), ("2022_2025", report.loc[report.index.year < 2026])]
    groups += [(str(y), g) for y, g in report.groupby(report.index.year)]
    rows = []
    for period, frame in groups:
        row = summarize_report(frame, case_id).iloc[0].to_dict()
        row.update(period=period, start_date=str(frame.index[0].date()), end_date=str(frame.index[-1].date()),
                   total_fees_cny=float(frame.fees_cash.sum()), average_holdings=float(frame.holdings_count.mean()),
                   minimum_holdings=int(frame.holdings_count.min()), maximum_holdings=int(frame.holdings_count.max()),
                   turnover_sum=float(frame.turnover.sum()))
        rows.append(row)
    without = report.loc[report.index.year != 2024]
    diagnostic = summarize_report(without, case_id).iloc[0].to_dict()
    diagnostic.update(period="excluding_2024_diagnostic", discontinuous_diagnostic=True)
    rows.append(diagnostic)
    return rows


def replay(prediction_path, portfolio, rebalance, offset=0, friction=0.):
    from qlib.data import D
    store = Store()
    configuration = dict(prediction_path=str(prediction_path), prediction_sha256=digest(prediction_path),
                         portfolio=portfolio, rebalance=rebalance, offset=offset, friction=friction,
                         initial_cash=1_000_000, risk_degree=.95, trade_unit=100,
                         execution="T+1 open", annualization=252, engine="audited_qlib_v2")

    def execute(key):
        prediction = pd.read_parquet(prediction_path)
        calendar = pd.DatetimeIndex(D.calendar(freq="day"))
        trade_dates = calendar[(calendar >= START) & (calendar <= AS_OF)]
        schedule = trade_dates[offset::rebalance]
        signal_dates = calendar[calendar.get_indexer(schedule)-1]
        signal = prediction.loc[prediction.index.get_level_values("datetime").isin(signal_dates), ["prediction"]]
        if len(signal_dates) != signal.index.get_level_values("datetime").nunique():
            raise ValueError("Missing scheduled signal dates")
        # Use the same source universe of quote codes for every model and portfolio.
        raw_schema = pd.read_parquet(DATA / "raw_features.parquet", columns=[])
        codes = sorted(raw_schema.index.get_level_values("instrument").unique())
        market = exchange(codes, calendar)
        market.reset_audit(friction, portfolio["top_k"])
        strategy = AuditedStrategy(signal=signal, risk_degree=.95, only_tradable=True, **portfolio)
        module = importlib.import_module("qlib.backtest.backtest")
        with patch.object(module, "tqdm", partial(module.tqdm, disable=True)):
            raw, positions = backtest_daily(start_time=calendar[calendar < START][-1], end_time=AS_OF,
                strategy=strategy, executor=SimulatorExecutor(time_per_step="day",
                generate_portfolio_metrics=True, verbose=False,
                trade_type=SimulatorExecutor.TT_SERIAL), account=1_000_000,
                benchmark="SH000300", exchange_kwargs={"exchange": market})
        report = enrich_daily_report(raw.reindex(trade_dates), key)
        if report[["net_return", "cost", "account"]].isna().any().any():
            raise ValueError("Account does not cover the common comparison interval")
        opening = D.features(["SH000300"], ["$open", "$close"], start_time=trade_dates[0], end_time=trade_dates[0]).iloc[0]
        report.loc[trade_dates[0], "bench"] = opening.iloc[1]/opening.iloc[0]-1
        report.loc[AS_OF, "bench"] = read_json(DATA / "benchmark_audit.json")["last_daily_return"]
        if report.bench.isna().any():
            raise ValueError("Benchmark missing outside the independently verified terminal patch")
        report["active_return"] = report.net_return-report.bench
        report["benchmark_nav"] = (1+report.bench).cumprod()
        report["relative_excess_nav"] = report.portfolio_nav/report.benchmark_nav
        prior = report.account.shift(1, fill_value=1_000_000.)
        orders = pd.DataFrame(market.journal)
        trades = orders.loc[orders.notional.gt(0)]
        totals = trades.groupby("datetime")[["notional", "fee"]].sum().reindex(trade_dates, fill_value=0)
        report["fees_cash"] = totals.fee
        report["traded_notional"] = totals.notional
        report["turnover"] = totals.notional/prior
        if not np.allclose(report.cost*prior, totals.fee, atol=1e-5):
            raise ValueError("Qlib cost rates do not match the transaction fee ledger")
        holdings, contributions = [], []
        previous_values = {}
        for day in trade_dates:
            position = positions[day]
            market.check_holding_limit(position, day, "daily_close")
            current_values = {}
            for stock in position.get_stock_list():
                quantity, price = position.get_stock_amount(stock), position.get_stock_price(stock)
                value = quantity*price
                current_values[stock] = value
                holdings.append(dict(datetime=day, instrument=stock, adjusted_quantity=quantity,
                                     adjusted_close=price, market_value=value, cash=position.get_cash()))
            today = trades.loc[trades.datetime.eq(day)]
            flows = today.assign(flow=np.where(today.direction.eq(Order.BUY), today.notional, -today.notional)).groupby("instrument").flow.sum()
            for stock in set(previous_values) | set(current_values) | set(flows.index):
                pnl = current_values.get(stock, 0)-previous_values.get(stock, 0)-flows.get(stock, 0)
                contributions.append(dict(datetime=day, instrument=stock, pnl_before_fees=pnl))
            previous_values = current_values
            if position.get_cash() < -1e-5:
                raise ValueError("Negative available cash")
        holding_frame = pd.DataFrame(holdings)
        pnl_frame = pd.DataFrame(contributions)
        report["holdings_count"] = holding_frame.groupby("datetime").size().reindex(trade_dates, fill_value=0)
        gross_pnl = pnl_frame.groupby("datetime").pnl_before_fees.sum().reindex(trade_dates, fill_value=0)
        if not np.allclose(prior+gross_pnl-totals.fee, report.account, atol=1e-4, rtol=1e-10):
            raise ValueError("Holdings and transaction cash flows do not reconstruct account NAV")
        if not np.allclose((1+report.net_return).cumprod()*1_000_000, report.account, atol=1e-4, rtol=1e-10):
            raise ValueError("Returns do not reconstruct account NAV")
        rows = summarize_accounts(report, key)
        year_rows = [r for r in rows if r["period"].isdigit()]
        all_row = next(r for r in rows if r["period"] == "all")
        if not np.isclose(np.prod([1+r["total_return"] for r in year_rows])-1, all_row["total_return"], atol=1e-10):
            raise ValueError("Annual returns do not compound to the whole period")
        directory = REPORT / "accounts" / key
        directory.mkdir(parents=True, exist_ok=True)
        report.to_parquet(directory/"daily.parquet")
        orders.to_parquet(directory/"orders.parquet", index=False)
        pd.DataFrame(market.generation_journal, columns=["datetime", "instrument", "direction", "reason"]).to_parquet(
            directory/"generation_diagnostics.parquet", index=False)
        holding_frame.to_parquet(directory/"holdings.parquet", index=False)
        pnl_frame.to_parquet(directory/"contributions.parquet", index=False)
        for row in rows:
            row.update(case_id=key, prediction_path=str(prediction_path), rebalance=rebalance,
                       offset=offset, friction=friction, **portfolio)
        pd.DataFrame(rows).to_csv(directory/"metrics.csv", index=False, encoding="utf-8-sig")
        result = dict(rows=rows, directory=str(directory), config=configuration)
        return result, list(directory.iterdir()), dict(annualized_return=all_row["annualized_return"],
                                                      max_drawdown=all_row["max_drawdown"], fees_cny=all_row["total_fees_cny"])
    return store.run("account", configuration, execute)
