import pytest

from paper_futu.research_s6_stop_loss import CostBook


def test_average_cost_includes_purchase_fees_and_partial_sales_preserve_it():
    book = CostBook()
    book.fill("X", 1, 100, 1000, 5)
    book.fill("X", 1, 100, 1200, 5)
    assert book.return_at("X", 8.84) == pytest.approx(-.2)
    book.fill("X", 0, 50, 400, 5)
    assert book.return_at("X", 8.84) == pytest.approx(-.2)


def test_new_position_does_not_inherit_old_loss_basis():
    book = CostBook()
    book.fill("X", 1, 100, 1000, 0)
    book.fill("X", 0, 100, 700, 5)
    assert book.return_at("X", 7) is None
    book.fill("X", 1, 100, 700, 0)
    assert book.return_at("X", 7) == 0


def test_missing_price_and_rejected_fill_cannot_trigger_stop():
    book = CostBook()
    book.fill("X", 1, 100, 1000, 5)
    book.fill("X", 0, 100, 0, 0)
    assert book.return_at("X", float("nan")) is None
    assert book.return_at("X", None) is None
    assert book.positions["X"][0] == 100


@pytest.fixture
def stop_harness():
    """Exercise the real decision method with a tiny, deterministic exchange."""
    from types import SimpleNamespace
    import pandas as pd
    from qlib.backtest.decision import Order
    from paper_futu.research_s6_stop_loss import classes

    _, strategy_class = classes()

    class Position:
        def __init__(self):
            self.amounts = {"A": 100., "B": 100.}
        def get_stock_list(self): return list(self.amounts)
        def get_stock_amount(self, stock): return self.amounts[stock]
        def get_stock_amount_dict(self): return self.amounts.copy()

    class Calendar:
        day = pd.Timestamp("2024-01-03")
        def get_trade_step(self): return 1
        def get_step_time(self, step=None, shift=0):
            day = self.day-pd.Timedelta(days=shift)
            return day, day+pd.Timedelta(hours=23)

    class Exchange:
        def __init__(self):
            self.book = CostBook()
            self.book.fill("A", 1, 100, 1000, 0)
            self.book.fill("B", 1, 100, 1000, 0)
            self.prices = {"A": 8., "B": 12.}
            self.blocked = False
            self.targets = []
        def get_close(self, stock, start, end):
            assert start < calendar.day  # Cannot inspect the execution day's close.
            return self.prices[stock]
        def generate_order_for_target_amount_position(self, target, held, start, end):
            self.targets.append(target.copy())
            if self.blocked: return []
            return [Order(stock, quantity-target.get(stock, 0.), Order.SELL, start, end)
                    for stock, quantity in held.items() if quantity > target.get(stock, 0.)]

    calendar, position, exchange = Calendar(), Position(), Exchange()
    signal = SimpleNamespace(value=None)
    signal.get_signal = lambda **kwargs: signal.value
    captured = {}
    def weights(score, current, start, end):
        captured["score"] = score
        # Deliberately return a stopped holding to verify the final safety filter.
        return {"A": .4, "B": .6}
    def generator(**kwargs):
        captured["weights"] = kwargs["target_weight_position"]
        return exchange.generate_order_for_target_amount_position(
            {s:100. for s in captured["weights"]}, position.amounts,
            calendar.day, calendar.day)
    strategy = SimpleNamespace(threshold=.2, pending={}, stop_events=[],
        trade_calendar=calendar, trade_position=position, trade_exchange=exchange,
        signal=signal, generate_target_weight_position=weights,
        order_generator=SimpleNamespace(generate_order_list_from_target_weight_position=generator),
        get_risk_degree=lambda step: .95)
    return SimpleNamespace(decide=lambda: strategy_class.generate_trade_decision(strategy),
        strategy=strategy, exchange=exchange, calendar=calendar, position=position,
        signal=signal, captured=captured)


def test_stop_only_day_preserves_other_names_and_pending_exit_survives_rebound(stop_harness):
    h = stop_harness
    h.exchange.blocked = True
    assert h.decide().get_decision() == []
    assert set(h.strategy.pending) == {"A"}
    assert h.exchange.targets[-1] == {"B": 100.}
    h.exchange.prices["A"] = 11.
    h.exchange.blocked = False
    orders = h.decide().get_decision()
    assert len(orders) == 1
    assert (orders[0].stock_id, orders[0].amount, orders[0].direction) == ("A", 100., 0)
    assert orders[0].stop_loss
    assert len(h.strategy.stop_events) == 1


def test_rebalance_cannot_buy_back_stop_and_does_not_renormalize(stop_harness):
    import pandas as pd
    h = stop_harness
    h.signal.value = pd.Series({"A": 1., "B": .5})
    h.decide()
    assert "A" not in h.captured["score"].index
    assert h.captured["weights"] == {"B": .6}
    h.exchange.book.fill("A", 0, 100, 800, 5)
    del h.position.amounts["A"]
    h.calendar.day += pd.Timedelta(days=5)
    h.decide()
    assert not h.strategy.pending
    assert "A" in h.captured["score"].index  # Eligible again at a later rebalance.


def test_unavailable_close_does_not_create_a_stop(stop_harness):
    h = stop_harness
    h.exchange.prices["A"] = float("nan")
    assert h.decide().get_decision() == []
    assert h.strategy.stop_events == []
