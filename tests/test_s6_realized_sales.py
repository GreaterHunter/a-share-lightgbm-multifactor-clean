import pandas as pd
import pytest

from scripts.analyze_s6_realized_sales import reconstruct_sales, stock_date_sales, summarize


def order(day, direction, qty, price, fee=5):
    return dict(datetime=pd.Timestamp(day), instrument="TEST", direction=direction,
                adjusted_quantity=qty, notional=qty*price, fee=fee)


def test_partial_sale_releases_average_purchase_cost_and_fees():
    orders = pd.DataFrame([order("2022-01-04", 1, 100, 10),
                           order("2022-01-05", 1, 100, 12),
                           order("2022-01-06", 0, 50, 13)])
    fills, ledger = reconstruct_sales(orders)
    assert fills.iloc[0].gross_basis == pytest.approx(550)
    assert fills.iloc[0].net_basis == pytest.approx(552.5)
    assert fills.iloc[0].net_pnl == pytest.approx(92.5)
    assert ledger["TEST"]["quantity"] == pytest.approx(150)
    assert ledger["TEST"]["net_cost"] == pytest.approx(1657.5)


def test_flat_price_is_a_net_loss_and_rejections_do_not_change_basis():
    orders = pd.DataFrame([order("2022-01-04", 1, 100, 10),
                           order("2022-01-05", 0, 0, 10, 0),
                           order("2022-01-06", 0, 100, 10)])
    fills, ledger = reconstruct_sales(orders)
    holdings = pd.DataFrame(columns=["datetime", "instrument", "adjusted_quantity"])
    events = stock_date_sales(fills, holdings)
    assert events.iloc[0].gross_class == "flat"
    assert events.iloc[0].net_class == "loss"
    assert events.iloc[0].net_return == pytest.approx(-10/1005)
    assert events.iloc[0].full_exit
    assert ledger["TEST"]["net_cost"] == 0


def test_multiple_fills_count_once_and_zero_sale_schedule_days_are_explicit():
    orders = pd.DataFrame([order("2022-01-04", 1, 200, 10),
                           order("2022-01-06", 0, 100, 12),
                           order("2022-01-06", 0, 100, 14)])
    fills, _ = reconstruct_sales(orders)
    holdings = pd.DataFrame(columns=["datetime", "instrument", "adjusted_quantity"])
    events = stock_date_sales(fills, holdings)
    result = summarize(events, ["2022-01-04", "2022-01-06"])
    assert result["sold_stock_date_events"] == 1
    assert result["gross"]["profit"]["average_return"] == pytest.approx(.3)
    assert result["gross"]["profit"]["average_count_per_scheduled_rebalance"] == .5
    assert result["gross"]["profit"]["average_count_per_selling_rebalance"] == 1


def test_missing_buy_fails_closed():
    with pytest.raises(ValueError, match="Oversell"):
        reconstruct_sales(pd.DataFrame([order("2022-01-04", 0, 100, 10)]))
