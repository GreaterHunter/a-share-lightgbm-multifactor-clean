import pandas as pd
import pytest

from scripts.analyze_s6_trade_quality import sale_ages


def order(day, direction, qty):
    return dict(datetime=pd.Timestamp(day), instrument="TEST", direction=direction,
                adjusted_quantity=qty, notional=qty*10)


def test_weighted_buy_dates_and_partial_sales_do_not_reset_age():
    calendar = pd.bdate_range("2022-01-03", periods=6)
    orders = pd.DataFrame([order(calendar[0], 1, 100),
                           order(calendar[2], 1, 100),
                           order(calendar[3], 0, 100),
                           order(calendar[5], 0, 100)])
    result = sale_ages(orders, calendar)
    assert (result.age_trading_mass/result.quantity).tolist() == pytest.approx([2, 4])
    assert (result.age_calendar_mass/result.quantity).tolist() == pytest.approx([2, 6])


def test_full_exit_then_reentry_resets_entry_date():
    calendar = pd.bdate_range("2022-01-03", periods=6)
    orders = pd.DataFrame([order(calendar[0], 1, 100),
                           order(calendar[2], 0, 100),
                           order(calendar[4], 1, 100),
                           order(calendar[5], 0, 100)])
    result = sale_ages(orders, calendar)
    assert (result.age_trading_mass/result.quantity).tolist() == pytest.approx([2, 1])


def test_missing_calendar_date_fails_closed():
    with pytest.raises(ValueError, match="missing"):
        sale_ages(pd.DataFrame([order("2022-01-01", 1, 100)]),
                  pd.bdate_range("2022-01-03", periods=6))
