"""Account filtering must never fall back to a live trading account."""

import pandas as pd
import pytest

from paper_futu.check_connection import paper_accounts, stock_paper_accounts
from paper_futu.smoke_order import checked_price


def test_only_paper_accounts_are_returned():
    frame = pd.DataFrame(
        {
            "acc_id": [111, 222, 333],
            "trd_env": ["REAL", "SIMULATE", "REAL"],
        }
    )
    result = paper_accounts(frame)
    assert result.acc_id.tolist() == [222]
    assert result.trd_env.eq("SIMULATE").all()


def test_no_paper_account_does_not_select_a_live_account():
    frame = pd.DataFrame({"acc_id": [111], "trd_env": ["REAL"]})
    assert paper_accounts(frame).empty


def test_missing_environment_field_is_rejected():
    with pytest.raises(ValueError):
        paper_accounts(pd.DataFrame({"acc_id": [111]}))


def test_stock_account_filter_rejects_real_futures_disabled_and_wrong_market():
    frame = pd.DataFrame(
        {
            "acc_id": [1, 2, 3, 4, 5],
            "trd_env": ["SIMULATE", "REAL", "SIMULATE", "SIMULATE", "SIMULATE"],
            "sim_acc_type": ["STOCK_AND_OPTION", "STOCK_AND_OPTION", "FUTURES", "STOCK_AND_OPTION", "STOCK"],
            "trdmarket_auth": [["US"], ["US"], ["US"], ["US"], ["HK"]],
            "acc_status": ["ACTIVE", "ACTIVE", "ACTIVE", "DISABLED", "ACTIVE"],
        }
    )
    assert stock_paper_accounts(frame, "US")["acc_id"].tolist() == [1]
    assert stock_paper_accounts(frame, "HK")["acc_id"].tolist() == [5]


def test_stock_account_filter_requires_market_and_account_type():
    with pytest.raises(ValueError):
        stock_paper_accounts(pd.DataFrame({"acc_id": [1], "trd_env": ["SIMULATE"]}), "US")


@pytest.mark.parametrize("price", [0, -1, 1000.01, float("nan"), float("inf")])
def test_smoke_order_rejects_invalid_or_oversized_prices(price):
    with pytest.raises(ValueError):
        checked_price(price)


def test_smoke_price_is_rounded_to_legal_us_tick():
    assert checked_price(779.1897) == 779.19
