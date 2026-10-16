from datetime import datetime

import pytest

from paper_futu.cn_overnight_probe import SCHEDULED, test_price as checked_price, validate_time


def test_probe_is_one_lot_under_500_cny():
    assert checked_price(4.6) == 4.6


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), 5.01])
def test_probe_rejects_invalid_price_or_excess_budget(value):
    with pytest.raises(ValueError):
        checked_price(value)


@pytest.mark.parametrize("stamp", ["2026-10-12T21:54:59+08:00", "2026-10-12T21:58:01+08:00",
                                    "2026-10-13T21:55:00+08:00"])
def test_probe_rejects_early_stale_or_wrong_date(stamp):
    with pytest.raises(ValueError):
        validate_time(datetime.fromisoformat(stamp), SCHEDULED)


def test_probe_accepts_only_authorized_time_window():
    validate_time(datetime.fromisoformat(SCHEDULED), SCHEDULED)
