import pandas as pd
import pytest

from paper_futu.research_s6_top12_r3_stop15 import schedule_dates, runtime_floor
from paper_futu.s6_explicit_account import schedule_dates as original_schedule


@pytest.mark.parametrize("offset",range(5))
def test_five_day_calendar_reproduces_existing_engine(offset):
    calendar=pd.bdate_range("2021-12-31","2023-01-20")
    expected=original_schedule(calendar,"2022-01-04","2023-01-20",offset)
    actual=schedule_dates(calendar,"2022-01-04","2023-01-20",offset,5)
    for a,b in zip(actual,expected): pd.testing.assert_index_equal(a,b)


@pytest.mark.parametrize("offset",range(3))
def test_three_day_cadence_has_causal_signals_and_no_year_reset(offset):
    calendar=pd.bdate_range("2021-12-31","2024-01-20")
    days,trades,signals=schedule_dates(calendar,"2022-01-04","2024-01-20",offset)
    assert (calendar.get_indexer(trades)-calendar.get_indexer(signals)==1).all()
    assert (pd.Series(calendar.get_indexer(trades)).diff().dropna()==3).all()
    assert (signals<trades).all()
    assert days[0]==pd.Timestamp("2022-01-04")
    assert days[-1]==pd.Timestamp("2024-01-19")
    assert len(set(trades.year))==3


def test_three_distinct_phases_cover_all_trade_days_without_duplication():
    calendar=pd.bdate_range("2021-12-31","2022-02-01")
    results=[schedule_dates(calendar,"2022-01-04","2022-02-01",o) for o in range(3)]
    phases=[set(r[1]) for r in results]
    assert set.union(*phases)==set(results[0][0])
    assert not phases[0]&phases[1] and not phases[1]&phases[2] and not phases[0]&phases[2]


def test_invalid_calendar_boundary_is_rejected():
    calendar=pd.bdate_range("2022-01-04","2022-02-01")
    with pytest.raises(ValueError): schedule_dates(calendar,"2022-01-04","2022-02-01",0)
    with pytest.raises(ValueError): schedule_dates(calendar,"2022-01-05","2022-02-01",3)
    with pytest.raises(ValueError): schedule_dates(calendar[::-1],"2022-01-05","2022-02-01",0)


@pytest.mark.parametrize("before,after",[(1.5,.8),(1.0,.5),(2.0,2.0)])
def test_only_the_explicitly_authorized_floors_are_remapped(before,after):
    assert runtime_floor(before)==after
