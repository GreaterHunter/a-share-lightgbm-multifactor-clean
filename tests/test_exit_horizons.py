import pandas as pd
import pytest
from paper_futu.diagnose_exit_horizons import future_session,kept_minus_sold,describe_holds
from paper_futu.research_rank_buffer_futu_fees import fee

def test_exact_tenth_session_and_no_future_padding():
    cal=pd.bdate_range('2025-12-01','2025-12-31')
    assert future_session(cal,cal[0])==cal[10]
    assert future_session(cal,cal[-10]) is None
    assert future_session(cal,cal[-11])==cal[-1]
    assert future_session(cal,'2025-12-06') is None

def test_counterfactual_sign_and_no_duplicate_entry_fee():
    assert kept_minus_sold(100,100,100,fee(10000,False))==pytest.approx(0)
    assert kept_minus_sold(100,100,105,fee(10000,False))==pytest.approx(500-fee(10500,False)+fee(10000,False))
    assert kept_minus_sold(100,100,95,fee(10000,False))<0

def test_holding_days_denominator_closed_cycles():
    r=describe_holds(pd.DataFrame({'holding_days':[1,9,10,20],'pnl':[1,-1,1,-1]}))
    assert r['share_lt10']==.5 and r['share_eq10']==.25 and r['share_gt10']==.25
    assert r['median_holding_days']==9.5
