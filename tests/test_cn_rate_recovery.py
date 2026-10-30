import pandas as pd
import pytest

from paper_futu.cn_broker_queries import WindowBudget,PositionCache
from paper_futu.resume_cn_rebalance import verified_original,remaining_buys


def test_sliding_budget_never_exceeds_eight_per_thirty_seconds():
    now=[0.];sleeps=[]
    def wait(seconds):sleeps.append(seconds);now[0]+=seconds
    b=WindowBudget(clock=lambda:now[0],sleep=wait)
    stamps=[]
    for _ in range(25):b.acquire();stamps.append(now[0])
    for stamp in stamps:assert sum(stamp-30<s<=stamp for s in stamps)<=8
    assert len(sleeps)==3


def test_position_cache_avoids_per_stock_remote_queries():
    now=[0.];calls=[]
    c=PositionCache(lambda:calls.append(now[0]) or pd.DataFrame(),initial=pd.DataFrame(),clock=lambda:now[0])
    for _ in range(20):c.get()
    assert calls==[]
    now[0]=10;c.get();c.get()
    assert calls==[10]
    c.get(force=True);assert calls==[10,10]


def fixture_batch():
    plan=dict(signal_date="2026-10-19",execution_date="2026-10-20",model_dir="frozen",top_k=2,n_drop=1,
        risk_degree=.95,top_tier_size=2,top_tier_weight=1.,predictions={"SH.600000":1.,"SZ.000001":.5})
    start=dict(**{k:plan[k] for k in ["signal_date","execution_date","model_dir","top_k","n_drop"]},
        selected=["SH.600000","SZ.000001"],weights={"SH.600000":.475,"SZ.000001":.475},
        actions=[dict(code="SH.600000",target=100),dict(code="SZ.000001",target=200)])
    receipt=dict(event="submitted",side="SELL",code="OLD",qty=100,order_id="9")
    before=dict(event="before_submit",side="SELL",code="OLD",qty=100)
    events=[dict(event="execution_start",**start),receipt,before]
    orders=pd.DataFrame([dict(order_id="9",code="OLD",trd_side="SELL",qty=100.,dealt_qty=100.,order_status="FILLED_ALL")])
    return plan,start,events,orders


def test_original_sells_need_exact_terminal_receipts():
    plan,start,events,orders=fixture_batch()
    assert verified_original(plan,events,orders)[1]==["9"]
    bad=orders.copy();bad.loc[0,"dealt_qty"]=50
    with pytest.raises(ValueError,match="fully verified"):verified_original(plan,events,bad)
    bad=orders.copy();bad.loc[0,"order_status"]="SUBMITTED"
    with pytest.raises(ValueError,match="fully verified"):verified_original(plan,events,bad)
    with pytest.raises(ValueError,match="Unaccounted"):verified_original(plan,events,pd.concat([orders,orders.assign(order_id="manual")]))


def test_resume_only_positive_buy_gaps_no_repeated_sells_or_reselection():
    _,start,_,_=fixture_batch()
    pos=pd.DataFrame([dict(code="SH.600000",qty=100)])
    quotes=pd.DataFrame(dict(code=start["selected"],last_price=[10.,10.])).set_index("code")
    actions=remaining_buys(start,pos,10000.,quotes)
    assert actions==[dict(code="SZ.000001",qty=200,held=0,original_target=200,live_capped_target=200)]
    pos=pd.DataFrame([dict(code="UNEXPECTED",qty=100)])
    with pytest.raises(ValueError,match="Unexpected holding"):remaining_buys(start,pos,10000.,quotes)
