import numpy as np
import pandas as pd
import pytest
from paper_futu.research_rank_cooldown_h9 import labels_from_open,make_account,decomposition

def test_h9_endpoints_and_maturity():
    cal=pd.bdate_range('2025-01-01',periods=30)
    opens=pd.DataFrame({'x':np.arange(100.,130.)},index=cal)
    idx=pd.MultiIndex.from_product([['x'],cal],names=['instrument','datetime'])
    y=labels_from_open(opens,idx,cal,9)
    assert y.label.iloc[0]==pytest.approx(110/101-1)
    assert y.label_end_date.iloc[0]==cal[10]
    assert y.label.iloc[-10:].isna().all()
    cut=cal[18];changed=opens.copy();changed.loc[cal[19]:]*=7
    new=labels_from_open(changed,idx,cal,9);mask=y.label_end_date<=cut
    pd.testing.assert_frame_equal(y.loc[mask],new.loc[mask])

def test_missing_open_does_not_compress_calendar():
    cal=pd.bdate_range('2025-01-01',periods=20)
    opens=pd.DataFrame({'x':np.arange(100.,120.)},index=cal);opens.iloc[10,0]=np.nan
    idx=pd.MultiIndex.from_product([['x'],cal],names=['instrument','datetime'])
    y=labels_from_open(opens,idx,cal,9);assert pd.isna(y.label.iloc[0]);assert y.label_end_date.iloc[0]==cal[10]

@pytest.mark.parametrize('variant',['full_10_5','full_8_4','staged_10_5','staged_8_4'])
@pytest.mark.parametrize('rank,cooldown',[(50,3),(50,2),(65,3),(65,2)])
def test_grid_only_intended_changes(variant,rank,cooldown):
    a=make_account(variant,rank,cooldown);assert a.rules.rank_exit==rank and a.rules.top_k==10
    assert a.rules.profit==(.10 if variant.endswith('10_5') else .08)

def test_factor_decomposition_reconciles():
    d=decomposition(.1,.11,.13,.16)
    assert d['rank_only']+d['cooldown_only']+d['interaction']==pytest.approx(d['joint'])
    assert d['allocated_rank']+d['allocated_cooldown']==pytest.approx(d['joint'])
