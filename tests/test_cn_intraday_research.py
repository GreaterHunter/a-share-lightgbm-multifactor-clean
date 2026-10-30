import numpy as np
import pandas as pd

from src.factor_library import ENHANCED_FEATURE_NAMES
from src.cn_intraday_features import FEATURE_NAMES, _grid
from scripts.normalize_cn_intraday_v2 import quarantine_volume_conflicts
from src.cn_intraday_source import normalize_public_bars


def test_conflicting_source_record_quarantines_whole_bucket():
    t=pd.date_range('2025-01-02 09:30',periods=11,freq='min')
    raw=pd.DataFrame(dict(timestamp=t,symbol='000001',exchange='SZ',open=10.,high=10.,low=10.,close=10.,volume=100,turnover=1000.))
    raw.loc[7,['volume','turnover']]=[0,.3]
    good,bad=quarantine_volume_conflicts(raw)
    assert len(bad)==1
    bars,_,audit=normalize_public_bars(good,'SZ000001','2025-01-01','2025-12-31')
    assert len(bars)==1 and audit['incomplete_5m_buckets']==1


def test_intraday_label_horizon_and_future_feature_invariance(tmp_path,monkeypatch):
    from paper_futu import research_cn_intraday as r
    cal=pd.bdate_range('2021-12-30',periods=17)
    days=cal[1:];grid=pd.DatetimeIndex(np.concatenate([_grid(d,5).values for d in days]),name='datetime')
    price=10+np.arange(len(grid))*.001
    bars=pd.DataFrame(dict(open=price,close=price+.001,high=price+.002,low=price-.001,volume=1000.,amount=price*1000),index=grid)
    (tmp_path/'bars_5m').mkdir();path=tmp_path/'bars_5m/SZ000001.parquet';bars.to_parquet(path)
    monkeypatch.setattr(r,'V2',tmp_path)
    monkeypatch.setattr(r,'TEST_START','2021-01-01')
    monkeypatch.setattr(r,'binary_series',lambda *args:pd.Series(1.,index=cal))
    daily=pd.DataFrame(np.repeat(np.arange(len(cal))[:,None],15,axis=1),index=cal,columns=ENHANCED_FEATURE_NAMES)
    a,p,m=r.feature_stock('SZ000001',daily,bars[['open','close']],cal,days,grid,[('2021-01-01','2025-12-31')])
    first=a.index[0];i=grid.get_loc(first)
    assert a.loc[first,'label_end']==grid[i+2+480]
    np.testing.assert_allclose(a.loc[first,'label'],price[i+2+480]/price[i+2]-1,rtol=1e-6)
    assert p.iloc[0][ENHANCED_FEATURE_NAMES].eq(0).all()
    cutoff=grid[400]
    changed=bars.copy();changed.loc[changed.index>cutoff,['open','close','high','low','amount']]*=2;changed.to_parquet(path)
    _,p2,_=r.feature_stock('SZ000001',daily,changed[['open','close']],cal,days,grid,[('2021-01-01','2025-12-31')])
    pd.testing.assert_frame_equal(p.loc[:cutoff,FEATURE_NAMES],p2.loc[:cutoff,FEATURE_NAMES])
