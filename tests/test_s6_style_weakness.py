import numpy as np
import pandas as pd
import pytest
from paper_futu.research_s6_style_weakness import lagged_regimes,conditional_mean_hac,close_observation
from paper_futu.research_barra import save
from paper_futu.research_barra_full import hac_mean


def test_regime_uses_completed_previous_sessions_only():
    idx=pd.bdate_range('2020-01-01',periods=80)
    f=pd.DataFrame({'industry:SW:银行':-.001,'book_to_price':-.002,'earnings_yield':-.002},index=idx)
    before=lagged_regimes(f);f.iloc[50:]=999.;after=lagged_regimes(f)
    pd.testing.assert_frame_equal(before.loc[:idx[50]],after.loc[:idx[50]])
    assert not before.known.iloc[:20].any() and before.regime.iloc[20:].eq(0).all()


def test_conditional_hac_with_every_day_matches_unconditional():
    r=np.random.default_rng(42).normal(.0001,.01,300)
    a=conditional_mean_hac(r,np.ones(len(r),bool));b=hac_mean(r)
    assert a['annual_conditional_arithmetic_mean']==pytest.approx(b['annual_arithmetic_mean'])
    assert a['ci_low']==pytest.approx(b['ci_low'])


def test_empty_regime_is_not_reported_as_zero_return():
    with pytest.raises(ValueError):conditional_mean_hac(np.zeros(100),np.zeros(100,bool))


def test_overnight_observation_cannot_be_relabeled_as_previous_close(tmp_path):
    p=tmp_path/'nav.json';save(p,dict(time='2026-10-22T00:19:00+08:00',assets=100.,cash=5.,holdings=20))
    with pytest.raises(ValueError):close_observation(p,'2026-10-21')
    with pytest.raises(ValueError):close_observation(p,'2026-10-22')
