import numpy as np
import pandas as pd
import pytest
from src.barra_risk import (standardize,fit_cross_section,covariance_forecast,
    specific_forecast,portfolio_risk,lagged_events,external_events)
from src.barra_risk import rolling_market_beta


def test_rolling_beta_divides_along_dates_not_stock_columns():
    rng=np.random.default_rng(4);dates=pd.bdate_range('2020-01-01',periods=160)
    market=pd.Series(rng.normal(size=len(dates))*.01,index=dates)
    returns=pd.DataFrame({'stock_a':market*2,'stock_b':market*.5},index=dates)
    beta=rolling_market_beta(returns,market,window=60,minimum=30)
    assert beta.columns.tolist()==['stock_a','stock_b']
    np.testing.assert_allclose(beta.iloc[-60:].stock_a,2,rtol=1e-10)
    np.testing.assert_allclose(beta.iloc[-60:].stock_b,.5,rtol=1e-10)


def fixture():
    rng=np.random.default_rng(42);n=100
    x=pd.DataFrame({'country':np.ones(n),'industry:a':np.r_[np.ones(40),np.zeros(60)],
        'industry:b':np.r_[np.zeros(40),np.ones(60)],'value':rng.normal(size=n)})
    beta=pd.Series([.01,.003,-.002,.004],index=x.columns)
    return x,beta,x@beta


def test_restricted_regression_recovers_identifiable_industry_returns():
    x,b,y=fixture();estimate,resid,audit=fit_cross_section(x,y)
    np.testing.assert_allclose(estimate,b,atol=1e-12)
    assert resid.abs().max()<1e-12 and audit['constraint_error']<1e-12


def test_stock_order_does_not_change_fit_and_no_missing_return_fill():
    x,b,y=fixture();y.iloc[0]=np.nan
    a,_,info=fit_cross_section(x,y);c,_,_=fit_cross_section(x.iloc[::-1],y)
    np.testing.assert_allclose(a,c,atol=1e-12);assert info['observations']==99


def test_covariance_is_psd_and_future_cannot_change_past_forecast():
    rng=np.random.default_rng(1);hist=pd.DataFrame(rng.normal(size=(200,4))*.01)
    a=covariance_forecast(hist.iloc[:150]);hist.iloc[150:]=100
    np.testing.assert_array_equal(a,covariance_forecast(hist.iloc[:150]))
    assert np.linalg.eigvalsh(a).min()>=-1e-12


def test_missing_specific_returns_use_conservative_fallback():
    rng=np.random.default_rng(1);f=pd.DataFrame(rng.normal(size=(130,2))*.01,columns=['a','b']);f['c']=np.nan
    s,fallback=specific_forecast(f)
    assert fallback['c'] and s['c']>0 and not fallback['a']


def test_euler_risk_contributions_reconcile_and_missing_holdings_block():
    x,_,_=fixture();w=pd.Series({0:.4,1:.3,80:.25});s=pd.Series(.0001,index=x.index)
    r=portfolio_risk(w,x,np.eye(4)*.0001,s)
    assert r['factor_variance_contribution'].sum()+r['stock_specific_contribution'].sum()==pytest.approx(r['daily_variance'])
    with pytest.raises(ValueError,match='Unmodeled'):portfolio_risk(pd.Series({999:.9}),x,np.eye(4),s)


def test_missing_exposure_and_constant_are_explicit():
    v=pd.Series([1.]*45+[np.nan]*5);z,a=standardize(v)
    assert z.eq(0).all() and a['constant'] and a['missing']==5


def test_disclosure_join_and_future_perturbation():
    d=pd.DataFrame({'instrument':['a','a'],'datetime':pd.to_datetime(['2025-04-01','2025-04-03'])})
    e=pd.DataFrame({'instrument':['a','a'],'available_date':pd.to_datetime(['2025-03-01','2025-04-02']),
        'report_date':pd.to_datetime(['2024-12-31','2025-03-31']),'value':[1.,2.]})
    assert lagged_events(d,e).value.tolist()==[1.,2.]
    e.loc[1,'value']=999.;assert lagged_events(d.iloc[:1],e).value.iloc[0]==1.


def test_current_industry_is_not_backfilled_into_history(tmp_path):
    p=tmp_path/'industry.parquet';pd.DataFrame({'instrument':['a'],'effective_date':['2020-01-01'],
        'available_date':['2026-10-26'],'industry':['bank'],'source_url':['https://example.com/source']}).to_parquet(p)
    e=external_events(p,'industry');assert e.usable_date.iloc[0]==pd.Timestamp('2026-10-26')
