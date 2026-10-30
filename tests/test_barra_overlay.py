import numpy as np
import pandas as pd
from src.barra_overlay import causal_risk, asset_covariance, adjusted_scores, optimize_weights, POLICY


def test_future_returns_do_not_change_snapshot():
    rng=np.random.default_rng(42);dates=pd.bdate_range('2020-01-01',periods=200)
    f=pd.DataFrame(rng.normal(0,.01,(200,3)),index=dates,columns=list('abc'))
    u=pd.DataFrame(rng.normal(0,.02,(200,4)),index=dates,columns=list('wxyz'))
    x=pd.DataFrame(rng.normal(size=(4,3)),index=list('wxyz'),columns=list('abc'));x['missing_descriptors']=0
    cutoff=dates[150];a=causal_risk(cutoff,f,u,x,list('abc'))
    f.loc[f.index>cutoff]=999;u.loc[u.index>cutoff]=-999
    b=causal_risk(cutoff,f,u,x,list('abc'))
    for left,right in zip(a,b):np.testing.assert_array_equal(left,right)


def test_covariance_is_positive_and_sums_to_factor_risk():
    x=np.array([[1,1],[1,-1.]])
    f=np.array([[.01,.002],[.002,.02]])
    d=np.array([.001,.003]);w=np.array([.5,.45])
    cov=asset_covariance(x,f,d)
    assert np.linalg.eigvalsh(cov).min()>0
    np.testing.assert_allclose(w@cov@w,((w@x)@f@(w@x)+sum(w*w*d))*252)


def test_optimizer_preserves_budget_bounds_score_and_reduces_objective():
    n=20;industry=np.eye(4)[np.arange(n)//5]
    cov=.04*industry@industry.T+.01*np.eye(n)
    ref=np.r_[np.repeat(.95*.4/6,6),np.repeat(.95*.6/14,14)]
    score=np.linspace(1,.8,n)
    w,a=optimize_weights(cov,industry,score,ref,ref)
    assert a['success'];assert abs(w.sum()-.95)<1e-8
    assert w.min()>=.01-1e-8 and w.max()<=.08+1e-8
    assert a['target_score']>=a['reference_score']-.0200001
    assert a['objective_after']<=a['objective_before']+1e-8


def test_infeasible_bounds_are_not_relaxed():
    w,a=optimize_weights(np.eye(2),np.eye(2),np.ones(2),np.array([.5,.45]),np.zeros(2))
    assert not a['success'] and a['reason']=='infeasible_bounds'
    np.testing.assert_array_equal(w,[.5,.45])


def test_locked_holding_cannot_be_cut():
    n=20;ref=np.full(n,.95/n);lo=np.full(n,.01);lo[0]=.07
    w,a=optimize_weights(np.eye(n),np.eye(n),np.ones(n),ref,ref,lower=lo)
    assert a['success'] and w[0]>=.07-1e-8


def test_risk_adjustment_penalizes_duplicate_industry():
    industry=np.array([[1,0],[1,0],[0,1.]])
    covariance=.04*industry@industry.T+.01*np.eye(3)
    s=adjusted_scores(np.ones(3),covariance,industry,np.array([.5,.45,0]))
    assert s[2]>s[0] and s[2]>s[1]
