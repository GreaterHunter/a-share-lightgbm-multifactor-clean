"""Transparent Barra-style research primitives, independent of model/broker code.

This is not an implementation or licensed data copy of MSCI Barra/CNE5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import null_space


def rolling_market_beta(returns, market, window=252, minimum=126):
    """Broadcast the time-series denominator down dates, never across stocks."""
    return returns.rolling(window,min_periods=minimum).cov(market).div(
        market.rolling(window,min_periods=minimum).var(),axis=0)


def standardize(values, weights=None, minimum=30):
    """MAD winsorization, weighted mean zero, population variance one.

    Missing descriptors are neutral-imputed only after measuring coverage.
    A constant or insufficient descriptor remains zero with an explicit audit.
    """
    s=pd.Series(values,dtype=float).replace([np.inf,-np.inf],np.nan)
    good=s.notna(); info=dict(coverage=float(good.mean()),missing=int((~good).sum()),constant=False)
    if good.sum()<minimum:
        info['insufficient']=True
        return pd.Series(0.,index=s.index),info
    median=s[good].median(); mad=(s[good]-median).abs().median()
    scale=1.4826*mad
    if scale>1e-12: s=s.clip(median-5*scale,median+5*scale)
    w=pd.Series(1.,index=s.index) if weights is None else pd.Series(weights,index=s.index)
    if (~np.isfinite(w)|w.le(0)).any():raise ValueError('Invalid standardization weights')
    center=np.average(s[good],weights=w[good]); s=s.fillna(center)
    sd=np.sqrt(np.average((s-center)**2,weights=w))
    info['constant']=bool(sd<1e-12)
    return ((s-center)/sd if sd>=1e-12 else pd.Series(0.,index=s.index)),info


def fit_cross_section(exposures, returns, regression_weights=None, industry_mass=None):
    """Restricted WLS: cap/universe-weighted industry returns sum to zero.

    Matrix has country=1, all industry dummies and standardized style columns.
    Styles constant on this date are excluded, then returned as zero.
    Returns a full coefficient vector, never fills a missing realized return.
    """
    x=exposures.astype(float); y=pd.Series(returns,index=x.index,dtype=float)
    valid=np.isfinite(y)&np.isfinite(x).all(axis=1)
    x,y=x.loc[valid],y.loc[valid]
    if len(x)<max(40,len(x.columns)+10):raise ValueError('Insufficient regression cross-section')
    w=pd.Series(1.,index=x.index) if regression_weights is None else regression_weights.reindex(x.index)
    if (~np.isfinite(w)|w.le(0)).any():raise ValueError('Invalid regression weights')
    industry=[c for c in x if c.startswith('industry:') and x[c].abs().sum()>0]
    styles=[c for c in x if c!='country' and not c.startswith('industry:') and x[c].std()>1e-10]
    columns=['country',*industry,*styles]; a=x[columns].to_numpy()
    mass=x[industry].sum().to_numpy() if industry_mass is None else industry_mass.reindex(industry).to_numpy()
    if not np.isfinite(mass).all() or mass.sum()<=0:raise ValueError('Invalid industry constraint')
    constraint=np.r_[0.,mass/mass.sum(),np.zeros(len(styles))][None,:]
    basis=null_space(constraint)
    design=a@basis; root=np.sqrt(w.to_numpy()/w.mean())
    if np.linalg.matrix_rank(design*root[:,None])<design.shape[1]:raise ValueError('Unidentified risk factors')
    theta=np.linalg.lstsq(design*root[:,None],y.to_numpy()*root,rcond=None)[0]
    beta=pd.Series(0.,index=exposures.columns);beta.loc[columns]=basis@theta
    residual=y-x@beta
    total=np.sum(w*(y-np.average(y,weights=w))**2)
    audit=dict(observations=len(y),rank=design.shape[1],
        r_squared=float(1-np.sum(w*residual**2)/total) if total>0 else None,
        constraint_error=float(abs(constraint@beta.loc[columns].to_numpy()).item()))
    return beta,residual,audit


def covariance_forecast(history, half_life=63, window=252, minimum=126, shrink=.2):
    """Past-only EW covariance; diagonal shrinkage and PSD numerical floor."""
    a=np.asarray(history,dtype=float)[-window:]
    if len(a)<minimum or not np.isfinite(a).all():raise ValueError('Insufficient factor return history')
    w=2.**(-np.arange(len(a)-1,-1,-1)/half_life);w/=w.sum()
    z=a-np.sum(a*w[:,None],axis=0)
    cov=(z*w[:,None]).T@z/(1-np.sum(w*w))
    cov=(1-shrink)*cov+shrink*np.diag(np.diag(cov))
    val,vec=np.linalg.eigh((cov+cov.T)/2)
    return (vec*np.maximum(val,1e-12))@vec.T


def specific_forecast(history, half_life=42, window=126, minimum=60, shrink=.5):
    """EW residual variances shrunk toward the universe median.

    Missing residual observations do not become zero. Short histories receive
    twice the median variance and an explicit fallback flag.
    """
    frame=history.tail(window)
    w=2.**(-np.arange(len(frame)-1,-1,-1)/half_life)
    x=frame.to_numpy(float);good=np.isfinite(x);weights=w[:,None]*good
    count=good.sum(axis=0);mass=weights.sum(axis=0)
    mean=np.nansum(np.where(good,x,0)*weights,axis=0)/np.maximum(mass,1e-12)
    var=np.nansum(np.where(good,(x-mean)**2,0)*weights,axis=0)/np.maximum(mass,1e-12)
    adequate=(count>=minimum)&np.isfinite(var)&(var>0)
    if not adequate.any():raise ValueError('Insufficient specific return history')
    median=float(np.median(var[adequate]))
    result=np.where(adequate,(1-shrink)*var+shrink*median,2*median)
    return pd.Series(np.maximum(result,1e-10),index=frame.columns),pd.Series(~adequate,index=frame.columns)


def portfolio_risk(weights, exposures, covariance, specific, minimum_coverage=.98):
    weights=pd.Series(weights,dtype=float)
    if not np.isfinite(weights).all() or (weights<0).any() or weights.sum()>1.02:
        raise ValueError('Invalid long-only portfolio weights')
    available=weights.index.intersection(exposures.index).intersection(specific.index)
    covered=float(weights.loc[available].sum()/weights.sum()) if weights.sum()>0 else 1.
    if covered<minimum_coverage:raise ValueError(f'Unmodeled holdings: coverage={covered:.3%}')
    w=weights.reindex(available); x=exposures.loc[available]
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(specific.loc[available]).all():
        raise ValueError('Nonfinite portfolio risk inputs')
    b=w@x; marginal=np.asarray(covariance)@b.to_numpy()
    contributions=b.to_numpy()*marginal
    common=float(contributions.sum());individual=w*w*specific.loc[available]
    special=float(individual.sum());total=common+special
    return dict(annual_volatility=float(np.sqrt(total*252)),daily_variance=total,
        common_variance=common,specific_variance=special,coverage=covered,
        factor_exposure=b,factor_variance_contribution=pd.Series(contributions,index=x.columns),
        stock_specific_contribution=individual)


def lagged_events(daily, events, date_column='available_date'):
    """Point-in-time join, rejecting duplicate events and impossible dates."""
    if events.duplicated(['instrument',date_column]).any():raise ValueError('Duplicate events')
    if (events[date_column]<=events.report_date).any():raise ValueError('Non-causal disclosure')
    return pd.merge_asof(daily.sort_values(['datetime','instrument']),
        events.sort_values([date_column,'instrument']),left_on='datetime',right_on=date_column,
        by='instrument',direction='backward')


def external_events(path, kind):
    """Optional verified historical industry/capital data, with availability dates.

    A current snapshot must use its actual availability date; it is never
    backward-filled. Source URLs are mandatory, proof remains user auditable.
    """
    frame=pd.read_parquet(path)
    needed=['instrument','effective_date','available_date','source_url']
    needed+=['industry'] if kind=='industry' else ['market_cap']
    if set(needed)-set(frame):raise ValueError('Missing historical metadata fields')
    for c in ['effective_date','available_date']:frame[c]=pd.to_datetime(frame[c])
    if frame[needed].isna().any().any() or not frame.source_url.astype(str).str.startswith('https://').all():
        raise ValueError('Unverified historical metadata')
    frame['usable_date']=frame[['effective_date','available_date']].max(axis=1)
    if frame.duplicated(['instrument','usable_date']).any():raise ValueError('Duplicate usable dates')
    if kind=='capital' and (frame.market_cap<=0).any():raise ValueError('Invalid market capitalization')
    return frame
