"""Fixed, score-preserving long-only Barra overlay; no return-model fitting.

Weights are fractions of NAV. Scores are cross-sectional percentiles, NOT
expected returns. Objective coefficients are research policy, not fitted alpha.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

POLICY = dict(equity=.95, minimum_weight=.01, maximum_weight=.08,
              score_slack=.02, volatility_scale=.20, anchor_penalty=2.,
              industry_threshold=.30, industry_penalty=1., turnover_penalty=.02,
              selection_marginal_penalty=.10, selection_industry_penalty=.10,
              candidate_count=60)


def causal_risk(day, factor_returns, specific_returns, exposure, columns):
    """The explicitly sliced histories make future observations irrelevant."""
    from src.barra_risk import covariance_forecast, specific_forecast
    cov = covariance_forecast(factor_returns.loc[:day, columns])
    specific, _ = specific_forecast(specific_returns.loc[:day])
    specific = specific.reindex(exposure.index).fillna(specific.median()*2)
    specific *= 1+.5*exposure.missing_descriptors.gt(0)
    x = exposure[columns].to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(specific).all():
        raise ValueError('Nonfinite risk snapshot')
    return x, cov, specific.to_numpy(float)


def asset_covariance(x, f, specific):
    result = (x@f@x.T + np.diag(specific))*252
    return (result+result.T)/2


def adjusted_scores(percentiles, cov, industry, reference, policy=POLICY):
    """Deterministic selection heuristic, not a joint cardinality optimum."""
    import pandas as pd
    marginal = cov@reference/np.sqrt(max(reference@cov@reference, 1e-14))
    rank = pd.Series(marginal).rank(method='average', pct=True).to_numpy()
    mass = industry.T@reference
    concentration = industry@np.clip((mass-policy['industry_threshold'])/
                                      policy['industry_threshold'], 0, 1)
    return (percentiles-policy['selection_marginal_penalty']*rank
            -policy['selection_industry_penalty']*concentration)


def optimize_weights(cov, industry, scores, reference, current, friction=0.,
                     lower=None, upper=None, policy=POLICY):
    """Convex objective with score floor, weight bounds and frozen budget.

    On infeasible/failed numerical solve return explicit failure. The caller
    must log its fallback; there is no hidden relaxation of constraints.
    """
    n = len(reference)
    reference, current, scores = [np.asarray(v, float) for v in (reference,current,scores)]
    lo = np.full(n,policy['minimum_weight']) if lower is None else np.asarray(lower,float)
    hi = np.full(n,policy['maximum_weight']) if upper is None else np.asarray(upper,float)
    total = float(reference.sum())
    floor = float(scores@reference-policy['score_slack']*total)
    if lo.sum()>total+1e-8 or hi.sum()<total-1e-8 or np.any(lo>hi):
        return reference.copy(), dict(success=False,reason='infeasible_bounds')
    smooth=1e-6
    def objective(w):
        variance = w@cov@w/policy['volatility_scale']**2
        anchor = policy['anchor_penalty']*np.sum((w-reference)**2)/total**2
        excess=np.maximum(industry.T@w-policy['industry_threshold'],0)
        sector=policy['industry_penalty']*np.sum(excess**2)/policy['industry_threshold']**2
        # Fixed dimensionless preference for low turnover. Higher spread stresses
        # change account fills/fees, not the frozen construction policy.
        turnover=policy['turnover_penalty']*np.sqrt((w-current)**2+smooth**2).sum()/total
        return variance+anchor+sector+turnover
    def gradient(w):
        excess=np.maximum(industry.T@w-policy['industry_threshold'],0)
        return (2*cov@w/policy['volatility_scale']**2+
                2*policy['anchor_penalty']*(w-reference)/total**2+
                2*policy['industry_penalty']*industry@excess/policy['industry_threshold']**2+
                policy['turnover_penalty']*(w-current)/np.sqrt((w-current)**2+smooth**2)/total)
    constraints=[dict(type='eq',fun=lambda w:w.sum()-total,jac=lambda w:np.ones(n)),
                 dict(type='ineq',fun=lambda w:scores@w-floor,jac=lambda w:scores)]
    result=minimize(objective,np.clip(reference,lo,hi),jac=gradient,method='SLSQP',
                    bounds=list(zip(lo,hi)),constraints=constraints,
                    options=dict(maxiter=150,ftol=1e-10))
    w=result.x
    valid=(result.success and np.isfinite(w).all() and abs(w.sum()-total)<1e-7
           and scores@w>=floor-1e-7 and (w>=lo-1e-7).all() and (w<=hi+1e-7).all())
    if not valid:
        return reference.copy(),dict(success=False,reason=str(result.message),iterations=int(result.nit))
    return w,dict(success=True,reason='optimized',iterations=int(result.nit),
        reference_volatility=float(np.sqrt(reference@cov@reference)),
        target_volatility=float(np.sqrt(w@cov@w)),
        reference_score=float(scores@reference/total),target_score=float(scores@w/total),
        reference_max_industry=float((industry.T@reference).max()),
        target_max_industry=float((industry.T@w).max()),
        objective_before=float(objective(reference)),objective_after=float(objective(w)))
