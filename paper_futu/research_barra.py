"""Independent S6 Barra-style risk research; never imports broker entry points.

python -m paper_futu.research_barra --stage all
Optional PIT industry/capital data: --industry-events / --capital-events.
Use --require-full to reject the explicitly incomplete local-data model.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np
import pandas as pd

from src.barra_risk import (standardize,fit_cross_section,covariance_forecast,
    specific_forecast,portfolio_risk,lagged_events,external_events,rolling_market_beta)

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data/processed/final_ablation_asof_20261013'
ACCOUNT=ROOT/'reports/experiments/s6_quarterly_l1l2_small_asof_20261013'
DEFAULT_OUT=ROOT/'reports/experiments/barra_s6_v1_asof_20261013'
START,END='2022-01-04','2026-10-13'
RF=float(np.expm1(np.log1p(.01)/252))
STYLES=['beta','momentum','residual_volatility','book_to_price','earnings_yield',
    'liquidity_proxy','cashflow_quality','growth','leverage','profitability']
LABELS=dict(country='市场共同因子',beta='Beta风格',momentum='中期动量',
    residual_volatility='残余波动',book_to_price='账面估值',earnings_yield='盈利收益率',
    liquidity_proxy='流动性代理',cashflow_quality='现金流质量',growth='营收增长',
    leverage='财务杠杆',profitability='ROE盈利能力',size='市值规模',nonlinear_size='非线性规模',
    specific='个股残差',execution_timing='交易时点与仓位变化',fees='交易费用',stock_rf='股票无风险基准部分')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str),encoding='utf-8');tmp.replace(path)


def verify_sources(out):
    p=read(out/'protocol.json')
    for name,expected in {**p['sources'],**p['implementation']}.items():
        if sha(name)!=expected:raise ValueError('Immutable input changed: '+name)
    return p


def stage_done(out,name):
    marker=out/f'{name}_complete.json'
    if not marker.exists():return False
    for path,digest in read(marker)['artifacts'].items():
        if sha(path)!=digest:raise ValueError('Output changed: '+path)
    return True


def complete(out,name,files,**details):
    save(out/f'{name}_complete.json',dict(status='complete',artifacts={str(p):sha(p) for p in files},**details))
    print('COMPLETE',name,flush=True)


def audit(out,industry=None,capital=None,require_full=False):
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(DATA/'manifest.json');sources={str(DATA/'manifest.json'):sha(DATA/'manifest.json')}
    for n,h in manifest['artifacts'].items():
        if n.startswith('provider/') or n in ['raw_features.parquet','financial_events.parquet']:
            sources[str(DATA/n)]=h
    for phase in range(5):
        task=read(ACCOUNT/f'tasks/account_history_QREG_{phase}.json')
        assert task['status']=='complete'
        sources[str(ACCOUNT/f'tasks/account_history_QREG_{phase}.json')]=sha(ACCOUNT/f'tasks/account_history_QREG_{phase}.json')
        sources.update(task['artifacts'])
    for name in ['configs/futu_cn_paper.json','models/paper_futu/s6_qreg_20260930/model.txt',
                 'models/paper_futu/s6_qreg_20260930/metadata.json']:
        sources[str(ROOT/name)]=sha(ROOT/name)
    for path in [industry,capital]:
        if path:sources[str(Path(path).resolve())]=sha(path)
    # Explicitly test adapter contracts before doing any expensive operation.
    if industry:external_events(industry,'industry')
    if capital:external_events(capital,'capital')
    if require_full and not (industry and capital):raise ValueError('Full risk model requires verified PIT industry and capital events')
    for name,h in sources.items():
        if sha(name)!=h:raise ValueError('Frozen source changed: '+name)
    p=dict(name='S6 Barra-style research v1',licensed_barra=False,snapshot_id=manifest['snapshot_id'],
        account='S6_QREG_Top20_head6_40_R5_drop6',start=START,end=END,estimation_start='2020-01-01',
        styles=STYLES,industry_events=str(Path(industry).resolve()) if industry else None,
        capital_events=str(Path(capital).resolve()) if capital else None,require_full=require_full,
        risk_free_annual=.01,regression='prior-session exposures, current close-to-close excess return; constrained WLS',
        weighting='equal weight absent verified market cap; sqrt-cap WLS if present',
        covariance=dict(window=252,half_life=63,minimum=126,diagonal_shrinkage=.2),
        specific_risk=dict(window=126,half_life=42,minimum=60,median_shrinkage=.5,short_history_multiplier=2),
        missing='neutral descriptor imputation with explicit coverage, 1.5x specific variance for stocks with missing descriptors',
        industry_policy='No current classifications or bank labels backfilled. One UNCLASSIFIED group if no PIT history.',
        capital_policy='No share-price or turnover surrogate mislabeled as market capitalization. Size omitted if unavailable.',
        sources=sources,implementation={str(Path(__file__)):sha(__file__),str(ROOT/'src/barra_risk.py'):sha(ROOT/'src/barra_risk.py')},
        limitations=['CSI300 estimation universe, not all A shares','Historical interval already researched',
            'Financial values are not a full as-reported revision archive','Specific residual is not proof of alpha',
            'Static model parameters are not tuned to S6 returns','Diagonal specific risk ignores residual stock correlations',
            'No commercial MSCI adjustments, volatility regime adjustment or optimization bias adjustment'])
    path=out/'protocol.json'
    if path.exists() and read(path)!=p:raise ValueError('Protocol differs; use a new --output directory')
    save(path,p);verify_sources(out);print('AUDIT passed',flush=True)


def provider_field(field,stocks,dates,full_calendar):
    a=np.full((len(dates),len(stocks)),np.nan,dtype=np.float32)
    locations=full_calendar.get_indexer(dates)
    for j,stock in enumerate(stocks):
        path=DATA/f'provider/features/{stock.lower()}/{field}.day.bin'
        if not path.exists():continue
        values=np.fromfile(path,dtype='<f4');start=int(values[0]);idx=locations-start
        good=(idx>=0)&(idx<len(values)-1);a[good,j]=values[1:][idx[good]]
    return pd.DataFrame(a,index=dates,columns=stocks)


def prepare(out):
    p=verify_sources(out)
    if stage_done(out,'data'):return
    from src.runtime_budget import require_memory
    require_memory(.8,'Barra sequential data preparation')
    raw=pd.read_parquet(DATA/'raw_features.parquet',filters=[('datetime','>=',pd.Timestamp('2018-01-01'))])
    raw=raw.swaplevel().sort_index() if raw.index.names[0]!='datetime' else raw.sort_index()
    cal=pd.DatetimeIndex(pd.to_datetime((DATA/'provider/calendars/day.txt').read_text().splitlines()))
    dates=cal[(cal>='2018-01-01')&(cal<=END)];stocks=sorted(raw.index.get_level_values('instrument').unique())
    close=provider_field('close',stocks,dates,cal);volume=provider_field('volume',stocks,dates,cal)
    returns=close/close.shift(1)-1
    returns=returns.replace([np.inf,-np.inf],np.nan)
    # Same-day dynamic CSI300 membership; estimation only uses previous-session members.
    pool=pd.Series(1.,index=raw.index).unstack('instrument').reindex(index=dates,columns=stocks).notna()
    market=returns.where(pool.shift(1).eq(True)).mean(axis=1)
    beta=rolling_market_beta(returns,market)
    residual=returns.sub(beta.mul(market,axis=0))
    momentum=np.log1p(returns.where(returns.gt(-1))).shift(20).rolling(126,min_periods=100).sum()
    residvol=residual.rolling(60,min_periods=40).std()
    amount=close*volume*100 # Qlib anchored price * adjusted hand volume cancels adjustment.
    liquidity=-np.log((returns.abs()/amount.where(amount>0)).rolling(20,min_periods=15).mean().clip(lower=1e-20))
    panel=raw[['book_to_price','earnings_to_price','operating_cashflow_to_revenue']].copy()
    for name,wide in [('beta',beta),('momentum',momentum),('residual_volatility',residvol),('liquidity_proxy',liquidity)]:
        wide.index.name='datetime';wide.columns.name='instrument';panel[name]=wide.stack(dropna=False).reindex(panel.index)
    events=pd.read_parquet(DATA/'financial_events.parquet')
    # Later report wins if reports are released on the same available date.
    events=events.sort_values(['available_date','report_date']).drop_duplicates(['instrument','available_date'],keep='last')
    fields=['instrument','report_date','available_date','roe','debt_to_assets','revenue_growth']
    general=lagged_events(panel.reset_index()[['instrument','datetime']],events[fields])
    general=general.set_index(['datetime','instrument']).reindex(panel.index)
    panel=panel.rename(columns={'earnings_to_price':'earnings_yield','operating_cashflow_to_revenue':'cashflow_quality'})
    panel['profitability']=general.roe;panel['leverage']=general.debt_to_assets;panel['growth']=general.revenue_growth
    panel['industry']='UNCLASSIFIED';panel['capital']=np.nan
    for kind,path in [('industry',p['industry_events']),('capital',p['capital_events'])]:
        if not path:continue
        e=external_events(path,kind);field='industry' if kind=='industry' else 'market_cap'
        joined=pd.merge_asof(panel.reset_index()[['datetime','instrument']].sort_values(['datetime','instrument']),
            e[['instrument','usable_date',field]].sort_values(['usable_date','instrument']),
            left_on='datetime',right_on='usable_date',by='instrument',direction='backward').set_index(['datetime','instrument'])
        if kind=='industry':panel['industry']=joined.industry.reindex(panel.index).fillna('UNCLASSIFIED')
        else:
            # Capitalization is a daily observation: stale values are never held indefinitely.
            fresh=(joined.index.get_level_values('datetime')-joined.usable_date).dt.days.le(7)
            panel['capital']=joined.market_cap.where(fresh).reindex(panel.index)
    styles=list(STYLES)
    if p['capital_events']:
        styles+=['size','nonlinear_size'];panel['size']=np.log(panel.capital.where(panel.capital>0))
        panel['nonlinear_size']=panel['size']**3
    panel=panel.loc[panel.index.get_level_values('datetime')>='2019-12-01']
    tested=panel.loc[panel.index.get_level_values('datetime')>=START]
    for required in ['beta','residual_volatility','momentum','liquidity_proxy']:
        if tested[required].notna().mean()<.8:
            raise ValueError('Critical descriptor coverage below 80%: '+required)
    industry_names=sorted(panel.industry.unique());columns=['country',*[f'industry:{n}' for n in industry_names],*styles]
    parts=[];coverage=[]
    for day,g in panel.groupby(level='datetime',sort=True):
        g=g.droplevel('datetime');caps=g.capital
        have_cap=caps.gt(0)&np.isfinite(caps)
        # Partial cap coverage must not silently exclude stocks or mix incompatible weights.
        w=caps if have_cap.all() else None
        x=pd.DataFrame(0.,index=g.index,columns=columns);x['country']=1.
        for n in industry_names:x[f'industry:{n}']=g.industry.eq(n).astype(float)
        missing=pd.Series(0,index=g.index)
        for name in styles:
            z,info=standardize(g[name],w);x[name]=z;missing+=g[name].isna()
            coverage.append(dict(datetime=day,descriptor=name,**info))
        if 'nonlinear_size' in styles:
            b=np.linalg.lstsq(np.column_stack([np.ones(len(x)),x['size']]),x.nonlinear_size,rcond=None)[0]
            x['nonlinear_size'],_=standardize(x.nonlinear_size-b[0]-b[1]*x['size'],w)
        x['missing_descriptors']=missing;x['capital']=caps;x['datetime']=day
        parts.append(x.reset_index())
    exposures=pd.concat(parts,ignore_index=True).set_index(['datetime','instrument']).sort_index()
    exposures.to_parquet(out/'exposures.parquet');returns.to_parquet(out/'asset_returns.parquet')
    pd.DataFrame(coverage).to_csv(out/'descriptor_coverage.csv',index=False)
    profile=dict(styles=styles,factor_columns=columns,industry_groups=industry_names,
        industry_observation_coverage=float(panel.industry.ne('UNCLASSIFIED').mean()),
        capital_observation_coverage=float(panel.capital.gt(0).mean()),
        full_model_eligible=bool(panel.industry.ne('UNCLASSIFIED').mean()>=.98 and panel.capital.gt(0).mean()>=.98),
        missing_financial_revision_archive=True,liquidity='negative log 20-day mean abs return / (adjusted close * adjusted hand-volume * 100); approximate value traded',
        historical_sector_and_size_backfill=False)
    save(out/'data_profile.json',profile)
    if p['require_full'] and not profile['full_model_eligible']:raise ValueError('Historical industry/capital coverage below 98%')
    complete(out,'data',[out/n for n in ['exposures.parquet','asset_returns.parquet','descriptor_coverage.csv','data_profile.json']])


def estimate(out):
    verify_sources(out)
    if stage_done(out,'factors'):return
    assert stage_done(out,'data')
    profile=read(out/'data_profile.json');cols=profile['factor_columns']
    e=pd.read_parquet(out/'exposures.parquet');r=pd.read_parquet(out/'asset_returns.parquet')
    dates=pd.DatetimeIndex(sorted(e.index.get_level_values('datetime').unique()));factor_rows=[];residuals=[];audits=[]
    for i in range(len(dates)-1):
        signal,day=dates[i],dates[i+1]
        if day<pd.Timestamp('2020-01-01'):continue
        x=e.xs(signal);y=r.loc[day].reindex(x.index)-RF
        caps=x.capital;w=np.sqrt(caps) if caps.notna().all() and caps.gt(0).all() else None
        indcols=[n for n in cols if n.startswith('industry:')]
        mass=x[indcols].mul(caps if w is not None else 1.,axis=0).sum()
        try:b,u,a=fit_cross_section(x[cols],y,w,mass)
        except ValueError as error:raise ValueError(f'{day.date()} regression failed: {error}') from error
        b.name=day;factor_rows.append(b)
        # Also compute residuals for stocks outside today's estimation membership
        # only when an actually observed exposure is available (no backward fill).
        u.name=day;residuals.append(u);audits.append(dict(datetime=day,exposure_date=signal,cap_weighted=w is not None,**a))
    f=pd.DataFrame(factor_rows);f.index.name='datetime';u=pd.DataFrame(residuals);u.index.name='datetime'
    f.to_parquet(out/'factor_returns.parquet');u.to_parquet(out/'specific_returns.parquet')
    pd.DataFrame(audits).to_csv(out/'regression_audit.csv',index=False)
    assert (pd.to_datetime(pd.DataFrame(audits).exposure_date)<pd.to_datetime(pd.DataFrame(audits).datetime)).all()
    complete(out,'factors',[out/n for n in ['factor_returns.parquet','specific_returns.parquet','regression_audit.csv']],days=len(f))


def analyze(out):
    verify_sources(out)
    if stage_done(out,'analysis'):return
    assert stage_done(out,'factors')
    profile=read(out/'data_profile.json');cols=profile['factor_columns']
    exposures=pd.read_parquet(out/'exposures.parquet');f=pd.read_parquet(out/'factor_returns.parquet')
    u=pd.read_parquet(out/'specific_returns.parquet');asset=pd.read_parquet(out/'asset_returns.parquet')
    # Risk for close t uses factor/residual observations through t, predicts t+1.
    caches={};exposure_dates=pd.DatetimeIndex(sorted(exposures.index.get_level_values('datetime').unique()))
    analysis_dates=f.index[f.index>=START]
    for number,day in enumerate(analysis_dates):
        hist=f.loc[:day]
        if len(hist)<126:continue
        cov=covariance_forecast(hist)
        specific,short=specific_forecast(u.loc[:day])
        x=exposures.xs(day);specific=specific.reindex(x.index).fillna(specific.median()*2)
        specific=specific*(1+.5*x.missing_descriptors.gt(0))
        caches[day]=(x,cov,specific)
        if number%250==0:print('Risk dates prepared',number,flush=True)
    forecasts=[];attribution=[];risk_parts=[];latest=[];stock_risks=[];summary=[];daily_exposure=[]
    for phase in range(5):
        d=ACCOUNT/f'accounts/history_QREG_{phase}';account=pd.read_parquet(d/'daily.parquet')
        h=pd.read_parquet(d/'holdings.parquet');hold={day:g.set_index('instrument') for day,g in h.groupby('datetime')}
        components=[]
        for index,(day,row) in enumerate(account.iterrows()):
            previous_day=exposure_dates[exposure_dates<day][-1]
            previous=hold.get(account.index[index-1]) if index else None
            prior_nav=float(account.account.iloc[index-1]) if index else 1e6
            w=previous.market_value/prior_nav if previous is not None else pd.Series(dtype=float)
            part=pd.Series(0.,index=[*cols,'stock_rf','specific','execution_timing','fees'])
            if len(w):
                x=exposures.xs(previous_day);known=w.index.intersection(x.index)
                b=w.reindex(known)@x.loc[known,cols]
                part.loc[cols]=b*f.loc[day]
                rr=asset.loc[day].reindex(known)
                # Missing realized returns remain part of the explicit unexplained/timing bucket.
                valid=rr.notna();known2=known[valid.to_numpy()]
                observed_resid=rr.loc[known2]-RF-x.loc[known2,cols]@f.loc[day]
                part['specific']=float((w.loc[known2]*observed_resid).sum())
                part['stock_rf']=float(w.loc[known2].sum()*RF)
            part['fees']=-float(row.fees_cash)/prior_nav
            part['execution_timing']=float(row.net_return)-part.sum()
            if abs(part.sum()-row.net_return)>1e-11:raise ValueError('Attribution does not reconcile')
            part.name=day;components.append(part)
            current=hold.get(day)
            if current is None or day not in caches:continue
            x,cov,specific=caches[day];weights=current.market_value/float(row.account)
            try:risk=portfolio_risk(weights,x[cols],cov,specific)
            except ValueError as error:
                forecasts.append(dict(offset=phase,datetime=day,valid=False,reason=str(error)));continue
            next_days=asset.index[asset.index>day]
            realized=None
            if len(next_days):
                nextret=asset.loc[next_days[0]].reindex(weights.index)
                if nextret.notna().all():realized=float(weights@nextret)
            forecasts.append(dict(offset=phase,datetime=day,valid=True,predicted_annual_vol=risk['annual_volatility'],
                daily_variance=risk['daily_variance'],common_fraction=risk['common_variance']/risk['daily_variance'],
                specific_fraction=risk['specific_variance']/risk['daily_variance'],coverage=risk['coverage'],
                next_fixed_holdings_return=realized,missing_descriptor_weight=float(weights.reindex(x.index,fill_value=0).loc[x.missing_descriptors.gt(0)].sum())))
            daily_exposure.append(dict(offset=phase,datetime=day,**risk['factor_exposure'].to_dict()))
            if day==pd.Timestamp(END):
                latest.append(dict(offset=phase,annual_volatility=risk['annual_volatility'],
                    common_fraction=risk['common_variance']/risk['daily_variance'],specific_fraction=risk['specific_variance']/risk['daily_variance'],
                    holdings=len(weights),missing_descriptor_weight=forecasts[-1]['missing_descriptor_weight']))
                for n,value in risk['factor_exposure'].items():risk_parts.append(dict(offset=phase,factor=n,exposure=value,
                    variance_contribution=float(risk['factor_variance_contribution'][n]),risk_share=float(risk['factor_variance_contribution'][n]/risk['daily_variance'])))
                for stock,value in risk['stock_specific_contribution'].items():stock_risks.append(dict(offset=phase,instrument=stock,
                    account_weight=float(weights[stock]),specific_risk_share=float(value/risk['daily_variance'])))
        comp=pd.DataFrame(components); comp.index.name='datetime'
        assert np.allclose(comp.sum(axis=1),account.net_return,atol=1e-11)
        for period,part in [('all',comp),*[(str(y),g) for y,g in comp.groupby(comp.index.year)]]:
            net=account.net_return.reindex(part.index);wealth=(1+net).cumprod().shift(1,fill_value=1.)
            linked=part.mul(wealth,axis=0).sum()
            total=float((1+net).prod()-1)
            assert abs(linked.sum()-total)<1e-10
            for name,value in linked.items():attribution.append(dict(offset=phase,period=period,factor=name,linked_contribution=value))
        comp['offset']=phase;comp.reset_index().to_parquet(out/f'attribution_daily_{phase}.parquet',index=False)
        summary.append(dict(offset=phase,total_return=float((1+account.net_return).prod()-1),attribution_reconciled=True))
    pd.DataFrame(forecasts).to_csv(out/'risk_forecasts.csv',index=False)
    pd.DataFrame(attribution).to_csv(out/'return_attribution.csv',index=False)
    pd.DataFrame(risk_parts).to_csv(out/'latest_factor_risk.csv',index=False)
    pd.DataFrame(stock_risks).to_csv(out/'latest_stock_specific_risk.csv',index=False)
    pd.DataFrame(daily_exposure).to_parquet(out/'portfolio_exposures.parquet',index=False)
    # Snapshot ingredients support a standalone downstream risk query.
    x,cov,specific=caches[pd.Timestamp(END)]
    pd.DataFrame(cov,index=cols,columns=cols).to_csv(out/'latest_factor_covariance_daily.csv')
    x.to_parquet(out/'latest_asset_exposures.parquet');specific.rename('specific_variance_daily').to_csv(out/'latest_specific_risk.csv')
    forecast=pd.DataFrame(forecasts);valid=forecast.loc[forecast.valid.eq(True)].copy()
    calibration=[]
    for phase,g in valid.groupby('offset'):
        g=g.dropna(subset=['next_fixed_holdings_return']);z=g.next_fixed_holdings_return/np.sqrt(g.daily_variance)
        calibration.append(dict(offset=int(phase),days=len(z),standardized_mean=float(z.mean()),
            standardized_rms=float(np.sqrt(np.mean(z*z))),outside_2sigma=float(z.abs().gt(2).mean())))
    result=dict(status='complete',latest=latest,calibration=calibration,accounts=summary,
        coverage=dict(valid_forecasts=len(valid),attempted=len(forecast),rejected=int((~forecast.valid).sum())),
        factor_columns=cols,full_model_eligible=profile['full_model_eligible'],
        caution='Specific returns and execution timing are not pure alpha; historical descriptive research only')
    save(out/'analysis.json',result)
    files=[out/n for n in ['risk_forecasts.csv','return_attribution.csv','latest_factor_risk.csv',
        'latest_stock_specific_risk.csv','portfolio_exposures.parquet','latest_factor_covariance_daily.csv',
        'latest_asset_exposures.parquet','latest_specific_risk.csv','analysis.json']]
    files += list(out.glob('attribution_daily_*.parquet'))
    complete(out,'analysis',files)


def report(out):
    p=verify_sources(out);assert stage_done(out,'analysis')
    profile=read(out/'data_profile.json');a=read(out/'analysis.json')
    latest=pd.DataFrame(a['latest']);risk=pd.read_csv(out/'latest_factor_risk.csv')
    attr=pd.read_csv(out/'return_attribution.csv');forecast=pd.read_csv(out/'risk_forecasts.csv')
    grouped=attr.groupby(['period','factor']).linked_contribution.mean().unstack('factor')
    def table(headers,rows):return '<table><tr>'+''.join(f'<th>{html.escape(str(v))}</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</table>'
    def pct(x):return f'{x:.2%}'
    exposure=risk.groupby('factor')[['exposure','risk_share']].mean()
    riskrows=[[LABELS.get(n,n),f'{r.exposure:+.3f}',pct(r.risk_share)] for n,r in exposure.iterrows() if n!='industry:UNCLASSIFIED']
    attrrows=[]
    for period,g in grouped.iterrows():
        attrrows.append([period,*[pct(g.get(c,0.)) for c in ['country']],
            pct(g.reindex(STYLES).sum()),pct(g.filter(like='industry:').sum()),pct(g.specific),
            pct(g.execution_timing),pct(g.fees),pct(g.stock_rf),pct(g.sum())])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained')
    selected=exposure.drop(index=['country','industry:UNCLASSIFIED'],errors='ignore').exposure
    axes[0].barh(selected.index,selected.values,color=np.where(selected>=0,'#176da5','#c15d43'))
    axes[0].axvline(0,color='gray',lw=.8);axes[0].set_title('S6 style exposures, five phases mean, 2026-10-13')
    valid=forecast.loc[forecast.valid.eq(True)].copy();valid.datetime=pd.to_datetime(valid.datetime)
    v=valid.groupby('datetime').predicted_annual_vol.mean()
    axes[1].plot(v.index,v,label='Forecast annual volatility');axes[1].set_ylim(bottom=0);axes[1].legend();axes[1].grid(alpha=.2)
    fig.savefig(out/'risk_overview.png',dpi=150);plt.close(fig)
    limitations='历史行业、市值均缺失；当前是市场＋10项风格的风险模型，没有行业、规模风险覆盖。' if not profile['full_model_eligible'] else '行业、市值按历史可得日期对齐，覆盖率见数据报告。'
    overview=dict(as_of=END,mean_forecast_annual_volatility=float(latest.annual_volatility.mean()) if len(latest) else None,
        common_risk_fraction=float(latest.common_fraction.mean()) if len(latest) else None,
        specific_risk_fraction=float(latest.specific_fraction.mean()) if len(latest) else None,
        missing_descriptor_weight=float(latest.missing_descriptor_weight.mean()) if len(latest) else None,
        factor_exposures=exposure.exposure.to_dict(),mean_calibration_rms=float(np.mean([x['standardized_rms'] for x in a['calibration']])),
        full_model_eligible=profile['full_model_eligible'],limitations=limitations)
    save(out/'summary.json',overview)
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6 Barra风格风险研究</title>
<style>body{{max-width:1200px;margin:32px auto;padding:0 22px;font:16px/1.7 system-ui;color:#163449}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #ccd9e3;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#ecf3f8}}.note{{padding:15px;background:#fff2d8}}img{{max-width:100%}}</style>
<h1>S6 Barra风格风险模型：独立研究版</h1><p class="note">{limitations} 本模型自行实现，非MSCI商业Barra、非完整CNE5复刻。当前结果不能解释为已剔除所有行业、规模影响后的Alpha。</p>
<p>冻结数据截至{END}；S6季度QREG、Top20、前6只40%、每5日调仓、最多替换6只，五条预设起点账户。风险研究不修改预测模型、模拟账户或下单调度。2022—2026为已经反复研究的历史。</p>
<h2>模型结构</h2><p>用t−1日风格暴露解释t日个股收盘到收盘超额收益。MAD五倍去极值、横截面标准化；财务按披露次日起可用。回归估计市场、行业与风格收益，行业收益加权和约束为零；没有市值时采用等权损失。样本仅限当时沪深300股票池。缺失描述符中性填充并计数，相应个股特异方差提高50%，不会把缺失收益补零。</p>
<p>共同因子协方差：过去252日、半衰期63日、20%向对角矩阵收缩；个股残差方差：过去126日、半衰期42日、50%向横截面中位数收缩，短历史为中位数两倍。协方差保证半正定。所有风险预测仅使用截至预测当日收盘的信息。未实现商业Barra的状态调整及优化偏差修正。</p>
<h2>截至{END}的持仓风险</h2><p>五账户平均预测年化波动率{pct(overview['mean_forecast_annual_volatility']) if len(latest) else '不可计算'}；共同风险占比{pct(overview['common_risk_fraction']) if len(latest) else '不可计算'}，特异风险占比{pct(overview['specific_risk_fraction']) if len(latest) else '不可计算'}。这是年化波动率预测，不是收益预测或最大回撤保证，也不是10月26日实时风险。</p>
{table(['因子','账户暴露/标准差单位（市场为股票仓位）','总方差贡献比例'],riskrows)}
<p>因子Euler贡献可能为负，表示与其他因子的协方差降低总风险，不能逐项解释成独立波动率。股票特异风险与因子贡献合计核对总方差。相对基准主动风险尚未提供：缺少历史指数成分权重，不能用等权池伪称沪深300权重。</p>
<img src="risk_overview.png" alt="S6 risk overview">
<h2>已实现收益的复利衔接归因</h2>{table(['年份','市场','风格合计','行业','个股残差','交易时点与仓位变化','费用','股票无风险部分','账户净收益'],attrrows)}
<p>各项是归属于期末累计收益的百分点贡献：逐日乘此前账户净值后累计，合计严格等于账户净收益。先逐账户计算再取五起点均值。上一日收盘持仓用于风险归因；实际开盘换仓造成的差异、建仓日、未覆盖股票和缺失收益进入明确的交易时点桶。该桶及个股残差都不叫纯Alpha。2026仅截至10月13日。</p>
<h2>预测风险校准</h2>{table(['起点','次日固定持仓观测数','标准化收益RMS','超出±2σ比例'],[[x['offset'],x['days'],f"{x['standardized_rms']:.3f}",pct(x['outside_2sigma'])] for x in a['calibration']])}
<p>固定当日收盘持仓，比较次日实际个股收益与预测风险；不把次日交易当成固定组合。RMS接近1只是校准参考，不是有效性证明，也不是独立测试。总计{a['coverage']['valid_forecasts']}条有效风险快照、{a['coverage']['rejected']}条因持仓覆盖等原因拒绝计算。持仓缺失超过2%时不强行给出风险。</p>
<h2>数据与边界</h2><p>金融指标原始值没有完整历史修订档案；流动性为日线成交额近似代理；残差相关性忽略可能低估共同遗漏风险。风格为本项目明确定义的代理，不等于商业Barra标准。缺少行业和规模时，共同/特异风险分解也会受遗漏风险影响。所有源文件、账户、模型及部署哈希均复核，封存资料不改。</p>
<p>行业导入需instrument、industry、effective_date、available_date、source_url；市值导入需instrument、market_cap、effective_date、available_date、source_url。可用日期取生效和已知日期的较晚者，市值超过7个自然日不沿用。完整模式要求行业和市值均达到98%覆盖；当前模式不会回填今天的分类到历史。</p>
<p>方法参考：<a href="https://www.msci.com/documents/10199/2935796a-0a80-4050-934a-12966d1e2518">MSCI Barra China Equity Model</a>。协议与源文件哈希见protocol.json，数据覆盖见data_profile.json。</p></html>'''
    (out/'report.html').write_text(text,encoding='utf-8')
    complete(out,'report',[out/n for n in ['summary.json','report.html','risk_overview.png']])
    save(out/'completion.json',dict(status='complete',full_model_eligible=profile['full_model_eligible'],sources_unchanged=True,
        protocol_sha256=sha(out/'protocol.json'),stages={n:sha(out/f'{n}_complete.json') for n in ['data','factors','analysis','report']}))
    print(json.dumps(overview,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['audit','data','factors','risk','report','all'],default='all')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--industry-events',type=Path);parser.add_argument('--capital-events',type=Path)
    parser.add_argument('--require-full',action='store_true');args=parser.parse_args();out=args.output.resolve()
    if not out.is_relative_to((ROOT/'reports/experiments').resolve()):raise ValueError('Output must stay in isolated experiments directory')
    from filelock import FileLock
    out.mkdir(parents=True,exist_ok=True)
    with FileLock(str(out/'controller.lock'),timeout=0):
        save(out/'status.json',dict(stage=args.stage,state='running',pid=os.getpid(),started=time.time()))
        try:
            if args.stage in ['audit','all']:audit(out,args.industry_events,args.capital_events,args.require_full)
            if args.stage in ['data','all']:prepare(out)
            if args.stage in ['factors','all']:estimate(out)
            if args.stage in ['risk','all']:analyze(out)
            if args.stage in ['report','all']:report(out)
        except BaseException as error:
            save(out/'status.json',dict(stage=args.stage,state='failed',error=repr(error)));raise
        save(out/'status.json',dict(stage=args.stage,state='complete',finished=time.time()))


if __name__=='__main__':main()
