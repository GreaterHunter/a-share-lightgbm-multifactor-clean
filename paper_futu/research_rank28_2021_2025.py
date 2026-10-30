"""Extend the fixed 28 rules to continuous 2021-2025 accounts, excluding 2026."""
import os,gc
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='4'
import numpy as np
import pandas as pd
from paper_futu.top5_buffer_data import ROOT,read,save,digest,fingerprint,choice_from_folds,init_qlib
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu import research_rank80_two_thresholds as grid
from paper_futu import research_top10_rank65_cooldown as cooldown
from paper_futu.top5_buffer_report import metrics,trade_metrics
from paper_futu.final_ablation_core import daily_ic,mature
SOURCE=fees.SOURCE
OUT=ROOT/'reports/experiments/top28_continuous_2021_2025'
START,END='2021-01-04','2025-12-31'
CANDIDATES=[10,20,40,80,160,320,640]

def calendar():return pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))

def compatible_sources(sources):
    """Preserve original scientific protocol, separately bind a presentation-only repair."""
    result=dict(sources);path=str(ROOT/'paper_futu/research_rank28_2021_2025.py')
    if result[path]!=digest(path):
        repair=read(OUT/'column_alignment_repair.json')
        assert repair['old_code_sha256']==result[path] and repair['new_code_sha256']==digest(path)
        assert repair['original_protocol_sha256']==digest(OUT/'protocol.json')
        fees.verify(repair['completed_model_artifacts'])
        result[path]=repair['new_code_sha256']
    return result

def audit():
    c=read(SOURCE/'completion.json');assert c['status']=='complete' and c['report_manifest_sha256']==digest(SOURCE/'report_manifest.json')
    fees.verify(read(SOURCE/'report_manifest.json')['artifacts'])
    sources=read(SOURCE/'audit_complete.json')['artifacts'];fees.verify(sources)
    protected=read(SOURCE/'protocol.json')['protected'];fees.verify(protected)
    for name in ('models_audit.json','data_audit.json','audit_complete.json'):
        sources[str(SOURCE/name)]=digest(SOURCE/name)
    for name in ('research_rank28_2021_2025.py','research_rank80_two_thresholds.py','rank_buffer_full_exit_engine.py',
                 'rank_buffer_10_20_engine.py','research_top10_rank65_cooldown.py','research_rank_buffer_futu_fees.py','top5_buffer_report.py'):
        path=ROOT/'paper_futu'/name;sources[str(path)]=digest(path)
    models=read(SOURCE/'models_audit.json')
    for row in models['periods']:
        assert row['label_end_max']<=row['cutoff']<row['start'] and row['end']<=END
        assert digest(row['model_path'])==row['model_sha256'];sources[row['model_path']]=row['model_sha256']
    for y,a in models['annual_bindings'].items():assert choice_from_folds(a['fold_records'],int(y))==a['trees']
    # Fixed base parameter family is the original H10 p00, not a search selected on 2021 outcomes.
    base=models['annual_bindings']['2022']['parameters'];qreg=models['periods'][0]['parameters']
    assert base['learning_rate']==qreg['learning_rate']==.03
    p=dict(start=START,end=END,sources=sources,protected=protected,configs=[list(x) for x in grid.CONFIGS],
        initial_cash=1000000,continuous=True,reset_on_year_boundary=False,
        validation_years=[2018,2019,2020],tree_candidates=CANDIDATES,base_parameters=base,final_parameters=qreg,
        tree_selection='Original procedure: fixed p00 parameters, three preceding annual mean Rank IC; ties fold std then tree complexity. Apply established QREG final fit after tree selection.',
        ranking='50% rank of net CAGR descending + 50% rank of continuous max drawdown ascending; ties worst annual return then lower fees. 30% drawdown flag not relaxed.',
        model_scope='Four new quarterly fits for 2021; reuse audited 16 quarterly models and predictions for 2022-2025. No 2026 outcome input.',
        strategy='28 fixed existing rules; H10, Top10, quarter refit, TP cooldown2 SL3, same stage scaling and fee-inclusive budgets.',
        fee_policy='Same published Futu transaction fee approximation as previous experiments; custody excluded, no extra slippage.',
        limitations='Observed historical parameter search; not independent test; incomplete historical financial revision vintages; ideal daily opens.',threads=4,workers=1)
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        original=read(OUT/'protocol.json')
        assert compatible_sources(original['sources'])==p['sources']
        p['sources']=original['sources'];assert original==p
    else:save(OUT/'protocol.json',p)
    return p

def train(p):
    import lightgbm as lgb
    if (OUT/'training_complete.json').exists():
        record=read(OUT/'training_complete.json');assert record['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(record['artifacts']);return
    features=pd.read_parquet(SOURCE/'data/features.parquet');labels=pd.read_parquet(SOURCE/'data/labels.parquet')
    assert features.index.get_level_values('datetime').max()<=pd.Timestamp(END) and not features.index.has_duplicates
    frame=features.join(labels);frame['target']=frame.rank_target
    records=[]
    for year in p['validation_years']:
        path=OUT/'folds'/f'{year}.json'
        if path.exists():
            cached=read(path);assert cached['protocol_sha256']==digest(OUT/'protocol.json');records+=cached['records'];continue
        trainset=mature(frame,'2010-01-01',f'{year-1}-12-31');valid=mature(frame,f'{year}-01-01',f'{year}-12-31')
        print('FIT VALIDATION',year,len(trainset),flush=True)
        fit=lgb.LGBMRegressor(**p['base_parameters'],n_estimators=max(CANDIDATES),random_state=42)
        fit.fit(trainset[features.columns],trainset.target)
        rows=[]
        for trees in CANDIDATES:
            score=pd.Series(fit.predict(valid[features.columns],num_iteration=trees),index=valid.index)
            ic=daily_ic(score,valid.label)
            assert np.isfinite(ic.rank_ic.mean())
            rows.append(dict(parameter_id='p00',trees=trees,num_leaves=p['base_parameters']['num_leaves'],validation_year=year,
                mean_rank_ic=float(ic.rank_ic.mean()),mean_ic=float(ic.ic.mean()),valid_days=int(ic.rank_ic.notna().sum()),
                train_rows=len(trainset),valid_rows=len(valid),train_label_end_max=str(trainset.label_end_date.max().date()),
                valid_label_end_max=str(valid.label_end_date.max().date()),training_fingerprint=fingerprint(trainset),validation_fingerprint=fingerprint(valid)))
        save(path,dict(protocol_sha256=digest(OUT/'protocol.json'),records=rows));records+=rows
        del fit,trainset,valid;gc.collect();print('DONE VALIDATION',year,flush=True)
    trees=choice_from_folds(records,2021);save(OUT/'tree_choice_2021.json',dict(trees=trees,records=records))
    print('SELECTED 2021 TREES',trees,flush=True)
    cal=calendar();assert cal.max()==pd.Timestamp(END)
    dates=features.index.get_level_values('datetime');pos=cal.get_indexer(dates)+1
    mask=(pos<len(cal)) & (dates>=pd.Timestamp('2020-12-31')) & (dates<=pd.Timestamp('2021-12-30'))
    expected=pd.DataFrame(index=features.index[mask]);expected['execution_date']=cal.to_numpy()[pos[mask]]
    pieces=[];models=[]
    for quarter in range(1,5):
        quarterdays=cal[(cal.year==2021)&(cal.quarter==quarter)]
        cutoff=cal[cal.get_loc(quarterdays[0])-1];cutstr=str(cutoff.date())
        tr=mature(frame,'2010-01-01',cutstr)
        part=expected.loc[expected.execution_date.between(quarterdays[0],quarterdays[-1])].copy()
        assert part.index.get_level_values('datetime').min()>=cutoff
        d=OUT/'models'/cutstr;d.mkdir(parents=True,exist_ok=True)
        binding=dict(cutoff=cutstr,start=str(quarterdays[0].date()),end=str(quarterdays[-1].date()),trees=trees,
            parameters=p['final_parameters'],train_rows=len(tr),label_end_max=str(tr.label_end_date.max().date()),
            training_fingerprint=fingerprint(tr),protocol_sha256=digest(OUT/'protocol.json'))
        if (d/'complete.json').exists():
            r=read(d/'complete.json');assert r['binding']==binding;fees.verify(r['artifacts']);model=lgb.Booster(model_file=str(d/'model.txt'))
        else:
            print('FIT QUARTER',cutstr,len(tr),flush=True)
            fit=lgb.LGBMRegressor(**p['final_parameters'],n_estimators=trees,random_state=42);fit.fit(tr[features.columns],tr.target)
            fit.booster_.save_model(str(d/'model.txt'));model=lgb.Booster(model_file=str(d/'model.txt'))
            np.testing.assert_allclose(fit.predict(features.loc[part.index]),model.predict(features.loc[part.index],num_threads=4),atol=1e-12,rtol=0)
            save(d/'complete.json',dict(binding=binding,reload_equal=True,artifacts={str(d/'model.txt'):digest(d/'model.txt')}));del fit
        part['prediction']=model.predict(features.loc[part.index],num_threads=4);part['model_cutoff']=cutoff
        assert np.isfinite(part.prediction).all() and (part.model_cutoff<part.execution_date).all()
        pieces.append(part);models.append(read(d/'complete.json'));del tr,model;gc.collect()
        print('DONE QUARTER',cutstr,flush=True)
    new=pd.concat(pieces).sort_index();pd.testing.assert_index_equal(new.index,expected.sort_index().index)
    history=pd.read_parquet(SOURCE/'data/predictions.parquet');assert history.execution_date.min()>new.execution_date.max()
    new=new.reindex(columns=history.columns).astype(history.dtypes.to_dict())
    combined=pd.concat([new,history]).sort_index();assert not combined.index.has_duplicates and combined.execution_date.max()<=pd.Timestamp(END)
    pd.testing.assert_frame_equal(combined.loc[history.index],history)
    (OUT/'data').mkdir(exist_ok=True);combined.to_parquet(OUT/'data/predictions.parquet')
    save(OUT/'models_audit.json',dict(new_models=models,existing_models=read(SOURCE/'models_audit.json')['periods'],annual_tree_choice=trees,no_2026=True))
    save(OUT/'training_complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and (f.suffix in ('.txt','.parquet') or f.name in ('models_audit.json','tree_choice_2021.json'))}))
    del features,labels,frame;gc.collect()

def prepare_market():
    if (OUT/'market_complete.json').exists():fees.verify(read(OUT/'market_complete.json')['artifacts']);return
    init_qlib(SOURCE/'data/provider');from qlib.data import D
    codes=sorted({line.split()[0] for line in (SOURCE/'data/provider/instruments/csi300.txt').read_text().splitlines()})
    panel=D.features(codes,['$open','$close','$factor'],start_time='2020-12-31',end_time=END);panel.columns=['open','close','factor']
    panel.to_parquet(OUT/'data/market.parquet')
    from paper_futu.benchmark_attribution import fetch_official,official_frame
    fetch_official(OUT/'data/benchmark','2020-12-31',END)
    bm=official_frame(read(OUT/'data/benchmark/H00300.json'),'H00300');days=calendar();days=days[(days>=START)&(days<=END)]
    b=bm.close.pct_change(fill_method=None).reindex(days);basis='Previous-year close; official first open unavailable'
    if np.isfinite(bm.loc[days[0],'open']) and bm.loc[days[0],'open']>0:
        b.iloc[0]=bm.loc[days[0],'close']/bm.loc[days[0],'open']-1;basis='First-day close/open, thereafter close/previous close'
    assert np.isfinite(b).all();b.rename('benchmark_return').to_frame().to_parquet(OUT/'data/benchmark.parquet')
    save(OUT/'market_complete.json',dict(benchmark_basis=basis,artifacts={str(f):digest(f) for f in (OUT/'data').rglob('*') if f.is_file()}))

def replay(p):
    cal=calendar();days=cal[(cal>=START)&(cal<=END)]
    panel=pd.read_parquet(OUT/'data/market.parquet').reset_index();market={d:g.set_index('instrument').to_dict('index') for d,g in panel.groupby('datetime')};del panel
    pred=dict(tuple(pd.read_parquet(OUT/'data/predictions.parquet').reset_index().groupby('execution_date')))
    b=pd.read_parquet(OUT/'data/benchmark.parquet').benchmark_return
    records=[];annual=[];checks={}
    for cfg in grid.CONFIGS:
        rank,strong,mode,t=cfg;name=f'r{rank}_s{strong}_{mode}_{t}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
        if not (d/'complete.json').exists():
            a=grid.make_account(*cfg)
            for step,day in enumerate(days):
                prev=cal[cal.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all() and (r.model_cutoff<=r.datetime).all()
                a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],{c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
            a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
            for n,value in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():pd.DataFrame(value).to_parquet(d/f'{n}.parquet',index=False)
            save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
        cached=read(d/'complete.json');assert cached['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(cached['artifacts'])
        daily,orders,held,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
        for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
        assert pd.DatetimeIndex(daily.date).equals(days)
        checks[name]=fees.ledger_audit(daily,orders,held,closed,opened,10);checks[name]['cooldown_checks']=cooldown.audit_cooldown(orders,closed,cal)
        info=dict(name=name,rank_exit=rank,strong_rank=strong,mode=mode,threshold=t)
        records.append(dict(**info,**metrics(daily,b),**trade_metrics(closed,opened),fills=len(orders)))
        for year,g in daily.groupby(daily.date.dt.year):annual.append(dict(**info,year=int(year),**metrics(g,b)))
        print('ACCOUNT COMPLETE',name,round(daily.nav.iloc[-1],2),flush=True)
    table=pd.DataFrame(records);years=pd.DataFrame(annual);assert len(table)==28 and len(years)==140
    table['cagr_rank']=table.cagr.rank(ascending=False,method='average');table['drawdown_rank']=table.max_drawdown.rank(method='average')
    table['mean_metric_rank']=(table.cagr_rank+table.drawdown_rank)/2
    table=table.join(years.groupby('name').total_return.min().rename('worst_year'),on='name')
    table=table.sort_values(['mean_metric_rank','worst_year','fees'],ascending=[True,False,True]).reset_index(drop=True);table.insert(0,'priority',range(1,29))
    table.to_csv(OUT/'ranking.csv',index=False,encoding='utf-8-sig');years.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    display=[]
    for r in table.itertuples():
        yr=years[years.name.eq(r.name)].set_index('year')
        display.append({'顺序':r.priority,'排名':f'{r.rank_exit}/{r.strong_rank}','退出':'直接清仓' if r.mode=='full' else '分段退出',
            '止盈/止损':r.threshold.replace('_','% / ')+'%','净年化':f'{r.cagr:.2%}','全期最大回撤':f'{r.max_drawdown:.2%}',
            '五年累计':f'{r.total_return:.2%}','费用':f'{r.fees/10000:.2f}万','平均名次':r.mean_metric_rank,
            **{str(y):f'{yr.loc[y,"total_return"]:.2%}' for y in range(2021,2026)}})
    show=pd.DataFrame(display)
    html='<html lang="zh"><meta charset="utf-8"><title>2021—2025连续回测与28版本排序</title><style>body{font:16px/1.7 system-ui;margin:30px}table{border-collapse:collapse;white-space:nowrap;font-size:13px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}th{background:#eef3fa}tbody tr:nth-child(-n+5){background:#eaf6ee}</style><h1>2021—2025连续回测：28个版本</h1>'
    html+='<p>初始100万元，2021年1月4日至2025年12月31日连续运行，跨年不清仓、不重置状态。主排序：净年化与连续最大回撤分别排名，名次等权平均；并列依次比较最差年度收益、费用。30%回撤线未放宽，研究关注顺序不等于风险合格。</p><p>补训2021四个季度模型，树数按原p00固定参数在2018—2020三个验证年选择，再按原流程应用QREG参数训练季度模型。2022—2025的16个已审计模型和预测复用。所有标签边界过滤，未用2026绩效筛选。</p>'
    html+='<div style="overflow:auto">'+show.to_html(index=False)+'</div><p>逐年数字为净收益，全期最大回撤跨年度计算。2022—2025可能与先前年初2022现金起跑结果不同，因为本次承接2021的真实持仓、冷却状态及资金，不能直接拼接年度表。</p><p>历史已被反复研究；固定规则本身也是事后选择，不能称为独立测试。缺少完整财务修订档案，理想次日开盘成交、既定费用假设，无额外滑点；未切换模拟盘。</p><p><a href="ranking.csv">全指标</a> · <a href="yearly.csv">逐年收益与回撤</a> · <a href="models_audit.json">训练边界</a></p></html>'
    (OUT/'report.html').write_text(html,encoding='utf-8')
    fees.verify(compatible_sources(p['sources']));fees.verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',accounts=28,checks=checks,new_quarterly_models=4,existing_quarterly_models=16,no_2026=True,continuous=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('report_manifest.json','completion.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=28,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(show.head(10).to_string(index=False),flush=True)

def run():
    p=audit();train(p);prepare_market();replay(p)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
