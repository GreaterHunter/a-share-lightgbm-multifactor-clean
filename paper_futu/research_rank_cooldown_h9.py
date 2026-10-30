"""Two-factor strategy attribution plus fixed-capacity H9 quarterly models."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='4'
import argparse
import gc
import time
from dataclasses import replace
import numpy as np
import pandas as pd
from paper_futu import research_top10_rank65_cooldown as joint
from paper_futu import research_top5_10_20_staged as staged
from paper_futu import research_top5_head2_full_exit as full
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu.top5_buffer_data import ROOT,START,END,read,save,digest,fingerprint,choice_from_folds
from paper_futu.top5_buffer_report import metrics,trade_metrics

SOURCE=fees.SOURCE
OUT=ROOT/'reports/experiments/rank_cooldown_h9_asof_20251231'
VARIANTS={'full_10_5':('full',.10,.05),'full_8_4':('full',.08,.04),
          'staged_10_5':('staged',.10,.05),'staged_8_4':('staged',.08,.04)}

def status(stage,**kw):save(OUT/'status.json',dict(stage=stage,pid=os.getpid(),time=time.time(),**kw))

def protocol():
    p=read(OUT/'protocol.json');assert p['code_sha256']==digest(__file__)
    fees.verify(p['protected']);return p

def audit():
    sources={};protected={}
    for base in (full.OUT,staged.OUT,joint.OUT):
        c=read(base/'completion.json');assert c['status']=='complete' and c['manifest_sha256']==digest(base/'report_manifest.json')
        p=read(base/'protocol.json');sources.update(p.get('sources',p.get('source',{})));sources.update(read(base/'report_manifest.json')['artifacts']);protected.update(p['protected'])
    sources[str(ROOT/'paper_futu/research_top10_rank65_cooldown.py')]=digest(ROOT/'paper_futu/research_top10_rank65_cooldown.py')
    fees.verify(sources);fees.verify(protected)
    audit=read(SOURCE/'models_audit.json');assert len(audit['periods'])==16
    for y,a in audit['annual_bindings'].items():assert choice_from_folds(a['fold_records'],int(y))==a['trees']
    for m in audit['periods']:assert m['label_end_max']<=m['cutoff']<m['start'] and m['end']<=END
    p=dict(start=START,end=END,code_sha256=digest(__file__),sources=sources,protected=protected,
        strategy_grid=dict(horizon=10,rank_exit=[50,65],profit_cooldown=[3,2],loss_cooldown=3,variants=list(VARIANTS)),
        model_test=dict(horizon=9,rank_exit=65,profit_cooldown=2,loss_cooldown=3,quarters=16,variants=list(VARIANTS),
        label='Open[T+10]/Open[T+1]-1, daily cross-sectional percentile rank minus 0.5',
        capacity='Keep original H10-selected annual trees and all QREG parameters; no H9 hyperparameter/tree search',
        boundary='Only signal dates with real label end <= quarterly cutoff; use immature dates for prediction only'),
        fees=read(full.OUT/'protocol.json')['fee_policy'],workers=1,threads=4,no_2026=True,no_live_change=True,
        inheritance='Same 15 raw factors, 2010 expanding training, seed42, quarterly fits, fixed historical universe and financial event snapshot',
        limitations=['Historical interval has been repeatedly researched, not an independent test.',
                    'H9 comparison is fixed model capacity, not best-tuned H9 versus best-tuned H10.',
                    'Published transaction fees exclude unverified custody fee and extra slippage; ideal daily opens.',
                    'Historical financial revisions and corporate-action vintage limitations remain.'])
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==p
    else:save(OUT/'protocol.json',p)
    save(OUT/'audit.json',dict(status='passed',quarters=16,causal_original_tree_choices=True))

def calendar():return pd.DatetimeIndex(pd.to_datetime((SOURCE/'data/provider/calendars/day.txt').read_text().splitlines()))

def labels_from_open(open_frame,index,cal,h):
    """Exact trading-session endpoints; never compress missing stock prices."""
    dates=index.get_level_values('datetime');codes=index.get_level_values('instrument')
    positions=cal.get_indexer(dates);assert (positions>=0).all()
    start=positions+1;end=positions+h+1;known=end<len(cal)
    values=np.full(len(index),np.nan,dtype='float64');ends=np.full(len(index),np.datetime64('NaT'),dtype='datetime64[ns]')
    ends[known]=cal.to_numpy()[end[known]]
    cols=open_frame.columns.get_indexer(codes);ok=known & (cols>=0)
    raw=open_frame.reindex(cal).to_numpy()
    x=raw[start[ok],cols[ok]];z=raw[end[ok],cols[ok]]
    values[ok]=np.where((x>0)&(z>0),z/x-1,np.nan)
    result=pd.DataFrame({'label':values,'label_end_date':ends},index=index)
    result['target']=result.label.groupby(level='datetime').rank(method='average',pct=True)-.5
    return result.replace([np.inf,-np.inf],np.nan)

def train():
    import lightgbm as lgb
    p=protocol();fees.verify(p['sources'])
    cal=calendar();assert cal.max()==pd.Timestamp(END)
    features=pd.read_parquet(SOURCE/'data/features.parquet');assert features.index.get_level_values('datetime').max()<=pd.Timestamp(END)
    assert not features.index.has_duplicates and features.shape[1]==15
    wide={}
    for code in features.index.get_level_values('instrument').unique():
        path=SOURCE/'data/provider/features'/code.lower()/'open.day.bin'
        if not path.exists():continue
        data=np.fromfile(path,dtype='<f4');start=int(data[0]);assert start+len(data)-1<=len(cal)
        a=np.full(len(cal),np.nan,dtype=np.float32);a[start:start+len(data)-1]=data[1:];wide[code]=a
    opens=pd.DataFrame(wide,index=cal);del wide
    label=labels_from_open(opens,features.index,cal,9)
    label.to_parquet(OUT/'labels_h9.parquet')
    # Boundary perturbation: altering any strictly post-cutoff opens cannot change matured training labels.
    cut=pd.Timestamp('2023-12-29');changed=opens.copy();changed.loc[changed.index>cut]*=11
    mutated=labels_from_open(changed,features.index,cal,9);mature=label.label_end_date.le(cut)
    pd.testing.assert_frame_equal(label.loc[mature],mutated.loc[mature],check_exact=True)
    del opens,changed,mutated;gc.collect()
    model_audit=read(SOURCE/'models_audit.json');expected=pd.read_parquet(SOURCE/'data/predictions.parquet')
    pieces=[];records=[]
    for n,m in enumerate(model_audit['periods']):
        cutoff=m['cutoff'];d=OUT/'models'/cutoff;d.mkdir(parents=True,exist_ok=True)
        part=expected.loc[expected.execution_date.between(m['start'],m['end']),['execution_date','model_cutoff']].copy()
        mask=label.label_end_date.le(pd.Timestamp(cutoff)) & label.target.notna() & label.label.notna()
        trainx=features.loc[mask];target=label.loc[mask,'target']
        assert label.loc[mask,'label_end_date'].max()<=pd.Timestamp(cutoff)
        assert (part.index.get_level_values('datetime')>=pd.Timestamp(cutoff)).all()
        params=dict(m['parameters']);params['n_jobs']=4
        binding=dict(cutoff=cutoff,start=m['start'],end=m['end'],trees=m['trees'],parameters=params,
          rows=len(trainx),training_hash=fingerprint(trainx.join(label.loc[mask])),feature_names=list(features.columns),
          label_end_max=str(label.loc[mask,'label_end_date'].max().date()),signal_end=str(trainx.index.get_level_values('datetime').max().date()),
          protocol_sha256=digest(OUT/'protocol.json'))
        status('train',quarter=n+1,total_quarters=16,cutoff=cutoff,trees=m['trees'])
        if (d/'complete.json').exists():
            old=read(d/'complete.json');assert old['binding']==binding;fees.verify(old['artifacts'])
            model=lgb.Booster(model_file=str(d/'model.txt'))
        else:
            started=time.time();print(f'FIT {n+1}/16 cutoff={cutoff} trees={m["trees"]} rows={len(trainx)}',flush=True)
            fit=lgb.LGBMRegressor(**params,n_estimators=m['trees'],random_state=42)
            fit.fit(trainx,target);fit.booster_.save_model(str(d/'model.txt'))
            model=lgb.Booster(model_file=str(d/'model.txt'))
            x=features.reindex(part.index)
            np.testing.assert_allclose(fit.predict(x),model.predict(x,num_threads=4),rtol=0,atol=1e-12)
            save(d/'complete.json',dict(binding=binding,reload_equal=True,elapsed=time.time()-started,artifacts={str(d/'model.txt'):digest(d/'model.txt')}))
            del fit,x
        part['prediction']=model.predict(features.reindex(part.index),num_threads=4)
        assert np.isfinite(part.prediction).all();pieces.append(part)
        records.append(read(d/'complete.json'));del trainx,target,model;gc.collect()
        print(f'COMPLETE H9 quarter {n+1}/16',flush=True)
    predictions=pd.concat(pieces).sort_index();pd.testing.assert_index_equal(predictions.index,expected.index)
    pd.testing.assert_series_equal(predictions.execution_date,expected.execution_date)
    predictions.to_parquet(OUT/'predictions_h9.parquet')
    save(OUT/'training_audit.json',dict(status='passed',models=records,quarters=16,reload_equal=True,future_perturbation_invariant=True,
         prediction_rows=len(predictions),predictions_sha256=digest(OUT/'predictions_h9.parquet'),labels_sha256=digest(OUT/'labels_h9.parquet')))
    del features,label,expected,pieces;gc.collect()

def make_account(variant,rank,cooldown):
    mode,tp,sl=VARIANTS[variant]
    if cooldown==2:a=joint.make_account(mode,tp,sl)
    elif mode=='staged':a=staged.make_account(10,tp,sl)
    else:a=full.engine().Account(10,profit=tp,stop=sl)
    a.rules=replace(a.rules,rank_exit=rank);return a

def replay(horizon,rank,cooldown,variant,market,pred,cal):
    name=f'h{horizon}_r{rank}_c{cooldown}_{variant}'
    if horizon==10 and (rank,cooldown) in ((50,3),(65,2)):
        base=joint.OUT if rank==65 else (staged.OUT if variant.startswith('staged') else full.OUT)
        return dict(horizon=horizon,rank=rank,cooldown=cooldown,variant=variant,path=str(base/'accounts'/f'top10_{variant}'),reused=True)
    d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
    if not (d/'complete.json').exists():
        a=make_account(variant,rank,cooldown);days=cal[(cal>=START)&(cal<=END)]
        for step,day in enumerate(days):
            prev=cal[cal.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all()
            a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],{c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
        a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
        if variant.startswith('full'):assert all(not e['half_mode'] for e in a.episodes+opened)
        for n,r in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():pd.DataFrame(r).to_parquet(d/f'{n}.parquet',index=False)
        save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
        print(f'COMPLETE {name} NAV {a.daily[-1]["nav"]:.2f}',flush=True)
    c=read(d/'complete.json');assert c['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(c['artifacts'])
    return dict(horizon=horizon,rank=rank,cooldown=cooldown,variant=variant,path=str(d),reused=False)

def backtest(horizon):
    p=protocol();fees.verify(p['sources']);cal=calendar()
    panel=pd.read_parquet(SOURCE/'data/market.parquet').reset_index()
    market={d:g.set_index('instrument')[['open','close','factor']].to_dict('index') for d,g in panel.groupby('datetime')};del panel
    if horizon==9:
        a=read(OUT/'training_audit.json');assert a['predictions_sha256']==digest(OUT/'predictions_h9.parquet')
    frame=pd.read_parquet(SOURCE/'data/predictions.parquet' if horizon==10 else OUT/'predictions_h9.parquet').reset_index()
    assert frame.execution_date.max()<=pd.Timestamp(END);pred=dict(tuple(frame.groupby('execution_date')))
    configs=[(r,c,v) for r in (50,65) for c in (3,2) for v in VARIANTS] if horizon==10 else [(65,2,v) for v in VARIANTS]
    results=[]
    for i,(rank,cooldown,variant) in enumerate(configs):
        status('backtest',horizon=horizon,account=i+1,total=len(configs))
        results.append(replay(horizon,rank,cooldown,variant,market,pred,cal))
    save(OUT/f'accounts_h{horizon}.json',results)

def decomposition(a,b,c,d):
    """A=50/3, B=50/2, C=65/3, D=65/2. Values may be CAGR or net P&L."""
    return dict(rank_only=c-a,cooldown_only=b-a,interaction=d-c-b+a,joint=d-a,
                allocated_rank=((c-a)+(d-b))/2,allocated_cooldown=((b-a)+(d-c))/2)

def report(final=False):
    from pathlib import Path
    p=protocol();benchmark=pd.read_parquet(SOURCE/'data/benchmark.parquet').benchmark_return
    configs=read(OUT/'accounts_h10.json')+(read(OUT/'accounts_h9.json') if final else [])
    rows=[];years=[];checks={}
    for config in configs:
        d=Path(config['path']);fees.verify(read(d/'complete.json')['artifacts'])
        daily,orders,held,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
        for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
        assert pd.DatetimeIndex(daily.date).equals(benchmark.index)
        key=f'{config["horizon"]}_{config["rank"]}_{config["cooldown"]}_{config["variant"]}'
        checks[key]=fees.ledger_audit(daily,orders,held,closed,opened,10)
        info={k:config[k] for k in ('horizon','rank','cooldown','variant')}
        rows.append(dict(**info,**metrics(daily,benchmark),mean_holding_days=float(closed.holding_days.mean()),fills=len(orders),**trade_metrics(closed,opened)))
        for y,g in daily.groupby(daily.date.dt.year):years.append(dict(**info,year=int(y),**metrics(g,benchmark)))
    table=pd.DataFrame(rows);table.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    annual=pd.DataFrame(years);annual.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    effects=[]
    for v,g in table[table.horizon==10].groupby('variant'):
        q=g.set_index(['rank','cooldown'])
        for measure in ('cagr','total_return','max_drawdown','fees'):
            effects.append(dict(variant=v,measure=measure,**decomposition(*(q.loc[pair,measure] for pair in ((50,3),(50,2),(65,3),(65,2))))))
    effects=pd.DataFrame(effects);effects.to_csv(OUT/'factor_effects.csv',index=False,encoding='utf-8-sig')
    grid=table[table.horizon==10].pivot(index='variant',columns=['rank','cooldown'],values='cagr').map(lambda x:f'{x:.2%}')
    risk=table[table.horizon==10].pivot(index='variant',columns=['rank','cooldown'],values='max_drawdown').map(lambda x:f'{x:.2%}')
    page='<html lang="zh"><meta charset="utf-8"><title>排名冷却消融与H9</title><style>body{font:16px/1.7 system-ui;max-width:1300px;margin:35px auto}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style><h1>排名退出与止盈冷却：四格消融；9日模型对照</h1><p>2022—2025，初始100万元，Top10入场等权、富途公开交易费用暂定口径。止损冷却始终D+3，盈利跟踪止损仍3日；仅止盈全清比较D+3与D+2。模型、强势换股及其他规则不变。</p><h2>10日模型净年化：列为排名门槛／止盈冷却</h2>'+grid.to_html()+'<h2>最大回撤</h2>'+risk.to_html()
    page+='<p>A=50/3，B=50/2，C=65/3，D=65/2。排名单独效果=C−A；冷却单独效果=B−A；交互=D−C−B+A。联合变化=D−A。另以两种修改顺序的平均分摊联合变化；这是指标差值分解，不是统计显著性或因果证明。</p><h2>年化变化分解（小数）</h2>'+effects[effects.measure=='cagr'].round(6).to_html(index=False)
    if final:
        paired=table[(table['rank']==65)&(table.cooldown==2)].copy();paired.to_csv(OUT/'h9_vs_h10.csv',index=False,encoding='utf-8-sig')
        show=paired[['horizon','variant','total_return','cagr','max_drawdown','fees','mean_holding_days']].copy()
        for col in ('total_return','cagr','max_drawdown'):show[col]=show[col].map(lambda x:f'{x:.2%}')
        page+='<h2>相同65/2策略：9日与10日模型</h2>'+show.round(2).to_html(index=False)
        page+='<p>H9标签=Open[T+10]/Open[T+1]−1，目标仍为横截面排名减0.5。16个季度从头训练，仅使用各截止日已成熟标签。树数沿用原H10年度历史验证选择，QREG其他参数不变；没有针对H9重新调参，故这是固定模型容量对照。模型保存重载预测一致，标签未来扰动测试通过。</p>'
        # Comparable IC on common mature dates, each target plus shared H10 outcome.
        from paper_futu.final_ablation_core import daily_ic
        h10=pd.read_parquet(SOURCE/'data/predictions.parquet');h9=pd.read_parquet(OUT/'predictions_h9.parquet')
        lab10=pd.read_parquet(SOURCE/'data/labels.parquet').label;lab9=pd.read_parquet(OUT/'labels_h9.parquet').label
        ic=[]
        for h,predictions in ((10,h10),(9,h9)):
            for target,label in ((10,lab10),(9,lab9)):
                both=lab10.reindex(predictions.index).notna() & lab9.reindex(predictions.index).notna()
                frame=daily_ic(predictions.loc[both,'prediction'],label.reindex(predictions.index[both]))
                frame.to_csv(OUT/f'ic_h{h}_target{target}.csv',encoding='utf-8-sig')
                ic.append(dict(horizon=h,target=target,**frame.mean(numeric_only=True).to_dict(),days=len(frame)))
        pd.DataFrame(ic).to_csv(OUT/'ic_comparison.csv',index=False,encoding='utf-8-sig');page+='<h2>相同成熟样本上的预测相关性</h2>'+pd.DataFrame(ic).round(5).to_html(index=False)
    else:page+='<p>当前为策略消融阶段报告；9日模型尚未完成。</p>'
    page+='<p>原始财务修订档案不完整；公开费用未含未核实组合费及额外滑点。收盘决策次日理想开盘成交，未使用2026数据；历史已反复研究，不能称为独立测试。不修改S6模拟盘。</p><p><a href="comparison.csv">全指标</a> · <a href="factor_effects.csv">分解</a> · <a href="yearly.csv">逐年结果</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf-8')
    fees.verify(p['sources']);fees.verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',accounts=checks,count=len(configs),sources_and_live_unchanged=True,final=final))
    if final:
        save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock','status.json')}))
        save(OUT/'completion.json',dict(status='complete',accounts=20,new_accounts=12,quarters=16,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(grid.to_string(),flush=True);print(effects[effects.measure=='cagr'].to_string(index=False),flush=True)

def main():
    import msvcrt
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['audit','strategy','train','model','report','all'],default='all');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:
            if args.stage in ('audit','all'):audit()
            if args.stage in ('strategy','all'):backtest(10);report(False)
            if args.stage in ('train','all'):train()
            if args.stage in ('model','all'):backtest(9)
            if args.stage in ('report','all'):report(True)
            status('complete',requested_stage=args.stage)
        except Exception as e:status('failed',error=repr(e));raise
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)

if __name__=='__main__':main()
