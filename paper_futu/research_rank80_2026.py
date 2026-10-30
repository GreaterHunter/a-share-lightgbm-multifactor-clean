"""Fresh-cash 2026 replay of the ten fixed rank65/rank80 exit policies."""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='4'
from paper_futu.top5_buffer_data import ROOT,SOURCE,STUDY,MODELS,read,save,digest,fingerprint,choice_from_folds,init_qlib
from paper_futu import research_rank_buffer_futu_fees as fees
from paper_futu import research_top10_rank65_cooldown as old
from paper_futu import research_top10_rank80_strong1 as new
START,END='2026-01-05','2026-10-13'
OUT=ROOT/'reports/experiments/top10_rank80_strong1_2026_asof_20261013'

def validate_boundary(meta,period):
    assert meta['cutoff']==period['cutoff']
    assert meta['label_end_max']<=meta['cutoff']<period['start']
    assert meta['train_signal_end']<=meta['cutoff']

def prepare():
    import gc
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from src.factor_library import ENHANCED_FEATURE_NAMES
    from paper_futu.benchmark_attribution import fetch_official,official_frame
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    prior=read(new.OUT/'completion.json');assert prior['manifest_sha256']==digest(new.OUT/'report_manifest.json')
    fees.verify(read(new.OUT/'report_manifest.json')['artifacts'])
    protected=read(new.OUT/'protocol.json')['protected'];fees.verify(protected)
    snap=read(SOURCE/'manifest.json');p=read(STUDY/'protocol.json')
    assert snap['snapshot_id']==p['snapshot_id'] and snap['as_of']==END
    # Validate the sealed provider, feature and label files against the original snapshot.
    hashes=next(v for k,v in snap.items() if isinstance(v,dict) and 'raw_features.parquet' in v)
    sources={str(SOURCE/k):v for k,v in hashes.items()}
    fees.verify(sources)
    manifest=read(STUDY/'report_manifest.json')['artifacts']
    for path in (STUDY/'protocol.json',STUDY/'predictions/history_QREG.parquet'):
        assert digest(path)==manifest[str(path)];sources[str(path)]=digest(path)
    sources.update({str(ROOT/'paper_futu'/n):digest(ROOT/'paper_futu'/n) for n in
        ('research_rank80_2026.py','research_top10_rank80_strong1.py','research_top10_rank65_cooldown.py',
         'rank_buffer_full_exit_engine.py','rank_buffer_10_20_engine.py','research_rank_buffer_futu_fees.py','top5_buffer_report.py')})
    calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'provider/calendars/day.txt').read_text().splitlines()))
    assert calendar.max()==pd.Timestamp(END)
    features=pd.read_parquet(SOURCE/'raw_features.parquet',columns=ENHANCED_FEATURE_NAMES)
    labels=pd.read_parquet(SOURCE/'labels_h10.parquet')
    frame=features.join(labels);frame['target']=frame.rank_target
    assert not frame.index.has_duplicates and frame.index.get_level_values('datetime').max()<=pd.Timestamp(END)
    positions=calendar.get_indexer(features.index.get_level_values('datetime'))+1
    valid=(positions<len(calendar)) & (features.index.get_level_values('datetime')>=pd.Timestamp('2025-12-31'))
    expected=pd.DataFrame(index=features.index[valid]);expected['execution_date']=calendar.to_numpy()[positions[valid]]
    legacy=pd.read_parquet(STUDY/'predictions/history_QREG.parquet',filters=[('execution_date','>=',pd.Timestamp(START))])
    choice=p['annual_bindings']['2026'];assert choice_from_folds(choice['fold_records'],2026)==choice['trees']
    pieces=[];audits=[]
    for period in [r for r in p['periods'] if r['year']==2026]:
        cutoff=period['cutoff'];mp=MODELS/cutoff/'metadata.json';meta=read(mp);validate_boundary(meta,period)
        taskpath=STUDY/'tasks'/f'fit_{cutoff}.json';assert digest(taskpath)==manifest[str(taskpath)]
        task=read(taskpath);assert task['status']=='complete';fees.verify(task['artifacts']);sources.update(task['artifacts'])
        assert meta==task['result'] and meta['trees']==choice['trees']
        train=frame.loc[frame.label_end_date.le(pd.Timestamp(cutoff))].dropna(subset=['label','target'])
        assert len(train)==meta['train_rows'] and fingerprint(train)==meta['training_fingerprint']
        assert str(train.label_end_date.max().date())==meta['label_end_max']
        part=expected.loc[expected.execution_date.between(period['start'],period['end'])].copy()
        model=lgb.Booster(model_file=meta['model_path']);part['prediction']=model.predict(features.reindex(part.index),num_threads=4)
        part['model_cutoff']=pd.Timestamp(cutoff)
        assert (part.model_cutoff<=part.index.get_level_values('datetime')).all()
        assert (part.index.get_level_values('datetime')<part.execution_date).all()
        previous=legacy.loc[legacy.execution_date.between(period['start'],period['end'])]
        assert previous.index.isin(part.index).all()
        np.testing.assert_allclose(part.loc[previous.index,'prediction'],previous.prediction,atol=1e-12,rtol=0)
        audits.append(dict(**period,train_rows=len(train),label_end_max=meta['label_end_max'],trees=meta['trees'],
            model_sha256=meta['model_sha256'],training_fingerprint_verified=True,reload_equal=True,
            recovered_signal_rows=len(part)-len(previous)))
        pieces.append(part);del train,model;gc.collect()
        print('AUDIT quarter',cutoff,'rows',len(part),flush=True)
    predictions=pd.concat(pieces).sort_index()
    assert len(audits)==4 and predictions.index.equals(expected.sort_index().index)
    assert not predictions.index.has_duplicates and np.isfinite(predictions.prediction).all()
    predictions.to_parquet(OUT/'data/predictions.parquet')
    del frame,features,labels,legacy;gc.collect()
    init_qlib(SOURCE/'provider');from qlib.data import D
    codes=sorted({line.split()[0] for line in (SOURCE/'provider/instruments/csi300.txt').read_text().splitlines()})
    market=D.features(codes,['$open','$close','$factor'],start_time='2025-12-31',end_time=END);market.columns=['open','close','factor']
    assert market.index.get_level_values('datetime').max()==pd.Timestamp(END)
    market.to_parquet(OUT/'data/market.parquet')
    fetch_official(OUT/'data/benchmark','2025-12-31',END)
    bm=official_frame(read(OUT/'data/benchmark/H00300.json'),'H00300')
    days=calendar[(calendar>=START)&(calendar<=END)]
    # Official 2026 H00300 responses have null opens. Do not invent a first-open
    # return: explicitly report the calendar-YTD close-to-close benchmark instead.
    b=bm.close.pct_change(fill_method=None).reindex(days)
    assert np.isfinite(b).all();b.rename('benchmark_return').to_frame().to_parquet(OUT/'data/benchmark.parquet')
    protocol=dict(start=START,end=END,initial_cash=1000000,fresh_start=True,no_2025_positions=True,quarterly_models=audits,
        sources=sources,protected=protected,annual_choice=choice,fee_policy=read(new.OUT/'protocol.json')['fee_policy'],
        rules=read(new.OUT/'protocol.json')['cooldown'],accounts=10,
        benchmark_basis='H00300 prior year-end close to cutoff close; official opens unavailable. Includes first overnight whereas fresh cash strategy enters at first open.',
        limitations='Historical researched period; not unseen test. Quarterly fits may use mature 2026 outcomes. Ideal next-open fills, no additional slippage. Financial historical revision archive incomplete. Published fees not exact verified simulator bills.')
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==protocol
    else:save(OUT/'protocol.json',protocol)
    save(OUT/'data_complete.json',dict(artifacts={str(f):digest(f) for f in (OUT/'data').rglob('*') if f.is_file()}))
    return calendar

def run():
    import pandas as pd
    from paper_futu.top5_buffer_report import metrics,trade_metrics
    if (OUT/'data_complete.json').exists():
        p=read(OUT/'protocol.json');fees.verify(p['sources']);fees.verify(p['protected']);fees.verify(read(OUT/'data_complete.json')['artifacts'])
        calendar=pd.DatetimeIndex(pd.to_datetime((SOURCE/'provider/calendars/day.txt').read_text().splitlines()))
    else:calendar=prepare()
    p=read(OUT/'protocol.json');days=calendar[(calendar>=START)&(calendar<=END)]
    panel=pd.read_parquet(OUT/'data/market.parquet').reset_index()
    market={d:g.set_index('instrument').to_dict('index') for d,g in panel.groupby('datetime')}
    pred=dict(tuple(pd.read_parquet(OUT/'data/predictions.parquet').reset_index().groupby('execution_date')))
    benchmark=pd.read_parquet(OUT/'data/benchmark.parquet').benchmark_return
    rows=[];checks={};curves={};months=[]
    for rank,strong,threshold in [(65,2,t) for t in ('10_5','8_4')]+[(80,1,t) for t in new.THRESHOLDS]:
        for mode in ('full','staged'):
            name=f'r{rank}_s{strong}_{mode}_{threshold}';d=OUT/'accounts'/name;d.mkdir(parents=True,exist_ok=True)
            if not (d/'complete.json').exists():
                a=new.make_account(mode,threshold) if rank==80 else old.make_account(mode,*new.THRESHOLDS[threshold])
                for step,day in enumerate(days):
                    prev=calendar[calendar.get_loc(day)-1];r=pred[day];assert r.datetime.eq(prev).all()
                    a.step(str(day.date()),step,dict(zip(r.instrument,r.prediction)),market[day],
                        {c:q['close'] for c,q in market[prev].items()},signal_day=str(prev.date()))
                a.reconcile();opened=a.open_episodes(str(days[-1].date()),len(days)-1)
                for n,records in dict(daily=a.daily,orders=a.orders,holdings=a.holdings,events=a.events,episodes=a.episodes,open_episodes=opened).items():
                    pd.DataFrame(records).to_parquet(d/f'{n}.parquet',index=False)
                save(d/'complete.json',dict(protocol_sha256=digest(OUT/'protocol.json'),artifacts={str(f):digest(f) for f in d.glob('*.parquet')}))
            done=read(d/'complete.json');assert done['protocol_sha256']==digest(OUT/'protocol.json');fees.verify(done['artifacts'])
            daily,orders,held,closed,opened=[pd.read_parquet(d/f'{n}.parquet') for n in ('daily','orders','holdings','episodes','open_episodes')]
            for f in (daily,orders,held):f['date']=pd.to_datetime(f.date)
            assert pd.DatetimeIndex(daily.date).equals(days)
            assert orders.date.min()>=pd.Timestamp(START) and orders.date.max()<=pd.Timestamp(END)
            checks[name]=fees.ledger_audit(daily,orders,held,closed,opened,10)
            checks[name]['reentries_checked']=old.audit_cooldown(orders,closed,calendar)
            info=dict(name=name,rank_exit=rank,strong_rank=strong,mode=mode,threshold=threshold)
            rows.append(dict(**info,**metrics(daily,benchmark),**trade_metrics(closed,opened),fills=len(orders),
                mean_holding_days=float(closed.holding_days.mean()),net_profit=float(daily.nav.iloc[-1]-1e6)))
            for month,g in daily.groupby(daily.date.dt.to_period('M')):
                months.append(dict(**info,month=str(month),**metrics(g,benchmark)))
            curves[name]=daily;print('COMPLETE',name,round(daily.nav.iloc[-1],2),flush=True)
    table=pd.DataFrame(rows);table.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(months).to_csv(OUT/'monthly.csv',index=False,encoding='utf-8-sig')
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,9),sharex=True)
    for mode,ax in zip(('full','staged'),axes):
        for name,f in curves.items():
            if f'_{mode}_' in name:ax.plot(f.date,f.nav/1e6,label=name)
        ax.plot(days,(1+benchmark).cumprod(),label='H00300',color='black',linestyle='--');ax.legend(fontsize=8);ax.set_title(mode);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'nav.png',dpi=150);plt.close(fig)
    display=table[['rank_exit','strong_rank','mode','threshold','total_return','cagr','max_drawdown','geometric_excess','closed','fills','mean_holding_days','fees','win_rate','avg_gain','avg_loss']].copy()
    for c in ('total_return','cagr','max_drawdown','geometric_excess','win_rate','avg_gain','avg_loss'):display[c]=display[c].map(lambda v:f'{v:.2%}')
    body='<html lang="zh"><meta charset="utf-8"><style>body{font:16px/1.7 system-ui;margin:30px}table{border-collapse:collapse;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>2026年Top10排名缓冲与止盈止损对照</h1>'
    body+=f'<p>{START}至{END}，{len(days)}个交易日，年初100万元现金独立起跑，不继承2025持仓。区间收益为实际回放收益，cagr只是年化折算，并非2026全年收益。H00300上年末收盘至截止日的全收益为{((1+benchmark).prod()-1):.2%}。官方开盘值缺失，因此基准包含年初首个隔夜，策略从首日开盘入场；超额和信息比率保留这一暴露差异，非精确开盘起点对照。</p>'
    body+='<p>沿用H10季度模型，2026年树数仅使用2023—2025验证折；季度拟合可以使用此前已成熟的2026标签，并非全年冻结模型。全部10组同一起点、费用与交易假设。2026数据此前已经观察，不能称为独立测试。</p>'
    body+='<div style="overflow-x:auto">'+display.round(2).to_html(index=False)+'</div><img src="nav.png">'
    body+='<p>closed为完整结束周期，fills为买卖成交笔数。未退出仓位按末日收盘估值，未强制平仓；胜率和平均盈亏仅统计已结束周期。费用已扣；现行公开费用假设不是经逐笔账单验证的模拟盘实际收费。日线理想开盘成交，跳空可使亏损超过触发线，缺少完整历史财务修订档案。</p><p>排名与止盈止损规则保持上一轮；不修改模拟盘、不重训、不重新调参。<a href="comparison.csv">完整指标</a> · <a href="monthly.csv">逐月指标</a> · <a href="verification.json">核验</a></p></html>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    fees.verify(p['sources']);fees.verify(p['protected'])
    save(OUT/'verification.json',dict(status='passed',checks=checks,quarterly_models=4,accounts=10,source_and_live_unchanged=True))
    save(OUT/'report_manifest.json',dict(artifacts={str(f):digest(f) for f in OUT.rglob('*') if f.is_file() and f.name not in ('completion.json','report_manifest.json','run.lock')}))
    save(OUT/'completion.json',dict(status='complete',accounts=10,manifest_sha256=digest(OUT/'report_manifest.json')))
    print(display.round(2).to_string(index=False),flush=True)

if __name__=='__main__':
    import msvcrt
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'run.lock').open('a+b') as f:
        f.seek(0);f.write(b'0');f.flush();f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
        try:run()
        finally:f.seek(0);msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
