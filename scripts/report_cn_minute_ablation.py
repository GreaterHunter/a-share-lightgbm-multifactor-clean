"""Prediction diagnostics, paired block intervals and independent ledger audits."""
from __future__ import annotations
import math
import gc
import time
import numpy as np
import pandas as pd
from paper_futu.research_cn_minute_ablation import (
    OUT, BASE, SOURCE, END, FEATURES, REUSED, read, save, digest, verify, account_key, model_dir,
)

LABELS={'M0_daily':'M0 日频15','M1_clock':'M1 加时段进度','M2_return':'M2 加分钟收益','M3_volatility':'M3 加分钟波动','M4_position':'M4 加量价位置','M5_full':'M5 全部29因子'}


def block_indices(n,length,seed,replications=2000):
    rng=np.random.default_rng(seed);starts=rng.integers(0,n,size=(replications,math.ceil(n/length)))
    return ((starts[:,:,None]+np.arange(length)[None,None,:])%n).reshape(replications,-1)[:,:n]


def diagnose():
    verify()
    if (OUT/'diagnostics_complete.json').exists():return
    folder=OUT/'evaluation_labels';folder.mkdir(exist_ok=True)
    for source in sorted((BASE/'market_parts').glob('*.parquet')):
        target=folder/source.name
        if target.exists():
            if read(target.with_suffix('.json'))['sha256']!=digest(target):raise RuntimeError('Evaluation label changed')
            continue
        m=pd.read_parquet(source,columns=['open'])
        label=m.open.shift(-482)/m.open.shift(-2)-1
        end=m.index.to_series().shift(-482)
        if end.dropna().max()>pd.Timestamp(END+' 23:59:59'):raise RuntimeError('Evaluation label crossed cutoff')
        f=pd.DataFrame(dict(instrument=source.stem,label=label,label_end=end),index=m.index)
        f=f.replace([np.inf,-np.inf],np.nan);f.to_parquet(target)
        save(target.with_suffix('.json'),dict(sha256=digest(target)))
    ic_rows=[];coverage=[];overlap=[]
    for path in sorted((OUT/'predictions').glob('*.parquet')):
        if read(path.with_suffix('.json'))['sha256']!=digest(path):raise RuntimeError('Prediction changed')
        month=pd.Period(path.stem,freq='M');lo,hi=month.start_time,month.end_time
        p=pd.read_parquet(path).reset_index()
        labels=pd.concat([pd.read_parquet(f,filters=[('datetime','>=',lo),('datetime','<=',hi)]) for f in folder.glob('*.parquet')]).reset_index()
        x=p.merge(labels,on=['datetime','instrument'],how='left',validate='one_to_one')
        if x.label_end.dropna().max()>pd.Timestamp(END+' 23:59:59'):raise RuntimeError('Future evaluation label')
        valid=x.label.notna();coverage.append(dict(month=str(month),predicted=len(x),mature_with_prices=int(valid.sum()),coverage=float(valid.mean())))
        x=x.loc[valid].set_index(['datetime','instrument']);r=x[list(FEATURES)+['label']].groupby(level=0).rank(method='average')
        z=r-r.groupby(level=0).transform('mean');counts=z.label.groupby(level=0).count()
        for model in FEATURES:
            cross=(z[model]*z.label).groupby(level=0).sum()
            denom=np.sqrt(z[model].pow(2).groupby(level=0).sum()*z.label.pow(2).groupby(level=0).sum())
            result=(cross/denom.replace(0,np.nan)).where(counts>=30).rename('rank_ic').reset_index()
            result['model']=model;ic_rows.append(result)
            top=p.sort_values(['datetime',model,'instrument'],ascending=[True,False,True]).groupby('datetime').head(10)
            previous=None;previous_day=None;values=[]
            for stamp,g in top.groupby('datetime',sort=True):
                current=set(g.instrument)
                if previous is not None and stamp.normalize()==previous_day:values.append(len(current&previous)/10.)
                previous=current;previous_day=stamp.normalize()
            overlap.append(dict(model=model,month=str(month),mean_top10_overlap=float(np.mean(values)),pairs=len(values)))
        save(OUT/'progress.json',dict(stage='diagnose',month=str(month),time=time.time()))
        print('DIAGNOSE '+str(month),flush=True)
        del p,labels,x,r,z;gc.collect()
    ic=pd.concat(ic_rows,ignore_index=True);ic.to_parquet(OUT/'rank_ic_intraday.parquet',index=False)
    ic['date']=ic.datetime.dt.normalize()
    daily=ic.groupby(['date','model']).rank_ic.mean().unstack('model')
    daily.to_csv(OUT/'rank_ic_daily.csv',encoding='utf-8-sig')
    annual=ic.groupby([ic.datetime.dt.year.rename('year'),'model']).rank_ic.agg(['mean','std','count']).reset_index()
    quarterly=ic.groupby([ic.datetime.dt.to_period('Q').astype(str).rename('quarter'),'model']).rank_ic.agg(['mean','std','count']).reset_index()
    annual.to_csv(OUT/'rank_ic_yearly.csv',index=False);quarterly.to_csv(OUT/'rank_ic_quarterly.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'label_coverage.csv',index=False);pd.DataFrame(overlap).to_csv(OUT/'top10_stability.csv',index=False)
    pairs=[('M1_clock','M0_daily'),('M5_full','M0_daily')]+[(m,'M1_clock') for m in ('M2_return','M3_volatility','M4_position','M5_full')]
    intervals=[]
    for length in (60,120):
        common=daily.dropna();ix=block_indices(len(common),length,20261030+length)
        for model,baseline in pairs:
            delta=(common[model]-common[baseline]).to_numpy();rep=delta[ix].mean(axis=1)
            intervals.append(dict(model=model,baseline=baseline,metric='daily_rank_ic_difference',extra_bp=0,block_days=length,
                difference=float(delta.mean()),lower=float(np.quantile(rep,.025)),upper=float(np.quantile(rep,.975)),replications=2000))
        for bp in (0,10):
            returns={m:pd.read_parquet(OUT/'accounts'/account_key(dict(model=m,cadence='5m',lag=2,bp=bp))/'daily.parquet').set_index('date').net_return for m in FEATURES}
            frame=pd.DataFrame(returns);idx=block_indices(len(frame),length,20261030+length)
            logs=np.log1p(frame)
            for model,baseline in pairs:
                rp=np.expm1(252*logs[model].to_numpy()[idx].mean(axis=1))-np.expm1(252*logs[baseline].to_numpy()[idx].mean(axis=1))
                observed=np.expm1(252*logs[model].mean())-np.expm1(252*logs[baseline].mean())
                intervals.append(dict(model=model,baseline=baseline,metric='net_cagr_difference',extra_bp=bp,block_days=length,
                    difference=float(observed),lower=float(np.quantile(rp,.025)),upper=float(np.quantile(rp,.975)),replications=2000))
    pd.DataFrame(intervals).to_csv(OUT/'paired_block_intervals.csv',index=False)
    save(OUT/'diagnostics_complete.json',dict(status='complete',intraday_rows=len(ic),daily_rows=len(daily),paired_intervals=len(intervals),
        note='Date-aggregated IC and paired circular time blocks; conditional historical uncertainty, not correction for prior strategy selection'))


def report():
    from scripts.audit_cn_intraday_results import check_account
    from paper_futu.top5_buffer_report import metrics
    import lightgbm as lgb
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    verify();audits={};summary=[];years=[];reasons=[];importance=[]
    records=read(OUT/'train_complete.json')['models']
    if len(records)!=96 or sum(not x['reused'] for x in records)!=64:raise RuntimeError('Missing model fit')
    for record in records:
        r=record['metadata'];cutoff=pd.Timestamp(r['cutoff'])
        if pd.Timestamp(r['label_end_max'])>cutoff or pd.Timestamp(r['signal_max'])>cutoff:raise RuntimeError('Model boundary invalid')
        if digest(record['path'])!=r['model_sha256'] or r['reload_error']>1e-10:raise RuntimeError('Model audit mismatch')
        b=lgb.Booster(model_file=record['path']);g=b.feature_importance(importance_type='gain').astype(float);g=g/g.sum() if g.sum() else g
        for name,value in zip(b.feature_name(),g):importance.append(dict(model=record['model'],year=r['year'],quarter=r['quarter'],feature=name,gain_share=float(value)))
    pd.DataFrame(importance).to_csv(OUT/'feature_importance.csv',index=False)
    benchmark=pd.read_parquet(SOURCE/'benchmark.parquet').benchmark_return
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    for n,row in enumerate(read(OUT/'backtest_complete.json')['accounts'],1):
        key=row['key'];folder=OUT/'accounts'/key
        for f,h in row['artifacts'].items():
            if digest(f)!=h:raise RuntimeError('Account changed')
        audits[key]=check_account(folder,row['bp']);print(f'AUDITED {n}/28 {key}',flush=True)
        d=pd.read_parquet(folder/'daily.parquet');c=pd.read_parquet(folder/'closed.parquet');o=pd.read_parquet(folder/'orders.parquet')
        d['date']=pd.to_datetime(d.date)
        if not pd.DatetimeIndex(d.date).equals(benchmark.index):raise RuntimeError('Unequal evaluation dates')
        calendar=pd.DatetimeIndex(d.date);hold=calendar.get_indexer(pd.to_datetime(c.exit_day))-calendar.get_indexer(pd.to_datetime(c.entry_day))
        spec={k:row[k] for k in ('key','model','cadence','lag','bp')}
        summary.append(dict(**spec,**metrics(d,benchmark),fills=len(o),closed=len(c),rank_exits=int(c.reason.isin(['rank','strong']).sum()),
            win_rate=float(c.pnl.gt(0).mean()),mean_hold=float(np.mean(hold))))
        for y,g in d.groupby(d.date.dt.year):years.append(dict(**spec,year=int(y),**metrics(g,benchmark)))
        for reason,g in c.groupby('reason'):reasons.append(dict(**spec,reason=reason,closed=len(g),pnl=float(g.pnl.sum())))
        if row['bp']==0 and row['cadence']=='5m' and row['lag']==2:
            nav=d.nav.to_numpy()/1e6;axes[0].plot(d.date,nav,label=row['model'])
            axes[1].plot(d.date,nav/np.maximum.accumulate(np.r_[1.,nav])[1:]-1,label=row['model'])
    axes[0].plot(benchmark.index,(1+benchmark).cumprod(),color='black',linestyle='--',label='H00300')
    for ax in axes:ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Net asset value');axes[1].set_ylabel('Drawdown');fig.tight_layout();fig.savefig(OUT/'curves.png',dpi=150);plt.close(fig)
    s=pd.DataFrame(summary);y=pd.DataFrame(years)
    s.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig');y.to_csv(OUT/'yearly.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(reasons).to_csv(OUT/'exit_reasons.csv',index=False)
    save(OUT/'independent_audit.json',dict(status='passed',accounts=audits,models=96,new_fits=64,production_unchanged=True,stop4_baseline_reproduced_exactly=True))
    def table(frame,cols,percent=()):
        f=frame[list(cols)].copy()
        if 'model' in f:f['model']=f.model.map(LABELS)
        for col in percent:f[col]=f[col].map(lambda v:f'{v:.2%}')
        for col in ('fees','mean_hold'):
            if col in f:f[col]=f[col].map(lambda v:f'{v:,.2f}')
        return f.rename(columns=cols).to_html(index=False)
    cols=dict(model='模型',cadence='排名检查频率',lag='执行滞后线数',bp='每边额外摩擦bp',cagr='净年化',max_drawdown='最大回撤',fees='费用元',closed='完整交易次数',rank_exits='排名退出次数')
    main=s[s.cadence.eq('5m')&s.lag.eq(2)]
    other=s[~(s.cadence.eq('5m')&s.lag.eq(2))]
    annual_ic=pd.read_csv(OUT/'rank_ic_yearly.csv').pivot(index='year',columns='model',values='mean')
    intervals=pd.read_csv(OUT/'paired_block_intervals.csv')
    limits=''.join('<li>'+x+'</li>' for x in read(OUT/'protocol.json')['limitations'])
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分钟因子有效性消融</title><style>body{font:16px/1.7 system-ui;max-width:1550px;margin:30px auto;padding:0 20px}table{border-collapse:collapse;font-size:14px}td,th{padding:7px;border:1px solid #ddd}th{background:#eef3f8}.table{overflow:auto}img{max-width:100%}</style><h1>分钟因子有效性：输入、排名换股频率与执行延迟</h1><p>2022—2025，初始100万元；固定Top10、前80退出、新进第1名替换、止盈10%、含费止损4%、T+1、止盈无额外冷却、止损次交易日可重入。共64次新增季度拟合，复用32个模型；28个账户。没有扩大参数网格或切换V10。</p><p>M0＝日频15；M1＝M0＋时段进度；M2＝M1＋6个分钟/相对收益；M3＝M1＋4个分钟波动/振幅；M4＝M1＋量比、VWAP偏离、开盘以来收益；M5＝全部29因子。M1是控制时段信息后的参照。</p><p>各输入使用完全相同的成熟样本、收益排名标签、年度树数和其他训练参数。新增模型没有重新选树数，因此结论只适用于当前固定训练预算，不能证明分钟因子在所有参数、容量或其他预测期限下无效。训练每天抽样4个时点，评价涵盖全部可用5分钟时点。</p><h2>输入主对照</h2><div class="table">'+table(main,cols,('cagr','max_drawdown'))+'</div><img src="curves.png"><h2>排名频率与延迟</h2><p>30分钟／每日仅限制主动排名退出及强势替换，止盈止损和空位补仓仍每5分钟判断。每日排名检查09:45；强势信号相对上一排名检查时点定义。lag2为原等待一根完整5分钟线，lag3额外再延迟5分钟。不同费用会改变含费止损和资金路径，压力测试不是简单扣固定收益。</p><div class="table">'+table(other,cols,('cagr','max_drawdown'))+'</div><h2>成熟H10标签的年度Rank IC</h2>'+annual_ic.to_html(float_format=lambda v:f'{v:.4f}')+'<p>统计推断先按日期聚合，不能把同日48个时点当独立样本。配对区块60/120交易日、2000次重采样；区间不能抵消此前历史选择偏差。</p><details><summary>全部配对差异区间</summary>'+intervals.to_html(index=False,float_format=lambda v:f'{v:.5f}')+'</details><h2>限制与核验</h2><ul>'+limits+'</ul><p>费用沿用项目富途模拟分项费用，未计托管费、未核验券商舍入；绩效基准H00300、无风险年利率1%。逐笔核验28个账户现金、费用、净值、T+1、源分钟报价与量能。原止损4%账户严格复现。2022—2025已反复使用，不能称新独立测试，不自动挑选新冠军。</p><p><a href="comparison.csv">账户比较</a> · <a href="yearly.csv">逐年结果</a> · <a href="paired_block_intervals.csv">差异区间</a> · <a href="rank_ic_daily.csv">每日IC</a> · <a href="top10_stability.csv">排名稳定性</a> · <a href="label_coverage.csv">标签覆盖率</a> · <a href="independent_audit.json">独立审计</a> · <a href="protocol.json">协议</a></p></html>'
    (OUT/'report.html').write_text(page,encoding='utf8')
    verify()
    artifacts={str(f):digest(f) for f in OUT.glob('*') if f.suffix in ('.csv','.html','.png')}
    artifacts[str(OUT/'independent_audit.json')]=digest(OUT/'independent_audit.json')
    save(OUT/'completion.json',dict(status='complete_and_audited',models=96,new_fits=64,accounts=28,artifacts=artifacts,protocol_sha256=digest(OUT/'protocol.json')))
    print(s.to_string(index=False),flush=True)
