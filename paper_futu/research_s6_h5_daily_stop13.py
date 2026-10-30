"""Fresh H5 quarterly models and daily Top10/head4_60/13% stop research.

All artifacts are isolated. No broker entry point or deployment write is used.
"""
from __future__ import annotations

import argparse
from functools import partial
import html
import importlib
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

from paper_futu import research_s6_stop_loss as stop_engine

ROOT,DATA,START,END=stop_engine.ROOT,stop_engine.DATA,stop_engine.START,stop_engine.END
OUT=ROOT/"reports/experiments/s6_h5_qreg_r1_top10_head4_60_stop13_asof_20261013"
read,save,digest=stop_engine.read,stop_engine.save,stop_engine.digest
ENV=stop_engine.ENV
TREES=[10,20,40,80,160,320,640]
PARAMS=dict(objective="regression",metric="l2",boosting_type="gbdt",learning_rate=.03,
    num_leaves=15,max_depth=-1,min_child_samples=300,subsample=.8,subsample_freq=1,
    colsample_bytree=.8,reg_alpha=1.,reg_lambda=5.,verbosity=-1,n_jobs=4,random_state=42)
PORTFOLIO=dict(top_k=10,top_tier_size=4,top_tier_weight=.6,n_drop=6,risk_degree=.95)


def calendar():
    import pandas as pd
    return pd.DatetimeIndex(pd.to_datetime((DATA/"provider/calendars/day.txt").read_text().splitlines()))


def periods(sessions=None,start=START,end=END):
    import pandas as pd
    sessions=calendar() if sessions is None else sessions
    result=[]
    for year in range(pd.Timestamp(start).year,pd.Timestamp(end).year+1):
        for month in (1,4,7,10):
            boundary=pd.Timestamp(year,month,1)
            dates=sessions[(sessions>=boundary)&(sessions<boundary+pd.DateOffset(months=3))&(sessions>=start)&(sessions<=end)]
            if len(dates):
                prior=sessions[sessions<dates[0]]
                if not len(prior): raise ValueError("Missing pre-quarter cutoff")
                result.append(dict(year=year,cutoff=str(prior[-1].date()),start=str(dates[0].date()),end=str(dates[-1].date())))
    return result


def mature(frame,start,cutoff):
    from paper_futu.final_ablation_core import mature as filtered
    result=filtered(frame,start,cutoff)
    assert result.label_end_date.max()<=__import__('pandas').Timestamp(cutoff)
    return result


def binding():
    p=read(OUT/"protocol.json")
    for path,value in p["implementation"].items():
        if digest(path)!=value: raise ValueError(f"Code binding changed: {path}")
    return p


def cached(name):
    path=OUT/f"tasks/{name}.json"
    if not path.exists(): return None
    result=read(path)
    if result["status"]!="complete": return None
    assert result["protocol_sha256"]==digest(OUT/"protocol.json")
    for path,value in result["artifacts"].items():
        if digest(path)!=value: raise ValueError(f"Cache changed: {path}")
    return result["result"]


def task(name,execute):
    binding()
    result=cached(name)
    if result is not None: return result
    path=OUT/f"tasks/{name}.json"
    state=dict(status="running",pid=os.getpid(),started=time.time(),protocol_sha256=digest(OUT/"protocol.json"))
    save(path,state)
    try:
        result,files=execute()
        state.update(status="complete",elapsed=time.time()-state["started"],result=result,
            artifacts={str(f):digest(f) for f in files})
        save(path,state)
        print(f"COMPLETE {name} {state['elapsed']:.1f}s",flush=True)
        return result
    except BaseException as error:
        state.update(status="failed",error=repr(error),elapsed=time.time()-state["started"])
        save(path,state)
        raise


def audit():
    from src.factor_library import ENHANCED_FEATURE_NAMES
    snapshot=read(DATA/"manifest.json")
    sources={str(DATA/name):value for name,value in snapshot["artifacts"].items()
        if name.startswith("provider") or name in ("raw_features.parquet","labels_h5.parquet",
            "financial_events.parquet","financial_alignment.parquet","benchmark_audit.json")}
    for name in ("raw_features.parquet","labels_h5.parquet"):
        if str(DATA/name) not in sources: raise ValueError("Missing frozen training source")
    sources[str(DATA/"manifest.json")]=digest(DATA/"manifest.json")
    template=stop_engine.SOURCE/"predictions/history_QREG.parquet"
    expected=read(stop_engine.SOURCE/"streams.json")["streams"]["history_QREG"]["sha256"]
    sources[str(template)]=expected
    config=ROOT/"configs/experiments/s6_head6_40_research_main.json"
    sources[str(config)]=digest(config)
    deployed=read(config)
    sources[deployed["model_path"]]=deployed["model_sha256"]
    for path,value in sources.items():
        if digest(path)!=value: raise ValueError(f"Frozen source changed: {path}")
    paths=[Path(__file__),Path(stop_engine.__file__)]
    paths += [ROOT/"paper_futu"/n for n in ("final_ablation_core.py","final_ablation_account.py","s6_explicit_account.py","overfit_four_core.py")]
    paths += [ROOT/"src"/n for n in ("factor_library.py","portfolio_strategy.py","compare_models.py","runtime_budget.py")]
    p=dict(model="S6_QREG_H5",snapshot_id=snapshot["snapshot_id"],start=START,end=END,
        features=ENHANCED_FEATURE_NAMES,input="raw",target="daily cross-sectional future return rank minus 0.5",
        horizon=5,label="Open(T+6)/Open(T+1)-1",retrain_months=3,training_start="2010-01-01",
        parameters=PARAMS,candidate_trees=TREES,selection="preceding three complete calendar-year folds; select anew for H5; hold tree count within execution year",
        validation_years=list(range(2019,2026)),periods=periods(),portfolio=PORTFOLIO,
        rebalance=1,offsets=[0],stop_threshold=.13,initial_cash=1e6,
        stop_rule="previous completed close vs moving average purchase cost including buy fees; next tradable open exit; no same-day reentry",
        fees=dict(buy=.0005,sell=.0015,minimum=5),template_path=str(template),
        runtime=dict(max_workers=1,threads=4,startup_free_gib=1.5,loaded_free_gib=.8),
        sources=sources,implementation={str(f):digest(f) for f in paths},
        limitations=["historically reused evaluation interval, not independent test", "ideal daily open fills",
            "old sealed round-lot sizing", "uniform +/-9.5% price filter", "quarterly fitting may use earlier observations from the same year only after labels mature",
            "joint horizon/holding/weight/cadence/stop changes are not a one-variable causal ablation"])
    path=OUT/"protocol.json"
    if path.exists() and read(path)!=p: raise ValueError("Protocol mismatch; preserve prior experiment")
    save(path,p)
    print("Source audit passed",flush=True)


def load_frame():
    import numpy as np
    import pandas as pd
    from src.runtime_budget import require_memory
    p=binding()
    require_memory(1.5,"H5 sequential training startup")
    features=pd.read_parquet(DATA/"raw_features.parquet",columns=p["features"])
    labels=pd.read_parquet(DATA/"labels_h5.parquet")
    if features.index.has_duplicates or labels.index.has_duplicates or not features.index.equals(labels.index):
        raise ValueError("Invalid or unmatched stock/date training sample")
    if np.isinf(features.to_numpy()).any(): raise ValueError("Infinite feature")
    result=features.join(labels)
    result["target"]=result.rank_target
    require_memory(.8,"H5 loaded training safety floor")
    return result


def data_check():
    def execute():
        import numpy as np
        import pandas as pd
        from paper_futu.overfit_four_core import verify_forward_labels
        checks=verify_forward_labels(5)
        frame=load_frame()
        # A future-data perturbation must not change any quarterly supervised slice.
        audits=[]
        for period in binding()["periods"]:
            training=mature(frame,"2010-01-01",period["cutoff"])
            audits.append(dict(cutoff=period["cutoff"],rows=len(training),
                signal_end=str(training.index.get_level_values("datetime").max().date()),
                label_end_max=str(training.label_end_date.max().date())))
        cutoff="2025-12-31"
        before=mature(frame,"2010-01-01",cutoff)
        digest_before=pd.util.hash_pandas_object(before,index=True).sum()
        future=frame.index.get_level_values("datetime")>pd.Timestamp(cutoff)
        frame.loc[future,binding()["features"]]=12345.
        frame.loc[frame.label_end_date>pd.Timestamp(cutoff),["label","target"]]=98765.
        after=mature(frame,"2010-01-01",cutoff)
        assert digest_before==pd.util.hash_pandas_object(after,index=True).sum()
        result=dict(label_recomputation=checks,boundaries=audits,future_data_invariance=True,
            training_rows=len(frame),duplicate_index=False,infinite_features=False)
        path=OUT/"data_audit.json"; save(path,result)
        return result,[path]
    return task("data",execute)


def fold(year):
    if year not in range(2019,2026): raise ValueError("Validation year outside protocol")
    def execute():
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
        from paper_futu.final_ablation_core import daily_ic
        frame=load_frame()
        train=mature(frame,"2010-01-01",f"{year-1}-12-31")
        valid=mature(frame,f"{year}-01-01",f"{year}-12-31")
        features=binding()["features"]
        del frame
        model=lgb.LGBMRegressor(**PARAMS,n_estimators=max(TREES))
        model.fit(train[features],train.target)
        rows=[]
        for count in TREES:
            score=pd.Series(model.predict(valid[features],num_iteration=count,num_threads=4),index=valid.index)
            ic=daily_ic(score,valid.label)
            if not np.isfinite(ic.rank_ic.mean()): raise ValueError("Invalid fold IC")
            rows.append(dict(parameter_id="qreg_h5",trees=count,num_leaves=15,validation_year=year,
                mean_rank_ic=float(ic.rank_ic.mean()),mean_ic=float(ic.ic.mean()),
                train_rows=len(train),valid_rows=len(valid),train_label_end_max=str(train.label_end_date.max().date()),
                valid_label_end_max=str(valid.label_end_date.max().date())))
        path=OUT/f"folds/{year}.json"; save(path,rows)
        return dict(records=rows),[path]
    return task(f"fold_{year}",execute)


def select():
    import pandas as pd
    from paper_futu.final_ablation_core import choose_candidates
    result={}
    for year in range(2022,2027):
        rows=[]
        for validation_year in range(year-3,year):
            fold_result=cached(f"fold_{validation_year}")
            if fold_result is None: raise ValueError("Missing validation fold")
            rows += fold_result["records"]
        ranked=choose_candidates(pd.DataFrame(rows))
        result[str(year)]=dict(trees=int(ranked.iloc[0].trees),fold_years=list(range(year-3,year)),
            selected_mean_rank_ic=float(ranked.iloc[0].mean_rank_ic),ranking=ranked.to_dict(orient="records"))
    save(OUT/"selected_trees.json",result)
    return result


def fit(cutoff):
    def execute():
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
        from paper_futu.final_ablation_core import daily_ic
        p=binding(); period=next(x for x in p["periods"] if x["cutoff"]==cutoff)
        selection=read(OUT/"selected_trees.json")[str(period["year"])]
        assert max(selection["fold_years"])<period["year"]
        frame=load_frame()
        train=mature(frame,"2010-01-01",cutoff)
        model=lgb.LGBMRegressor(**PARAMS,n_estimators=selection["trees"])
        model.fit(train[p["features"]],train.target)
        directory=OUT/f"models/{cutoff}"; directory.mkdir(parents=True,exist_ok=True)
        model_path=directory/"model.txt"; model.booster_.save_model(str(model_path))
        loaded=lgb.Booster(model_file=str(model_path))
        sample=train.iloc[::max(1,len(train)//2000)][p["features"]]
        np.testing.assert_allclose(model.predict(sample),loaded.predict(sample,num_threads=4),atol=1e-12,rtol=0)
        template=pd.read_parquet(p["template_path"],columns=["execution_date"])
        pred=template.loc[template.execution_date.between(period["start"],period["end"])].copy()
        pred["prediction"]=loaded.predict(frame.reindex(pred.index)[p["features"]],num_threads=4)
        pred=pred.join(frame[["label","rank_target","label_end_date"]])
        pred["model_cutoff"]=pd.Timestamp(cutoff)
        assert (pred.execution_date>pred.model_cutoff).all()
        assert (pred.index.get_level_values("datetime")>=pd.Timestamp(cutoff)).all()
        assert np.isfinite(pred.prediction).all() and not pred.index.has_duplicates
        pred_path=directory/"prediction.parquet"; pred.to_parquet(pred_path)
        available=pred.loc[pred.label_end_date.le(pd.Timestamp(END))]
        ic=daily_ic(available.prediction,available.label)
        ic_path=directory/"mature_ic.parquet"; ic.to_parquet(ic_path)
        result=dict(**period,trees=selection["trees"],train_rows=len(train),
            train_signal_end=str(train.index.get_level_values("datetime").max().date()),
            label_end_max=str(train.label_end_date.max().date()),
            training_fingerprint=int(pd.util.hash_pandas_object(train,index=True).sum()),
            model_path=str(model_path),model_sha256=digest(model_path),reload_equal=True,
            prediction_rows=len(pred),prediction_path=str(pred_path),
            mature_rank_ic=float(ic.rank_ic.mean()) if np.isfinite(ic.rank_ic.mean()) else None)
        path=directory/"metadata.json"; save(path,result)
        return result,[model_path,pred_path,ic_path,path]
    return task(f"fit_{cutoff}",execute)


def assemble():
    def execute():
        import numpy as np
        import pandas as pd
        p=binding(); parts=[]
        for period in p["periods"]:
            fitted=cached(f"fit_{period['cutoff']}")
            if fitted is None: raise ValueError("Missing quarterly model")
            parts.append(pd.read_parquet(fitted["prediction_path"]))
        pred=pd.concat(parts).sort_index()
        template=pd.read_parquet(p["template_path"],columns=["execution_date"])
        pd.testing.assert_index_equal(pred.index,template.index)
        pd.testing.assert_series_equal(pred.execution_date,template.execution_date)
        assert not pred.index.has_duplicates and np.isfinite(pred.prediction).all()
        # Labels are diagnostic only, including immature rows required for live-like prediction.
        assert pred.loc[pred.execution_date.eq(END),"prediction"].notna().all()
        path=OUT/"predictions/history_H5_QREG.parquet"; path.parent.mkdir(exist_ok=True); pred.to_parquet(path)
        result=dict(path=str(path),sha256=digest(path),rows=len(pred),models=len(parts))
        save(OUT/"streams.json",result)
        return result,[path,OUT/"streams.json"]
    return task("assemble",execute)


def account():
    def execute():
        import numpy as np
        import pandas as pd
        from src.runtime_budget import require_memory
        from paper_futu.s6_explicit_account import init_qlib
        from src.compare_models import enrich_daily_report
        from qlib.data import D
        from qlib.backtest.executor import SimulatorExecutor
        from qlib.backtest.decision import Order
        from qlib.contrib.evaluate import backtest_daily
        require_memory(1.5,"H5 daily account startup")
        stream=cached("assemble"); assert stream
        init_qlib(SimpleNamespace(DATA=DATA,OUT=OUT,NAME=OUT.name))
        Market,Strategy=stop_engine.classes()
        pred=pd.read_parquet(stream["path"],columns=["prediction"])
        sessions=pd.DatetimeIndex(D.calendar(freq="day"))
        days=sessions[(sessions>=START)&(sessions<=END)]
        signals=sessions[sessions.get_indexer(days)-1]
        signal=pred.loc[pred.index.get_level_values("datetime").isin(signals)]
        assert signal.index.get_level_values("datetime").nunique()==len(signals)
        market=Market(freq="day",start_time=signals[0],end_time=END,
            codes=sorted(pred.index.get_level_values("instrument").unique()),deal_price="open",
            open_cost=.0005,close_cost=.0015,min_cost=5.,trade_unit=100,
            limit_threshold=("($open / Ref($close, 1) - 1) >= 0.095","($open / Ref($close, 1) - 1) <= -0.095"))
        market.reset_audit(0.,10)
        strategy=Strategy(signal=signal,only_tradable=True,threshold=.13,**PORTFOLIO)
        module=importlib.import_module("qlib.backtest.backtest")
        with patch.object(module,"tqdm",partial(module.tqdm,disable=True)):
            raw,positions=backtest_daily(start_time=signals[0],end_time=END,strategy=strategy,
                executor=SimulatorExecutor(time_per_step="day",generate_portfolio_metrics=True,verbose=False,
                    trade_type=SimulatorExecutor.TT_SERIAL),account=1e6,benchmark="SH000300",exchange_kwargs={"exchange":market})
        report=enrich_daily_report(raw.reindex(days),"H5_QREG_R1_Top10_Stop13")
        orders=pd.DataFrame(market.journal); trades=orders.loc[orders.notional.gt(0)]
        assert trades.datetime.isin(days).all()
        assert trades.loc[trades.stop_loss,"direction"].eq(Order.SELL).all()
        totals=trades.groupby("datetime")[["notional","fee"]].sum().reindex(days,fill_value=0.)
        prior=report.account.shift(1,fill_value=1e6)
        report["fees_cash"],report["traded_notional"]=totals.fee,totals.notional
        report["turnover"]=totals.notional/prior
        holding,pnl,previous=[],[],{}
        for day in days:
            position=positions[day]; market.check_holding_limit(position,day,"daily_close")
            current={}
            for stock in position.get_stock_list():
                qty,price=position.get_stock_amount(stock),position.get_stock_price(stock)
                current[stock]=qty*price
                holding.append(dict(datetime=day,instrument=stock,adjusted_quantity=qty,
                    adjusted_close=price,market_value=qty*price,cash=position.get_cash()))
            today=trades.loc[trades.datetime.eq(day)]
            flow=today.assign(flow=np.where(today.direction.eq(Order.BUY),today.notional,-today.notional)).groupby("instrument").flow.sum()
            for stock in set(current)|set(previous)|set(flow.index):
                pnl.append(dict(datetime=day,instrument=stock,pnl_before_fees=current.get(stock,0)-previous.get(stock,0)-flow.get(stock,0)))
            previous=current
        holding,pnl=pd.DataFrame(holding),pd.DataFrame(pnl)
        report["holdings_count"]=holding.groupby("datetime").size().reindex(days,fill_value=0)
        values=holding.groupby("datetime").market_value.sum().reindex(days,fill_value=0)
        gross=pnl.groupby("datetime").pnl_before_fees.sum().reindex(days,fill_value=0)
        for actual,expected in ((values+report.cash,report.account),(prior+gross-totals.fee,report.account),
            ((1+report.net_return).cumprod()*1e6,report.account),(report.cost*prior,totals.fee)):
            np.testing.assert_allclose(actual,expected,atol=1e-4,rtol=1e-10)
        assert report.cash.min()>=-1e-5 and report.holdings_count.max()<=10
        baseline=pd.read_parquet(stop_engine.OUT/"accounts/baseline_0/daily.parquet")
        pd.testing.assert_index_equal(report.index,baseline.index)
        report["bench"],report["benchmark_nav"]=baseline.bench,baseline.benchmark_nav
        report["active_return"]=report.net_return-report.bench
        report["relative_excess_nav"]=report.portfolio_nav/report.benchmark_nav
        directory=OUT/"account"; directory.mkdir(exist_ok=True); files=[]
        for name,frame in (("daily",report),("orders",orders),("holdings",holding),("contributions",pnl)):
            path=directory/f"{name}.parquet"; frame.to_parquet(path,index=name=="daily"); files.append(path)
        path=directory/"stop_events.json"; save(path,strategy.stop_events); files.append(path)
        result=dict(periods={"all":stop_engine.metrics(report),**{str(y):stop_engine.metrics(g) for y,g in report.groupby(report.index.year)}},
            stop_fills=int(trades.stop_loss.sum()),unresolved_stops=sorted(set(strategy.pending)&set(positions[days[-1]].get_stock_list())),
            account_audit_passed=True)
        path=directory/"metrics.json"; save(path,result); files.append(path)
        return result,files
    return task("account",execute)


def report_results():
    import numpy as np
    import pandas as pd
    from paper_futu.final_ablation_core import daily_ic
    p=binding()
    for path,value in p["sources"].items():
        if digest(path)!=value: raise ValueError(f"Input changed: {path}")
    data=cached("data"); stream=cached("assemble"); result=cached("account")
    assert data and stream and result
    selected=read(OUT/"selected_trees.json")
    models=[cached(f"fit_{q['cutoff']}") for q in p["periods"]]
    assert all(m and m["reload_equal"] and m["label_end_max"]<=m["cutoff"] for m in models)
    pred=pd.read_parquet(stream["path"])
    mature_pred=pred.loc[pred.label_end_date.le(pd.Timestamp(END))]
    ic=daily_ic(mature_pred.prediction,mature_pred.label)
    ic.to_parquet(OUT/"mature_ic.parquet")
    yearly_ic={str(y):float(g.rank_ic.mean()) for y,g in ic.groupby(ic.index.year)}
    reference=read(stop_engine.OUT/"comparison.json")["cases"]["baseline"]["periods"]
    summary=dict(status="complete",account=result,selected_trees=selected,models=models,
        mature_rank_ic=float(ic.rank_ic.mean()),yearly_rank_ic=yearly_ic,reference_top20=reference,
        sources_unchanged=True,protocol_sha256=digest(OUT/"protocol.json"))
    save(OUT/"comparison.json",summary)
    def table(headers,rows):
        return "<table><tr>"+"".join(f"<th>{html.escape(str(v))}</th>" for v in headers)+"</tr>"+"".join(
            "<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in rows)+"</table>"
    rows=[]
    for key,label,fmt in (("cagr","复合年化",".2%"),("total_return","累计收益",".2%"),
        ("max_drawdown","最大回撤",".2%"),("fees","全期费用（元）",",.2f"),
        ("mean_daily_turnover","平均每日双边换手",".2%"),("mean_holdings","平均持股数",".2f")):
        rows.append([label,format(reference["all"][key],fmt),format(result["periods"]["all"][key],fmt)])
    years=[[y,f"{reference[y]['total_return']:.2%}",f"{result['periods'][y]['total_return']:.2%}",
        f"{result['periods'][y]['max_drawdown']:.2%}",selected[y]["trees"],f"{yearly_ic[y]:.4f}"] for y in map(str,range(2022,2027))]
    text=f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>S6 H5季度重训／每日Top10／13%止损</title>
<style>body{{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#173449}}table{{width:100%;border-collapse:collapse;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #dce5ed;text-align:right}}th{{background:#edf4f8}}th:first-child,td:first-child{{text-align:left}}</style>
<h1>S6新实验：预测5日、季度重训、每日调仓</h1><p>2022-01-04至2026-10-13。15个原始因子输入、未来5日开盘到开盘收益横截面排名目标；每年用前三个完整年度验证选择树数，当年四个季度分别从头拟合成熟样本。固定L1=1、L2=5、15叶、叶子最少300样本，种子42、4线程。</p>
<p>Top10、前4只合计60%股票仓位、后6只40%，股票目标95%，普通替换上限6只。每日调仓只有一个周期起点，不复制相同日历制造起点样本。单股13%成本止损每日收盘确认，下一可交易日开盘退出，不保证最大成交损失13%；禁止当天止损后回买，次日正常调仓可再次入选。</p>
<h2>全期含成本</h2>{table(['指标','原S6 Top20（5起点均值）','新H5每日Top10（单条账户）'],rows)}
<h2>逐年收益、风险和模型诊断</h2>{table(['年份','原S6收益','新版本收益','新版本年内回撤','树数','成熟标签Rank IC'],years)}
<p>2026截至10月13日。逐年收益非年化；全期复合年化按252交易日计算。原策略多起点均值与新策略单条账户仅作描述性参照，不能推断显著性。本次同时改变预测期、持仓、权重、频率及止损，不能单独归因于某一项。</p>
<h2>核验</h2><p>7个共享年度验证拟合、20个季度最终拟合、20个模型重载预测一致；标签均按真实label_end_date成熟过滤。预测样本覆盖与原S6一致，最新未成熟标签日期仍生成预测。账户净值、现金、费用、最多10只持股已复算。止损卖出{result['stop_fills']}次。模拟盘及部署未修改。</p>
<p>仍采用日线理想开盘成交、旧整手规则、统一±9.5%涨跌幅过滤；买入0.05%、卖出0.15%、最低5元。财务披露日期沿用固定快照，不新增2026年10月13日后的信息。2022—2026已反复参与研究，不是新的独立测试。协议SHA256：{digest(OUT/'protocol.json')}。</p></html>"""
    (OUT/"report.html").write_text(text,encoding="utf-8")
    save(OUT/"report_manifest.json",{str(path):digest(path) for path in
        (OUT/"comparison.json",OUT/"report.html",OUT/"mature_ic.parquet",OUT/"selected_trees.json")})
    print(result,flush=True)


def run_child(stage,name=None):
    args=[sys.executable,"-u","-m","paper_futu.research_s6_h5_daily_stop13","--stage",stage]
    if name is not None: args += ["--name",str(name)]
    logdir=OUT/"logs"; logdir.mkdir(exist_ok=True)
    save(OUT/"status.json",dict(state="running",stage=stage,name=name,pid=os.getpid(),updated=time.time()))
    with (logdir/f"{stage}_{name or 'main'}.log").open("a",encoding="utf-8") as log:
        completed=subprocess.run(args,cwd=ROOT,env={**os.environ,**ENV},stdout=log,stderr=subprocess.STDOUT)
    if completed.returncode:
        save(OUT/"status.json",dict(state="failed",stage=stage,name=name,exit_code=completed.returncode))
        raise SystemExit(completed.returncode)
    print(f"Finished {stage} {name or ''}",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["audit","data","fold","select","fit","assemble","account","report","all"],default="all")
    parser.add_argument("--name")
    args=parser.parse_args()
    if args.stage=="audit": audit()
    elif args.stage=="data": data_check()
    elif args.stage=="fold": fold(int(args.name))
    elif args.stage=="select": select()
    elif args.stage=="fit": fit(args.name)
    elif args.stage=="assemble": assemble()
    elif args.stage=="account": account()
    elif args.stage=="report": report_results()
    else:
        from filelock import FileLock
        OUT.mkdir(parents=True,exist_ok=True)
        with FileLock(str(OUT/"controller.lock"),timeout=0):
            audit()
            run_child("data")
            for year in range(2019,2026): run_child("fold",year)
            run_child("select")
            for period in binding()["periods"]: run_child("fit",period["cutoff"])
            for stage in ("assemble","account","report"): run_child(stage)
            save(OUT/"status.json",dict(state="complete",updated=time.time()))


if __name__=="__main__": main()
