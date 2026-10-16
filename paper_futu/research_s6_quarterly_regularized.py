"""One combined S6 quarterly/L1/L2/complexity experiment, on the same frozen data."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from paper_futu import research_s6_2026 as parent

ROOT,DATA = parent.ROOT,parent.DATA
NAME = "s6_quarterly_l1l2_small_asof_20261013"
OUT = ROOT / "reports/experiments" / NAME
MODELS = ROOT / "models/experiments" / NAME
START,END,HISTORY_START = parent.START,parent.END,parent.HISTORY_START
PORTFOLIO,ENV = dict(parent.PORTFOLIO),dict(parent.ENV)
read,save,digest,identity = parent.read,parent.save,parent.digest,parent.identity
CASES = ("QREG",)


def implementation():
    return identity(dict(parent=parent.implementation(),files={name:digest(ROOT / "paper_futu" / name)
        for name in ("research_s6_quarterly_regularized.py","s6_explicit_account.py")}))


def parameters(base):
    return {**base,"reg_alpha":1.,"reg_lambda":5.,"num_leaves":15,"min_child_samples":300,"n_jobs":4}


def protocol():
    p = read(OUT / "protocol.json")
    if p["implementation_sha256"] != implementation():
        raise ValueError("Combined-experiment code changed; review cache compatibility first")
    return p


def check_sources():
    p = protocol()
    parent.check_sources()
    for path,expected in p["sources"].items():
        if digest(path) != expected:
            raise ValueError(f"Combined-experiment source changed: {path}")
    return p


def cached(kind,name):
    path = OUT / "tasks" / f"{kind}_{name}.json"
    if not path.exists(): return None
    task = read(path)
    if task["status"] != "complete": return None
    if task["implementation_sha256"] != implementation() or task["protocol_sha256"] != digest(OUT / "protocol.json"):
        raise ValueError("Combined cache binding changed")
    parent.verify_artifacts(task)
    return task["result"]


def run_task(kind,name,execute):
    result = cached(kind,name)
    if result is not None: return result
    import psutil
    task = dict(kind=kind,name=name,status="running",pid=os.getpid(),created=psutil.Process().create_time(),
        started=time.time(),implementation_sha256=implementation(),protocol_sha256=digest(OUT / "protocol.json"))
    path = OUT / "tasks" / f"{kind}_{name}.json"
    save(path,task)
    try:
        result,artifacts = execute()
        task.update(status="complete",result=result,elapsed=time.time()-task["started"],
            artifacts={str(p):digest(p) for p in artifacts})
        save(path,task)
        print(f"COMPLETE {kind} {name} seconds={task['elapsed']:.1f}",flush=True)
        return result
    except BaseException as error:
        task.update(status="failed",error=repr(error),elapsed=time.time()-task["started"])
        save(path,task)
        raise


def financial_audit():
    import pandas as pd
    events = pd.read_parquet(DATA / "financial_events.parquet")
    alignment = pd.read_parquet(DATA / "financial_alignment.parquet")
    features = pd.read_parquet(DATA / "raw_features.parquet",columns=["book_to_price","earnings_to_price","operating_cashflow_to_revenue"])
    coverage = []
    for day in ("2025-12-31","2026-03-31","2026-06-30","2026-09-30",END):
        f = features.loc[features.index.get_level_values("datetime")==day]
        a = alignment.loc[alignment.datetime.eq(day) & alignment.instrument.isin(f.index.get_level_values("instrument"))]
        if (a.fundamental_available_date > a.datetime).any() or (a.annual_eps_available_date > a.datetime).any():
            raise ValueError("Financial disclosure alignment uses future information")
        missing = a.loc[a.fundamental_report_date.isna(),"instrument"].tolist()
        coverage.append(dict(day=day,pool_stocks=len(f),aligned_rows=len(a),
            general_reports={str(k):int(v) for k,v in a.fundamental_report_date.value_counts(dropna=False).items()},
            annual_eps_reports={str(k):int(v) for k,v in a.annual_eps_report_date.value_counts(dropna=False).items()},
            missing_general_stocks=missing,
            missing_from_event_universe=sorted(set(missing)-set(events.instrument)),
            missing_factors={k:int(v) for k,v in f.isna().sum().items()}))
    reports = []
    for date,part in events.loc[events.report_date.dt.year.eq(2026)].groupby("report_date"):
        reports.append(dict(report_date=str(date.date()),stocks=part.instrument.nunique(),
            first_available=str(part.available_date.min().date()),last_available=str(part.available_date.max().date())))
    return dict(as_of=END,max_report_date=str(events.report_date.max().date()),
        max_disclosure_date=str(events.disclosure_date.max().date()),max_available_date=str(events.available_date.max().date()),
        reports_2026=reports,coverage=coverage,
        annual_eps_definition="Latest disclosed full-year basic EPS, not quarterly EPS or TTM EPS",
        financial_revision_archive_complete=False,data_updated_for_experiment=False,
        source_sha256={str(DATA/name):digest(DATA/name) for name in
            ("financial_events.parquet","financial_alignment.parquet","raw_features.parquet")} )


def fit_jobs():
    return parent.serving_periods(3)


def audit():
    from paper_futu import overfit_four_core as core
    p = parent.check_sources()
    verification = read(parent.OUT / "final_verification.json")
    if verification["status"] != "passed": raise ValueError("Parent experiment audit failed")
    core.verify_protected(core.Store().protocol,full_snapshot=True)
    parent_completion = read(parent.OUT / "completion.json")
    if parent_completion["manifest_sha256"] != digest(parent.OUT / "report_manifest.json"):
        raise ValueError("Parent report manifest changed")
    parent_manifest = read(parent.OUT / "report_manifest.json")
    for path,expected in parent_manifest["artifacts"].items():
        if digest(path) != expected: raise ValueError(f"Parent artifact changed: {path}")
    sources = {str(parent.OUT / name):digest(parent.OUT / name) for name in
        ("protocol.json","comparison.json","summary.csv","account_period_metrics.csv","report_manifest.json","final_verification.json")}
    for case in ("V0","M3","M6","V7","V8"):
        for mode in (("fresh","history") if case in ("V0","M3","M6") else ("fresh",)):
            for offset in range(5):
                directory = (parent.OUT / "accounts" / f"{mode}_{case}_{offset}")
                if mode == "history" and case == "V0":
                    directory = Path(next(a for a in p["historical_baseline_accounts"] if a["offset"]==offset)["directory"])
                for path in directory.iterdir():
                    if path.is_file(): sources[str(path)] = digest(path)
    model = dict(snapshot_id=p["snapshot_id"],implementation_sha256=implementation(),
        parent_implementation_sha256=parent.implementation(),sources=sources,banks=p["banks"],
        annual_bindings=p["annual_bindings"],portfolio=PORTFOLIO,seed=42,
        parameters_by_year={y:parameters(b["parameters"]) for y,b in p["annual_bindings"].items()},
        policy="Quarterly fresh fits; original pre-year three-fold tree choice held fixed within year",
        data_cutoff=END,label="Open(T+11)/Open(T+1)-1",new_fits=20,new_accounts=10,
        periods=fit_jobs(),initial_cash=1_000_000,risk_degree=.95,rebalance=5,offsets=list(range(5)),
        min_start_gib=3.,min_task_gib=2.,max_workers=2,threads=4,
        limitations=p["limitations"])
    path = OUT / "protocol.json"
    if path.exists() and read(path) != model: raise ValueError("Existing combined protocol differs")
    save(path,model)
    save(OUT / "financial_coverage.json",financial_audit())
    save(OUT / "audit.json",dict(status="passed",parent_verified=True,protected_unchanged=True,
        protocol_sha256=digest(path),snapshot_id=p["snapshot_id"]))


def fit(cutoff):
    def execute():
        import lightgbm as lgb
        import numpy as np
        from paper_futu import overfit_four_core as core
        from src.runtime_budget import require_memory
        require_memory(2.,"Combined quarterly S6 fit")
        p = protocol()
        period = next(x for x in p["periods"] if x["cutoff"]==cutoff)
        choice = p["annual_bindings"][str(period["year"])]
        params = parameters(choice["parameters"])
        training = core.mature(parent.load_frame(),"2010-01-01",cutoff)
        model = lgb.LGBMRegressor(**{**params,"n_estimators":choice["trees"],"random_state":42})
        model.fit(training[list(core.frozen.ENHANCED_FEATURE_NAMES)],training.target)
        directory = MODELS / cutoff; directory.mkdir(parents=True,exist_ok=True)
        path = directory / "model.txt"; model.booster_.save_model(str(path))
        loaded = lgb.Booster(model_file=str(path))
        sample = training.iloc[::max(1,len(training)//2000)][list(core.frozen.ENHANCED_FEATURE_NAMES)]
        np.testing.assert_allclose(model.predict(sample),loaded.predict(sample,num_threads=4),atol=1e-12,rtol=0)
        result = dict(cutoff=cutoff,year=period["year"],train_rows=len(training),
            train_signal_end=str(training.index.get_level_values("datetime").max().date()),
            label_end_max=str(training.label_end_date.max().date()),training_fingerprint=parent.frame_fingerprint(training),
            model_path=str(path),model_sha256=digest(path),trees=choice["trees"],parameters=params,seed=42,reload_equal=True)
        if result["label_end_max"] > cutoff: raise ValueError("Quarterly training labels cross cutoff")
        save(directory / "metadata.json",result)
        return result,[path,directory / "metadata.json"]
    return run_task("fit",cutoff,execute)


def assemble():
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from paper_futu import final_ablation_core as frozen
    p = protocol()
    base = pd.read_parquet(parent.OUT / "predictions/history_V0.parquet")
    features = pd.read_parquet(DATA / "raw_features.parquet",columns=list(frozen.ENHANCED_FEATURE_NAMES))
    pieces = []
    for period in p["periods"]:
        fitted = cached("fit",period["cutoff"])
        if fitted is None: raise ValueError("Missing quarterly combined fit")
        part = base.loc[base.execution_date.between(period["start"],period["end"])].copy()
        model = lgb.Booster(model_file=fitted["model_path"])
        part["prediction"] = model.predict(features.reindex(part.index),num_threads=4)
        part["model_cutoff"] = pd.Timestamp(period["cutoff"])
        if (part.model_cutoff >= part.execution_date).any(): raise ValueError("Future model serves past date")
        pieces.append(part)
    prediction = pd.concat(pieces).sort_index()
    if not prediction.index.equals(base.index) or not np.isfinite(prediction.prediction).all():
        raise ValueError("Combined prediction stock/date sample changed")
    directory = OUT / "predictions"; directory.mkdir(exist_ok=True)
    prediction.to_parquet(directory / "history_QREG.parquet")
    prediction.loc[prediction.execution_date.between(START,END)].to_parquet(directory / "QREG.parquet")
    save(OUT / "streams.json",dict(streams={name:dict(path=str(directory / f"{name}.parquet"),
        sha256=digest(directory / f"{name}.parquet")) for name in ("QREG","history_QREG")}))


def worker(kind,names):
    protocol()
    if kind == "fit":
        allowed = {p["cutoff"] for p in fit_jobs()}
        for name in names:
            if name not in allowed: raise ValueError("Unexpected fit")
            fit(name)
    else:
        from paper_futu.s6_explicit_account import init_qlib,replay
        init_qlib(sys.modules[__name__])
        for name in names:
            mode,case,offset = name.split("_")
            if mode not in ("fresh","history") or case != "QREG" or int(offset) not in range(5):
                raise ValueError("Unexpected combined account")
            run_task("account",name,lambda mode=mode,offset=offset:replay(sys.modules[__name__],"QREG",int(offset),mode))


def execute_queue(kind,names):
    import psutil
    pending = [n for n in names if cached(kind,n) is None]
    active = {}
    retry_path = OUT / "memory_retries.json"
    retries = read(retry_path) if retry_path.exists() else {}
    maximum,wait_since,failure,peak = 2,None,None,0
    minimum = psutil.virtual_memory().available/2**30
    for name in pending:
        path = OUT / "tasks" / f"{kind}_{name}.json"
        if path.exists() and read(path)["status"] == "failed":
            raise RuntimeError(f"Previous failed task requires explicit diagnosis: {path}")
    try:
        while pending or active:
            for pid,(child,batch,stdout,stderr) in list(active.items()):
                if child.poll() is None: continue
                stdout.close();stderr.close();del active[pid]
                missing = [n for n in batch if cached(kind,n) is None]
                if child.returncode or missing:
                    records = [read(OUT / "tasks" / f"{kind}_{n}.json") for n in missing
                               if (OUT / "tasks" / f"{kind}_{n}.json").exists()]
                    errors = [r.get("error","") for r in records]
                    is_memory = bool(errors) and all("GiB" in e or "MemoryError" in e for e in errors)
                    if is_memory and all(f"{kind}_{n}" not in retries for n in missing):
                        for n in missing: retries[f"{kind}_{n}"] = dict(used=True,error=errors)
                        save(retry_path,retries);pending = missing+pending;maximum = 1
                    else: failure = f"Worker {pid} failed: {errors}; inspect logs"
            if failure and not active: raise RuntimeError(failure)
            free = psutil.virtual_memory().available/2**30;minimum = min(minimum,free)
            if pending and not active and free < 3.:
                wait_since = wait_since or time.time()
                if time.time()-wait_since >= 1800: raise MemoryError("RAM below 3 GiB for 30 minutes")
            else: wait_since = None
            while not failure and pending and len(active)<maximum and psutil.virtual_memory().available/2**30>=3.:
                size = 1 if kind=="fit" else 5
                batch,pending = pending[:size],pending[size:]
                logs = OUT / "logs";logs.mkdir(exist_ok=True)
                stamp = f"{kind}_{time.time_ns()}"
                stdout=(logs/f"{stamp}.stdout.log").open("w",encoding="utf-8")
                stderr=(logs/f"{stamp}.stderr.log").open("w",encoding="utf-8")
                child = subprocess.Popen([sys.executable,"-u","-m","paper_futu.research_s6_quarterly_regularized",
                    "--stage","worker","--kind",kind,"--names",",".join(batch)],cwd=ROOT,
                    stdout=stdout,stderr=stderr,env={**os.environ,**ENV},
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0)
                active[child.pid]=(child,batch,stdout,stderr);peak=max(peak,len(active))
            info=[]
            for pid,(child,batch,_,_) in active.items():
                try:
                    proc=psutil.Process(pid)
                    info.append(dict(pid=pid,created=proc.create_time(),names=batch,command=proc.cmdline(),
                        rss_gib=proc.memory_info().rss/2**30,cpu_seconds=sum(proc.cpu_times()[:2]),io=proc.io_counters()._asdict()))
                except psutil.Error: pass
            save(OUT/"status.json",dict(status="draining_error" if failure else "running",phase=kind,
                controller_pid=os.getpid(),controller_created=psutil.Process().create_time(),updated=time.time(),
                completed=sum(cached(kind,n) is not None for n in names),total=len(names),active=info,
                free_gib=free,min_free_gib=minimum,peak_parallel=peak,error=failure))
            if pending or active: time.sleep(2)
    finally:
        for child,_,stdout,stderr in active.values(): child.wait();stdout.close();stderr.close()
        save(OUT/"runtime"/f"{kind}_{time.time_ns()}.json",dict(kind=kind,min_free_gib=minimum,peak_parallel=peak,error=failure))


def report():
    import numpy as np
    import pandas as pd
    from html import escape
    from paper_futu import overfit_four_core as core
    from paper_futu.s6_2026_report import statistics,periods,safe
    p = check_sources()
    account_rows,contributions = [],[]
    for mode in ("fresh","history"):
        for offset in range(5):
            result = cached("account",f"{mode}_QREG_{offset}")
            if result is None: raise ValueError("Combined account incomplete")
            directory = Path(result["directory"]);core.verify_account(directory)
            daily = pd.read_parquet(directory/"daily.parquet")
            holdings = pd.read_parquet(directory/"holdings.parquet")
            h=holdings.assign(weight=holdings.market_value/holdings.datetime.map(daily.account))
            daily["bank_weight"]=h.loc[h.instrument.isin(p["banks"])].groupby("datetime").weight.sum().reindex(daily.index,fill_value=0)
            daily["largest_stock_weight"]=h.groupby("datetime").weight.max().reindex(daily.index,fill_value=0)
            orders=pd.read_parquet(directory/"orders.parquet");fills=orders.loc[orders.notional.gt(0)]
            expected=np.maximum(5.,fills.notional*np.where(fills.direction.eq(1),.0005,.0015))
            np.testing.assert_allclose(fills.fee,expected,atol=1e-7,rtol=1e-10)
            pnl=pd.read_parquet(directory/"contributions.parquet").set_index(["datetime","instrument"]).pnl_before_fees
            net=pnl.sub(fills.groupby(["datetime","instrument"]).fee.sum(),fill_value=0).rename("pnl").reset_index()
            prior=daily.account.shift(1,fill_value=1e6)
            for kind,name,part in periods(daily):
                account_rows.append(dict(mode=mode,case="QREG",offset=offset,period_kind=kind,period=name,**statistics(part)))
                if kind in ("all","year"):
                    by=net.loc[net.datetime.isin(part.index)].groupby("instrument").pnl.sum()
                    capital=prior.loc[part.index[0]]
                    np.testing.assert_allclose(by.sum(),part.account.iloc[-1]-capital,atol=1e-4,rtol=1e-10)
                    contributions.extend(dict(mode=mode,offset=offset,period=name,instrument=code,pnl_cny=float(value),
                        return_contribution=float(value/capital),bank=code in p["banks"]) for code,value in by.items())
    own=pd.DataFrame(account_rows)
    reference=pd.read_csv(parent.OUT/"account_period_metrics.csv")
    reference=reference.loc[(reference["mode"].eq("history")&reference.case.isin(["V0","M3","M6"]))
        |(reference["mode"].eq("fresh")&reference.case.isin(["V0","V7","V8","M3","M6"]))]
    metrics=pd.concat([reference,own],ignore_index=True)
    metrics.to_csv(OUT/"account_period_metrics.csv",index=False,encoding="utf-8-sig")
    summary=[]
    for key,part in metrics.groupby(["mode","case","period_kind","period"],sort=False):
        assert sorted(part.offset)==list(range(5))
        summary.append(dict(zip(("mode","case","period_kind","period"),key)) | dict(
            mean_return=float(part.total_return.mean()),mean_cagr=float(part.annualized_return.mean()),
            phase_return_std=float(part.total_return.std(ddof=0)),phase_cagr_std=float(part.annualized_return.std(ddof=0)),
            min_return=float(part.total_return.min()),max_return=float(part.total_return.max()),
            worst_drawdown=float(part.max_drawdown.max()),mean_fees_cny=float(part.fees_cny.mean()),
            mean_excess=float(part.geometric_excess.mean()),bank_weight=float(part.bank_weight.mean()),
            mean_daily_turnover=float(part.daily_turnover.mean()),mean_holdings=float(part.average_holdings.mean())))
    frame=pd.DataFrame(summary);frame.to_csv(OUT/"summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(contributions).to_csv(OUT/"stock_contributions_by_phase.csv",index=False,encoding="utf-8-sig")
    pairs=[]
    for mode in ("fresh","history"):
        base=reference.loc[reference["mode"].eq(mode)&reference.case.eq("M3")]
        joined=own.loc[own["mode"].eq(mode)].merge(base,on=["offset","period_kind","period"],suffixes=("_new","_old"))
        for _,row in joined.iterrows():
            pairs.append(dict(mode=mode,offset=int(row.offset),period_kind=row.period_kind,period=row.period,
                return_difference=float(row.total_return_new-row.total_return_old),
                cagr_difference=float(row.annualized_return_new-row.annualized_return_old),
                drawdown_difference=float(row.max_drawdown_new-row.max_drawdown_old)))
    pd.DataFrame(pairs).to_csv(OUT/"paired_vs_quarterly.csv",index=False,encoding="utf-8-sig")
    label=pd.read_parquet(DATA/"labels_h10.parquet")
    pred=pd.read_parquet(OUT/"predictions/history_QREG.parquet")
    joined=pred.join(label[["label","label_end_date"]]);joined=joined.loc[joined.label_end_date.le(pd.Timestamp(END))]
    ic=core.frozen.daily_ic(joined.prediction,joined.label)
    ic["execution_date"]=pred.execution_date.groupby(level="datetime").first().reindex(ic.index)
    ic.to_parquet(OUT/"daily_ic.parquet")
    icrows=[dict(year=int(y),mean_rank_ic=float(g.rank_ic.mean()),valid_days=int(g.rank_ic.notna().sum()))
            for y,g in ic.groupby(ic.execution_date.dt.year)]
    data=safe(dict(summary=summary,paired_vs_quarterly=pairs,signal_ic=icrows,
        financial_coverage=read(OUT/"financial_coverage.json"),new_fits=20,new_accounts=10,
        parameters=parameters(p["annual_bindings"]["2026"]["parameters"]),main_changed=False,limitations=p["limitations"]))
    save(OUT/"comparison.json",data)
    labels={"V0":"年度原版","V7":"年度L1/L2增强","V8":"年度降低复杂度","M6":"半年原版","M3":"季度原版","QREG":"季度＋L1/L2＋降低复杂度"}
    def table(part):
        part=part[["case","mean_return","mean_cagr","worst_drawdown","phase_return_std","mean_fees_cny","bank_weight"]].copy()
        part.case=part.case.map(labels)
        for name in part.columns[1:]: part[name]=part[name].map((lambda v:f"{v:,.0f}") if name=="mean_fees_cny" else (lambda v:f"{v:.2%}"))
        return part.rename(columns=dict(case="版本",mean_return="累计净收益",mean_cagr="平均年化",worst_drawdown="最差回撤",
            phase_return_std="收益起点标准差",mean_fees_cny="平均费用（元）",bank_weight="银行平均仓位")).to_html(index=False,border=0)
    body='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>S6季度重训与联合约束</title>
<style>body{font:16px/1.7 system-ui,'Microsoft YaHei';max-width:1300px;margin:30px auto;padding:0 20px;color:#20374a}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;text-align:right;border-bottom:1px solid #ddd}th{background:#24465d;color:white}h1{font-size:26px}h2{font-size:21px}</style>
<h1>S6季度重训＋L1/L2增强＋降低树复杂度</h1>
<p>仅新增一个联合设置：reg_alpha=1、reg_lambda=5、num_leaves=15、min_child_samples=300。每个执行年仍沿用原S6此前三年验证选出的树数；年内不选树或搜索参数。</p>
<p>Top20，前6只占股票仓位40%，股票仓位95%；每5个交易日调仓，最多替换6只。买入0.05%、卖出0.15%、最低5元。五起点均值；最差回撤取五个日收盘净值最大回撤的最差值，包含初始净值。</p>
<h2>2022—2026连续账户</h2><p>2022-01-04至2026-10-13，从100万元现金开始，跨年不重置调仓节奏。</p>'''
    a=frame.loc[frame["mode"].eq("history")&frame.period_kind.eq("all")].set_index("case")
    body+=table(a.loc[["V0","M6","M3","QREG"]].reset_index())
    years=frame.loc[frame["mode"].eq("history")&frame.period_kind.eq("year")].pivot(index="period",columns="case",values="mean_return")
    body+="<h2>逐年累计净收益</h2>"+years[["V0","M3","QREG"]].rename(columns=labels).map(lambda v:f"{v:.2%}").to_html(border=0)
    a=frame.loc[frame["mode"].eq("fresh")&frame.period_kind.eq("all")].set_index("case")
    body+="<h2>2026年空仓起步</h2><p>2026-01-05至2026-10-13，每个账户100万元现金开始。该表的2026收益与连续账户不同。</p>"+table(a.loc[["V0","V7","V8","M3","QREG"]].reset_index())
    financial=data["financial_coverage"];latest=financial["coverage"][-1]
    body+="<h2>财务数据实际覆盖</h2><p>截至2026-10-13，300只股票中290只对齐2026年半年报，10只缺少通用财务记录；没有纳入2026年三季报。E/P使用最新已披露的年度基本EPS，290只对齐2025年年报，并非TTM口径。因单项财务字段缺失，B/P、E/P各有11个缺失值，现金流/营收有10个缺失值。缺失保留给LightGBM处理，没有填入未来财报。</p>"
    body+="<p>缺失财务记录代码："+escape(", ".join(latest["missing_general_stocks"]))+"。</p><h2>局限</h2><ul>"+"".join("<li>"+escape(x)+"</li>" for x in p["limitations"])+"</ul>"
    body+='<p><a href="comparison.json">完整结果</a> · <a href="summary.csv">汇总表</a> · <a href="paired_vs_quarterly.csv">同起点配对差异</a> · <a href="financial_coverage.json">财务覆盖审计</a> · <a href="protocol.json">配置</a></p></html>'
    (OUT/"comparison.html").write_text(body,encoding="utf-8")
    core.verify_protected(core.Store().protocol,full_snapshot=True)
    # Reconstruct every actual training input independently of its saved metadata.
    training_frame=parent.load_frame()
    for period in p["periods"]:
        fitted=cached("fit",period["cutoff"])
        train=core.mature(training_frame,"2010-01-01",period["cutoff"])
        assert parent.frame_fingerprint(train)==fitted["training_fingerprint"]
        assert fitted["label_end_max"]<=period["cutoff"]
    artifacts={str(f):digest(f) for f in OUT.rglob("*") if f.is_file() and f.suffix!=".log"
        and f.name not in ("controller.lock","status.json","report_manifest.json","completion.json")}
    save(OUT/"report_manifest.json",dict(status="complete",implementation_sha256=implementation(),artifacts=artifacts,
        source_hashes=p["sources"],source_snapshot=p["snapshot_id"],new_fits=20,new_accounts=10,
        exact_training_inputs_verified=True,all_accounting_verified=True,protected_unchanged=True))
    save(OUT/"completion.json",dict(status="complete",finished=time.time(),new_fits=20,new_accounts=10,
        manifest_sha256=digest(OUT/"report_manifest.json"),comparison_sha256=digest(OUT/"comparison.json"),protected_unchanged=True))
    save(OUT/"status.json",dict(status="complete",phase="report",finished=time.time(),new_fits=20,new_accounts=10))
    print(frame.loc[frame.period_kind.eq("all"),["mode","case","mean_return","mean_cagr","worst_drawdown"]].to_string(index=False),flush=True)


@contextmanager
def controller_lock():
    import psutil
    OUT.mkdir(parents=True,exist_ok=True)
    for process in psutil.process_iter(["pid","cmdline"]):
        if process.pid!=os.getpid() and "paper_futu.research_s6_quarterly_regularized" in (process.info["cmdline"] or []):
            raise RuntimeError("An experiment controller or worker is already running")
    path=OUT/"controller.lock"
    if path.exists():
        old=read(path)
        try:
            if abs(psutil.Process(old["pid"]).create_time()-old["created"])<.01: raise RuntimeError("Live controller lock")
        except psutil.NoSuchProcess: pass
        path.unlink()
    with path.open("x",encoding="utf-8") as stream:
        __import__("json").dump(dict(pid=os.getpid(),created=psutil.Process().create_time()),stream)
    try: yield
    finally: path.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["audit","train","backtest","report","all","status","worker"],default="all")
    parser.add_argument("--kind",choices=["fit","account"]);parser.add_argument("--names",default="")
    args=parser.parse_args()
    if args.stage=="status":
        print(read(OUT/"status.json") if (OUT/"status.json").exists() else dict(status="not_started"));return
    if args.stage=="worker": worker(args.kind,args.names.split(","));return
    with controller_lock():
        try:
            if args.stage in ("audit","all") and not (OUT/"audit.json").exists(): audit()
            else: check_sources()
            if args.stage in ("train","all"):
                execute_queue("fit",[p["cutoff"] for p in fit_jobs()])
                subprocess.run([sys.executable,"-u","-c","from paper_futu.research_s6_quarterly_regularized import assemble; assemble()"],
                    cwd=ROOT,env={**os.environ,**ENV},check=True)
            if args.stage in ("backtest","all"):
                for s in read(OUT/"streams.json")["streams"].values():
                    if digest(s["path"])!=s["sha256"]: raise ValueError("Prediction stream changed")
                execute_queue("account",[f"{mode}_QREG_{offset}" for mode in ("fresh","history") for offset in range(5)])
            if args.stage in ("report","all"): report()
        except BaseException as error:
            previous=read(OUT/"status.json") if (OUT/"status.json").exists() else {}
            save(OUT/"status.json",{**previous,"status":"failed","error":repr(error),"updated":time.time()})
            traceback.print_exc();raise


if __name__=="__main__": main()
