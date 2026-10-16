"""Paired temporal, bank/style, seed, friction and accounting diagnostics."""
from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
import time

import numpy as np
import pandas as pd

from paper_futu import overfit_four_core as core
from paper_futu import final_ablation_core as frozen


def residual_correlation(group):
    """Pearson correlation of residuals of global ranks on contemporaneous controls."""
    complete = group[["prediction", "label", "bank", *core.STYLE]].dropna()
    if len(complete) < 30:
        return dict(style_subset_size=len(complete), style_coverage=len(complete)/len(group),
                    style_subset_rank_ic=np.nan, style_residual_rank_ic=np.nan)
    ranks = complete[["prediction", "label"]].rank(method="average").to_numpy(dtype=float)
    controls = np.column_stack([np.ones(len(complete)), complete[["bank", *core.STYLE]].to_numpy(dtype=float)])
    residual = ranks-controls @ np.linalg.lstsq(controls, ranks, rcond=None)[0]
    def corr(values):
        centered = values-values.mean(axis=0)
        denominator = np.sqrt((centered**2).sum(axis=0).prod())
        # Numerical regression remnants of a constant/exactly-explained rank are not a signal.
        if denominator <= 1e-12 or (np.linalg.norm(centered, axis=0) < 1e-8).any():
            return np.nan
        return float(np.sum(centered[:, 0]*centered[:, 1])/denominator)
    return dict(style_subset_size=len(complete), style_coverage=len(complete)/len(group),
                style_subset_rank_ic=corr(ranks), style_residual_rank_ic=corr(residual))


def summarize_ic(frame, identity):
    rows = []
    for kind in ["all", "year", "quarter"]:
        grouping = [("all", frame)] if kind == "all" else frame.groupby(
            frame.execution_date.dt.year if kind == "year" else frame.execution_date.dt.to_period("Q").astype(str))
        for period, group in grouping:
            record = dict(**identity, period_kind=kind, period=str(period), signal_days=len(group),
                          valid_ic_days=int(group.rank_ic.notna().sum()))
            for name in ["ic", "rank_ic", "bank_rank_ic", "nonbank_rank_ic", "style_subset_rank_ic", "style_residual_rank_ic"]:
                valid = group[name].dropna()
                record[f"mean_{name}"] = valid.mean()
                record[f"positive_{name}_ratio"] = valid.gt(0).mean() if len(valid) else np.nan
            for name in ["style_coverage", "prediction_rank_correlation", "topk_overlap", "top_bank_fraction"]:
                record[name] = group[name].mean()
            rows.append(record)
    return rows


def source_fold(protocol, family, parent, year):
    config = dict(spec=protocol["families"][family]["spec"], horizon=core.FAMILIES[family]["horizon"],
        parameter=parent["config"], validation_year=year,
        candidates=frozen.TREES[parent["config"]["parameters"]["learning_rate"]])
    key = frozen.identity(dict(version=frozen.VERSION, kind="validation_fold", config=config,
        code=protocol["scientific_source_sha256"], data=protocol["snapshot_id"]))[:24]
    record = core.read(core.SOURCE / "tasks" / f"{key}.json")
    if record["status"] != "complete":
        raise ValueError("Original validation fold not complete")
    for name, expected in record["artifacts"].items():
        if frozen.digest(name) != expected:
            raise ValueError("Original validation fold hash mismatch")
    return record["result"]


def generalization(stream, store):
    rows, files = [], []
    family, variant, seed = stream["family"], stream["variant"], stream["seed"]
    selections = core.read(core.OUT / "selected_trees.json")
    baseline_frame = core.load_frame(family) if variant == "V0" and seed == 42 else None
    for parent in store.protocol["families"][family]["periods"]:
        year = parent["selection_year"]
        choice = next(c for c in selections if c["family"] == family and c["variant"] == variant and c["execution_year"] == year)
        if variant == "V0" and seed == 42:
            import lightgbm as lgb
            training = core.mature(baseline_frame, "2010-01-01", parent["cutoff"])
            model = lgb.Booster(model_file=parent["model_path"])
            tail = core.training_tail(training, model, parent["trees"])
            train_rows, train_start = len(training), "2010-01-01"
            label_end_max, cutoff = str(training.label_end_date.max().date()), parent["cutoff"]
            del training, model
        else:
            period = next(p for p in stream["periods"] if p["execution_year"] == year)
            tail = pd.read_parquet(period["train_tail_path"])
            train_rows, train_start = period["train_rows"], period["train_start"]
            label_end_max, cutoff = period["label_end_max"], period["cutoff"]
        rows.append(dict(family=family, variant=variant, seed=seed, execution_year=year,
            sample="training_tail_252", trees=choice["trees"], mean_ic=float(tail.ic.mean()),
            mean_rank_ic=float(tail.rank_ic.mean()), positive_rank_ic_ratio=float(tail.rank_ic.dropna().gt(0).mean()),
            valid_days=int(tail.rank_ic.notna().sum()), train_rows=train_rows, train_start=train_start,
            train_label_end_max=label_end_max, train_cutoff=cutoff))
        for validation_year in range(year-3, year):
            if variant == "V0":
                records = source_fold(store.protocol, family, parent, validation_year)
            else:
                job = dict(kind="fold", family=family, variant=variant, parameters=choice["parameters"],
                           validation_year=validation_year)
                records = store.record(job)["result"]["records"]
            selected = next(r for r in records if r["trees"] == choice["trees"])
            rows.append(dict(family=family, variant=variant, seed=seed, execution_year=year,
                             sample=f"validation_{validation_year}", **selected))
    return rows, files


def signal_diagnostic(job, key, store):
    stream = job["stream"]
    family, variant, seed = stream["family"], stream["variant"], stream["seed"]
    prediction = pd.read_parquet(stream["prediction_path"])
    baseline = pd.read_parquet(store.protocol["families"][family]["prediction_path"])
    style = pd.read_parquet(core.DATA / "rank_features.parquet", columns=core.STYLE).reindex(prediction.index)+.5
    label = pd.read_parquet(core.DATA / f"labels_h{core.FAMILIES[family]['horizon']}.parquet")
    label = label.loc[label.label_end_date.le(frozen.AS_OF), ["label"]]
    frame = prediction[["prediction", "execution_date"]].join(style).join(label)
    frame["bank"] = frame.index.get_level_values("instrument").isin(store.protocol["bank_names"])
    frame["baseline"] = baseline.prediction
    ic = frozen.daily_ic(frame.prediction, frame.label)
    ic["bank_rank_ic"] = frozen.daily_ic(frame.loc[frame.bank].prediction, frame.loc[frame.bank].label, minimum=10).rank_ic
    ic["nonbank_rank_ic"] = frozen.daily_ic(frame.loc[~frame.bank].prediction, frame.loc[~frame.bank].label, minimum=10).rank_ic
    # K is fixed by the two portfolio rules sharing a stream (20 or 10).
    k = 20 if family == "raw10" else 10
    records = []
    for day, group in frame.groupby(level="datetime", sort=True):
        mature = group.loc[group.label.notna()]
        residual = residual_correlation(mature) if len(mature) else dict(style_subset_size=0,
            style_coverage=np.nan, style_subset_rank_ic=np.nan, style_residual_rank_ic=np.nan)
        predicted_names = set(group.nlargest(k, "prediction").index.get_level_values("instrument"))
        original_names = set(group.nlargest(k, "baseline").index.get_level_values("instrument"))
        records.append(dict(datetime=day, execution_date=group.execution_date.iloc[0],
            prediction_rank_correlation=group.prediction.corr(group.baseline, method="spearman"),
            topk_overlap=len(predicted_names & original_names)/k,
            top_bank_fraction=float(group.nlargest(k, "prediction").bank.mean()), **residual))
    daily = pd.DataFrame(records).set_index("datetime").join(ic)
    identity = dict(family=family, variant=variant, seed=seed)
    directory = core.OUT / "diagnostics" / key
    directory.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(directory/"daily_ic.parquet")
    pd.DataFrame(summarize_ic(daily, identity)).to_csv(directory/"ic_summary.csv", index=False, encoding="utf-8-sig")
    training, _ = generalization(stream, store)
    pd.DataFrame(training).to_csv(directory/"generalization.csv", index=False, encoding="utf-8-sig")
    return dict(directory=str(directory), **identity), list(directory.iterdir())


def account_diagnostic(job, key, store):
    source_job = job["account_job"]
    record = store.record(source_job)
    if not record:
        raise ValueError("Missing audited account")
    directory = Path(record["result"]["directory"])
    core.verify_account(directory)
    daily = pd.read_parquet(directory/"daily.parquet")
    holdings = pd.read_parquet(directory/"holdings.parquet")
    orders = pd.read_parquet(directory/"orders.parquet")
    pnl = pd.read_parquet(directory/"contributions.parquet")
    hh = holdings.merge(daily[["account"]], left_on="datetime", right_index=True)
    hh["weight"] = hh.market_value/hh.account
    style = pd.read_parquet(core.DATA/"rank_features.parquet", columns=core.STYLE,
                            filters=[("datetime", ">=", pd.Timestamp("2022-01-01"))])+.5
    hh = hh.set_index(["instrument", "datetime"]).join(style)
    bank = hh.loc[hh.index.get_level_values("instrument").isin(store.protocol["bank_names"])].weight.groupby(level="datetime").sum().reindex(daily.index, fill_value=0.)
    exposure = pd.DataFrame(dict(bank_weight=bank, largest_stock_weight=hh.weight.groupby(level="datetime").max())).reindex(daily.index)
    for field in core.STYLE:
        valid = hh.loc[hh[field].notna()]
        exposure[field] = (valid.market_value*valid[field]).groupby(level="datetime").sum()/valid.market_value.groupby(level="datetime").sum()
        exposure[field+"_coverage"] = valid.market_value.groupby(level="datetime").sum()/hh.market_value.groupby(level="datetime").sum()
    fills = orders.loc[orders.notional.gt(0)]
    fees = fills.groupby(["datetime", "instrument"]).fee.sum().rename("fee")
    profit = pnl.set_index(["datetime", "instrument"]).join(fees).fillna({"fee": 0})
    profit["net_pnl"] = profit.pnl_before_fees-profit.fee
    stocks = profit.groupby(level="instrument").net_pnl.sum().sort_values(ascending=False)
    stock_output = stocks.rename("net_pnl").to_frame()
    stock_output["bank"] = stock_output.index.isin(store.protocol["bank_names"])
    identity = {k: source_job[k] for k in ["strategy", "family", "variant", "seed", "offset", "friction"]}
    metrics = pd.read_csv(directory/"metrics.csv")
    rows = []
    for period in ["all", "2022", "2023", "2024", "2025", "2026"]:
        days = daily if period == "all" else daily.loc[daily.index.year == int(period)]
        original = metrics.loc[metrics.period.astype(str).eq(period)].iloc[0].to_dict()
        original = {k: v for k, v in original.items() if k not in identity}
        original.update(**identity, period=period, **exposure.loc[days.index].mean().to_dict(),
                        maximum_stock_weight=float(exposure.loc[days.index].largest_stock_weight.max()))
        rows.append(original)
    for excluded in [[2024], [2022, 2024]]:
        days = daily.loc[~daily.index.year.isin(excluded)]
        row = dict(**identity, period="excluding_"+"_".join(map(str, excluded))+"_diagnostic",
            annualized_return=float((1+days.net_return).prod()**(252/len(days))-1), discontinuous_diagnostic=True)
        rows.append(row)
    overall = rows[0]
    overall["net_profit_cny"] = float(daily.account.iloc[-1]-1e6)
    overall["bank_net_profit_cny"] = float(stocks.loc[stocks.index.isin(store.protocol["bank_names"])].sum())
    overall["bank_profit_ratio"] = overall["bank_net_profit_cny"]/overall["net_profit_cny"] if abs(overall["net_profit_cny"])>1e-8 else np.nan
    log_years = np.log1p(daily.net_return).groupby(daily.index.year).sum()
    overall["log_growth_share_2024"] = log_years.get(2024, 0.)/log_years.sum() if abs(log_years.sum())>1e-12 else np.nan
    target = core.OUT / "diagnostics" / key
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(target/"metrics_exposure.csv", index=False, encoding="utf-8-sig")
    exposure.to_parquet(target/"daily_exposure.parquet")
    stock_output.to_csv(target/"stock_contributions.csv", encoding="utf-8-sig")
    profit.reset_index().assign(year=lambda x: x.datetime.dt.year).groupby(["year", "instrument"]).net_pnl.sum().to_csv(target/"stock_contributions_yearly.csv", encoding="utf-8-sig")
    return dict(directory=str(target), account_directory=str(directory), **identity), list(target.iterdir())


def diagnostic_jobs():
    streams = core.read(core.OUT/"streams.json")
    accounts = core.read(core.OUT/"account_jobs.json")
    return ([dict(kind="signal_diagnostic", stream=stream) for stream in streams]
        + [dict(kind="account_diagnostic", account_job=account) for account in accounts])


def paired_bootstrap(a, b, block, repeats=2000, seed=20261015, cagr=False):
    """Paired moving blocks; columns are the five phases when cagr=True."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or len(a) < block:
        raise ValueError("Paired bootstrap needs the same adequately long interval")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Bootstrap inputs must be finite and pre-aligned")
    rng = np.random.default_rng(seed)
    n = len(a)
    samples = []
    if cagr:
        if (a <= -1).any() or (b <= -1).any():
            raise ValueError("Invalid account return")
        a, b = np.log1p(a), np.log1p(b)
    count = int(np.ceil(n/block))
    for start in range(0, repeats, 100):
        size = min(100, repeats-start)
        starts = rng.integers(0, n-block+1, size=(size, count))
        indices = (starts[..., None]+np.arange(block)).reshape(size, -1)[:, :n]
        if cagr:
            result = (np.expm1(a[indices].sum(axis=1)*252/n)-np.expm1(b[indices].sum(axis=1)*252/n)).mean(axis=1)
        else:
            result = (a[indices]-b[indices]).mean(axis=1)
        samples.extend(result.tolist())
    values = np.asarray(samples)
    point = float((np.expm1(a.sum(axis=0)*252/n)-np.expm1(b.sum(axis=0)*252/n)).mean()) if cagr else float((a-b).mean())
    return dict(point_delta=point, ci_2_5=float(np.quantile(values, .025)),
                ci_97_5=float(np.quantile(values, .975)), block=block, repeats=repeats,
                conditional_history_only=True)


def aggregate(store, jobs):
    summary, general, exposures = [], [], []
    signals, accounts = {}, {}
    for job in jobs:
        record = store.record(job)
        if not record:
            raise ValueError("Diagnostics incomplete")
        result = record["result"]
        directory = Path(result["directory"])
        if job["kind"] == "signal_diagnostic":
            summary.append(pd.read_csv(directory/"ic_summary.csv"))
            general.append(pd.read_csv(directory/"generalization.csv"))
            signals[(result["family"], result["variant"], result["seed"])] = pd.read_parquet(directory/"daily_ic.parquet")
        else:
            exposures.append(pd.read_csv(directory/"metrics_exposure.csv"))
            accounts[(result["strategy"], result["variant"], result["seed"], result["friction"], result["offset"])] = result["account_directory"]
    def write(name, frame):
        frame.to_csv(core.OUT/name, index=False, encoding="utf-8-sig")
    all_metrics = pd.concat(exposures, ignore_index=True)
    write("account_metrics_exposure.csv", all_metrics)
    write("signal_ic_summary.csv", pd.concat(summary, ignore_index=True))
    write("generalization.csv", pd.concat(general, ignore_index=True))
    groups = ["strategy", "family", "variant", "seed", "friction", "period"]
    means = all_metrics.groupby(groups, sort=True).mean(numeric_only=True).reset_index()
    risks = all_metrics.groupby(groups, sort=True).agg(
        worst_phase_drawdown=("max_drawdown", "min"), maximum_stock_weight=("maximum_stock_weight", "max")).reset_index()
    means = means.drop(columns="maximum_stock_weight").merge(risks, on=groups)
    spread = all_metrics.groupby(groups, sort=True).annualized_return.std(ddof=0).rename("phase_std_cagr").reset_index()
    means = means.merge(spread, on=groups)
    write("strategy_summary.csv", means)
    primary = means.loc[means.seed.eq(42)&means.friction.eq(0)]
    baseline = primary.loc[primary.variant.eq("V0")].set_index(["strategy", "period"])
    differences = []
    for row in primary.itertuples(index=False):
        reference = baseline.loc[(row.strategy, row.period)]
        differences.append(dict(strategy=row.strategy, variant=row.variant, period=row.period,
            delta_cagr=row.annualized_return-reference.annualized_return,
            delta_worst_drawdown=row.worst_phase_drawdown-reference.worst_phase_drawdown,
            delta_fees=row.total_fees_cny-reference.total_fees_cny,
            delta_turnover=row.average_daily_turnover-reference.average_daily_turnover))
    write("paired_strategy_differences.csv", pd.DataFrame(differences))
    boot = []
    for family in core.FAMILIES:
        reference = signals[(family, "V0", 42)]
        for variant in list(core.VARIANTS)[1:]:
            compared = signals[(family, variant, 42)]
            joined = compared[["rank_ic"]].join(reference[["rank_ic"]], lsuffix="_variant", rsuffix="_base").dropna()
            for block in [60, 120]:
                boot.append(dict(kind="rank_ic", family=family, variant=variant,
                    **paired_bootstrap(joined.rank_ic_variant, joined.rank_ic_base, block)))
    for strategy, rule in core.STRATEGIES.items():
        phases = frozen.offsets(rule["rebalance"])
        ref = pd.concat([pd.read_parquet(Path(accounts[(strategy, "V0", 42, 0., phase)])/"daily.parquet").net_return.rename(phase) for phase in phases], axis=1)
        for variant in list(core.VARIANTS)[1:]:
            new = pd.concat([pd.read_parquet(Path(accounts[(strategy, variant, 42, 0., phase)])/"daily.parquet").net_return.rename(phase) for phase in phases], axis=1)
            if not ref.index.equals(new.index):
                raise ValueError("Bootstrap account intervals differ")
            for block in [60,120]:
                boot.append(dict(kind="mean_phase_cagr", strategy=strategy, family=rule["family"], variant=variant,
                    **paired_bootstrap(new.to_numpy(), ref.to_numpy(), block, cagr=True)))
    write("paired_block_bootstrap.csv", pd.DataFrame(boot))
    core.save(core.OUT/"diagnostic_aggregation.json", dict(status="complete", accounts=len(accounts),
        signal_streams=len(signals), jobs=len(jobs), timestamp=time.time()))


def report(store):
    aggregate_manifest = core.read(core.OUT/"diagnostic_aggregation.json")
    if aggregate_manifest["accounts"] != 340 or aggregate_manifest["signal_streams"] != 30:
        raise ValueError("Incomplete report scope")
    summary = pd.read_csv(core.OUT/"strategy_summary.csv")
    signals = pd.read_csv(core.OUT/"signal_ic_summary.csv")
    bootstrap = pd.read_csv(core.OUT/"paired_block_bootstrap.csv")
    if len(bootstrap) != 96:
        raise ValueError("Paired uncertainty analysis incomplete")
    sections = []
    evidence = []
    for strategy, rule in core.STRATEGIES.items():
        frame = summary.loc[summary.strategy.eq(strategy)&summary.seed.eq(42)&summary.friction.eq(0)&summary.period.eq("all")].sort_values("variant")
        if len(frame) != 9:
            raise ValueError("Missing strategy/variant rows")
        baseline = frame.loc[frame.variant.eq("V0")].iloc[0]
        columns = ["variant", "annualized_return", "phase_std_cagr", "worst_phase_drawdown", "total_fees_cny", "bank_weight", "maximum_stock_weight"]
        display = frame[columns].copy()
        for field in columns[1:]:
            display[field] = display[field].map(lambda x: f"{x:,.0f}" if field == "total_fees_cny" else f"{x:.2%}")
        sections.append(f"<h2>{strategy}</h2>"+display.to_html(index=False, escape=True))
        years = summary.loc[summary.strategy.eq(strategy)&summary.variant.eq("V0")&summary.seed.eq(42)&summary.friction.eq(0)&summary.period.isin(["2022","2023","2024","2025","2026"])]
        observed = signals.loc[signals.family.eq(rule["family"])&signals.variant.eq("V0")&signals.seed.eq(42)&signals.period_kind.eq("year")]
        negative_years = observed.loc[observed.mean_rank_ic.lt(0), "period"].astype(str).tolist()
        variants = frame.loc[frame.variant.ne("V0")]
        item = dict(strategy=strategy, baseline_cagr=float(baseline.annualized_return),
            baseline_worst_drawdown=float(baseline.worst_phase_drawdown),
            variants_under_30pct_drawdown=int(variants.worst_phase_drawdown.ge(-.3).sum()),
            variants_higher_cagr=int(variants.annualized_return.gt(baseline.annualized_return).sum()),
            baseline_negative_rank_ic_years=negative_years,
            interpretation="Historical sensitivity evidence only; no champion selected or overfit probability estimated")
        evidence.append(item)
        sections.append("<p>原版执行年Rank IC为负的年份："+escape("、".join(negative_years) or "无")+"。变体表完整保留；收益提高不能单独证明泛化改善。</p>")
    core.save(core.OUT/"evidence_summary.json", evidence)
    heading = "S2、S4、S5、S6：训练窗口、正则化与风格依赖研究"
    html = "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>"+heading+"</title><style>body{font:16px system-ui;max-width:1250px;margin:36px auto;padding:0 20px;color:#183047}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}h1{font-size:26px}h2{margin-top:32px}p{line-height:1.7}</style><h1>"+heading+"</h1>"
    html += "<p>2022年初至2026-10-13；截至2026-10-13的固定快照。平均年化为五个预设调仓起点的平均值，回撤取最差起点，费用为全期人民币金额。股票目标仓位95%。</p>"
    html += "<p>V0原版；V1/V2/V3近3/5/10年；V4近期加权；V5增强L1；V6增强L2；V7联合增强；V8降低树复杂度。年度重训，修改规则后使用此前三年验证折重新选树数。</p>"+"".join(sections)
    html += "<h2>解释边界</h2><p>2022—2026已经反复用于研究，均非新的独立测试。区块重采样区间仅描述给定历史的不确定性。银行标记及估值、波动、动量残差诊断不是完整行业中性，也不等于可交易Alpha。剔除年份指标是非连续依赖性诊断。未选择新冠军、未修改模拟盘。</p>"
    html += "<p>回测使用日线理想开盘成交、统一9.5%涨跌幅过滤及原公司行动近似；财务数据不含完整历史修订档案。2026为截至截止日的部分年度。</p></html>"
    (core.OUT/"research_report.html").write_text(html, encoding="utf-8")
    outputs = [p for p in core.OUT.iterdir() if p.suffix in (".csv", ".html", ".json") and p.name not in ("status.json", "report_manifest.json", "completion.json")]
    core.save(core.OUT/"report_manifest.json", dict(status="complete", snapshot_id=store.protocol["snapshot_id"],
        implementation_sha256=core.scientific_hash(), results={str(p): frozen.digest(p) for p in outputs},
        account_count=340, main_streams=18, extra_seed_streams=12, no_champion_selected=True,
        limitations=store.protocol["limitations"]))
