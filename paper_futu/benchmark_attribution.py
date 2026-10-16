"""CSI300 total-return attribution without changing historical account ledgers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
API = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
RF = 0.01
STYLE_COLUMNS = ["book_to_price", "earnings_to_price", "volatility_60d", "mom_20d", "mom_60d"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def official_frame(payload, code):
    if str(payload.get("code")) != "200" or not payload.get("data"):
        raise ValueError("Official index response is empty or unsuccessful")
    f = pd.DataFrame(payload["data"])
    if not f.indexCode.eq(code).all():
        raise ValueError("Wrong index identity")
    expected_name = "沪深300全收益指数" if code == "H00300" else "沪深300指数"
    if not f.indexNameCnAll.eq(expected_name).all():
        raise ValueError("Wrong index name")
    f["datetime"] = pd.to_datetime(f.tradeDate, format="%Y%m%d", errors="raise")
    if f.datetime.duplicated().any():
        raise ValueError("Duplicate official trading dates")
    f = f.set_index("datetime").sort_index()
    for col in ["open", "close"]:
        f[col] = pd.to_numeric(f[col], errors="raise")
    if not np.isfinite(f.close).all() or not f.close.gt(0).all():
        raise ValueError("Invalid official closing levels")
    return f


def fetch_official(raw_dir, start, end):
    import requests
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata = raw_dir / "sources.json"
    if metadata.exists():
        sources = json.loads(metadata.read_text(encoding="utf-8"))
        if sources["start"] != start or sources["end"] != end:
            raise ValueError("Do not overwrite a different benchmark snapshot")
        for item in sources["sources"]:
            if digest(raw_dir / item["file"]) != item["sha256"]:
                raise ValueError("Cached official response changed")
        return sources
    entries = []
    for code in ["000300", "H00300"]:
        params = dict(indexCode=code, startDate=start.replace("-", ""), endDate=end.replace("-", ""))
        response = requests.get(API, params=params, timeout=30)
        response.raise_for_status()
        f = official_frame(response.json(), code)
        if f.index.min() != pd.Timestamp(start) or f.index.max() != pd.Timestamp(end):
            raise ValueError("Official series does not reach the requested boundaries")
        path = raw_dir / (code + ".json")
        if path.exists() and path.read_bytes() != response.content:
            raise ValueError("Existing raw response differs; use a new versioned directory")
        path.write_bytes(response.content)
        entries.append(dict(index_code=code, file=path.name, sha256=digest(path), url=response.url,
                            retrieved_at_utc=datetime.now(timezone.utc).isoformat(), rows=len(f)))
    value = dict(start=start, end=end, sources=entries, provider="China Securities Index Co., Ltd.",
                 redistribution="Raw official responses remain local; only research aggregates are published.")
    save_json(metadata, value)
    return value


def align_benchmarks(price, total, daily, tolerance=5e-6):
    dates = pd.DatetimeIndex(sorted(daily.datetime.unique()))
    if daily.duplicated(["offset", "datetime"]).any() or len(dates) < 30:
        raise ValueError("Duplicate account observations or insufficient observations")
    if set(daily.offset) != set(range(5)):
        raise ValueError("All five predetermined phases are required")
    for frame in [price, total]:
        if not dates.isin(frame.index).all():
            raise ValueError("Missing benchmark trading day; no filling or interpolation is permitted")
        if not frame.index[frame.index < dates[0]].size:
            raise ValueError("Previous-close anchor is absent")
    benchmark = pd.DataFrame(index=dates)
    for code, frame in [("price", price), ("total_return", total)]:
        returns = frame.close.pct_change(fill_method=None).reindex(dates)
        opening = frame.loc[dates[0], "open"]
        if not np.isfinite(opening) or opening <= 0:
            raise ValueError("The first account day requires an official opening level")
        returns.iloc[0] = frame.loc[dates[0], "close"] / opening - 1
        if not np.isfinite(returns).all():
            raise ValueError("Invalid aligned benchmark returns")
        benchmark[code] = returns
    maximum = 0.0
    for _, g in daily.groupby("offset"):
        g = g.sort_values("datetime")
        pd.testing.assert_index_equal(pd.DatetimeIndex(g.datetime), dates, check_names=False)
        np.testing.assert_allclose((1 + g.net_return).cumprod(), g.portfolio_nav, atol=1e-10)
        delta = np.max(np.abs(g.benchmark_return.to_numpy() - benchmark.price.to_numpy()))
        maximum = max(maximum, float(delta))
        if delta > tolerance:
            raise ValueError("Official price index does not match the frozen account benchmark")
    ratio = total.close.reindex(dates) / price.close.reindex(dates)
    if ratio.pct_change(fill_method=None).min() < -1e-5:
        raise ValueError("Unexpected negative dividend wedge beyond quote rounding")
    return benchmark, dict(days=len(dates), maximum_price_return_difference=maximum,
                           tolerance=tolerance, missing_dates=0, first_day_basis="official close/open",
                           remaining_days_basis="official close/previous close")


def capm(net, factors, lags=20):
    rf = np.expm1(np.log1p(RF) / 252)
    y = np.asarray(net, dtype=float) - rf
    values = np.asarray(factors, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    X = np.column_stack([np.ones(len(y)), values])
    if len(y) <= X.shape[1] + lags or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Insufficient or nonfinite regression observations")
    if np.linalg.matrix_rank(X) != X.shape[1]:
        raise ValueError("Collinear risk factors")
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    residual = y - X @ b
    z = X * residual[:, None]
    meat = z.T @ z
    for lag in range(1, lags + 1):
        term = z[lag:].T @ z[:-lag]
        meat += (1 - lag / (lags + 1)) * (term + term.T)
    inverse = np.linalg.inv(X.T @ X)
    covariance = inverse @ meat @ inverse * len(y) / (len(y) - X.shape[1])
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    p = math.erfc(abs(b[0] / se[0]) / math.sqrt(2)) if se[0] > 0 else None
    total = np.sum((y - y.mean()) ** 2)
    return dict(observations=len(y), alpha_annual_arithmetic=float(b[0] * 252),
                alpha_ci_low=float((b[0] - 1.95996398454 * se[0]) * 252),
                alpha_ci_high=float((b[0] + 1.95996398454 * se[0]) * 252),
                alpha_nominal_p=p, market_beta=float(b[1]),
                r_squared=float(1 - residual @ residual / total) if total else None, hac_lags=lags)


def account_metrics(g, benchmark):
    net = g.net_return.to_numpy()
    active = net - benchmark.to_numpy()
    n = len(net)
    strategy = np.prod(1 + net)
    base = np.prod(1 + benchmark)
    nav = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1., nav])[1:]
    te = np.std(active, ddof=1) * np.sqrt(252)
    return dict(trading_days=n, strategy_total_return=float(strategy-1),
                strategy_cagr=float(strategy ** (252/n) - 1),
                benchmark_cagr=float(base ** (252/n) - 1), benchmark_total_return=float(base - 1),
                geometric_excess_annual=float((strategy/base) ** (252/n) - 1),
                information_ratio=float(np.mean(active)*252/te),
                tracking_error_annual=float(te), max_drawdown=float(-(nav/peak-1).min()),
                fees_cny=float(g.fees_cny.sum()))


def style_factors(panel, banks):
    dates = pd.DatetimeIndex(sorted(panel.index.get_level_values("datetime").unique()))
    returns = panel.return_1d.unstack("instrument").reindex(dates)
    pool = pd.Series(True, index=panel.index).unstack("instrument").reindex(dates).shift(1).eq(True)
    is_bank = pd.Series(returns.columns.isin(banks), index=returns.columns)
    result = pd.DataFrame(index=dates)
    bank_return = returns.where(pool & is_bank).mean(axis=1)
    other_return = returns.where(pool & ~is_bank).mean(axis=1)
    counts = [(returns.notna() & pool & mask).sum(axis=1) for mask in [is_bank, ~is_bank]]
    result["bank_spread"] = (bank_return-other_return).where((counts[0] >= 10) & (counts[1] >= 10))
    for column in STYLE_COLUMNS:
        prior = panel[column].unstack("instrument").reindex(dates).shift(1).where(pool)
        rank = prior.rank(axis=1, pct=True, method="average")
        high, low = rank.gt(.7), rank.le(.3)
        enough = ((high & returns.notna()).sum(axis=1) >= 10) & ((low & returns.notna()).sum(axis=1) >= 10)
        result[column] = (returns.where(high).mean(axis=1)-returns.where(low).mean(axis=1)).where(enough)
    coverage = float((returns.notna() & pool).sum().sum()/pool.sum().sum())
    return result, dict(return_coverage=coverage, feature_lag_sessions=1,
                        style_definition="Prior-day cross-sectional top 30% minus bottom 30%, equal weighted",
                        limitation="Research proxies, not official traded factor indices; missing returns are not zero-filled.")


def analyze(daily_path, raw_dir, output, feature_panel=None, bank_protocol=None):
    daily_path, raw_dir, output = Path(daily_path), Path(raw_dir), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sources = json.loads((raw_dir/"sources.json").read_text(encoding="utf-8"))
    for item in sources["sources"]:
        if digest(raw_dir/item["file"]) != item["sha256"]:
            raise ValueError("Official response hash mismatch")
    frames = {code: official_frame(json.loads((raw_dir/(code+".json")).read_text(encoding="utf-8")), code)
              for code in ["000300", "H00300"]}
    daily = pd.read_csv(daily_path, parse_dates=["datetime"]).sort_values(["offset", "datetime"])
    if daily.datetime.max() != pd.Timestamp(sources["end"]):
        raise ValueError("Account and benchmark cutoff differ")
    benchmark, alignment = align_benchmarks(frames["000300"], frames["H00300"], daily)
    metrics, regressions = [], []
    rf = np.expm1(np.log1p(RF)/252)
    for offset, group in daily.groupby("offset"):
        group = group.set_index("datetime")
        periods = [("all", group), *[(str(y), part) for y, part in group.groupby(group.index.year)]]
        for period, g in periods:
            for basis in ["price", "total_return"]:
                b = benchmark[basis].reindex(g.index)
                metrics.append(dict(offset=int(offset), period=period, benchmark=basis, **account_metrics(g, b)))
                for lag in ([5,20,60] if period == "all" else [20]):
                    regressions.append(dict(offset=int(offset), period=period, model=basis+"_market",
                                            **capm(g.net_return, b-rf, lag)))
    style_audit = None
    if bool(feature_panel) != bool(bank_protocol):
        raise ValueError("Style attribution requires both its panel and audited bank classification")
    if feature_panel:
        panel = pd.read_parquet(feature_panel, columns=["return_1d", *STYLE_COLUMNS],
                                filters=[("datetime", ">=", pd.Timestamp(sources["start"]))])
        banks = json.loads(Path(bank_protocol).read_text(encoding="utf-8"))["banks"]
        factors, style_audit = style_factors(panel, banks)
        style_audit.update(panel_sha256=digest(feature_panel), bank_protocol_sha256=digest(bank_protocol))
        for offset, group in daily.groupby("offset"):
            # The account's first day is open-to-close; style proxies are close-to-close.
            g = group.set_index("datetime").iloc[1:].join(factors).dropna(subset=list(factors))
            style_audit["matched_trading_days"] = len(g)
            for model, names in [("total_return_market_bank", ["bank_spread"]),
                                  ("total_return_market_bank_styles", list(factors))]:
                x = pd.concat([(benchmark.total_return.reindex(g.index)-rf).rename("market"), g[names]], axis=1)
                regressions.append(dict(offset=int(offset), period="all", model=model,
                                        **capm(g.net_return, x, 20)))
    m, reg = pd.DataFrame(metrics), pd.DataFrame(regressions)
    m.to_csv(output/"per_phase.csv", index=False, lineterminator="\n")
    reg.to_csv(output/"regressions.csv", index=False, lineterminator="\n")
    summary = m.groupby(["period", "benchmark"], sort=True).agg(
        mean_strategy_total_return=("strategy_total_return","mean"),
        mean_strategy_cagr=("strategy_cagr","mean"), benchmark_cagr=("benchmark_cagr","first"),
        benchmark_total_return=("benchmark_total_return","first"),
        mean_geometric_excess_annual=("geometric_excess_annual","mean"),
        mean_information_ratio=("information_ratio","mean"), min_information_ratio=("information_ratio","min"),
        max_information_ratio=("information_ratio","max"), mean_tracking_error=("tracking_error_annual","mean"),
        worst_drawdown=("max_drawdown","max"), mean_fees_cny=("fees_cny","mean")).reset_index()
    summary.to_csv(output/"summary.csv", index=False, lineterminator="\n")
    if style_audit:
        save_json(output/"style_audit.json", style_audit)
    allrows = summary.loc[summary.period.eq("all")].set_index("benchmark")
    rows = []
    for basis, label in [("price","沪深300价格指数"),("total_return","沪深300全收益指数")]:
        v = allrows.loc[basis]
        a = reg.loc[reg.period.eq("all") & reg.model.eq(basis+"_market") & reg.hac_lags.eq(20)]
        rows.append(f"| {label} | {v.benchmark_total_return:.2%} | {v.benchmark_cagr:.2%} | "
                    f"{v.mean_geometric_excess_annual:.2%} | {v.mean_information_ratio:.3f} | "
                    f"{a.market_beta.mean():.3f} | {a.alpha_annual_arithmetic.mean():.2%} |")
    style_rows = []
    for model, label in [("total_return_market","全收益市场"),
                         ("total_return_market_bank","再控制银行"),
                         ("total_return_market_bank_styles","再控制五项风格")]:
        part = reg.loc[reg.period.eq("all") & reg.model.eq(model) & reg.hac_lags.eq(20)]
        if len(part):
            style_rows.append(f"| {label} | {part.alpha_annual_arithmetic.mean():.2%} | "
                              f"{part.market_beta.mean():.3f} | {part.r_squared.mean():.1%} | "
                              f"{part.alpha_nominal_p.min():.4f} 至 {part.alpha_nominal_p.max():.4f} |")
    year_rows=[]
    for period,g in summary.loc[summary.period.ne("all")].groupby("period"):
        t=g.set_index("benchmark");tr=t.loc["total_return"];pr=t.loc["price"]
        year_rows.append(f"| {period} | {tr.mean_strategy_total_return:.2%} | {pr.benchmark_total_return:.2%} | "
                         f"{tr.benchmark_total_return:.2%} | {tr.mean_information_ratio:.3f} |")
    report = "\n".join(["# S6全收益基准归因", "",
        "当前季度联合约束S6的账户、交易和费用保持不变，只替换评价基准。完整区间为2022-01-04至"+sources["end"]+"，五个预设调仓起点分别计算后汇总。", "",
        "| 基准 | 基准累计收益 | 基准复合年化 | 策略年化几何超额 | 信息比率 | 市场Beta | 年化回归Alpha |",
        "|---|---:|---:|---:|---:|---:|---:|", *rows, "",
        "无风险基准固定有效年利率1%，按252交易日折算。回归Alpha是日截距乘252，不是复合年化或可交易的对冲收益。HAC使用20阶Bartlett核与小样本修正，另附5/60阶敏感性；名义显著性未校正历史选优。", "",
        "## 行业及风格控制", "",
        "| 控制项 | 平均年化回归Alpha | 条件市场Beta | 日收益波动解释度 | 各起点名义p值范围 |",
        "|---|---:|---:|---:|---:|",*style_rows,"",
        "单市场回归使用完整1154日；风格回归剔除首个建仓日，使用匹配的1153日。p值是各账户自己的检验，不将五个高度相关起点合并为独立样本。", "",
        "## 分年度对照", "",
        "| 年份 | 策略平均净收益 | 价格指数收益 | 全收益指数收益 | 相对全收益指数的信息比率 |",
        "|---|---:|---:|---:|---:|",*year_rows,"",
        "2026仅截至截止日，表中收益为该区间累计收益而非完整年度。", "",
        "首日采用官方收盘/开盘，此后采用收盘/前收盘。官方000300与冻结基准逐日核对；没有补零、插值或用ETF替代H00300。原价格基准继续保留供追溯，前瞻主评价基准固定H00300。", "",
        "银行和五项风格控制使用前一日可得特征及研究收益代理，剔除首个建仓日。不是完整行业中性模型，不控制全部规模、质量及非线性动态风险。原始官方响应保留本地，不随公开报告发布。", "",
        "2022—2026已反复参与研究，这次改变基准不让它变成独立测试。策略公司行动使用复权近似，并非逐笔税后分红现金账；H00300为税前红利再投资基准，税费口径仍须区分。", "",
        "[官方指数资料](https://www.csindex.com.cn/zh-CN/indices/index-detail/000300)；"
        "[官方历史接口]("+API+")。原始来源、取数时间、响应和账户哈希见manifest.json。", ""])
    (output/"README.md").write_text(report, encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(1,2,figsize=(12,4.4), constrained_layout=True)
    for offset, g in daily.groupby("offset"):
        axes[0].plot(g.datetime, g.portfolio_nav, lw=1, alpha=.65, label="S6 phase "+str(offset))
    for basis, label, color in [("price","CSI300 price","#777777"),("total_return","CSI300 total return","#111111")]:
        axes[0].plot(benchmark.index,(1+benchmark[basis]).cumprod(),label=label,color=color,lw=2)
    axes[0].set(title="Historical NAV (not prospective)", ylabel="NAV, initial capital = 1")
    axes[0].set_xlim(benchmark.index.min(),benchmark.index.max())
    axes[0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].legend(fontsize=7, ncol=2)
    for basis, shift, color in [("price",-.18,"#9aa5b1"),("total_return",.18,"#225e8a")]:
        v=m.loc[m.period.eq("all") & m.benchmark.eq(basis)].sort_values("offset")
        axes[1].bar(v.offset+shift,v.information_ratio,width=.36,color=color,label=basis.replace("_"," "))
    axes[1].set(title="Annualized information ratio",xlabel="Predetermined phase",ylabel="Information ratio",xticks=range(5))
    axes[1].legend()
    for ax in axes: ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.savefig(output/"benchmark_attribution.png",dpi=160);plt.close(fig)
    artifacts = {p.name:digest(p) for p in output.iterdir() if p.is_file() and p.name != "manifest.json"}
    manifest = dict(status="complete", strategy_id="S6_head6_40_QREG", as_of=sources["end"],
                    created_at_utc=datetime.now(timezone.utc).isoformat(), risk_free_annual=RF,
                    account_daily_sha256=digest(daily_path), implementation_sha256=digest(__file__),
                    sources=sources, alignment=alignment, artifacts=artifacts,
                    historical_account_unchanged=True, independent_test=False)
    save_json(output/"manifest.json",manifest)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["fetch","analyze","all"],default="all")
    parser.add_argument("--as-of",default="2026-10-13")
    parser.add_argument("--daily-csv",type=Path,default=ROOT/"reports/current_strategy/phase_daily.csv")
    parser.add_argument("--raw-dir",type=Path,default=ROOT/"data/external/csi300_total_return_asof_20261013")
    parser.add_argument("--output",type=Path,default=ROOT/"reports/current_strategy/total_return")
    parser.add_argument("--feature-panel",type=Path)
    parser.add_argument("--bank-protocol",type=Path)
    args=parser.parse_args()
    if args.stage in ["fetch","all"]:fetch_official(args.raw_dir,"2021-12-31",args.as_of)
    if args.stage in ["analyze","all"]:
        result=analyze(args.daily_csv,args.raw_dir,args.output,args.feature_panel,args.bank_protocol)
        print(result.loc[result.period.eq("all")].to_json(orient="records"))
    else: print("Official benchmark responses validated and archived")


if __name__ == "__main__":
    main()

