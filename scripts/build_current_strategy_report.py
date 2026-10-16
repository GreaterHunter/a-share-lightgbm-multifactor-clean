"""Rebuild the selected S6 report from sanitized, aggregate research evidence.

The default command uses only files shipped with this repository. The optional
import step reads an already completed local experiment; it never trains a model
or connects to a broker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/current_strategy"
SUMMARY = ROOT / "reports/experiments/s6_quarterly_l1l2_small_asof_20261013/summary.csv"
OFFSETS = tuple(range(5))
START, END = "2022-01-04", "2026-10-13"
INITIAL_CASH = 1_000_000
ANNUAL_RISK_FREE_RATE = 0.01
COLUMNS = ["offset", "datetime", "net_return", "benchmark_return", "portfolio_nav",
           "benchmark_nav", "fees_cny", "turnover", "holdings_count"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def drawdown(nav):
    """Include initial capital, including a loss on the first observed day."""
    values = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(np.r_[1.0, values])[1:]
    return values / peak - 1.0


def risk_adjusted_metrics(net_return, benchmark_return, annual_risk_free_rate=ANNUAL_RISK_FREE_RATE):
    """Use a fixed effective annual cash benchmark; IR remains index-relative."""
    if not np.isfinite(annual_risk_free_rate) or annual_risk_free_rate <= -1:
        raise ValueError("Invalid annual risk-free rate")
    daily_risk_free = np.expm1(np.log1p(annual_risk_free_rate) / 252)
    excess = net_return - daily_risk_free
    active = net_return - benchmark_return
    excess_std, active_std = excess.std(ddof=1), active.std(ddof=1)
    return dict(risk_free_rate_annual=float(annual_risk_free_rate),
        sharpe=float(excess.mean()/excess_std*np.sqrt(252)) if excess_std>0 else float("nan"),
        information_ratio=float(active.mean()/active_std*np.sqrt(252)) if active_std>0 else float("nan"),
        annualized_volatility=float(net_return.std(ddof=1)*np.sqrt(252)),
        annualized_tracking_error=float(active_std*np.sqrt(252)))


def period_stats(part):
    wealth = (1 + part.net_return).cumprod()
    benchmark = (1 + part.benchmark_return).prod()
    return dict(trading_days=len(part), total_return=float(wealth.iloc[-1] - 1),
        annualized_return=float(wealth.iloc[-1] ** (252 / len(part)) - 1),
        benchmark_return=float(benchmark - 1),
        geometric_excess=float(wealth.iloc[-1] / benchmark - 1),
        max_drawdown=float(-drawdown(wealth).min()), fees_cny=float(part.fees_cny.sum()),
        daily_turnover=float(part.turnover.mean()), average_holdings=float(part.holdings_count.mean()),
        **risk_adjusted_metrics(part.net_return,part.benchmark_return))


def calculate_tables(daily):
    records = []
    for offset, part in daily.groupby("offset", sort=True):
        part = part.sort_values("datetime")
        records.append(dict(offset=int(offset), period="all", **period_stats(part)))
        for year, year_part in part.groupby(part.datetime.dt.year):
            records.append(dict(offset=int(offset), period=str(year), **period_stats(year_part)))
    phases = pd.DataFrame(records)
    aggregates = []
    for period, part in phases.groupby("period", sort=True):
        aggregates.append(dict(period=period, trading_days=int(part.trading_days.iloc[0]),
            mean_return=part.total_return.mean(), mean_cagr=part.annualized_return.mean(),
            min_return=part.total_return.min(), max_return=part.total_return.max(),
            min_cagr=part.annualized_return.min(), max_cagr=part.annualized_return.max(),
            phase_cagr_std=part.annualized_return.std(ddof=0),
            worst_drawdown=part.max_drawdown.max(), benchmark_return=part.benchmark_return.iloc[0],
            mean_excess=part.geometric_excess.mean(), mean_fees_cny=part.fees_cny.mean(),
            mean_daily_turnover=part.daily_turnover.mean(), mean_holdings=part.average_holdings.mean(),
            risk_free_rate_annual=ANNUAL_RISK_FREE_RATE, mean_sharpe=part.sharpe.mean(),
            min_sharpe=part.sharpe.min(), max_sharpe=part.sharpe.max(),
            mean_information_ratio=part.information_ratio.mean(),
            min_information_ratio=part.information_ratio.min(), max_information_ratio=part.information_ratio.max(),
            mean_annualized_volatility=part.annualized_volatility.mean(),
            mean_annualized_tracking_error=part.annualized_tracking_error.mean()))
    return phases, pd.DataFrame(aggregates)


def validate_daily(daily):
    if list(daily.columns) != COLUMNS or daily.duplicated(["offset", "datetime"]).any():
        raise ValueError("Unexpected public fields or duplicate phase/date")
    if set(daily.offset) != set(OFFSETS) or daily[COLUMNS[2:]].isna().any().any():
        raise ValueError("Missing phase or numerical evidence")
    if not np.isfinite(daily[COLUMNS[2:]].to_numpy()).all():
        raise ValueError("Non-finite research result")
    first = None
    for offset, part in daily.groupby("offset", sort=True):
        part = part.sort_values("datetime")
        if part.datetime.min() != pd.Timestamp(START) or part.datetime.max() != pd.Timestamp(END):
            raise ValueError("Comparison interval changed")
        if first is None:
            first = part
        else:
            np.testing.assert_array_equal(part.datetime, first.datetime)
            np.testing.assert_allclose(part.benchmark_return, first.benchmark_return, atol=1e-12, rtol=0)
        np.testing.assert_allclose((1 + part.net_return).cumprod(), part.portfolio_nav, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose((1 + part.benchmark_return).cumprod(), part.benchmark_nav, atol=1e-10, rtol=1e-10)
        if (part.fees_cny < 0).any() or not part.holdings_count.between(0, 20).all():
            raise ValueError("Invalid fees or holdings count")
        if offset and not part.portfolio_nav.iloc[:offset].eq(1.0).all():
            raise ValueError("Phase initial cash period changed")


def verify_summary(aggregates):
    audited = pd.read_csv(SUMMARY)
    audited = audited.loc[audited["mode"].eq("history") & audited.case.eq("QREG")]
    for _, row in aggregates.iterrows():
        expected = audited.loc[audited.period.astype(str).eq(row.period)]
        if len(expected) != 1:
            raise ValueError("Audited period is not unique")
        for column in ("mean_return", "mean_cagr", "phase_cagr_std", "worst_drawdown", "mean_fees_cny", "mean_excess"):
            np.testing.assert_allclose(row[column], expected.iloc[0][column], atol=1e-8, rtol=1e-10)


def import_evidence(source):
    source = Path(source).resolve()
    completion = read_json(source / "completion.json")
    verification = read_json(source / "final_verification.json")
    if completion["status"] != "complete" or verification["status"] != "passed":
        raise ValueError("Only completed, independently verified experiments can be imported")
    if completion["manifest_sha256"] != digest(source / "report_manifest.json"):
        raise ValueError("Source manifest changed")
    manifest = read_json(source / "report_manifest.json")
    if not manifest["all_accounting_verified"] or not manifest["protected_unchanged"]:
        raise ValueError("Source accounting or protected-state audit did not pass")
    source_hashes = {}

    def verified_file(path, bindings):
        expected = bindings.get(str(path)) or bindings.get(path.as_posix())
        actual = digest(path)
        if expected != actual:
            raise ValueError(f"Source binding mismatch: {path.name}")
        source_hashes[path.relative_to(source).as_posix()] = actual

    verified_file(source / "summary.csv", manifest["artifacts"])
    if digest(source / "summary.csv") != digest(SUMMARY):
        raise ValueError("Public audited summary differs from source")
    verified_file(source / "account_period_metrics.csv", manifest["artifacts"])
    frames = []
    for offset in OFFSETS:
        task_path = source / "tasks" / f"account_history_QREG_{offset}.json"
        verified_file(task_path, manifest["artifacts"])
        task = read_json(task_path)
        if task["status"] != "complete" or task["protocol_sha256"] != digest(source / "protocol.json"):
            raise ValueError("Account task or protocol changed")
        path = source / "accounts" / f"history_QREG_{offset}" / "daily.parquet"
        verified_file(path, manifest["artifacts"])
        verified_file(path, task["artifacts"])
        raw = pd.read_parquet(path)
        np.testing.assert_allclose(raw.account / INITIAL_CASH, raw.portfolio_nav, atol=1e-10, rtol=1e-10)
        if (raw.cash < -1e-5).any():
            raise ValueError("Negative research-account cash")
        frame = raw.rename(columns={"bench": "benchmark_return", "fees_cash": "fees_cny"}).copy()
        frame["datetime"], frame["offset"] = pd.to_datetime(raw.index), offset
        frames.append(frame[COLUMNS])
    daily = pd.concat(frames, ignore_index=True)
    validate_daily(daily)
    _, aggregates = calculate_tables(daily)
    verify_summary(aggregates)
    OUT.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUT / "phase_daily.csv", index=False, float_format="%.17g", lineterminator="\n")
    evidence = dict(strategy_id="S6_head6_40_QREG", experiment=source.name,
        data_cutoff=END, initial_cash=INITIAL_CASH, source_snapshot=manifest["source_snapshot"],
        source_manifest_sha256=digest(source / "report_manifest.json"),
        source_protocol_sha256=digest(source / "protocol.json"), source_sha256=source_hashes,
        public_daily_sha256=digest(OUT / "phase_daily.csv"),
        scope="Aggregate historical research accounts only; no broker state, orders or stock-level holdings",
        validation="Source task/manifest hashes and daily NAV/fees checked; period metrics match audited summary")
    (OUT / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_figures(daily, aggregates, versions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter

    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.labelcolor": "#334155",
        "text.color": "#172b4d", "axes.edgecolor": "#cbd5e1", "savefig.facecolor": "white"})
    colors = ["#2563eb", "#0f766e", "#a16207", "#7c3aed", "#0891b2"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.55, 1]}, layout="constrained")
    for offset, part in daily.groupby("offset", sort=True):
        dates = pd.DatetimeIndex([pd.Timestamp("2021-12-31"), *part.datetime])
        nav = np.r_[1.0, part.portfolio_nav]
        axes[0].plot(dates, nav, color=colors[int(offset)], lw=1.4, label=f"S6 phase {offset}")
        axes[1].plot(dates, drawdown(nav), color=colors[int(offset)], lw=1.1)
    base = daily.loc[daily.offset.eq(0)]
    dates = pd.DatetimeIndex([pd.Timestamp("2021-12-31"), *base.datetime])
    benchmark = np.r_[1.0, base.benchmark_nav]
    axes[0].plot(dates, benchmark, color="#475569", ls="--", lw=1.8, label="CSI 300 benchmark")
    axes[1].plot(dates, drawdown(benchmark), color="#475569", ls="--", lw=1.4)
    axes[1].axhline(-.30, color="#dc2626", ls=":", lw=1.3, label="30% research risk threshold")
    axes[0].set(title="Selected S6: quarterly refit + L1/L2 + smaller trees", ylabel="Net asset value (initial = 1)")
    axes[1].set(ylabel="Drawdown from prior high", xlabel="Historical research interval | 2026 ends on Oct 13")
    axes[0].legend(ncol=3, frameon=False, fontsize=9, loc="upper left")
    axes[1].legend(frameon=False, fontsize=9, loc="lower right")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[1].set_xlim(pd.Timestamp("2021-12-31"), pd.Timestamp(END)+pd.Timedelta(days=14))
    for ax in axes: ax.grid(alpha=.2)
    fig.savefig(figures / "nav_drawdown.png", dpi=180)
    plt.close(fig)

    years = aggregates.loc[aggregates.period.ne("all")].sort_values("period")
    x = np.arange(len(years)); width = .34
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    error = np.vstack([years.mean_return-years.min_return, years.max_return-years.mean_return])
    bars = ax.bar(x-width/2, years.mean_return, width, color="#2563eb", label="S6: mean of 5 phase returns",
                  yerr=error, capsize=4, error_kw={"elinewidth":1.1, "ecolor":"#172b4d"})
    base_bars = ax.bar(x+width/2, years.benchmark_return, width, color="#94a3b8", label="CSI 300")
    for bar, upper in zip(bars, years.max_return):
        ax.annotate(f"{bar.get_height():.1%}", (bar.get_x()+bar.get_width()/2,upper), xytext=(0,8),
                    textcoords="offset points",ha="center",fontsize=9)
    for bar in base_bars:
        h=bar.get_height()
        ax.annotate(f"{h:.1%}", (bar.get_x()+bar.get_width()/2, h), xytext=(0, 7 if h>=0 else -15),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x, [y if y!="2026" else "2026 YTD" for y in years.period])
    ax.set(title="Annual net returns | whiskers show the minimum and maximum phase", ylabel="Period return")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.axhline(0, color="#64748b", lw=.8)
    ax.set_ylim(min(years.benchmark_return.min(),years.min_return.min())-.08, years.max_return.max()+.12)
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, loc="upper left")
    fig.savefig(figures / "annual_returns.png", dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    x=np.arange(len(versions))
    first=ax.bar(x-width/2, versions.mean_cagr, width, color="#2563eb", label="Mean compound annual net return")
    second=ax.bar(x+width/2, versions.worst_drawdown, width, color=["#f59e0b"]*3+["#dc2626"], label="Worst maximum drawdown across phases")
    ax.axhline(.30, color="#dc2626", lw=1.1, ls=":", label="30% drawdown threshold")
    for bar in [*first,*second]:
        ax.annotate(f"{bar.get_height():.2%}", (bar.get_x()+bar.get_width()/2,bar.get_height()),
                    xytext=(0,5),textcoords="offset points",ha="center",fontsize=10)
    ax.set_xticks(x,["Annual baseline","6-month baseline","Quarterly baseline","Quarterly regularized\n(selected)"])
    ax.set(title="S6 training variants | Net annual return and worst drawdown", ylabel="Return / drawdown magnitude",ylim=(0,.39))
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y",alpha=.2)
    ax.legend(frameon=False,fontsize=9,loc="upper left")
    fig.savefig(figures / "version_comparison.png",dpi=180);plt.close(fig)


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "|"+"---|"*len(headers),
                       *["| "+" | ".join(map(str,row))+" |" for row in rows]])


def make_report(phases, aggregates, versions):
    full=aggregates.loc[aggregates.period.eq("all")].iloc[0]
    years=aggregates.loc[aggregates.period.ne("all")].sort_values("period")
    phase_rows=phases.loc[phases.period.eq("all")].sort_values("offset")
    phase_table=table(["起点偏移", "累计净收益", "复合年化", "最大回撤", "夏普", "信息比率", "累计费用"],
        [[int(r.offset),f"{r.total_return:.2%}",f"{r.annualized_return:.2%}",f"{r.max_drawdown:.2%}",f"{r.sharpe:.3f}",f"{r.information_ratio:.3f}",f"{r.fees_cny:,.0f}元"] for r in phase_rows.itertuples()])
    yearly_table=table(["年份", "策略平均收益", "起点最小至最大收益", "沪深300收益", "平均几何超额", "年内最差回撤"],
        [[r.period if r.period!="2026" else "2026截至10月13日",f"{r.mean_return:.2%}",f"{r.min_return:.2%} 至 {r.max_return:.2%}",f"{r.benchmark_return:.2%}",f"{r.mean_excess:.2%}",f"{r.worst_drawdown:.2%}"] for r in years.itertuples()])
    version_table=table(["训练版本", "平均复合年化", "五起点最差回撤", "平均累计费用"],
        [[r.variant,f"{r.mean_cagr:.2%}",f"{r.worst_drawdown:.2%}",f"{r.mean_fees_cny:,.0f}元"] for r in versions.itertuples()])
    report=f"""# S6季度联合约束策略研究报告

当前选定版本为 **S6季度重训、L1/L2增强及低复杂度树**（`S6_head6_40_QREG`）。固定2026-10-13数据快照，回放2022-01-04至2026-10-13，五个预设调仓起点的平均复合年化为 **{full.mean_cagr:.2%}**，最差最大回撤为 **{full.worst_drawdown:.2%}**。这些是成本后的历史研究结果。

## 训练版本与收益口径

S6是组合和模型线的名称，不同重训与约束版本的结果不同。原年度版本约17.13%，原季度版本约17.36%；当前选定的是季度联合约束版本。四组使用相同历史区间、Top20前6只占股票仓位40%、最多替换6只及5日调仓。

{version_table}

![S6不同训练版本的收益与回撤](figures/version_comparison.png)

## 模型和可用信息

- 股票池采用历史CSI300成分区间。12个价量因子及3个基本面因子使用原始数值输入。
- 目标为 `Open(T+11)/Open(T+1)-1` 的每日横截面排名减0.5；T日收盘形成信号，次日开盘执行。
- 从2010年开始扩展训练。每个执行年前用此前三个年度验证折选定树数，年内树数保持固定；每季度创建新模型，从零拟合当时所有成熟样本。
- 训练与验证按真实 `label_end_date` 截止，未知的未来10日收益不进入训练。最近特征仍可用于预测。
- L1=1、L2=5、叶子数15、叶子最小样本数300，随机种子42。具体参数见[公开配置](../../configs/experiments/s6_head6_40_public.json)。
- 财务值按可核验披露时间对齐；当前快照包含2026半年报，年度EPS主要来自2025年。修订历史不完整，覆盖见[财务证据](../../docs/evidence/financial_coverage.json)。

## 组合和成交假设

初始资金100万元，目标股票仓位95%。Top20中前6只合计占股票仓位40%，即账户的38%；每只目标约6.33%。其余14只合计占账户57%，每只约4.07%。权重会随价格漂移。

每5个交易日调仓，最多替换6只；保留股票也可能交易目标仓位差。计入现金、整手、T+1、缺报价及不可成交处理。买入费率0.05%、卖出0.15%，每笔最低5元，已从净值扣除。现有统一涨跌幅过滤仍是简化。

所有账户在相同区间从现金开始，分别固定偏移0/1/2/3/4个交易日，跨年不重置。偏移期间保留现金；初期或不可成交时实际持仓可以不足20只。

## 净值和最大回撤

下图展示全部五条起点账户及同区间沪深300基准，没有挑选表现最好的起点。初始净值为1，回撤包括初始现金高点；展示的是每日收盘净值回撤，不含日内极端波动。

![S6五起点净值及回撤](figures/nav_drawdown.png)

{phase_table}

五起点平均累计净收益为{full.mean_return:.2%}，平均累计费用{full.mean_fees_cny:,.0f}元，日均双边成交额换手{full.mean_daily_turnover:.2%}，平均持仓{full.mean_holdings:.2f}只。同区间基准累计收益{full.benchmark_return:.2%}。

平均年化是每条账户 `(期末净值/初始净值)^(252/交易日数)-1` 的算术平均。不能把累计收益除以年数，也不能将五条净值平均后当作真实账户。最大回撤为各账户相对此前高点的最大跌幅，最差值取五起点中的最大者。

## 夏普和信息比率

本报告将固定年化无风险利率设为 **{ANNUAL_RISK_FREE_RATE:.0%}**，作为研究中的机会成本假设，不代表某家银行的历史报价。按有效年利率折算日利率 `rf_daily=(1+0.01)^(1/252)-1`，约为0.00394862%。

- 夏普：`mean(日净收益-rf_daily) / std(日净收益-rf_daily, ddof=1) * sqrt(252)`。
- 信息比率：`mean(日净收益-沪深300日收益) / std(日净收益-沪深300日收益, ddof=1) * sqrt(252)`。

五起点分别计算后，平均年化夏普为 **{full.mean_sharpe:.3f}**，范围{full.min_sharpe:.3f}至{full.max_sharpe:.3f}；平均年化信息比率为 **{full.mean_information_ratio:.3f}**，范围{full.min_information_ratio:.3f}至{full.max_information_ratio:.3f}。两项指标均使用成本后日收益和252交易日年化。

1%的基准只用于夏普计算，不会给研究账户的闲置现金增加利息，净值、年化收益和最大回撤保持原回放结果。信息比率相对沪深300计算，不扣除该无风险基准。历史冻结实验中的零无风险利率口径仍保留供追溯，当前报告以上述1%口径为准。

## 分年度收益

{yearly_table}

![S6逐年净收益与沪深300对比](figures/annual_returns.png)

年度柱形为五起点收益的算术平均，误差线为最小至最大值。2026是截至10月13日的累计收益，没有将其冒充完整年度。年内回撤在当年起点归一化计算；全期回撤不在跨年时重置。

## 研究结论和限制

2024年贡献显著，2025年虽然绝对收益为正，仍跑输沪深300。平均银行仓位约40.87%，组合存在明显风格依赖。当前版本最差回撤超过原30%历史筛选线，不能称为通过全部风险筛选。

2022至2026已反复参与策略研究；因果时间切分、正则化和多起点检查均不能消除历史选择偏差。真正的独立验证需要冻结规则后的新数据。

日线回放假设理想开盘成交。模拟执行使用09:31后限价单，可能出现价差或未成交；本报告不代表已实现的模拟盘或实盘收益。企业行动、财报修订档案及市场容量仍有建模限制。

## 数据与图表复算

本目录提供[五起点每日汇总](phase_daily.csv)、[逐起点指标](phase_metrics.csv)、[年度汇总](summary.csv)、[版本对照](version_comparison.csv)及[来源哈希](evidence.json)。只包含历史研究账户的汇总净值、收益、费用与持仓数量，不包含券商账户、订单或逐股票持仓。

```powershell
python -m scripts.build_current_strategy_report
```

该命令从仓库内汇总复算表格与PNG，不下载数据、不训练模型、不连接券商。完整研究复现另需原始快照和父实验，见[数据契约](../../docs/data_contract.md)。旧年度Top50结果见[历史报告](../../docs/history/annual_top50_report.md)。
"""
    (OUT / "README.md").write_text(report,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-evidence-from",type=Path,help="Read-only import from a verified local QREG experiment")
    args=parser.parse_args()
    if args.import_evidence_from:
        import_evidence(args.import_evidence_from)
    evidence=read_json(OUT / "evidence.json")
    if digest(OUT / "phase_daily.csv") != evidence["public_daily_sha256"]:
        raise ValueError("Public daily evidence changed")
    daily=pd.read_csv(OUT / "phase_daily.csv",parse_dates=["datetime"],float_precision="round_trip")
    validate_daily(daily)
    phases,aggregates=calculate_tables(daily)
    verify_summary(aggregates)
    source=pd.read_csv(SUMMARY)
    variants=[]
    for case,label in [("V0","年度原版"),("M6","半年原版"),("M3","季度原版"),("QREG","季度联合约束（当前）")]:
        row=source.loc[source["mode"].eq("history") & source.case.eq(case) & source.period_kind.eq("all")].iloc[0]
        variants.append(dict(variant=label,case=case,mean_cagr=row.mean_cagr,
            worst_drawdown=row.worst_drawdown,mean_fees_cny=row.mean_fees_cny))
    versions=pd.DataFrame(variants)
    phases.to_csv(OUT / "phase_metrics.csv",index=False,float_format="%.17g",lineterminator="\n")
    aggregates.to_csv(OUT / "summary.csv",index=False,float_format="%.17g",lineterminator="\n")
    versions.to_csv(OUT / "version_comparison.csv",index=False,float_format="%.17g",lineterminator="\n")
    make_figures(daily,aggregates,versions)
    make_report(phases,aggregates,versions)
    print(aggregates.loc[aggregates.period.eq("all")].to_string(index=False))
    print("Rebuilt the selected S6 report from aggregate research evidence.")


if __name__=="__main__":
    main()
