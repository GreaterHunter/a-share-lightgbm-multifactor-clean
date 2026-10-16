"""Diagnose existing top-candidate accounts; does not train or replay any account."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/experiments/final_ablation_asof_20261013"
BASE = ROOT / "reports/experiments/all_full15_offsets_asof_20261013"
DATA = ROOT / "data/processed/final_ablation_asof_20261013"
OUT = BASE / "top_candidate_diagnostic"
CANDIDATE = "b3891f2971a5fd1a23fa"
BANKS = dict(zip(
    "SZ000001 SZ002142 SH600000 SH600015 SH600016 SH600036 SH600919 SH600926 "
    "SH601009 SH601166 SH601169 SH601229 SH601288 SH601328 SH601398 SH601658 "
    "SH601818 SH601939 SH601988 SH601998 SH601916 SH601838 SH601077 SH601825".split(),
    "平安银行 宁波银行 浦发银行 华夏银行 民生银行 招商银行 江苏银行 杭州银行 "
    "南京银行 兴业银行 北京银行 上海银行 农业银行 交通银行 工商银行 邮储银行 "
    "光大银行 建设银行 中国银行 中信银行 浙商银行 成都银行 渝农商行 沪农商行".split(),
))
STYLE = ["book_to_price", "earnings_to_price", "volatility_20d", "volatility_60d",
         "mom_20d", "mom_60d", "operating_cashflow_to_revenue"]


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    candidate = next(x for x in manifest["candidates"] if x["candidate_key"] == CANDIDATE)
    ranks = pd.read_parquet(DATA / "rank_features.parquet", columns=STYLE,
                            filters=[("datetime", ">=", pd.Timestamp("2022-01-01"))]) + .5
    files = [BASE / "manifest.json", BASE / "all_full15_phase_ranking.csv", DATA / "rank_features.parquet"]
    phase_rows, yearly_rows, pnl_series, yearly_pnl_series = [], [], [], []
    for offset, case in zip(candidate["offsets"], candidate["case_ids"]):
        directory = (SOURCE if case in manifest["known"] else BASE) / "accounts" / case
        names = ["daily.parquet", "holdings.parquet", "contributions.parquet", "orders.parquet", "metrics.csv"]
        files.extend(directory / name for name in names)
        daily = pd.read_parquet(directory / "daily.parquet")
        holdings = pd.read_parquet(directory / "holdings.parquet")
        pnl = pd.read_parquet(directory / "contributions.parquet")
        orders = pd.read_parquet(directory / "orders.parquet")
        metrics = pd.read_csv(directory / "metrics.csv")
        assert daily.index.is_unique and not holdings.duplicated(["datetime", "instrument"]).any()
        trades = orders.loc[orders.notional.gt(0)]
        fees = trades.groupby(["datetime", "instrument"]).fee.sum().rename("fee")
        pnl = pnl.set_index(["datetime", "instrument"]).join(fees).fillna({"fee": 0})
        pnl["net_pnl"] = pnl.pnl_before_fees - pnl.fee
        reconstructed = 1_000_000 + pnl.net_pnl.groupby(level="datetime").sum().reindex(daily.index, fill_value=0).cumsum()
        assert np.allclose(reconstructed, daily.account, atol=1e-4, rtol=1e-10)
        assert np.isclose(fees.sum(), daily.fees_cash.sum(), atol=1e-4)
        stocks = pnl.groupby(level="instrument").net_pnl.sum().sort_values(ascending=False)
        pnl_series.append(stocks.rename(offset))
        p = pnl.reset_index()
        p["year"] = p.datetime.dt.year
        yearly_pnl_series.append(p.groupby(["year", "instrument"]).net_pnl.sum().rename(offset))
        holdings = holdings.merge(daily[["account"]], left_on="datetime", right_index=True)
        holdings["weight"] = holdings.market_value / holdings.account
        bank_weight = holdings.loc[holdings.instrument.isin(BANKS)].groupby("datetime").weight.sum().reindex(daily.index, fill_value=0)
        largest = holdings.groupby("datetime").weight.max()
        joined = holdings.set_index(["instrument", "datetime"]).join(ranks)
        exposures = {}
        for factor in STYLE:
            valid = joined.loc[joined[factor].notna()]
            exposures[factor] = (valid[factor] * valid.market_value).groupby(level="datetime").sum() / valid.market_value.groupby(level="datetime").sum()
        exposures = pd.DataFrame(exposures).reindex(daily.index)
        exposures["bank_account_weight"] = bank_weight
        for _, row in metrics.loc[metrics.period.astype(str).str.fullmatch(r"202[2-6]")].iterrows():
            year = int(row.period)
            yearly_rows.append(dict(offset=offset, year=year,
                total_return=float(row.total_return), benchmark_return=float(row.benchmark_total_return),
                geometric_excess=float(row.total_excess_return), max_drawdown=float(row.max_drawdown),
                fees_cny=float(row.total_fees_cny), **exposures.loc[exposures.index.year == year].mean().to_dict()))
        remaining = daily.loc[~daily.index.year.isin([2022, 2024])]
        without24 = daily.loc[daily.index.year != 2024]
        year_log_return = np.log1p(daily.net_return).groupby(daily.index.year).sum()
        net_profit = daily.account.iloc[-1] - 1_000_000
        phase_rows.append(dict(offset=offset, net_profit=net_profit,
            annualized_return=float((1 + daily.net_return).prod() ** (252 / len(daily)) - 1),
            excluded_2022_2024_diagnostic_cagr=float((1 + remaining.net_return).prod() ** (252 / len(remaining)) - 1),
            excluded_2024_diagnostic_cagr=float((1 + without24.net_return).prod() ** (252 / len(without24)) - 1),
            log_growth_share_2022_2024=float(year_log_return.loc[[2022, 2024]].sum() / year_log_return.sum()),
            bank_net_profit=float(stocks.loc[stocks.index.isin(BANKS)].sum()),
            bank_share_net_profit=float(stocks.loc[stocks.index.isin(BANKS)].sum() / net_profit),
            mean_bank_account_weight=float(bank_weight.mean()), maximum_bank_account_weight=float(bank_weight.max()),
            mean_largest_account_weight=float(largest.mean()), maximum_single_stock_account_weight=float(largest.max()),
            top5_stock_share_net_profit=float(stocks.iloc[:5].sum() / net_profit),
            distinct_stocks=len(stocks)))

    prediction_path = Path(candidate["prediction_path"])
    ic_path = prediction_path.with_name(prediction_path.stem + "_daily_ic.csv")
    predictions = pd.read_parquet(prediction_path, columns=["execution_date"])
    dates = predictions.execution_date.groupby(level="datetime").first()
    ic = pd.read_csv(ic_path, parse_dates=["datetime"]).set_index("datetime")
    ic["year"] = dates.reindex(ic.index).dt.year
    ic_yearly = ic.groupby("year").agg(mean_rank_ic=("rank_ic", "mean"), mature_ic_days=("rank_ic", "count"))
    files.extend([prediction_path, ic_path])
    yearly = pd.DataFrame(yearly_rows)
    mean_yearly = yearly.groupby("year").mean(numeric_only=True).drop(columns="offset").join(ic_yearly)
    stock_pnl = pd.concat(pnl_series, axis=1).fillna(0)
    stock_pnl["mean_net_pnl_cny"] = stock_pnl.mean(axis=1)
    stock_pnl["bank_name"] = stock_pnl.index.map(BANKS)
    stock_pnl = stock_pnl.sort_values("mean_net_pnl_cny", ascending=False)
    yearly_stock_pnl = pd.concat(yearly_pnl_series, axis=1).fillna(0)
    yearly_stock_pnl["mean_net_pnl_cny"] = yearly_stock_pnl.mean(axis=1)
    ranking = pd.read_csv(BASE / "all_full15_phase_ranking.csv")
    nearby = ranking.loc[(ranking.spec == candidate["spec"]) & (ranking.horizon == 20)
        & (ranking.rebalance == 5) & (ranking.policy == candidate["policy"])
        & (ranking.months == 12) & (ranking.top_k == 5) & (ranking.n_drop == 2)]
    nearby = nearby.sort_values("top_tier_weight")
    phases = pd.DataFrame(phase_rows)
    reference_rows = []
    reference = ranking.loc[(ranking.spec == "lgb_15_rank_rank") & (ranking.horizon == 20)
        & (ranking.rebalance == 20) & (ranking.months == 6) & (ranking.policy == "fixed")
        & (ranking.top_k == 5) & (ranking.top_tier_weight == .6) & (ranking.n_drop == 5)]
    assert len(reference) == 1
    reference_candidate = next(x for x in manifest["candidates"] if x["candidate_key"] == reference.iloc[0].candidate_key)
    for offset, case in zip(reference_candidate["offsets"], reference_candidate["case_ids"]):
        directory = (SOURCE if case in manifest["known"] else BASE) / "accounts" / case
        daily = pd.read_parquet(directory / "daily.parquet")
        holding = pd.read_parquet(directory / "holdings.parquet")
        files.extend([directory / "daily.parquet", directory / "holdings.parquet"])
        holding = holding.merge(daily[["account"]], left_on="datetime", right_index=True)
        holding["bank_weight"] = holding.market_value / holding.account * holding.instrument.isin(BANKS)
        bank = holding.groupby("datetime").bank_weight.sum().reindex(daily.index, fill_value=0)
        other_years = daily.loc[~daily.index.year.isin([2022, 2024])]
        reference_rows.append(dict(offset=offset, mean_bank_weight=float(bank.mean()),
            excluding_2022_2024_cagr=float((1 + other_years.net_return).prod() ** (252 / len(other_years)) - 1),
            **{f"bank_weight_{year}": float(group.mean()) for year, group in bank.groupby(bank.index.year)}))
    reference_phases = pd.DataFrame(reference_rows)
    summary = dict(candidate=candidate, yearly_means=mean_yearly.reset_index().to_dict("records"),
        phase_means=phases.mean(numeric_only=True).drop(labels="offset").to_dict(),
        bank_profit_share_ratio_of_means=float(phases.bank_net_profit.mean() / phases.net_profit.mean()),
        top5_average_pnl_share=float(stock_pnl.mean_net_pnl_cny.iloc[:5].sum() / phases.net_profit.mean()),
        removed_two_years_negative_phases=int(phases.excluded_2022_2024_diagnostic_cagr.lt(0).sum()),
        frozen_reference_candidate=reference_candidate,
        frozen_reference_phase_means=reference_phases.mean(numeric_only=True).drop(labels="offset").to_dict(),
        bank_names=BANKS,
        bank_identity_sources=[
            "https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-03-31/512820_20220331_1_Xrwj9Zqm.pdf",
            "https://www.sse.com.cn/assortment/stock/list/info/announcement/index.shtml?productId=601825"],
        methodology=[
            "Five existing start offsets, no new model fits or account simulations.",
            "Yearly return columns are means of five separate account-period returns, not a tradable blended-account NAV.",
            "Bank weight is daily bank market value divided by total account NAV, including cash in denominator.",
            "Style percentiles are stock-value-weighted contemporaneous cross-sectional factor percentiles, then equal-day/equal-phase averaged; they are descriptive exposures, not factor-return attribution.",
            "Stock contributions subtract instrument transaction fees and reconcile to ending account profit.",
            "Removing years is a discontinuous return-dependency diagnostic, not a feasible alternative trading strategy.",
            "Rank IC includes only finite/mature 20-day labels, grouped by execution year.",
            "No statistical probability of overfitting or independent industry-adjusted alpha has been estimated."
        ], source_sha256={str(path): digest(path) for path in files})
    OUT.mkdir(parents=True, exist_ok=True)
    for name, frame in [("yearly_by_phase", yearly), ("yearly_means", mean_yearly),
                        ("phase_diagnostics", phases), ("stock_profit_contribution", stock_pnl),
                        ("stock_profit_by_year", yearly_stock_pnl), ("same_model_concentration", nearby),
                        ("frozen_reference_diagnostic", reference_phases)]:
        frame.to_csv(OUT / (name + ".csv"), encoding="utf-8-sig", index=name not in ["yearly_by_phase", "phase_diagnostics", "same_model_concentration", "frozen_reference_diagnostic"])
    (OUT / "diagnostic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(mean_yearly.to_string())
    print(json.dumps({"phase_means": summary["phase_means"], "bank_profit_share_ratio_of_means": summary["bank_profit_share_ratio_of_means"], "top5_average_pnl_share": summary["top5_average_pnl_share"]}, ensure_ascii=False))
    print(f"Saved diagnostics: {OUT}")


if __name__ == "__main__":
    main()
