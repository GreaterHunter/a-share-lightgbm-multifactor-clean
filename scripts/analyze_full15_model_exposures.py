"""Read-only diagnostics of existing prediction streams and audited accounts."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analyze_top_candidate_overfit import BANKS as PREVIOUS_BANKS, digest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "reports/experiments/all_full15_offsets_asof_20261013"
SOURCE = ROOT / "reports/experiments/final_ablation_asof_20261013"
DATA = ROOT / "data/processed/final_ablation_asof_20261013"
OUT = BASE / "cross_model_diagnostic"
BANKS = {**PREVIOUS_BANKS, **dict(zip(
    "SH600908 SH600928 SH601128 SH601187 SH601577 SH601665 SH601860 SH601963 SH601997 SH603323 "
    "SZ001227 SZ002807 SZ002839 SZ002936 SZ002948 SZ002958 SZ002966 SZ002987".split(),
    "无锡银行 西安银行 常熟银行 厦门银行 长沙银行 齐鲁银行 紫金银行 重庆银行 贵阳银行 苏农银行 "
    "兰州银行 江阴银行 张家港行 郑州银行 青岛银行 青农商行 苏州银行 瑞丰银行".split()))}
STYLE = ["book_to_price", "earnings_to_price", "volatility_60d", "mom_20d", "mom_60d"]
FILES: set[Path] = set()


def read_parquet(path, **kwargs):
    FILES.add(Path(path))
    return pd.read_parquet(path, **kwargs)


def main():
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))
    FILES.update([BASE / "manifest.json", BASE / "all_full15_phase_ranking.csv", SOURCE / "model_ablation.csv"])
    ranking = pd.read_csv(BASE / "all_full15_phase_ranking.csv")
    candidates = {c["candidate_key"]: c for c in manifest["candidates"]}
    selected: dict[str, set[str]] = {}

    def select(key, role):
        selected.setdefault(key, set()).add(role)

    common = ranking.loc[(ranking.horizon == 20) & (ranking.rebalance == 10)
        & (ranking.top_k == 20) & (ranking.top_tier_size == 5) & (ranking.top_tier_weight == .25)
        & (ranking.n_drop == 3) & (ranking.months == 12)]
    for row in common.itertuples():
        select(row.candidate_key, "common_fixed" if row.policy == "fixed" else "common_tuned")
    streams = {}
    for candidate in candidates.values():
        streams.setdefault(candidate["prediction_path"], candidate)
    for path in streams:
        keys = {c["candidate_key"] for c in candidates.values() if c["prediction_path"] == path}
        select(ranking.loc[ranking.candidate_key.isin(keys)].iloc[0].candidate_key, "best_existing_for_stream")
    for key in ranking.head(10).candidate_key:
        select(key, "overall_top10")
    old = pd.read_csv(SOURCE / "model_ablation.csv")
    old = old.loc[old.period.eq("all")]
    for row in old.itertuples():
        streams.setdefault(row.prediction_path, dict(spec=row.spec, horizon=int(row.horizon),
            policy=row.policy, months=int(row.months), seed=int(row.seed), prediction_path=row.prediction_path))

    style = read_parquet(DATA / "rank_features.parquet", columns=STYLE,
                         filters=[("datetime", ">=", pd.Timestamp("2021-12-31"))]) + .5
    all_account_yearly = []

    @lru_cache(maxsize=None)
    def account(case):
        directory = BASE / "accounts" / case
        if not directory.exists():
            directory = SOURCE / "accounts" / case
        daily = read_parquet(directory / "daily.parquet")
        holdings = read_parquet(directory / "holdings.parquet")
        contributions = read_parquet(directory / "contributions.parquet")
        orders = read_parquet(directory / "orders.parquet")
        assert daily.index.is_unique and len(daily) == 1154
        assert not holdings.duplicated(["instrument", "datetime"]).any()
        fills = orders.loc[orders.notional.gt(0)]
        fees = fills.groupby(["datetime", "instrument"]).fee.sum().rename("fee")
        pnl = contributions.set_index(["datetime", "instrument"]).join(fees).fillna({"fee": 0})
        pnl["net_pnl"] = pnl.pnl_before_fees - pnl.fee
        nav = 1e6 + pnl.net_pnl.groupby(level="datetime").sum().reindex(daily.index, fill_value=0).cumsum()
        assert np.allclose(nav, daily.account, atol=1e-4, rtol=1e-10)
        assert np.isclose(fees.sum(), daily.fees_cash.sum(), atol=1e-4)
        holdings = holdings.merge(daily[["account"]], left_on="datetime", right_index=True)
        holdings["weight"] = holdings.market_value / holdings.account
        bank = holdings.loc[holdings.instrument.isin(BANKS)].groupby("datetime").weight.sum().reindex(daily.index, fill_value=0)
        largest = holdings.groupby("datetime").weight.max().reindex(daily.index, fill_value=0)
        hh = holdings.set_index(["instrument", "datetime"]).join(style)
        exposures = {}
        for factor in STYLE:
            valid = hh.loc[hh[factor].notna()]
            exposures[factor] = (valid.market_value * valid[factor]).groupby(level="datetime").sum() / valid.market_value.groupby(level="datetime").sum()
        exposure = pd.DataFrame(exposures).reindex(daily.index)
        exposure["bank_weight"] = bank
        exposure["largest_weight"] = largest
        stock = pnl.groupby(level="instrument").net_pnl.sum().sort_values(ascending=False)
        net_profit = float(daily.account.iloc[-1] - 1e6)
        yearly = []
        for year, days in daily.groupby(daily.index.year):
            total_return = float((1 + days.net_return).prod() - 1)
            benchmark = float((1 + days.bench).prod() - 1)
            record = dict(case_id=case, year=int(year), total_return=total_return,
                benchmark_return=benchmark, geometric_excess=(1 + total_return) / (1 + benchmark) - 1,
                **exposure.loc[days.index].mean().to_dict())
            yearly.append(record)
            all_account_yearly.append(record)
        no24 = daily.loc[daily.index.year != 2024]
        no22_24 = daily.loc[~daily.index.year.isin([2022, 2024])]
        summary = dict(case_id=case, annualized_return=float((1 + daily.net_return).prod() ** (252 / len(daily)) - 1),
            excluded2024_cagr=float((1 + no24.net_return).prod() ** (252 / len(no24)) - 1),
            excluded2022_2024_cagr=float((1 + no22_24.net_return).prod() ** (252 / len(no22_24)) - 1),
            bank_net_profit=float(stock.loc[stock.index.isin(BANKS)].sum()), net_profit=net_profit,
            positive_stock_profit=float(stock.clip(lower=0).sum()),
            top5_positive_profit=float(stock.clip(lower=0).head(5).sum()),
            mean_bank_weight=float(bank.mean()), mean_largest_weight=float(largest.mean()),
            maximum_largest_weight=float(largest.max()), fees_cny=float(fees.sum()),
            **{f"mean_{f}": float(exposure[f].mean()) for f in STYLE})
        return summary, yearly

    summaries, candidate_yearly = [], []
    for index, (key, roles) in enumerate(selected.items(), 1):
        candidate = candidates[key]
        results = [account(case) for case in candidate["case_ids"]]
        phase = pd.DataFrame([x[0] for x in results])
        yr = pd.DataFrame([y for x in results for y in x[1]]).groupby("year").mean(numeric_only=True)
        original = ranking.loc[ranking.candidate_key.eq(key)].iloc[0]
        assert np.isclose(phase.annualized_return.mean(), original.mean_cagr, atol=1e-12)
        record = dict(candidate_key=key, roles=";".join(sorted(roles)), spec=candidate["spec"],
            horizon=candidate["horizon"], rebalance=candidate["rebalance"], policy=candidate["policy"],
            months=candidate["months"], **candidate["portfolio"], ranking=int(original["rank"]),
            worst_drawdown=float(original.worst_drawdown), phase_std_cagr=float(original.phase_std_cagr),
            phase_count=len(phase), **phase.mean(numeric_only=True).to_dict())
        for year, row in yr.iterrows():
            candidate_yearly.append(dict(candidate_key=key, spec=candidate["spec"], roles=record["roles"], year=int(year), **row.to_dict()))
            for metric in ["bank_weight", "total_return", "book_to_price", "volatility_60d"]:
                record[f"{metric}_{year}"] = float(row[metric])
        summaries.append(record)
        if index % 5 == 0:
            print(f"Account diagnostics {index}/{len(selected)}", flush=True)
    legacy = []
    for row in old.itertuples():
        summary, yearly = account(row.case_id)
        record = dict(spec=row.spec, case_id=row.case_id, phase_count=1, offset=0,
            max_drawdown=float(-row.max_drawdown), **{k: v for k, v in summary.items() if k != "case_id"})
        for period in yearly:
            for metric in ["bank_weight", "total_return", "book_to_price", "volatility_60d"]:
                record[f"{metric}_{period['year']}"] = period[metric]
        legacy.append(record)

    @lru_cache(maxsize=5)
    def labels(horizon):
        return read_parquet(DATA / f"labels_h{horizon}.parquet", columns=["label"],
                            filters=[("datetime", ">=", pd.Timestamp("2021-12-31"))])

    def correlation_by_day(pair, ranked=False):
        if not ranked:
            pair = pair.groupby(level="datetime").rank(method="average")
        centered = pair - pair.groupby(level="datetime").transform("mean")
        sums = pd.DataFrame(dict(product=centered.prediction * centered.label,
            p2=centered.prediction ** 2, y2=centered.label ** 2)).groupby(level="datetime").sum()
        value = sums["product"] / np.sqrt(sums.p2 * sums.y2)
        return value.where(pair.groupby(level="datetime").size().ge(10))

    signal_yearly, signal_overall = [], []
    max_ic_reconciliation_error = 0.
    for index, (path, stream) in enumerate(streams.items(), 1):
        predictions = read_parquet(path, columns=["prediction", "execution_date"])
        assert predictions.index.is_unique and np.isfinite(predictions.prediction).all()
        frame = predictions.join(style).join(labels(stream["horizon"]))
        frame["bank"] = frame.index.get_level_values("instrument").isin(BANKS)
        frame["prediction_rank"] = frame.prediction.groupby(level="datetime").rank(method="first", ascending=False)
        chosen = frame.loc[frame.prediction_rank.le(20)]
        signal = chosen.groupby(level="datetime")[STYLE + ["bank"]].mean().rename(columns={"bank": "top20_bank_fraction"})
        signal["top1_is_bank"] = frame.loc[frame.prediction_rank.eq(1)].bank.droplevel("instrument")
        signal["universe_bank_fraction"] = frame.bank.groupby(level="datetime").mean()
        complete = frame.loc[frame.label.notna()]
        pair = complete[["prediction", "label"]]
        overall_ic = correlation_by_day(pair)
        bank_ic = correlation_by_day(pair.loc[complete.bank])
        nonbank_ic = correlation_by_day(pair.loc[~complete.bank])
        full_ranks = pair.groupby(level="datetime").rank(method="average")
        groups = [full_ranks.index.get_level_values("datetime"), complete.bank.to_numpy()]
        residual = full_ranks - full_ranks.groupby(groups).transform("mean")
        within_binary_group_ic = correlation_by_day(residual, ranked=True)
        signal["rank_ic"] = overall_ic
        signal["bank_only_rank_ic"] = bank_ic
        signal["nonbank_only_rank_ic"] = nonbank_ic
        signal["bank_group_demeaned_rank_ic"] = within_binary_group_ic
        execution = predictions.execution_date.groupby(level="datetime").first()
        signal["year"] = execution.reindex(signal.index).dt.year
        ic_path = Path(path).with_name(Path(path).stem + "_daily_ic.csv")
        FILES.add(ic_path)
        known_ic = pd.read_csv(ic_path, parse_dates=["datetime"]).set_index("datetime").rank_ic
        delta = (known_ic.reindex(overall_ic.index) - overall_ic).abs().max()
        assert delta < 1e-10
        max_ic_reconciliation_error = max(max_ic_reconciliation_error, float(delta))
        identifier = {k: stream[k] for k in ["spec", "horizon", "policy", "months", "seed"]}
        identifier["stream"] = Path(path).stem
        for year, group in signal.groupby("year"):
            signal_yearly.append(dict(**identifier, year=int(year), mature_ic_days=int(group.rank_ic.notna().sum()),
                **group.drop(columns="year").mean(numeric_only=True).to_dict()))
        signal_overall.append(dict(**identifier, mature_ic_days=int(signal.rank_ic.notna().sum()),
            **signal.drop(columns="year").mean(numeric_only=True).to_dict()))
        if index % 5 == 0:
            print(f"Prediction diagnostics {index}/{len(streams)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    tables = {"candidate_summary": pd.DataFrame(summaries), "candidate_yearly": pd.DataFrame(candidate_yearly),
              "legacy_ablation_single_offset": pd.DataFrame(legacy), "signal_yearly": pd.DataFrame(signal_yearly),
              "signal_overall": pd.DataFrame(signal_overall), "account_yearly": pd.DataFrame(all_account_yearly)}
    for name, frame in tables.items():
        frame.to_csv(OUT / (name + ".csv"), index=False, encoding="utf-8-sig")
    details = dict(selected_existing_configs=len(selected), audited_distinct_accounts=account.cache_info().currsize,
        full15_prediction_streams=29, all_prediction_streams_including_completed_legacy=len(streams),
        bank_names=BANKS, max_daily_ic_reconciliation_error=max_ic_reconciliation_error,
        methodology=[
            "Common model comparison: H20/R10, Top20 equal equity weights, max3 substitutions, 95% equity target, m12, seed42, fixed parameter policy; five existing phases.",
            "Also inspect each full15 prediction stream's best existing historical portfolio, overall Top10, and two tuned common-rule controls. These winner selections are descriptive and biased.",
            "Legacy 12-factor and leave-one-feature-out accounts are existing single-offset completed runs only; no new ablations have been run.",
            "42 bank identity lookup used descriptively; does not replace historical universe membership and is not a full industry classification.",
            "All model signal diagnostics select the highest-scoring 20 names each date equally; they are not additional trading backtests.",
            "Style percentiles are cross-sectional ranks in the existing snapshot; holding exposure is value-weighted excluding cash, bank weight divides by whole account NAV.",
            "Bank/nonbank group-demeaned IC ranks predictions and labels across all eligible names then removes each group's daily mean from both ranks. It controls only this binary classification, not all industries or other styles.",
            "Within-group IC requires at least10 complete pairs and nonconstant ranks. IC is descriptive with overlapping mature future labels, not a significance test.",
            "Year-exclusion statistics concatenate selected realized returns and annualize by retained trading days; not implementable counterfactual strategies.",
            "Historical accounts were already used during model/strategy selection. No formal overfit probability or clean out-of-sample alpha is established."
        ], sources={str(path): digest(path) for path in sorted(FILES)},
        classification_sources=["https://www.sse.com.cn/disclosure/fund/announcement/c/new/2022-03-31/512820_20220331_1_Xrwj9Zqm.pdf",
            "https://www.sse.com.cn/assortment/stock/list/info/announcement/index.shtml?productId=601825",
            "https://www.szse.cn/disclosure/notice/company/t20220114_590607.html"])
    (OUT / "manifest.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    cols = ["spec", "annualized_return", "worst_drawdown", "bank_weight_2024", "bank_weight_2025", "excluded2024_cagr", "excluded2022_2024_cagr"]
    print(tables["candidate_summary"].loc[tables["candidate_summary"].roles.str.contains("common_fixed"), cols].to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
