"""Run the failed study case and all three causal cadence pilots before restart."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from paper_futu.final_ablation_core import (
    A, AS_OF, REPORT, code_hash, fee, initial_specs, protected_state, save_json,
)
from paper_futu.run_final_ablation import ANCHOR, job_accounts, job_stream, table


def verify_output(path, top_k):
    summary = pd.read_csv(path)
    whole = summary.loc[summary.period.eq("all")].iloc[0]
    directory = REPORT / "accounts" / whole.case_id
    daily = pd.read_parquet(directory / "daily.parquet")
    orders = pd.read_parquet(directory / "orders.parquet")
    holdings = pd.read_parquet(directory / "holdings.parquet")
    assert daily.index[-1] == AS_OF
    assert daily.holdings_count.max() <= top_k
    assert orders.holdings_before.max() <= top_k and orders.holdings_after.max() <= top_k
    assert orders.cash_before.min() >= -1e-5 and orders.cash_after.min() >= -1e-5
    assert holdings.cash.min() >= -1e-5
    expected_fees = [fee(r.notional, r.direction, r.friction) for r in orders.itertuples()]
    assert np.allclose(orders.fee, expected_fees, atol=1e-6)
    signed = np.where(orders.direction.eq(1), -orders.notional, orders.notional)-orders.fee
    assert np.allclose(orders.cash_before+signed, orders.cash_after, atol=1e-5)
    filled_buys = orders.loc[orders.direction.eq(1) & orders.notional.gt(0)]
    assert np.allclose(filled_buys.raw_quantity/100, np.round(filled_buys.raw_quantity/100), atol=1e-6)
    assert orders.groupby("datetime").direction.apply(lambda x: x.is_monotonic_increasing).all()
    annual = summary.loc[summary.period.astype(str).str.fullmatch(r"202[2-6]")]
    assert np.isclose(np.prod(1+annual.total_return)-1, whole.total_return, atol=1e-10)
    return dict(case_id=whole.case_id, output=str(path), account_directory=str(directory),
        daily_rows=len(daily), maximum_holdings=int(daily.holdings_count.max()),
        minimum_cash=float(orders.cash_after.min()), order_count=len(orders),
        blocked_new_name_orders=int(orders.reason.eq("no_slot_after_actual_sells").sum()),
        annualized_return=float(whole.annualized_return), max_drawdown=float(whole.max_drawdown),
        ledger_and_nav_checks="passed")


def main():
    before = protected_state()
    started = time.time()
    audit_path = REPORT / "recovery_acceptance.json"
    record = dict(status="running", started=started, code_sha256=code_hash(), checks=[])
    save_json(audit_path, record)
    try:
        failed = next(s for s in initial_specs() if s.name == "lgb_15_rank_raw")
        output = REPORT / "model_runs" / f"{failed.name}.csv"
        job_accounts(failed, 20, "fixed", [dict(portfolio=ANCHOR, rebalance=10)], output, anchor=True)
        record["checks"].append(dict(name="original_failed_case", **verify_output(output, ANCHOR["top_k"])))
        save_json(audit_path, record)
        pilot_paths = []
        for months in [12, 6, 3]:
            job_stream(A, 20, months=months)
            output = REPORT / "cadence_runs" / f"pilot_m{months}.csv"
            job_accounts(A, 20, "fixed", [dict(portfolio=ANCHOR, rebalance=10)], output, months=months)
            pilot_paths.append(output)
            record["checks"].append(dict(name=f"cadence_pilot_m{months}", **verify_output(output, ANCHOR["top_k"])))
            save_json(audit_path, record)
        table(pilot_paths, REPORT / "cadence_pilot.csv")
        record.update(status="complete", elapsed_seconds=time.time()-started,
                      protected_state_unchanged=before == protected_state())
        assert record["protected_state_unchanged"]
        save_json(audit_path, record)
        print(f"ACCEPTANCE COMPLETE: {len(record['checks'])} cases", flush=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error), elapsed_seconds=time.time()-started,
                      protected_state_unchanged=before == protected_state())
        save_json(audit_path, record)
        raise


if __name__ == "__main__":
    main()
