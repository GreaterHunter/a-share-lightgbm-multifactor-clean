"""Execute a saved CSI300 plan on the next session, using only CN paper funds."""

from __future__ import annotations

import argparse
import csv
import json
import math
import socket
import time

import pandas as pd

from paper_futu.check_connection import stock_paper_accounts
from paper_futu.cn_common import CHINA_TZ, STATE_DIR, china_now, load_deployment, select_holdings, select_audited_holdings, sellable_quantity, target_share_quantity, tiered_weights, validate_execution_session, validate_plan_deployment
from paper_futu.smoke_order import TERMINAL, append_event, require_ok
from paper_futu.cn_broker_queries import query_positions, PositionCache


def account_context():
    from futu import OpenSecTradeContext, SecurityFirm, TrdMarket
    return OpenSecTradeContext(filter_trdmarket=TrdMarket.CN, host="127.0.0.1", port=11111,
                               security_firm=SecurityFirm.FUTUSECURITIES)


def paper_account_id(trade):
    paper = stock_paper_accounts(require_ok(trade.get_acc_list(), "读取沪深模拟账户"), "CN")
    if len(paper) != 1 or str(paper.iloc[0].acc_id)[-4:] != "0000":
        raise ValueError("沪深股票模拟账户不唯一或不是已确认的账户尾号 2301")
    return int(paper.iloc[0].acc_id)


def save_account_snapshot(trade, account_id):
    from futu import Currency, OpenQuoteContext, TrdEnv
    assets = require_ok(trade.accinfo_query(trd_env=TrdEnv.SIMULATE, acc_id=account_id,
                                           refresh_cache=True, currency=Currency.CNH), "查询模拟资产")
    positions = query_positions(trade, account_id)
    now = china_now()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = STATE_DIR / "account_baseline.json"
    if not baseline_path.exists():
        baseline_path.write_text(json.dumps({"time": now.isoformat(), "initial_assets": float(assets.iloc[0].total_assets),
                                             "account_suffix": "0000"}, indent=2), encoding="utf-8")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    benchmark_context = OpenQuoteContext(host="127.0.0.1", port=11111)
    try:
        benchmark = require_ok(benchmark_context.get_market_snapshot(["SH.000300"]), "查询沪深300基准")
        benchmark_price = float(benchmark.iloc[0].last_price)
    finally:
        benchmark_context.close()
    if "benchmark_price" not in baseline:
        baseline["benchmark_price"] = benchmark_price
        baseline["benchmark_start_time"] = now.isoformat()
        baseline_path.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    row = {"time": now.isoformat(), "total_assets": float(assets.iloc[0].total_assets),
           "cash": float(assets.iloc[0].cash), "market_val": float(assets.iloc[0].market_val),
           "holdings": int(positions.qty.gt(0).sum()), "return_since_start": float(assets.iloc[0].total_assets) / baseline["initial_assets"] - 1,
           "benchmark_price": benchmark_price,
           "benchmark_return": benchmark_price / baseline["benchmark_price"] - 1,
           "geometric_excess_return": (float(assets.iloc[0].total_assets) / baseline["initial_assets"])
                                      / (benchmark_price / baseline["benchmark_price"]) - 1}
    path = STATE_DIR / "account_snapshots.csv"
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.reader(source))
        header = rows[0]
        columns = list(row)
        extended_header = header + [name for name in columns if name not in header]
        earlier = pd.DataFrame([dict(zip(extended_header, values)) for values in rows[1:]])
        pd.concat([earlier, pd.DataFrame([row])], ignore_index=True).reindex(columns=columns).to_csv(path, index=False)
    else:
        pd.DataFrame([row]).to_csv(path, index=False)
    positions.to_csv(STATE_DIR / "positions_latest.csv", index=False)
    return assets.iloc[0], positions


def fresh_quotes(quote, codes, execution=False):
    parts = []
    for begin in range(0, len(codes), 100):
        parts.append(require_ok(quote.get_market_snapshot(codes[begin:begin + 100]), "查询交易行情"))
    frame = pd.concat(parts).set_index("code", drop=False)
    if set(frame.index) != set(codes):
        raise ValueError("交易行情缺少代码")
    if execution:
        now = china_now()
        for row in frame.itertuples():
            if not row.suspension:
                timestamp = pd.Timestamp(row.update_time).tz_localize(CHINA_TZ)
                if timestamp.date() != now.date() or abs((pd.Timestamp(now) - timestamp).total_seconds()) > 120:
                    raise ValueError(f"交易行情过期：{row.code}, {row.update_time}")
    return frame


def main():
    from paper_futu.cn_execution_session import rebalance_main
    return rebalance_main()


if __name__ == "__main__":
    main()
