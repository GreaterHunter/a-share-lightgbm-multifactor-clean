"""Read-only market-data probe; never connect to a broker or fabricate bars."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/experiments/cn_intraday_data_feasibility_20261029'


def probe(spec):
    label, secid, period = spec
    params = dict(secid=secid, fields1='f1,f2,f3,f4,f5,f6',
                  fields2='f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                  klt=period, fqt='0', beg='20220101', end='20251231')
    row = dict(label=label, symbol=secid, period_minutes=int(period),
               requested_start='2022-01-01', requested_end='2025-12-31')
    try:
        r = requests.get('https://push2his.eastmoney.com/api/qt/stock/kline/get',
                         params=params, timeout=(5, 15))
        row['http_status'] = r.status_code
        r.raise_for_status()
        payload = r.json()
        bars = (payload.get('data') or {}).get('klines') or []
        dates = [b.split(',')[0] for b in bars]
        in_range = [d for d in dates if '2022-01-01' <= d[:10] <= '2025-12-31']
        row.update(rows=len(bars), first=min(dates) if dates else None,
                   last=max(dates) if dates else None,
                   rows_in_requested_range=len(in_range),
                   response_sha256=hashlib.sha256(r.content).hexdigest(),
                   status=('rejected_out_of_range' if len(in_range) != len(bars) else
                           'returned_requires_coverage_audit' if bars else 'no_bars'))
        if bars:
            (OUT / f'{label}_{period}min_response.json').write_text(
                json.dumps(payload, ensure_ascii=True), encoding='utf-8')
    except Exception as exc:
        # Do not serialize URLs/proxy configuration or account information.
        row.update(status='request_failed', error_type=type(exc).__name__)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths = [ROOT/'configs/futu_cn_forward_protocol.json',
             ROOT/'paper_futu/cn_rank80_strategy.py',
             ROOT/'paper_futu/cn_rank80_runtime.py']
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    specs = [('stock', '0.000001', '1'), ('stock', '0.000001', '5'),
             ('csi300', '1.000300', '1'), ('csi300', '1.000300', '5')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        probes = list(pool.map(probe, specs))
    after = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert before == after
    result = dict(
        checked_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        status='data_not_yet_qualified', probes=probes, protected_hashes=after,
        broker_connected=False, backtest_run=False, model_trained=False,
        contract=dict(market='CN_A', benchmark_intraday='CSI300 price index',
                      t_plus_one=True, take_profit_cooldown_sessions=0,
                      stop_loss_cooldown='sold on D; earliest rebuy D+1',
                      unchanged='V10 production strategy and schedule',
                      feature_count=29),
        other_sources=dict(
            local='No historical minute dataset found in project data inventory',
            akshare_1min='Installed stock_zh_a_hist_min_em requests ndays=5',
            baostock='Official documentation excludes index minute data; supports 5/15/30/60 minute stock bars',
            fmp='Connected endpoint rejects intraday-5-min; current server accepts EOD endpoints only'),
        required='Stocks and CSI300 aligned historical bars, turnover/volume units, timestamp semantics, corporate-action and membership checks; adequate pre-test training history')
    (OUT/'data_probe.json').write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
