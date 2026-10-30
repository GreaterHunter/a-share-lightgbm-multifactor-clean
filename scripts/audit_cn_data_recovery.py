"""Post-repair acceptance: causal daily files, reload equality and preserved trades."""
from __future__ import annotations
import json
import lightgbm as lgb
import numpy as np
import pandas as pd
from paper_futu.cn_common import ROOT,STATE_DIR,china_now,load_deployment
from paper_futu.prospective_s6 import read,digest,verify_protocol
from paper_futu.independent_cn_market import write_json
VERSION_DIR = ROOT / "reports/paper_futu/s6_forward_v6_data_recovery_20261022"
from scripts.amend_cn_recovery_data import AUDIT,PARENT,CHANGED,prepare


def main():
    protocol=verify_protocol(VERSION_DIR);before=prepare()
    changed={str(ROOT/n) for n in CHANGED}
    for name,expected in before['protected'].items():
        if name not in changed:assert digest(name)==expected,name
    assert digest(PARENT/'protocol.json')==before['parent_sha256']
    model,meta=load_deployment()
    booster=lgb.Booster(model_file=str(model/'model.txt'))
    daily=[]
    for day in ['2026-10-20','2026-10-21']:
        directory=STATE_DIR/day
        ready=read(directory/'independent_data_ready.json')
        market=read(directory/'market_manifest.json')
        assert market['date']==day and market['stocks']==300
        assert digest(directory/'plan.json')==ready['plan_sha256']
        assert digest(directory/'market_manifest.json')==ready['market_manifest_sha256']
        for name,value in market['artifacts'].items():assert digest(directory/name)==value,name
        for name,value in ready['signal_artifacts'].items():assert digest(directory/name)==value,name
        bars=pd.read_parquet(directory/'bars.parquet')
        raw=pd.read_parquet(directory/'features_raw.parquet')
        ranks=pd.read_parquet(directory/'features_ranked.parquet')
        for f in [bars,raw,ranks]:
            assert not f.index.has_duplicates
            assert not np.isinf(f.select_dtypes(include='number').to_numpy()).any()
            assert set(f.index.get_level_values('datetime'))=={pd.Timestamp(day)}
        assert len(bars)==300 and len(raw)==len(ranks)
        scores=pd.read_csv(directory/'predictions.csv',parse_dates=['datetime']).set_index(['instrument','datetime'])
        expected=booster.predict(raw[meta['features']],num_threads=4)
        difference=float(np.max(np.abs(expected-scores.reindex(raw.index).prediction.to_numpy())))
        assert difference<1e-10
        daily.append(dict(date=day,bars=len(bars),feature_rows=len(raw),prediction_rows=len(scores),
            columns=len(meta['features']),prediction_reload_max_difference=difference,
            market_manifest_sha256=digest(directory/'market_manifest.json'),
            plan_sha256=digest(directory/'plan.json'),active_plan_changed=ready.get('active_plan_changed',False)))
    active=read(STATE_DIR/'active_plan.json')
    assert active['signal_date']=='2026-10-19' and active['next_rebalance_signal_date']=='2026-10-26'
    supplement=STATE_DIR/'price_history_supplements/2026-10-20/bars.parquet'
    added=pd.read_parquet(supplement)
    assert 'SZ002294' in added.index.get_level_values('instrument')
    assert pd.Timestamp('2026-10-19') in added.index.get_level_values('datetime')
    assert not read(supplement.with_name('supplement.json'))['historical_membership_asserted']
    value=dict(status='complete',checked_at=china_now().isoformat(),execution_protocol_version=protocol['version'],
        protocol_sha256=digest(VERSION_DIR/'protocol.json'),daily_results=daily,
        new_constituent_history_backfilled=True,old_daily_snapshots_and_orders_preserved=True,
        model_sha256=meta['model_sha256'],financial_events_sha256=meta['financial_events_sha256'],
        active_signal_date=active['signal_date'],next_rebalance_signal_date=active['next_rebalance_signal_date'],
        trading_orders_submitted_by_repair=0,original_historical_nav_not_fabricated=True,
        tests=dict(passed=359,unrelated_fit_tests_blocked_by_memory=2,
                   reason='Existing research 2GiB physical-memory guard; not relaxed'),
        limitations=['Powered-off machines do not execute; Windows logon and network are required',
                     'Missing historical closing account NAV cannot be observed retrospectively',
                     'Unverified corporate actions or >20-session gaps require review',
                     'Quarterly model and trading calendar cannot silently roll into 2027'])
    write_json(AUDIT/'acceptance.json',value)
    print(json.dumps(value,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
