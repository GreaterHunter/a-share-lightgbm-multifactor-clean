"""Read-only count of remaining portfolio work using the unchanged scientific keys."""
from __future__ import annotations

import json
import pandas as pd

from paper_futu.final_ablation_core import REPORT, SCHEDULES, Store, digest, portfolios, read_json
from paper_futu.run_final_ablation import COARSE, fine_pairs
from paper_futu.resume_full_factor_ablation import ALLOWED_NAMES


def main():
    store = Store()
    planned, streams = {}, []
    for name in ALLOWED_NAMES:
        for policy in ["fixed", "tuned"]:
            frames = [pd.read_csv(REPORT/"portfolio_runs"/f"{name}_{policy}_h{h}_coarse.csv") for h in SCHEDULES]
            pairs = fine_pairs(pd.concat(frames,ignore_index=True))
            streams.append(dict(spec=name,policy=policy,fine_pairs=sorted(pairs)))
            predictions = {}
            for horizon in SCHEDULES:
                path = REPORT/"predictions"/f"{name}_h{horizon}_{policy}_m12_s42.parquet"
                predictions[horizon] = (str(path),digest(path))
            cases = [(h,r,p,"coarse") for h,rs in SCHEDULES.items() for r in rs for p in COARSE]
            cases += [(h,r,p,"fine") for h,r in sorted(pairs) for p in portfolios()]
            for h,r,p,stage in cases:
                path,sha = predictions[h]
                config = dict(prediction_path=path,prediction_sha256=sha,portfolio=p,rebalance=r,
                    offset=0,friction=0.,initial_cash=1_000_000,risk_degree=.95,trade_unit=100,
                    execution="T+1 open",annualization=252,engine="audited_qlib_v2")
                key = store.key("account",config)
                task_path = REPORT/"tasks"/f"{key}.json"
                status = read_json(task_path).get("status") if task_path.exists() else "not_started"
                planned.setdefault(key,dict(status=status,stages=set()))["stages"].add(stage)
    statuses = {}
    for item in planned.values():
        statuses[item["status"]] = statuses.get(item["status"],0)+1
    print(json.dumps(dict(scientific_code_sha256=store.code,streams=streams,
        planned_unique_portfolio_accounts=len(planned),status_counts=statuses,
        remaining_new_portfolio_accounts=sum(v for s,v in statuses.items() if s != "complete"),
        note="Counts complete manifests by original key; the worker verifies artifact hashes before reuse."),indent=2))


if __name__ == "__main__":
    main()
