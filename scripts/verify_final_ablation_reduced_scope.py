"""Verify the authorized grid subset and one complete account before restart."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from paper_futu.final_ablation_core import (
    A, DATA, REPORT, SCHEDULES, code_hash, digest, parameter_configs,
    protected_state, read_json, save_json,
)
from paper_futu.run_final_ablation import ANCHOR, job_accounts, search_counts
from scripts.verify_final_ablation_recovery import verify_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-protocol", required=True, type=Path)
    args = parser.parse_args()
    before = protected_state()
    old = read_json(args.expanded_protocol)
    assert parameter_configs() == old["parameter_configs"][:24]
    assert 60 not in SCHEDULES
    assert not {1,3} & {r for values in SCHEDULES.values() for r in values}
    counts = search_counts()
    assert (counts["coarse_replays_max"],counts["fine_replays_max"],counts["h20_validation_fits_max"]) == (240,2112,504)
    started = time.time()
    path = REPORT / "reduced_scope_acceptance.json"
    record = dict(status="running", started=started, code_sha256=code_hash(),
        expanded_protocol_path=str(args.expanded_protocol.resolve()),
        expanded_protocol_sha256=digest(args.expanded_protocol),
        old_parameter_prefix_identical=True, search_counts=counts,
        snapshot_id=read_json(DATA/"manifest.json")["snapshot_id"])
    save_json(path, record)
    try:
        output = REPORT / "model_runs" / f"{A.name}.csv"
        # Fresh code identity reruns all annual folds and model save/reload checks.
        job_accounts(A, 20, "fixed", [dict(portfolio=ANCHOR, rebalance=10)], output, anchor=True)
        check = verify_output(output, ANCHOR["top_k"])
        record.update(status="complete", account=check,
            model_save_reload_checks="passed in fit_stream for all five annual models",
            elapsed_seconds=time.time()-started, protected_state_unchanged=before == protected_state())
        assert record["protected_state_unchanged"]
        save_json(path, record)
        print("REDUCED SCOPE ACCEPTANCE COMPLETE", flush=True)
    except BaseException as error:
        record.update(status="failed", error=repr(error), elapsed_seconds=time.time()-started,
                      protected_state_unchanged=before == protected_state())
        save_json(path, record)
        raise


if __name__ == "__main__":
    main()
