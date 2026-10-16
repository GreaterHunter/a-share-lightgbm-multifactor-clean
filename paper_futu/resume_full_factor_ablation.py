"""Resume only the two retained full-factor streams without changing science keys."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import sys
import time

import psutil

from paper_futu import run_final_ablation as runner
from paper_futu.final_ablation_core import (
    A, DATA, REPORT, SCHEDULES, Store, code_hash, digest, protected_state, read_json, save_json,
)

SCIENTIFIC_HASH = "003db1603e155bec273d419e6d3457db6752f5dabf2c9c58d6b175da4f306a16"
SNAPSHOT = "658e24bcaa36c1a52769c70871737e72c5c69a2aac79fb755e385377d5e87992"
ALLOWED_NAMES = ("lgb_15_rank_rank", "lgb_15_raw_rank")
STAGES = ["portfolio", "cadence", "robustness", "report"]
MODULE = "paper_futu.resume_full_factor_ablation"
SCOPE = "two_full_15_factor_streams_resume"
_ORIGINAL_STAGES = {name: getattr(runner, "stage_"+name) for name in STAGES}


def validate_spec(record):
    if (record.get("name") not in ALLOWED_NAMES or tuple(record.get("features", ())) != A.features
        or record.get("target_kind") != "rank" or record.get("engine") != "lightgbm"
        or record.get("window") != 0
        or record.get("input_kind") != ("raw" if record["name"] == ALLOWED_NAMES[1] else "rank")):
        raise ValueError(f"Spec outside the user-approved full-factor scope: {record.get('name')}")


def scoped_selection(selection):
    selected = [r for r in selection["selected"] if r["name"] in ALLOWED_NAMES]
    if {r["name"] for r in selected} != set(ALLOWED_NAMES):
        raise ValueError("Both existing full-factor specifications must be available")
    for record in selected:
        validate_spec(record)
    return {**selection, "selected":selected, "user_scope_revision":dict(
        scope=SCOPE, removed_from_future=[r["name"] for r in selection["selected"] if r["name"] not in ALLOWED_NAMES],
        historical_all_specs_and_results_retained=True)}


def selected_specs():
    records = read_json(REPORT/"model_selection.json")["selected"]
    for record in records:
        validate_spec(record)
    if {r["name"] for r in records} != set(ALLOWED_NAMES):
        raise ValueError("Retained model selection changed during the resumed study")
    return [runner.spec_from(r) for r in records]


def validate_candidates(filename):
    for candidate in read_json(REPORT/filename):
        validate_spec(candidate["spec"])


def guarded_stage(name):
    def execute():
        selected_specs()
        if name == "cadence":
            validate_candidates("cadence_candidates.json")
        elif name == "robustness":
            validate_candidates("robustness_candidates.json")
        elif name == "report":
            validate_spec(read_json(REPORT/"frozen_research_version.json")["configuration"]["spec"])
        _ORIGINAL_STAGES[name]()
        if name == "portfolio":
            validate_candidates("cadence_candidates.json")
        elif name == "cadence":
            validate_candidates("robustness_candidates.json")
        elif name == "robustness":
            validate_spec(read_json(REPORT/"frozen_research_version.json")["configuration"]["spec"])
    return execute


def scoped_save_json(path, value):
    if Path(path).name in {"runner_process.json", "protocol.json", "reduced_scope.json"}:
        value = {**value, "resume_entry":MODULE, "resume_entry_sha256":digest(__file__),
                 "retained_model_names":ALLOWED_NAMES, "execution_stages":STAGES,
                 "model_and_tune_stages_repeated":False}
    save_json(path, value)


def configure_runner():
    # Only process-local orchestration changes; original worker math and Store keys stay intact.
    runner.STAGES = list(STAGES)
    runner.MAX_SELECTED_SPECS = 2
    runner.SCOPE_REVISION = SCOPE
    runner.selected_specs = selected_specs
    runner.save_json = scoped_save_json
    for name in STAGES:
        setattr(runner, "stage_"+name, guarded_stage(name))


@contextmanager
def resume_lock():
    root = str(Path(__file__).resolve().parents[1])
    active = [p.info["pid"] for p in psutil.process_iter(["pid", "cmdline", "cwd"])
              if p.pid != os.getpid() and p.info["cmdline"] and p.info["cwd"] == root
              and any(m in p.info["cmdline"] for m in (MODULE, "paper_futu.run_final_ablation"))
              and "--dry-run" not in p.info["cmdline"]]
    if active:
        raise RuntimeError(f"Study processes already active: {active}")
    lock = REPORT/"resume_full_factor.lock"
    if lock.exists():
        old_pid = int(lock.read_text())
        if psutil.pid_exists(old_pid):
            raise RuntimeError(f"Resume lock still belongs to live PID {old_pid}")
        lock.unlink()
    descriptor = os.open(lock, os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(descriptor, "w") as output:
        output.write(str(os.getpid()))
    try:
        yield
    finally:
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


def prepare_resume():
    if code_hash() != SCIENTIFIC_HASH or read_json(DATA/"manifest.json")["snapshot_id"] != SNAPSHOT:
        raise ValueError("The approved scientific code or frozen snapshot changed")
    before = protected_state()
    store = Store()
    selection = scoped_selection(read_json(REPORT/"model_selection.json"))
    verified = []
    for record in selection["selected"]:
        spec = runner.spec_from(record)
        for horizon in SCHEDULES:
            for policy in ["fixed", "tuned"]:
                path = runner.prediction_path(spec, horizon, policy)
                meta = read_json(path.with_suffix(".json"))
                validate_spec(meta["spec"])
                if digest(path) != meta["prediction_sha256"]:
                    raise ValueError(f"Stitched predictions changed: {path}")
                for period in meta["periods"]:
                    config = {k:v for k,v in period.items() if k != "path"}
                    if store.cached("serve", config) is None:
                        raise ValueError(f"No verified current-code serving cache for {path}")
                verified.append(str(path))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archive = REPORT/"archives"/f"before_full_factor_resume_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ["model_selection.json", "protocol.json", "scope_confirmation_pending.json",
                 "model_ablation.csv", "pipeline_status.json", "runner_process.json"]:
        path = REPORT/name
        if path.exists():
            shutil.copy2(path, archive/name)
    counts = {}
    for path in (REPORT/"tasks").glob("*.json"):
        task = read_json(path)
        if task.get("code_sha256") == SCIENTIFIC_HASH and task.get("status") == "complete":
            counts[task["kind"]] = counts.get(task["kind"],0)+1
    save_json(REPORT/"model_selection.json", selection)
    result = dict(status="prepared", scope=SCOPE, entry_sha256=digest(__file__),
        scientific_code_sha256=SCIENTIFIC_HASH, snapshot_id=SNAPSHOT,
        retained_model_names=ALLOWED_NAMES, verified_annual_prediction_streams=verified,
        completed_tasks_retained=counts, search_counts=runner.search_counts(),
        execution_stages=STAGES, historical_model_ablation_specs=14,
        model_and_tune_stages_repeated=False, archive=str(archive),
        protected_state=before, protected_state_unchanged=before == protected_state(), updated=time.time())
    if not result["protected_state_unchanged"]:
        raise ValueError("Protected Futu files changed during scope preparation")
    save_json(REPORT/"full_factor_resume.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-10-13", choices=["2026-10-13"])
    parser.add_argument("--stage", default="all", choices=STAGES+["all"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    configure_runner()
    if args.dry_run:
        runner.main()
        return
    REPORT.mkdir(parents=True, exist_ok=True)
    with resume_lock():
        record = prepare_resume()
        record.update(status="running", runner_pid=os.getpid(), started=time.time())
        save_json(REPORT/"full_factor_resume.json", record)
        try:
            runner.main()
            record.update(status="complete", updated=time.time())
        except BaseException as error:
            record.update(status="failed", error=repr(error), updated=time.time())
            raise
        finally:
            save_json(REPORT/"full_factor_resume.json", record)


if __name__ == "__main__":
    main()
