"""Package completed research artifacts without repeating any scientific calculation.

Paths in the v2 manifest are relative to the project root. Mutable control files
are classified, not assigned pretend hashes. Unknown static read errors fail.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time

import pandas as pd
import psutil

from paper_futu.final_ablation_core import (
    A, DATA, MODELS, REPORT, SCHEDULES, Store, code_hash, digest, offsets,
    portfolios, protected_state, read_json, save_json,
)
from paper_futu import run_final_ablation as science
from paper_futu import resume_full_factor_ablation as scope
from paper_futu.parallel_training_control import account_configuration, file_lock
from scripts.inspect_final_ablation import research_command

PROJECT = Path(__file__).resolve().parents[1]
MUTABLE_DIRECTORIES = {
    "logs": "append-only execution logs, not scientific result artifacts",
    "monitor": "live monitoring observations",
    "mlruns": "mutable tracker internals; authoritative completed task artifacts are included separately",
    "parallel_training": "live scheduler telemetry, locks, private fragments and worker control records",
}
MUTABLE_ROOT_FILES = {
    "pipeline_status.json", "parallel_resume.json", "full_factor_resume.json",
    "runner_process.json", "current_job.json", "scope_confirmation_pending.json",
    "delivery_finalization.json", "delivery_manifest.json", "delivery_manifest.preview.json",
}
REQUIRED_OUTPUTS = [
    "protocol.json", "model_selection.json", "model_ablation.csv", "parameter_search.csv",
    "portfolio_coarse.csv", "portfolio_fine.csv", "cadence_pilot.csv", "cadence_comparison.csv",
    "robustness_all.csv", "randomness.csv", "final_selection.csv", "reference.csv",
    "frozen_research_version.json", "frozen_latest_ranking.csv", "frozen_vs_reference_yearly.csv",
    "research_report.html",
]


def relative(path, project=PROJECT):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(project).resolve()):
        raise ValueError(f"Artifact outside project: {path}")
    return path.relative_to(Path(project).resolve()).as_posix()


def exclusion_reason(path, report, incomplete_ids=()):
    path, report = Path(path), Path(report)
    if path.suffix.lower() in {".lock", ".tmp", ".log"}:
        return "lock, temporary file, or mutable execution log"
    if path.is_relative_to(report):
        parts = path.relative_to(report).parts
        if parts and parts[0] in MUTABLE_DIRECTORIES:
            return MUTABLE_DIRECTORIES[parts[0]]
        if len(parts) == 1 and path.name in MUTABLE_ROOT_FILES:
            return "live completion/control metadata or self-referential manifest"
        if len(parts) >= 2 and parts[0] == "accounts" and parts[1] in incomplete_ids:
            return "retained incomplete historical account artifacts"
        if len(parts) == 2 and parts[0] == "tasks" and path.stem in incomplete_ids:
            return "retained incomplete historical task; not promoted to a completed result"
    return None


def artifact_inventory(project, report, data, models, incomplete_ids=(), extra_paths=()):
    included, excluded = set(), []
    for root in (Path(report), Path(data), Path(models)):
        if not root.is_dir():
            raise FileNotFoundError(root)
        for directory, folders, names in os.walk(root):
            directory = Path(directory)
            for folder in list(folders):
                child = directory/folder
                reason = exclusion_reason(child, report, incomplete_ids)
                if reason:
                    folders.remove(folder)
                    excluded.append(dict(path=relative(child,project),kind="directory_prefix",reason=reason))
            for name in names:
                path = directory/name
                reason = exclusion_reason(path, report, incomplete_ids)
                if reason:
                    excluded.append(dict(path=relative(path,project),kind="file",reason=reason))
                else:
                    included.add(path.resolve())
    included.update(Path(p).resolve() for p in extra_paths)
    for path in included:
        relative(path,project)
    return sorted(included), sorted(excluded,key=lambda r:r["path"])


def build_manifest(project, report, data, models, *, incomplete_ids=(), extra_paths=(), metadata=None, hash_file=digest):
    included, excluded = artifact_inventory(project,report,data,models,incomplete_ids,extra_paths)
    # No exception suppression: permission errors in unknown/static files must fail delivery.
    files = {relative(path,project):hash_file(path) for path in included}
    return dict(schema_version=2,path_base="project_root",files=files,excluded=excluded,
        excluded_directory_rules=MUTABLE_DIRECTORIES,excluded_root_control_files=sorted(MUTABLE_ROOT_FILES),
        incomplete_historical_task_ids=sorted(incomplete_ids),metadata=metadata or {})


def verify_manifest(manifest, project=PROJECT, hash_file=digest, inventory_roots=None, extra_paths=()):
    for path, expected in manifest["files"].items():
        absolute = Path(project)/path
        relative(absolute,project)
        if hash_file(absolute) != expected:
            raise ValueError(f"Delivery artifact changed: {path}")
    if inventory_roots:
        report,data,models = inventory_roots
        actual,_ = artifact_inventory(project,report,data,models,
            manifest["incomplete_historical_task_ids"],extra_paths)
        if {relative(p,project) for p in actual} != set(manifest["files"]):
            raise ValueError("Immutable delivery inventory changed; unknown new static files cannot be silently skipped")
    return len(manifest["files"])


def assert_no_research_processes():
    active = []
    for process in psutil.process_iter(["pid","cmdline","cwd"]):
        command = process.info["cmdline"] or []
        if research_command(command) or research_command(command,worker=True):
            if process.info["cwd"] is None:
                raise RuntimeError(f"Cannot establish study process cwd: {process.pid}")
            if Path(process.info["cwd"]).resolve() == PROJECT:
                active.append(process.pid)
    if active:
        raise RuntimeError(f"Do not package a live study: {active}")


def locked_digest_probe(path):
    with file_lock(path,timeout=0):
        try:
            digest(path)
        except PermissionError as error:
            result = dict(reproduced=True,error=repr(error),errno=error.errno,
                          explanation="Windows byte-range lock blocks even the same process's independent read handle")
        else:
            if os.name == "nt":
                raise RuntimeError("Expected Windows controller byte-lock read failure was not reproduced")
            result = dict(reproduced=False,explanation="This OS allows reading advisory-locked files")
    result["read_after_lock_release_sha256"] = digest(path)
    return result


def task_inventory():
    complete, incomplete, counts = {}, set(), {}
    for path in sorted((REPORT/"tasks").glob("*.json")):
        task = read_json(path)
        if task["task_id"] != path.stem:
            raise ValueError(f"Task filename/id mismatch: {path}")
        current = task["code_sha256"] == scope.SCIENTIFIC_HASH
        bucket = ("current" if current else "historical")+":"+task["kind"]+":"+task["status"]
        counts[bucket] = counts.get(bucket,0)+1
        if current and task["status"] != "complete":
            raise ValueError(f"Current-code scientific task is unfinished: {path}")
        if task["status"] == "complete":
            complete[path.stem] = task
        else:
            incomplete.add(path.stem)
    return complete,incomplete,counts


def calculation_plan(complete):
    """Reconstruct existing keys only; never calls a training or replay function."""
    store = object.__new__(Store)
    store.code,store.data = scope.SCIENTIFIC_HASH,scope.SNAPSHOT
    accounts,streams,dependencies = {},{},set()
    selection = read_json(REPORT/"model_selection.json")
    if {r["name"] for r in selection["selected"]} != set(scope.ALLOWED_NAMES):
        raise ValueError("Final retained specifications changed")
    for record in selection["selected"]:
        scope.validate_spec(record)
    if len(selection["all_specs"]) != 14:
        raise ValueError("Historical fourteen-model ablation was not preserved")

    def account(record,horizon,policy,case,months=12,seed=42):
        spec = science.spec_from(record)
        path = science.prediction_path(spec,horizon,policy,months,seed)
        streams[str(path)] = path
        sha = read_json(path.with_suffix(".json"))["prediction_sha256"]
        key = store.key("account",account_configuration(path,case,sha))
        accounts[key] = dict(spec=record["name"],horizon=horizon,policy=policy,months=months,seed=seed)

    for record in selection["all_specs"]:
        account(record,20,"fixed",dict(portfolio=science.ANCHOR,rebalance=10))
    for record in selection["selected"]:
        for policy in ["fixed","tuned"]:
            coarse = []
            for horizon,rs in SCHEDULES.items():
                for r in rs:
                    for p in science.COARSE:
                        account(record,horizon,policy,dict(portfolio=p,rebalance=r))
                coarse.append(pd.read_csv(REPORT/"portfolio_runs"/f"{record['name']}_{policy}_h{horizon}_coarse.csv"))
            for h,r in sorted(science.fine_pairs(pd.concat(coarse,ignore_index=True))):
                for p in portfolios():
                    account(record,h,policy,dict(portfolio=p,rebalance=r))
    for months in [12,6,3]:
        account(A.record(),20,"fixed",dict(portfolio=science.ANCHOR,rebalance=10),months)
    cadence = read_json(REPORT/"cadence_candidates.json")
    robust = read_json(REPORT/"robustness_candidates.json")
    for c in cadence:
        scope.validate_spec(c["spec"])
        for months in [12,6,3]:
            account(c["spec"],c["horizon"],c["policy"],dict(portfolio=c["portfolio"],rebalance=c["rebalance"]),months)
    for c in robust:
        scope.validate_spec(c["spec"])
        for o in offsets(c["rebalance"]):
            for f in [0.,.001,.002]:
                account(c["spec"],c["horizon"],c["policy"],dict(portfolio=c["portfolio"],rebalance=c["rebalance"],offset=o,friction=f),c["months"])
    frozen = read_json(REPORT/"frozen_research_version.json")
    if frozen["code_sha256"] != scope.SCIENTIFIC_HASH or frozen["snapshot_id"] != scope.SNAPSHOT:
        raise ValueError("Frozen winner provenance changed")
    scope.validate_spec(frozen["configuration"]["spec"])
    _,order = science.choose_final(robust,pd.read_csv(REPORT/"robustness_all.csv",float_precision="round_trip"))
    if order != frozen["selection_order"]:
        raise ValueError("Complete seed/winner selection differs from the original scientific stage")
    for candidate_id in order[:3]:
        c = next(x for x in robust if x["candidate_id"] == candidate_id)
        for seed in [2024,2025]:
            for o in offsets(c["rebalance"]):
                account(c["spec"],c["horizon"],c["policy"],dict(portfolio=c["portfolio"],rebalance=c["rebalance"],offset=o),c["months"],seed)
    account(A.record(),10,"fixed",dict(portfolio=science.COARSE[1],rebalance=10))
    for path in streams.values():
        meta = read_json(path.with_suffix(".json"))
        if digest(path) != meta["prediction_sha256"]:
            raise ValueError(f"Stitched prediction changed: {path}")
        for period in meta["periods"]:
            dependencies.add(store.key("serve",{k:v for k,v in period.items() if k != "path"}))
            dependencies.add(store.key("final_fit",{k:period[k] for k in
                ["spec","horizon","config","trees","seed","train_start","cutoff"]}))
    for key in set(accounts)|dependencies:
        if key not in complete or complete[key]["code_sha256"] != scope.SCIENTIFIC_HASH or complete[key]["snapshot_id"] != scope.SNAPSHOT:
            raise ValueError(f"Required original-key computation not complete under approved provenance: {key}")
    for name in REQUIRED_OUTPUTS:
        if not (REPORT/name).is_file():
            raise FileNotFoundError(REPORT/name)
    return dict(required_unique_accounts=len(accounts),required_prediction_streams=len(streams),
        required_fit_and_serve_tasks=len(dependencies),required_account_keys=sorted(accounts),
        required_prediction_paths=[relative(p) for p in sorted(streams.values())],selection_order=order,
        seed_pretraining_full_selection_matches_source=True,model_ablation_specs_retained=14)


def verify_execution_sources(record):
    bindings = {
        "resume_entry_sha256":"paper_futu/resume_parallel_full_factor_ablation.py",
        "base_resume_entry_sha256":"paper_futu/resume_full_factor_ablation.py",
        "training_worker_sha256":"paper_futu/parallel_full_factor_worker.py",
        "account_worker_sha256":"paper_futu/parallel_account_worker.py",
        "training_scheduling_sha256":"paper_futu/parallel_training_control.py",
    }
    for key,path in bindings.items():
        if digest(PROJECT/path) != record[key]:
            raise ValueError(f"Approved non-scientific execution source changed: {path}")
    return {path:record[key] for key,path in bindings.items()}


def archive_failure(record):
    previous = REPORT/"delivery_finalization.json"
    if previous.exists():
        path = PROJECT/read_json(previous)["failure_archive"]
        if path.is_dir():
            return path
    archive = REPORT/"archives"/f"report_delivery_failure_{time.strftime('%Y%m%d_%H%M%S')}"
    archive.mkdir(parents=True,exist_ok=False)
    for name in ["pipeline_status.json","parallel_resume.json","runner_process.json","current_job.json","protocol.json"]:
        shutil.copy2(REPORT/name,archive/(Path(name).stem+"_failed.json"))
    for key in ["stdout_log","stderr_log"]:
        source = Path(record[key])
        if not source.resolve().is_relative_to(REPORT.resolve()):
            raise ValueError(f"Failure log outside report root: {source}")
        shutil.copy2(source,archive/("runner_"+key+"_failed.txt"))
    save_json(archive/"failure_explanation.json",dict(
        original_error=read_json(REPORT/"pipeline_status.json"),
        cause="stage_report attempted to digest the OS-byte-locked parallel controller.lock",
        scope="delivery packaging only; no scientific rerun",scientific_code_sha256=scope.SCIENTIFIC_HASH))
    return archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only",action="store_true")
    mode.add_argument("--finalize",action="store_true")
    mode.add_argument("--verify-only",action="store_true")
    parser.add_argument("--review-files",nargs="+",default=[])
    args = parser.parse_args(argv)
    if args.verify_only:
        m = read_json(REPORT/"delivery_manifest.json")
        extras = [PROJECT/p for p in m["metadata"]["execution_sources"]]
        count = verify_manifest(m,inventory_roots=(REPORT,DATA,MODELS),extra_paths=extras)
        print(f"VERIFIED {count} immutable delivery files; mutable logs/monitor/locks remain excluded",flush=True)
        return
    assert_no_research_processes()
    if code_hash() != scope.SCIENTIFIC_HASH or read_json(DATA/"manifest.json")["snapshot_id"] != scope.SNAPSHOT:
        raise ValueError("Approved science or frozen snapshot changed")
    if args.finalize and not args.review_files:
        parser.error("--finalize requires the parent's completed --review-files before sealing")
    reviews = [Path(p).resolve() for p in args.review_files]
    for path in reviews:
        if not path.is_file() or not path.is_relative_to(REPORT.resolve()) or exclusion_reason(path,REPORT):
            raise ValueError(f"Review must be an existing immutable file inside the report: {path}")
    with file_lock(REPORT/"delivery_finalize.lock",timeout=0):
        before = protected_state()
        runner = read_json(REPORT/"runner_process.json")
        resume = read_json(REPORT/"parallel_resume.json")
        if before != resume["protected_state"] or not resume["protected_state_unchanged"]:
            raise ValueError("Protected Futu state changed")
        sources = verify_execution_sources(runner)
        sources[relative(__file__)] = digest(__file__)
        if reviews:
            review_source = PROJECT/"scripts/review_final_ablation_results.py"
            if not review_source.is_file():
                raise FileNotFoundError(review_source)
            sources[relative(review_source)] = digest(review_source)
        scientific_paths = sorted((PROJECT/"paper_futu").glob("final_ablation_*.py"))
        scientific_paths += [PROJECT/p for p in ["paper_futu/run_final_ablation.py",
            "src/factor_library.py","src/build_enhanced_dataset.py","src/compare_models.py",
            "src/portfolio_strategy.py","src/runtime_budget.py"]]
        scientific_sources = {relative(p):digest(p) for p in scientific_paths}
        sources.update(scientific_sources)
        archive = archive_failure(runner)
        probe = locked_digest_probe(REPORT/"parallel_training/controller.lock")
        save_json(archive/"controller_lock_reproduction.json",probe)
        complete,incomplete,counts = task_inventory()
        print(f"CHECK {len(complete)} completed task manifests; {len(incomplete)} retained incomplete historical tasks",flush=True)
        plan = calculation_plan(complete)
        print(f"CHECK reconstructed {plan['required_unique_accounts']} required original-key accounts and {plan['required_prediction_streams']} streams",flush=True)
        meta = dict(finalized_at=time.time(),scientific_code_sha256=scope.SCIENTIFIC_HASH,
            snapshot_id=scope.SNAPSHOT,execution_sources=sources,
            source_hash_does_not_cover_new_orchestration_or_finalizer=True,
            scientific_sources=scientific_sources,
            protected_state=before,protected_state_unchanged=True,
            task_counts=counts,calculation_plan=plan,failure_archive=relative(archive),
            review_files=[relative(p) for p in reviews],
            historical_completed_artifacts_retained_with_own_task_provenance=True,
            historical_results_not_promoted_into_current_fair_candidate_ranking=True,
            no_training_or_account_replay_performed=True,controller_lock_reproduction=probe)
        manifest = build_manifest(PROJECT,REPORT,DATA,MODELS,incomplete_ids=incomplete,
            extra_paths=[PROJECT/p for p in sources],metadata=meta)
        print(f"CHECK hashed {len(manifest['files'])} immutable files; validating task and data artifact fingerprints",flush=True)
        for task in complete.values():
            for path,expected in task["artifacts"].items():
                rel = relative(path)
                if manifest["files"].get(rel) != expected:
                    raise ValueError(f"Completed task artifact changed, missing, or wrongly classified mutable: {path}")
        for path,expected in read_json(DATA/"manifest.json")["artifacts"].items():
            if manifest["files"].get(relative(DATA/path)) != expected:
                raise ValueError(f"Frozen data artifact changed: {path}")
        destination = REPORT/("delivery_manifest.json" if args.finalize else "delivery_manifest.preview.json")
        save_json(destination,manifest)
        count = verify_manifest(read_json(destination),inventory_roots=(REPORT,DATA,MODELS),
            extra_paths=[PROJECT/p for p in sources])
        if protected_state() != before or code_hash() != scope.SCIENTIFIC_HASH:
            raise ValueError("Protected or scientific files changed during packaging")
        record = dict(status="complete" if args.finalize else "prepared_waiting_for_final_review",
            updated=time.time(),manifest=relative(destination),manifest_sha256=digest(destination),
            immutable_files_verified=count,failure_archive=relative(archive),
            finalizer_sha256=digest(__file__),scientific_code_sha256=scope.SCIENTIFIC_HASH,
            snapshot_id=scope.SNAPSHOT,protected_state_unchanged=True,
            computation_status="complete",delivery_status="complete" if args.finalize else "pending_final_review",
            no_scientific_tasks_reexecuted=True)
        save_json(REPORT/"delivery_finalization.json",record)
        if args.finalize:
            prior = read_json(REPORT/"pipeline_status.json")
            save_json(REPORT/"pipeline_status.json",dict(stage=runner["stage"],status="complete",updated=time.time(),
                computation_status="complete",delivery_status="complete",delivery_manifest_sha256=record["manifest_sha256"],
                original_delivery_failure=prior,finalization_record=relative(REPORT/"delivery_finalization.json")))
            resume.update(status="complete",updated=time.time(),computation_status="complete",delivery_status="complete",
                original_delivery_failure=resume.get("error"),failure_archive=relative(archive),
                finalization_record=relative(REPORT/"delivery_finalization.json"))
            resume.pop("error",None)
            save_json(REPORT/"parallel_resume.json",resume)
        print(f"{record['status'].upper()}: {count} immutable files verified; {plan['required_unique_accounts']} original-key accounts complete",flush=True)


if __name__ == "__main__":
    main()
