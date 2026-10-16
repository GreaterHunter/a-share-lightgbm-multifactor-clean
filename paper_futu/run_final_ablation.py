"""Run the frozen final study without modifying the Futu deployment.

python -m paper_futu.run_final_ablation --as-of 2026-10-13 --stage all
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from paper_futu.final_ablation_core import (
    A, AS_OF, BASE, DATA, LIMITATIONS, NAME, REPORT, SCHEDULES, Spec, Store,
    code_hash, digest, initial_specs, init_qlib, offsets, parameter_configs,
    portfolios, read_json, save_json,
    protected_state,
)

ANCHOR = dict(top_k=20, top_tier_size=5, top_tier_weight=.25, n_drop=3)
COARSE = [dict(top_k=20, top_tier_size=5, top_tier_weight=.25, n_drop=6),
          dict(top_k=10, top_tier_size=2, top_tier_weight=.4, n_drop=3)]
STAGES = ["data", "model", "tune", "portfolio", "cadence", "robustness", "report"]
MAX_SELECTED_SPECS = 3
FINE_TOP_PAIRS = 2
FINE_ANCHOR_PAIRS = {(10, 10), (20, 10)}
SCOPE_REVISION = "reduced_24_parameters_20_schedules"


def search_counts():
    streams = MAX_SELECTED_SPECS * 2  # fixed and tuned parameter policies
    pairs = sum(map(len, SCHEDULES.values()))
    portfolio_count = len(portfolios())
    validation_years = {v for year in range(2022, 2027) for v in range(year-3, year)}
    fine_pair_limit = FINE_TOP_PAIRS + len(FINE_ANCHOR_PAIRS)
    return dict(model_ablation_specs=14, parameter_configs=len(parameter_configs()),
        horizon_rebalance_pairs=pairs, portfolios_per_pair=portfolio_count,
        selected_specs_max=MAX_SELECTED_SPECS, model_streams_max=streams,
        fine_top_pairs=FINE_TOP_PAIRS, fine_pairs_per_stream_max=fine_pair_limit,
        coarse_replays_max=streams*pairs*len(COARSE),
        fine_replays_max=streams*fine_pair_limit*portfolio_count,
        robustness_replays_max=225,
        h20_validation_fits_max=MAX_SELECTED_SPECS*len(parameter_configs())*len(validation_years))


def spec_from(record):
    return Spec(**{**record, "features": tuple(record["features"])})


def prediction_path(spec, horizon, policy="fixed", months=12, seed=42):
    return REPORT / "predictions" / f"{spec.name}_h{horizon}_{policy}_m{months}_s{seed}.parquet"


def worker(job):
    from paper_futu.final_ablation_training import fit_stream
    spec = spec_from(job["spec"])
    horizon, policy, months, seed = job["horizon"], job.get("policy", "fixed"), job.get("months", 12), job.get("seed", 42)
    if job["kind"] in ["stream", "anchor"]:
        path = fit_stream(spec, horizon, policy, months, seed)
    else:
        init_qlib()
        path = prediction_path(spec, horizon, policy, months, seed)
        meta = read_json(path.with_suffix(".json"))
        if digest(path) != meta["prediction_sha256"]:
            raise ValueError("Stitched prediction artifact changed")
    if job["kind"] == "stream":
        return
    from paper_futu.final_ablation_account import replay
    cases = job.get("cases", [dict(portfolio=ANCHOR, rebalance=10, offset=0, friction=0.)])
    rows = []
    for case in cases:
        result = replay(path, **case)
        rows += [{**r, "spec": spec.name, "engine": spec.engine, "policy": policy,
                  "horizon": horizon, "months": months, "seed": seed} for r in result["rows"]]
        pd.DataFrame(rows).to_csv(job["output"], index=False, encoding="utf-8-sig")
        gc.collect()


def dispatch(job):
    from paper_futu.final_ablation_core import identity
    REPORT.mkdir(parents=True, exist_ok=True)
    logs = REPORT / "logs"
    logs.mkdir(exist_ok=True)
    task = identity(job)[:16]
    log = logs / f"{task}.log"
    print(f"START {job['kind']} {job['spec']['name']} H{job['horizon']} {job.get('policy','fixed')} M{job.get('months',12)}; log={log}", flush=True)
    started = time.time()
    progress = dict(job=job, log=str(log), started=started, status="running", code_sha256=code_hash())
    save_json(REPORT / "current_job.json", progress)
    command = [sys.executable, "-u", "-m", "paper_futu.run_final_ablation", "--worker", json.dumps(job)]
    with log.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[1],
            stdout=output, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS":"4",
                 "MKL_NUM_THREADS":"4", "OPENBLAS_NUM_THREADS":"4"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        progress.update(worker_pid=process.pid, command=command)
        save_json(REPORT / "current_job.json", progress)
        while process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                progress.update(updated=time.time(), elapsed_seconds=time.time()-started)
                save_json(REPORT / "current_job.json", progress)
    status = "complete" if process.returncode == 0 else "failed"
    save_json(REPORT / "current_job.json", {**progress, "status":status, "updated":time.time(),
        "elapsed_seconds":time.time()-started, "exit_code":process.returncode})
    if process.returncode:
        raise RuntimeError(f"Worker failed; full log: {log}")
    print(f"DONE {task}: {time.time()-started:.1f}s", flush=True)


def table(paths, output):
    frames = [pd.read_csv(p) for p in paths if Path(p).exists()]
    if not frames:
        raise ValueError("No completed experiment rows")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["case_id", "period"])
    result.to_csv(output, index=False, encoding="utf-8-sig")
    return result


def qualified(frame):
    return frame.loc[frame.period.eq("all") & frame.max_drawdown.ge(-.30)].sort_values(
        ["annualized_return", "max_drawdown", "total_fees_cny", "case_id"], ascending=[False, False, True, True])


def fine_pairs(frame):
    good = qualified(frame).drop_duplicates(["horizon", "rebalance"]).head(FINE_TOP_PAIRS)
    return {(int(r.horizon), int(r.rebalance)) for r in good.itertuples()} | FINE_ANCHOR_PAIRS


def job_stream(spec, horizon, policy="fixed", months=12, seed=42):
    dispatch(dict(kind="stream", spec=spec.record(), horizon=horizon, policy=policy, months=months, seed=seed))


def job_accounts(spec, horizon, policy, cases, output, months=12, seed=42, anchor=False):
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    dispatch(dict(kind="anchor" if anchor else "accounts", spec=spec.record(), horizon=horizon,
                  policy=policy, months=months, seed=seed, cases=cases, output=str(output)))


def stage_model():
    specs = initial_specs()
    files = []
    for spec in specs:
        output = REPORT / "model_runs" / f"{spec.name}.csv"
        job_accounts(spec, 20, "fixed", [dict(portfolio=ANCHOR, rebalance=10)], output, anchor=True)
        files.append(output)
        table(files, REPORT / "model_ablation.csv")
    scores = table(files, REPORT / "model_ablation.csv")
    good = qualified(scores.loc[scores.engine.eq("lightgbm")])
    source = next((s for s in specs if len(good) and s.name == good.iloc[0].spec), A)
    for window in [5, 10]:
        spec = replace(source, name=f"{source.name}_window{window}", window=window)
        specs.append(spec)
        output = REPORT / "model_runs" / f"{spec.name}.csv"
        job_accounts(spec, 20, "fixed", [dict(portfolio=ANCHOR, rebalance=10)], output, anchor=True)
        files.append(output)
    scores = table(files, REPORT / "model_ablation.csv")
    selected = [A]
    for row in qualified(scores.loc[scores.engine.eq("lightgbm")]).itertuples():
        if row.spec != A.name:
            selected.append(next(s for s in specs if s.name == row.spec))
        if len(selected) == MAX_SELECTED_SPECS:
            break
    save_json(REPORT / "model_selection.json", dict(selected=[s.record() for s in selected],
              all_specs=[s.record() for s in specs], window_source=source.name,
              rule="A plus highest net CAGR non-A LightGBM specifications with MDD <=30%; no Ridge",
              eliminated=[r._asdict() for r in scores.loc[scores.period.eq("all")].itertuples(index=False)
                          if r.spec not in {s.name for s in selected}]))


def selected_specs():
    return [spec_from(s) for s in read_json(REPORT / "model_selection.json")["selected"]]


def stage_tune():
    for spec in selected_specs():
        for policy in ["fixed", "tuned"]:
            job_stream(spec, 20, policy)
        for horizon in [h for h in SCHEDULES if h != 20]:
            for policy in ["fixed", "tuned"]:
                job_stream(spec, horizon, policy)
    # Stable reference always exists even if fewer challenge specs qualify.
    job_stream(A, 10)


def stage_portfolio():
    coarse_files, fine_files = [], []
    for spec in selected_specs():
        for policy in ["fixed", "tuned"]:
            stream_files = []
            for horizon, schedules in SCHEDULES.items():
                cases = [dict(portfolio=p, rebalance=r) for r in schedules for p in COARSE]
                output = REPORT / "portfolio_runs" / f"{spec.name}_{policy}_h{horizon}_coarse.csv"
                job_accounts(spec, horizon, policy, cases, output)
                stream_files.append(output)
                coarse_files.append(output)
                table(coarse_files, REPORT / "portfolio_coarse.csv")
            stream = pd.concat([pd.read_csv(p) for p in stream_files], ignore_index=True)
            pairs = fine_pairs(stream)
            for horizon in sorted({h for h, _ in pairs}):
                cases = [dict(portfolio=p, rebalance=r) for h, r in sorted(pairs) if h == horizon for p in portfolios()]
                output = REPORT / "portfolio_runs" / f"{spec.name}_{policy}_h{horizon}_fine.csv"
                job_accounts(spec, horizon, policy, cases, output)
                fine_files.append(output)
                table(fine_files, REPORT / "portfolio_fine.csv")
    results = table(fine_files, REPORT / "portfolio_fine.csv")
    winners = qualified(results).head(5)
    specs = {s.name: s for s in selected_specs()}
    candidates = [dict(spec=specs[r.spec].record(), horizon=int(r.horizon), policy=r.policy,
                       rebalance=int(r.rebalance), portfolio=dict(top_k=int(r.top_k), n_drop=int(r.n_drop),
                       top_tier_size=int(r.top_tier_size), top_tier_weight=float(r.top_tier_weight))) for r in winners.itertuples()]
    save_json(REPORT / "cadence_candidates.json", candidates)


def stage_cadence():
    # Causal calendar pilot is diagnostic and does not change filtering rules.
    files = []
    for months in [12, 6, 3]:
        job_stream(A, 20, months=months)
        path = REPORT / "cadence_runs" / f"pilot_m{months}.csv"
        job_accounts(A, 20, "fixed", [dict(portfolio=ANCHOR, rebalance=10)], path, months=months)
        files.append(path)
    table(files, REPORT / "cadence_pilot.csv")
    candidates = read_json(REPORT / "cadence_candidates.json")
    files = []
    expanded = []
    for index, config in enumerate(candidates):
        spec = spec_from(config["spec"])
        for months in [12, 6, 3]:
            job_stream(spec, config["horizon"], config["policy"], months)
            path = REPORT / "cadence_runs" / f"candidate{index}_m{months}.csv"
            case = dict(portfolio=config["portfolio"], rebalance=config["rebalance"])
            job_accounts(spec, config["horizon"], config["policy"], [case], path, months)
            files.append(path)
            expanded.append({**config, "months": months, "candidate_id": f"c{index}_m{months}"})
    if files:
        table(files, REPORT / "cadence_comparison.csv")
    save_json(REPORT / "robustness_candidates.json", expanded)


def choose_final(candidates, scores):
    qualified_rows = []
    for config in candidates:
        group = scores.loc[scores.candidate_id.eq(config["candidate_id"]) & scores.period.eq("all")]
        base = group.loc[group.friction.eq(0)]
        stressed = group.loc[np.isclose(group.friction, .002)]
        expected = len(offsets(config["rebalance"]))
        if len(base) != expected or len(stressed) != expected:
            raise ValueError("Incomplete robustness phases")
        passed = bool(base.max_drawdown.ge(-.30).all() and stressed.annualized_return.median() > 0)
        qualified_rows.append(dict(candidate_id=config["candidate_id"], passed=passed,
             median_cagr=float(base.annualized_return.median()), worst_drawdown=float(-base.max_drawdown.min()),
             stress_median_cagr=float(stressed.annualized_return.median()),
             median_turnover=float(base.average_daily_turnover.median()),
             median_fees=float(base.total_fees_cny.median()), months=config["months"],
             reason="passed" if passed else "drawdown phase exceeds30% or 20bp median return nonpositive"))
    ranking = pd.DataFrame(qualified_rows, columns=["candidate_id", "passed", "median_cagr",
        "worst_drawdown", "stress_median_cagr", "median_turnover", "median_fees", "months", "reason"])
    if ranking.empty:
        return ranking, []
    remaining = ranking.loc[ranking.passed].copy()
    selected = []
    while len(remaining):
        near = remaining.loc[remaining.median_cagr > remaining.median_cagr.max()-.01]
        row = near.sort_values(["worst_drawdown", "median_turnover", "median_fees", "months", "candidate_id"],
                               ascending=[True, True, True, False, True]).iloc[0]
        selected.append(row.candidate_id)
        remaining = remaining.drop(row.name)
    return ranking, selected


def stage_robustness():
    candidates = read_json(REPORT / "robustness_candidates.json")
    frames = []
    for config in candidates:
        spec = spec_from(config["spec"])
        cases = [dict(portfolio=config["portfolio"], rebalance=config["rebalance"], offset=o, friction=f)
                 for o in offsets(config["rebalance"]) for f in [0., .001, .002]]
        output = REPORT / "robustness_runs" / f"{config['candidate_id']}.csv"
        job_accounts(spec, config["horizon"], config["policy"], cases, output, config["months"])
        frame = pd.read_csv(output).assign(candidate_id=config["candidate_id"])
        frames.append(frame)
        pd.concat(frames, ignore_index=True).to_csv(REPORT / "robustness_all.csv", index=False, encoding="utf-8-sig")
    scores = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ranking, order = choose_final(candidates, scores) if candidates else (pd.DataFrame(), [])
    ranking.to_csv(REPORT / "final_selection.csv", index=False, encoding="utf-8-sig")
    for candidate_id in order[:3]:
        config = next(c for c in candidates if c["candidate_id"] == candidate_id)
        spec = spec_from(config["spec"])
        for seed in [2024, 2025]:
            job_stream(spec, config["horizon"], config["policy"], config["months"], seed)
            cases = [dict(portfolio=config["portfolio"], rebalance=config["rebalance"], offset=o) for o in offsets(config["rebalance"])]
            output = REPORT / "seed_runs" / f"{candidate_id}_seed{seed}.csv"
            job_accounts(spec, config["horizon"], config["policy"], cases, output, config["months"], seed)
            primary = pd.read_parquet(prediction_path(spec, config["horizon"], config["policy"], config["months"]))
            alternate = pd.read_parquet(prediction_path(spec, config["horizon"], config["policy"], config["months"], seed))
            overlaps = []
            for day, a in primary.groupby(level="datetime"):
                b = alternate.xs(day, level="datetime", drop_level=False)
                ka = set(a.nlargest(config["portfolio"]["top_k"], "prediction").index.get_level_values("instrument"))
                kb = set(b.nlargest(config["portfolio"]["top_k"], "prediction").index.get_level_values("instrument"))
                overlaps.append(dict(datetime=day, top_k_overlap=len(ka&kb)/len(ka)))
            pd.DataFrame(overlaps).to_csv(output.with_name(output.stem+"_overlap.csv"), index=False)
    if order:
        frozen = next(c for c in candidates if c["candidate_id"] == order[0])
        passed = True
    else:
        frozen = dict(spec=A.record(), horizon=10, policy="fixed", months=12, rebalance=10, portfolio=COARSE[1])
        passed = False
        job_stream(A, 10)
    spec = spec_from(frozen["spec"])
    manifest = read_json(prediction_path(spec, frozen["horizon"], frozen["policy"], frozen["months"]).with_suffix(".json"))
    save_json(REPORT / "frozen_research_version.json", dict(configuration=frozen, passed=passed,
              seed=42, offset=0, snapshot_id=read_json(DATA/"manifest.json")["snapshot_id"],
              code_sha256=code_hash(), latest_model=manifest["periods"][-1], selection_order=order,
              deployment_changed=False, independent_holdout=False, limitations=LIMITATIONS))


def stage_report():
    init_qlib()
    output = REPORT / "reference.csv"
    job_accounts(A, 10, "fixed", [dict(portfolio=COARSE[1], rebalance=10)], output)
    selections = read_json(REPORT / "frozen_research_version.json")
    config = selections["configuration"]
    spec = spec_from(config["spec"])
    from paper_futu.final_ablation_training import load_frame
    import lightgbm as lgb
    frame = load_frame(spec, config["horizon"])
    latest = frame.xs(AS_OF, level="datetime")
    model = lgb.Booster(model_file=selections["latest_model"]["model_path"])
    scores = pd.Series(model.predict(latest[list(spec.features)], num_threads=4), index=latest.index, name="prediction").sort_values(ascending=False)
    scores.to_csv(REPORT / "frozen_latest_ranking.csv", encoding="utf-8-sig")
    selection_tables = []
    for chosen_spec in selected_specs():
        for horizon in SCHEDULES:
            for policy in ["fixed", "tuned"]:
                for year in range(2022,2027):
                    selection_file = REPORT/"selections"/f"{chosen_spec.name}_h{horizon}_{policy}_{year}.csv"
                    if selection_file.exists():
                        selection_tables.append(pd.read_csv(selection_file).assign(
                            spec=chosen_spec.name,horizon=horizon,policy=policy,execution_year=year))
    if selection_tables:
        pd.concat(selection_tables,ignore_index=True).to_csv(REPORT/"parameter_search.csv",index=False,encoding="utf-8-sig")
    seed_tables = []
    for path in (REPORT/"seed_runs").glob("*.csv"):
        if not path.name.endswith("_overlap.csv"):
            seed_tables.append(pd.read_csv(path).assign(candidate_id=path.stem.split("_seed")[0]))
    if seed_tables:
        pd.concat(seed_tables,ignore_index=True).to_csv(REPORT/"randomness.csv",index=False,encoding="utf-8-sig")
    files = ["model_ablation.csv", "parameter_search.csv", "portfolio_coarse.csv", "portfolio_fine.csv", "cadence_pilot.csv",
             "cadence_comparison.csv", "final_selection.csv", "robustness_all.csv", "reference.csv"]
    sections = []
    percentage_columns = {"annualized_return", "max_drawdown", "annualized_excess_return",
                          "median_cagr", "worst_drawdown", "stress_median_cagr", "top_tier_weight", "friction"}
    for name in files:
        path = REPORT/name
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "period" in frame:
            frame = frame.loc[frame.period.eq("all")]
        columns = [c for c in ["spec", "policy", "horizon", "rebalance", "top_k", "top_tier_weight", "n_drop", "months",
                   "offset", "friction", "annualized_return", "max_drawdown", "annualized_excess_return", "total_fees_cny",
                   "average_holdings", "candidate_id", "passed", "median_cagr", "worst_drawdown", "stress_median_cagr", "reason"] if c in frame]
        if "annualized_return" in frame:
            frame = frame.sort_values("annualized_return", ascending=False)
        sections.append(f"<h2>{html.escape(name)}</h2><p>完整文件：{html.escape(name)}；以下显示前100行。</p>"+
                        frame[columns].head(100).to_html(index=False, float_format=lambda x:f"{x:.5f}",
                        formatters={c:lambda x:f"{x:.2%}" for c in columns if c in percentage_columns}))
    annual_tables = [pd.read_csv(output).assign(version="B0研究参照")]
    robust_path = REPORT/"robustness_all.csv"
    if selections["passed"] and robust_path.exists():
        robust = pd.read_csv(robust_path)
        winner = robust.loc[robust.candidate_id.eq(config["candidate_id"]) & robust.offset.eq(0) & robust.friction.eq(0)]
        annual_tables.append(winner.assign(version="冻结研究主版本"))
    annual = pd.concat(annual_tables,ignore_index=True)
    annual = annual.loc[annual.period.astype(str).str.fullmatch(r"202[2-6]")].copy()
    annual["period"] = annual.period.astype(str).replace({"2026":"2026截至10/13"})
    annual_columns = ["version","period","total_return","benchmark_total_return","total_excess_return",
                      "max_drawdown","total_fees_cny","average_daily_turnover","average_holdings"]
    annual.to_csv(REPORT/"frozen_vs_reference_yearly.csv",index=False,encoding="utf-8-sig")
    sections.insert(0,"<h2>冻结版本与参照的逐年区间收益</h2>"+annual[annual_columns].to_html(index=False,
                    formatters={c:lambda x:f"{x:.2%}" for c in annual_columns if "return" in c or c=="max_drawdown"}))
    status = "已筛出通过预设压力条件的研究候选" if selections["passed"] else "未找到通过预设标准的替代，保留B0参照；不声称B0通过压力标准"
    page = "<!doctype html><meta charset='utf-8'><title>最终消融研究</title><style>body{font:15px sans-serif;margin:32px;color:#182331}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccd5df}th{background:#eef3f8}</style>"
    page += f"<h1>最终消融研究，截止2026-10-13</h1><p>{status}</p><pre>{html.escape(json.dumps(config,ensure_ascii=False,indent=2))}</pre>"
    page += "<ul>"+"".join(f"<li>{html.escape(x)}</li>" for x in LIMITATIONS)+"</ul>"+"".join(sections)
    (REPORT/"research_report.html").write_text(page, encoding="utf-8")
    paths = [p for p in REPORT.rglob("*") if p.is_file() and "mlruns" not in p.parts and p.name != "delivery_manifest.json"]
    save_json(REPORT/"delivery_manifest.json", {str(p.relative_to(REPORT)):digest(p) for p in paths})


def dry_run():
    task_files = list((REPORT/"tasks").glob("*.json"))
    counts = {}
    durations = []
    current_code = code_hash()
    for p in task_files:
        r = read_json(p)
        if r.get("code_sha256") != current_code:
            continue
        key = f"{r['kind']}:{r['status']}"
        counts[key] = counts.get(key,0)+1
        if r['kind']=="validation_fold" and r['status']=="complete":
            durations.append(r["elapsed_seconds"])
    budget = search_counts()
    print(json.dumps(dict(as_of=str(AS_OF.date()), scope_revision=SCOPE_REVISION, **budget,
        retraining_months=[12,6,3], cached_task_status=counts,
        memory_policy="one isolated fitting worker, 4 threads, 2 GiB available minimum",
        estimated_h20_validation_seconds=(float(np.median(durations))*max(0,budget["h20_validation_fits_max"]-len(durations))) if durations else None,
        estimate_note="Measured fold median; excludes final fits, other horizons and account replay; not an ETA guarantee",
        writes=False), ensure_ascii=False,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2026-10-13", choices=["2026-10-13"])
    parser.add_argument("--stage", choices=STAGES+["all"], default="all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.dry_run:
        dry_run()
        return
    if args.worker:
        worker(json.loads(args.worker))
        return
    protected = protected_state()
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = REPORT/"pipeline.lock"
    if lock.exists():
        import psutil
        old_pid = int(lock.read_text())
        try:
            old_command = psutil.Process(old_pid).cmdline()
        except psutil.NoSuchProcess:
            old_command = []
        if "paper_futu.run_final_ablation" in old_command:
            raise RuntimeError(f"Study runner {old_pid} is already active; inspect its status rather than launching a duplicate")
        lock.unlink()
    descriptor = os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    with os.fdopen(descriptor,"w") as lock_file:
        lock_file.write(str(os.getpid()))
    save_json(REPORT/"runner_process.json",dict(pid=os.getpid(),started=time.time(),stage=args.stage,
        as_of=args.as_of, executable=sys.executable, command=[sys.executable, *sys.argv],
        stdout_log=os.environ.get("FINAL_ABLATION_STDOUT_LOG"),
        stderr_log=os.environ.get("FINAL_ABLATION_STDERR_LOG"), code_sha256=code_hash()))
    protocol = dict(as_of=str(AS_OF.date()), schedules=SCHEDULES, parameter_configs=parameter_configs(),
                    portfolio_rules=portfolios(), seed=42, threads=4, limitations=LIMITATIONS,
                    code_sha256=code_hash(), protected_state=protected, scope_revision=SCOPE_REVISION,
                    search_counts=search_counts(), fine_top_pairs=FINE_TOP_PAIRS,
                    fine_anchor_pairs=sorted(FINE_ANCHOR_PAIRS),
                    fine_selection_rule="Two highest qualified distinct H/R pairs plus 10/10 and 20/10")
    save_json(REPORT/"protocol.json", protocol)
    save_json(REPORT/"reduced_scope.json", dict(scope_revision=SCOPE_REVISION,
        user_requested=True, schedules=SCHEDULES, search_counts=search_counts(),
        removed_horizons=[60], removed_rebalances=[1,3], legacy_joint_sample_size=32,
        retained_joint_configurations="first 8 of the original 32, p16..p23 unchanged",
        fine_top_pairs=FINE_TOP_PAIRS, fine_anchor_pairs=sorted(FINE_ANCHOR_PAIRS),
        frozen_snapshot_reused=True, old_h60_snapshot_artifacts_retained=True,
        old_code_task_cache_migrated=False, code_sha256=code_hash()))
    stages = STAGES if args.stage == "all" else [args.stage]
    try:
        for stage in stages:
            print(f"STAGE {stage}", flush=True)
            save_json(REPORT/"pipeline_status.json", dict(stage=stage,status="running",updated=time.time()))
            if stage == "data":
                from paper_futu.final_ablation_data import prepare_data
                prepare_data()
            else:
                if stage == "model":
                    for months in [12,6,3]:
                        job_stream(A,20,months=months)
                        path = REPORT/"cadence_runs"/f"pilot_m{months}.csv"
                        job_accounts(A,20,"fixed",[dict(portfolio=ANCHOR,rebalance=10)],path,months)
                    table([REPORT/"cadence_runs"/f"pilot_m{m}.csv" for m in [12,6,3]],REPORT/"cadence_pilot.csv")
                globals()[f"stage_{stage}"]()
            save_json(REPORT/"pipeline_status.json",dict(stage=stage,status="complete",updated=time.time()))
        save_json(REPORT/"pipeline_status.json",dict(stage=args.stage,status="complete",updated=time.time()))
    except BaseException as error:
        save_json(REPORT/"pipeline_status.json",dict(stage=stage,status="failed",error=repr(error),updated=time.time()))
        raise
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):
            lock.unlink()
        after = protected_state()
        save_json(REPORT/"protected_state_check.json",dict(unchanged=protected==after,before=protected,after=after))
        if protected != after:
            raise RuntimeError("Protected deployment or active plan changed during the experiment; no writer was invoked")


if __name__ == "__main__":
    main()
