"""Frozen experiment protocol, causal calendars, and verified task artifacts."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.factor_library import ENHANCED_FEATURE_NAMES, FUNDAMENTAL_FEATURE_NAMES

AS_OF = pd.Timestamp("2026-10-13")
START = pd.Timestamp("2022-01-01")
NAME = "final_ablation_asof_20261013"
REPORT = PROJECT_ROOT / "reports/experiments" / NAME
DATA = PROJECT_ROOT / "data/processed" / NAME
MODELS = PROJECT_ROOT / "models/experiments" / NAME
SOURCE_PROVIDER = PROJECT_ROOT / "data/processed/morning_alpha158_20261013/provider"
PROVIDER = DATA / "provider"
FINANCIAL_SOURCE = PROJECT_ROOT / "data/processed/cn_latest_mature_open10_20261012/financial_events.parquet"
SCHEDULES = {5: [5, 10], 10: [5, 10, 20, 30],
             20: [5, 10, 20, 30], 30: [5, 10, 15, 20, 30],
             40: [5, 10, 20, 30, 40]}
BASE = dict(objective="regression", metric="l2", boosting_type="gbdt",
            learning_rate=.03, num_leaves=31, max_depth=-1, min_child_samples=100,
            subsample=.8, subsample_freq=1, colsample_bytree=.8,
            reg_alpha=.1, reg_lambda=1., verbosity=-1, n_jobs=4)
GRID = dict(learning_rate=[.015, .03, .06], num_leaves=[15, 31, 63, 127],
            min_child_samples=[100, 300, 1000], reg_alpha=[0., .1, 1.],
            reg_lambda=[1., 5., 20.], subsample=[.6, .8, 1.],
            colsample_bytree=[.6, .8, 1.])
TREES = {.015: [20, 40, 80, 160, 320, 640, 1280],
         .03: [10, 20, 40, 80, 160, 320, 640],
         .06: [5, 10, 20, 40, 80, 160, 320]}
VERSION = "1"
_VERIFIED_SNAPSHOT = None
LIMITATIONS = ["日线理想开盘成交，不代表09:31模拟盘成交或盘口容量已验证。",
               "统一9.5%开盘涨跌幅过滤，不重建板块、ST及每日精确涨跌停。",
               "复权价格及Qlib调整数量用于公司行动收益近似，未建立逐笔分红现金与配股台账。",
               "财务披露日期已对齐，但财务值不是完整历史修订档案。",
               "2022—2026反复参与研究，最终选择存在选择偏差，不是新的独立测试。",
               "分阶段剪枝不能证明全网格全局最优，30%历史回撤线不保证未来回撤。"]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str,
                      separators=(",", ":"), allow_nan=False)


def identity(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hash():
    files = sorted((PROJECT_ROOT / "paper_futu").glob("final_ablation_*.py"))
    files += [PROJECT_ROOT / "paper_futu/run_final_ablation.py",
              PROJECT_ROOT / "src/factor_library.py", PROJECT_ROOT / "src/build_enhanced_dataset.py",
              PROJECT_ROOT / "src/compare_models.py", PROJECT_ROOT / "src/portfolio_strategy.py",
              PROJECT_ROOT / "src/runtime_budget.py"]
    return identity({p.relative_to(PROJECT_ROOT).as_posix(): digest(p) for p in files})


def protected_state():
    """Hash deployment bindings and scheduler sources without invoking an order API."""
    config = PROJECT_ROOT / "configs/futu_cn_paper.json"
    model_dir = (PROJECT_ROOT / read_json(config)["model_dir"]).resolve()
    if not model_dir.is_relative_to((PROJECT_ROOT / "models").resolve()):
        raise ValueError("Unexpected deployed model directory")
    paths = [config, model_dir/"model.txt", model_dir/"metadata.json",
             PROJECT_ROOT/"data/interim/futu_cn_paper/active_plan.json",
             PROJECT_ROOT/"data/interim/futu_cn_paper/catchup_plan.json",
             PROJECT_ROOT/"paper_futu/cn_worker.py", PROJECT_ROOT/"scripts/install_cn_paper_schedule.ps1"]
    return {str(p.relative_to(PROJECT_ROOT)): digest(p) for p in paths if p.exists()}


@dataclass(frozen=True)
class Spec:
    name: str
    features: tuple[str, ...]
    input_kind: str = "rank"
    target_kind: str = "rank"
    window: int = 0
    engine: str = "lightgbm"

    def record(self):
        return asdict(self)


A = Spec("lgb_15_rank_rank", tuple(ENHANCED_FEATURE_NAMES))


def initial_specs():
    specs = [Spec(f"lgb_{n}_{x}_{y}", tuple(ENHANCED_FEATURE_NAMES[:n]), x, y)
             for n, x, y in itertools.product([12, 15], ["raw", "rank"], ["raw", "rank"])]
    specs += [replace(A, name=f"without_{f}", features=tuple(x for x in A.features if x != f))
              for f in FUNDAMENTAL_FEATURE_NAMES]
    return specs + [replace(A, name="ridge_rank_rank", engine="ridge")]


def parameter_configs():
    result = [{"id": "p00", "parameters": dict(BASE)}]
    for key, values in GRID.items():
        for value in values:
            if value != BASE[key]:
                result.append({"id": f"p{len(result):02d}", "parameters": {**BASE, key: value}})
    assert len(result) == 16
    excluded = {identity(r["parameters"]) for r in result}
    choices = [{**BASE, **dict(zip(GRID, row))} for row in itertools.product(*GRID.values())]
    choices = [p for p in choices if identity(p) not in excluded]
    rng = np.random.default_rng(20261014)
    # Keep exactly the first eight joint configurations from the original 48.
    # Drawing eight anew would change p16..p23 and break the paired comparison.
    for index in rng.choice(len(choices), size=32, replace=False)[:8]:
        result.append({"id": f"p{len(result):02d}", "parameters": choices[int(index)]})
    assert len({identity(r["parameters"]) for r in result}) == 24
    return result


def window_start(period_start, window):
    return max(pd.Timestamp("2010-01-01"), pd.Timestamp(period_start) - pd.DateOffset(years=window)) if window else pd.Timestamp("2010-01-01")


def serving_periods(calendar, months):
    dates = pd.DatetimeIndex(calendar)
    trades = dates[(dates >= START) & (dates <= AS_OF)]
    periods = []
    for year in range(2022, 2027):
        for month in range(1, 13, months):
            p = pd.Timestamp(year, month, 1)
            end = p + pd.DateOffset(months=months)
            span = trades[(trades >= p) & (trades < end)]
            if len(span):
                previous = dates[dates < span[0]]
                if not len(previous):
                    raise ValueError("Serving period has no preceding signal session")
                periods.append(dict(period_start=p, cutoff=previous[-1], execution_start=span[0],
                                    execution_end=span[-1], year=year, retrain_months=months))
    return periods


def mature(frame, start, cutoff):
    dates = frame.index.get_level_values("datetime")
    result = frame.loc[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(cutoff)) &
                       frame.label_end_date.le(pd.Timestamp(cutoff))].dropna(subset=["label", "target"])
    if result.empty or result.label_end_date.gt(pd.Timestamp(cutoff)).any():
        raise ValueError("Empty or non-causal supervised period")
    return result


def daily_ic(score, label, minimum=30):
    pair = pd.DataFrame({"prediction": score, "outcome": label}).replace([np.inf, -np.inf], np.nan).dropna()
    count = pair.groupby(level="datetime").size()
    out = pd.DataFrame({"observations": count})
    for name, values in [("ic", pair), ("rank_ic", pair.groupby(level="datetime").rank(method="average"))]:
        centered = values - values.groupby(level="datetime").transform("mean")
        product = (centered.prediction * centered.outcome).groupby(level="datetime").sum()
        square = centered.pow(2).groupby(level="datetime").sum()
        denominator = np.sqrt(square.prediction * square.outcome)
        out[name] = (product / denominator.where(denominator > 0)).where(count >= minimum)
    return out


def choose_candidates(frame):
    """Joint parameter/tree selection; each of three folds has equal weight."""
    records = []
    for (parameter_id, trees), group in frame.groupby(["parameter_id", "trees"]):
        if group.validation_year.nunique() != 3 or not np.isfinite(group.mean_rank_ic).all():
            raise ValueError("Selection requires three finite annual folds")
        records.append(dict(parameter_id=parameter_id, trees=int(trees),
                            mean_rank_ic=float(group.mean_rank_ic.mean()),
                            fold_std=float(group.mean_rank_ic.std()),
                            complexity=int(group.num_leaves.iloc[0]) * int(trees)))
    remaining = pd.DataFrame(records)
    ordered = []
    while len(remaining):
        near = remaining.loc[remaining.mean_rank_ic >= remaining.mean_rank_ic.max() - 1e-8]
        winner = near.sort_values(["fold_std", "complexity", "parameter_id", "trees"]).iloc[0]
        ordered.append(winner.to_dict())
        remaining = remaining.drop(winner.name)
    return pd.DataFrame(ordered)


def offsets(rebalance):
    return sorted({j * rebalance // 5 for j in range(5)})


def portfolios():
    result = []
    for k, head in [(5, 1), (10, 2), (20, 5), (30, 5), (50, 10)]:
        drops = sorted({math.ceil(k * p) for p in [.1, .3, .6, 1.]} | {3})
        for weight, drop in itertools.product(["equal", .4, .5, .6], drops):
            result.append(dict(top_k=k, top_tier_size=head,
                               top_tier_weight=head / k if weight == "equal" else weight,
                               n_drop=drop))
    assert len(result) == 88
    return result


def fee(notional, direction, friction=0):
    if notional <= 0:
        return 0.
    return max(5., notional * (.0005 if direction == 1 else .0015)) + friction * notional


def init_qlib():
    # Qlib 0.9.7 uses the local MLflow file recorder; current MLflow requires an explicit opt-in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import qlib
    from qlib.config import REG_CN
    qlib.init(provider_uri=str(PROVIDER), region=REG_CN, kernels=1,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (REPORT / "mlruns").as_uri(), "default_exp_name": NAME}})


class Store:
    """Only verified output hashes mark a task complete; failed runs remain visible."""
    def __init__(self):
        global _VERIFIED_SNAPSHOT
        self.code = code_hash()
        manifest = read_json(DATA / "manifest.json")
        self.data = manifest["snapshot_id"]
        if _VERIFIED_SNAPSHOT != self.data:
            for path, expected in manifest["artifacts"].items():
                if not (DATA/path).exists() or digest(DATA/path) != expected:
                    raise ValueError(f"Frozen snapshot artifact changed: {path}")
            _VERIFIED_SNAPSHOT = self.data

    def key(self, kind, config):
        return identity(dict(version=VERSION, kind=kind, config=config, code=self.code, data=self.data))[:24]

    def cached(self, kind, config):
        key = self.key(kind, config)
        path = REPORT / "tasks" / f"{key}.json"
        if not path.exists():
            return None
        record = read_json(path)
        if record.get("status") != "complete":
            return None
        if any(not Path(p).exists() or digest(p) != h for p, h in record["artifacts"].items()):
            raise ValueError(f"Cached artifact changed for task {key}")
        return record["result"]

    def run(self, kind, config, function):
        cached = self.cached(kind, config)
        if cached is not None:
            return cached
        from qlib.workflow import R
        key = self.key(kind, config)
        path = REPORT / "tasks" / f"{key}.json"
        started = time.time()
        record = dict(task_id=key, kind=kind, config=config, code_sha256=self.code,
                      snapshot_id=self.data, status="running", started=started)
        save_json(path, record)
        print(f"RUN {kind} {key} {config.get('spec', {}).get('name', '')}", flush=True)
        try:
            with R.start(experiment_name=NAME, recorder_name=key):
                R.log_params(task_id=key, kind=kind, config_sha256=identity(config),
                             snapshot_id=self.data, code_sha256=self.code)
                result, files, metrics = function(key)
                hashes = {str(p): digest(p) for p in files}
                R.log_metrics(**{k: float(v) for k, v in metrics.items() if np.isfinite(v)})
                R.save_objects(task_manifest=dict(config=config, result=result, artifacts=hashes,
                                                 code_sha256=self.code, snapshot_id=self.data))
                record.update(status="complete", elapsed_seconds=time.time() - started,
                              result=result, artifacts=hashes, recorder_id=R.get_recorder().id)
                save_json(path, record)
                print(f"COMPLETE {kind} {key} {record['elapsed_seconds']:.1f}s", flush=True)
                return result
        except BaseException as error:
            record.update(status="failed", error=repr(error), elapsed_seconds=time.time() - started)
            save_json(path, record)
            raise
