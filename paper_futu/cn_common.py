"""Paths and pure portfolio rules shared by CSI300 paper modules."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib
import json
import math

from src.forward_labels import LEGACY_OPEN_TO_CLOSE, OPEN_TO_OPEN, validate_open_label_metadata

ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = ROOT / "data/interim/futu_cn_paper"
MODEL_DIR = ROOT / "models/paper_futu/cn_2026"
HISTORICAL_PROVIDER = ROOT / "data/interim/futu_cn_provider/2026-10-11/extracted/qlib_bin"
CHINA_TZ = timezone(timedelta(hours=8))
DEPLOYMENT_PATH = ROOT / "configs/futu_cn_paper.json"


def load_deployment(require_aligned=True):
    """Explicit paper-only deployment; never silently fall back to a different strategy."""
    config = json.loads(DEPLOYMENT_PATH.read_text(encoding="utf-8"))
    if config.get("environment") != "SIMULATE" or config.get("market") != "CN" or config.get("account_suffix") != "0000":
        raise ValueError("Deployment must use the confirmed CN simulated account")
    model_dir = (ROOT / config["model_dir"]).resolve()
    if not model_dir.is_relative_to((ROOT / "models").resolve()):
        raise ValueError("Deployed model must stay inside the project's models directory")
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    if config.get("model_metadata_sha256") and hashlib.sha256((model_dir / "metadata.json").read_bytes()).hexdigest() != config["model_metadata_sha256"]:
        raise ValueError("Deployed model metadata changed")
    if hashlib.sha256((model_dir / "model.txt").read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Deployed model hash changed")
    p = metadata["portfolio"]
    for name in ["top_k", "n_drop", "top_tier_size", "rebalance_every"]:
        if type(p[name]) is not int or p[name] <= 0:
            raise ValueError(f"Invalid deployment parameter: {name}")
    if p["n_drop"] > p["top_k"] or p["top_tier_size"] > p["top_k"]:
        raise ValueError("Invalid deployment holding or tier size")
    if not 0 < p["risk_degree"] <= 1 or not 0 < p["top_tier_weight"] <= 1:
        raise ValueError("Invalid deployment exposure")
    if config.get("label_price_basis") == OPEN_TO_OPEN:
        validate_open_label_metadata(metadata)
    elif require_aligned:
        pending = config.get("pending_deployment", {})
        if (config.get("label_price_basis") != LEGACY_OPEN_TO_CLOSE
                or pending.get("previous_model_sha256") != metadata["model_sha256"]
                or not pending.get("effective_signal_date")
                or china_now().date().isoformat() > pending["effective_signal_date"]):
            raise ValueError("Legacy close-label deployment needs an explicit, unexpired open-label transition")
    return model_dir, metadata


def validate_plan_deployment(plan, model_dir, metadata):
    """A signal's model hash alone cannot identify its portfolio allocation."""
    expected_dir = model_dir.relative_to(ROOT.resolve()).as_posix()
    if plan.get("model_dir") != expected_dir or plan.get("model_sha256") != metadata["model_sha256"] or plan.get("model_year") != metadata["model_year"]:
        raise ValueError("Signal/deployed model version mismatch")
    for name in ["top_k", "n_drop", "top_tier_size", "top_tier_weight", "risk_degree", "rebalance_every"]:
        if plan.get(name) != metadata["portfolio"][name]:
            raise ValueError(f"Signal/deployment portfolio mismatch: {name}")
    if plan.get("label_horizon") != metadata["label_horizon"]:
        raise ValueError("Signal/deployment prediction horizon mismatch")
    if metadata.get("label_price_basis") == OPEN_TO_OPEN:
        validate_open_label_metadata(metadata)
        for name in ["label_price_basis", "label_expression", "target", "execution_price_basis"]:
            if plan.get(name) != metadata[name]:
                raise ValueError(f"Signal/deployment label contract mismatch: {name}")
    if metadata.get("strategy_id") == "S6_head6_40_QREG":
        for name in ["input_kind", "retrain_months", "information_cutoff", "selection_rule"]:
            if plan.get(name) != metadata[name]:
                raise ValueError(f"Signal/deployment strategy mismatch: {name}")
        validate_serving_date(metadata, plan["signal_date"])


def validate_serving_date(metadata, signal_date):
    """A quarterly model cannot silently continue serving the next quarter."""
    if metadata.get("retrain_months") == 3:
        if not metadata["information_cutoff"] <= signal_date <= metadata["serving_signal_end"]:
            raise ValueError("Quarterly model is outside its serving quarter; a validated fresh fit is required")


def select_audited_holdings(predictions, eligible, current_codes, top_k, n_drop, sellable):
    """Exact research name-selection rule, including missing-score held names."""
    values = {code: value for code, value in predictions.items() if math.isfinite(value)}
    rank = lambda code: (-values.get(code, -math.inf), code)
    current = set(current_codes)
    if len(current) > top_k:
        raise ValueError("Actual holdings exceed the configured holding limit; reconcile first")
    selected = set(current)
    new = sorted(set(eligible).intersection(values) - current, key=rank)
    while len(selected) < top_k and new:
        selected.add(new.pop(0))
    eligible_old = current.intersection(sellable)
    changes = 0
    for candidate in new:
        old = sorted(eligible_old, key=rank)
        if not old or changes >= n_drop:
            break
        worst = old[-1]
        if values[candidate] <= values.get(worst, -math.inf):
            break
        selected.remove(worst)
        selected.add(candidate)
        eligible_old.remove(worst)
        changes += 1
    return sorted(selected, key=rank)


def china_now():
    return datetime.now(CHINA_TZ)


def to_futu_code(instrument):
    if len(instrument) != 8 or instrument[:2] not in {"SH", "SZ"} or not instrument[2:].isdigit():
        raise ValueError(f"Invalid A-share symbol: {instrument}")
    return instrument[:2] + "." + instrument[2:]


def to_qlib_code(code):
    if len(code) != 9 or code[2] != ".":
        raise ValueError(f"Invalid Futu A-share symbol: {code}")
    instrument = code[:2] + code[3:]
    to_futu_code(instrument)
    return instrument


def target_share_quantity(code, weight, nav, price):
    """Round initial buying to board-specific units; never exceed target value."""
    if price <= 0 or weight < 0 or nav <= 0:
        raise ValueError("Invalid allocation inputs")
    desired = int(weight * nav / price)
    if code.startswith("SH.688"):
        return desired if desired >= 200 else 0
    return desired // 100 * 100


def tiered_weights(ranked_codes, risk_degree=0.95, top_k=50, top_tier_size=10, top_tier_weight=0.4):
    if len(ranked_codes) != top_k or len(set(ranked_codes)) != top_k:
        raise ValueError(f"Need exactly {top_k} different selected stocks")
    if not 0 < top_tier_size <= top_k or not math.isfinite(risk_degree) or not 0 < risk_degree <= 1 or not 0 < top_tier_weight <= 1:
        raise ValueError("Invalid tiered allocation")
    if top_tier_size == top_k:
        return {code: risk_degree / top_k for code in ranked_codes}
    return {code: risk_degree * (top_tier_weight / top_tier_size if i < top_tier_size
            else (1 - top_tier_weight) / (top_k - top_tier_size)) for i, code in enumerate(ranked_codes)}


def select_holdings(predictions, eligible, current_codes, top_k=50, n_drop=5):
    """Mirror the research strategy's limited replacement and tiered ranking."""
    order = lambda codes: sorted(codes, key=lambda code: (-predictions[code], code))
    current = list(dict.fromkeys(current_codes))
    scored_current = order([code for code in current if code in predictions])
    new = order([code for code in eligible if code not in current])
    proposed = new[:n_drop + max(top_k - len(current), 0)]
    combined = order(list(dict.fromkeys(scored_current + proposed)))
    worst = set(combined[-n_drop:])
    sells = set(current).intersection(worst)
    survivors = [code for code in current if code not in sells and code in predictions]
    buy_count = len(sells) + max(top_k - len(current), 0)
    selected = list(dict.fromkeys(survivors + proposed[:buy_count]))
    if len(selected) < top_k:
        fill = [code for code in order(eligible) if code not in selected]
        selected += fill[:top_k - len(selected)]
    return order(selected)[:top_k]


def sellable_quantity(position_qty, platform_sellable, bought_today):
    return max(0, int(min(platform_sellable, position_qty - bought_today)))


def validate_execution_session(today, execution_date, model_year, market_sh, market_sz):
    if today != execution_date or today[:4] != str(model_year):
        raise ValueError("不能在信号日、过期日期或模型年份外执行调仓")
    if market_sh not in {"MORNING", "AFTERNOON"} or market_sz not in {"MORNING", "AFTERNOON"}:
        raise ValueError("沪深市场当前不在连续交易时段，不提交订单")
