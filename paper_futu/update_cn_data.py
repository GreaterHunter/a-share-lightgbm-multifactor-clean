"""Bounded, locked public-data update and inference. No broker SDK or orders.

Run after 15:35 or before the following session to catch up. Publication is
all-or-nothing per session; bad/stale feeds never replace an active plan.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import msvcrt
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from paper_futu.cn_common import HISTORICAL_PROVIDER, ROOT, STATE_DIR, china_now, to_futu_code
from paper_futu.independent_cn_market import PublicClient, normalize_stock, sha, write_json
from paper_futu.cn_daily_history import prior_saved_parts,merge_history,save_supplement


def calendar():
    return pd.DatetimeIndex(pd.to_datetime((HISTORICAL_PROVIDER/"calendars/day_future.txt").read_text().splitlines()))


def completed_date(now, dates):
    today = pd.Timestamp(now.date())
    eligible = dates[(dates < today) | ((dates == today) & ((now.hour, now.minute) >= (15, 35)))]
    if eligible.empty or today > dates.max():
        raise ValueError("Trading calendar missing/expired")
    return str(eligible.max().date())


def run(module, *args):
    subprocess.run([sys.executable, "-X", "utf8", "-m", module, *map(str, args)], cwd=ROOT,
                   check=True, timeout=1200, stdout=sys.stdout, stderr=sys.stderr,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def load_history(day, members):
    import qlib
    from qlib.config import REG_CN
    from qlib.data import D
    qlib.init(provider_uri=str(HISTORICAL_PROVIDER), region=REG_CN, kernels=1)
    history = D.features(members.instrument.tolist(), ["$open", "$close", "$high", "$low", "$volume", "$factor"],
        start_time=str((pd.Timestamp(day)-pd.Timedelta(days=500)).date()), freq="day")
    history.columns = ["open", "close", "high", "low", "volume", "factor"]
    history = history.loc[history.index.get_level_values("datetime") < pd.Timestamp(day)]
    parts=[history,*prior_saved_parts(day)]
    return merge_history(parts)


def make_package(day, members_dir, output, cache, progress=None):
    output.mkdir(parents=True, exist_ok=True)
    members = pd.read_csv(members_dir/"members.csv")
    if len(members) != 300 or members.instrument.nunique() != 300:
        raise ValueError("Expected 300 distinct official constituents")
    if not pd.to_datetime(members.publisher_asof).le(pd.Timestamp(day)).all():
        raise ValueError("Future constituent snapshot")
    history = load_history(day, members)
    dates = calendar()
    first = history.index.get_level_values("datetime").max()
    required = dates[(dates > first) & (dates <= pd.Timestamp(day))]
    if required.empty or len(required) > 20 or required[-1] != pd.Timestamp(day):
        raise ValueError("Automatic backfill is limited to 20 completed sessions; review larger gaps")
    client = PublicClient(cache)
    rows, snapshots, suspended = [], [], []
    for i, member in enumerate(members.itertuples(), 1):
        instrument = member.instrument
        stock = history.xs(instrument, level="instrument", drop_level=False).dropna(subset=["factor"])
        if stock.empty:
            raise ValueError(f"No verified normalization history: {instrument}")
        previous = stock.iloc[-1]
        valid = stock.loc[stock.close.notna() & stock.factor.gt(0)]
        if valid.empty:
            raise ValueError(f"Missing price anchor: {instrument}")
        last_valid = valid.iloc[-1]
        try:
            source = client.stock(instrument, str(last_valid.name[1].date()), day)
            # A new constituent was not in yesterday's saved 300-name slice.
            # Backfill its own missing price history from its verified Qlib anchor,
            # without pretending it belonged to a past constituent list.
            stock_required=dates[(dates>pd.Timestamp(previous.name[1])) & (dates<=pd.Timestamp(day))]
            if len(stock_required)>320:raise ValueError('Stock anchor exceeds verified source coverage: '+instrument)
            new, halts = normalize_stock(instrument, previous, last_valid, *source, stock_required)
        except Exception:
            client.invalidate(instrument)
            raise
        rows += new
        suspended_today = day in halts
        suspended += [instrument] if suspended_today else []
        snapshots.append(dict(code=to_futu_code(instrument), name=member.name, suspension=suspended_today,
            update_time=day+" 15:35:00", source="validated_public_daily_bars_not_a_broker_quote"))
        if progress:
            progress(i, instrument)
    bars = pd.DataFrame(rows).set_index(["instrument", "datetime"]).sort_index()
    if bars.index.has_duplicates or np.isinf(bars.to_numpy()).any():
        raise ValueError("Duplicate or infinite normalized bars")
    supplements=bars.loc[bars.index.get_level_values('datetime')<required.min()]
    supplement_path=save_supplement(day,supplements,cache)
    for name in ["members.csv", "members_source.json", "members_source.xls"]:
        if (members_dir/name).exists() and (members_dir/name).resolve() != (output/name).resolve():
            shutil.copy2(members_dir/name, output/name)
    for past in required[:-1]:
        backfill = output/"backfill"/str(past.date())
        backfill.mkdir(parents=True, exist_ok=True)
        bars.loc[bars.index.get_level_values("datetime") == past].to_parquet(backfill/"bars.parquet")
        write_json(backfill/"bars_units.json", dict(volume="hands_divided_by_adjustment_factor"))
        write_json(backfill/"backfill_source.json", dict(price_date=str(past.date()), fetched_for_signal_date=day,
            universe_source_date=day, historical_membership_asserted=False,
            purpose="price history only; no retrospective signal/plan/NAV is fabricated"))
    bars = bars.loc[bars.index.get_level_values("datetime") == pd.Timestamp(day)]
    bars.to_parquet(output/"bars.parquet")
    write_json(output/"bars_units.json", dict(volume="hands_divided_by_adjustment_factor"))
    pd.DataFrame(snapshots).to_csv(output/"snapshots.csv", index=False, encoding="utf-8-sig")
    manifest = dict(status="validated", validation_schema=2, date=day, retrieved_at=china_now().isoformat(), stocks=len(bars),
        suspended=suspended, primary="Sina raw OHLC/share volume + dated cumulative hfq factor",
        cross_check="Tencent unadjusted daily OHLC and share-normalized volume; every active stock",
        factor_policy="existing saved factor anchored to ratio of dated Sina factors; no history restatement",
        volume_policy="Sina shares / 100 / normalized factor; Tencent STAR shares, other stock hands",
        quote_timestamp_policy="update_time is a completed-session convention, NOT an observed broker timestamp",
        backfilled_sessions=[str(d.date()) for d in required[:-1]],
        price_history_supplement=supplement_path,
        source_cache=str(cache.relative_to(ROOT)),
        source_artifacts={p.relative_to(cache).as_posix():sha(p) for p in sorted(cache.glob("*/*.json"))},
        artifacts={name:sha(output/name) for name in ["bars.parquet", "bars_units.json", "members.csv", "snapshots.csv"]
            + [str(p.relative_to(output).as_posix()) for p in sorted((output/"backfill").glob("*/*"))]})
    write_json(output/"market_manifest.json", manifest)
    return manifest


def activation_allowed(day,current,now):
    """Data catchup must not be confused with permission to chase an old rebalance."""
    due=current['next_rebalance_signal_date']
    if day>due:return False,'missed_rebalance_signal_manual_review'
    if day!=due:return False,None
    dates=calendar();future=dates[dates>pd.Timestamp(day)]
    if future.empty:return False,'trading_calendar_expired'
    execution=str(future.min().date())
    from paper_futu.execution_override import late_activation_allowed
    if (now.date().isoformat()<execution or
        (now.date().isoformat()==execution and (now.hour,now.minute)<(9,25)) or late_activation_allowed(day,now)):
        return True,None
    return False,'missed_execution_window_no_stale_activation'


def build_signal(day, package, check_only=False):
    from paper_futu.cn_deployment import model_for_signal
    current = json.loads((STATE_DIR/"active_plan.json").read_text(encoding="utf-8"))
    args = ["--date", day, "--prebuilt-bars", package/"bars.parquet"]
    if check_only:
        directory, _ = model_for_signal(day)
        args += ["--daily-only", "--model-dir", directory.relative_to(ROOT).as_posix(),
                 "--output-dir", package.relative_to(ROOT).as_posix()]
    elif json.loads((ROOT/'configs/futu_cn_paper.json').read_text(encoding='utf-8')).get('execution_strategy') == 'R80_S1_TOP10_FULL_TP10_SL5':
        # Same predictor, separate stateful executor. Never activate a legacy
        # five-session Top20 rebalance from the daily data updater.
        allowed,blocked=False,'rank80_daily_signal_separate_executor'
        args += ['--daily-only']
    else:
        allowed,blocked=activation_allowed(day,current,china_now())
        if not allowed:
            args += ["--daily-only"]
    run("paper_futu.build_cn_signal", *args)
    return dict(active_plan_changed=bool(not check_only and allowed),
                activation_blocked_reason=None if check_only else blocked)


def update(day=None, check_only=False):
    now = china_now()
    latest = completed_date(now, calendar())
    day = day or latest
    if day > latest or pd.Timestamp(day) not in calendar():
        raise ValueError("Cannot use intraday/future/non-trading-day data")
    status_path = STATE_DIR/"independent_data_status.json"
    def progress(count, instrument):
        write_json(status_path, dict(time=china_now().isoformat(), pid=os.getpid(), stage="fetching",
            date=day, stocks_done=count, stocks_total=300, last_stock=instrument, check_only=check_only,
            broker_connection_required=False))
    target = STATE_DIR/day
    package = ROOT/"reports/paper_futu/independent_data_validation"/day if check_only else target
    if not check_only and (target/"independent_data_ready.json").exists():
        ready = json.loads((target/"independent_data_ready.json").read_text(encoding="utf-8"))
        if sha(target/"plan.json") != ready["plan_sha256"]:
            raise ValueError("Saved daily signal changed after independent update")
        manifest=json.loads((target/'market_manifest.json').read_text(encoding='utf-8'))
        for name,expected in manifest['artifacts'].items():
            if sha(target/name)!=expected:raise ValueError('Saved market artifact changed: '+name)
        for name,expected in ready.get('signal_artifacts',{}).items():
            if sha(target/name)!=expected:raise ValueError('Saved signal artifact changed: '+name)
        result=dict(stage='already_complete',date=day,time=now.isoformat(),broker_connection_required=False,orders_submitted=0)
        write_json(status_path,result)
        return result
    # Never mutate a legacy historical session to switch data suppliers.
    if (target/"bars.parquet").exists() and not check_only and not (target/"market_manifest.json").exists():
        return dict(stage="verified_legacy_bars_preserved", date=day, broker_connection_required=False)
    if not (target/"members.csv").exists():
        run("paper_futu.fetch_cn_members", "--date", day)
    cache = ROOT/"data/interim/public_cn_daily"/day
    manifest_path = package/"market_manifest.json"
    needs_rebuild = not manifest_path.exists()
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")).get("validation_schema") != 2:
        if not check_only:
            raise ValueError("Existing market package uses an earlier validation version; review before replacing")
        needs_rebuild = True
    if needs_rebuild:
        # Cache and staging survive a network error, but incomplete packages are
        # never published to the live bars path.
        staging = package.parent/(package.name+".pending")
        manifest = make_package(day, target, staging, cache, progress)
        package.mkdir(parents=True, exist_ok=True)
        for name in [*manifest["artifacts"], "members_source.json", "members_source.xls", "market_manifest.json"]:
            if (staging/name).exists():
                (package/name).parent.mkdir(parents=True, exist_ok=True)
                os.replace(staging/name, package/name)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Never fabricate past trades, signals or account observations. These are
    # only price rows needed to calculate the latest signal's rolling features.
    if not check_only:
        for past in manifest.get("backfilled_sessions", []):
            source = package/"backfill"/past
            destination = STATE_DIR/past
            destination.mkdir(parents=True, exist_ok=True)
            for name in ["bars.parquet", "bars_units.json", "backfill_source.json"]:
                if sha(source/name) != manifest["artifacts"][f"backfill/{past}/{name}"]:
                    raise ValueError("Backfill package changed")
                if (destination/name).exists():
                    if sha(destination/name) != sha(source/name):
                        raise ValueError("Refusing to replace existing historical market evidence")
                else:
                    shutil.copy2(source/name, destination/name)
    signal_status=build_signal(day, package, check_only)
    result = dict(stage="validated_check_only" if check_only else "data_and_signal_ready", date=day,
        time=china_now().isoformat(), broker_connection_required=False,
        market_manifest_sha256=sha(package/"market_manifest.json"), orders_submitted=0,**signal_status)
    if not check_only:
        write_json(target/"independent_data_ready.json", dict(**result, plan_sha256=sha(target/"plan.json"),
            signal_artifacts={name:sha(target/name) for name in ['features_raw.parquet','features_ranked.parquet','predictions.csv','plan.json']}))
    write_json(status_path, result)
    return result


def catchup():
    """Recover saved-universe dates in order, then latest; never invent old membership."""
    latest=completed_date(china_now(),calendar())
    ready_days=sorted(p.parent.name for p in STATE_DIR.glob('*/independent_data_ready.json')
                      if len(p.parent.name)==10 and p.parent.name<=latest)
    last=ready_days[-1] if ready_days else None
    dates=calendar(); latest_stamp=pd.Timestamp(latest)
    if last and len(dates[(dates>pd.Timestamp(last))&(dates<=latest_stamp)])>20:
        raise ValueError('Over 20 missed sessions; bounded source/model-calendar review required')
    pending=sorted({p.parent.name for p in STATE_DIR.glob('*/members.csv')
                    if len(p.parent.name)==10 and (last is None or p.parent.name>last) and p.parent.name<=latest}|{latest})
    results=[]
    for day in pending:results.append(update(day))
    value=dict(stage='catchup_complete',date=latest,time=china_now().isoformat(),sessions=results,
               broker_connection_required=False,orders_submitted=0,
               historical_account_nav_fabricated=False,historical_membership_fabricated=False)
    write_json(STATE_DIR/'data_catchup_status.json',value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date")
    parser.add_argument("--check-only", action="store_true", help="Isolated report; no active plan or holdings changes")
    parser.add_argument('--catch-up',action='store_true',help='Recover missed saved-universe sessions then latest completed date')
    args = parser.parse_args()
    if args.catch_up and (args.date or args.check_only):parser.error('--catch-up cannot be combined with --date/--check-only')
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if sys.stdout is None:
        sys.stdout = (STATE_DIR/"independent_data.stdout.log").open("a", encoding="utf-8", buffering=1)
    if sys.stderr is None:
        sys.stderr = (STATE_DIR/"independent_data.stderr.log").open("a", encoding="utf-8", buffering=1)
    with (STATE_DIR/"independent_data.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0"); lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("Independent data update already running; no duplicate", flush=True)
            return
        try:
            print(json.dumps(catchup() if args.catch_up else update(args.date, args.check_only), ensure_ascii=False), flush=True)
        except Exception as error:
            write_json(STATE_DIR/"independent_data_status.json", dict(stage="error", time=china_now().isoformat(),
                reason=str(error), broker_connection_required=False, active_plan_not_replaced_by_error=True))
            raise


if __name__ == "__main__":
    main()
