"""Bounded pre-2026 extraction and causal model reuse checks for Top5 research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'data/processed/final_ablation_asof_20261013'
MODELS = ROOT/'models/experiments/s6_quarterly_l1l2_small_asof_20261013'
STUDY = ROOT/'reports/experiments/s6_quarterly_l1l2_small_asof_20261013'
START, END = '2022-01-04', '2025-12-31'


def observe_memory(_former_threshold, purpose):
    """Observation only: user explicitly removed this experiment's RAM gate."""
    from src.runtime_budget import available_memory_bytes
    available = available_memory_bytes()/2**30
    print(f'MEMORY observed={available:.2f}GiB gate=disabled purpose={purpose}', flush=True)
    return available


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def fingerprint(frame):
    import pandas as pd
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()).hexdigest()


def bounded_parquet(path, field='datetime', end=END, columns=None, extra=()):
    """Push date predicate into Arrow before constructing a pandas dataframe."""
    import pandas as pd
    return pd.read_parquet(path, columns=columns, filters=[(field, '<=', pd.Timestamp(end)), *extra])


def truncate_binary(source, destination, count):
    """Never read the suffix after the permitted calendar position."""
    import numpy as np
    with Path(source).open('rb') as f:
        head = f.read(4)
        if not head: return
        start = int(np.frombuffer(head, dtype='<f4')[0])
        if start >= count: return
        values = head+f.read(4*max(0, count-start))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(values)


def provider_view(out):
    """Prefix-only binary reads, interval clipping, no future calendar needed."""
    import pandas as pd
    source = SOURCE/'provider'
    provider = out/'data/provider'
    dates = []
    with (source/'calendars/day.txt').open(encoding='utf-8') as f:
        for row in f:
            day = row.strip()
            if day > END: break
            dates.append(day)
    if dates[-1] != END: raise ValueError('Calendar does not reach 2025 cutoff')
    for name in ('day.txt', 'day_future.txt'):
        path = provider/'calendars'/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('\n'.join(dates)+'\n', encoding='utf-8')
    lines, codes = [], set()
    with (source/'instruments/csi300.txt').open(encoding='utf-8') as f:
        for row in f:
            code, start, end = row.split()
            if start <= END:
                lines.append(f'{code}\t{start}\t{min(end, END)}')
                codes.add(code)
    path = provider/'instruments/csi300.txt'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    missing = []
    for code in sorted(codes):
        for field in ('open', 'close', 'high', 'low', 'volume', 'factor'):
            name = f'features/{code.lower()}/{field}.day.bin'
            if not (source/name).exists():
                # Some historical membership intervals refer to securities absent
                # from the frozen provider. Qlib represents these fields as NaN.
                # Preserve the universe and missingness; never invent prices.
                missing.append(name)
                continue
            truncate_binary(source/name, provider/name, len(dates))
    hashes = {str(p.relative_to(provider)): digest(p) for p in sorted(provider.rglob('*')) if p.is_file()}
    save(out/'provider_prefix_manifest.json', dict(end=END, files=hashes, stocks=len(codes),missing_source_fields=missing,
         method='read binary header then only pre-cutoff bytes; clip historical membership intervals'))
    return provider, pd.DatetimeIndex(pd.to_datetime(dates)), codes


def init_qlib(provider):
    import qlib
    from qlib.constant import REG_CN
    qlib.init(provider_uri=str(provider), region=REG_CN, kernels=1, expression_cache=None,
              dataset_cache=None, mem_cache_size_limit=1, logging_level=30)


def recompute_features(provider, end=END):
    import numpy as np
    from qlib.data import D
    from src.factor_library import FEATURE_EXPRESSIONS, ENHANCED_FEATURE_NAMES
    from src.build_enhanced_dataset import (merge_latest_event, prepare_general_report_events,
                                            prepare_annual_eps_events, validate_point_in_time)
    daily = D.features(D.instruments('csi300'), list(FEATURE_EXPRESSIONS.values())+['$close / ($factor + 1e-12)'],
                       start_time='2010-01-01', end_time=end, freq='day')
    daily.columns = list(FEATURE_EXPRESSIONS)+['raw_close']
    events = bounded_parquet(SOURCE/'financial_events.parquet', 'available_date', end)
    if (events.available_date <= events.report_date).any(): raise ValueError('Financial date error')
    panel = merge_latest_event(daily.reset_index(), prepare_general_report_events(events), 'fundamental_available_date')
    panel = merge_latest_event(panel, prepare_annual_eps_events(events), 'annual_eps_available_date')
    validate_point_in_time(panel)
    price = panel.raw_close.where(panel.raw_close.gt(0))
    panel['book_to_price'] = panel.book_value_per_share/price
    panel['earnings_to_price'] = panel.annual_eps/price
    raw = panel.set_index(['instrument', 'datetime'])[ENHANCED_FEATURE_NAMES]
    raw = raw.replace([np.inf, -np.inf], np.nan).sort_index()
    raw = raw.loc[~raw.isna().all(axis=1)].astype('float32')
    alignment = panel[['instrument', 'datetime', 'fundamental_report_date', 'fundamental_available_date',
                       'annual_eps_report_date', 'annual_eps_available_date']]
    return raw, events, alignment


def choice_from_folds(records, year):
    import pandas as pd
    f = pd.DataFrame(records)
    if set(f.validation_year) != set(range(year-3, year)):
        raise ValueError('Tree selection does not use exactly previous 3 years')
    for r in records:
        y = r['validation_year']
        if r['train_label_end_max'] >= f'{y}-01-01' or r['valid_label_end_max'] > f'{y}-12-31':
            raise ValueError('Fold label boundary violation')
    summary = f.groupby(['parameter_id','trees']).agg(mean=('mean_rank_ic','mean'),
                 std=('mean_rank_ic','std'), leaves=('num_leaves','first'), count=('validation_year','nunique')).reset_index()
    if not summary['count'].eq(3).all(): raise ValueError('Incomplete fold candidates')
    summary['complexity'] = summary.trees*summary.leaves
    return int(summary.loc[summary['mean'] >= summary['mean'].max()-1e-8]
               .sort_values(['std','complexity','parameter_id','trees']).iloc[0].trees)


def prepare(out):
    require_memory = observe_memory
    require_memory(1.5, 'Pre-2026 Top5 audit startup')
    import gc
    import lightgbm as lgb
    import numpy as np
    import pandas as pd
    from qlib.data import D
    from src.factor_library import ENHANCED_FEATURE_NAMES
    print('AUDIT building physically truncated provider', flush=True)
    provider, calendar, codes = provider_view(out)
    init_qlib(provider)
    print('AUDIT reconstructing all 15 features without future records', flush=True)
    features, events, alignment = recompute_features(provider)
    old = bounded_parquet(SOURCE/'raw_features.parquet')
    pd.testing.assert_index_equal(features.index, old.index)
    changed = not np.allclose(features.to_numpy(), old.to_numpy(), equal_nan=True, rtol=0, atol=0)
    maxdiff = (features-old).abs().max().to_dict()
    features.to_parquet(out/'data/features.parquet')
    events.to_parquet(out/'data/financial_events.parquet', index=False)
    alignment.to_parquet(out/'data/financial_alignment.parquet', index=False)
    del old, events, alignment; gc.collect()
    print(f'AUDIT feature equality changed={changed}', flush=True)
    label = D.features(D.instruments('csi300'), ['Ref($open, -11) / Ref($open, -1) - 1'],
                       start_time='2010-01-01', end_time=END)
    label.columns = ['label']; label = label.reindex(features.index).replace([np.inf, -np.inf], np.nan)
    positions = calendar.get_indexer(features.index.get_level_values('datetime'))+11
    known = positions < len(calendar)
    ends = np.full(len(positions), np.datetime64('NaT'), dtype='datetime64[ns]')
    ends[known] = calendar.to_numpy()[positions[known]]
    label['label_end_date'] = ends
    label.loc[~known, 'label'] = np.nan
    label['rank_target'] = label.label.groupby(level='datetime').rank(method='average', pct=True)-.5
    label.to_parquet(out/'data/labels.parquet')
    frame = features.join(label); frame['target'] = frame.rank_target
    protocol = read(STUDY/'protocol.json')  # metadata only; select only <=2025 bindings
    periods = [p for p in protocol['periods'] if p['start'] <= END]
    if len(periods) != 16 or any(p['end'] > END for p in periods): raise ValueError('Expected 16 complete quarters')
    source_pred = bounded_parquet(STUDY/'predictions/history_QREG.parquet', 'execution_date', END)
    if source_pred.index.get_level_values('datetime').max() > pd.Timestamp(END): raise ValueError('Future signal')
    # Independently derive the trading sample, including signals with no mature
    # outcome. Never let a label-filtered legacy prediction file drop these rows.
    signal_pos = calendar.get_indexer(features.index.get_level_values('datetime'))
    executable = (signal_pos+1 < len(calendar)) & (features.index.get_level_values('datetime') >= '2021-12-31')
    expected_pred = pd.DataFrame(index=features.index[executable])
    expected_pred['execution_date'] = calendar.to_numpy()[signal_pos[executable]+1]
    rows, pieces, bindings = [], [], {}
    for year in range(2022, 2026):
        choice = protocol['annual_bindings'][str(year)]
        if choice_from_folds(choice['fold_records'], year) != choice['trees']:
            raise ValueError('Annual tree selection mismatch')
        bindings[str(year)] = {k: choice[k] for k in ('trees','parameters','validation_years','fold_records')}
    for period in periods:
        require_memory(1., 'Top5 causal model audit')
        cutoff = period['cutoff']; meta_path = MODELS/cutoff/'metadata.json'
        meta = read(meta_path)
        choice = bindings[str(period['year'])]
        if meta['cutoff'] != cutoff or meta['label_end_max'] > cutoff or meta['train_signal_end'] > cutoff:
            raise ValueError('Invalid model maturity boundary')
        if meta['trees'] != choice['trees'] or digest(meta['model_path']) != meta['model_sha256']:
            raise ValueError('Model binding mismatch')
        expected_params = {**choice['parameters'], 'reg_alpha':1., 'reg_lambda':5.,
                           'num_leaves':15, 'min_child_samples':300, 'n_jobs':4}
        if meta['parameters'] != expected_params: raise ValueError('Unexpected S6 parameters')
        training = frame.loc[frame.label_end_date.le(pd.Timestamp(cutoff))].dropna(subset=['label','target'])
        training_hash = fingerprint(training)
        reusable = not changed and training_hash == meta['training_fingerprint'] and len(training) == meta['train_rows']
        part = expected_pred.loc[expected_pred.execution_date.between(period['start'], period['end'])].copy()
        part['model_cutoff'] = pd.Timestamp(cutoff)
        old_part = source_pred.loc[source_pred.execution_date.between(period['start'], period['end'])]
        if not (old_part.model_cutoff.eq(pd.Timestamp(cutoff)) & old_part.model_cutoff.lt(old_part.execution_date)).all():
            raise ValueError('Source serving model cutoff mismatch')
        if not old_part.index.isin(part.index).all(): raise ValueError('Source prediction outside historical pool')
        if part.empty or not (part.model_cutoff <= part.index.get_level_values('datetime')).all():
            raise ValueError('Model unavailable at signal close')
        if not (part.index.get_level_values('datetime') < part.execution_date).all():
            raise ValueError('Signal was not known before execution')
        if reusable:
            model_path = Path(meta['model_path'])
            model = lgb.Booster(model_file=str(model_path))
        else:
            print(f'AUDIT rebuilding affected model {cutoff}, no new parameter search', flush=True)
            require_memory(1.5, 'Rebuild causal quarter')
            fit = lgb.LGBMRegressor(**expected_params, n_estimators=choice['trees'], random_state=42)
            fit.fit(training[ENHANCED_FEATURE_NAMES], training.target)
            model_path = out/'models'/cutoff/'model.txt'; model_path.parent.mkdir(parents=True, exist_ok=True)
            fit.booster_.save_model(str(model_path)); model = lgb.Booster(model_file=str(model_path))
            np.testing.assert_allclose(fit.predict(features.reindex(part.index)),
                                       model.predict(features.reindex(part.index), num_threads=4), atol=1e-12, rtol=0)
            del fit
        values = model.predict(features.reindex(part.index), num_threads=4)
        part['prediction'] = values
        if reusable: np.testing.assert_allclose(part.loc[old_part.index,'prediction'], old_part.prediction, atol=1e-12, rtol=0)
        pieces.append(part)
        rows.append(dict(**period, reused=reusable, model_path=str(model_path), model_sha256=digest(model_path),
                         train_rows=len(training), train_signal_end=str(training.index.get_level_values('datetime').max().date()),
                         label_end_max=str(training.label_end_date.max().date()), training_fingerprint=training_hash,
                         trees=choice['trees'], reload_equal=True, parameters=expected_params,
                         recovered_missing_signal_rows=len(part)-len(old_part)))
        print(f'AUDIT {cutoff} reused={reusable} rows={len(training)}', flush=True)
        del training, model; gc.collect()
    pred = pd.concat(pieces).sort_index()
    if pred.index.has_duplicates or not np.isfinite(pred.prediction).all(): raise ValueError('Invalid predictions')
    pred.to_parquet(out/'data/predictions.parquet')
    save(out/'models_audit.json', dict(periods=rows, annual_bindings=bindings))
    del frame, features, source_pred; gc.collect()
    fields = ['open','close','factor','volume']
    market = D.features(sorted(codes), ['$'+f for f in fields], start_time='2021-12-30', end_time=END)
    market.columns = fields
    market.to_parquet(out/'data/market.parquet')
    # Evaluate IC only for fully matured labels; don't drop valid trading signals.
    from paper_futu.final_ablation_core import daily_ic
    ic = daily_ic(pred.prediction, label.label.reindex(pred.index))
    ic.to_csv(out/'daily_ic.csv', encoding='utf-8-sig')
    # Bounded official request, independent of old endpoint/terminal corrections.
    from paper_futu.benchmark_attribution import fetch_official, official_frame
    rawdir = out/'data/benchmark'
    fetch_official(rawdir, '2021-12-31', END)
    benchmark = official_frame(read(rawdir/'H00300.json'), 'H00300')
    dates = calendar[(calendar >= START) & (calendar <= END)]
    if not dates.isin(benchmark.index).all(): raise ValueError('Benchmark day missing')
    returns = benchmark.close.pct_change(fill_method=None).reindex(dates)
    returns.iloc[0] = benchmark.loc[dates[0], 'close']/benchmark.loc[dates[0], 'open']-1
    if not np.isfinite(returns).all(): raise ValueError('Invalid benchmark return')
    returns.rename('benchmark_return').to_frame().to_parquet(out/'data/benchmark.parquet')
    save(out/'data_audit.json', dict(status='passed', end=END, start=START,
        feature_changed=changed, feature_max_difference=maxdiff, rebuilt_models=sum(not r['reused'] for r in rows),
        prediction_rows=len(pred), trading_days=len(dates), immature_ic_rows=int(label.label.isna().sum()),
        evidence='Features rebuilt from a physically truncated provider and pre-cutoff financial events; all serving predictions reloaded.',
        financial_revision_archive_complete=False,
        limitation='No vintage financial revision archive or point-in-time corporate-action vintage. Prefix invariance is not proof against vendor backfills.',
        benchmark_first_day='close/open; subsequent close/previous close; no legacy terminal patch'))
