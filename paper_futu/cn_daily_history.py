"""Merge immutable daily evidence and explicit price-only membership supplements."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from paper_futu.cn_common import ROOT,STATE_DIR
from paper_futu.independent_cn_market import sha,write_json


def read_bars(path):
    path=Path(path)
    bars=pd.read_parquet(path)
    marker=path.with_name('bars_units.json')
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')).get('volume')!='hands_divided_by_adjustment_factor':
            raise ValueError('Unknown saved-bar volume unit: '+str(path))
    elif path.name=='history_repairs.parquet' or path.parent.name in {'2026-10-12','2026-10-13'}:
        bars=bars.copy();bars['volume']/=100
    else:raise ValueError('Unmarked saved-bar volume unit: '+str(path))
    return bars


def merge_history(parts):
    result=pd.concat(parts).sort_index()
    if result.index.has_duplicates:
        duplicated=result.loc[result.index.duplicated(keep=False)]
        for _,group in duplicated.groupby(level=['instrument','datetime']):
            values=group.to_numpy(dtype=float)
            if not np.all(np.isclose(values,values[0],rtol=1e-10,atol=1e-10,equal_nan=True)):
                raise ValueError('Conflicting historical/live bar evidence')
        result=result.loc[~result.index.duplicated(keep='first')]
    return result


def prior_saved_parts(day,state_dir=STATE_DIR):
    parts=[]
    for path in sorted(state_dir.glob('*/bars.parquet')):
        if len(path.parent.name)==10 and path.parent.name < day:
            parts.append(read_bars(path))
    for path in sorted((state_dir/'price_history_supplements').glob('*/bars.parquet')):
        meta=json.loads(path.with_name('supplement.json').read_text(encoding='utf-8'))
        if sha(path)!=meta['bars_sha256']:raise ValueError('Price supplement changed')
        frame=read_bars(path)
        parts.append(frame.loc[frame.index.get_level_values('datetime') < pd.Timestamp(day)])
    return parts


def save_supplement(day,frame,cache):
    if frame.empty:return None
    directory=STATE_DIR/'price_history_supplements'/day
    path=directory/'bars.parquet'
    if path.exists():
        prior=read_bars(path)
        frame=merge_history([prior,frame])
    directory.mkdir(parents=True,exist_ok=True)
    frame.sort_index().to_parquet(path)
    write_json(directory/'bars_units.json',dict(volume='hands_divided_by_adjustment_factor'))
    write_json(directory/'supplement.json',dict(bars_sha256=sha(path),generated_for=day,
        source_cache=str(Path(cache).relative_to(ROOT)),historical_membership_asserted=False,
        purpose='Causal rolling-price history for newly observed members; never historical selections, trades or NAV'))
    return str(path.relative_to(ROOT))
