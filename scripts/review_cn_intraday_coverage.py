"""Read-only source comparison; missing bars never imply a suspension by default."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.prepare_cn_intraday_public import OUT, SOURCE, START, END, save


def binary_series(provider, code, field, calendar):
    path=provider/'features'/code.lower()/f'{field}.day.bin'
    if not path.exists(): return pd.Series(np.nan,index=calendar)
    with path.open('rb') as f:
        head=f.read(4)
        if len(head)!=4: return pd.Series(np.nan,index=calendar)
        offset=int(np.frombuffer(head,dtype='<f4')[0])
        values=np.frombuffer(f.read(max(0,len(calendar)-offset)*4),dtype='<f4').copy()
    return pd.Series(values,index=calendar[offset:offset+len(values)]).reindex(calendar)


def run():
    summary=json.loads((OUT/'data_summary.json').read_text(encoding='utf8'))
    provider=SOURCE/'provider'
    dates=[]
    with (provider/'calendars/day.txt').open(encoding='utf8') as f:
        for line in f:
            day=line.strip()
            if day>END: break
            dates.append(day)
    calendar=pd.DatetimeIndex(pd.to_datetime(dates))
    features=pd.read_parquet(SOURCE/'features.parquet',columns=['book_to_price'],
        filters=[('datetime','>=',pd.Timestamp(START)),('datetime','<=',pd.Timestamp(END))])
    expected={code:g.index.get_level_values('datetime') for code,g in features.groupby(level='instrument')}
    del features
    gaps=[];differences=[]
    for record in summary['results']:
        code=record['instrument']
        if code=='SH510300': continue
        daily=pd.read_parquet(OUT/'daily'/f'{code}.parquet')
        volume=binary_series(provider,code,'volume',calendar)
        factor=binary_series(provider,code,'factor',calendar)
        absent=expected[code].difference(daily.index)
        for day in absent:
            v=volume.get(day,np.nan)
            reason='active_daily_volume' if np.isfinite(v) and v>0 else 'zero_daily_volume' if v==0 else 'unknown_daily_volume'
            gaps.append(dict(instrument=code,date=day,year=day.year,classification=reason,
                             reference_volume=None if not np.isfinite(v) else float(v)))
        for field in ('open','close'):
            price=binary_series(provider,code,field,calendar)/factor
            common=daily.index.intersection(expected[code])
            reference=price.reindex(common);observed=daily[field].reindex(common)
            delta=(reference-observed).abs();relative=delta/reference
            bad=(reference>0)&delta.gt(.011)&relative.gt(.001)
            for day in common[bad]:
                differences.append(dict(instrument=code,date=day,year=day.year,field=field,
                    minute_aggregate=float(observed[day]),daily_reference=float(reference[day]),
                    relative_error=float(relative[day])))
    g=pd.DataFrame(gaps);d=pd.DataFrame(differences)
    g.to_csv(OUT/'coverage_gaps.csv',index=False,encoding='utf-8-sig')
    d.to_csv(OUT/'daily_price_differences.csv',index=False,encoding='utf-8-sig')
    grouped=g.groupby(['year','classification']).size().rename('count').reset_index()
    price_group=d.groupby(['year','field']).size().rename('count').reset_index()
    result=dict(status='review_required_before_training',files=len(summary['results']),
        gaps=len(g),gap_classification=g.classification.value_counts().to_dict(),
        gaps_by_year=grouped.to_dict('records'),price_differences=len(d),
        price_differences_by_year=price_group.to_dict('records'),
        test_period_active_gaps=int(((g.year>=2022)&g.classification.eq('active_daily_volume')).sum()),
        note='Zero volume is a no-trade indication, not independently verified suspension; unknowns are not filled. Price checks now include 2015-2025, unlike initial 2022-2025 sample checks.',
        training_started=False,backtest_complete=False)
    save(OUT/'coverage_review.json',result)
    print(json.dumps(result,ensure_ascii=True),flush=True)


if __name__=='__main__': run()
