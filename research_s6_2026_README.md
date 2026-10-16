# S6 head-six 40% research

Research main: raw 15-factor LightGBM, ranked forward 10-session open-to-open
return, Top20 with the first six names sharing 40% of the 95% equity sleeve,
five-session rebalance, at most six substitutions. This research registration
does not change the Futu deployment.

```powershell
python -m paper_futu.research_s6_2026 --stage all
python -m paper_futu.research_s6_2026 --stage status
```

Stages: `audit`, `train`, `backtest`, `report`, `all`, `status`.
The independent directory is
`reports/experiments/s6_head6_40_2026_holdout_asof_20261013`.

Six frozen-year comparisons V0/V2/V3/V4/V7/V8 use only labels matured by
2025-12-31 and validation years 2023/2024/2025. The existing audited models
and their tree counts are rechecked, then reused. Each receives five fresh
one-million-yuan accounts from 2026-01-05 to 2026-10-13.

Following the user's clarified scope, half-year and quarter-year refits run
through **every execution year 2022 through 2026**, not just 2026. Each year
uses its own previous-three-year validation choice: the original S6 tree counts
are 640/160/40/80/40 for 2022/2023/2024/2025/2026. Within a year the parameters,
tree count and seed stay fixed. Updates fit from scratch on all mature labels
since 2010. Quarter and half-year June cutoffs share a fit, yielding 15 new
fits at actual trading-calendar quarter ends. Model updates never introduce
extra trades or reset the five-session schedule at a year boundary.

The full-history frequency comparison has 15 accounts: five audited annual
accounts reused from the existing head-six experiment and ten new continuous
accounts beginning 2022-01-04. The two new frequency streams also receive ten
fresh 2026 accounts so their 2026 comparison shares the originally agreed
cash starting state. Total: **55 accounts, 50 new, five reused; 15 new fits**.
Training-rule variants are not crossed with cadence variants.

Training membership is filtered by actual `label_end_date`, including at every
quarter/half-year boundary. Saved-model reload predictions, old annual tree
choices, source hashes, fees, cash plus holdings, stock PnL, and equal account
trajectories before the first update are audited. Offsets preserve the same
comparison interval and continuous five-trading-day schedule.

At most two worker processes with four threads each, at least 3 GiB available
before launch and a 2 GiB in-task floor. Workers recycle after at most five
accounts. Confirmed memory failures permit one durable serial retry; ordinary
errors stop scheduling. A continuous 30-minute launch-memory shortage stops
the controller. Resume reuses only exact, hash-verified completed tasks; source
or protocol changes fail closed. No other application is closed.

The final HTML, CSV and JSON reports separate frozen-year tests, continuous
2022–2026 cadence tests, and fresh-cash 2026 cadence tests. They report return,
geometric excess, drawdown, phase dispersion, fees, turnover, holdings, monthly/
quarterly results, mature-label IC, bank exposure, stock contributions and paired
differences. Initial cash is included in drawdown. No Git commit is automatic.

2022–2026 has already been examined repeatedly. These tests audit causal
information boundaries and sensitivity; they are not untouched independent
holdouts. Existing ideal-open execution, uniform 9.5% limit filtering and
corporate-action/PIT-revision approximations remain disclosed.
