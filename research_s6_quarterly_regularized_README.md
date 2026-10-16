# S6 quarterly refitting with combined constraints

```powershell
python -m paper_futu.research_s6_quarterly_regularized --stage all
python -m paper_futu.research_s6_quarterly_regularized --stage status
```

One new combination only: `reg_alpha=1`, `reg_lambda=5`, `num_leaves=15`,
`min_child_samples=300`; other S6 parameters are unchanged. This is a paired
fixed-tree-count experiment, not a new hyperparameter search. At the start of
each execution year the original S6 prior-three-year tree choice is retained:
640/160/40/80/40 for 2022/2023/2024/2025/2026. Trees and parameters stay fixed
within the execution year. Each quarterly fit starts from scratch using only
labels with actual end dates at or before the prior quarter's last session.

Quarterly refits cover 2022–2026, including the initial model before each year:
20 new final fits and ten accounts (five continuous from 2022, five fresh from
2026). Initial capital is one million yuan. Compare against hash-verified
annual and quarterly S6 accounts from the completed parent experiment.
Top20, first six sharing 40% of the 95% equity sleeve, five-session rebalance,
at most six substitutions, unchanged commission/lot/T+1 and tradability logic.
The explicit-context account adapter has an AST-identical replay body to the
previous audited adapter, with only a local context binding added.

Independent outputs:
`reports/experiments/s6_quarterly_l1l2_small_asof_20261013`.
The original S6 research-main registration and Futu deployment are unchanged.
Source and output hashes, exact training input fingerprints, model reload
predictions, account cash/holdings/fees and stock profit accounting are checked.
No automated Git commit. Maximum two workers, four threads each, 3 GiB launch
and 2 GiB in-task RAM gates, short account batches, one durable serial retry
for a confirmed memory error, no unlimited retries.

Financial information is audited, not silently refreshed in this comparison.
The fixed 2026-10-13 snapshot contains 2026 Q1 and half-year reports. Of the
300 stocks at the final date, 290 align to the 2026 half-year report and ten
have no general financial alignment; B/P and E/P each have eleven missing
values, and cash-flow/revenue has ten. E/P uses the latest disclosed full-year
basic EPS (2025 for the 290 covered stocks), not quarterly or TTM EPS. The
latest report end is 2026-06-30; 2026 Q3 reports have not been incorporated.
Disclosure dates govern availability; missing values stay missing. The
financial-value revision archive remains incomplete.

The historical period has already been inspected repeatedly. Positive returns
are not proof of no overfitting; ideal opening fills and the existing uniform
limit/corporate-action approximations remain disclosed in the final report.
