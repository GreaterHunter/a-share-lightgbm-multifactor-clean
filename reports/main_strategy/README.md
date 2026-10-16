# 最终主策略报告

最终模型为10年窗口、年度滚动重训的 `ranked_enhanced` LightGBM。模型通过时间序列多折验证选择树数，预测未来10日横截面收益排名，每10个交易日调仓。

## 总体结果（2022—2024）

| 年化收益 | 年化超额 | 夏普 | 信息比率 | 最大回撤 | 日均换手 |
|---:|---:|---:|---:|---:|---:|
| 8.57% | 17.32% | 0.599 | 1.613 | -18.22% | 4.28% |

## 图表

![净值曲线](figures/nav_curve.png)

![回撤曲线](figures/drawdown.png)

![年度收益](figures/annual_returns.png)

## 文件

- `backtest_comparison.csv`：全区间指标；
- `backtest_yearly.csv`：分年度指标；
- `backtest_daily.csv`：每日账户净值、基准、超额、换手和成本。
- `test_ic_summary.csv`：各测试年度IC与Rank IC。

2022—2024已经用于研究方案比较，因此这里报告的是滚动样本外研究结果，而不是完全未触碰的最终测试结果。
