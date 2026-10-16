# 主模型诊断

本目录诊断年度滚动 `ranked_enhanced` 模型。模型本身保持不变，只比较组合构建与稳健性。

## 分层 Top 50

- 最终持有 50 只股票；
- 每次调仓最多替换 5 只；
- 排名前 10 的股票合计目标权重 40%；
- 第 11—50 名合计目标权重 60%；
- 账户总投资比例仍由 `risk_degree=0.95` 控制；
- 比较 3、5、10 个交易日调仓；
- 额外检验 5 日调仓、买卖成本同时翻倍的压力情景。

调仓周期比较属于敏感性分析，不能因为某一周期在已经查看过的测试期表现最好，就把它解释成新发现的最优参数。正式基准仍采用与 5 日预测标签一致的 5 日调仓。

## 预测诊断

- `prediction_yearly_ic.csv`：逐年 IC 与 Rank IC；
- `prediction_decile_returns.csv`：按预测分数分成十组后的未来 5 日平均收益；
- `feature_importance_by_year.csv`：三个年度模型的特征重要性；
- `feature_importance_stability.csv`：跨年度重要性汇总；
- `feature_rank_correlation.csv`：年度重要性排名的 Spearman 相关系数。

`tiered_portfolio_daily.csv` 是可复算净值和回撤的日频明细，文件较大；其他 CSV 是用于报告和面试展示的汇总结果。
