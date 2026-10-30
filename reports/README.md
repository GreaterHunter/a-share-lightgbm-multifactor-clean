# 报告导航

仅发布小体积、脱敏研究汇总。原始数据、模型、逐股票账本与本机模拟记录不分发。

| 内容 | 入口 | 口径 |
|---|---|---|
| 项目全貌 | [成果](../docs/achievements.md) | 已有成果与边界 |
| 早期Top20 S6 | [报告](current_strategy/README.md) | 2022—2026-10-13五起点；目录名保留，不代表最新模拟规则 |
| 全收益归因 | [H00300报告](current_strategy/total_return/README.md) | 同一S6账户，rf=1% |
| 排名缓冲 | [配置表](research/rank_buffer/ranking.csv)、[逐年表](research/rank_buffer/yearly.csv) | 2021—2025与2026单独回放 |
| 行业规模归因 | [说明](research/barra/README.md) | 残差不等于Alpha |
| 风险约束 | [对照](research/risk_overlay/comparison.csv) | 各费用及年份均保留 |
| 经典动量 | [结果](research/momentum/comparison.csv) | 2022—2025，同一研究费用 |
| 小资金 | [1万元实验](research/capital/comparison.csv) | 整手、固定费用及严重亏损也披露 |
| 分钟消融 | [结论](research/minute_ablation/findings.html)、[28组](research/minute_ablation/comparison.csv) | 2022—2025，分钟执行，TP10／SL4，非V10 |
| 早期研究 | [实验索引](experiments/README.md)、[Top50](main_strategy/README.md) | 历史记录，不是当前部署成绩 |

HTML下载后用浏览器打开；CSV、Markdown和图片可在GitHub查看。公开evidence.json记录相对来源及哈希，原协议与完整账本本地保留。[清理规则](../docs/artifact_retention.md)。


> 产物生命周期更新：2026-10-30按用户要求进一步精简：59个本地历史实验原目录已永久删除，仅保留300张结果表、主要图表、方法与结论摘要。旧分钟行情副本、特征、预测、逐笔账户、任务缓存及上一轮两个MLflow压缩包不再保留；原有哈希只用于识别当时来源，不代表原件仍可取得。历史结果不再支持完整逐笔复算或原缓存接续，重新研究须重建数据与产物。当前部署模型、运行数据、活动V10协议、交易状态及源码保留。
