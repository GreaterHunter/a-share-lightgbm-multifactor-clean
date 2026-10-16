# 架构与代码导航

研究与模拟执行共享标签和组合约束，但产物、入口及权限分开。为保留科学代码哈希，本次整理不搬动实验实现，也不批量格式化已冻结模块。

```text
历史行情 + 当时股票池 + 已披露财务值
                  ↓
因子面板 + 真实label_end_date
                  ↓
成熟样本过滤 → 历史多折选树 → 季度从零拟合
                  ↓
预测分数 ──→ 离线账户回放 ──→ 起点/费用/风格诊断
                  ↓ 经单独验证与授权
部署模型及哈希 → 收盘信号 → 次日模拟订单 → 回执与账户核对
```

## 阅读顺序

| 层 | 关键文件 | 应掌握的内容 |
|---|---|---|
| 因子 | `src/factor_library.py` | 价量与基本面定义，不同单位的处理 |
| 标签 | `src/forward_labels.py` | H日持有对应H+1交易日结束偏移、成熟样本筛选 |
| 历史选择 | `src/train_walk_forward_cv.py` | 时间折、候选树数、重新拟合 |
| 当前季度研究 | `paper_futu/research_s6_quarterly_regularized.py` | 年内固定参数树数、季度成熟边界、缓存绑定 |
| 父级频率研究 | `paper_futu/research_s6_2026.py` | 年度/半年/季度对照与共享拟合 |
| 账户 | `paper_futu/s6_explicit_account.py` | 现金、订单、整手、持仓及费用核算 |
| 诊断 | `paper_futu/overfit_four_diagnostics.py` | IC、风格残差、时间区块重采样 |
| 模型部署 | `paper_futu/deploy_s6_quarterly.py` | 模型包、哈希、服务季度、明确切换 |
| 交易保护 | `paper_futu/cn_common.py`、`execute_cn_plan.py` | 模拟账户限制、仓位差、不可成交处理 |
| 调度 | `paper_futu/cn_worker.py` | 信号与执行状态机，不依赖对话窗口 |

`paper_futu/` 是历史形成的包名，其中既有研究账户回放，也有券商适配；并不意味着导入所有研究模块都会交易。具体权限以各入口为准，初学者从 `examples/offline_demo.py` 开始。

## 三类配置不能混用

1. `configs/experiments/s6_head6_40_public.json`：公开的研究参数说明，不包含部署绑定。实际本机登记 `s6_head6_40_research_main.json` 包含模型指纹与计划状态，仅在本地保留，不随最新版发布。
2. `configs/main_strategy.yaml`：保留的旧年度Top50流程配置。
3. `configs/futu_cn_paper.json`：本机模拟部署绑定，不上传Git；实际交易还取决于模型元数据、计划和账户状态。

## 产物与并行边界

每个实验使用独立输出目录；协议、源代码、数据、模型与任务产物哈希用于识别兼容缓存。研究调度最多两个工作进程，每进程四线程；可用内存启动门槛与任务内保护不能为追求速度直接取消。账户内部按时间顺序执行，并行仅分配互斥任务。

新功能优先新增独立模块和测试。修改已有科学代码会改变哈希，应先审查缓存兼容性，不能覆盖protocol让旧结果看似由新代码产生。
