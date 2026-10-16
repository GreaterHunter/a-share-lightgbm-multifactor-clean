# 安装与运行指南

从仓库根目录执行命令。研究环境以Windows、Python 3.11为主；模拟交易调度依赖Windows。其他系统可运行合成数据示例，尚不承诺完整调度兼容。

## 1 获取代码

```powershell
git clone https://github.com/GreaterHunter/a-share-lightgbm-multifactor-clean.git
cd a-share-lightgbm-multifactor-clean
python -m venv .venv
.venv\Scripts\Activate.ps1
```

如果PowerShell不允许激活脚本，可直接使用 `.venv\Scripts\python.exe` 替代下面的 `python`，不必修改系统执行策略。已有研究环境不要为运行示例盲目升级依赖。

Windows下建议项目与模型输出目录使用纯英文路径。当前LightGBM原生模型保存接口在含中文路径时可能报无法写入；这与Python本身能否读取该目录不是一回事。

## 2 无行情也能运行的示例

```powershell
python -m pip install -r requirements-demo.txt
python -m examples.offline_demo
```

输出包含训练截止日期、最新训练标签结束日期、样本数与合成数据Rank IC。检查 `max_train_label_end <= train_cutoff`。数据使用虚构股票和工作日日历，不代表A股交易日历；只有3个演示特征，不复刻15因子策略、收益或账户回测。

示例不请求网络、不接入OpenD、不保存模型，也不提交订单。随机合成价格没有经济意义，不应追求示例IC为正。

## 3 开发环境与测试

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
```

测试主要使用临时文件、合成数据和模拟接口，覆盖标签边界、时点过滤、模型重载、账户成本、调度互斥和部署保护。依赖中直接使用的psutil、scipy、matplotlib也显式列出，不依赖偶然的传递安装。

2026-10-16展示副本验证：269项通过、2项跳过，合成数据示例运行成功。
跳过项分别需要未公开的本机模拟交易日志和冻结财务快照，不以空数据冒充验证成功。

CI只运行离线示例和测试；不下载真实数据、不运行大型研究、不加载交易凭据。GitHub Actions的实际状态以仓库Actions页为准，不能用本地通过替代远端结果。

默认临时目录遇到权限错误时，为本次测试指定一个**新建且独占**的目录：

```powershell
$testTemp = Join-Path $env:TEMP ('quant-tests-' + [guid]::NewGuid().ToString('N'))
python -m pytest -q --basetemp $testTemp
```

pytest可能清空指定的basetemp，不能填数据目录、项目根目录或其他已有重要目录。
如果系统临时目录本身包含非ASCII字符，请改为一个可写的英文目录下的新子目录。

## 4 完整历史研究不是一条命令的下载即跑

先阅读[数据契约](data_contract.md)，准备获授权的数据、冻结快照、父实验和模型产物。下列是已有本地研究环境的入口，不是新克隆后的快速开始：

| 任务 | 入口 | 前提 |
|---|---|---|
| 查看季度实验状态 | `python -m paper_futu.research_s6_quarterly_regularized --stage status` | 依赖已安装；缺少状态文件会显示not_started |
| 季度联合约束完整研究 | `python -m paper_futu.research_s6_quarterly_regularized --stage all` | 父实验、快照及哈希绑定齐全 |
| 四策略风险研究 | `python -m paper_futu.research_overfit_four --stage all` | 原版预测、账户和数据快照齐全 |
| 旧年度Top50研究 | `python -m src.run_main_pipeline --legacy --stage all` | 旧研究配置与数据已准备 |

`python main.py` 指向旧研究入口；在本机S6已登记时会阻止无提示地跑旧年度模型。公开仓库不包含该本机登记，因此不要把这个命令作为S6的一键训练入口；新克隆请先运行离线示例。

研究代码检测源文件和产物哈希；缺数据、源变化或缓存不兼容时，应建立新版本实验并审计，不能删除校验强行续跑。重现历史数值需要原始快照，换成今天下载的数据即是另一次实验。

## 5 模拟交易另行配置

仅在本机已验证的环境中安装 `paper_futu/requirements.txt` 中的富途依赖，并按[部署说明](../paper_futu/S6_CURRENT_DEPLOYMENT.md)核查。
本机配置、模型、账户状态与Windows任务不随GitHub发布；不能靠复制示例配置或删除账户尾号限制来完成部署。

本项目没有授权实盘，测试通过也不等于可以无人值守交易。下一季度的新模型、数据新鲜度、OpenD连接和交易回执都需要检查。
