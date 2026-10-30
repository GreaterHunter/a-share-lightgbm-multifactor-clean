# 日频更新、关机及OpenD退出后的恢复

行情与因子更新不依赖OpenD或Codex。Windows必须开机、用户登录且网络可用；关机期间不能运行。重启后登录Windows，会在约30秒后执行独立恢复任务，最多尝试3次（间隔60秒）。固定08:50及收盘后的多次重试保留。没有后台订单执行入口，恢复历史数据不补发过期交易。

## 数据与账户分离

- 公开数据来自新浪原始日线及带日期的复权因子，并用腾讯逐股票核对OHLC和成交量。HTTPS保留证书校验，新浪连接失败有备用HTTPS入口。
- 本地保存过成员名单的缺失日期按顺序恢复，再更新最新完整交易日；15:35之前不使用当天数据。新增成分股按自身可核验价格锚补历史，单独存储，不覆盖旧300只股票快照。
- 缺少当时成员名单时仅补可核验价格历史并计算最新因子，不伪造过去名单/选股。最多自动补20个全市场交易日，超限、交易日历或季度模型过期会明确阻断并要求检查。
- 原始分红额与除息扣减额不一定相同：回购股份不分红时使用已核验发行人公告中的扣减额。当前登记徐工机械2026-10-21和蓝思科技2026-10-20两项特殊事件，公告PDF及哈希保存；其他未知事项仍停止核查，不放宽因子误差。
- 每日完成标记同时记录行情和因子/预测文件哈希。中途失败不发布新信号；再次运行验证并复用完成文件。

独立只读账户任务在09:20—23:00每10分钟查询一次。OpenD关闭/账号退出时正常等待，不影响公开行情。稍后登录OpenD后，下一轮会补记**当时**现金、净值和实际非零持仓数，不会伪造成断线期间的每日收盘净值。漏失收盘净值仍显示缺口。

交易后台的行情错误会显示但不再阻断账户快照，行情重试至少间隔10分钟；交易规则、模拟账户限制、每5日调仓和09:31—09:51窗口均不变。错过调仓信号或执行窗口时可以补行情/预测，但不会重置调仓节奏、自动追买或改成隔夜下单。跨季度仍需核验新模型。

## 手动命令

```text
python -m paper_futu.update_cn_data --catch-up
python -m paper_futu.recover_cn_session --attempts 3
python -m paper_futu.record_cn_account
```

前两项的公开数据子进程不导入富途SDK。第三项仅观察账户，不下单、不撤单。重新安装恢复任务：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_cn_data_schedule.ps1 -Python D:\Anaconda\envs\quant-lightgbm\pythonw.exe
```

状态文件在`data/interim/futu_cn_paper/`：`independent_data_status.json`、`data_catchup_status.json`、`session_recovery_status.json`、`account_recovery_status.json`。V6是数据/恢复修订，不是新的独立样本外验证；旧协议和已完成交易保留，训练模型及财务因子快照不变。
