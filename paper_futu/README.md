# 富途模拟交易适配器

本目录包含离线研究、账户回放与富途模拟交易适配代码。公开仓库不展示个人账户尾号、持仓数量、订单价格或具体切换日期；实时状态以本机已授权配置和回执为准。

## 先明确运行范围

- 当前研究版本和执行限制见 [S6部署说明](S6_CURRENT_DEPLOYMENT.md)。
- 账户、资产、持仓、下单和撤单仅允许明确绑定的模拟环境；不授权实盘。
- 本机配置、模型、计划、交易日志不随仓库发布，不能克隆后直接启动交易。
- 源码保留账户匹配等安全检查，不能为方便演示而移除。

## 只读连接检查

账号本人安装并登录官方OpenD，监听本机回环地址，不把交易接口暴露到公网。
在已准备富途SDK的独立环境中执行：

```powershell
python -m paper_futu.check_connection --market CN
```

其他市场可使用对应参数，但A股模型不能直接用于港股或美股。此检查会查询模拟账户信息；日志仅保存在本地，分享故障时需脱敏。连接成功不代表下单成功或策略已经部署。

## 状态与停止

Windows任务负责启动本机工作进程，不需要聊天软件保持打开。电脑必须开机、联网、不休眠，Windows用户已登录，OpenD在线。锁屏与关闭显示器不等于关机或休眠。

```powershell
Get-Content .\data\interim\futu_cn_paper\worker_status.json
python -m paper_futu.cn_worker --stop
```

停止工作进程不会清仓；恢复前应核对停止标记、未完成订单和实际持仓，不要另起重复进程。不要为了运行示例安装交易调度。

## 执行边界

日线回测是理想开盘成交；模拟程序在允许的连续交易时段提交限价单。价格、整手约束、停牌、涨跌停和资金不足会影响成交，持仓可能少于目标数量。

只交易仓位差，检查可卖数量及T+1；已有提交日志不得盲目重放。部分成交、撤单或异常应以券商回执核对，不能把预览当成成交。

`smoke_order` 是受限的模拟接口测试，不是策略收益验证。未经单独授权，不运行提交订单选项。历史测试记录和补仓细节不在公开说明中复述。

## 参考文档

[OpenD说明](https://openapi.futunn.com/futu-api-doc/quick/opend-base.html) · [账号与权限](https://openapi.futunn.com/futu-api-doc/intro/authority.html) · [模拟交易说明](https://openapi.futunn.com/futu-api-doc/qa/trade.html)
