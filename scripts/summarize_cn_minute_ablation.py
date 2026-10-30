"""Interpret audited ablations without changing the sealed experiment or models."""
from pathlib import Path
import hashlib
import json
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/experiments/cn_minute_factor_ablation_asof_20251231'
NAMES = {'M0_daily':'M0 日频15因子','M1_clock':'M1 日频＋时段进度',
         'M2_return':'M2 M1＋分钟收益','M3_volatility':'M3 M1＋分钟波动',
         'M4_position':'M4 M1＋量价位置','M5_full':'M5 完整29因子'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    complete = json.loads((OUT/'completion.json').read_text(encoding='utf8'))
    if complete['status'] != 'complete_and_audited':
        raise RuntimeError('Wait for completed independent audits')
    for path, expected in complete['artifacts'].items():
        if sha(path) != expected:
            raise RuntimeError('Audited output changed: '+path)
    s = pd.read_csv(OUT/'comparison.csv')
    y = pd.read_csv(OUT/'yearly.csv')
    ic = pd.read_csv(OUT/'rank_ic_daily.csv').drop(columns='date').mean()
    main_rows = s[s.cadence.eq('5m') & s.lag.eq(2)]
    base = main_rows[main_rows.bp.eq(0)].set_index('model').reindex(NAMES)
    stress = main_rows[main_rows.bp.eq(10)].set_index('model').reindex(NAMES)
    table = pd.DataFrame({
        '输入':pd.Series(NAMES),
        '净年化':base.cagr.map(lambda v:f'{v:.2%}'),
        '最大回撤':base.max_drawdown.map(lambda v:f'{v:.2%}'),
        '全期费用/万元':base.fees.map(lambda v:f'{v/10000:.2f}'),
        '完整交易周期':base.closed.astype(int),
        '每边额外10bp净年化':stress.cagr.map(lambda v:f'{v:.2%}'),
        '每日平均Rank IC':ic.reindex(NAMES).map(lambda v:f'{v:.5f}'),
    })
    annual = y[y.cadence.eq('5m') & y.lag.eq(2) & y.bp.eq(0)].pivot(index='model',columns='year',values='total_return').reindex(NAMES)
    annual.index = annual.index.map(NAMES)
    # Inspect actual tree splits, not just a rounded importance display.
    splits=[]
    for path in sorted((OUT/'models/M1_clock').glob('*/model.txt')):
        b=lgb.Booster(model_file=str(path));i=b.feature_name().index('min_session_progress')
        splits.append({'quarter':path.parent.name,'model_sha256':sha(path),
                       'clock_splits':int(b.feature_importance('split')[i])})
    if len(splits)!=16 or any(row['clock_splits'] for row in splits):
        raise RuntimeError('Revisit interpretation of the clock control')
    (OUT/'clock_split_evidence.json').write_text(json.dumps(splits,indent=2),encoding='utf8')
    table.to_csv(OUT/'findings_main.csv',index=False,encoding='utf-8-sig')
    intervals=pd.read_csv(OUT/'paired_block_intervals.csv')
    ic_intervals=intervals[intervals.metric.eq('daily_rank_ic_difference')]
    if not ((ic_intervals.lower<=0)&(ic_intervals.upper>=0)).all():
        raise RuntimeError('Revisit IC uncertainty statement')
    diagnostic=s[s.model.eq('M5_full')][['cadence','lag','bp','cagr','max_drawdown','closed','fees']].copy()
    diagnostic['cadence']=diagnostic.cadence.map({'5m':'每5分钟','30m':'每30分钟','daily':'每日09:45'})
    diagnostic=diagnostic.rename(columns={'cadence':'排名检查','lag':'执行滞后线数','bp':'每边额外bp','cagr':'净年化','max_drawdown':'最大回撤','closed':'完整交易周期','fees':'全期费用/元'})
    for col in ('净年化','最大回撤'):
        diagnostic[col]=diagnostic[col].map(lambda v:f'{v:.2%}')
    diagnostic['全期费用/元']=diagnostic['全期费用/元'].map(lambda v:f'{v:,.2f}')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>分钟因子消融：结论与证据</title>
<style>body{font:16px/1.75 system-ui;max-width:1350px;margin:35px auto;padding:0 20px;color:#172738}h1,h2{line-height:1.4}table{border-collapse:collapse;font-size:14px;width:100%}th,td{padding:8px;border:1px solid #dce3eb;text-align:right}th{background:#edf3f8}img{max-width:100%}.notice{padding:18px;background:#fff3df;border-left:4px solid #ce9b48}.table{overflow:auto}</style>
<h1>分钟因子有效性消融：目前没有稳定增益证据</h1>
<p>2022—2025，初始100万元。季度训练、10交易日收益排名目标；固定Top10、排名跌出80退出、新进第1名替换、止盈10%、含费止损4%，保留T+1；止盈无额外冷却，止损下一交易日可重入。所有主组每5分钟检查，成交滞后至少一根完整5分钟线。</p>
<div class="notice"><b>结论：</b>在当前固定训练设置和交易规则下，完整分钟输入未优于日频对照。分组输入的结果不一致；所有预设Rank IC差异区间均包含0。不能据此宣称分钟因子普遍无效，也没有依据将完整分钟组自动升级为主策略。</div>
<h2>1. 六组输入主对照</h2>'''+table.to_html(index=False)+'''
<p>M2包括5/15/60分钟收益及相对510300的收益；M3包括15/60分钟波动和振幅；M4包括量比、VWAP偏离及开盘以来收益。M2—M4均包含M1。成本列均为扣费后的结果；额外10bp是每边加收并重新回放，交易路径也可能随之改变。</p>
<h2>2. 逐年收益：优势并不稳定</h2>'''+annual.to_html(float_format=lambda v:f'{v:.2%}')+'''
<p>完整分钟组2025年收益高于纯日频组，但2023年明显落后；2024年收益类输入较好。这支持进一步关注时间稳定性，而不支持用某一年或全期冠军证明因子有效。</p>
<h2>3. 时段组的收益不能解释为时段Alpha</h2>
<p>核对全部16个M1季度模型后，时段进度列的实际分裂次数均为0。M1虽有较高历史年化，但最终树并未使用时段列；日内Top10也没有变化。当前colsample_bytree=0.8，添加列会改变随机特征子采样及训练出的树结构。因此，这更应视为模型实现敏感性的证据，不能将收益提升归因于时段信息。尚未用多种随机种子或关闭特征子采样做专门归因，本轮不额外扩大训练。</p>
<h2>4. 预测增益与不确定性</h2>
<p>按日期聚合的平均Rank IC：日频约0.07792，完整分钟约0.07745；收益类约0.07858。完整分钟相对日频差异约−0.00047，120日区块95%区间约[−0.00201，0.00130]；收益类相对时段对照约+0.00099，区间约[−0.00107，0.00267]。60日和120日区块的全部预设IC对照均跨0，尚无法确认稳定增益。</p>
<p>完整分钟相对日频的基础成本净年化差约−0.81个百分点，120日区块区间约[−10.28，7.17]个百分点。区间很宽；不能断言真实效果一定为负。分钟组全期完整交易1225次，日频966次，约多26.8%；费用由18.69万元增加至23.42万元。预测能力没有明显增加，而交易摩擦增加，是当前主要问题之一。</p>
<h2>5. 完整分钟组的排名频率与延迟检查</h2>'''+diagnostic.to_html(index=False)+'''
<p>30分钟或每日仅降低排名退出与强势替换频率，止盈止损和空位补仓仍每5分钟判断。lag2为原执行规则；lag3额外延迟5分钟。较慢执行并未使收益单调恶化，但收益变化明显，说明成交时点和路径需要前瞻验证。不能事后挑最好延迟或检查频率作为已验证结论。</p>
<h2>6. 使用边界与交付</h2>
<p>64次新拟合、32个模型复用、28个账户、36组配对区间均已完成。模型重载、标签边界、源文件哈希、现金、费用、净值、T+1和原止损4%基准复现已核验。当前V10及模拟盘未修改，没有自动提交Git。</p>
<p>所有输入使用相同成熟样本、年度树数和其他参数，不重新搜索各组最佳容量；每日4个时点训练，全部可用5分钟时点评价。仅单一模型种子，历史已反复使用；区块区间不能抵消历史选择偏差。分钟数据未经交易所认证，财务修订档案不完整，复权持仓为合成处理；成交量参与率、涨跌停及价差仍有简化，不能称实盘可实现收益。</p>
<img src="curves.png" alt="六组模型的净值与回撤"><p><a href="report.html">完整实验报告与28组对照</a> · <a href="comparison.csv">全部账户表</a> · <a href="yearly.csv">全部逐年结果</a> · <a href="paired_block_intervals.csv">全部差异区间</a> · <a href="clock_split_evidence.json">时段列分裂核验</a> · <a href="independent_audit.json">账户审计</a></p></html>'''
    (OUT/'findings.html').write_text(html,encoding='utf8')
    paths=[OUT/'findings.html',OUT/'findings_main.csv',OUT/'clock_split_evidence.json']
    manifest={'source_completion_sha256':sha(OUT/'completion.json'),'script_sha256':sha(__file__),
              'outputs':{str(p):sha(p) for p in paths},'scientific_results_unchanged':True}
    (OUT/'findings_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=True),encoding='utf8')
    print(table.to_string(index=False))


if __name__=='__main__':
    main()
