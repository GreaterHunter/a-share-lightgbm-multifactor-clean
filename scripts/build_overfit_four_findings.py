"""Reader-facing interpretation of the audited study; no experiment is rerun."""
from __future__ import annotations

from html import escape
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/experiments/overfit_four_asof_20261013"
VARIANTS = {
    "V0": "原版扩展窗口", "V1": "近3年", "V2": "近5年", "V3": "近10年",
    "V4": "近期样本加权", "V5": "增强L1", "V6": "增强L2", "V7": "L1+L2", "V8": "降低树复杂度",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(frame, names, percent=(), cash=(), decimals=()):
    display = frame[list(names)].copy()
    def formatted(value, pattern):
        return "不可计算" if pd.isna(value) else pattern.format(value)
    for column in percent:
        display[column] = display[column].map(lambda value: formatted(value, "{:.2%}"))
    for column in cash:
        display[column] = display[column].map(lambda value: formatted(value / 10000, "{:.2f}万"))
    for column in decimals:
        display[column] = display[column].map(lambda value: formatted(value, "{:.2f}"))
    return '<div class="scroll">' + display.rename(columns=names).to_html(index=False, border=0, escape=True) + "</div>"


def main():
    verification = json.loads((OUT / "final_verification.json").read_text(encoding="utf-8"))
    original = json.loads((OUT / "report_manifest.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed"
    assert verification["original_report_manifest_sha256"] == digest(OUT / "report_manifest.json")
    for path, expected in original["results"].items():
        assert digest(path) == expected, path
    s = pd.read_csv(OUT / "strategy_summary.csv")
    ic = pd.read_csv(OUT / "signal_ic_summary.csv")
    general = pd.read_csv(OUT / "generalization.csv")
    boot = pd.read_csv(OUT / "paired_block_bootstrap.csv")
    main_rows = s.loc[s.seed.eq(42) & s.friction.eq(0)]
    overall = main_rows.loc[main_rows.period.eq("all")]
    baseline = overall.loc[overall.variant.eq("V0")]
    assert len(overall) == 36 and len(baseline) == 4 and len(boot) == 96
    parts = ["""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>四策略过拟合与风格依赖研究结论</title>
<style>body{font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;color:#183047;max-width:1220px;margin:36px auto;padding:0 22px}h1{font-size:28px;line-height:1.4}h2{font-size:22px;margin-top:38px;border-bottom:1px solid #d5e0e6;padding-bottom:8px}h3{font-size:18px;margin-top:24px}.lead{padding:18px 22px;background:#edf3f6;border-left:4px solid #315c78}.note{font-size:14px;color:#526575}.scroll{overflow-x:auto;margin:14px 0}table{border-collapse:collapse;width:100%;font-size:14px;white-space:nowrap}th,td{padding:8px 10px;text-align:right;border-bottom:1px solid #e1e8ec}th{background:#24475e;color:white}td:first-child,th:first-child{text-align:left}tbody tr:nth-child(even){background:#f7f9fa}details{margin:18px 0}summary{cursor:pointer;font-weight:600}a{color:#205e8d}code{background:#eef2f4;padding:2px 5px}li{margin:7px 0}@media print{body{max-width:none;margin:10px}details{display:block}th{color:#183047;background:#eee}.scroll{overflow:visible}table{font-size:10px;white-space:normal}}</style></head><body>
<h1>S2、S4、S5、S6：过拟合与风格依赖研究</h1>
<p>固定快照截至2026-10-13；回放从2022年第一个交易日至2026-10-13。2026为不完整年度。全部结果均为既有历史研究诊断，不是新的独立测试。</p>
<p class="lead"><strong>结论：</strong>不能认定这些策略“已排除过拟合”。S4/S5的银行、估值和低波动依赖很明显；S2/S6相对分散、对随机种子更稳定，但仍有风格与2024年度依赖。缩短训练窗口没有解决问题，部分版本反而出现更大的训练—执行差距。增加L1/L2没有带来一致改善；降低复杂度对S2/S6有积极迹象，但证据不足以自动替换原版。</p>
<h2>一、固定配置与统计口径</h2>
<ul><li>S2：原始15因子，10日预测/5日调仓，Top20等权，最多替换3只；S6共用同一预测，只将最多替换改为6只。</li>
<li>S4：排名15因子，20日预测/10日调仓，Top10、前2名合计占股票仓位50%，最多替换6只；S5共用预测，前2名合计占股票仓位60%。总股票目标仓位95%，不是头部股票占整个账户50%/60%。</li>
<li>标签为 Open(T+H+1)/Open(T+1)−1，目标为该收益的当日横截面排名减0.5。年度从头训练，沿用原固定/年度参数方案，变体只重新选择树数。</li>
<li>R5预设起点0/1/2/3/4，R10为0/2/4/6/8；跨年不重置。平均年化是五个账户年化的算术平均，回撤取五起点中最差值。这五个账户高度相关，不是五份独立样本。</li>
<li>账户初始100万元，买入0.05%、卖出0.15%、每笔最低5元，只交易仓位差。费用是各起点全期人民币金额的均值，不是固定资金费率。</li></ul>
<h2>二、全部V0—V8结果</h2><p>每格依次为：成本后平均年化 / 最差起点最大回撤（绝对值）。历史回撤30%仅作风险标记，不用它判定是否过拟合。</p>"""]
    matrix = []
    for variant, label in VARIANTS.items():
        row = {"设置": f"{variant} {label}"}
        for strategy in ["S2", "S4", "S5", "S6"]:
            value = overall.loc[overall.strategy.eq(strategy) & overall.variant.eq(variant)].iloc[0]
            row[strategy] = f"{value.annualized_return:.2%} / {-value.worst_phase_drawdown:.2%}" + (" *" if value.worst_phase_drawdown < -.3 else "")
        matrix.append(row)
    parts.append(table(pd.DataFrame(matrix), {k: k for k in matrix[0]}))
    parts.append("""<p class="note">* 至少一个预设起点回撤超过30%。V4半衰期756个交易日且训练权重均值归一化为1；V5的L1取max(1,原值×10)，V6的L2乘5；V8叶子数减半、叶子最小样本数乘3。没有做全排列。</p>
<p>近5年窗口下，S4/S5年化降至1.43%/0.92%，最差回撤约51%；并不是删去老数据就更能适应市场。V8下S2/S6年化分别为18.88%/20.18%，但这里只保留为诊断结果，不据此选新冠军。</p>
<h2>三、收益是否只来自少数年份</h2><p>以下是V0的自然年净收益（2026为截至10月13日累计），不是当年年化；均为五个起点的均值。</p>""")
    yearly = main_rows.loc[main_rows.variant.eq("V0") & main_rows.period.isin(["2022", "2023", "2024", "2025", "2026"])]
    year_view = yearly.pivot(index="period", columns="strategy", values="total_return").reset_index()
    year_view["benchmark"] = year_view.period.map(yearly.loc[yearly.strategy.eq("S2")].set_index("period").benchmark_total_return)
    parts.append(table(year_view, dict(period="年份", S2="S2", S4="S4", S5="S5", S6="S6", benchmark="沪深300"), percent=["S2", "S4", "S5", "S6", "benchmark"]))
    dependence = baseline[["strategy", "annualized_return", "log_growth_share_2024"]].copy()
    for field, period in [("exclude2024", "excluding_2024_diagnostic"), ("exclude2022and2024", "excluding_2022_2024_diagnostic")]:
        values = main_rows.loc[main_rows.variant.eq("V0") & main_rows.period.eq(period)].set_index("strategy").annualized_return
        dependence[field] = dependence.strategy.map(values)
    parts.append(table(dependence, dict(strategy="策略", annualized_return="原版平均年化", log_growth_share_2024="2024对数增长占比", exclude2024="剔除2024的年化诊断", exclude2022and2024="剔除2022及2024的年化诊断"), percent=["annualized_return", "log_growth_share_2024", "exclude2024", "exclude2022and2024"]))
    parts.append("""<p>2024约占各原版账户全期对数增长的58%—60%，确有集中。但剔除2024后的诊断年化仍约8%—9%，再剔除2022后约6%—12%，不是收益完全消失。S2/S6在2022分别仅赚约4.00%/1.34%，不能把此前单股重仓版本的“2022和2024驱动”结论直接照搬到这四个策略。</p><p class="note">剔除年份后拼接的是非连续日收益，只用于年度依赖诊断，不是新的一条可实施连续回测。对数增长占比不是人民币利润贡献占比。逐年几何超额见下方明细。</p>
<h2>四、银行、估值与低波动暴露</h2>""")
    exposure = baseline.copy()
    exposure["max_drawdown_abs"] = -exposure.worst_phase_drawdown
    parts.append(table(exposure, dict(strategy="策略", bank_weight="银行占净值均值", bank_profit_ratio="银行净利润贡献占比均值", book_to_price="B/P平均分位", earnings_to_price="E/P平均分位", volatility_60d="60日波动平均分位", maximum_stock_weight="观察到的单股最高权重", total_fees_cny="全期平均费用"), percent=["bank_weight", "bank_profit_ratio", "book_to_price", "earnings_to_price", "volatility_60d", "maximum_stock_weight"], cash=["total_fees_cny"]))
    parts.append("""<p>S4/S5平均约三分之二账户净值在银行，约89%的净利润来自银行；2024及2025银行平均权重进一步达到约88%—90%。Top10并没有实现行业分散。较高B/P、E/P分位和较低波动分位也符合明显的估值/低波动偏好。行业盈利贡献不等于行业独立Alpha，不能单凭它区分行业贝塔、风格与个股能力。</p>
<p>S2/S6银行净值权重约41%/38%，银行利润贡献约52%/38%，相对分散，但仍非风格中性。以下用相同完整数据子集比较调整前排名相关与控制银行和五项风格后的残差相关。</p>""")
    signals = ic.loc[ic.seed.eq(42) & ic.period_kind.eq("all")].copy()
    base_signal = signals.loc[signals.variant.eq("V0")]
    parts.append(table(base_signal, dict(family="预测流", mean_style_subset_rank_ic="同一子集调整前Rank IC", mean_style_residual_rank_ic="控制风格后的残差相关", mean_bank_rank_ic="银行内Rank IC", mean_nonbank_rank_ic="非银行内Rank IC", style_coverage="每日样本覆盖率均值"), percent=["mean_style_subset_rank_ic", "mean_style_residual_rank_ic", "mean_bank_rank_ic", "mean_nonbank_rank_ic", "style_coverage"]))
    parts.append("""<p>raw10从8.34%降到2.99%，rank20从8.52%降到2.63%，关联明显减弱。两个非银行组内Rank IC仍为正，说明“全部只是买银行”也过于绝对。控制仅包括银行标记、B/P、E/P、60日波动、20/60日动量及截距；这不是完整行业中性，残差相关也不能直接称为可交易Alpha。</p>
<h2>五、训练成绩、执行期与2025年变化</h2>""")
    train = general.loc[general.seed.eq(42) & general["sample"].eq("training_tail_252")].groupby(["family", "variant"]).mean_rank_ic.mean()
    valid = general.loc[general.seed.eq(42) & general["sample"].str.startswith("validation_")].groupby(["family", "variant"]).mean_rank_ic.mean()
    training_view = signals.set_index(["family", "variant"])[["mean_rank_ic", "prediction_rank_correlation", "topk_overlap"]].join(train.rename("train_tail_rank_ic")).join(valid.rename("selected_validation_rank_ic")).reset_index()
    parts.append(table(training_view, dict(family="预测流", variant="设置", train_tail_rank_ic="训练尾252日Rank IC均值", selected_validation_rank_ic="所选树数验证折均值", mean_rank_ic="历史执行期Rank IC", prediction_rank_correlation="与V0预测排名相关", topk_overlap="与V0 TopK重合"), percent=["train_tail_rank_ic", "selected_validation_rank_ic", "mean_rank_ic", "prediction_rank_correlation", "topk_overlap"]))
    parts.append("""<p>训练尾段和验证均值按执行年汇总；验证年在不同执行年的三折中会重复，不能视作独立样本数。训练尾段与执行期也不是同一市场时期，因此差距只是诊断，不是精确过拟合量。</p>
<p>raw10的3年窗口训练Rank IC约29.12%，执行期仅5.10%；原版分别15.47%和8.41%。短窗口训练更好、执行更弱，符合样本变少后的过度拟合迹象，但样本减少及市场关系变化也可能共同作用。V8使raw10训练指标降到12.52%、执行指标约8.50%，支持降低复杂度的方向，但不能单凭此排除历史选择偏差。</p>
<p>2025年raw10原版Rank IC约0.83%，rank20约−4.33%；后者风格调整后残差相关约0.32%。这支持市场关系/风格适用性发生变化，而不是仅仅因为原始收益MSE变差。该年四个原版策略仍有正收益，但都跑输沪深300。</p>
<h2>六、随机种子及交易摩擦</h2><p>以下只改变最终拟合种子，参数及树数保持seed42选定值。没有挑表现最好的种子。V2/V7同时包含规则变化，与V0的重合率不能解读为纯种子效应。</p>""")
    seed_view = s.loc[s.period.eq("all") & s.friction.eq(0) & s.variant.isin(["V0", "V2", "V7"])].copy()
    seed_view["drawdown_abs"] = -seed_view.worst_phase_drawdown
    parts.append(table(seed_view, dict(strategy="策略", variant="设置", seed="种子", annualized_return="平均年化", drawdown_abs="最差起点回撤"), percent=["annualized_return", "drawdown_abs"]))
    parts.append("""<p>原版S2/S6的三个种子年化约16.30%—16.93%及17.05%—17.43%，相对稳定；S4约14.64%—16.97%，S5约15.39%—17.51%。S5在seed2025下最差回撤达到31.80%，超过30%风险标记。原版额外种子与seed42的TopK重合率，raw10约84%，rank20约70%，头部集中使排名变动更容易转化为收益差异。</p>
<p>摩擦是在原有买卖成本之外每边增加10bp/20bp，重新回放账户，不是从年化中扣一个常数：</p>""")
    friction = s.loc[s.period.eq("all") & s.variant.eq("V0") & s.seed.eq(42)].copy()
    friction["added"] = friction.friction.map({0.: "原成本", .001: "每边+10bp", .002: "每边+20bp"})
    friction["drawdown_abs"] = -friction.worst_phase_drawdown
    parts.append(table(friction, dict(strategy="策略", added="成本设置", annualized_return="平均年化", drawdown_abs="最差起点回撤", total_fees_cny="全期平均费用"), percent=["annualized_return", "drawdown_abs"], cash=["total_fees_cny"]))
    parts.append("""<p>原成本下S6比S2多约0.49个百分点年化，但费用约18.38万元对10.00万元；每边再加20bp后S6年化约10.98%，低于S2的13.53%。较快换仓的优势对摩擦更敏感，不能只比较原成本收益。日线理想开盘成交仍不能替代实际09:31成交验证。</p>
<h2>七、配对区块重采样与不确定性</h2><p>固定种子20261015，2,000次，同一次对照双方及所有起点使用相同日期区块。60/120交易日两种区块长度全部保留；区间是给定历史的条件性不确定性，不是过拟合概率。</p>""")
    chosen = boot.loc[boot.kind.eq("mean_phase_cagr") & boot.variant.isin(["V6", "V7", "V8"])].copy()
    for column in ["point_delta", "ci_2_5", "ci_97_5"]:
        chosen[column] = chosen[column] * 100
    parts.append(table(chosen, dict(strategy="策略", variant="设置", block="区块天数", point_delta="年化差值（百分点）", ci_2_5="95%区间下限（百分点）", ci_97_5="95%区间上限（百分点）"), decimals=["point_delta", "ci_2_5", "ci_97_5"]))
    parts.append("""<p>S2的V8年化提高约1.95个百分点，但两种区块区间都包含0；S6的V8提高约2.75个百分点，60日区块区间约−0.05至+6.15个百分点，120日区块约+0.01至+6.05个百分点，结论对区块长度敏感。V8没有被本轮预先指定做额外种子检查，不能临时补挑种子或据此宣布稳健胜出。S2增强L2、S6联合正则化的收益差区间同样包含0。</p>
<h2>八、逐个策略的判断</h2>
<h3>S2：相对分散、种子稳定，但有风格和年份依赖</h3><p>支持风险的证据：约41%银行暴露，2024增长集中，3/5年窗口显著退化。反对“纯偶然重仓银行”解释的证据：非银行组内Rank IC仍正，原版种子较稳定，剔除2024后并非无收益，每边20bp压力下仍正。尚不能判断：剩余相关能否在完整行业/风格控制和未来独立数据上成立。</p>
<h3>S4：银行/估值依赖是比单纯复杂树更明显的问题</h3><p>银行约贡献89%净利润，2024/2025银行权重接近九成，2025整体Rank IC转负。简单加强L1/L2或降低复杂度均未一致提升成本后收益；近5年窗口严重退化。没有证据说明“加正则就修好了”，也不能因原版优于变体而宣布无过拟合。</p>
<h3>S5：提高头部权重的增益有限，随机性风险更大</h3><p>与S4共用预测，不能把它当作另一份独立模型证据。年化仅提高约0.54个百分点，银行依赖相近，原种子最差回撤更高；额外种子有31.80%回撤。这支持头部权重放大暴露和尾部不确定性的判断，不支持低历史回撤就等于低未来风险。</p>
<h3>S6：可疑点较少，但更依赖执行成本</h3><p>银行暴露和银行利润占比较S2低，种子较稳定；V8相对原版在五个执行年份的平均收益差均为正，值得保留为复杂度敏感性的积极证据。但V8的区块区间敏感且未做额外种子，原版较高换手在成本压力下明显吃亏。不能自动选它替换S2，更不切换模拟盘。</p>
<h2>九、停止条件与下一步</h2><p>本轮已完成预先确认的范围，不再扩大历史网格，不自动选新冠军。S2/S6与S4/S5各自共用预测，因此不是四个独立模型。下一步应冻结待验证规则，用研究结束后未观察的数据做前瞻模拟，记录真实成交、交易摩擦和行业暴露；不把已反复看过的2022—2026重新命名为独立测试。</p>
<p>接受性核验：136验证拟合、140最终拟合，30预测流，340账户，96个配对区间。140个新最终模型逐个重载、每个最多抽128个预测验证一致；340账户净值、现金与费用复算，比较区间与基准一致，训练/验证成熟边界及冻结快照/交易保护文件核验通过。原版模型在前置审计中重载核对。未重新拟合任何模型。</p>
<p class="note">局限：现有银行标记不是完整历史行业体系；财务值没有完整历史修订档案；统一9.5%开盘涨跌幅过滤，未逐板块/ST重建精确限制；复权股价与调整数量近似公司行动，未建立逐笔现金分红和配股台账；不能据此证明盘口容量或真实成交已验证。</p>
<h2>十、明细与可复算依据</h2><ul>""")
    sources = ["strategy_summary.csv", "signal_ic_summary.csv", "generalization.csv", "paired_block_bootstrap.csv", "account_metrics_exposure.csv", "paired_strategy_differences.csv", "final_verification.json", "protocol.json", "report_manifest.json"]
    for name in sources:
        parts.append(f'<li><a href="{escape(name)}">{escape(name)}</a></li>')
    parts.append("</ul><details><summary>全部逐年几何超额、回撤、费用、换手及实际持仓</summary>")
    annual_detail = main_rows.loc[main_rows.period.isin(["2022", "2023", "2024", "2025", "2026"])].copy()
    annual_detail["drawdown_abs"] = -annual_detail.worst_phase_drawdown
    parts.append(table(annual_detail, dict(strategy="策略", variant="设置", period="年份", total_return="净收益", total_excess_return="几何超额", drawdown_abs="最差起点回撤", total_fees_cny="平均费用", average_daily_turnover="平均日双边换手", average_holdings="平均实际持仓数", bank_weight="银行占净值均值"), percent=["total_return", "total_excess_return", "drawdown_abs", "average_daily_turnover", "bank_weight"], cash=["total_fees_cny"], decimals=["average_holdings"]))
    parts.append("</details><details><summary>全部年度及季度IC、分组IC和风格残差</summary>")
    ic_detail = ic.loc[ic.seed.eq(42) & ic.period_kind.ne("all")]
    parts.append(table(ic_detail, dict(family="预测流", variant="设置", period_kind="时期类型", period="时期", valid_ic_days="有效IC日", mean_rank_ic="Rank IC", mean_bank_rank_ic="银行内", mean_nonbank_rank_ic="非银行内", mean_style_residual_rank_ic="风格残差相关", style_coverage="覆盖率"), percent=["mean_rank_ic", "mean_bank_rank_ic", "mean_nonbank_rank_ic", "mean_style_residual_rank_ic", "style_coverage"]))
    parts.append("</details><details><summary>全部96个区块区间</summary>")
    parts.append(boot.fillna({"strategy": "不适用"}).to_html(index=False, border=0))
    parts.append("</details></body></html>")
    destination = OUT / "research_findings.html"
    destination.write_text("".join(parts), encoding="utf-8")
    manifest = dict(status="complete", no_new_experiments=True, no_champion_selected=True,
        original_report_manifest_sha256=digest(OUT / "report_manifest.json"),
        source_hashes={name: digest(OUT / name) for name in sources},
        findings_path=str(destination), findings_sha256=digest(destination),
        builder_sha256=digest(__file__), analysis_runtime="bundled Python 26.1013.11914")
    (OUT / "findings_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(path=str(destination), size=destination.stat().st_size, sha256=digest(destination)), ensure_ascii=True))


if __name__ == "__main__":
    main()
