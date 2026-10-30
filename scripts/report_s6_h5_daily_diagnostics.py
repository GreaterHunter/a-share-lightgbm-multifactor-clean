"""Post-run realized stop audit; no model, source, or broker mutations."""
import html
import pandas as pd
from paper_futu.research_s6_h5_daily_stop13 import OUT, ROOT, read, save, digest
from scripts.analyze_s6_realized_sales import reconstruct_sales, stock_date_sales


def main():
    summary=read(OUT/"comparison.json")
    assert summary["status"]=="complete"
    orders=pd.read_parquet(OUT/"account/orders.parquet")
    fills,_=reconstruct_sales(orders)
    sales=stock_date_sales(fills,pd.read_parquet(OUT/"account/holdings.parquet"))
    keys=orders.loc[orders.stop_loss & orders.notional.gt(0),["datetime","instrument"]].drop_duplicates()
    stops=sales.merge(keys,on=["datetime","instrument"],validate="one_to_one")
    result=dict(stop_sales=len(stops),below_minus13=int(stops.net_return.lt(-.13).sum()),
        worst_stop_sale=stops.sort_values("net_return").iloc[0].to_dict() if len(stops) else None,
        mean_stop_return=float(stops.net_return.mean()) if len(stops) else None,
        mature_rank_ic=summary["mature_rank_ic"],yearly_rank_ic=summary["yearly_rank_ic"])
    assert len(stops)==summary["account"]["stop_fills"]
    save(OUT/"execution_diagnostics.json",result)
    doc=(OUT/"report.html").read_text(encoding="utf-8")
    section=f"<h2>实际止损成交</h2><p>共{len(stops)}次止损卖出，其中{result['below_minus13']}次含费用收益低于−13%；平均止损收益{result['mean_stop_return']:.2%}。最差：{html.escape(str(result['worst_stop_sale']))}。全期成熟标签平均Rank IC为{summary['mature_rank_ic']:.4f}，正IC不能保证扣除高换手费用后盈利。</p>"
    if "<h2>实际止损成交</h2>" not in doc:
        (OUT/"report.html").write_text(doc.replace("</html>",section+"</html>"),encoding="utf-8")
    manifest=read(OUT/"report_manifest.json")
    for p in (OUT/"report.html",OUT/"execution_diagnostics.json",ROOT/"scripts/report_s6_h5_daily_diagnostics.py"):
        manifest[str(p)]=digest(p)
    save(OUT/"report_manifest.json",manifest)
    print(result)


if __name__=="__main__": main()
