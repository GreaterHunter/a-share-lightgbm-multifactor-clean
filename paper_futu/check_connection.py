"""Read-only OpenD check: display paper accounts, never submit orders."""

from __future__ import annotations

import argparse
import importlib.metadata
import socket
import sys


def paper_accounts(accounts):
    """Filter the server response before exposing account identifiers."""
    if "trd_env" not in accounts.columns:
        raise ValueError("富途账户响应缺少 trd_env 字段")
    return accounts.loc[accounts["trd_env"].eq("SIMULATE")].copy()


def stock_paper_accounts(accounts, market):
    """Select active stock paper accounts, excluding options-only and futures."""
    paper = paper_accounts(accounts)
    required = {"sim_acc_type", "trdmarket_auth", "acc_status"}
    if not required.issubset(paper.columns):
        raise ValueError("账户响应缺少模拟账户品类、市场权限或状态字段")
    account_type = {"HK": "STOCK", "US": "STOCK_AND_OPTION", "CN": "STOCK"}[market]
    has_market = paper["trdmarket_auth"].map(
        lambda markets: isinstance(markets, (list, tuple)) and market in markets
    )
    return paper.loc[
        paper["sim_acc_type"].eq(account_type)
        & paper["acc_status"].eq("ACTIVE")
        & has_market
    ].copy()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--market", choices=["HK", "US", "CN"], default="US")
    args = parser.parse_args()
    try:
        sdk_version = importlib.metadata.version("futu-api")
    except importlib.metadata.PackageNotFoundError:
        print("缺少富途SDK，请使用 paper_futu 的独立Python环境运行。")
        return 2

    print("SDK版本：", sdk_version)
    try:
        with socket.create_connection((args.host, args.port), timeout=3):
            pass
    except OSError:
        print(f"OpenD未就绪：{args.host}:{args.port}")
        print("请先在官方OpenD中登录牛牛账号，完成它要求的首次使用确认。")
        print("如果OpenD修改过监听端口，请用 --port 指定实际端口。")
        return 3

    from futu import Currency, OpenSecTradeContext, RET_OK, SecurityFirm, TrdEnv, TrdMarket

    context = OpenSecTradeContext(
        filter_trdmarket=getattr(TrdMarket, args.market),
        host=args.host,
        port=args.port,
        security_firm=SecurityFirm.FUTUSECURITIES,
    )
    try:
        status, accounts = context.get_acc_list()
        if status != RET_OK:
            print("账户查询失败：", accounts)
            return 4
        paper = stock_paper_accounts(accounts, args.market)
        if paper.empty:
            print(f"连接成功，但没有可见的 {args.market} 股票模拟账户。")
            print("请在富途模拟交易页面检查账户是否已开通；权限以账号实际返回为准。")
            return 5
        if "acc_id" in paper.columns:
            paper["account_suffix"] = paper["acc_id"].map(lambda value: "***" + str(value)[-4:])
        columns = [
            name for name in ["account_suffix", "trd_env", "acc_type", "sim_acc_type", "trdmarket_auth", "acc_status"]
            if name in paper.columns
        ]
        print("可见的模拟账户：")
        print(paper[columns].to_string(index=False))
        for account_id in paper["acc_id"]:
            account_id = int(account_id)
            status, assets = context.accinfo_query(
                trd_env=TrdEnv.SIMULATE,
                acc_id=account_id,
                currency={"US": Currency.USD, "HK": Currency.HKD, "CN": Currency.CNH}[args.market],
            )
            if status != RET_OK:
                print("模拟资产查询失败：", assets)
                return 6
            asset_columns = [
                name for name in ["currency", "total_assets", "cash", "market_val", "power"]
                if name in assets.columns
            ]
            print(f"模拟账户 ***{str(account_id)[-4:]} 资产：")
            print(assets[asset_columns].to_string(index=False))
            status, positions = context.position_list_query(
                trd_env=TrdEnv.SIMULATE, acc_id=account_id
            )
            if status != RET_OK:
                print("模拟持仓查询失败：", positions)
                return 7
            print("模拟持仓数量：", len(positions))
            if not positions.empty:
                position_columns = [
                    name for name in ["code", "qty", "can_sell_qty", "cost_price", "market_val"]
                    if name in positions.columns
                ]
                print(positions[position_columns].to_string(index=False))
        print("连接检查通过；本程序仅查询模拟账户、资产与持仓，没有下单。")
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    sys.exit(main())
