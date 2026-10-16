import qlib
from qlib.config import REG_CN
from qlib.data import D

from src.config import PROJECT_ROOT, load_research_config

def main() -> None:
    config = load_research_config()

    provider_uri = PROJECT_ROOT / config["provider_uri"]

    qlib.init(
        provider_uri=str(provider_uri),
        region=REG_CN,
    )

    instruments = D.instruments(config["universe"])

    stocks = D.list_instruments(
        instruments=instruments,
        start_time=config["start_date"],
        end_time=config["end_date"],
        as_list=True,
    )
    print(f"股票池历史涉及股票数：{len(stocks)}")
    print("前 10 只股票：")
    print(stocks[:10])

    fields = [
        "$open",
        "$close",
        "$volume",
        "($close / Ref($close, 1) - 1)",
    ]

    data = D.features(
        instruments=instruments,
        fields=fields,
        start_time=config["start_date"],
        end_time=config["end_date"],
        freq="day",
    )

    print("\n数据前 10 行：")
    print(data.head(10))

    print("\n数据形状（行数, 字段数）：")
    print(data.shape)


if __name__ == "__main__":
    main()
