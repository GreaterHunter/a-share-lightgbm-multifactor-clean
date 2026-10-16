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

    factor_window = config["factor_window"]
    label_horizon = config["label_horizon"]


    momentum_expression = (
        f"$close / Ref($close, {factor_window}) - 1"
    )


    label_expression = (
        f"Ref($close, -{label_horizon}) / Ref($open, -1) - 1"
    )


    instruments = D.instruments(config["universe"])

    factor_panel = D.features(
        instruments=instruments,
        fields=[momentum_expression, label_expression],
        start_time=config["start_date"],
        end_time=config["end_date"],
        freq="day",
    )

    factor_name = f"mom_{factor_window}d"
    label_name = f"future_return_{label_horizon}d"

    factor_panel.columns = [factor_name, label_name]

    factor_panel = factor_panel.dropna()

    output_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "factor_panel_sample.parquet"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    factor_panel.to_parquet(output_path)

    print("因子面板前 10 行：")
    print(factor_panel.head(10))

    print("\n因子与标签的描述统计：")
    print(factor_panel.describe())

    print(f"\n保存位置：{output_path}")
    print(f"数据形状：{factor_panel.shape}")


if __name__ == "__main__":
    main()
