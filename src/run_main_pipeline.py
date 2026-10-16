"""Run the frozen 10-day ranked-enhanced annual-rolling strategy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_main_strategy_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and/or backtest the frozen main strategy.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "train", "backtest", "status"),
        default="all",
        help="all: train then backtest; train/backtest: run one stage only",
    )
    parser.add_argument("--legacy", action="store_true", help="Explicitly run the previous annual Top50 research pipeline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = Path(__file__).resolve().parents[1] / "configs/experiments/s6_head6_40_research_main.json"
    if registry.exists() and not args.legacy:
        selected = json.loads(registry.read_text(encoding="utf-8"))
        if selected.get("strategy_id") == "S6_head6_40_QREG":
            print("Current main strategy: S6 quarterly QREG, raw 15 factors, H10/R5, Top20/head6 40%, max6 replacements.")
            if args.stage == "status":
                from paper_futu.deploy_s6_quarterly import status
                print(json.dumps(status(),indent=2,ensure_ascii=True))
                return
            raise ValueError("The old annual Top50 pipeline is not the selected main model. "
                "Use python -m paper_futu.deploy_s6_quarterly --stage status/verify for the selected version; "
                "--legacy explicitly runs the old research pipeline. Quarterly refresh requires a new validated fit.")
    config = load_main_strategy_config()
    horizon = int(config["label_horizon"])
    rebalance_every = int(config["rebalance_every"])

    print("主策略配置已锁定：")
    print(f"策略名称：{config['strategy_name']}")
    print(f"模型：{config['model_variant']}")
    print(f"预测期限：{horizon}个交易日")
    print(f"标签与成交口径：T+1开盘至T+{horizon + 1}开盘；成熟标签按实际结束日筛选")
    print(f"调仓周期：{rebalance_every}个交易日")

    if args.stage in {"all", "train"}:
        from src.train_walk_forward_cv import run_walk_forward_training

        run_walk_forward_training()
    if args.stage in {"all", "backtest"}:
        from src.train_walk_forward_cv import run_walk_forward_backtest

        run_walk_forward_backtest()


if __name__ == "__main__":
    main()
