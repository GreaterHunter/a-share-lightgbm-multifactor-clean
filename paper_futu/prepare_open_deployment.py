"""Prepare the aligned model; no broker connection, order or off-cycle rebalance."""

import argparse
import json

from paper_futu.cn_deployment import prepare_transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    result = prepare_transition(args.as_of)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
