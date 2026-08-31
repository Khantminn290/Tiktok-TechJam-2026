#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback

import train_lib


def parse_args():
    p = argparse.ArgumentParser(description="KuaiRand-Pure seed solution via train_lib.run")
    p.add_argument("--menu-choices", type=str, required=True,
                   help="JSON dict of menu choices")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to write metrics.json, scores_valid.npy, scores_test.npy")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)

        required = ["metrics.json", "scores_valid.npy", "scores_test.npy"]
        missing = [name for name in required if not os.path.exists(os.path.join(args.output_dir, name))]
        if missing:
            raise RuntimeError(f"train_lib.run completed but missing required outputs: {missing}")

        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
