import argparse
import json
import os
import sys
import traceback

import train_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    p.add_argument("--output-dir", required=True, help="Directory to write metrics/scores")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        menu_choices = json.loads(args.menu_choices)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
