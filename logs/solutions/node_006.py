import argparse
import json
import os
import sys

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")
    except Exception as e:
        print(f"Failed to parse --menu-choices: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w") as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f)
        return 0
    except Exception as e:
        print(f"Training/evaluation failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
