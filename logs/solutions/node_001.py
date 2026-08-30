import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        os.makedirs(args.output_dir, exist_ok=True)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)

        required = ["metrics.json", "scores_valid.npy", "scores_test.npy"]
        missing = [name for name in required if not os.path.exists(os.path.join(args.output_dir, name))]
        if missing:
            raise RuntimeError(f"train_lib.run completed but missing expected outputs: {missing}")

    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
