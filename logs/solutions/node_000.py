import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        os.makedirs(args.output_dir, exist_ok=True)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w") as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f)

        required = [
            os.path.join(args.output_dir, "metrics.json"),
            os.path.join(args.output_dir, "scores_valid.npy"),
            os.path.join(args.output_dir, "scores_test.npy"),
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("Missing required output files: " + ", ".join(missing))

        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
