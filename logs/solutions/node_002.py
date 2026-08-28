import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        os.makedirs(args.output_dir, exist_ok=True)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f)

        for name in ["scores_valid.npy", "scores_test.npy"]:
            path = os.path.join(args.output_dir, name)
            if not os.path.exists(path):
                raise RuntimeError(f"train_lib.run did not create required output: {name}")

    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
