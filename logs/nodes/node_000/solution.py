import argparse
import json
import os
import sys
import traceback

import train_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    p.add_argument("--output-dir", required=True, help="Directory to write metrics.json and score npy files")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        if metrics is None:
            metrics_path = os.path.join(args.output_dir, "metrics.json")
            if not os.path.exists(metrics_path):
                raise RuntimeError("train_lib.run returned None and did not write metrics.json")
        return 0
    except Exception as e:
        sys.stderr.write("ERROR: " + str(e) + "\n")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
