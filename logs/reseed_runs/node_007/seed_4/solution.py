import argparse
import json
import os
import sys
import traceback

import numpy as np

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu_choices = json.loads(args.menu_choices)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w") as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f)

        valid_path = os.path.join(args.output_dir, "scores_valid.npy")
        test_path = os.path.join(args.output_dir, "scores_test.npy")
        if not (os.path.exists(valid_path) and os.path.exists(test_path)):
            raise RuntimeError(
                "train_lib.run did not produce required score files: "
                f"scores_valid.npy exists={os.path.exists(valid_path)}, "
                f"scores_test.npy exists={os.path.exists(test_path)}"
            )

        with open(metrics_path, "r") as f:
            loaded = json.load(f)
        required = {"GAUC", "nDCG@5", "primary"}
        if not required.issubset(loaded.keys()):
            raise RuntimeError(f"metrics.json missing keys: expected {required}, got {set(loaded.keys())}")

        np.load(valid_path)
        np.load(test_path)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
