import argparse
import json
import os
import sys
import traceback
from typing import List

import numpy as np
import train_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    p.add_argument("--output-dir", required=True, help="Directory to write outputs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        ensemble_seeds: List[int] = [int(args.seed) + i for i in range(3)]
        valid_scores = []
        test_scores = []

        for i, seed in enumerate(ensemble_seeds):
            member_dir = os.path.join(args.output_dir, f"member_{i}_seed_{seed}")
            os.makedirs(member_dir, exist_ok=True)
            metrics = train_lib.run(menu_choices, member_dir, seed=seed)
            if not isinstance(metrics, dict):
                raise RuntimeError(f"train_lib.run did not return a metrics dict for seed {seed}")

            valid_path = os.path.join(member_dir, "scores_valid.npy")
            test_path = os.path.join(member_dir, "scores_test.npy")
            if not os.path.exists(valid_path) or not os.path.exists(test_path):
                raise FileNotFoundError(f"Missing score files for seed {seed} in {member_dir}")

            valid_scores.append(np.load(valid_path))
            test_scores.append(np.load(test_path))

        avg_valid = np.mean(np.stack(valid_scores, axis=0), axis=0)
        avg_test = np.mean(np.stack(test_scores, axis=0), axis=0)

        splits, _ = train_lib.load_cache()
        user_ids = splits["valid"]["user_raw"]
        labels = splits["valid"]["long_view"]
        metrics = train_lib.evaluate(user_ids, labels, avg_valid)
        metrics = {k: float(v) for k, v in metrics.items()}

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
            json.dump(metrics, fh)
        np.save(os.path.join(args.output_dir, "scores_valid.npy"), avg_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), avg_test)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
