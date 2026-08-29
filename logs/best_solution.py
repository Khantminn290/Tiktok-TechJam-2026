#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", type=str, required=True,
                        help="JSON string accepted for contract compatibility; ignored by this custom Path B script.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Confirmation target: incumbent config + validation-safe snapshot averaging.
    menu_choices = {
        "loss": "bpr_pairwise",
        "neg_sampling": "uniform_1",
        "user_history": "mean_pool_positives",
        "multitask": "none",
        "model": "fm_numpy",
        "temporal": "hour_plus_dow",
        "training": "lower_lr_longer",
        "data_extras": "none",
        "sample_weighting": "per_row",
        "regularization": "l2_default",
        "snapshot_ensemble": 5,
        "snapshot_force": True,
    }

    metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write("ERROR: %s\n" % str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
