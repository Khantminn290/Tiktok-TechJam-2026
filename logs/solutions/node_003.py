import argparse
import json
import os
import sys
import traceback

import train_lib


FIXED_MENU_CHOICES = {
    "loss": "bpr_pairwise",
    "neg_sampling": "uniform_1",
    "user_history": "recency_weighted_pool",
    "multitask": "none",
    "model": "fm_numpy",
    "temporal": "none",
    "training": "lower_lr_longer",
    "data_extras": "none",
    "sample_weighting": "per_row",
    "regularization": "l2_default",
    "lr": 0.0005,
    "epochs": 12,
    "patience": 4,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        user_menu = json.loads(args.menu_choices)
        if not isinstance(user_menu, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        os.makedirs(args.output_dir, exist_ok=True)

        # Faithfully execute the selected confirmation candidate from the policy.
        # Ignore caller-side substitutions except for exact matching informational use;
        # the experiment definition is fixed by this node.
        metrics = train_lib.run(FIXED_MENU_CHOICES, args.output_dir, seed=args.seed)

        # Ensure metrics.json exists with plain Python floats per contract, even if
        # train_lib already wrote it.
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
