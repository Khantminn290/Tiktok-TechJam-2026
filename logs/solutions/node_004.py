#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
import numpy as np

import train_lib

INCUMBENT = {
    "loss": "bpr_pairwise",
    "neg_sampling": "uniform_1",
    "user_history": "recency_weighted_pool",
    "multitask": "none",
    "model": "fm_numpy",
    "temporal": "hour_plus_dow",
    "training": "lower_lr_longer",
    "data_extras": "none",
    "sample_weighting": "per_row",
    "regularization": "l2_default",
}


def monotone_global_transform(scores: np.ndarray):
    x = np.asarray(scores, dtype=np.float64)
    center = float(np.median(x))
    scale = float(np.std(x))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    z = (x - center) / scale
    z = np.clip(z, -40.0, 40.0)
    y = 1.0 / (1.0 + np.exp(-z))
    return y.astype(np.float64), center, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        # Parse but intentionally do not use the external menu for this Path-B null test.
        # The incumbent configuration is fixed by the selected candidate definition.
        _ = json.loads(args.menu_choices)

        base_dir = os.path.join(args.output_dir, "_base_run")
        os.makedirs(base_dir, exist_ok=True)

        train_lib.run(INCUMBENT, base_dir, seed=args.seed)

        scores_valid = np.load(os.path.join(base_dir, "scores_valid.npy"))
        scores_test = np.load(os.path.join(base_dir, "scores_test.npy"))

        splits, _meta = train_lib.load_cache()
        labels_valid = splits["valid"]["long_view"]
        user_ids_valid = splits["valid"]["user_raw"]

        scores_valid_tx, center, scale = monotone_global_transform(scores_valid)
        scores_test_tx, _, _ = monotone_global_transform(scores_test)

        metrics = train_lib.evaluate(user_ids_valid, labels_valid, scores_valid_tx)
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics["transform_center"] = float(center)
        metrics["transform_scale"] = float(scale)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)

        np.save(os.path.join(args.output_dir, "scores_valid.npy"), scores_valid_tx)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), scores_test_tx)

    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
