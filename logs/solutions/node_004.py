#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import shutil
import numpy as np

import train_lib


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_base_scores_via_train_lib(output_dir, seed):
    menu_choices = {
        "loss": "bpr_pairwise",
        "neg_sampling": "uniform_1",
        "user_history": "mean_pool_positives",
        "multitask": "none",
        "model": "fm_numpy",
        "temporal": "none",
        "training": "default",
        "data_extras": "none",
        "sample_weighting": "per_row",
        "regularization": "l2_default",
    }
    tmpdir = tempfile.mkdtemp(prefix="base_run_", dir=output_dir)
    try:
        train_lib.run(menu_choices, tmpdir, seed=seed)
        valid_scores = np.load(os.path.join(tmpdir, "scores_valid.npy"))
        test_scores = np.load(os.path.join(tmpdir, "scores_test.npy"))
        return valid_scores, test_scores
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def build_user_stats(train_split, valid_split, test_split):
    n_users = int(max(train_split["user"].max(), valid_split["user"].max(), test_split["user"].max()) + 1)
    train_user = train_split["user"].astype(np.int64)
    train_y = train_split["long_view"].astype(np.float64)

    impressions = np.bincount(train_user, minlength=n_users).astype(np.float64)
    positives = np.bincount(train_user, weights=train_y, minlength=n_users).astype(np.float64)

    global_rate = float(train_y.mean())
    # Beta-Binomial shrinkage toward global rate.
    prior_strength = 20.0
    alpha0 = global_rate * prior_strength
    beta0 = (1.0 - global_rate) * prior_strength
    shrunk_rate = (positives + alpha0) / (impressions + alpha0 + beta0)

    # Confidence from amount of user history, saturating slowly.
    conf = np.sqrt(impressions / (impressions + prior_strength))

    return {
        "shrunk_rate": shrunk_rate,
        "conf": conf,
        "global_rate": global_rate,
        "impressions": impressions,
        "positives": positives,
    }


def calibrate_scores(scores, user_ids, user_stats):
    scores = scores.astype(np.float64, copy=True)
    shrunk_rate = user_stats["shrunk_rate"]
    conf = user_stats["conf"]
    global_rate = float(user_stats["global_rate"])

    # Train-derived user factor: centered log-odds relative to global, then shrunk by confidence.
    eps = 1e-6
    user_p = np.clip(shrunk_rate[user_ids], eps, 1.0 - eps)
    global_p = min(max(global_rate, eps), 1.0 - eps)
    user_lo = np.log(user_p / (1.0 - user_p))
    global_lo = np.log(global_p / (1.0 - global_p))
    delta = (user_lo - global_lo) * conf[user_ids]

    # Mild bounded scaling to avoid destabilizing well-ranked lists.
    alpha = 1.0 + 0.08 * np.tanh(delta / 2.0)

    # Per-user affine transform preserving user mean but changing dispersion.
    out = scores.copy()
    user_ids = user_ids.astype(np.int64)
    unique_users, inverse = np.unique(user_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=scores, minlength=len(unique_users))
    cnts = np.bincount(inverse, minlength=len(unique_users)).astype(np.float64)
    means = sums / np.maximum(cnts, 1.0)
    user_means = means[inverse]
    out = user_means + alpha * (scores - user_means)
    return out


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    try:
        _ = json.loads(args.menu_choices)
    except Exception:
        # Accepted but unused; this iteration is custom Path B.
        pass

    try:
        base_valid, base_test = load_base_scores_via_train_lib(args.output_dir, args.seed)
        splits, meta = train_lib.load_cache()
        valid = splits["valid"]
        test = splits["test"]
        train = splits["train"]

        user_stats = build_user_stats(train, valid, test)
        cal_valid = calibrate_scores(base_valid, valid["user"], user_stats)
        cal_test = calibrate_scores(base_test, test["user"], user_stats)

        metrics = train_lib.evaluate(valid["user_raw"], valid["long_view"], cal_valid)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, "scores_valid.npy"), cal_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), cal_test)
        return 0
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
