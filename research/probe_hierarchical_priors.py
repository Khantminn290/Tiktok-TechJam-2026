"""Cheap train-only target-prior probes against a saved validation prediction.

This is an exploration tool, not a final model.  It asks whether a categorical
interaction contains ranking signal that the incumbent did not already learn.
Only train labels are used to construct priors; validation labels are used only
by the official evaluator.  Test labels and test scores are never loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from runtime import train_lib


def _codes_from_train(train_key: np.ndarray, query_key: np.ndarray):
    """Map arbitrary int64 keys to train vocabulary; unseen query keys are -1."""
    vocab, inverse = np.unique(train_key.astype(np.int64), return_inverse=True)
    pos = np.searchsorted(vocab, query_key.astype(np.int64))
    seen = pos < len(vocab)
    safe = np.minimum(pos, max(len(vocab) - 1, 0))
    seen &= vocab[safe] == query_key
    return inverse, pos, seen, len(vocab)


def empirical_bayes_logit(train_key, query_key, y, row_weight, strength):
    """Smoothed, train-only target logit residual for each query row."""
    inv, query_pos, seen, n = _codes_from_train(train_key, query_key)
    counts = np.bincount(inv, weights=row_weight, minlength=n)
    positives = np.bincount(inv, weights=row_weight * y, minlength=n)
    global_rate = float(np.sum(row_weight * y) / np.sum(row_weight))
    global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
    rates = (positives + strength * global_rate) / (counts + strength)
    rates = np.clip(rates, 1e-5, 1.0 - 1e-5)
    residual = np.log(rates / (1.0 - rates)) - np.log(
        global_rate / (1.0 - global_rate))
    out = np.zeros(len(query_key), dtype=np.float64)
    out[seen] = residual[query_pos[seen]]
    return out, {
        "groups": int(n),
        "seen_fraction": float(np.mean(seen)),
        "strength": float(strength),
    }


def per_user_zscore(values, users):
    """Vectorized normalization; ranking metrics are computed within user."""
    values = np.asarray(values, dtype=np.float64)
    users = np.asarray(users, dtype=np.int64)
    n = int(users.max()) + 1
    count = np.bincount(users, minlength=n).astype(np.float64)
    total = np.bincount(users, weights=values, minlength=n)
    mean = total / np.maximum(count, 1.0)
    centered = values - mean[users]
    var = np.bincount(users, weights=centered * centered, minlength=n)
    std = np.sqrt(var / np.maximum(count, 1.0))
    return centered / np.maximum(std[users], 1e-12)


def _duration_bins(train, valid, n_bins):
    edges = np.unique(np.quantile(
        train["duration_ms"], np.linspace(0, 1, n_bins + 1)[1:-1]))
    return (np.searchsorted(edges, train["duration_ms"]).astype(np.int64),
            np.searchsorted(edges, valid["duration_ms"]).astype(np.int64),
            len(edges) + 1)


def _pair(a, b, b_dim):
    return a.astype(np.int64) * int(b_dim) + b.astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-scores", required=True)
    ap.add_argument("--output", default="scratch/prior_probe.json")
    args = ap.parse_args()

    splits, meta = train_lib.load_cache()
    tr, va = splits["train"], splits["valid"]
    base = np.load(args.base_scores).astype(np.float64)
    if base.shape != (len(va["user"]),):
        raise ValueError(f"base score shape {base.shape} does not match validation")

    y = tr["long_view"].astype(np.float64)
    age = tr["time_ms"].max() - tr["time_ms"].astype(np.float64)
    recency = np.exp(-age / (3.0 * 86400e3))
    dur_tr, dur_va, dur_dim = _duration_bins(tr, va, 20)
    hour_tr = (tr["hourmin"] // 100).astype(np.int64)
    hour_va = (va["hourmin"] // 100).astype(np.int64)
    day_tr = (tr["date"] % 100).astype(np.int64) - 8
    day_va = (va["date"] % 100).astype(np.int64) - 8
    dow_tr = (day_tr + 4) % 7
    dow_va = (day_va + 4) % 7
    tab_dim = meta["field_dims"]["tab"]

    candidates = {
        "tab": (tr["tab"].astype(np.int64),
                va["tab"].astype(np.int64), 1000.0),
        "video": (tr["video"].astype(np.int64),
                  va["video"].astype(np.int64), 20.0),
        "author": (tr["author"].astype(np.int64),
                   va["author"].astype(np.int64), 100.0),
        "duration20": (dur_tr, dur_va, 500.0),
        "tab_x_duration20": (_pair(tr["tab"], dur_tr, dur_dim),
                              _pair(va["tab"], dur_va, dur_dim), 300.0),
        "user_x_tab": (_pair(tr["user"], tr["tab"], tab_dim),
                       _pair(va["user"], va["tab"], tab_dim), 12.0),
        "user_x_duration20": (_pair(tr["user"], dur_tr, dur_dim),
                              _pair(va["user"], dur_va, dur_dim), 10.0),
        "author_x_tab": (_pair(tr["author"], tr["tab"], tab_dim),
                         _pair(va["author"], va["tab"], tab_dim), 25.0),
        "video_x_tab": (_pair(tr["video"], tr["tab"], tab_dim),
                        _pair(va["video"], va["tab"], tab_dim), 15.0),
        "author_x_duration20": (_pair(tr["author"], dur_tr, dur_dim),
                                _pair(va["author"], dur_va, dur_dim), 25.0),
        "user_x_author": (_pair(tr["user"], tr["author"],
                                meta["field_dims"]["author"]),
                          _pair(va["user"], va["author"],
                                meta["field_dims"]["author"]), 8.0),
        "user_x_video": (_pair(tr["user"], tr["video"],
                               meta["field_dims"]["video"]),
                         _pair(va["user"], va["video"],
                               meta["field_dims"]["video"]), 6.0),
        "tab_x_hour_x_dow": (
            _pair(_pair(tr["tab"], hour_tr, 24), dow_tr, 7),
            _pair(_pair(va["tab"], hour_va, 24), dow_va, 7), 150.0),
    }

    users = va["user"]
    user_raw = list(va["user_raw"])
    labels = va["long_view"]
    base_z = per_user_zscore(base, users)
    report = {
        "base": {k: float(v) for k, v in
                 train_lib.evaluate(user_raw, labels, base).items()
                 if k in ("GAUC", "nDCG@5", "primary")},
        "candidates": {},
        "notes": "Diagnostic only; promote a family only after paired multi-seed training.",
    }

    normalized_priors = {}
    for name, (train_key, valid_key, strength) in candidates.items():
        prior, info = empirical_bayes_logit(
            train_key, valid_key, y, recency, strength)
        prior_z = per_user_zscore(prior, users)
        normalized_priors[name] = prior_z
        row = {"metadata": info, "blends": {}}
        for alpha in (0.05, 0.10, 0.20, 0.30):
            score = base_z + alpha * prior_z
            metrics = train_lib.evaluate(user_raw, labels, score)
            row["blends"][f"{alpha:.2f}"] = {
                k: float(metrics[k]) for k in ("GAUC", "nDCG@5", "primary")}
        report["candidates"][name] = row

    combinations = {
        "scenario_item_creator": {
            "video_x_tab": 0.5,
            "author_x_tab": 0.5,
        },
        "scenario_item_creator_user": {
            "video_x_tab": 0.4,
            "author_x_tab": 0.4,
            "user_x_tab": 0.2,
        },
        "scenario_plus_duration": {
            "video_x_tab": 0.4,
            "author_x_tab": 0.3,
            "author_x_duration20": 0.3,
        },
    }
    report["combinations"] = {}
    for name, components in combinations.items():
        combined = sum(weight * normalized_priors[part]
                       for part, weight in components.items())
        combined = per_user_zscore(combined, users)
        row = {"components": components, "blends": {}}
        for alpha in (0.03, 0.05, 0.08, 0.10):
            metrics = train_lib.evaluate(
                user_raw, labels, base_z + alpha * combined)
            row["blends"][f"{alpha:.2f}"] = {
                key: float(metrics[key])
                for key in ("GAUC", "nDCG@5", "primary")}
        report["combinations"][name] = row

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
