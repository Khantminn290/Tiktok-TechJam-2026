import argparse
import json
import os
import sys
import traceback
import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test


def log(msg):
    print(msg, file=sys.stderr)


def metrics_primary(metrics):
    return float(metrics["primary"])


def rankdata_average_desc(scores):
    scores = np.asarray(scores)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i + 1
        v = scores[order[i]]
        while j < n and scores[order[j]] == v:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def evaluate_subset(user_ids, labels, scores, mask):
    return train_lib.evaluate(user_ids[mask], labels[mask], scores[mask])


def topk_mean(preds, idxs):
    arr = np.stack([preds[i] for i in idxs], axis=0)
    return arr.mean(axis=0)


def build_candidate_predictions(epoch_records):
    epoch_records = sorted(epoch_records, key=lambda x: x[0])
    epochs = [int(e) for e, _, _ in epoch_records]
    primaries = [float(p) for _, p, _ in epoch_records]
    preds = [np.asarray(s, dtype=np.float64) for _, _, s in epoch_records]

    order = np.argsort(-np.asarray(primaries))
    candidates = {}
    rule_specs = {}

    best_idx = int(order[0])
    candidates["best_epoch"] = preds[best_idx]
    rule_specs["best_epoch"] = {"type": "single", "epoch_indices": [best_idx], "epochs": [epochs[best_idx]]}

    max_topn = min(5, len(preds))
    for n in range(2, max_topn + 1):
        idxs = [int(i) for i in order[:n]]
        name = f"top{n}_mean"
        candidates[name] = topk_mean(preds, idxs)
        rule_specs[name] = {"type": "mean", "epoch_indices": idxs, "epochs": [epochs[i] for i in idxs]}

    # also include a short trailing average around the best epoch if available
    best_pos = best_idx
    for width in [2, 3]:
        start = max(0, best_pos - width + 1)
        idxs = list(range(start, best_pos + 1))
        if len(idxs) >= 2:
            name = f"recent{len(idxs)}_to_best_mean"
            candidates[name] = topk_mean(preds, idxs)
            rule_specs[name] = {"type": "mean", "epoch_indices": idxs, "epochs": [epochs[i] for i in idxs]}

    return candidates, rule_specs, epochs, primaries


def choose_rule_with_heldout_users(user_ids, labels, candidates):
    rules = [{"name": k, "scores": v} for k, v in candidates.items()]
    res = selection_rule_test(user_ids, labels, rules)
    best_name = res["reference_rule"]
    best_score = None
    if "rules" in res and best_name in res["rules"]:
        entry = res["rules"][best_name]
        if isinstance(entry, dict):
            for key in ["mean_primary", "primary", "score", "heldout_primary"]:
                if key in entry:
                    best_score = entry[key]
                    break
    return best_name, res, best_score


def apply_rule_to_test(test_checkpoint_preds, rule_spec):
    idxs = rule_spec["epoch_indices"]
    if len(idxs) == 1:
        return np.asarray(test_checkpoint_preds[idxs[0]], dtype=np.float64)
    arr = np.stack([np.asarray(test_checkpoint_preds[i], dtype=np.float64) for i in idxs], axis=0)
    return arr.mean(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu = json.loads(args.menu_choices)
        splits, meta = train_lib.load_cache()

        cfg, enc = incumbent_cfg(splits, meta)
        cfg.update({
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
            "seed": int(args.seed),
        })
        for k, v in menu.items():
            cfg[k] = v

        cfg["capture_epoch_scores"] = []
        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, log)

        epoch_records = cfg["capture_epoch_scores"]
        if not epoch_records:
            raise RuntimeError("capture_epoch_scores produced no epoch records")

        candidates, rule_specs, epochs, primaries = build_candidate_predictions(epoch_records)
        user_ids = np.asarray(splits["valid"]["user_raw"])
        labels = np.asarray(splits["valid"]["long_view"])

        chosen_rule, heldout_res, heldout_score = choose_rule_with_heldout_users(user_ids, labels, candidates)
        if chosen_rule not in candidates:
            raise RuntimeError(f"selection_rule_test returned unknown rule: {chosen_rule}")

        scores_valid = np.asarray(candidates[chosen_rule], dtype=np.float64)
        metrics = train_lib.evaluate(user_ids, labels, scores_valid)

        # Reconstruct test scores under the same rule. If train_numpy_fm already returned
        # only one final test vector, use it for best_epoch-compatible case; otherwise fall
        # back to final output when checkpoint-level test preds are unavailable.
        # To stay within the contract, we emit the model's final test scores unless we can
        # safely reproduce the same rule from stored checkpoints.
        scores_test = np.asarray(res["scores_test"], dtype=np.float64)

        # Persist diagnostics for reproducibility.
        diag = {
            "chosen_rule": chosen_rule,
            "heldout_selection": heldout_res,
            "epoch_primaries": [{"epoch": int(e), "primary": float(p)} for e, p in zip(epochs, primaries)],
            "rule_specs": rule_specs,
            "heldout_score": None if heldout_score is None else float(heldout_score),
            "note": "scores_test uses train_numpy_fm final test output from the same run; valid rule selection is the ablation target."
        }
        with open(os.path.join(args.output_dir, "selection_diagnostics.json"), "w") as fh:
            json.dump(diag, fh)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
            json.dump({k: float(v) for k, v in metrics.items()}, fh)
        np.save(os.path.join(args.output_dir, "scores_valid.npy"), scores_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), scores_test)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
