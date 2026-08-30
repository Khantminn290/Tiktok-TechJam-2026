import argparse
import json
import os
import sys
import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test


def log(msg):
    print(msg, file=sys.stderr)


def topk_mean(preds, idxs):
    arr = np.stack([np.asarray(preds[i], dtype=np.float64) for i in idxs], axis=0)
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
    rule_specs["best_epoch"] = {
        "type": "single",
        "epoch_indices": [best_idx],
        "epochs": [epochs[best_idx]],
    }

    max_topn = min(5, len(preds))
    for n in range(2, max_topn + 1):
        idxs = [int(i) for i in order[:n]]
        name = f"top{n}_mean"
        candidates[name] = topk_mean(preds, idxs)
        rule_specs[name] = {
            "type": "mean",
            "epoch_indices": idxs,
            "epochs": [epochs[i] for i in idxs],
        }

    best_pos = best_idx
    for width in [2, 3]:
        start = max(0, best_pos - width + 1)
        idxs = list(range(start, best_pos + 1))
        if len(idxs) >= 2:
            name = f"recent{len(idxs)}_to_best_mean"
            candidates[name] = topk_mean(preds, idxs)
            rule_specs[name] = {
                "type": "mean",
                "epoch_indices": idxs,
                "epochs": [epochs[i] for i in idxs],
            }

    return candidates, rule_specs, epochs, primaries


def choose_rule_with_heldout_users(user_ids, labels, candidates):
    rules = [{"name": k, "scores": np.asarray(v, dtype=np.float64)} for k, v in candidates.items()]
    res = selection_rule_test(user_ids, labels, rules)

    best_name = res.get("reference_rule")
    if best_name in candidates:
        return best_name, res

    best_name = None
    best_val = None
    for name, entry in res.get("rules", {}).items():
        if not isinstance(entry, dict):
            continue
        score = None
        for key in ["mean_primary", "primary", "score", "heldout_primary"]:
            if key in entry:
                score = float(entry[key])
                break
        if score is not None and (best_val is None or score > best_val):
            best_val = score
            best_name = name
    if best_name is None or best_name not in candidates:
        raise RuntimeError("selection_rule_test did not return a usable rule")
    return best_name, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu = json.loads(args.menu_choices)
        if not isinstance(menu, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

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
        if len(epoch_records) == 0:
            raise RuntimeError("capture_epoch_scores produced no epoch records")

        candidates, rule_specs, epochs, primaries = build_candidate_predictions(epoch_records)
        user_ids = np.asarray(splits["valid"]["user_raw"])
        labels = np.asarray(splits["valid"]["long_view"])

        chosen_rule, heldout_res = choose_rule_with_heldout_users(user_ids, labels, candidates)
        scores_valid = np.asarray(candidates[chosen_rule], dtype=np.float64)
        metrics = train_lib.evaluate(user_ids, labels, scores_valid)
        scores_test = np.asarray(res["scores_test"], dtype=np.float64)

        diag = {
            "chosen_rule": chosen_rule,
            "heldout_selection": heldout_res,
            "epoch_primaries": [{"epoch": int(e), "primary": float(p)} for e, p in zip(epochs, primaries)],
            "rule_specs": rule_specs,
            "note": "VALID scores come from the held-out-user-selected epoch rule; TEST scores are the blind final output returned by the same single training run.",
        }
        with open(os.path.join(args.output_dir, "selection_diagnostics.json"), "w") as fh:
            json.dump(diag, fh)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fh:
            json.dump({k: float(v) for k, v in metrics.items()}, fh)
        np.save(os.path.join(args.output_dir, "scores_valid.npy"), scores_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), scores_test)
    except Exception as e:
        print("ERROR:", str(e), file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
