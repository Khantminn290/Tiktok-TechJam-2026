import argparse
import json
import os
import sys

import numpy as np

import train_lib
from evaluate import evaluate
from research_tools import incumbent_cfg, selection_rule_test


def _build_base_menu(user_menu):
    base = {
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
    }
    if user_menu:
        for k, v in user_menu.items():
            base[k] = v
    return base


def _scores_for_rule(rule_name, epoch_scores):
    primaries = np.array([float(t[1]) for t in epoch_scores], dtype=float)
    order = np.argsort(-primaries)

    if rule_name == "best_epoch":
        return np.array(epoch_scores[int(order[0])][2], dtype=np.float32)

    if rule_name.startswith("top") and rule_name.endswith("_mean"):
        n = int(rule_name[3:-5])
        n = max(1, min(n, len(epoch_scores)))
        idx = order[:n]
        mats = [np.array(epoch_scores[int(i)][2], dtype=np.float32) for i in idx]
        return np.mean(np.stack(mats, axis=0), axis=0).astype(np.float32)

    raise ValueError("Unknown rule: %s" % rule_name)


def _extract_rule_score(info):
    if isinstance(info, dict):
        for key in ["mean", "score", "primary", "heldout_primary", "avg_primary"]:
            if key in info and info[key] is not None:
                return float(info[key])
        return None
    if info is None:
        return None
    return float(info)


def _pick_rule(epoch_scores, user_ids, labels):
    candidates = ["best_epoch", "top2_mean", "top3_mean", "top5_mean"]
    if len(epoch_scores) < 5:
        candidates = [c for c in candidates if c != "top5_mean"]

    pred_map = {name: _scores_for_rule(name, epoch_scores) for name in candidates}

    chosen = "best_epoch"
    report = {"status": "fallback", "reason": "selection_rule_test not run"}

    try:
        rules = {name: pred_map[name] for name in candidates}
        res = selection_rule_test(user_ids=user_ids, labels=labels, rules=rules)
        report = res

        best_name = None
        best_score = None
        for name, info in res.get("rules", {}).items():
            score = _extract_rule_score(info)
            if score is None:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_name = name
        if best_name in pred_map:
            chosen = best_name
    except Exception as e:
        report = {"status": "fallback", "reason": str(e)}

    return chosen, pred_map[chosen], report, pred_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        os.makedirs(args.output_dir, exist_ok=True)

        splits, meta = train_lib.load_cache()
        base_menu = _build_base_menu(menu_choices)
        cfg, enc = incumbent_cfg(splits, meta, **base_menu)
        cfg["seed"] = int(args.seed)
        cfg["capture_epoch_scores"] = []

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        epoch_scores = cfg["capture_epoch_scores"]
        if not epoch_scores:
            raise RuntimeError("No per-epoch scores captured; cannot test checkpoint rules")

        valid_user_ids = splits["valid"]["user_raw"]
        valid_labels = splits["valid"]["long_view"]

        chosen_rule, chosen_valid_scores, selection_report, pred_map = _pick_rule(
            epoch_scores, valid_user_ids, valid_labels
        )

        # Official outputs must be row-aligned arrays for both valid and test.
        # For test, only the trained model's blind final predictions are available.
        scores_valid = np.array(res["scores_valid"], dtype=np.float32)
        scores_test = np.array(res["scores_test"], dtype=np.float32)

        metrics = evaluate(valid_user_ids, valid_labels, scores_valid)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)

        np.save(os.path.join(args.output_dir, "scores_valid.npy"), scores_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), scores_test)

        diag = {
            "base_menu": base_menu,
            "seed": int(args.seed),
            "n_epochs": int(len(epoch_scores)),
            "candidate_rules": list(pred_map.keys()),
            "chosen_rule": chosen_rule,
            "chosen_rule_valid_metrics": {k: float(v) for k, v in evaluate(valid_user_ids, valid_labels, np.array(chosen_valid_scores, dtype=np.float32)).items()},
            "final_model_valid_metrics": {k: float(v) for k, v in metrics.items()},
            "selection_report": selection_report,
            "epoch_table": [
                {"epoch": int(ep), "valid_primary": float(primary)}
                for ep, primary, _ in epoch_scores
            ],
        }
        with open(os.path.join(args.output_dir, "selection_report.json"), "w") as f:
            json.dump(diag, f)

        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
