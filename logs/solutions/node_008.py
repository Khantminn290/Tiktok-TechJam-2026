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


def _mean_score_from_rule_info(info):
    if isinstance(info, dict):
        for key in ("mean", "score", "primary"):
            if key in info and info[key] is not None:
                return float(info[key])
    elif info is not None:
        return float(info)
    return None


def _run_selection_rule_test(user_ids, labels, epoch_scores):
    candidates = ["best_epoch", "top2_mean", "top3_mean", "top5_mean"]
    if len(epoch_scores) < 5:
        candidates = [c for c in candidates if c != "top5_mean"]
    if len(epoch_scores) < 3:
        candidates = [c for c in candidates if c != "top3_mean"]
    if len(epoch_scores) < 2:
        candidates = [c for c in candidates if c != "top2_mean"]

    pred_map = {name: _scores_for_rule(name, epoch_scores) for name in candidates}

    report = {
        "status": "not_run",
        "reason": None,
        "candidate_rules": candidates,
        "winner": None,
    }

    try:
        res = selection_rule_test(user_ids=user_ids, labels=labels, rules=pred_map)
        report = {
            "status": "ok",
            "candidate_rules": candidates,
            "raw": res,
            "winner": None,
        }
        best_name = None
        best_val = None
        for name, info in res.get("rules", {}).items():
            score = _mean_score_from_rule_info(info)
            if score is None:
                continue
            if best_val is None or score > best_val:
                best_val = score
                best_name = name
        if best_name in pred_map:
            report["winner"] = best_name
    except Exception as e:
        report = {
            "status": "fallback",
            "reason": str(e),
            "candidate_rules": candidates,
            "winner": None,
        }

    return report


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

        scores_valid = np.array(res["scores_valid"], dtype=np.float32)
        scores_test = np.array(res["scores_test"], dtype=np.float32)
        metrics = evaluate(splits["valid"]["user_raw"], splits["valid"]["long_view"], scores_valid)

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)

        np.save(os.path.join(args.output_dir, "scores_valid.npy"), scores_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), scores_test)

        epoch_scores = cfg.get("capture_epoch_scores", [])
        selection_report = {
            "base_menu": base_menu,
            "n_epochs_captured": int(len(epoch_scores)),
            "final_output_rule": "train_lib_internal_selection",
            "valid_metrics_of_final_output": {k: float(v) for k, v in metrics.items()},
        }

        if epoch_scores:
            selection_report["checkpoint_rule_test"] = _run_selection_rule_test(
                user_ids=splits["valid"]["user_raw"],
                labels=splits["valid"]["long_view"],
                epoch_scores=epoch_scores,
            )

            per_rule_valid = {}
            for rule_name in selection_report["checkpoint_rule_test"]["candidate_rules"]:
                rule_scores = _scores_for_rule(rule_name, epoch_scores)
                rule_metrics = evaluate(splits["valid"]["user_raw"], splits["valid"]["long_view"], rule_scores)
                per_rule_valid[rule_name] = {k: float(v) for k, v in rule_metrics.items()}
            selection_report["in_sample_valid_metrics_by_rule"] = per_rule_valid
        else:
            selection_report["checkpoint_rule_test"] = {
                "status": "no_epoch_scores",
                "reason": "capture_epoch_scores was empty",
                "candidate_rules": [],
                "winner": None,
            }

        with open(os.path.join(args.output_dir, "selection_report.json"), "w") as f:
            json.dump(selection_report, f)

        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
