import argparse
import json
import os
import sys
import numpy as np

import train_lib
from evaluate import evaluate


FALLBACK_MENU = {
    "loss": "bpr_pairwise",
    "neg_sampling": "uniform_1",
    "user_history": "none",
    "multitask": "none",
    "model": "gru4rec_seq",
    "temporal": "none",
    "training": "lower_lr_longer",
    "data_extras": "none",
    "sample_weighting": "per_row",
    "regularization": "l2_default"
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--menu-choices", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def normalize_menu(menu):
    if not isinstance(menu, dict):
        raise ValueError("--menu-choices must decode to a JSON object")
    out = dict(FALLBACK_MENU)
    out.update(menu)
    return out


def build_cfg_from_menu(menu, seed):
    cfg = dict(menu)

    training = cfg.get("training", "default")
    if training == "default":
        cfg.setdefault("k", 16)
        cfg.setdefault("lr", 1e-3)
        cfg.setdefault("bs", 8192)
        cfg.setdefault("epochs", 40)
        cfg.setdefault("patience", 4)
    elif training == "k32":
        cfg.setdefault("k", 32)
        cfg.setdefault("lr", 1e-3)
        cfg.setdefault("bs", 8192)
        cfg.setdefault("epochs", 40)
        cfg.setdefault("patience", 4)
    elif training == "lower_lr_longer":
        cfg.setdefault("k", 16)
        cfg.setdefault("lr", 5e-4)
        cfg.setdefault("bs", 8192)
        cfg.setdefault("epochs", 12)
        cfg.setdefault("patience", 4)
    elif training == "two_stage_finetune":
        cfg.setdefault("k", 16)
        cfg.setdefault("lr", 1e-3)
        cfg.setdefault("bs", 8192)
        cfg.setdefault("epochs", 40)
        cfg.setdefault("patience", 4)
    else:
        raise ValueError("Unknown training schedule: %s" % training)

    reg = cfg.get("regularization", "l2_default")
    if reg == "l2_default":
        cfg.setdefault("l2", 1e-6)
    elif reg == "l2_1e5":
        cfg.setdefault("l2", 1e-5)
    elif reg == "l2_1e4":
        cfg.setdefault("l2", 1e-4)
    elif reg == "l2_1e3":
        cfg.setdefault("l2", 1e-3)
    else:
        raise ValueError("Unknown regularization: %s" % reg)

    cfg["history"] = cfg.get("user_history", "none")
    cfg.setdefault("n_checkpoints", 1)
    cfg.setdefault("checkpoint_combine", False)
    cfg["seed"] = seed
    cfg["capture_epoch_scores"] = []
    return cfg


def user_half_splits(user_ids, n_splits=6, seed=0):
    uniq = np.unique(user_ids)
    rng = np.random.RandomState(seed)
    splits = []
    for _ in range(n_splits):
        perm = rng.permutation(len(uniq))
        mid = len(uniq) // 2
        a_users = uniq[perm[:mid]]
        mask_a = np.isin(user_ids, a_users)
        mask_b = ~mask_a
        if np.any(mask_a) and np.any(mask_b):
            splits.append((mask_a, mask_b))
    return splits


def eval_primary(user_ids, labels, scores):
    return float(evaluate(user_ids, labels, scores)["primary"])


def mean_selected_predictions(masked_preds, idxs):
    arr = np.stack([masked_preds[j] for j in idxs], axis=0)
    return np.mean(arr, axis=0)


def analyze_epoch_selection(capture, valid_user_ids, valid_labels, output_dir, seed):
    if not capture:
        return

    epochs = []
    primaries = []
    preds = []
    for item in capture:
        if len(item) < 3:
            continue
        ep, primary, score_vec = item
        epochs.append(int(ep))
        primaries.append(float(primary))
        preds.append(np.asarray(score_vec, dtype=np.float64))

    if len(preds) == 0:
        return

    full_argmax_idx = int(np.argmax(np.asarray(primaries, dtype=np.float64)))
    full_argmax_metrics = evaluate(valid_user_ids, valid_labels, preds[full_argmax_idx])

    candidate_rules = [{"name": "single_best_epoch", "kind": "single"}]
    if len(preds) >= 2:
        candidate_rules.append({"name": "top2_avg", "kind": "topk_avg", "k": 2})
    if len(preds) >= 3:
        candidate_rules.append({"name": "top3_avg", "kind": "topk_avg", "k": 3})

    splits = user_half_splits(valid_user_ids, n_splits=6, seed=seed)
    details = []
    heldout_scores = {rule["name"]: [] for rule in candidate_rules}
    baseline_scores = []

    for split_id, (sel_mask, eval_mask) in enumerate(splits):
        directions = [
            (sel_mask, eval_mask, "a_to_b"),
            (eval_mask, sel_mask, "b_to_a"),
        ]
        for choose_mask, score_mask, tag in directions:
            choose_users = valid_user_ids[choose_mask]
            choose_labels = valid_labels[choose_mask]
            score_users = valid_user_ids[score_mask]
            score_labels = valid_labels[score_mask]

            choose_primary_by_epoch = np.asarray(
                [eval_primary(choose_users, choose_labels, p[choose_mask]) for p in preds],
                dtype=np.float64,
            )

            baseline = evaluate(score_users, score_labels, preds[full_argmax_idx][score_mask])
            baseline_scores.append(float(baseline["primary"]))

            row = {
                "split": int(split_id),
                "direction": tag,
                "full_valid_argmax_epoch": int(epochs[full_argmax_idx]),
                "full_valid_argmax_primary_on_scored_half": float(baseline["primary"]),
                "rule_results": [],
            }

            masked_preds = [p[score_mask] for p in preds]
            for rule in candidate_rules:
                if rule["kind"] == "single":
                    chosen_idx = int(np.argmax(choose_primary_by_epoch))
                    pred_scored = masked_preds[chosen_idx]
                    met = evaluate(score_users, score_labels, pred_scored)
                    heldout_scores[rule["name"]].append(float(met["primary"]))
                    row["rule_results"].append({
                        "rule": rule["name"],
                        "selected_epochs": [int(epochs[chosen_idx])],
                        "heldout_primary": float(met["primary"]),
                    })
                elif rule["kind"] == "topk_avg":
                    k = int(rule["k"])
                    idxs = np.argsort(choose_primary_by_epoch)[::-1][:k]
                    pred_scored = mean_selected_predictions(masked_preds, idxs)
                    met = evaluate(score_users, score_labels, pred_scored)
                    heldout_scores[rule["name"]].append(float(met["primary"]))
                    row["rule_results"].append({
                        "rule": rule["name"],
                        "selected_epochs": [int(epochs[j]) for j in idxs.tolist()],
                        "heldout_primary": float(met["primary"]),
                    })
            details.append(row)

    summary = {
        "epochs": [int(x) for x in epochs],
        "full_valid_primary_by_epoch": [float(x) for x in primaries],
        "full_valid_argmax_epoch": int(epochs[full_argmax_idx]),
        "full_valid_argmax_metrics": {k: float(v) for k, v in full_argmax_metrics.items()},
        "heldout_user_split_summary": {
            "mean_primary_full_valid_argmax": float(np.mean(np.asarray(baseline_scores, dtype=np.float64))) if baseline_scores else float(full_argmax_metrics["primary"])
        },
        "heldout_user_split_details": details,
    }

    for rule in candidate_rules:
        vals = np.asarray(heldout_scores[rule["name"]], dtype=np.float64)
        base = np.asarray(baseline_scores, dtype=np.float64)
        summary["heldout_user_split_summary"][rule["name"]] = {
            "mean_primary": float(np.mean(vals)) if len(vals) else None,
            "delta_vs_full_valid_argmax": float(np.mean(vals - base)) if len(vals) and len(base) == len(vals) else None,
        }

    with open(os.path.join(output_dir, "checkpoint_selection_analysis.json"), "w") as f:
        json.dump(summary, f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    raw_menu = json.loads(args.menu_choices)
    menu = normalize_menu(raw_menu)

    metrics = train_lib.run(menu, args.output_dir, seed=args.seed)

    cfg = build_cfg_from_menu(menu, args.seed)
    splits, meta = train_lib.load_cache()
    temporal = menu.get("temporal", "none")
    enc, dim, offsets, dims = train_lib.encode_features(splits, meta, temporal)

    model_name = menu.get("model", "")
    if model_name == "fm_numpy":
        train_lib.train_numpy_fm(cfg, enc, splits, meta, log=lambda *a, **k: None)
    elif model_name in ("deepfm_mlp", "dcn_lite", "gru4rec_seq"):
        train_lib.train_torch(cfg, enc, splits, meta, log=lambda *a, **k: None)
    else:
        raise ValueError("Unknown model: %s" % model_name)

    valid_user_ids = np.asarray(splits["valid"]["user_raw"])
    valid_labels = np.asarray(splits["valid"]["long_view"])
    analyze_epoch_selection(cfg["capture_epoch_scores"], valid_user_ids, valid_labels, args.output_dir, args.seed)

    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        sys.stderr.write(str(e) + "\n")
        raise
