import argparse
import json
import os
import sys

import train_lib
from research_tools import selection_rule_test


DEFAULT_MENU = {
    "loss": "bpr_pairwise",
    "neg_sampling": "uniform_1",
    "user_history": "none",
    "multitask": "none",
    "model": "gru4rec_seq",
    "temporal": "none",
    "training": "lower_lr_longer",
    "data_extras": "none",
    "sample_weighting": "per_row",
    "regularization": "l2_default",
}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def merge_menu(user_menu):
    cfg = dict(DEFAULT_MENU)
    if user_menu:
        cfg.update(user_menu)
    return cfg


def topk_mean_indices(order, k):
    return list(order[: min(k, len(order))])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        user_menu = json.loads(args.menu_choices) if args.menu_choices else {}
        cfg = merge_menu(user_menu)

        splits, meta = train_lib.load_cache()
        enc, dim, offsets, dims = train_lib.encode_features(splits, meta, cfg.get("temporal", "none"))

        low_cfg = dict(cfg)
        low_cfg["dim"] = dim
        low_cfg["offsets"] = offsets
        low_cfg["dims"] = dims
        low_cfg["seed"] = args.seed
        low_cfg["capture_epoch_scores"] = []

        def log(msg):
            print(msg, file=sys.stderr)

        valid_scores, test_scores = train_lib.train_torch(low_cfg, enc, splits, meta, log)

        captured = low_cfg.get("capture_epoch_scores", [])
        if not captured:
            metrics = train_lib.evaluate(
                splits["valid"]["user_raw"],
                splits["valid"]["long_view"],
                valid_scores,
            )
            with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f)
            import numpy as np
            np.save(os.path.join(args.output_dir, "scores_valid.npy"), valid_scores)
            np.save(os.path.join(args.output_dir, "scores_test.npy"), test_scores)
            return

        import numpy as np

        epoch_primaries = []
        epoch_valid_preds = []
        epoch_test_preds = []
        for item in captured:
            if len(item) >= 4:
                epoch, valid_primary, vp, tp = item[0], item[1], item[2], item[3]
            else:
                raise RuntimeError("capture_epoch_scores format unsupported; expected (epoch, valid_primary, valid_scores, test_scores)")
            epoch_primaries.append(float(valid_primary))
            epoch_valid_preds.append(np.asarray(vp, dtype=np.float32))
            epoch_test_preds.append(np.asarray(tp, dtype=np.float32))

        order = np.argsort(-np.asarray(epoch_primaries))

        candidate_rules = {
            "single_best": topk_mean_indices(order, 1),
            "top2_mean": topk_mean_indices(order, 2),
            "top3_mean": topk_mean_indices(order, 3),
        }
        if len(order) >= 5:
            candidate_rules["top5_mean"] = topk_mean_indices(order, 5)

        candidate_valid = {}
        candidate_test = {}
        for name, idxs in candidate_rules.items():
            v = np.mean(np.stack([epoch_valid_preds[i] for i in idxs], axis=0), axis=0)
            t = np.mean(np.stack([epoch_test_preds[i] for i in idxs], axis=0), axis=0)
            candidate_valid[name] = v
            candidate_test[name] = t

        rule_names = list(candidate_valid.keys())
        pred_matrix = np.stack([candidate_valid[n] for n in rule_names], axis=1)

        srt = selection_rule_test(
            user_ids=splits["valid"]["user_raw"],
            labels=splits["valid"]["long_view"],
            candidate_scores=pred_matrix,
            candidate_names=rule_names,
            metric="primary",
        )

        selected_name = None
        if isinstance(srt, dict):
            for key in ["winner", "best_name", "selected_name", "chosen_name"]:
                if key in srt and srt[key] in candidate_valid:
                    selected_name = srt[key]
                    break
            if selected_name is None:
                for key in ["mean_scores", "candidate_means", "scores", "generalization_scores"]:
                    if key in srt and isinstance(srt[key], dict):
                        best_k = max(srt[key], key=lambda kk: srt[key][kk])
                        if best_k in candidate_valid:
                            selected_name = best_k
                            break
                    if key in srt and isinstance(srt[key], (list, tuple)) and len(srt[key]) == len(rule_names):
                        arr = list(srt[key])
                        selected_name = rule_names[int(np.argmax(arr))]
                        break
        if selected_name is None:
            selected_name = "single_best"

        final_valid = candidate_valid[selected_name]
        final_test = candidate_test[selected_name]
        metrics = train_lib.evaluate(
            splits["valid"]["user_raw"],
            splits["valid"]["long_view"],
            final_valid,
        )

        with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, "scores_valid.npy"), final_valid)
        np.save(os.path.join(args.output_dir, "scores_test.npy"), final_test)

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
