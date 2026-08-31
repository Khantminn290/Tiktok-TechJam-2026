import argparse
import json
import os
import sys
import traceback

import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test
from evaluate import evaluate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu-choices', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def mean_scores(arrs):
    if not arrs:
        raise ValueError('No score arrays to average')
    base = np.asarray(arrs[0], dtype=np.float64)
    if len(arrs) == 1:
        return base.astype(np.float32)
    out = np.zeros_like(base, dtype=np.float64)
    for a in arrs:
        aa = np.asarray(a, dtype=np.float64)
        if aa.shape != out.shape:
            raise ValueError('Score arrays have mismatched shapes')
        out += aa
    out /= float(len(arrs))
    return out.astype(np.float32)


def build_rules():
    def _sorted_epoch_ids(epoch_scores):
        items = sorted(epoch_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [int(ep) for ep, _ in items]

    def best1(epoch_scores):
        ordered = _sorted_epoch_ids(epoch_scores)
        return ordered[:1]

    def top2_mean(epoch_scores):
        ordered = _sorted_epoch_ids(epoch_scores)
        return ordered[: min(2, len(ordered))]

    def top3_mean(epoch_scores):
        ordered = _sorted_epoch_ids(epoch_scores)
        return ordered[: min(3, len(ordered))]

    def top5_mean(epoch_scores):
        ordered = _sorted_epoch_ids(epoch_scores)
        return ordered[: min(5, len(ordered))]

    return {
        'best1': best1,
        'top2_mean': top2_mean,
        'top3_mean': top3_mean,
        'top5_mean': top5_mean,
    }


def extract_rule_value(info):
    if isinstance(info, (int, float, np.floating)):
        return float(info)
    if isinstance(info, dict):
        for key in ('mean', 'score', 'primary', 'heldout_mean', 'avg', 'delta_vs_reference'):
            if key in info and isinstance(info[key], (int, float, np.floating)):
                return float(info[key])
        nums = [float(v) for v in info.values() if isinstance(v, (int, float, np.floating))]
        if nums:
            return float(sum(nums) / len(nums))
    return float('-inf')


def choose_rule_from_audit(audit, fallback_name='best1'):
    if not isinstance(audit, dict):
        return fallback_name
    rules_info = audit.get('rules', {})
    if not isinstance(rules_info, dict) or not rules_info:
        return fallback_name
    scored = [(name, extract_rule_value(info)) for name, info in rules_info.items()]
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[0][0] if scored else fallback_name


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        menu_choices = json.loads(args.menu_choices)
        splits, meta = train_lib.load_cache()

        cfg, enc = incumbent_cfg(splits, meta)
        cfg.update({
            'loss': 'bpr_pairwise',
            'neg_sampling': 'uniform_1',
            'user_history': 'recency_weighted_pool',
            'multitask': 'none',
            'model': 'fm_numpy',
            'temporal': 'hour_plus_dow',
            'training': 'lower_lr_longer',
            'data_extras': 'none',
            'sample_weighting': 'per_row',
            'regularization': 'l2_default',
            'seed': int(args.seed),
            'capture_epoch_scores': [],
        })
        for k, v in menu_choices.items():
            cfg[k] = v

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        epoch_caps = cfg['capture_epoch_scores']
        if not epoch_caps:
            raise RuntimeError('No epoch scores were captured; cannot audit checkpoint rules.')

        epoch_primary = {}
        epoch_valid_scores = {}
        ordered_epochs = []
        ordered_valid_arrays = []

        for item in epoch_caps:
            if len(item) != 3:
                raise RuntimeError('capture_epoch_scores returned unexpected tuple shape')
            epoch, valid_primary, scores_valid = item
            epoch = int(epoch)
            valid_primary = float(valid_primary)
            scores_valid = np.asarray(scores_valid, dtype=np.float32)
            epoch_primary[epoch] = valid_primary
            epoch_valid_scores[epoch] = scores_valid

        for epoch in sorted(epoch_valid_scores.keys()):
            ordered_epochs.append(epoch)
            ordered_valid_arrays.append(epoch_valid_scores[epoch])

        rules = build_rules()
        user_ids_valid = np.asarray(splits['valid']['user_raw'])
        labels_valid = np.asarray(splits['valid']['long_view'])

        audit = selection_rule_test(ordered_valid_arrays, user_ids_valid, labels_valid, rules)
        chosen_rule_name = choose_rule_from_audit(audit, fallback_name='best1')
        chosen_epochs = rules[chosen_rule_name](epoch_primary)
        if not chosen_epochs:
            raise RuntimeError('Chosen rule returned no epochs')

        final_valid = mean_scores([epoch_valid_scores[ep] for ep in chosen_epochs])
        metrics = evaluate(user_ids_valid, labels_valid, final_valid)
        final_test = np.asarray(res['scores_test'], dtype=np.float32)

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), np.asarray(final_valid, dtype=np.float32))
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), final_test)

    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        sys.stderr.write(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()
