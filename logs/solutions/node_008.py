import argparse
import json
import os
import sys

import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test
from evaluate import evaluate


def rank_average(score_matrix):
    score_matrix = np.asarray(score_matrix, dtype=np.float64)
    if score_matrix.ndim != 2:
        raise ValueError(f'score_matrix must be 2D, got shape {score_matrix.shape}')
    n_models, n_rows = score_matrix.shape
    out = np.zeros(n_rows, dtype=np.float64)
    for i in range(n_models):
        s = score_matrix[i]
        order = np.argsort(s, kind='mergesort')
        ranks = np.empty(n_rows, dtype=np.float64)
        ranks[order] = np.arange(n_rows, dtype=np.float64)
        out += ranks
    out /= float(n_models)
    return out.astype(np.float32)


def parse_rule_metric(info):
    if not isinstance(info, dict):
        return None
    for key in ('mean', 'score', 'primary', 'heldout_mean', 'estimate'):
        if key in info:
            try:
                return float(info[key])
            except Exception:
                pass
    return None


def build_rules(n_epochs):
    def best_epoch_rule(primary_curve, epoch_scores):
        idx = int(np.argmax(np.asarray(primary_curve, dtype=np.float64)))
        return np.asarray(epoch_scores[idx], dtype=np.float32)

    rules = {'best_epoch': best_epoch_rule}

    for k in (2, 3, 5):
        kk = min(k, n_epochs)

        def make_topk_rankavg(k_take):
            def _rule(primary_curve, epoch_scores):
                p = np.asarray(primary_curve, dtype=np.float64)
                order = np.argsort(p)[::-1]
                take = order[:k_take]
                return rank_average(np.asarray(epoch_scores[take], dtype=np.float32))
            return _rule

        rules[f'top{kk}_rankavg'] = make_topk_rankavg(kk)

    return rules


def reconstruct_rule(rule_name, curve_scores, epoch_primary):
    n_epochs = curve_scores.shape[0]
    if n_epochs < 1:
        raise ValueError('No epochs captured')

    if rule_name is None or rule_name == 'best_epoch':
        idx = int(np.argmax(epoch_primary))
        return np.asarray(curve_scores[idx], dtype=np.float32)

    if rule_name.startswith('top') and rule_name.endswith('_rankavg'):
        middle = rule_name[3:-8]
        k = int(middle)
        order = np.argsort(epoch_primary)[::-1]
        take = order[:min(k, len(order))]
        return rank_average(curve_scores[take])

    if rule_name.startswith('epoch_'):
        idx = int(rule_name.split('_', 1)[1])
        if idx < 0 or idx >= n_epochs:
            raise ValueError(f'epoch index {idx} out of range for {n_epochs} epochs')
        return np.asarray(curve_scores[idx], dtype=np.float32)

    idx = int(np.argmax(epoch_primary))
    return np.asarray(curve_scores[idx], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError('--menu-choices must decode to a JSON object')

        os.makedirs(args.output_dir, exist_ok=True)

        splits, meta = train_lib.load_cache()
        cfg, enc = incumbent_cfg(splits, meta)

        default_incumbent = {
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
        }
        cfg.update(default_incumbent)
        cfg.update(menu_choices)
        cfg['seed'] = args.seed
        cfg['capture_epoch_scores'] = []

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        captured = cfg['capture_epoch_scores']
        if not captured:
            raise ValueError('capture_epoch_scores is empty; training produced no epoch predictions')

        epoch_primary = []
        epoch_valid_scores = []
        for tup in captured:
            if len(tup) != 3:
                raise ValueError(f'Expected capture tuple of length 3, got {len(tup)}')
            _epoch, primary, scores_valid = tup
            epoch_primary.append(float(primary))
            epoch_valid_scores.append(np.asarray(scores_valid, dtype=np.float32))

        curve_scores = np.stack(epoch_valid_scores, axis=0)
        per_epoch_scores = np.expand_dims(curve_scores, axis=0)
        epoch_primary_arr = np.asarray(epoch_primary, dtype=np.float64)

        users = np.asarray(splits['valid']['user_raw'])
        labels = np.asarray(splits['valid']['long_view'])

        rules = build_rules(curve_scores.shape[0])
        srt = selection_rule_test(per_epoch_scores, users, labels, rules)

        best_rule_name = None
        best_rule_score = -1e18
        if not isinstance(srt, dict):
            raise ValueError('selection_rule_test did not return a dict')
        rule_results = srt.get('rules', {})
        if not isinstance(rule_results, dict):
            raise ValueError('selection_rule_test returned malformed rules results')

        for name, info in rule_results.items():
            score = parse_rule_metric(info)
            if score is not None and score > best_rule_score:
                best_rule_score = score
                best_rule_name = name

        if best_rule_name is None:
            ref = srt.get('reference_rule')
            if isinstance(ref, str):
                best_rule_name = ref
            else:
                best_rule_name = 'best_epoch'

        chosen_valid = reconstruct_rule(best_rule_name, curve_scores, epoch_primary_arr)
        metrics = evaluate(users, labels, chosen_valid)
        scores_test = np.asarray(res['scores_test'], dtype=np.float32)

        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), np.asarray(chosen_valid, dtype=np.float32))
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)

    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
