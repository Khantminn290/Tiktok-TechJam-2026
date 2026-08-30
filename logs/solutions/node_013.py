import argparse
import json
import os
import sys

import numpy as np

import train_lib
from evaluate import evaluate
from research_tools import incumbent_cfg, selection_rule_test


BASE_MENU = {
    'loss': 'bpr_pairwise',
    'neg_sampling': 'uniform_1',
    'user_history': 'recency_weighted_pool',
    'multitask': 'none',
    'model': 'fm_numpy',
    'temporal': 'none',
    'training': 'lower_lr_longer',
    'data_extras': 'none',
    'sample_weighting': 'per_row',
    'regularization': 'l2_default',
}


def build_cfg(splits, meta, seed):
    cfg, enc = incumbent_cfg(splits, meta)
    for k, v in BASE_MENU.items():
        cfg[k] = v
    cfg['seed'] = seed
    cfg['capture_epoch_scores'] = []
    return cfg, enc


def choose_top_epochs(epoch_records, max_candidates=5):
    ordered = sorted(epoch_records, key=lambda x: x['primary'], reverse=True)
    return ordered[:max_candidates]


def make_rule_candidates(top_epochs):
    candidates = {}
    epoch_ids = [rec['epoch'] for rec in top_epochs]
    for n in [1, 2, 3, 5]:
        if len(top_epochs) >= n:
            name = f'top{n}_uniform'
            arr = np.stack([top_epochs[i]['scores_valid'] for i in range(n)], axis=0)
            candidates[name] = arr.mean(axis=0)
    return epoch_ids, candidates


def build_per_epoch_tensor(candidates, reference_order):
    curves = []
    for name in reference_order:
        curves.append(np.asarray(candidates[name], dtype=np.float64))
    single_seed = np.stack(curves, axis=0)
    return np.asarray([single_seed], dtype=np.float64)


def pick_rule_from_selection_result(sel_result, fallback_order):
    rules = sel_result.get('rules', {})
    best_name = None
    best_val = None
    for name in fallback_order:
        info = rules.get(name, {})
        val = info.get('mean_delta', None)
        if val is None:
            continue
        if best_val is None or float(val) > best_val:
            best_val = float(val)
            best_name = name
    if best_name is None:
        best_name = fallback_order[0]
    return best_name


def menu_for_rule(rule_name):
    menu = dict(BASE_MENU)
    if rule_name == 'top1_uniform':
        menu['checkpoint_combine'] = False
        menu['n_checkpoints'] = 1
    else:
        n = int(rule_name.replace('top', '').replace('_uniform', ''))
        menu['checkpoint_combine'] = True
        menu['n_checkpoints'] = n
    return menu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    try:
        json.loads(args.menu_choices)
        os.makedirs(args.output_dir, exist_ok=True)

        splits, meta = train_lib.load_cache()
        cfg, enc = build_cfg(splits, meta, args.seed)
        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        epoch_records = []
        for epoch, valid_primary, scores_valid in cfg['capture_epoch_scores']:
            epoch_records.append({
                'epoch': int(epoch),
                'primary': float(valid_primary),
                'scores_valid': np.asarray(scores_valid, dtype=np.float64),
            })

        if len(epoch_records) == 0:
            raise RuntimeError('capture_epoch_scores returned no epochs')

        top_epochs = choose_top_epochs(epoch_records, max_candidates=5)
        top_epoch_ids, candidates = make_rule_candidates(top_epochs)
        rule_order = [name for name in ['top1_uniform', 'top2_uniform', 'top3_uniform', 'top5_uniform'] if name in candidates]
        if len(rule_order) == 0:
            raise RuntimeError('no candidate rules constructed from captured epochs')

        per_epoch_scores = build_per_epoch_tensor(candidates, rule_order)
        users = np.asarray(splits['valid']['user_raw'])
        labels = np.asarray(splits['valid']['long_view'])
        sel_result = selection_rule_test(per_epoch_scores, users, labels, rule_order)
        chosen_rule = pick_rule_from_selection_result(sel_result, rule_order)

        final_scores_valid = np.asarray(candidates[chosen_rule], dtype=np.float64)
        metrics = evaluate(users, labels, final_scores_valid)
        metrics = {k: float(v) for k, v in metrics.items()}

        rerun_dir = os.path.join(args.output_dir, 'rerun_for_test_scores')
        rerun_menu = menu_for_rule(chosen_rule)
        train_lib.run(rerun_menu, rerun_dir, seed=args.seed)
        scores_test = np.load(os.path.join(rerun_dir, 'scores_test.npy'))

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)

        diagnostics = {
            'chosen_rule': chosen_rule,
            'candidate_rules': rule_order,
            'top_epochs_by_same_valid': [int(x) for x in top_epoch_ids],
            'epoch_primaries': {str(rec['epoch']): float(rec['primary']) for rec in top_epochs},
            'selection_rule_test': sel_result,
            'rerun_menu_for_test_scores': rerun_menu,
            'train_return_keys': list(res.keys()),
        }
        with open(os.path.join(args.output_dir, 'selection_diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)

        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
