import argparse
import json
import os
import sys

import numpy as np

import train_lib
from evaluate import evaluate
from research_tools import incumbent_cfg, selection_rule_test


def user_group_indices(user_ids):
    groups = {}
    for i, u in enumerate(user_ids):
        if u in groups:
            groups[u].append(i)
        else:
            groups[u] = [i]
    return groups


def subset_scores_from_groups(groups, selected_users, scores):
    idx = []
    for u in selected_users:
        idx.extend(groups[u])
    if len(idx) == 0:
        return np.zeros(0, dtype=scores.dtype), np.zeros(0, dtype=object), np.zeros(0, dtype=np.int8)
    idx = np.asarray(idx, dtype=np.int64)
    return scores[idx], idx


def metrics_on_user_subset(valid_user_raw, valid_labels, groups, selected_users, scores):
    idx = []
    for u in selected_users:
        idx.extend(groups[u])
    if not idx:
        return {'GAUC': 0.0, 'nDCG@5': 0.0, 'primary': 0.0}
    idx = np.asarray(idx, dtype=np.int64)
    m = evaluate(valid_user_raw[idx], valid_labels[idx], scores[idx])
    return {k: float(v) for k, v in m.items()}


def choose_rule_by_halves(valid_user_raw, valid_labels, epoch_entries, top_epoch_ids):
    candidates = {}

    sorted_epochs = list(top_epoch_ids)
    for n in [1, 2, 3, 5]:
        if len(sorted_epochs) >= n:
            name = f'top{n}_uniform'
            arrs = [epoch_entries[e]['scores_valid'] for e in sorted_epochs[:n]]
            candidates[name] = np.mean(np.stack(arrs, axis=0), axis=0)

    groups = user_group_indices(valid_user_raw)
    users = np.array(list(groups.keys()), dtype=object)

    rng = np.random.RandomState(12345)
    evals = []
    for _ in range(6):
        perm = users[rng.permutation(len(users))]
        mid = len(perm) // 2
        a = set(perm[:mid].tolist())
        b = set(perm[mid:].tolist())
        if len(a) == 0 or len(b) == 0:
            continue
        evals.append((a, b))
        evals.append((b, a))

    rule_stats = {k: [] for k in candidates}
    rule_means = {}
    for rule_name, scores in candidates.items():
        full_m = evaluate(valid_user_raw, valid_labels, scores)
        rule_means[rule_name] = float(full_m['primary'])

    for select_users, score_users in evals:
        best_rule = None
        best_select = None
        for rule_name, scores in candidates.items():
            m_sel = metrics_on_user_subset(valid_user_raw, valid_labels, groups, select_users, scores)
            val = m_sel['primary']
            if best_select is None or val > best_select:
                best_select = val
                best_rule = rule_name
        for rule_name, scores in candidates.items():
            m_score = metrics_on_user_subset(valid_user_raw, valid_labels, groups, score_users, scores)
            rule_stats[rule_name].append(m_score['primary'])
        rule_stats[best_rule].append(rule_stats[best_rule][-1] + 0.0)

    stable = []
    for rule_name, vals in rule_stats.items():
        if len(vals) == 0:
            mean_oos = -1e18
            std_oos = 1e18
        else:
            mean_oos = float(np.mean(vals))
            std_oos = float(np.std(vals))
        stable.append((mean_oos, -std_oos, rule_means[rule_name], rule_name))
    stable.sort(reverse=True)
    chosen = stable[0][3]

    return chosen, candidates, {
        'rule_full_valid_primary': rule_means,
        'stability_ranking': [
            {'rule': r, 'oos_mean_primary': float(m), 'oos_std_primary': float(-s), 'full_valid_primary': float(f)}
            for (m, s, f, r) in stable
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    try:
        _ = json.loads(args.menu_choices)
        os.makedirs(args.output_dir, exist_ok=True)

        splits, meta = train_lib.load_cache()
        cfg, enc = incumbent_cfg(splits, meta)
        cfg['loss'] = 'bpr_pairwise'
        cfg['neg_sampling'] = 'uniform_1'
        cfg['user_history'] = 'recency_weighted_pool'
        cfg['multitask'] = 'none'
        cfg['model'] = 'fm_numpy'
        cfg['temporal'] = 'none'
        cfg['training'] = 'lower_lr_longer'
        cfg['data_extras'] = 'none'
        cfg['sample_weighting'] = 'per_row'
        cfg['regularization'] = 'l2_default'
        cfg['seed'] = args.seed
        cfg['capture_epoch_scores'] = []

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        epoch_entries = {}
        for epoch, valid_primary, scores_valid in cfg['capture_epoch_scores']:
            epoch_entries[int(epoch)] = {
                'primary': float(valid_primary),
                'scores_valid': np.asarray(scores_valid)
            }

        if not epoch_entries:
            raise RuntimeError('capture_epoch_scores returned no epochs')

        ranked_epochs = sorted(epoch_entries.keys(), key=lambda e: epoch_entries[e]['primary'], reverse=True)
        top_epoch_ids = ranked_epochs[:5]

        chosen_rule, candidate_valid_scores, diag = choose_rule_by_halves(
            splits['valid']['user_raw'],
            splits['valid']['long_view'],
            epoch_entries,
            top_epoch_ids,
        )

        final_scores_valid = candidate_valid_scores[chosen_rule]
        metrics = evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], final_scores_valid)
        metrics = {k: float(v) for k, v in metrics.items()}

        # Produce test scores with the nearest expressible built-in checkpoint rule.
        if chosen_rule == 'top1_uniform':
            menu = {
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
                'checkpoint_combine': False,
                'n_checkpoints': 1,
            }
        else:
            n = int(chosen_rule.replace('top', '').replace('_uniform', ''))
            menu = {
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
                'checkpoint_combine': True,
                'n_checkpoints': n,
            }

        tmp_dir = os.path.join(args.output_dir, 'menu_rerun')
        train_lib.run(menu, tmp_dir, seed=args.seed)
        scores_test = np.load(os.path.join(tmp_dir, 'scores_test.npy'))

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)

        with open(os.path.join(args.output_dir, 'selection_diagnostics.json'), 'w') as f:
            json.dump({
                'chosen_rule': chosen_rule,
                'top_epochs_by_same_valid': [int(e) for e in top_epoch_ids],
                'epoch_primaries': {str(int(e)): float(epoch_entries[e]['primary']) for e in top_epoch_ids},
                'diagnostics': diag,
                'menu_rerun_rule': menu,
            }, f)

        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
