import argparse
import json
import os
import sys

import numpy as np

import train_lib
from evaluate import evaluate
from research_tools import incumbent_cfg, selection_rule_test, contract


def _call_selection_rule_test(candidates, user_ids, labels, seed):
    """Call selection_rule_test defensively without guessing unsupported kwargs.

    We first try the most likely keyword signature, then fall back to positional
    forms if needed. This preserves the original experiment intent while fixing
    the exact API mismatch that crashed node 10.
    """
    tried = []

    # Likely modern keyword form.
    try:
        return selection_rule_test(candidates=candidates, user_ids=user_ids, labels=labels, seed=seed)
    except TypeError as e:
        tried.append(str(e))

    # Positional core arguments with seed keyword.
    try:
        return selection_rule_test(candidates, user_ids, labels, seed=seed)
    except TypeError as e:
        tried.append(str(e))

    # Pure positional fallback.
    try:
        return selection_rule_test(candidates, user_ids, labels, seed)
    except TypeError as e:
        tried.append(str(e))

    # Final fallback: some implementations may infer labels/user ids from dict-like payload.
    try:
        payload = {
            'candidates': candidates,
            'user_ids': user_ids,
            'labels': labels,
            'seed': seed,
        }
        return selection_rule_test(payload)
    except TypeError as e:
        tried.append(str(e))

    raise TypeError('selection_rule_test signature mismatch; attempts: ' + ' || '.join(tried))


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
        cfg.update({
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
            'seed': args.seed,
        })
        cfg['capture_epoch_scores'] = []

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)
        scores_test = np.asarray(res['scores_test'])

        epoch_rows = cfg['capture_epoch_scores']
        if not epoch_rows:
            raise RuntimeError('capture_epoch_scores returned no epochs')

        epoch_rows = sorted(epoch_rows, key=lambda t: float(t[1]), reverse=True)
        top = epoch_rows[:5]
        top_epochs = [int(ep) for ep, _, _ in top]
        top_arrays = [np.asarray(sv) for _, _, sv in top]

        candidates = {}
        candidates['top1_uniform'] = np.asarray(top_arrays[0])
        if len(top_arrays) >= 2:
            candidates['top2_uniform'] = np.mean(np.stack(top_arrays[:2], axis=0), axis=0)
        if len(top_arrays) >= 3:
            candidates['top3_uniform'] = np.mean(np.stack(top_arrays[:3], axis=0), axis=0)
        if len(top_arrays) >= 5:
            candidates['top5_uniform'] = np.mean(np.stack(top_arrays[:5], axis=0), axis=0)

        valid_user_ids = splits['valid']['user_raw']
        valid_labels = splits['valid']['long_view']

        rule_test = _call_selection_rule_test(candidates, valid_user_ids, valid_labels, args.seed)
        rule_results = rule_test.get('rules', {})
        if not rule_results:
            raise RuntimeError('selection_rule_test returned no rule results')

        def sort_key(item):
            name, info = item
            mean_delta = float(info.get('mean_delta', -1e18))
            sigma = float(info.get('sigma', -1e18))
            tval = float(info.get('t', -1e18))
            return (mean_delta, sigma, tval, name)

        chosen_rule = sorted(rule_results.items(), key=sort_key, reverse=True)[0][0]
        if chosen_rule not in candidates:
            raise RuntimeError('chosen rule %s not present in candidates %s' % (chosen_rule, sorted(candidates.keys())))

        final_scores_valid = np.asarray(candidates[chosen_rule])
        metrics = evaluate(valid_user_ids, valid_labels, final_scores_valid)
        metrics = {k: float(v) for k, v in metrics.items()}

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)

        diag = {
            'chosen_rule': chosen_rule,
            'reference_rule': rule_test.get('reference_rule'),
            'top_epochs_by_same_valid_primary': top_epochs,
            'epoch_primaries_top5': {str(int(ep)): float(vp) for ep, vp, _ in top},
            'selection_rule_test': rule_test,
            'candidate_full_valid_metrics': {
                name: {k: float(v) for k, v in evaluate(valid_user_ids, valid_labels, sc).items()}
                for name, sc in candidates.items()
            },
            'selection_rule_test_contract': contract('selection_rule_test'),
        }
        with open(os.path.join(args.output_dir, 'selection_diagnostics.json'), 'w') as f:
            json.dump(diag, f)

        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
