import argparse
import json
import os
import sys

import numpy as np

import train_lib
from evaluate import evaluate
from research_tools import incumbent_cfg, selection_rule_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    try:
        # Contract requires accepting this argument; for this Path B run the
        # mechanism is fixed and does not depend on menu primitives.
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
        scores_test = res['scores_test']

        epoch_rows = cfg['capture_epoch_scores']
        if not epoch_rows:
            raise RuntimeError('capture_epoch_scores returned no epochs')

        epoch_rows = sorted(epoch_rows, key=lambda t: float(t[1]), reverse=True)
        top = epoch_rows[:5]

        candidates = {}
        top_epochs = [int(ep) for ep, _, _ in top]
        top_arrays = [np.asarray(sv) for _, _, sv in top]

        candidates['top1_uniform'] = np.asarray(top_arrays[0])
        if len(top_arrays) >= 2:
            candidates['top2_uniform'] = np.mean(np.stack(top_arrays[:2], axis=0), axis=0)
        if len(top_arrays) >= 3:
            candidates['top3_uniform'] = np.mean(np.stack(top_arrays[:3], axis=0), axis=0)
        if len(top_arrays) >= 5:
            candidates['top5_uniform'] = np.mean(np.stack(top_arrays[:5], axis=0), axis=0)

        rule_test = selection_rule_test(
            user_ids=splits['valid']['user_raw'],
            labels=splits['valid']['long_view'],
            candidate_scores=candidates,
            n_evaluations=8,
            seed=args.seed,
        )

        rule_results = rule_test.get('rules', {})
        if not rule_results:
            raise RuntimeError('selection_rule_test returned no rule results')

        def sort_key(item):
            name, info = item
            mean_delta = float(info.get('mean_delta', -1e18))
            sigma = float(info.get('sigma', 0.0))
            t = float(info.get('t', 0.0))
            return (mean_delta, sigma, t, name)

        chosen_rule = sorted(rule_results.items(), key=sort_key, reverse=True)[0][0]
        final_scores_valid = np.asarray(candidates[chosen_rule])
        metrics = evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], final_scores_valid)
        metrics = {k: float(v) for k, v in metrics.items()}

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)

        diag = {
            'chosen_rule': chosen_rule,
            'reference_rule': rule_test.get('reference_rule'),
            'top_epochs_by_same_valid_primary': top_epochs,
            'epoch_primaries': {str(int(ep)): float(vp) for ep, vp, _ in top},
            'selection_rule_test': rule_test,
            'candidate_full_valid_metrics': {
                name: {k: float(v) for k, v in evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], sc).items()}
                for name, sc in candidates.items()
            }
        }
        with open(os.path.join(args.output_dir, 'selection_diagnostics.json'), 'w') as f:
            json.dump(diag, f)

        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
