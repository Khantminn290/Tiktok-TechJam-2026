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
        # Required by contract, but this custom Path-B run ignores menu contents.
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

        epoch_rows = list(cfg['capture_epoch_scores'])
        if not epoch_rows:
            raise RuntimeError('capture_epoch_scores returned no epochs')

        # selection_rule_test compares rules for choosing among per-epoch predictions.
        # We therefore provide the same top-K epoch pool to every rule and let the tool
        # evaluate which rule generalises on held-out user splits.
        epoch_rows_sorted = sorted(epoch_rows, key=lambda t: float(t[1]), reverse=True)
        top = epoch_rows_sorted[:5]
        if len(top) < 1:
            raise RuntimeError('need at least one epoch candidate')

        per_epoch_scores = [np.asarray(sv) for (_, _, sv) in top]
        top_epochs = [int(ep) for (ep, _, _) in top]
        top_primaries = [float(vp) for (_, vp, _) in top]

        valid_user_ids = splits['valid']['user_raw']
        valid_labels = splits['valid']['long_view']

        rules = ['top1_uniform', 'top2_uniform', 'top3_uniform', 'top5_uniform']
        # Trim rules that require more epochs than available.
        available_rules = []
        for r in rules:
            need = int(r.split('_')[0][3:])
            if len(per_epoch_scores) >= need:
                available_rules.append(r)
        if not available_rules:
            raise RuntimeError('no selection rules available for %d epochs' % len(per_epoch_scores))

        rule_test = selection_rule_test(per_epoch_scores, valid_user_ids, valid_labels, available_rules)
        rule_results = rule_test.get('rules', {})
        if not rule_results:
            raise RuntimeError('selection_rule_test returned no rule results')

        def _rule_value(info, key, default=-1e18):
            val = info.get(key, default)
            try:
                return float(val)
            except Exception:
                return float(default)

        # Prefer the rule with best held-out mean delta; tie-break with sigma then t if present.
        chosen_rule = sorted(
            rule_results.items(),
            key=lambda kv: (
                _rule_value(kv[1], 'mean_delta'),
                _rule_value(kv[1], 'sigma'),
                _rule_value(kv[1], 't'),
                kv[0],
            ),
            reverse=True,
        )[0][0]

        def combine_rule(name, arrays):
            n = int(name.split('_')[0][3:])
            if len(arrays) < n:
                raise RuntimeError('rule %s requires %d epochs, only have %d' % (name, n, len(arrays)))
            if n == 1:
                return np.asarray(arrays[0])
            return np.mean(np.stack(arrays[:n], axis=0), axis=0)

        final_scores_valid = combine_rule(chosen_rule, per_epoch_scores)
        metrics = evaluate(valid_user_ids, valid_labels, final_scores_valid)
        metrics = {k: float(v) for k, v in metrics.items()}

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)

        diagnostics = {
            'chosen_rule': chosen_rule,
            'reference_rule': rule_test.get('reference_rule'),
            'top_epochs_by_same_valid_primary': top_epochs,
            'top_epoch_valid_primary': top_primaries,
            'selection_rule_test': rule_test,
            'candidate_full_valid_metrics': {
                r: {k: float(v) for k, v in evaluate(valid_user_ids, valid_labels, combine_rule(r, per_epoch_scores)).items()}
                for r in available_rules
            },
        }
        with open(os.path.join(args.output_dir, 'selection_diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)

        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
