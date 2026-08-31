import argparse
import json
import os
import sys

import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test
from evaluate import evaluate


def rank_average(score_matrix):
    # score_matrix: (n_models, n_rows)
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


def build_rule_predictions(curve_scores, rule_name):
    # curve_scores: (epochs, rows)
    if curve_scores.ndim != 2:
        raise ValueError(f'curve_scores must be 2D, got shape {curve_scores.shape}')
    n_epochs = curve_scores.shape[0]
    if n_epochs < 1:
        raise ValueError('No captured epoch scores found')

    if rule_name == 'best_epoch':
        raise RuntimeError('best_epoch should be resolved by selection_rule_test to a concrete epoch rule')

    if rule_name.startswith('epoch_'):
        idx = int(rule_name.split('_', 1)[1])
        if idx < 0 or idx >= n_epochs:
            raise ValueError(f'epoch index {idx} out of range for {n_epochs} epochs')
        return curve_scores[idx].astype(np.float32)

    if rule_name.startswith('top') and rule_name.endswith('_rankavg'):
        k = int(rule_name[3:-8])
        if k < 1:
            raise ValueError(f'Invalid top-k in rule {rule_name}')
        k = min(k, n_epochs)
        per_epoch_primary = []
        # We do not have per-half scores here; use captured primary to identify top epochs globally.
        # This mirrors the candidate family whose generalisation was judged by selection_rule_test.
        raise RuntimeError('top-k rankavg requires captured primary ordering alongside scores')

    raise ValueError(f'Unknown rule_name: {rule_name}')


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
        cfg.update(menu_choices)
        cfg['seed'] = args.seed
        cfg['capture_epoch_scores'] = []

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        captured = cfg['capture_epoch_scores']
        if not captured:
            raise ValueError('capture_epoch_scores is empty; training produced no epoch predictions')

        epoch_ids = []
        epoch_primary = []
        epoch_valid_scores = []
        for tup in captured:
            if len(tup) != 3:
                raise ValueError(f'Expected capture tuple of length 3, got {len(tup)}')
            epoch, primary, scores_valid = tup
            epoch_ids.append(int(epoch))
            epoch_primary.append(float(primary))
            epoch_valid_scores.append(np.asarray(scores_valid, dtype=np.float32))

        curve_scores = np.stack(epoch_valid_scores, axis=0)  # (epochs, rows)
        per_epoch_scores = np.array([curve_scores], dtype=np.float32)  # (seeds=1, epochs, rows)

        srt = selection_rule_test(
            per_epoch_scores=per_epoch_scores,
            user_ids=splits['valid']['user_raw'],
            labels=splits['valid']['long_view'],
        )

        rules = srt.get('rules', {})
        if not rules:
            raise ValueError(f'selection_rule_test returned no rules: {srt}')

        # Choose the rule with highest held-out estimated score.
        best_rule_name = None
        best_rule_score = -1e18
        for name, info in rules.items():
            score = None
            if isinstance(info, dict):
                for key in ('mean', 'score', 'primary', 'heldout_mean', 'estimate'):
                    if key in info:
                        score = float(info[key])
                        break
            if score is None:
                continue
            if score > best_rule_score:
                best_rule_score = score
                best_rule_name = name

        if best_rule_name is None:
            # Fall back to reference rule if shape/schema differs.
            best_rule_name = srt.get('reference_rule')
            if best_rule_name is None:
                raise ValueError(f'Could not identify a rule from selection_rule_test output: {srt}')

        # Reconstruct predictions for the chosen rule.
        # Supported concrete rules: epoch_i and topK_rankavg families.
        if best_rule_name.startswith('epoch_'):
            chosen_valid = build_rule_predictions(curve_scores, best_rule_name)
        elif best_rule_name.startswith('top') and best_rule_name.endswith('_rankavg'):
            k = int(best_rule_name[3:-8])
            order = np.argsort(np.asarray(epoch_primary))[::-1]
            take = order[: min(k, len(order))]
            chosen_valid = rank_average(curve_scores[take])
        elif best_rule_name in ('best_epoch', 'argmax'):
            idx = int(np.argmax(np.asarray(epoch_primary)))
            chosen_valid = curve_scores[idx].astype(np.float32)
        else:
            # If the tool returns a different rule label, try a conservative fallback to reference/best epoch.
            idx = int(np.argmax(np.asarray(epoch_primary)))
            chosen_valid = curve_scores[idx].astype(np.float32)

        metrics = evaluate(
            splits['valid']['user_raw'],
            splits['valid']['long_view'],
            chosen_valid,
        )

        # Keep blind test scores from the single trained run; no test-label access or test-metric computation.
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
