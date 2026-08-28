import argparse
import json
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np
import train_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def load_scores(path):
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f'score file {path} must be 1D, got shape {arr.shape}')
    return arr.astype(np.float64, copy=False)


def per_user_zscore(scores, user_ids):
    scores = np.asarray(scores, dtype=np.float64)
    user_ids = np.asarray(user_ids)
    out = np.empty_like(scores, dtype=np.float64)
    unique_users, inv = np.unique(user_ids, return_inverse=True)
    for uidx in range(len(unique_users)):
        m = inv == uidx
        s = scores[m]
        mu = s.mean()
        sd = s.std()
        if sd < 1e-12:
            out[m] = 0.0
        else:
            out[m] = (s - mu) / sd
    return out


def evaluate_scores(valid_scores, splits):
    metrics = train_lib.evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], valid_scores)
    return {k: float(v) for k, v in metrics.items()}


def run_menu(menu_choices, output_dir, seed):
    os.makedirs(output_dir, exist_ok=True)
    metrics = train_lib.run(menu_choices, output_dir, seed=seed)
    return {k: float(v) for k, v in metrics.items()}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    tmp_root = None
    try:
        _ = json.loads(args.menu_choices)

        splits, meta = train_lib.load_cache()

        fm_cfg = {
            'loss': 'bpr_pairwise',
            'score_prior': 'recency_bayesian_item_author',
            'user_history': 'mean_pool_positives',
            'multitask': 'aux_click_like_forward',
            'model': 'fm_numpy',
            'temporal': 'none',
            'training': 'lower_lr_longer',
            'data_extras': 'none',
        }
        din_cfg = {
            'loss': 'bpr_pairwise',
            'score_prior': 'recency_bayesian_item_author',
            'user_history': 'din_attention',
            'multitask': 'none',
            'model': 'deepfm_mlp',
            'temporal': 'none',
            'training': 'lower_lr_longer',
            'data_extras': 'none',
        }

        tmp_root = tempfile.mkdtemp(prefix='kuairand_blend_')
        out_a = os.path.join(tmp_root, 'fm_best')
        out_b = os.path.join(tmp_root, 'din_auxfree')

        run_menu(fm_cfg, out_a, seed=args.seed)
        run_menu(din_cfg, out_b, seed=args.seed)

        va = load_scores(os.path.join(out_a, 'scores_valid.npy'))
        ta = load_scores(os.path.join(out_a, 'scores_test.npy'))
        vb = load_scores(os.path.join(out_b, 'scores_valid.npy'))
        tb = load_scores(os.path.join(out_b, 'scores_test.npy'))

        user_valid = splits['valid']['user_raw']
        user_test = splits['test']['user_raw']

        va_n = per_user_zscore(va, user_valid)
        vb_n = per_user_zscore(vb, user_valid)
        ta_n = per_user_zscore(ta, user_test)
        tb_n = per_user_zscore(tb, user_test)

        candidate_weights = [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        best = None
        for w in candidate_weights:
            s_valid = (1.0 - w) * va_n + w * vb_n
            metrics = evaluate_scores(s_valid, splits)
            record = {
                'w_din': float(w),
                'metrics': metrics,
            }
            if best is None or metrics['primary'] > best['metrics']['primary']:
                best = record

        w = best['w_din']
        scores_valid = ((1.0 - w) * va_n + w * vb_n).astype(np.float32)
        scores_test = ((1.0 - w) * ta_n + w * tb_n).astype(np.float32)
        metrics = evaluate_scores(scores_valid.astype(np.float64), splits)

        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        with open(os.path.join(args.output_dir, 'blend_info.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'base_model': fm_cfg,
                'second_model': din_cfg,
                'selected_w_din': float(w),
                'valid_metrics': {k: float(v) for k, v in metrics.items()},
            }, f)
        return 0
    except Exception:
        sys.stderr.write('ERROR: solution failed\n')
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if tmp_root is not None:
            shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
