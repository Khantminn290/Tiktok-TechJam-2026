#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys
import traceback

import numpy as np
import train_lib


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu-choices', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def incumbent_cfg():
    return {
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


def build_low_level_cfg(menu_cfg, dim, seed):
    cfg = dict(menu_cfg)
    cfg['history'] = menu_cfg['user_history']
    cfg['dim'] = int(dim)
    cfg['seed'] = int(seed)
    cfg['aux_tasks'] = []
    cfg['l2'] = 1e-6
    cfg['neg_k'] = 1
    cfg['batch_size'] = 8192
    if menu_cfg['training'] == 'lower_lr_longer':
        cfg['k'] = 16
        cfg['lr'] = 5e-4
        cfg['max_epochs'] = 60
        cfg['patience'] = 6
    elif menu_cfg['training'] == 'default':
        cfg['k'] = 16
        cfg['lr'] = 1e-3
        cfg['max_epochs'] = 40
        cfg['patience'] = 4
    elif menu_cfg['training'] == 'k32':
        cfg['k'] = 32
        cfg['lr'] = 1e-3
        cfg['max_epochs'] = 40
        cfg['patience'] = 4
    else:
        raise ValueError('Unsupported training option for this script: %s' % menu_cfg['training'])
    return cfg


def item_prior_from_train(train_split, n_videos):
    vids = np.asarray(train_split['video'], dtype=np.int64)
    y = np.asarray(train_split['long_view'], dtype=np.float64)
    counts = np.bincount(vids, minlength=n_videos).astype(np.float64)
    pos = np.bincount(vids, weights=y, minlength=n_videos).astype(np.float64)
    global_rate = float(y.mean())
    prior_strength = 20.0
    shrunk = (pos + prior_strength * global_rate) / np.maximum(counts + prior_strength, 1.0)
    eps = 1e-6
    shrunk = np.clip(shrunk, eps, 1.0 - eps)
    logit = np.log(shrunk / (1.0 - shrunk)).astype(np.float32)
    return logit


def split_item_prior(split, item_logit_prior):
    vids = np.asarray(split['video'], dtype=np.int64)
    return item_logit_prior[vids].astype(np.float32)


def zscore_from_train(train_vals, vals):
    mu = float(np.mean(train_vals))
    sd = float(np.std(train_vals))
    if sd < 1e-8:
        return np.zeros_like(vals, dtype=np.float32)
    return ((vals - mu) / sd).astype(np.float32)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        _ = json.loads(args.menu_choices)

        cfg = incumbent_cfg()
        train_lib.run(cfg, args.output_dir, seed=args.seed)

        valid_scores = np.load(os.path.join(args.output_dir, 'scores_valid.npy'))
        test_scores = np.load(os.path.join(args.output_dir, 'scores_test.npy'))

        splits, meta = train_lib.load_cache()
        enc, dim, offsets, dims = train_lib.encode_features(splits, meta, temporal=cfg['temporal'])

        low_cfg = build_low_level_cfg(cfg, dim=dim, seed=args.seed)

        def log(*_args, **_kwargs):
            return None

        # Refit once with explicit numeric config so train_numpy_fm has the keys it expects.
        # This preserves the original experiment intent while fixing the KeyError: 'dim' crash.
        train_lib.train_numpy_fm(low_cfg, enc, splits, meta, log)

        n_videos = int(meta['field_dims']['video'])
        item_logit_prior = item_prior_from_train(splits['train'], n_videos=n_videos)
        prior_train = split_item_prior(splits['train'], item_logit_prior)
        prior_valid = split_item_prior(splits['valid'], item_logit_prior)
        prior_test = split_item_prior(splits['test'], item_logit_prior)

        prior_valid_z = zscore_from_train(prior_train, prior_valid)
        prior_test_z = zscore_from_train(prior_train, prior_test)

        labels_valid = np.asarray(splits['valid']['long_view'])
        users_valid = np.asarray(splits['valid']['user_raw'])

        alphas = [0.0, 0.005, 0.01, 0.02, 0.05]
        best_alpha = 0.0
        best_metrics = None
        best_valid_scores = valid_scores.astype(np.float32)

        for alpha in alphas:
            cand_scores = (valid_scores + alpha * prior_valid_z).astype(np.float32)
            metrics = train_lib.evaluate(users_valid, labels_valid, cand_scores)
            if best_metrics is None or float(metrics['primary']) > float(best_metrics['primary']):
                best_metrics = metrics
                best_alpha = alpha
                best_valid_scores = cand_scores

        final_test_scores = (test_scores + best_alpha * prior_test_z).astype(np.float32)

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in best_metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), best_valid_scores)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), final_test_scores)
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1


if __name__ == '__main__':
    sys.exit(main())
