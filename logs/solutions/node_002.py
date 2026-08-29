#!/usr/bin/env python3
import argparse
import json
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


def recency_centroids_from_train(train, n_users, n_videos, emb_dim, video_vecs, max_hist=30, decay=0.85):
    user = np.asarray(train['user'], dtype=np.int64)
    video = np.asarray(train['video'], dtype=np.int64)
    label = np.asarray(train['long_view'], dtype=np.int8)
    date = np.asarray(train['date'], dtype=np.int64)
    time_ms = np.asarray(train['time_ms'], dtype=np.int64)

    order = np.lexsort((time_ms, date, user))
    user_o = user[order]
    video_o = video[order]
    label_o = label[order]

    centroids = np.zeros((n_users, emb_dim), dtype=np.float32)
    norms = np.zeros(n_users, dtype=np.float32)

    start = 0
    n = len(order)
    while start < n:
        u = user_o[start]
        end = start + 1
        while end < n and user_o[end] == u:
            end += 1
        vids = video_o[start:end][label_o[start:end] == 1]
        if vids.size > 0:
            vids = vids[-max_hist:]
            vecs = video_vecs[vids]
            m = vecs.shape[0]
            w = decay ** np.arange(m - 1, -1, -1, dtype=np.float32)
            denom = float(np.sum(w))
            c = (vecs * w[:, None]).sum(axis=0) / max(denom, 1e-12)
            cn = float(np.linalg.norm(c))
            if cn > 0:
                centroids[u] = c.astype(np.float32)
                norms[u] = cn
        start = end
    return centroids, norms


def compute_similarity_prior(split, centroids, centroid_norms, video_vecs, video_norms):
    user = np.asarray(split['user'], dtype=np.int64)
    video = np.asarray(split['video'], dtype=np.int64)
    uv = centroids[user]
    vv = video_vecs[video]
    dot = np.sum(uv * vv, axis=1)
    denom = centroid_norms[user] * video_norms[video]
    sim = np.zeros_like(dot, dtype=np.float32)
    mask = denom > 1e-12
    sim[mask] = (dot[mask] / denom[mask]).astype(np.float32)
    return sim


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

        # Refit the same incumbent once to expose learned embeddings for the similarity prior.
        # Root-cause fix from the failed attempt: train_lib.train_numpy_fm expects cfg['history']
        # in some internal paths, so provide both aliases.
        cfg_train = dict(cfg)
        cfg_train['history'] = cfg['user_history']

        def log(*a, **k):
            return None

        model, _ = train_lib.train_numpy_fm(cfg_train, enc, splits, meta, log)

        state = model.state()
        V = None
        for key in ('V', 'embeddings', 'embed'):
            if key in state:
                V = np.asarray(state[key])
                break
        if V is None:
            raise RuntimeError('Could not find FM embedding matrix in model.state(); available keys: %s' % list(state.keys()))

        video_field = 1
        video_start = int(offsets[video_field])
        n_videos = int(meta['field_dims']['video'])
        video_vecs = np.asarray(V[video_start:video_start + n_videos], dtype=np.float32)
        video_norms = np.linalg.norm(video_vecs, axis=1).astype(np.float32)

        n_users = int(meta['field_dims']['user'])
        centroids, centroid_norms = recency_centroids_from_train(
            splits['train'], n_users=n_users, n_videos=n_videos,
            emb_dim=video_vecs.shape[1], video_vecs=video_vecs,
            max_hist=30, decay=0.85,
        )

        prior_train = compute_similarity_prior(splits['train'], centroids, centroid_norms, video_vecs, video_norms)
        prior_valid = compute_similarity_prior(splits['valid'], centroids, centroid_norms, video_vecs, video_norms)
        prior_test = compute_similarity_prior(splits['test'], centroids, centroid_norms, video_vecs, video_norms)

        prior_valid_z = zscore_from_train(prior_train, prior_valid)
        prior_test_z = zscore_from_train(prior_train, prior_test)

        labels_valid = np.asarray(splits['valid']['long_view'])
        users_valid = np.asarray(splits['valid']['user_raw'])

        alphas = [0.0, 0.01, 0.02, 0.05, 0.10]
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
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
