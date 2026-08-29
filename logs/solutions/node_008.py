import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import train_lib


BASE_MENU = {
    "loss": "bpr_pairwise",
    "neg_sampling": "uniform_1",
    "user_history": "mean_pool_positives",
    "multitask": "none",
    "model": "fm_numpy",
    "temporal": "none",
    "training": "lower_lr_longer",
    "data_extras": "none",
    "sample_weighting": "per_row",
    "regularization": "l2_default",
}


ALPHAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def resolve_menu(menu):
    if menu is None:
        return dict(BASE_MENU)
    if not isinstance(menu, dict):
        raise ValueError('--menu-choices must decode to a JSON object')
    if len(menu) == 0:
        return dict(BASE_MENU)
    if menu != BASE_MENU:
        raise ValueError(f'This confirmation script only supports the incumbent config or {{}}. Got: {menu}')
    return dict(BASE_MENU)


def build_cfg(menu):
    _ = resolve_menu(menu)
    return {
        'k': 16,
        'lr': 5e-4,
        'batch_size': 8192,
        'max_epochs': 60,
        'patience': 6,
        'l2': 1e-6,
    }


def grouped_indices_by_user(user_codes):
    d = defaultdict(list)
    for i, u in enumerate(user_codes):
        d[int(u)].append(i)
    return d


def make_bpr_pairs(splits, seed):
    rng = np.random.default_rng(seed)
    y = splits['train']['long_view'].astype(np.int8)
    user = splits['train']['user']
    by_user = grouped_indices_by_user(user)
    pos_idx = []
    neg_idx = []
    for _, idxs in by_user.items():
        idxs = np.asarray(idxs, dtype=np.int64)
        yi = y[idxs]
        pos = idxs[yi > 0]
        neg = idxs[yi <= 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        sampled_neg = neg[rng.integers(0, len(neg), size=len(pos))]
        pos_idx.append(pos)
        neg_idx.append(sampled_neg)
    if not pos_idx:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos_idx = np.concatenate(pos_idx)
    neg_idx = np.concatenate(neg_idx)
    perm = rng.permutation(len(pos_idx))
    return pos_idx[perm], neg_idx[perm]


def sigmoid_stable(x):
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def predict_in_batches(model, X, H, batch_size=32768):
    out = np.empty(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], batch_size):
        e = min(s + batch_size, X.shape[0])
        Hb = None if H is None else H[s:e]
        out[s:e] = model.predict(X[s:e], Hb).astype(np.float32)
    return out


def infer_embedding_matrix(model, dim, k):
    state = model.state()
    candidates = []
    if isinstance(state, dict):
        for v in state.values():
            if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == k:
                candidates.append(v)
    if hasattr(model, 'V') and isinstance(model.V, np.ndarray) and model.V.ndim == 2 and model.V.shape[1] == k:
        candidates.append(model.V)
    if not candidates:
        raise RuntimeError('Could not locate embedding matrix in RankFM state/model')
    for arr in candidates:
        if arr.shape[0] == dim:
            return arr.astype(np.float32, copy=False)
    best = max(candidates, key=lambda a: a.shape[0])
    return best.astype(np.float32, copy=False)


def row_item_vectors(X, V, batch_size=65536):
    n = X.shape[0]
    k = V.shape[1]
    out = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        out[s:e] = V[X[s:e]].sum(axis=1).astype(np.float32)
    return out


def build_user_centroids(train_users, train_labels, train_item_vecs, n_users):
    k = train_item_vecs.shape[1]
    sums = np.zeros((n_users, k), dtype=np.float32)
    counts = np.zeros(n_users, dtype=np.int32)
    pos_mask = train_labels.astype(np.int8) > 0
    users = train_users[pos_mask].astype(np.int64)
    vecs = train_item_vecs[pos_mask]
    np.add.at(sums, users, vecs)
    np.add.at(counts, users, 1)
    centroids = np.zeros_like(sums)
    nz = counts > 0
    centroids[nz] = sums[nz] / counts[nz, None]
    norms = np.linalg.norm(centroids, axis=1)
    return centroids, norms, counts


def cosine_affinity(users, item_vecs, centroids, centroid_norms, batch_size=65536):
    n = item_vecs.shape[0]
    out = np.zeros(n, dtype=np.float32)
    item_norms = np.linalg.norm(item_vecs, axis=1).astype(np.float32)
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        u = users[s:e].astype(np.int64)
        c = centroids[u]
        cn = centroid_norms[u].astype(np.float32)
        dots = np.sum(item_vecs[s:e] * c, axis=1).astype(np.float32)
        denom = item_norms[s:e] * cn
        mask = denom > 1e-12
        block = np.zeros(e - s, dtype=np.float32)
        block[mask] = dots[mask] / denom[mask]
        out[s:e] = block
    return out


def train_and_collect(menu, output_dir, seed):
    menu = resolve_menu(menu)
    cfg = build_cfg(menu)
    splits, meta = train_lib.load_cache()
    enc, dim, offsets, dims = train_lib.encode_features(splits, meta, temporal=menu['temporal'])

    H_train = H_valid = H_test = None
    if menu['user_history'] == 'mean_pool_positives':
        history = train_lib.History(splits, meta['field_dims']['user'], mode='mean_pool_positives')
        H_train = history.pooled['train']
        H_valid = history.pooled['valid']
        H_test = history.pooled['test']

    model = train_lib.RankFM(dim=dim, k=cfg['k'], lr=cfg['lr'], seed=seed, aux_tasks=[])

    X_train = enc['train']
    X_valid = enc['valid']
    X_test = enc['test']
    y_valid = splits['valid']['long_view']
    user_valid = splits['valid']['user_raw']

    best_primary = -1e18
    best_state = None
    wait = 0

    for epoch in range(cfg['max_epochs']):
        pos_idx, neg_idx = make_bpr_pairs(splits, seed + epoch)
        if len(pos_idx) == 0:
            raise RuntimeError('No BPR pairs could be formed from train split')

        for s in range(0, len(pos_idx), cfg['batch_size']):
            e = min(s + cfg['batch_size'], len(pos_idx))
            p = pos_idx[s:e]
            n = neg_idx[s:e]
            Xp = X_train[p]
            Xn = X_train[n]
            Hp = None if H_train is None else H_train[p]
            Hn = None if H_train is None else H_train[n]

            logit_p, cache_p = model.forward(Xp, Hp)
            logit_n, cache_n = model.forward(Xn, Hn)
            diff = (logit_p - logit_n).astype(np.float32)
            sig = sigmoid_stable(diff)
            g = (sig - 1.0).astype(np.float32) / max(1, (e - s))
            model.apply_grads([(cache_p, g), (cache_n, -g)], aux_contribs=None)

        valid_scores = predict_in_batches(model, X_valid, H_valid)
        metrics = train_lib.evaluate(user_valid, y_valid, valid_scores)
        primary = float(metrics['primary'])

        if primary > best_primary:
            best_primary = primary
            best_state = model.state()
            wait = 0
        else:
            wait += 1
            if wait >= cfg['patience']:
                break

    if best_state is None:
        raise RuntimeError('Training did not produce a valid best state')

    model.load_state(best_state)

    base_valid_scores = predict_in_batches(model, X_valid, H_valid)
    base_test_scores = predict_in_batches(model, X_test, H_test)

    V = infer_embedding_matrix(model, dim=dim, k=cfg['k'])
    train_item_vecs = row_item_vectors(X_train, V)
    valid_item_vecs = row_item_vectors(X_valid, V)
    test_item_vecs = row_item_vectors(X_test, V)

    n_users = meta['field_dims']['user']
    centroids, centroid_norms, counts = build_user_centroids(
        splits['train']['user'], splits['train']['long_view'], train_item_vecs, n_users
    )

    valid_aff = cosine_affinity(splits['valid']['user'], valid_item_vecs, centroids, centroid_norms)
    test_aff = cosine_affinity(splits['test']['user'], test_item_vecs, centroids, centroid_norms)

    best_alpha = 0.0
    best_metrics = train_lib.evaluate(user_valid, y_valid, base_valid_scores)
    best_valid_scores = base_valid_scores
    best_primary = float(best_metrics['primary'])

    for alpha in ALPHAS:
        cand_scores = (base_valid_scores + np.float32(alpha) * valid_aff).astype(np.float32)
        cand_metrics = train_lib.evaluate(user_valid, y_valid, cand_scores)
        cand_primary = float(cand_metrics['primary'])
        if cand_primary > best_primary:
            best_primary = cand_primary
            best_alpha = alpha
            best_metrics = cand_metrics
            best_valid_scores = cand_scores

    final_test_scores = (base_test_scores + np.float32(best_alpha) * test_aff).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, 'scores_valid.npy'), best_valid_scores)
    np.save(os.path.join(output_dir, 'scores_test.npy'), final_test_scores)
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({k: float(v) for k, v in best_metrics.items()}, f)

    return best_metrics


def main():
    args = parse_args()
    try:
        menu = json.loads(args.menu_choices)
        train_and_collect(menu, args.output_dir, args.seed)
    except Exception as e:
        print(f'ERROR: {e}', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
