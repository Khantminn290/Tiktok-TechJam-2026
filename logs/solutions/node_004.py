#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
import numpy as np
import train_lib


def sigmoid(x):
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def primary_from_scores(user_ids, labels, scores):
    m = train_lib.evaluate(user_ids, labels, scores)
    return {
        'GAUC': float(m['GAUC']),
        'nDCG@5': float(m['nDCG@5']),
        'primary': float((float(m['GAUC']) + float(m['nDCG@5'])) / 2.0),
    }


def build_user_masks(user_raw):
    uniq = np.unique(user_raw)
    rng = np.random.RandomState(12345)
    perm = uniq.copy()
    rng.shuffle(perm)
    half = len(perm) // 2
    a_users = set(perm[:half].tolist())
    mask_a = np.array([u in a_users for u in user_raw], dtype=bool)
    mask_b = ~mask_a
    return mask_a, mask_b


def metrics_on_mask(user_ids, labels, scores, mask):
    return primary_from_scores(user_ids[mask], labels[mask], scores[mask])


def make_history_vectors(choice, history_obj, rows=None):
    if choice == 'none':
        return None
    if rows is None:
        return history_obj.pooled
    return history_obj.batch_vectors(rows)


def train_epoch_bpr(model, X, y, users, hist_train, batch_size, rng):
    pos_idx = np.flatnonzero(y > 0.5)
    if len(pos_idx) == 0:
        return
    user_to_negs = {}
    for u in np.unique(users[pos_idx]):
        negs = np.flatnonzero((users == u) & (y <= 0.5))
        if len(negs) > 0:
            user_to_negs[int(u)] = negs
    pos_kept = [i for i in pos_idx if int(users[i]) in user_to_negs]
    if not pos_kept:
        return
    pos_kept = np.asarray(pos_kept, dtype=np.int64)
    rng.shuffle(pos_kept)

    for start in range(0, len(pos_kept), batch_size):
        pidx = pos_kept[start:start + batch_size]
        nidx = np.empty(len(pidx), dtype=np.int64)
        for j, pi in enumerate(pidx):
            negs = user_to_negs[int(users[pi])]
            nidx[j] = negs[rng.randint(len(negs))]

        Xp = X[pidx]
        Xn = X[nidx]
        Hp = None if hist_train is None else hist_train[pidx]
        Hn = None if hist_train is None else hist_train[nidx]

        sp, cp = model.forward(Xp, Hp)
        sn, cn = model.forward(Xn, Hn)
        diff = sp - sn
        g = -(1.0 - sigmoid(diff)) / len(pidx)
        model.apply_grads([(cp, g), (cn, -g)], aux_contribs=None)


def predict_in_batches(model, X, H, batch_size=65536):
    out = np.empty(X.shape[0], dtype=np.float32)
    for s in range(0, X.shape[0], batch_size):
        e = min(X.shape[0], s + batch_size)
        Hb = None if H is None else H[s:e]
        out[s:e] = model.predict(X[s:e], Hb).astype(np.float32)
    return out


def choose_rule(valid_user_ids, valid_labels, epoch_valid_scores, mask_a, mask_b):
    candidates = []
    n = len(epoch_valid_scores)
    for i in range(n):
        candidates.append((f'epoch_{i+1}', epoch_valid_scores[i]))
    if n >= 2:
        candidates.append(('avg_last2', np.mean(epoch_valid_scores[-2:], axis=0)))
    if n >= 3:
        candidates.append(('avg_last3', np.mean(epoch_valid_scores[-3:], axis=0)))

    best_name = None
    best_cv = -1e18
    best_scores = None
    details = {}
    for name, scores in candidates:
        ab = metrics_on_mask(valid_user_ids, valid_labels, scores, mask_b)['primary']
        ba = metrics_on_mask(valid_user_ids, valid_labels, scores, mask_a)['primary']
        cv = (ab + ba) / 2.0
        full = primary_from_scores(valid_user_ids, valid_labels, scores)['primary']
        details[name] = {'split_cv_primary': float(cv), 'full_valid_primary': float(full)}
        if cv > best_cv:
            best_cv = cv
            best_name = name
            best_scores = scores
    return best_name, best_scores, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu-choices', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu = json.loads(args.menu_choices)
        seed = int(args.seed)
        rng = np.random.RandomState(seed)

        cfg = {
            'loss': menu.get('loss', 'bpr_pairwise'),
            'neg_sampling': menu.get('neg_sampling', 'uniform_1'),
            'user_history': menu.get('user_history', 'recency_weighted_pool'),
            'multitask': menu.get('multitask', 'none'),
            'model': menu.get('model', 'fm_numpy'),
            'temporal': menu.get('temporal', 'none'),
            'training': menu.get('training', 'lower_lr_longer'),
            'data_extras': menu.get('data_extras', 'none'),
            'sample_weighting': menu.get('sample_weighting', 'per_row'),
            'regularization': menu.get('regularization', 'l2_default'),
            'lr': float(menu.get('lr', 5e-4)),
            'epochs': int(menu.get('epochs', 12)),
            'patience': int(menu.get('patience', 4)),
            'k': int(menu.get('k', 16)),
            'l2': float(menu.get('l2', 1e-6)),
            'bs': int(menu.get('bs', 8192)),
            'hist_tau_days': float(menu.get('hist_tau_days', 7.0)),
        }

        if cfg['loss'] != 'bpr_pairwise' or cfg['model'] != 'fm_numpy':
            raise ValueError('This custom confirmation script is implemented only for fm_numpy + bpr_pairwise.')

        splits, meta = train_lib.load_cache()
        enc, dim, offsets, dims = train_lib.encode_features(splits, meta, cfg['temporal'])

        history_mode = cfg['user_history']
        history_obj = None
        H_train = H_valid = H_test = None
        if history_mode != 'none':
            history_obj = train_lib.History(splits, meta['field_dims']['user'], history_mode)
            H_train = make_history_vectors(history_mode, history_obj, rows=np.arange(len(splits['train']['user'])))
            H_valid = make_history_vectors(history_mode, history_obj, rows=None)
            H_test = make_history_vectors(history_mode, history_obj, rows=None)

        model = train_lib.RankFM(dim=dim, k=cfg['k'], lr=cfg['lr'], seed=seed, aux_tasks=[])
        if hasattr(model, 'l2'):
            model.l2 = cfg['l2']

        Xtr = enc['train']
        ytr = splits['train']['long_view'].astype(np.float32)
        utr = splits['train']['user'].astype(np.int64)
        Xva = enc['valid']
        Xte = enc['test']
        valid_user_ids = splits['valid']['user_raw']
        valid_labels = splits['valid']['long_view']

        best_primary = -1e18
        best_epoch = -1
        best_state = None
        bad = 0
        epoch_valid_scores = []
        epoch_test_scores = []
        state_snapshots = []

        for epoch in range(cfg['epochs']):
            train_epoch_bpr(model, Xtr, ytr, utr, H_train, cfg['bs'], rng)
            s_valid = predict_in_batches(model, Xva, H_valid)
            s_test = predict_in_batches(model, Xte, H_test)
            epoch_valid_scores.append(s_valid)
            epoch_test_scores.append(s_test)
            state_snapshots.append(model.state())
            m = primary_from_scores(valid_user_ids, valid_labels, s_valid)
            if m['primary'] > best_primary:
                best_primary = m['primary']
                best_epoch = epoch
                best_state = model.state()
                bad = 0
            else:
                bad += 1
                if bad >= cfg['patience']:
                    break

        epoch_valid_scores = epoch_valid_scores[:best_epoch + 1 + bad if bad < cfg['patience'] else len(epoch_valid_scores)]
        epoch_test_scores = epoch_test_scores[:len(epoch_valid_scores)]

        mask_a, mask_b = build_user_masks(valid_user_ids)
        chosen_name, chosen_valid_scores, rule_details = choose_rule(
            valid_user_ids,
            valid_labels,
            epoch_valid_scores,
            mask_a,
            mask_b,
        )

        if chosen_name.startswith('epoch_'):
            idx = int(chosen_name.split('_')[1]) - 1
            final_valid = epoch_valid_scores[idx]
            final_test = epoch_test_scores[idx]
        elif chosen_name == 'avg_last2':
            final_valid = np.mean(epoch_valid_scores[-2:], axis=0).astype(np.float32)
            final_test = np.mean(epoch_test_scores[-2:], axis=0).astype(np.float32)
        elif chosen_name == 'avg_last3':
            final_valid = np.mean(epoch_valid_scores[-3:], axis=0).astype(np.float32)
            final_test = np.mean(epoch_test_scores[-3:], axis=0).astype(np.float32)
        else:
            raise RuntimeError('Unknown chosen rule')

        metrics = primary_from_scores(valid_user_ids, valid_labels, final_valid)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), final_test)
        with open(os.path.join(args.output_dir, 'selection_details.json'), 'w') as f:
            json.dump({'chosen_rule': chosen_name, 'rule_details': rule_details}, f)

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
