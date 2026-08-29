import argparse
import json
import os
import sys
import traceback
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_lib


class DeepFMNet(nn.Module):
    def __init__(self, field_dims: List[int], emb_dim: int, mlp_hidden: Tuple[int, ...] = (128, 64), aux_tasks: List[str] = None):
        super().__init__()
        self.field_dims = list(field_dims)
        self.num_fields = len(field_dims)
        self.emb_dim = emb_dim
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:-1]), dtype=np.int64)
        self.total_dim = int(np.sum(self.field_dims))

        self.first_order = nn.Embedding(self.total_dim, 1)
        self.embed = nn.Embedding(self.total_dim, emb_dim)

        in_dim = self.num_fields * emb_dim
        layers = []
        prev = in_dim
        for h in mlp_hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.main_head = nn.Linear(prev, 1)

        self.aux_tasks = aux_tasks or []
        self.aux_map = {t: f"task_{t}" for t in self.aux_tasks}
        self.aux_heads = nn.ModuleDict({self.aux_map[t]: nn.Linear(prev, 1) for t in self.aux_tasks})

        nn.init.xavier_uniform_(self.first_order.weight)
        nn.init.xavier_uniform_(self.embed.weight)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.main_head.weight)
        nn.init.zeros_(self.main_head.bias)
        for head in self.aux_heads.values():
            nn.init.xavier_uniform_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor, hist: torch.Tensor = None):
        first = self.first_order(x).sum(dim=1)
        emb = self.embed(x)
        if hist is not None:
            emb = torch.cat([emb, hist.unsqueeze(1)], dim=1)
        sum_emb = emb.sum(dim=1)
        sum_sq = sum_emb * sum_emb
        sq_sum = (emb * emb).sum(dim=1)
        fm_second = 0.5 * (sum_sq - sq_sum).sum(dim=1, keepdim=True)
        deep_in = emb.reshape(emb.size(0), -1)
        deep_feat = self.mlp(deep_in)
        main_logit = (first + fm_second + self.main_head(deep_feat)).squeeze(1)
        aux_logits = {t: self.aux_heads[self.aux_map[t]](deep_feat).squeeze(1) for t in self.aux_tasks}
        return main_logit, aux_logits


def make_history_vectors(splits, meta, mode: str):
    n_users = meta['field_dims']['user']
    hist = train_lib.History(splits, n_users, mode)
    train_H = hist.batch_vectors('train')
    valid_H = hist.pooled('valid')
    test_H = hist.pooled('test')
    return train_H.astype(np.float32), valid_H.astype(np.float32), test_H.astype(np.float32)


def user_group_index(user_ids: np.ndarray) -> Dict[int, np.ndarray]:
    groups = {}
    order = np.argsort(user_ids, kind='mergesort')
    sorted_users = user_ids[order]
    uniq, starts = np.unique(sorted_users, return_index=True)
    starts = list(starts) + [len(order)]
    for i, u in enumerate(uniq):
        groups[int(u)] = order[starts[i]:starts[i + 1]]
    return groups


def build_pair_arrays(users: np.ndarray, labels: np.ndarray, max_pairs_per_user: int, rng: np.random.Generator):
    groups = user_group_index(users)
    pos_list = []
    neg_list = []
    for _, idx in groups.items():
        y = labels[idx]
        pos = idx[y > 0]
        neg = idx[y <= 0]
        if len(pos) == 0 or len(neg) == 0:
            continue
        n_pairs = min(max_pairs_per_user, len(pos) * len(neg))
        pos_samp = pos[rng.integers(0, len(pos), size=n_pairs)]
        neg_samp = neg[rng.integers(0, len(neg), size=n_pairs)]
        pos_list.append(pos_samp)
        neg_list.append(neg_samp)
    if not pos_list:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos_idx = np.concatenate(pos_list).astype(np.int64)
    neg_idx = np.concatenate(neg_list).astype(np.int64)
    perm = rng.permutation(len(pos_idx))
    return pos_idx[perm], neg_idx[perm]


def evaluate_split(net, X, H, batch_size, device, user_raw, labels):
    net.eval()
    scores = np.zeros(len(X), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            e = min(len(X), s + batch_size)
            xb = torch.from_numpy(X[s:e]).to(device=device, dtype=torch.long)
            hb = torch.from_numpy(H[s:e]).to(device=device, dtype=torch.float32)
            logit, _ = net(xb, hb)
            scores[s:e] = logit.cpu().numpy().astype(np.float32)
    metrics = train_lib.evaluate(user_raw, labels, scores)
    return {k: float(v) for k, v in metrics.items()}, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu_choices = json.loads(args.menu_choices)
        expected = {
            'loss': 'bpr_pairwise',
            'user_history': 'recency_weighted_pool',
            'multitask': 'aux_click_like_forward',
            'model': 'deepfm_mlp',
            'temporal': 'hour_plus_dow',
            'training': 'default',
            'data_extras': 'none',
        }
        for k, v in expected.items():
            if menu_choices.get(k) != v:
                raise ValueError(f'This debug script is specialized for {expected}, got {menu_choices}')

        rng = np.random.default_rng(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        splits, meta = train_lib.load_cache()
        enc, dim, offsets, dims = train_lib.encode_features(splits, meta, temporal='hour_plus_dow')

        train_H, valid_H, test_H = make_history_vectors(splits, meta, mode='recency_weighted_pool')
        hist_dim = train_H.shape[1]

        field_dims = list(dims) + [hist_dim]
        net = DeepFMNet(field_dims=field_dims, emb_dim=16, mlp_hidden=(128, 64), aux_tasks=['is_click', 'is_like', 'is_forward'])
        device = torch.device('cpu')
        net.to(device)

        X_train = enc['train'].astype(np.int64)
        X_valid = enc['valid'].astype(np.int64)
        X_test = enc['test'].astype(np.int64)

        y_train = splits['train']['long_view'].astype(np.float32)
        y_valid = splits['valid']['long_view'].astype(np.float32)

        click_train = splits['train']['is_click'].astype(np.float32)
        like_train = splits['train']['is_like'].astype(np.float32)
        fwd_train = splits['train']['is_forward'].astype(np.float32)

        users_train = splits['train']['user'].astype(np.int64)

        pos_idx, neg_idx = build_pair_arrays(users_train, y_train, max_pairs_per_user=32, rng=rng)
        if len(pos_idx) == 0:
            raise RuntimeError('No valid BPR pairs constructed from training split.')

        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        batch_size = 4096
        eval_batch_size = 8192
        max_epochs = 12
        patience = 3
        aux_w = {'is_click': 0.15, 'is_like': 0.10, 'is_forward': 0.10}

        best_primary = -1e18
        best_state = None
        bad_epochs = 0

        for epoch in range(max_epochs):
            net.train()
            order = rng.permutation(len(pos_idx))
            pos_idx_e = pos_idx[order]
            neg_idx_e = neg_idx[order]
            total_loss = 0.0
            total_n = 0

            for s in range(0, len(pos_idx_e), batch_size):
                e = min(len(pos_idx_e), s + batch_size)
                p = pos_idx_e[s:e]
                n = neg_idx_e[s:e]

                xb_p = torch.from_numpy(X_train[p]).to(device=device, dtype=torch.long)
                hb_p = torch.from_numpy(train_H[p]).to(device=device, dtype=torch.float32)
                xb_n = torch.from_numpy(X_train[n]).to(device=device, dtype=torch.long)
                hb_n = torch.from_numpy(train_H[n]).to(device=device, dtype=torch.float32)

                logit_p, aux_p = net(xb_p, hb_p)
                logit_n, _ = net(xb_n, hb_n)
                bpr_loss = F.softplus(-(logit_p - logit_n)).mean()

                aux_loss = 0.0
                targets = {
                    'is_click': torch.from_numpy(click_train[p]).to(device=device, dtype=torch.float32),
                    'is_like': torch.from_numpy(like_train[p]).to(device=device, dtype=torch.float32),
                    'is_forward': torch.from_numpy(fwd_train[p]).to(device=device, dtype=torch.float32),
                }
                for t, w in aux_w.items():
                    aux_loss = aux_loss + w * F.binary_cross_entropy_with_logits(aux_p[t], targets[t])

                loss = bpr_loss + aux_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * (e - s)
                total_n += (e - s)

            valid_metrics, _ = evaluate_split(
                net, X_valid, valid_H, eval_batch_size, device,
                splits['valid']['user_raw'], splits['valid']['long_view']
            )
            primary = float(valid_metrics['primary'])

            if primary > best_primary:
                best_primary = primary
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

        if best_state is None:
            raise RuntimeError('Training did not produce any checkpoint.')
        net.load_state_dict(best_state)

        valid_metrics, valid_scores = evaluate_split(
            net, X_valid, valid_H, eval_batch_size, device,
            splits['valid']['user_raw'], splits['valid']['long_view']
        )
        _, test_scores = evaluate_split(
            net, X_test, test_H, eval_batch_size, device,
            splits['test']['user_raw'], np.zeros(len(X_test), dtype=np.float32)
        )

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump({k: float(v) for k, v in valid_metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), valid_scores)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), test_scores)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
