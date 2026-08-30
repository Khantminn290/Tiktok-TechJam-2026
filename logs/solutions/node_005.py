import argparse
import json
import os
import sys
import numpy as np

import train_lib
from research_tools import incumbent_cfg
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def write_outputs(output_dir, metrics, scores_valid, scores_test):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)
    np.save(os.path.join(output_dir, 'scores_valid.npy'), np.asarray(scores_valid, dtype=np.float64))
    np.save(os.path.join(output_dir, 'scores_test.npy'), np.asarray(scores_test, dtype=np.float64))


def main():
    args = parse_args()
    try:
        # Parse for CLI contract compatibility; experiment ignores menu overrides by design.
        _ = json.loads(args.menu_choices)

        splits, meta = train_lib.load_cache()

        cfg, enc = incumbent_cfg(
            splits,
            meta,
            loss='bpr_pairwise',
            neg_sampling='uniform_1',
            user_history='recency_weighted_pool',
            multitask='none',
            model='fm_numpy',
            temporal='none',
            training='lower_lr_longer',
            data_extras='none',
            sample_weighting='per_row',
            regularization='l2_default'
        )
        cfg['seed'] = args.seed
        cfg['capture_epoch_scores'] = []

        saved_states = []
        original_apply = train_lib.RankFM.apply_grads

        def apply_and_snapshot(self, grad_pairs, aux_contribs):
            out = original_apply(self, grad_pairs, aux_contribs)
            saved_states.append(self.state())
            return out

        train_lib.RankFM.apply_grads = apply_and_snapshot
        try:
            res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)
        finally:
            train_lib.RankFM.apply_grads = original_apply

        epoch_records = list(cfg['capture_epoch_scores'])
        if not epoch_records:
            raise RuntimeError('capture_epoch_scores returned no epochs')

        n_epochs = len(epoch_records)
        if len(saved_states) < n_epochs:
            raise RuntimeError('captured fewer model states than epoch score snapshots')
        if len(saved_states) > n_epochs:
            saved_states = saved_states[:n_epochs]

        epoch_primaries = np.asarray([float(t[1]) for t in epoch_records], dtype=np.float64)
        epoch_valid_scores = [np.asarray(t[2], dtype=np.float64) for t in epoch_records]

        top_k = min(5, n_epochs)
        top_idxs = np.argsort(epoch_primaries)[::-1][:top_k]
        top_idxs = sorted(int(i) for i in top_idxs.tolist())

        scores_valid = np.mean(np.stack([epoch_valid_scores[i] for i in top_idxs], axis=0), axis=0)

        dim = int(enc['train'].max()) + 1
        model = train_lib.RankFM(
            dim=dim,
            k=cfg['k'],
            lr=cfg['lr'],
            seed=args.seed,
            aux_tasks=cfg.get('aux_tasks', [])
        )

        history = None
        if cfg.get('user_history', 'none') != 'none':
            history = train_lib.History(splits, meta['field_dims']['user'], cfg['user_history'])

        X_test = enc['test']
        H_test = None
        if history is not None:
            H_test = history.pooled['test']

        test_preds = []
        for idx in top_idxs:
            model.load_state(saved_states[idx])
            pred = model.predict(X_test, H_test)
            test_preds.append(np.asarray(pred, dtype=np.float64))
        scores_test = np.mean(np.stack(test_preds, axis=0), axis=0)

        metrics = evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], scores_valid)
        write_outputs(args.output_dir, metrics, scores_valid, scores_test)

    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        raise


if __name__ == '__main__':
    main()
