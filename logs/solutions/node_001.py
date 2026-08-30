import argparse
import json
import os
import sys

import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_cfg_from_menu(menu_choices, splits, meta, seed):
    # Start from the known incumbent recipe, then allow explicit overrides from menu_choices.
    cfg, enc = incumbent_cfg(splits, meta)

    defaults = {
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
    }
    for k, v in defaults.items():
        cfg[k] = menu_choices.get(k, v)

    # Pipeline overrides if provided.
    override_keys = [
        'aux_weight', 'bs', 'checkpoint_combine', 'epochs', 'hist_tau_days',
        'k', 'l2', 'lr', 'n_checkpoints', 'patience'
    ]
    for k in override_keys:
        if k in menu_choices:
            cfg[k] = menu_choices[k]

    cfg['seed'] = seed
    cfg['capture_epoch_scores'] = []
    return cfg, enc


def mean_of_indices(arrays, idxs):
    sel = [arrays[i] for i in idxs]
    out = np.zeros_like(sel[0], dtype=np.float64)
    for a in sel:
        out += a.astype(np.float64)
    out /= float(len(sel))
    return out


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    try:
        menu_choices = json.loads(args.menu_choices)
        splits, meta = train_lib.load_cache()
        cfg, enc = build_cfg_from_menu(menu_choices, splits, meta, args.seed)

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)
        scores_test = res['scores_test']

        epoch_records = cfg['capture_epoch_scores']
        if not epoch_records:
            raise RuntimeError('No epoch records captured; capture_epoch_scores is empty.')

        epochs = []
        val_primaries = []
        epoch_valid_scores = []
        for rec in epoch_records:
            if len(rec) != 3:
                raise RuntimeError('capture_epoch_scores entry must have length 3.')
            ep, vp, sv = rec
            epochs.append(int(ep))
            val_primaries.append(float(vp))
            epoch_valid_scores.append(np.asarray(sv))

        order = np.argsort(np.asarray(val_primaries))[::-1]
        best_idx = int(order[0])

        candidates = {}
        candidates['argmax_epoch'] = epoch_valid_scores[best_idx]

        max_k = min(5, len(order))
        for k in range(2, max_k + 1):
            idxs = order[:k].tolist()
            candidates[f'top{k}_mean'] = mean_of_indices(epoch_valid_scores, idxs)

        user_ids = splits['valid']['user_raw']
        labels = splits['valid']['long_view']

        candidate_metrics = {}
        for name, sc in candidates.items():
            m = evaluate(user_ids, labels, sc)
            candidate_metrics[name] = {kk: float(vv) for kk, vv in m.items()}

        rule_test = selection_rule_test(user_ids, labels, candidates)

        chosen_name = 'argmax_epoch'
        if isinstance(rule_test, dict) and 'rules' in rule_test:
            ref = rule_test.get('reference_rule', 'argmax_epoch')
            ref_delta = 0.0
            if ref in rule_test['rules'] and isinstance(rule_test['rules'][ref], dict):
                ref_delta = float(rule_test['rules'][ref].get('mean_delta', 0.0))
            best_delta = ref_delta
            for name, info in rule_test['rules'].items():
                if not isinstance(info, dict):
                    continue
                delta = float(info.get('mean_delta', -1e18))
                if delta > best_delta:
                    best_delta = delta
                    chosen_name = name
            if chosen_name not in candidates:
                chosen_name = 'argmax_epoch'

        final_valid = candidates[chosen_name]
        final_metrics = evaluate(user_ids, labels, final_valid)

        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in final_metrics.items()}, f)

        diagnostics = {
            'epochs': epochs,
            'val_primaries': [float(x) for x in val_primaries],
            'candidate_metrics': candidate_metrics,
            'selection_rule_test': rule_test,
            'chosen_rule': chosen_name,
            'menu_choices': menu_choices,
        }
        with open(os.path.join(args.output_dir, 'diagnostics.json'), 'w') as f:
            json.dump(diagnostics, f)

    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        raise


if __name__ == '__main__':
    main()
