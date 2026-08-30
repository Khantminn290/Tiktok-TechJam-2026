import argparse
import json
import os
import sys
import numpy as np

import train_lib
from evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def build_cfg_from_menu(menu):
    cfg = dict(menu)
    training = cfg.get('training', 'default')
    if training == 'default':
        cfg.setdefault('k', 16)
        cfg.setdefault('lr', 1e-3)
        cfg.setdefault('bs', 8192)
        cfg.setdefault('epochs', 40)
        cfg.setdefault('patience', 4)
    elif training == 'k32':
        cfg.setdefault('k', 32)
        cfg.setdefault('lr', 1e-3)
        cfg.setdefault('bs', 8192)
        cfg.setdefault('epochs', 40)
        cfg.setdefault('patience', 4)
    elif training == 'lower_lr_longer':
        cfg.setdefault('k', 16)
        cfg.setdefault('lr', 5e-4)
        cfg.setdefault('bs', 8192)
        cfg.setdefault('epochs', 12)
        cfg.setdefault('patience', 4)
    elif training == 'two_stage_finetune':
        cfg.setdefault('k', 16)
        cfg.setdefault('lr', 1e-3)
        cfg.setdefault('bs', 8192)
        cfg.setdefault('epochs', 40)
        cfg.setdefault('patience', 4)
    cfg.setdefault('l2', 1e-6)
    cfg.setdefault('n_checkpoints', 1)
    cfg.setdefault('checkpoint_combine', False)
    return cfg


def user_half_splits(user_ids, n_splits=6, seed=0):
    uniq = np.unique(user_ids)
    rng = np.random.RandomState(seed)
    splits = []
    for i in range(n_splits):
        perm = rng.permutation(len(uniq))
        mid = len(uniq) // 2
        a_users = set(uniq[perm[:mid]].tolist())
        mask_a = np.array([u in a_users for u in user_ids], dtype=bool)
        mask_b = ~mask_a
        if mask_a.any() and mask_b.any():
            splits.append((mask_a, mask_b))
    return splits


def safe_primary(user_ids, labels, scores):
    m = evaluate(user_ids, labels, scores)
    return float(m['primary'])


def analyze_epoch_selection(capture, valid_user_ids, valid_labels, output_dir, seed):
    if not capture:
        return
    epochs = []
    primaries = []
    preds = []
    for item in capture:
        if len(item) < 3:
            continue
        ep, primary, score_vec = item
        epochs.append(int(ep))
        primaries.append(float(primary))
        preds.append(np.asarray(score_vec, dtype=np.float64))
    if len(preds) < 2:
        return

    full_argmax_idx = int(np.argmax(np.asarray(primaries)))
    full_best_metrics = evaluate(valid_user_ids, valid_labels, preds[full_argmax_idx])

    splits = user_half_splits(valid_user_ids, n_splits=6, seed=seed)
    analysis_rows = []
    heldout_best = []
    heldout_top3 = []
    heldout_argmax_full = []

    for split_id, (sel_mask, eval_mask) in enumerate(splits):
        sel_users = valid_user_ids[sel_mask]
        sel_labels = valid_labels[sel_mask]
        eval_users = valid_user_ids[eval_mask]
        eval_labels = valid_labels[eval_mask]

        sel_scores = [safe_primary(sel_users, sel_labels, p[sel_mask]) for p in preds]
        chosen = int(np.argmax(np.asarray(sel_scores)))
        eval_chosen = evaluate(eval_users, eval_labels, preds[chosen][eval_mask])
        eval_full_argmax = evaluate(eval_users, eval_labels, preds[full_argmax_idx][eval_mask])

        topk = min(3, len(preds))
        order = np.argsort(np.asarray(sel_scores))[::-1][:topk]
        avg_pred = np.mean(np.stack([preds[j][eval_mask] for j in order], axis=0), axis=0)
        eval_avg = evaluate(eval_users, eval_labels, avg_pred)

        heldout_best.append(float(eval_chosen['primary']))
        heldout_top3.append(float(eval_avg['primary']))
        heldout_argmax_full.append(float(eval_full_argmax['primary']))

        analysis_rows.append({
            'split': int(split_id),
            'selected_epoch': int(epochs[chosen]),
            'heldout_primary_selected_epoch': float(eval_chosen['primary']),
            'heldout_primary_full_valid_argmax_epoch': float(eval_full_argmax['primary']),
            'heldout_primary_top3_avg_by_selection_half': float(eval_avg['primary'])
        })

        sel_users2 = valid_user_ids[eval_mask]
        sel_labels2 = valid_labels[eval_mask]
        eval_users2 = valid_user_ids[sel_mask]
        eval_labels2 = valid_labels[sel_mask]

        sel_scores2 = [safe_primary(sel_users2, sel_labels2, p[eval_mask]) for p in preds]
        chosen2 = int(np.argmax(np.asarray(sel_scores2)))
        eval_chosen2 = evaluate(eval_users2, eval_labels2, preds[chosen2][sel_mask])
        eval_full_argmax2 = evaluate(eval_users2, eval_labels2, preds[full_argmax_idx][sel_mask])
        order2 = np.argsort(np.asarray(sel_scores2))[::-1][:topk]
        avg_pred2 = np.mean(np.stack([preds[j][sel_mask] for j in order2], axis=0), axis=0)
        eval_avg2 = evaluate(eval_users2, eval_labels2, avg_pred2)

        heldout_best.append(float(eval_chosen2['primary']))
        heldout_top3.append(float(eval_avg2['primary']))
        heldout_argmax_full.append(float(eval_full_argmax2['primary']))

        analysis_rows.append({
            'split': int(split_id) + 1000,
            'selected_epoch': int(epochs[chosen2]),
            'heldout_primary_selected_epoch': float(eval_chosen2['primary']),
            'heldout_primary_full_valid_argmax_epoch': float(eval_full_argmax2['primary']),
            'heldout_primary_top3_avg_by_selection_half': float(eval_avg2['primary'])
        })

    report = {
        'epochs': [int(x) for x in epochs],
        'full_valid_primary_by_epoch': [float(x) for x in primaries],
        'full_valid_argmax_epoch': int(epochs[full_argmax_idx]),
        'full_valid_argmax_metrics': {k: float(v) for k, v in full_best_metrics.items()},
        'heldout_user_split_summary': {
            'mean_primary_selected_epoch': float(np.mean(heldout_best)),
            'mean_primary_full_valid_argmax_epoch': float(np.mean(heldout_argmax_full)),
            'mean_primary_top3_avg': float(np.mean(heldout_top3)),
            'delta_selected_minus_full_argmax': float(np.mean(np.asarray(heldout_best) - np.asarray(heldout_argmax_full))),
            'delta_top3avg_minus_full_argmax': float(np.mean(np.asarray(heldout_top3) - np.asarray(heldout_argmax_full)))
        },
        'heldout_user_split_details': analysis_rows
    }

    with open(os.path.join(output_dir, 'checkpoint_selection_analysis.json'), 'w') as f:
        json.dump(report, f)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    menu = json.loads(args.menu_choices)

    metrics = train_lib.run(menu, args.output_dir, seed=args.seed)

    cfg = build_cfg_from_menu(menu)
    cfg['seed'] = args.seed
    cfg['capture_epoch_scores'] = []

    splits, meta = train_lib.load_cache()
    temporal = menu.get('temporal', 'none')
    enc, dim, offsets, dims = train_lib.encode_features(splits, meta, temporal)

    model_name = menu.get('model', '')
    if model_name == 'fm_numpy':
        train_lib.train_numpy_fm(cfg, enc, splits, meta, log=lambda *a, **k: None)
    elif model_name in ('deepfm_mlp', 'dcn_lite', 'gru4rec_seq'):
        train_lib.train_torch(cfg, enc, splits, meta, log=lambda *a, **k: None)
    else:
        raise ValueError('Unknown model: %s' % model_name)

    valid_user_ids = np.asarray(splits['valid']['user_raw'])
    valid_labels = np.asarray(splits['valid']['long_view'])
    analyze_epoch_selection(cfg['capture_epoch_scores'], valid_user_ids, valid_labels, args.output_dir, args.seed)

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        raise
