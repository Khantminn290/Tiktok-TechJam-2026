import argparse
import json
import os
import sys
import traceback

import numpy as np

import train_lib
from research_tools import incumbent_cfg, capture_selection_rule_test
from evaluate import evaluate


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu-choices', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    return ap.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def contiguous_average(score_list, center_idx, width):
    n = len(score_list)
    if width <= 1 or n == 1:
        return score_list[center_idx].copy()
    half = width // 2
    start = center_idx - half
    end = start + width
    if start < 0:
        start = 0
        end = min(width, n)
    if end > n:
        end = n
        start = max(0, end - width)
    arr = np.mean(np.stack(score_list[start:end], axis=0), axis=0)
    return arr.astype(np.float32, copy=False)


def build_rule_scores(epoch_records):
    epochs = [int(t[0]) for t in epoch_records]
    primaries = [float(t[1]) for t in epoch_records]
    valid_scores = [np.asarray(t[2], dtype=np.float32) for t in epoch_records]
    best_idx = int(np.argmax(np.asarray(primaries)))

    rules = {
        'best_epoch': valid_scores[best_idx],
        'avg_top2_contig': contiguous_average(valid_scores, best_idx, 2),
        'avg_top3_contig': contiguous_average(valid_scores, best_idx, 3),
        'avg_top5_contig': contiguous_average(valid_scores, best_idx, 5),
    }
    meta = {
        'epochs': epochs,
        'primaries': primaries,
        'best_epoch_idx': best_idx,
        'best_epoch_number': epochs[best_idx],
    }
    return rules, meta


def choose_rule_with_fallback(valid_user_ids, valid_labels, rule_scores):
    selected = 'best_epoch'
    diagnostic = None
    try:
        diagnostic = capture_selection_rule_test(valid_user_ids, valid_labels, rule_scores)
        if isinstance(diagnostic, dict) and 'rules' in diagnostic and diagnostic['rules']:
            best_name = None
            best_value = -1e18
            for name, info in diagnostic['rules'].items():
                value = None
                if isinstance(info, dict):
                    for k in ['mean', 'score', 'primary', 'heldout_primary', 'avg_primary']:
                        if k in info:
                            value = float(info[k])
                            break
                if value is not None and value > best_value:
                    best_value = value
                    best_name = name
            if best_name in rule_scores:
                selected = best_name
    except Exception as e:
        diagnostic = {'warning': 'selection_rule_test_failed', 'error': str(e)}
    return selected, diagnostic


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    try:
        _ = json.loads(args.menu_choices)

        splits, meta = train_lib.load_cache()
        cfg, enc = incumbent_cfg(splits, meta)
        cfg['seed'] = int(args.seed)
        cfg['capture_epoch_scores'] = []

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

        epoch_records = cfg['capture_epoch_scores']
        if not epoch_records:
            raise RuntimeError('No epoch scores were captured; cannot run checkpoint-rule selection test.')

        rule_scores, rule_meta = build_rule_scores(epoch_records)

        valid_user_ids = splits['valid']['user_raw']
        valid_labels = splits['valid']['long_view']
        selected_rule, selection_diag = choose_rule_with_fallback(valid_user_ids, valid_labels, rule_scores)

        final_valid_scores = np.asarray(rule_scores[selected_rule], dtype=np.float32)
        final_test_scores = np.asarray(res['scores_test'], dtype=np.float32)

        metrics = evaluate(valid_user_ids, valid_labels, final_valid_scores)
        metrics = {k: float(v) for k, v in metrics.items()}

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as fh:
            json.dump(metrics, fh)

        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_valid_scores)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), final_test_scores)

        debug = {
            'selected_rule': selected_rule,
            'rule_meta': rule_meta,
            'selection_diagnostic': selection_diag,
            'note': 'scores_test.npy is the blind test output returned by train_numpy_fm; per-epoch test checkpoints are not exposed by train_lib, so rule selection is applied directly only to validation in this experiment.'
        }
        with open(os.path.join(args.output_dir, 'debug_selection.json'), 'w') as fh:
            json.dump(debug, fh)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
