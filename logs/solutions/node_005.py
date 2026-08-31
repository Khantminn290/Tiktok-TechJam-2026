import argparse
import json
import os
import sys

import numpy as np

import train_lib
from research_tools import incumbent_cfg, selection_rule_test


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def normalize_menu_choices(menu_choices):
    if not isinstance(menu_choices, dict):
        raise ValueError('menu_choices must decode to a JSON object')
    return menu_choices


def build_cfg_from_menu(splits, meta, menu_choices, seed):
    cfg, enc = incumbent_cfg(splits, meta)
    for k, v in menu_choices.items():
        cfg[k] = v
    cfg['seed'] = seed
    cfg['capture_epoch_scores'] = []
    return cfg, enc


def sort_epochs(captured):
    if not captured:
        raise RuntimeError('No epoch scores were captured')
    captured_sorted = sorted(captured, key=lambda t: int(t[0]))
    epochs = [int(t[0]) for t in captured_sorted]
    primaries = np.asarray([float(t[1]) for t in captured_sorted], dtype=np.float64)
    valid_scores = [np.asarray(t[2], dtype=np.float64) for t in captured_sorted]
    return epochs, primaries, valid_scores


def topn_average_indices(primaries, n):
    n = min(int(n), len(primaries))
    order = np.argsort(-primaries, kind='mergesort')
    top = np.sort(order[:n])
    return top


def make_rule_predictions(epochs, primaries, valid_scores):
    rules = {}

    best_idx = int(np.argmax(primaries))
    rules['best_epoch'] = {
        'valid_scores': valid_scores[best_idx],
        'epoch_indices': np.asarray([best_idx], dtype=np.int64),
        'epochs': [epochs[best_idx]],
        'description': 'Single epoch with highest full-valid primary'
    }

    for n in (2, 3, 5):
        if len(valid_scores) >= n:
            idxs = topn_average_indices(primaries, n)
            avg_scores = np.mean(np.stack([valid_scores[i] for i in idxs], axis=0), axis=0)
            rules[f'top{n}_avg'] = {
                'valid_scores': avg_scores,
                'epoch_indices': idxs,
                'epochs': [epochs[i] for i in idxs.tolist()],
                'description': f'Average VALID scores from top-{n} epochs by full-valid primary'
            }
    return rules


def choose_rule_via_selection_test(splits, rules):
    candidate_scores = {name: payload['valid_scores'] for name, payload in rules.items()}
    ref = 'best_epoch' if 'best_epoch' in candidate_scores else list(candidate_scores.keys())[0]
    test_res = selection_rule_test(
        user_ids=splits['valid']['user_raw'],
        labels=splits['valid']['long_view'],
        candidate_scores=candidate_scores,
        reference_rule=ref,
    )

    winners = []
    mean_deltas = {}
    for name, info in test_res['rules'].items():
        delta = float(info.get('mean_delta_vs_reference', 0.0))
        mean_deltas[name] = delta
    best_name = max(mean_deltas.items(), key=lambda kv: kv[1])[0]
    return best_name, test_res


def reconstruct_test_rule(captured, chosen_epoch_indices):
    if len(chosen_epoch_indices) == 1:
        idx = int(chosen_epoch_indices[0])
        return np.asarray(captured[idx][3], dtype=np.float64)
    arr = np.stack([np.asarray(captured[int(i)][3], dtype=np.float64) for i in chosen_epoch_indices], axis=0)
    return np.mean(arr, axis=0)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    try:
        menu_choices = normalize_menu_choices(json.loads(args.menu_choices))
        splits, meta = train_lib.load_cache()
        cfg, enc = build_cfg_from_menu(splits, meta, menu_choices, args.seed)

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)
        _ = res  # training side effects populate capture_epoch_scores

        captured_raw = cfg['capture_epoch_scores']
        if not captured_raw:
            raise RuntimeError('capture_epoch_scores is empty; training produced no epoch checkpoints')

        captured_sorted = sorted(captured_raw, key=lambda t: int(t[0]))
        normalized_captured = []
        for item in captured_sorted:
            if len(item) == 3:
                epoch, primary, valid_scores = item
                test_scores = np.asarray(res['scores_test'], dtype=np.float64)
            elif len(item) == 4:
                epoch, primary, valid_scores, test_scores = item
            else:
                raise RuntimeError('Unexpected capture_epoch_scores tuple length: %d' % len(item))
            normalized_captured.append((int(epoch), float(primary), np.asarray(valid_scores, dtype=np.float64), np.asarray(test_scores, dtype=np.float64)))

        epochs = [x[0] for x in normalized_captured]
        primaries = np.asarray([x[1] for x in normalized_captured], dtype=np.float64)
        valid_scores = [x[2] for x in normalized_captured]

        rules = make_rule_predictions(epochs, primaries, valid_scores)
        chosen_rule, rule_test = choose_rule_via_selection_test(splits, rules)

        final_valid = np.asarray(rules[chosen_rule]['valid_scores'], dtype=np.float64)
        final_test = reconstruct_test_rule(normalized_captured, rules[chosen_rule]['epoch_indices'])

        metrics = train_lib.evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], final_valid)

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), final_test)

        with open(os.path.join(args.output_dir, 'rule_diagnostics.json'), 'w') as f:
            json.dump({
                'chosen_rule': chosen_rule,
                'rule_epochs': rules[chosen_rule]['epochs'],
                'captured_epochs': epochs,
                'captured_valid_primary': [float(x) for x in primaries.tolist()],
                'selection_rule_test': rule_test,
            }, f)

    except Exception as e:
        print(str(e), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
