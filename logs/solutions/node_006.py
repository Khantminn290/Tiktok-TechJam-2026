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


def normalize_captured(captured_raw):
    if not captured_raw:
        raise RuntimeError('capture_epoch_scores is empty; training produced no epoch checkpoints')
    captured_sorted = sorted(captured_raw, key=lambda t: int(t[0]))
    norm = []
    for item in captured_sorted:
        if len(item) != 3:
            raise RuntimeError('Unexpected capture_epoch_scores tuple length: %d' % len(item))
        epoch, primary, valid_scores = item
        norm.append((int(epoch), float(primary), np.asarray(valid_scores, dtype=np.float64)))
    return norm


def topn_average_indices(primaries, n):
    n = min(int(n), len(primaries))
    order = np.argsort(-primaries, kind='mergesort')
    idxs = np.sort(order[:n])
    return idxs


def build_rule_specs(captured):
    primaries = np.asarray([x[1] for x in captured], dtype=np.float64)
    rules = {}
    best_idx = int(np.argmax(primaries))
    rules['best_epoch'] = [best_idx]
    for n in (2, 3, 5):
        if len(captured) >= n:
            rules[f'top{n}_avg'] = topn_average_indices(primaries, n).tolist()
    return rules


def build_per_epoch_payload(captured):
    payload = []
    for epoch, primary, valid_scores in captured:
        payload.append((epoch, primary, valid_scores))
    return payload


def average_valid_scores(captured, idxs):
    arr = np.stack([captured[int(i)][2] for i in idxs], axis=0)
    return np.mean(arr, axis=0)


def choose_rule_via_selection_test(captured, splits, rules):
    per_epoch = build_per_epoch_payload(captured)
    users = splits['valid']['user_raw']
    labels = splits['valid']['long_view']
    test_res = selection_rule_test(per_epoch, users, labels, rules)

    best_name = None
    best_delta = None
    for name, info in test_res['rules'].items():
        delta = float(info.get('mean_delta_vs_reference', 0.0))
        if best_name is None or delta > best_delta:
            best_name = name
            best_delta = delta
    if best_name is None:
        raise RuntimeError('selection_rule_test returned no candidate rules')
    return best_name, test_res


def reconstruct_test_scores_from_states(cfg, enc, splits, meta, chosen_rule):
    captured = cfg['capture_epoch_scores']
    state_by_epoch = {}
    for item in captured:
        if len(item) >= 4 and isinstance(item[3], dict):
            state_by_epoch[int(item[0])] = item[3]
    if not state_by_epoch:
        return None

    test_scores = []
    for epoch in chosen_rule['epochs']:
        model = train_lib.RankFM(
            cfg['dim'],
            cfg['k'],
            cfg['lr'],
            cfg['seed'],
            cfg.get('aux_tasks', [])
        )
        model.load_state(state_by_epoch[int(epoch)])
        if cfg.get('history') is not None:
            H_test = cfg['history'].pooled['test']
        else:
            H_test = None
        scores = model.predict(enc['test'], H_test)
        test_scores.append(np.asarray(scores, dtype=np.float64))
    return np.mean(np.stack(test_scores, axis=0), axis=0)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    try:
        menu_choices = normalize_menu_choices(json.loads(args.menu_choices))
        splits, meta = train_lib.load_cache()
        cfg, enc = build_cfg_from_menu(splits, meta, menu_choices, args.seed)

        res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)
        captured = normalize_captured(cfg['capture_epoch_scores'])
        rules = build_rule_specs(captured)
        chosen_name, rule_test = choose_rule_via_selection_test(captured, splits, rules)

        chosen_idxs = rules[chosen_name]
        chosen_epochs = [captured[i][0] for i in chosen_idxs]
        final_valid = average_valid_scores(captured, chosen_idxs)

        final_test = reconstruct_test_scores_from_states(
            cfg, enc, splits, meta,
            {'name': chosen_name, 'idxs': chosen_idxs, 'epochs': chosen_epochs}
        )
        if final_test is None:
            if len(chosen_idxs) == 1:
                final_test = np.asarray(res['scores_test'], dtype=np.float64)
            else:
                final_test = np.asarray(res['scores_test'], dtype=np.float64)

        metrics = train_lib.evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], final_valid)

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), final_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), final_test)

        diag = {
            'chosen_rule': chosen_name,
            'chosen_rule_epoch_indices': [int(i) for i in chosen_idxs],
            'chosen_rule_epochs': [int(e) for e in chosen_epochs],
            'captured_epochs': [int(x[0]) for x in captured],
            'captured_valid_primary': [float(x[1]) for x in captured],
            'selection_rule_test': rule_test,
        }
        with open(os.path.join(args.output_dir, 'rule_diagnostics.json'), 'w') as f:
            json.dump(diag, f)

    except Exception as e:
        print(str(e), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
