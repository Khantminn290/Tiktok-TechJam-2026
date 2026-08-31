import argparse
import json
import os
import sys
import traceback

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


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    menu = json.loads(args.menu_choices)

    splits, meta = train_lib.load_cache()
    cfg, enc = incumbent_cfg(splits, meta)

    for k, v in menu.items():
        cfg[k] = v
    cfg['seed'] = args.seed
    cfg['capture_epoch_scores'] = []

    res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

    epoch_records = cfg['capture_epoch_scores']
    if len(epoch_records) == 0:
        raise RuntimeError('No epoch predictions captured; capture_epoch_scores is empty')

    epoch_to_scores = {int(epoch): scores for epoch, valid_primary, scores in epoch_records}
    available_epochs = sorted(epoch_to_scores.keys())

    desired = [2, 3, 4]
    chosen = [e for e in desired if e in epoch_to_scores]
    if not chosen:
        if len(available_epochs) >= 3:
            chosen = available_epochs[:3]
        else:
            chosen = available_epochs

    valid_stack = np.stack([np.asarray(epoch_to_scores[e], dtype=np.float64) for e in chosen], axis=0)
    scores_valid = valid_stack.mean(axis=0)

    labels_valid = splits['valid']['long_view']
    users_valid = splits['valid']['user_raw']
    metrics = evaluate(users_valid, labels_valid, scores_valid)

    scores_test = np.asarray(res['scores_test'])

    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)
    np.save(os.path.join(args.output_dir, 'scores_valid.npy'), scores_valid)
    np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('ERROR:', str(e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
