import argparse
import json
import os
import sys

import numpy as np

import train_lib
from research_tools import incumbent_cfg, capture_selection_rule_test, selection_pressure


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu_choices = json.loads(args.menu_choices)
        splits, meta = train_lib.load_cache()

        cfg, enc = incumbent_cfg(splits, meta,
                                 loss='bpr_pairwise',
                                 neg_sampling='uniform_1',
                                 user_history='none',
                                 multitask='none',
                                 model='gru4rec_seq',
                                 temporal='none',
                                 training='lower_lr_longer',
                                 data_extras='none',
                                 sample_weighting='per_row',
                                 regularization='l2_default')

        for k, v in menu_choices.items():
            cfg[k] = v
        cfg['seed'] = args.seed
        cfg['capture_epoch_scores'] = []

        res = train_lib.train_torch(cfg, enc, splits, meta, print)
        scores_valid = res['scores_valid']
        scores_test = res['scores_test']

        metrics = train_lib.evaluate(splits['valid']['user_raw'], splits['valid']['long_view'], scores_valid)

        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        np.save(os.path.join(args.output_dir, 'scores_valid.npy'), scores_valid)
        np.save(os.path.join(args.output_dir, 'scores_test.npy'), scores_test)

        epoch_scores = cfg['capture_epoch_scores']
        diag = {
            'n_epochs_captured': int(len(epoch_scores)),
            'epochs': [int(t[0]) for t in epoch_scores],
            'valid_primary_by_epoch': [float(t[1]) for t in epoch_scores],
        }

        if len(epoch_scores) >= 2:
            pred_matrix = np.stack([t[2] for t in epoch_scores], axis=0)
            try:
                srt = capture_selection_rule_test(
                    user_ids=splits['valid']['user_raw'],
                    labels=splits['valid']['long_view'],
                    epoch_predictions=pred_matrix,
                    epoch_numbers=[int(t[0]) for t in epoch_scores],
                )
                diag['selection_rule_test'] = srt
            except Exception as e:
                diag['selection_rule_test_error'] = str(e)

            try:
                sp = selection_pressure(n=len(epoch_scores))
                diag['selection_pressure'] = sp
            except Exception as e:
                diag['selection_pressure_error'] = str(e)

        with open(os.path.join(args.output_dir, 'diagnostics.json'), 'w') as f:
            json.dump(diag, f)

        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        raise


if __name__ == '__main__':
    raise SystemExit(main())
