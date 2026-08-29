import argparse
import json
import os
import traceback

import numpy as np
import train_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu-choices', type=str, required=True)
    ap.add_argument('--output-dir', type=str, required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu_choices = json.loads(args.menu_choices)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        metrics = {k: float(v) for k, v in metrics.items()}
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
