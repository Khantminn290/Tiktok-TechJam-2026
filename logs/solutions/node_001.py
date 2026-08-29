import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--menu-choices', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        menu_choices = json.loads(args.menu_choices)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


if __name__ == '__main__':
    main()
