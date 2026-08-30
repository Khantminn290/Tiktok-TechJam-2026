import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        os.makedirs(args.output_dir, exist_ok=True)
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
