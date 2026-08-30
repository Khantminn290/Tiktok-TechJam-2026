import argparse
import json
import os
import sys

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        os.makedirs(args.output_dir, exist_ok=True)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        metrics = {k: float(v) for k, v in metrics.items()}
        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        raise


if __name__ == '__main__':
    sys.exit(main())
