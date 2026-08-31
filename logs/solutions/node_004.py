import argparse
import json
import os
import sys

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
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    existing = json.load(f)
                with open(metrics_path, 'w') as f:
                    json.dump({k: float(v) for k, v in existing.items()}, f)
            except Exception:
                with open(metrics_path, 'w') as f:
                    json.dump({k: float(v) for k, v in metrics.items()}, f)
        else:
            with open(metrics_path, 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f)
    except Exception as e:
        print(str(e), file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
