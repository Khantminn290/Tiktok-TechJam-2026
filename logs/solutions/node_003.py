import argparse
import json
import os

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu-choices', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    menu_choices = json.loads(args.menu_choices)
    metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f)


if __name__ == '__main__':
    main()
