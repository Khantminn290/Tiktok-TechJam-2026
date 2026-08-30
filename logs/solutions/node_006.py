import argparse
import json
import os
import sys
import numpy as np
import train_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--menu-choices', type=str, required=True)
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    try:
        menu_choices = json.loads(args.menu_choices)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)
        if metrics is not None:
            metrics_path = os.path.join(args.output_dir, 'metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'w') as f:
                    json.dump({k: float(v) for k, v in metrics.items()}, f)
        return 0
    except Exception as e:
        sys.stderr.write(str(e) + '\n')
        return 1


if __name__ == '__main__':
    sys.exit(main())
