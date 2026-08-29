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
        with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
        return 0
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
