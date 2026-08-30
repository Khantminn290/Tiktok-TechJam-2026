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
        if not isinstance(menu_choices, dict):
            raise ValueError('--menu-choices must decode to a JSON object')

        os.makedirs(args.output_dir, exist_ok=True)
        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            with open(metrics_path, 'w') as f:
                json.dump({k: float(v) for k, v in metrics.items()}, f)

        required = [
            os.path.join(args.output_dir, 'metrics.json'),
            os.path.join(args.output_dir, 'scores_valid.npy'),
            os.path.join(args.output_dir, 'scores_test.npy'),
        ]
        missing = [p for p in required if not os.path.exists(p)]
        if missing:
            raise RuntimeError('train_lib.run completed but did not create required outputs: ' + ', '.join(missing))

        return 0
    except Exception as e:
        print(str(e), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
