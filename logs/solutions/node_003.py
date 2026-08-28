import argparse
import json
import os
import sys
import traceback

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    parser.add_argument("--output-dir", required=True, help="Directory to write metrics.json and score npy files")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        os.makedirs(args.output_dir, exist_ok=True)
        train_lib.run(menu_choices, args.output_dir, seed=args.seed)
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
