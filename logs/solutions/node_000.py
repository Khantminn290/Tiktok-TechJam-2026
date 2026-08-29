"""Reference solution script — the interface every generated solution must follow.

Usage:  python3 seed_solution.py --menu-choices '<json>' --output-dir <path> [--seed 0]

Contract (enforced by the harness):
  * on success: write to --output-dir
      metrics.json        {"GAUC": float, "nDCG@5": float, "primary": float}  (VALID split)
      scores_valid.npy    float array, one score per valid row, data.load() row order
      scores_test.npy     float array, one score per test row,  data.load() row order
    and exit 0.
  * on failure: exit non-zero with a readable traceback on stderr.
  * never read test labels or compute test metrics (hidden-test discipline).
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_lib  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--menu-choices", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    choices = json.loads(a.menu_choices)
    metrics = train_lib.run(choices, a.output_dir, seed=a.seed)
    print(json.dumps(metrics))


if __name__ == "__main__":
    main()
