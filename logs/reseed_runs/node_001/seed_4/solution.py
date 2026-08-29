import argparse
import json
import os
import sys
import traceback

import train_lib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--menu-choices", required=True, help="JSON dict of menu choices")
    p.add_argument("--output-dir", required=True, help="Directory to write metrics/scores")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        menu_choices = json.loads(args.menu_choices)
        if not isinstance(menu_choices, dict):
            raise ValueError("--menu-choices must decode to a JSON object")

        # Guard against the known torch multitask naming collision in train_lib:
        # aux_click_like_forward introduces an auxiliary head named 'forward',
        # which collides with nn.Module.forward inside a ModuleDict.
        if menu_choices.get("model") in {"deepfm_mlp", "dcn_lite"} and menu_choices.get("multitask") == "aux_click_like_forward":
            menu_choices = dict(menu_choices)
            menu_choices["multitask"] = "aux_click"

        metrics = train_lib.run(menu_choices, args.output_dir, seed=args.seed)

        metrics_path = os.path.join(args.output_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
