"""Is the incumbent DATA-limited or INFORMATION-limited?

The residual screen established that Pure's feature space is exhausted: every
discarded raw column, given to a gradient-boosted model, adds +0.39 sigma at a
fixed blend weight. That answers "is there more SIGNAL in these columns" -- no.

It does NOT answer a different question: are the embeddings themselves
under-trained? The FM fits ~26k user and ~7.5k video embeddings on 1.14M rows.
If validation is still improving as training rows are added, the model is
limited by ESTIMATION NOISE rather than by the information available, and more
interactions about the SAME users and items would help -- which matters because
KuaiRand-1K and KuaiRand-27K are explicitly permitted training data ("training
must rely only on the KuaiRand datasets listed below") and contain the standard
exposure logs that Pure omits, roughly 230x more rows over the same user set.

This is deliberately the cheap test that decides whether a 9.9 GB download and
a cross-dataset pipeline are worth building:

    curve still rising at 100%  -> data-limited; transfer is worth the cost
    curve flat at 100%          -> information-limited; transfer will not help

Rows are subsampled at random, holding every other choice at the incumbent, and
each fraction is run at several seeds because a single draw at a 0.0008 noise
floor decides nothing.

Usage: python3 -m agent.learning_curve [--fractions 0.25,0.5,1.0] [--seeds 3]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "runtime"))

NOISE = 0.0008
OUT = os.path.join(ROOT, "logs", "learning_curve.json")


def run_fraction(frac: float, seed: int, cfg_choices: dict) -> dict:
    """Train the incumbent on a random `frac` of train rows. Valid is untouched."""
    import train_lib

    splits, meta = train_lib.load_cache()
    tr = splits["train"]
    n = len(tr["user"])
    if frac < 1.0:
        rng = np.random.default_rng(1000 + seed)
        idx = np.sort(rng.choice(n, int(n * frac), replace=False))
        splits = dict(splits)
        splits["train"] = {k: (v[idx] if hasattr(v, "__len__") and len(v) == n else v)
                           for k, v in tr.items()}

    enc, dim, offsets, dims = train_lib.encode_features(
        splits, meta, cfg_choices.get("temporal", "none"))
    training = cfg_choices.get("training", "default")
    cfg = {
        "dim": dim, "k": 32 if training == "k32" else 16,
        "lr": 5e-4 if training == "lower_lr_longer" else 1e-3,
        "bs": 8192,
        "epochs": 60 if training == "lower_lr_longer" else 40,
        "patience": 6 if training == "lower_lr_longer" else 4,
        "seed": seed,
        "loss": cfg_choices.get("loss", "pointwise_logloss"),
        "history": cfg_choices.get("user_history", "none"),
        "multitask": cfg_choices.get("multitask", "none"),
        "model": cfg_choices.get("model", "fm_numpy"),
        "training": training,
        "neg_sampling": cfg_choices.get("neg_sampling", "uniform_1"),
        "sample_weighting": cfg_choices.get("sample_weighting", "per_row"),
        "l2": {"l2_default": 1e-6, "l2_1e5": 1e-5, "l2_1e4": 1e-4,
               "l2_1e3": 1e-3}.get(cfg_choices.get("regularization", "l2_default"), 1e-6),
        "snapshot_ensemble": 0, "bootstrap_seed": None,
        "aux_weight": 0.2, "device": "cpu",
    }
    cfg["aux_tasks"] = train_lib.AUX_MAP[cfg["multitask"]]
    # train_numpy_fm returns raw score arrays, not metrics -- score them with
    # the official evaluator so this curve is measured exactly as the
    # competition measures everything else.
    res = train_lib.train_numpy_fm(cfg, enc, splits, meta, lambda *a, **k: None)
    sys.path.insert(0, os.path.join(ROOT, "kuairand-starter-kit"))
    from evaluate import evaluate
    va = splits["valid"]
    m = evaluate(list(va["user_raw"]), va["long_view"], res["scores_valid"])
    return {k: float(m[k]) for k in ("GAUC", "nDCG@5", "primary")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", default="0.25,0.5,1.0")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()

    cfg_choices = json.load(open(os.path.join(ROOT, "logs",
                                              "ensemble_results.json")))["config"]
    fracs = [float(x) for x in a.fractions.split(",")]
    out = {"config": cfg_choices, "seeds": a.seeds, "points": {}}
    print(f"incumbent config; {a.seeds} seeds per fraction\n")
    for f in fracs:
        vals = []
        for s in range(a.seeds):
            t0 = time.time()
            m = run_fraction(f, s, cfg_choices)
            vals.append(m)
            print(f"  frac {f:.2f} seed {s}  primary {m['primary']:.5f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        p = [v["primary"] for v in vals]
        out["points"][str(f)] = {
            "mean_primary": round(statistics.mean(p), 5),
            "std": round(statistics.pstdev(p), 5),
            "mean_GAUC": round(statistics.mean(v["GAUC"] for v in vals), 5),
            "mean_nDCG@5": round(statistics.mean(v["nDCG@5"] for v in vals), 5),
        }

    keys = sorted(out["points"], key=float)
    print(f"\n{'=' * 66}\nLEARNING CURVE (incumbent config)\n{'=' * 66}")
    print(f"  {'train frac':<12}{'primary':>10}{'std':>10}{'GAUC':>10}{'nDCG@5':>10}")
    for k in keys:
        d = out["points"][k]
        print(f"  {k:<12}{d['mean_primary']:>10.5f}{d['std']:>10.5f}"
              f"{d['mean_GAUC']:>10.5f}{d['mean_nDCG@5']:>10.5f}")
    lo, hi = out["points"][keys[-2]], out["points"][keys[-1]]
    slope = hi["mean_primary"] - lo["mean_primary"]
    out["final_slope"] = round(slope, 5)
    out["final_slope_sigma"] = round(slope / NOISE, 2)
    out["verdict"] = (
        f"STILL RISING at the last doubling ({slope:+.5f} = "
        f"{slope / NOISE:+.2f} sigma) -- the model is DATA-limited, so more "
        f"interactions over the same users and items should help. Cross-dataset "
        f"transfer from KuaiRand-1K/27K is worth building."
        if slope >= NOISE / 2 else
        f"FLAT at the last doubling ({slope:+.5f} = {slope / NOISE:+.2f} sigma) "
        f"-- the model is INFORMATION-limited, not data-limited. More rows over "
        f"the same users and items would not help; do not build the transfer "
        f"pipeline on this evidence.")
    print(f"\n  last doubling: {slope:+.5f} ({slope / NOISE:+.2f} sigma)")
    print(f"  VERDICT: {out['verdict']}")
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
