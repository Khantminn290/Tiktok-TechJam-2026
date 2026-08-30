"""Controlled multi-seed experiment runner for the Opus research phase.

Bypasses the menu deliberately: this phase investigates parts of the pipeline
the menu cannot express (embedding capacity, regularisation shape, checkpoint
selection), so experiments are specified as direct cfg overrides against the
incumbent and compared paired-by-seed.

Everything else is held at the incumbent, and the control arm is the stored
final_ensemble members, which are the incumbent trained under exactly this
protocol -- so a delta here is attributable to the override alone.

The incumbent is never overwritten. This module only reports.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for p in (ROOT, os.path.join(ROOT, "runtime"),
          os.path.join(ROOT, "kuairand-starter-kit")):
    if p not in sys.path:
        sys.path.insert(0, p)

NOISE = 0.0008
OUT_DIR = os.path.join(ROOT, "logs", "opus_research")


def incumbent_cfg(splits, meta, choices=None):
    """The incumbent training config, rebuilt exactly as train_lib.run does.

    Delegates to runtime/research_tools.py so the orchestrator and generated
    experiment code build configs with the same code rather than two copies
    that can drift.
    """
    from research_tools import incumbent_cfg as _build
    return _build(splits, meta, choices)


def control_scores(n_seeds: int) -> dict:
    """Incumbent metrics per seed, from the stored ensemble members."""
    out = {}
    for s in range(n_seeds):
        p = os.path.join(ROOT, "logs", "final_ensemble", f"seed_{s:02d}",
                         "metrics.json")
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out


def run_variant(overrides: dict, seeds: list, tag: str,
                choices: dict | None = None, quiet: bool = True,
                save_dir: str | None = None) -> dict:
    """Train the incumbent with `overrides` applied, at each seed."""
    import numpy as np
    import train_lib
    from evaluate import evaluate

    splits, meta = train_lib.load_cache()
    base_cfg, enc = incumbent_cfg(splits, meta, choices)
    va = splits["valid"]
    res = {}
    for s in seeds:
        cfg = dict(base_cfg)
        cfg.update(overrides)
        cfg["seed"] = s
        t0 = time.time()
        r = train_lib.train_numpy_fm(cfg, enc, splits, meta,
                                     (lambda *a, **k: None) if quiet else print)
        m = evaluate(list(va["user_raw"]), va["long_view"], r["scores_valid"])
        res[s] = {k: float(m[k]) for k in ("GAUC", "nDCG@5", "primary")}
        res[s]["seconds"] = round(time.time() - t0, 1)
        if save_dir:
            # the prediction arrays are what an ensemble comparison needs; the
            # metrics alone cannot answer it
            d = os.path.join(save_dir, f"seed_{s:02d}")
            os.makedirs(d, exist_ok=True)
            np.save(os.path.join(d, "scores_valid.npy"), r["scores_valid"])
            np.save(os.path.join(d, "scores_test.npy"), r["scores_test"])
            with open(os.path.join(d, "metrics.json"), "w") as fh:
                json.dump({k: res[s][k] for k in ("GAUC", "nDCG@5", "primary")}, fh)
        print(f"    [{tag}] seed {s}  primary {res[s]['primary']:.5f}  "
              f"{res[s]['seconds']:.0f}s", flush=True)
    return res


def paired(control: dict, arm: dict, key: str = "primary") -> dict:
    shared = sorted(set(control) & set(arm))
    if len(shared) < 2:
        return {"usable": False, "n": len(shared)}
    d = [arm[s][key] - control[s][key] for s in shared]
    m = statistics.mean(d)
    sd = statistics.pstdev(d)
    t = m / (sd / len(d) ** 0.5) if sd > 0 else 0.0
    return {"usable": True, "n": len(shared),
            "control_mean": round(statistics.mean(control[s][key] for s in shared), 5),
            "arm_mean": round(statistics.mean(arm[s][key] for s in shared), 5),
            "delta": round(m, 5), "sigma": round(m / NOISE, 2),
            "sd": round(sd, 5), "t": round(t, 2),
            "wins": sum(1 for x in d if x > 0)}


def report(name: str, control: dict, arm: dict) -> str:
    L = [f"  {name}"]
    for k in ("primary", "GAUC", "nDCG@5"):
        p = paired(control, arm, k)
        if not p["usable"]:
            L.append(f"    {k}: only {p['n']} paired seeds")
            continue
        L.append(f"    {k:<8} {p['control_mean']:.5f} -> {p['arm_mean']:.5f}  "
                 f"{p['delta']:+.5f} ({p['sigma']:+.2f} sigma)  "
                 f"wins {p['wins']}/{p['n']}  t={p['t']}")
    return "\n".join(L)


def record(entry: dict, path: str | None = None) -> None:
    path = path or os.path.join(OUT_DIR, "experiments.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry.setdefault("timestamp_utc",
                     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")
