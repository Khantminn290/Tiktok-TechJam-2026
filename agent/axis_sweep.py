"""Controlled single-axis sweep: change ONE thing, hold everything else, N seeds.

The agent could previously only run single-seed experiments and then reseed the
survivors as a separate pass. That ordering decides which mechanisms look
promising using exactly the evidence -- one draw -- that this project has
repeatedly shown to be untrustworthy: seed noise here is 0.0008, and several
"wins" chosen that way did not survive reseeding.

This runs the comparison the right way round. One axis moves, every other
choice is held at the incumbent, and both arms get the same seeds, so the
difference is attributable to the mechanism rather than to a bundle of
simultaneous changes or to a lucky draw.

Reported as an effect size in units of the noise floor, never as a raw delta,
and paired across seeds -- the same seed in both arms shares its initialisation,
so the paired difference cancels most of the seed variance and needs far fewer
runs than two independent samples would.

Usage:
    python3 -m agent.axis_sweep --axis neg_sampling --values uniform_2,uniform_4
    python3 -m agent.axis_sweep --axis neg_sampling --values uniform_2 --seeds 8
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.executor import run_solution  # noqa: E402

NOISE = 0.0008
SWEEP_DIR = os.path.join(ROOT, "logs", "axis_sweep")


def incumbent() -> tuple:
    """The configuration and frozen script the current result was built from."""
    res = json.load(open(os.path.join(ROOT, "logs", "ensemble_results.json")))
    seed0 = (res.get("seeds_used") or [0])[0]
    code = os.path.join(ROOT, "logs", "final_ensemble", f"seed_{seed0:02d}",
                        "solution.py")
    if not os.path.exists(code):
        raise SystemExit(f"frozen incumbent script missing: {code}")
    return dict(res["config"]), open(code).read()


def run_arm(code: str, cfg: dict, name: str, seeds: list, timeout_s: int) -> dict:
    """Train one arm; reuse any seed already on disk so a rerun resumes."""
    out = {}
    for s in seeds:
        d = os.path.join(SWEEP_DIR, name, f"seed_{s:02d}")
        mp = os.path.join(d, "metrics.json")
        if os.path.exists(mp):
            out[s] = json.load(open(mp))
            continue
        t0 = time.time()
        r = run_solution(code, os.path.join(d, "solution.py"), cfg, d,
                         timeout_s=timeout_s, seed=s)
        if r.ok and os.path.exists(mp):
            out[s] = json.load(open(mp))
            print(f"    seed {s:2d}  {out[s]['primary']:.5f}  "
                  f"{time.time() - t0:.0f}s", flush=True)
        else:
            print(f"    seed {s:2d}  FAILED  {(r.error_trace or '')[:140]}", flush=True)
    return out


def compare(base: dict, arm: dict, key: str = "primary") -> dict:
    """Paired comparison over the seeds both arms actually completed."""
    shared = sorted(set(base) & set(arm))
    if len(shared) < 2:
        return {"n_paired": len(shared), "usable": False}
    diffs = [arm[s][key] - base[s][key] for s in shared]
    mean_d = statistics.mean(diffs)
    sd = statistics.pstdev(diffs)
    # paired t-like statistic; with few seeds this is indicative, not decisive
    t = (mean_d / (sd / (len(diffs) ** 0.5))) if sd > 0 else 0.0
    return {"n_paired": len(shared), "usable": True,
            "base_mean": round(statistics.mean([base[s][key] for s in shared]), 5),
            "arm_mean": round(statistics.mean([arm[s][key] for s in shared]), 5),
            "mean_delta": round(mean_d, 5),
            "sigma": round(mean_d / NOISE, 2),
            "paired_sd": round(sd, 5),
            "paired_t": round(t, 2),
            "wins": sum(1 for d in diffs if d > 0),
            # MAGNITUDE and DIRECTION are separate questions. A small effect
            # that loses on every paired seed with a large t is not "nothing
            # either way" -- its size is under the promotion bar, but its sign
            # is certain, and reporting only the size hides that.
            "verdict": ("BETTER beyond the noise floor" if mean_d >= NOISE else
                        "WORSE beyond the noise floor" if mean_d <= -NOISE else
                        (f"sub-noise in SIZE ({mean_d / NOISE:+.2f} sigma) but "
                         f"CONSISTENT in direction: "
                         f"{sum(1 for x in diffs if x > 0)}/{len(diffs)} wins, "
                         f"t={t:.2f}. Not promotable; also not a positive lever."
                         if abs(t) > 2.0 else
                         "INSIDE the noise floor -- says nothing either way"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--values", required=True, help="comma-separated")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--fresh-base", action="store_true",
                    help="retrain the control arm instead of reusing the stored "
                         "final_ensemble members. Required whenever the cache or "
                         "training library changed, so the two arms differ ONLY "
                         "by the axis under test.")
    a = ap.parse_args()

    cfg, code = incumbent()
    if a.axis not in cfg:
        raise SystemExit(f"{a.axis} is not in the incumbent config: {sorted(cfg)}")
    seeds = list(range(a.seeds))
    base_val = cfg[a.axis]
    print(f"incumbent {a.axis}={base_val}; sweeping "
          f"{a.values} over {a.seeds} seeds (paired)")

    # The base arm is the incumbent config -- reuse the already-trained
    # final_ensemble members rather than retraining identical work.
    base = {}
    if not a.fresh_base:
        for s in seeds:
            mp = os.path.join(ROOT, "logs", "final_ensemble", f"seed_{s:02d}",
                              "metrics.json")
            if os.path.exists(mp):
                base[s] = json.load(open(mp))
    missing = [s for s in seeds if s not in base]
    if missing:
        print(f"  base arm ({a.axis}={base_val}), {len(missing)} seed(s) to train")
        base.update(run_arm(code, cfg, f"{a.axis}__{base_val}", missing, a.timeout))
    else:
        print(f"  base arm ({a.axis}={base_val}): all {len(base)} seeds reused")

    results = {}
    for val in [v.strip() for v in a.values.split(",") if v.strip()]:
        print(f"  arm {a.axis}={val}")
        arm_cfg = dict(cfg); arm_cfg[a.axis] = val
        arm = run_arm(code, arm_cfg, f"{a.axis}__{val}", seeds, a.timeout)
        results[val] = {k: compare(base, arm, k)
                        for k in ("primary", "GAUC", "nDCG@5")}

    print(f"\n{'=' * 72}\nCONTROLLED SWEEP: {a.axis} (paired, "
          f"everything else held at the incumbent)\n{'=' * 72}")
    for val, r in results.items():
        p = r["primary"]
        if not p.get("usable"):
            print(f"  {a.axis}={val:<22} not enough paired seeds "
                  f"({p['n_paired']})")
            continue
        print(f"  {a.axis}={val:<22} {p['base_mean']} -> {p['arm_mean']}  "
              f"{p['mean_delta']:+.5f} ({p['sigma']:+.2f} sigma)  "
              f"wins {p['wins']}/{p['n_paired']}  t={p['paired_t']}")
        print(f"      GAUC {r['GAUC']['sigma']:+.2f} sigma | "
              f"nDCG@5 {r['nDCG@5']['sigma']:+.2f} sigma")
        print(f"      {p['verdict']}")

    out = os.path.join(ROOT, "logs", f"sweep_{a.axis}.json")
    with open(out, "w") as fh:
        json.dump({"axis": a.axis, "incumbent": base_val, "seeds": a.seeds,
                   "results": results}, fh, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
