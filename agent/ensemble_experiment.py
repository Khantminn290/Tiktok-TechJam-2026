"""Ensembling as something the agent can DO, not a step a human runs afterwards.

This closed a real and embarrassing gap. The submitted result is a 16-seed
rank-averaged ensemble scoring 0.60541, but the agent's search space had ten
axes -- loss, negative sampling, history, multitask, model, temporal, training,
data extras, sample weighting, regularisation -- and ensembling was not one of
them. The single largest measured gain available, +0.00078 (about 1 sigma), sat
permanently outside the agent's reach, and the step that captured it was a
human typing `python3 -m agent.final_ensemble --seeds 16`.

That made the honest claim awkward: the agent found the configuration, but a
person performed the operation that turned it into the submitted number.

What ensembling actually buys, and why it is not free:

    a single model is one draw from a distribution with a real spread
    (0.60463 +/- 0.00032 here). Averaging k independent seeds does not make any
    model better -- it removes the seed variance from the thing you submit.

So the effect to measure is the ensemble against the MEAN MEMBER, not against
the best member. Comparing against the best member measures luck, because the
best of k draws is above the mean by construction; that comparison would report
a gain even if ensembling did nothing at all.

Cost is stated up front and charged to the training-run budget: an ensemble at
k seeds is k training executions and one decision.
"""
from __future__ import annotations

import json
import os
import statistics
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
RUNTIME_DIR = os.path.join(ROOT, "runtime")

from . import evidence as EV  # noqa: E402
from . import experiment_spec as XS  # noqa: E402
from .executor import run_solution  # noqa: E402

SEED_SOLUTION = os.path.join(RUNTIME_DIR, "seed_solution.py")

# Below this many members the average is itself noisy, so an "ensemble gain"
# measured on 2-3 seeds is mostly the spread of a tiny sample.
MIN_MEMBERS = 4


def _flush(*a, **k):
    k.setdefault("flush", True)
    print(*a, **k)


def train_members(choices: dict, seeds, work_dir: str, timeout_s: int = 1800,
                  log=_flush) -> dict:
    """Train one member per seed. Returns {seed: {metrics, scores_valid path}}."""
    from . import execution_events as EX

    with open(SEED_SOLUTION) as fh:
        src = fh.read()
    os.makedirs(work_dir, exist_ok=True)
    out, events = {}, []
    for s in seeds:
        d = os.path.join(work_dir, f"seed_{s:02d}")
        sv = os.path.join(d, "scores_valid.npy")
        if os.path.exists(sv) and os.path.exists(os.path.join(d, "metrics.json")):
            # A real independent observation -- it WAS trained once at this
            # seed -- but no compute is being spent now. Charging it as a
            # training run reported spend that never happened.
            with open(os.path.join(d, "metrics.json")) as fh:
                out[s] = {"metrics": json.load(fh), "dir": d}
            events.append(EX.event(EX.REUSED_ARTIFACT, seed=s, config=choices,
                                   detail="member already on disk"))
            log(f"    [member] seed {s}  (reusing — no compute)")
            continue
        t0 = time.time()
        res = run_solution(src, os.path.join(work_dir, f"seed_{s:02d}.py"),
                           choices, d, timeout_s=timeout_s, seed=s)
        el = time.time() - t0
        if res.ok and res.metrics:
            out[s] = {"metrics": {k: float(res.metrics[k])
                                  for k in ("GAUC", "nDCG@5", "primary")},
                      "dir": d}
            events.append(EX.event(EX.FRESH_EXECUTION, seed=s, seconds=el,
                                   config=choices))
            log(f"    [member] seed {s}  primary {out[s]['metrics']['primary']:.5f}"
                f"  {el:.0f}s")
        else:
            head = (res.error_trace or "")[:110].replace("\n", " ")
            events.append(EX.event(EX.FAILED_EXECUTION, seed=s, seconds=el,
                                   config=choices, detail=head))
            log(f"    [member] seed {s}  FAILED — {head}")
    out["_events"] = events
    return out


def combine(members: dict, log=_flush) -> dict:
    """Rank-normalise and average, then score. This is the submitted recipe.

    Rank-normalising before averaging matters: both metrics read only the
    ORDER of scores, so averaging raw values would let whichever member happens
    to have the widest spread dominate the result.
    """
    import numpy as np
    import train_lib
    from evaluate import evaluate
    from .ensemble import rank_normalise

    members = {k: v for k, v in members.items() if k != "_events"}
    if len(members) < 2:
        return {"usable": False, "reason": f"only {len(members)} member(s) trained"}

    splits, _meta = train_lib.load_cache()
    va = splits["valid"]
    seeds = sorted(members)
    arrays = [np.load(os.path.join(members[s]["dir"], "scores_valid.npy"))
              for s in seeds]
    ranked = [rank_normalise(a) for a in arrays]
    agg = np.mean(ranked, axis=0)
    m = evaluate(list(va["user_raw"]), va["long_view"], agg)
    # Report every fixed prefix for diagnosis, never for choosing k. The action
    # always submits all pre-registered seeds; selecting the best prefix here
    # would tune ensemble size on validation.
    curve = {
        str(k): round(float(evaluate(
            list(va["user_raw"]), va["long_view"],
            np.mean(ranked[:k], axis=0))["primary"]), 5)
        for k in range(1, len(ranked) + 1)
    }

    singles = [members[s]["metrics"]["primary"] for s in seeds]
    mean_member = statistics.mean(singles)
    sd_member = statistics.pstdev(singles) if len(singles) > 1 else 0.0
    ens = float(m["primary"])

    return {
        "usable": True, "k": len(seeds), "seeds": seeds,
        "ensemble": {k: round(float(m[k]), 5)
                     for k in ("GAUC", "nDCG@5", "primary")},
        "primary": round(ens, 5),
        "mean_member": round(mean_member, 5),
        "sd_member": round(sd_member, 5),
        "best_member": round(max(singles), 5),
        # THE number: ensembling removes seed variance, so its value is against
        # the AVERAGE member. Against the best member it would look good even if
        # ensembling did nothing, because the best of k draws is above the mean
        # by construction.
        "gain_over_mean_member": round(ens - mean_member, 5),
        "gain_sigma": round((ens - mean_member) / EV.NOISE, 2),
        "gain_over_best_member": round(ens - max(singles), 5),
        "k_curve_diagnostic_only": curve,
    }


def grade(res: dict) -> dict:
    """Is this ensemble gain something we may act on?"""
    if not res.get("usable"):
        return {"state": EV.UNTESTED, "actionable": False,
                "why": res.get("reason", "not enough members"),
                "next_step": "train more members"}
    if res["k"] < MIN_MEMBERS:
        return {"state": EV.PRELIMINARY, "actionable": False,
                "why": f"only {res['k']} members; the average of a handful of "
                       f"seeds is itself noisy",
                "next_step": f"train at least {MIN_MEMBERS} members"}
    gain = res["gain_over_mean_member"]
    if gain < EV.NOISE / 2:
        return {"state": EV.REJECTED, "actionable": False,
                "why": f"{gain:+.5f} over the mean member is under half the "
                       f"noise floor",
                "next_step": "ensembling is not buying anything here"}
    # Deterministic given the members: re-aggregating the same arrays reproduces
    # it exactly, so there is no seed lottery left to survive.
    return {"state": EV.CONFIRMED, "actionable": True,
            "why": f"{gain:+.5f} ({res['gain_sigma']:+.2f} sigma) over the mean "
                   f"of {res['k']} members; reproducible by re-aggregating the "
                   f"stored predictions",
            "next_step": "this is the number to submit"}


def run(choices: dict, seeds, work_dir: str, timeout_s: int = 1800,
        log=_flush) -> dict:
    log(f"  [ensemble] training {len(list(seeds))} members")
    members = train_members(choices, seeds, work_dir, timeout_s, log)
    events = members.get("_events", [])
    res = combine(members, log)
    ev = grade(res)
    if res.get("usable"):
        log(f"  [ensemble] k={res['k']}  members {res['mean_member']:.5f} "
            f"+/- {res['sd_member']:.5f}  ->  ensemble {res['primary']:.5f}  "
            f"(gain {res['gain_over_mean_member']:+.5f}, "
            f"{res['gain_sigma']:+.2f}σ)")
    log(f"  [ensemble] {ev['state']} — {ev['why']}")
    from . import execution_events as EX
    tally = EX.tally(events)
    if tally["reused_artifacts"] or tally["duplicate_reuse_attempts"]:
        log(f"  [ensemble] {tally['fresh_executions']} fresh, "
            f"{tally['reused_artifacts']} reused artifacts (no compute), "
            f"{tally['unique_observations']} unique observations"
            + (f", {tally['duplicate_reuse_attempts']} duplicate"
               if tally["duplicate_reuse_attempts"] else ""))
    return {"members": {k: v for k, v in members.items() if k != "_events"},
            "result": res, "evidence": ev, "events": events,
            "execution_tally": tally,
            "promote": bool(ev.get("actionable"))}


def spec_for(choices: dict, k: int = 8, parent: int | None = None,
             timeout_s: int = 1800) -> XS.ExperimentSpec:
    """An ensemble experiment as a first-class spec the loop can schedule."""
    return XS.ExperimentSpec(
        hypothesis=(f"Averaging {k} independent seeds of this configuration "
                    f"should remove seed variance from the submitted score. "
                    f"The effect is measured against the MEAN member, not the "
                    f"best one."),
        experiment_type=XS.ENSEMBLE_CONSTRUCTION,
        control=dict(choices), treatment=dict(choices),
        seeds=tuple(range(k)), parent_node=parent,
        runtime_budget_s=timeout_s,
        notes="ensemble construction: k training runs, one decision")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="run an ensemble experiment")
    ap.add_argument("--choices", default=None,
                    help="JSON menu_choices; defaults to the submitted config")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "ensemble_exp"))
    a = ap.parse_args()
    ch = json.loads(a.choices) if a.choices else json.load(
        open(os.path.join(ROOT, "logs", "ensemble_results.json")))["config"]
    out = run(ch, range(a.k), a.out)
    raise SystemExit(0 if out["promote"] else 1)
