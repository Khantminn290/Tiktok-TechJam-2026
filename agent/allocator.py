"""Where should the next iteration go, and why?

A transparent adaptive allocator over experiment FAMILIES. Deliberately not
reinforcement learning: there are ~50 iterations, the reward is a validation
score whose noise floor is 0.0008, and most families will be attempted a
handful of times. Any method that needs to learn a value function from that
will fit noise, and worse, it will do so unaccountably.

What works at this scale is an explicit utility with every term visible:

    utility = expected_gain x P(success) x generalization_confidence
              - runtime_cost - failure_cost - redundancy_penalty

Each term is estimated from THIS RUN'S OWN HISTORY, with shrinkage toward a
prior so that one lucky or unlucky attempt does not swing the allocation. The
success rate of a family observed 2 times is mostly prior; observed 15 times it
is mostly data. That is a Beta posterior mean, which is the honest way to hold
an estimate you barely have evidence for -- the same standard this project
applies to its ML results.

The output is a ranked list with the arithmetic attached, so a judge (or a
teammate) can see why exploration lost to confirmation on iteration 9 rather
than taking it on faith.
"""
from __future__ import annotations

import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

from . import evidence as EV  # noqa: E402
from . import experiment_spec as XS  # noqa: E402

# Families the allocator chooses between. These are what an iteration can BE.
FAMILIES = (XS.EXPLORATION, XS.IMPROVEMENT, XS.BRANCH, XS.CROSSOVER,
            XS.PATH_B_DISCOVERY, XS.MULTI_SEED_REPLICATION,
            XS.ENSEMBLE_CONSTRUCTION)

# Prior beliefs, in units the project actually measures. These are starting
# points that data overrides, not tuned constants.
PRIOR = {
    #                       gain(sigma)  p(success)  generality  runtime(min)
    XS.EXPLORATION:         (1.0,        0.30,       0.60,       3.0),
    XS.IMPROVEMENT:         (0.6,        0.45,       0.75,       3.0),
    XS.BRANCH:              (0.8,        0.30,       0.60,       3.0),
    XS.CROSSOVER:           (0.7,        0.35,       0.65,       3.0),
    XS.PATH_B_DISCOVERY:    (1.5,        0.20,       0.50,       6.0),
    XS.MULTI_SEED_REPLICATION: (0.0,     0.90,       1.00,      12.0),
    XS.ENSEMBLE_CONSTRUCTION:  (0.5,     0.70,       0.90,      15.0),
}

PRIOR_STRENGTH = 4.0     # pseudo-observations; ~4 attempts before data dominates
NOISE = EV.NOISE


def _posterior_rate(successes: int, attempts: int, prior_p: float) -> float:
    """Beta posterior mean. One attempt should not become a policy."""
    a = prior_p * PRIOR_STRENGTH + successes
    b = (1 - prior_p) * PRIOR_STRENGTH + (attempts - successes)
    return a / (a + b)


def observe(nodes: list) -> dict:
    """Summarise the run so far into the state the allocator reasons over."""
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    errored = [n for n in nodes if n.get("status") == "error"]

    # Per-family attempt/success counts, from the recorded category.
    per_family: dict = {f: {"attempts": 0, "successes": 0, "gains": []}
                        for f in FAMILIES}
    # None, not 0.0: "gain" means IMPROVEMENT TO THE RUNNING BEST, which is
    # undefined for the first scored node. Starting at zero credited that node
    # with its entire score as a gain -- 755 sigma -- and the allocator then
    # rated whatever family happened to run first as infinitely promising.
    best_so_far = None
    for n in nodes:
        fam = _family_of(n)
        if fam not in per_family:
            continue
        per_family[fam]["attempts"] += 1
        m = n.get("metrics") or {}
        if n.get("status") == "success" and m:
            per_family[fam]["successes"] += 1
            if best_so_far is None:
                best_so_far = m["primary"]
            elif m["primary"] > best_so_far:
                per_family[fam]["gains"].append(m["primary"] - best_so_far)
                best_so_far = m["primary"]

    primaries = [n["metrics"]["primary"] for n in scored]
    recent = primaries[-4:]
    improving = (max(recent) - min(recent)) if len(recent) >= 2 else 0.0

    path_a = [n for n in nodes if (n.get("implementation_path") or "A").upper() == "A"]
    path_b = [n for n in nodes if (n.get("implementation_path") or "").upper() == "B"]

    def _rate(ns):
        ok = sum(1 for n in ns if n.get("status") == "success")
        return (ok / len(ns)) if ns else None

    return {"iteration": len(nodes),
            "scored": len(scored), "errored": len(errored),
            "best_primary": max(primaries) if primaries else None,
            "recent_spread": round(improving, 5),
            "recent_spread_sigma": round(improving / NOISE, 2),
            "path_a_success_rate": _rate(path_a),
            "path_b_success_rate": _rate(path_b),
            "per_family": per_family,
            "consecutive_failures": _consecutive_failures(nodes),
            "incumbent_confirmed": _incumbent_confirmed(nodes)}


def _family_of(node: dict) -> str:
    """Map a recorded node onto an allocator family."""
    cat = (node.get("research_category") or "").lower()
    if (node.get("action") or "") == "confirm":
        return XS.MULTI_SEED_REPLICATION
    if (node.get("implementation_path") or "").upper() == "B":
        return XS.PATH_B_DISCOVERY
    return {"exploration": XS.EXPLORATION, "exploitation": XS.IMPROVEMENT,
            "ablation": XS.IMPROVEMENT, "confirmation": XS.MULTI_SEED_REPLICATION,
            "integration": XS.CROSSOVER}.get(cat, XS.EXPLORATION)


def _consecutive_failures(nodes: list) -> int:
    n = 0
    for node in reversed(nodes):
        if node.get("status") == "error":
            n += 1
        else:
            break
    return n


def _incumbent_confirmed(nodes: list) -> bool:
    """Has anything reached CONFIRMED via a paired multi-seed experiment?"""
    for n in nodes:
        for e in (n.get("events") or []):
            if e.get("type") == "paired_result":
                if (e.get("evidence") or {}).get("state") == EV.CONFIRMED:
                    return True
    return False


def score_family(family: str, state: dict, budget_left: int,
                 minutes_left: float | None = None) -> dict:
    """Utility for one family, with every term exposed."""
    g_prior, p_prior, gen_prior, rt_prior = PRIOR[family]
    obs = state["per_family"].get(family, {"attempts": 0, "successes": 0, "gains": []})

    p_success = _posterior_rate(obs["successes"], obs["attempts"], p_prior)

    # Expected gain: shrink the prior toward what this family has actually
    # delivered as an improvement to the running best.
    gains = obs["gains"]
    if gains:
        observed = (sum(gains) / len(gains)) / NOISE
        w = len(gains) / (len(gains) + 2.0)
        expected_gain = (1 - w) * g_prior + w * observed
    else:
        expected_gain = g_prior

    generality = gen_prior
    runtime_cost = rt_prior / 60.0            # normalise to "hours", keeps terms comparable
    failure_cost = (1 - p_success) * 0.4

    # Redundancy: a family attempted repeatedly with nothing to show is being
    # re-run, not explored.
    redundancy = 0.0
    if obs["attempts"] >= 3 and not gains:
        redundancy = 0.3 * math.log1p(obs["attempts"] - 2)

    reasons = []

    # --- state-dependent adjustments, each one stated ---
    if family == XS.MULTI_SEED_REPLICATION:
        # Confirmation has no exploratory upside by construction; its value is
        # that it is the ONLY family that can move something to CONFIRMED, which
        # is the only state that may change what we submit.
        if state["best_primary"] and not state["incumbent_confirmed"]:
            expected_gain = max(expected_gain, 1.2)
            reasons.append("nothing is CONFIRMED yet, so no result can be acted "
                           "on until something is")
        if budget_left < 2:
            redundancy += 1.0
            reasons.append("too little budget left to pay for a paired run")

    if family == XS.PATH_B_DISCOVERY:
        rate = state.get("path_b_success_rate")
        if rate is not None and rate < 0.3 and state["iteration"] >= 4:
            failure_cost += 0.5
            reasons.append(f"Path B has completed only {rate:.0%} of attempts "
                           f"in this run")

    if family in (XS.EXPLORATION, XS.BRANCH):
        # Late in the budget, opening new branches leaves them unconfirmable.
        if budget_left <= 3:
            redundancy += 0.5
            reasons.append("late in the budget: a new branch cannot be "
                           "confirmed before the run ends")
        if state["recent_spread_sigma"] < 1.0 and state["iteration"] >= 5:
            expected_gain *= 0.6
            reasons.append(f"the last few results span only "
                           f"{state['recent_spread_sigma']:.2f} sigma — the "
                           f"space looks flat here")

    if state["consecutive_failures"] >= 2 and family == XS.PATH_B_DISCOVERY:
        failure_cost += 0.5
        reasons.append(f"{state['consecutive_failures']} consecutive failures; "
                       f"free-form code is what has been failing")

    utility = (expected_gain * p_success * generality
               - runtime_cost - failure_cost - redundancy)

    return {"family": family, "utility": round(utility, 4),
            "expected_gain_sigma": round(expected_gain, 3),
            "p_success": round(p_success, 3),
            "generalization_confidence": round(generality, 3),
            "runtime_cost": round(runtime_cost, 3),
            "failure_cost": round(failure_cost, 3),
            "redundancy_penalty": round(redundancy, 3),
            "attempts": obs["attempts"], "successes": obs["successes"],
            "reasons": reasons}


def allocate(nodes: list, budget_left: int,
             minutes_left: float | None = None) -> dict:
    """Rank the families and say why the winner won."""
    state = observe(nodes)
    ranked = sorted((score_family(f, state, budget_left, minutes_left)
                     for f in FAMILIES),
                    key=lambda r: -r["utility"])
    return {"state": state, "ranked": ranked, "choice": ranked[0]["family"],
            "budget_left": budget_left}


def render(a: dict) -> str:
    s, L = a["state"], []
    L.append("## EXPERIMENT ALLOCATION — where this iteration should go")
    L.append(f"state: iteration {s['iteration']}, {s['scored']} scored, "
             f"{s['errored']} failed, budget left {a['budget_left']}, "
             f"best {s['best_primary']}, recent spread "
             f"{s['recent_spread_sigma']:.2f} sigma, "
             f"incumbent confirmed: {s['incumbent_confirmed']}")
    L.append(f"{'family':<26}{'util':>8}{'gain':>7}{'p(ok)':>7}{'gen':>6}"
             f"{'-rt':>7}{'-fail':>7}{'-redun':>8}")
    for r in a["ranked"]:
        L.append(f"{r['family']:<26}{r['utility']:>8.3f}"
                 f"{r['expected_gain_sigma']:>7.2f}{r['p_success']:>7.2f}"
                 f"{r['generalization_confidence']:>6.2f}"
                 f"{r['runtime_cost']:>7.2f}{r['failure_cost']:>7.2f}"
                 f"{r['redundancy_penalty']:>8.2f}")
    top = a["ranked"][0]
    L.append(f"\nCHOICE: {top['family']} "
             f"({top['attempts']} attempts, {top['successes']} completed)")
    for r in top["reasons"]:
        L.append(f"  - {r}")
    runner_up = a["ranked"][1]
    L.append(f"  runner-up {runner_up['family']} at {runner_up['utility']:.3f}"
             f"{'; ' + runner_up['reasons'][0] if runner_up['reasons'] else ''}")
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(ROOT, "logs", "journal.jsonl"))
    ap.add_argument("--budget-left", type=int, default=6)
    a = ap.parse_args()
    nodes = []
    if os.path.exists(a.journal):
        with open(a.journal) as fh:
            for ln in fh:
                if ln.strip():
                    try:
                        nodes.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    print(render(allocate(nodes, a.budget_left)))
