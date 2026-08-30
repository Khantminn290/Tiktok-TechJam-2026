"""Research capabilities that generated experiment code can genuinely import.

This module is the real thing, not a shim written to make the LLM's guesses
come true. It lives in `runtime/` because that is what the experiment
subprocess actually has on its PYTHONPATH, and it depends on nothing from the
`agent/` package for the same reason.

Everything here is a pure function over arrays the calling script already has.
None of it trains a model and none of it touches the test labels, which is why
it is safe to hand to generated code.

The agent-side modules import these same implementations rather than keeping
their own copies -- `agent.validity` and `agent.pipeline_lab` both delegate
here. One implementation, two access paths, so the capability an experiment
imports is byte-for-byte the one the orchestrator ran.

Docstrings here are deliberately NEUTRAL. They say what a function measures and
what its result means, never what it previously found. A tool whose help text
contains a research conclusion is an answer key, and this module is readable by
the agent.
"""
from __future__ import annotations

import math
import statistics

import numpy as np

# The official FM baseline's own 5-seed standard deviation on this benchmark.
# Every "is this real?" judgement here is expressed relative to it.
NOISE = 0.0008

FATAL, WARN, NOTE = "FATAL", "WARN", "NOTE"


# --------------------------------------------------------- selection effects ---
def expected_max_of_n(n: int, sd: float | None = None) -> float:
    """Expected maximum of n independent zero-mean draws, in score units.

    This is the number that matters whenever something is reported as "the best
    of n": even with no real effect at all, the best of several noisy
    comparisons sits measurably above zero.
    """
    sd = NOISE if sd is None else sd
    if n <= 1:
        return 0.0
    # Blom's approximation to the expected largest order statistic.
    return sd * (math.sqrt(2 * math.log(n)) -
                 (math.log(math.log(max(n, 3))) + math.log(4 * math.pi))
                 / (2 * math.sqrt(2 * math.log(n))))


def selection_pressure(n_candidates_compared: int, seed_sd: float | None = None) -> dict:
    """How large an apparent gain does picking the best of n produce by itself?"""
    e = expected_max_of_n(n_candidates_compared, seed_sd)
    return {"n": n_candidates_compared, "expected_max_delta": round(e, 5),
            "expected_max_sigma": round(e / NOISE, 2),
            "reading": (f"choosing the best of {n_candidates_compared} noisy "
                        f"comparisons yields about {e / NOISE:+.2f} sigma even "
                        f"when nothing works; a real effect must clear this "
                        f"before it means anything")}


def convergence_epsilon(n_iterations: int, seed_sd: float | None = None) -> float:
    """Upward drift of a running MAXIMUM over n iterations, by luck alone.

    The calibrated threshold for "this search has stopped making progress": a
    best-so-far climbs on its own, so anything below this is not evidence of
    improvement.
    """
    return expected_max_of_n(max(n_iterations, 2), seed_sd)


def audit_comparison(delta: float, n_seeds: int = 1, paired: bool = False,
                     n_candidates_compared: int = 1,
                     selected_on_eval_data: bool = False,
                     seed_sd: float | None = None,
                     confirmed_out_of_sample: bool = False) -> dict:
    """Grade one claimed improvement by how it was MEASURED, not by its size.

    Returns a severity and a list of findings. Advisory only: it describes what
    a number is worth as evidence and never refuses anything.
    """
    sd = NOISE if seed_sd is None else seed_sd
    findings = []
    sigma = delta / NOISE

    if n_seeds < 2:
        findings.append((FATAL, f"rests on {n_seeds} seed(s); seed noise here is "
                                f"{NOISE}, so a single draw cannot distinguish "
                                f"this from luck"))
    elif n_seeds < 5 and abs(sigma) < 2:
        findings.append((WARN, f"{n_seeds} seeds is thin for a {sigma:+.2f} sigma "
                               f"effect; prefer >=5 paired seeds"))

    if abs(delta) < NOISE / 2:
        findings.append((WARN, f"{delta:+.5f} is under half the noise floor "
                               f"({NOISE / 2}); this is not a result whatever "
                               f"its sign"))

    if n_candidates_compared > 1:
        infl = expected_max_of_n(n_candidates_compared, sd)
        sev = FATAL if infl >= abs(delta) else WARN
        findings.append((sev,
                         f"best of {n_candidates_compared} comparisons: with NO "
                         f"real effect the max would sit about {infl:+.5f} "
                         f"({infl / NOISE:+.2f} sigma) above zero by selection "
                         f"alone, against a claimed {delta:+.5f}"))

    if selected_on_eval_data and not confirmed_out_of_sample:
        findings.append((FATAL, "the winner was chosen using the same data it is "
                                "scored on. That number measures fit to this "
                                "split, not generalisation -- confirm on a "
                                "held-out partition or by resampling"))

    if not paired and n_seeds >= 2:
        findings.append((NOTE, "unpaired arms; pairing by seed removes most of "
                               "the seed variance and needs far fewer runs"))

    worst = (FATAL if any(s == FATAL for s, _ in findings)
             else WARN if any(s == WARN for s, _ in findings)
             else NOTE if findings else "CLEAN")
    verdict = {
        FATAL: "NOT EVIDENCE as measured -- fix the design before believing it",
        WARN: "weak evidence -- treat as a lead, not a result",
        NOTE: "usable, with a caveat",
        "CLEAN": "soundly measured",
    }[worst]
    return {"delta": round(delta, 5), "sigma": round(sigma, 2),
            "severity": worst, "verdict": verdict,
            "findings": [{"level": s, "message": m} for s, m in findings],
            "trustworthy": worst not in (FATAL,)}


# ------------------------------------------------------------ rule selection ---
def selection_rule_test(per_epoch_scores, users, labels, rules: dict,
                        n_splits: int = 4, seed: int = 7) -> dict:
    """Does a SELECTION RULE generalise, or is it fitting the data that scored it?

    Chooses on one half of the users and scores on the other, both directions,
    over several independent splits. `rules` maps name -> fn(per_epoch_primaries,
    per_epoch_scores) -> chosen score vector. The FIRST rule is the reference
    every other rule is measured against.

    Use this whenever a number was produced by an argmax over candidates
    evaluated on the same data now being used to score the winner.
    """
    from evaluate import evaluate
    users = np.asarray(users)
    labels = np.asarray(labels)
    uniq = np.unique(users)
    E = np.asarray(per_epoch_scores)          # (seeds, epochs, rows)
    if E.ndim != 3:
        raise ValueError(
            f"per_epoch_scores must be (seeds, epochs, rows); got shape {E.shape}. "
            f"If you captured a single seed, wrap it: np.array([curve]).")
    if E.shape[2] != len(users):
        raise ValueError(f"score rows {E.shape[2]} != users {len(users)}")

    def sc(mask, s):
        return evaluate(list(users[mask]), labels[mask], s[mask])["primary"]

    deltas: dict = {n: [] for n in rules}
    ref = list(rules)[0]
    rng = np.random.default_rng(seed)
    for _ in range(n_splits):
        perm = rng.permutation(len(uniq))
        A = set(uniq[perm[:len(uniq) // 2]])
        mA = np.array([u in A for u in users])
        for mc, me in ((mA, ~mA), (~mA, mA)):
            for si in range(E.shape[0]):
                pc = np.array([sc(mc, E[si, e]) for e in range(E.shape[1])])
                chosen = {n: sc(me, fn(pc, E[si])) for n, fn in rules.items()}
                for n in rules:
                    deltas[n].append(chosen[n] - chosen[ref])
    out = {"reference_rule": ref, "n_evaluations": len(deltas[ref]), "rules": {}}
    for n, d in deltas.items():
        if n == ref:
            continue
        m, sd = statistics.mean(d), statistics.pstdev(d)
        t = m / (sd / len(d) ** 0.5) if sd > 0 else 0.0
        out["rules"][n] = {
            "mean_delta": round(m, 5), "sigma": round(m / NOISE, 2),
            "t": round(t, 2), "wins": sum(1 for x in d if x > 0), "n": len(d),
            "generalises": bool(m >= NOISE / 2 and t > 2.0)}
    return out


def free_recombination(member_scores, users, labels, rules: dict,
                       n_subsets: int = 24, subset: int = 8,
                       seed: int = 0) -> dict:
    """Compare aggregation rules over stored predictions -- no training at all.

    Resamples member subsets so a rule has to win repeatedly rather than once.
    `rules` maps name -> fn(subset_scores) -> combined score vector; the first
    rule is the reference.
    """
    from evaluate import evaluate
    M = np.asarray(member_scores)
    if M.ndim != 2:
        raise ValueError(f"member_scores must be (members, rows); got {M.shape}")
    ref = list(rules)[0]
    rng = np.random.default_rng(seed)
    d: dict = {n: [] for n in rules}
    for _ in range(n_subsets):
        idx = rng.choice(len(M), min(subset, len(M)), replace=False)
        sub = M[idx]
        base = evaluate(list(users), labels, rules[ref](sub))["primary"]
        for n, fn in rules.items():
            d[n].append(evaluate(list(users), labels, fn(sub))["primary"] - base)
    out = {"reference_rule": ref, "n_subsets": n_subsets, "rules": {}}
    for n, v in d.items():
        if n == ref:
            continue
        m, sd = statistics.mean(v), statistics.pstdev(v)
        t = m / (sd / len(v) ** 0.5) if sd > 0 else 0.0
        # "beats_reference", not "generalises": an aggregation rule is being
        # compared against another rule on the same predictions, which is a
        # different claim from a selection rule surviving held-out users.
        out["rules"][n] = {
            "mean_delta": round(m, 5), "sigma": round(m / NOISE, 2),
            "t": round(t, 2), "wins": sum(1 for x in v if x > 0), "n": len(v),
            "beats_reference": bool(m >= NOISE / 2 and t > 2.0)}
    return out


# --------------------------------------------------------------- redundancy ---
def redundancy(delta_a, delta_b, users=None) -> dict:
    """Do two interventions attack the SAME source of error, or different ones?

    An intervention can be genuinely useful alone and worthless in combination,
    because both remove the same variance. Stacking them then buys nothing, and
    the second one's solo measurement predicts nothing about the pair.

    Pass the per-row score CHANGES each intervention makes relative to a shared
    baseline (arm_scores - baseline_scores). The correlation between those
    change vectors is the signal:

        corr near +1   the two move the same rows the same way -- REDUNDANT,
                       expect the pair to gain much less than the sum
        corr near  0   they move different rows -- COMPLEMENTARY, the gains
                       plausibly add
        corr near -1   they fight each other

    This is a general question about any two interventions. It does not know or
    care which interventions they are.
    """
    a = np.asarray(delta_a, dtype=float).ravel()
    b = np.asarray(delta_b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"delta vectors must align; got {a.shape} vs {b.shape}")
    if a.std() == 0 or b.std() == 0:
        return {"usable": False,
                "reason": "one intervention changed no scores at all"}
    r = float(np.corrcoef(a, b)[0, 1])
    if r >= 0.6:
        verdict, expect = "REDUNDANT", ("the pair should gain much less than the "
                                        "sum of the two solo effects")
    elif r <= -0.3:
        verdict, expect = "OPPOSED", "the two partly cancel; combining may lose"
    elif abs(r) < 0.25:
        verdict, expect = "COMPLEMENTARY", ("the solo effects plausibly add; worth "
                                            "measuring the pair")
    else:
        verdict, expect = "PARTIALLY OVERLAPPING", ("some shared variance; expect "
                                                    "less than the sum")
    return {"usable": True, "correlation": round(r, 4), "verdict": verdict,
            "expectation": expect,
            "reading": f"per-row score changes correlate at r={r:+.3f} -> {verdict}. "
                       f"Do not assume solo gains add: {expect}."}


__all__ = ["NOISE", "expected_max_of_n", "selection_pressure",
           "convergence_epsilon", "audit_comparison", "selection_rule_test",
           "free_recombination", "redundancy"]
