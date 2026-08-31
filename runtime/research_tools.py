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

import json
import math
import os
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
    # `rules` is a MAPPING name -> callable, not a list of names. Passing a list
    # fails later with "'list' object has no attribute 'items'", which does not
    # say what was expected. Same class as the payload error below: caught here
    # because the caller usually builds it in a variable that no static check
    # can recognise.
    if not isinstance(rules, dict):
        raise TypeError(
            f"rules must be a dict mapping a name to a callable, not "
            f"{type(rules).__name__}. Each callable takes "
            f"(per_epoch_primaries, per_epoch_scores) and returns one score "
            f"vector; the FIRST entry is the reference the others are measured "
            f"against. For example:\n"
            f"    rules = {{\n"
            f"        'argmax_epoch': lambda prim, sc: sc[int(np.argmax(prim))],\n"
            f"        'mean_top3':    lambda prim, sc: np.mean(\n"
            f"                            [sc[i] for i in np.argsort(prim)[-3:]], axis=0),\n"
            f"    }}")
    bad = [k for k, v in rules.items() if not callable(v)]
    if bad:
        raise TypeError(
            f"these rules are not callable: {bad}. A rule is a function of "
            f"(per_epoch_primaries, per_epoch_scores), not a string or a score.")
    users = np.asarray(users)
    labels = np.asarray(labels)
    uniq = np.unique(users)
    # np.asarray raises on ragged input BEFORE the shape check below ever
    # runs, and its message -- "setting an array element with a sequence, the
    # requested array has an inhomogeneous shape after 2 dimensions" -- says
    # nothing about what the caller did wrong. Observed three times in live
    # runs, each costing a full training run, and each time the cause was the
    # same: the raw (epoch, valid_primary, scores) capture payload passed
    # straight in. Static preflight cannot reliably catch it because the caller
    # binds it to a differently-named variable first, so it is caught here.
    try:
        E = np.asarray(per_epoch_scores)      # (seeds, epochs, rows)
    except ValueError as e:
        raise ValueError(
            f"per_epoch_scores could not be read as an array ({e}). This is "
            f"almost always the raw capture payload: cfg['capture_epoch_scores'] "
            f"is a LIST of (epoch, valid_primary, scores_valid) tuples, while "
            f"this function needs a 3-D (seeds, epochs, rows) array of score "
            f"vectors only.\n"
            f"Use the adapter that converts it:\n"
            f"    from research_tools import capture_selection_rule_test\n"
            f"    out = capture_selection_rule_test("
            f"cfg['capture_epoch_scores'], users, labels)") from e
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


def capture_selection_rule_test(capture_epoch_scores, users, labels,
                                n_splits: int = 4, seed: int = 7) -> dict:
    """Validate fixed epoch-selection rules for one captured training curve.

    ``train_numpy_fm`` records ``(epoch, valid_primary, scores_valid)`` tuples.
    This adapter turns that documented payload into the 3-D tensor and callable
    rule mapping required by :func:`selection_rule_test`, so generated code does
    not have to reconstruct a subtle API contract after training has completed.
    The result is a diagnostic; it does not authorise a validation-selected rule
    to replace a submitted model.
    """
    rows = list(capture_epoch_scores or [])
    if not rows:
        raise ValueError("capture_epoch_scores is empty")
    try:
        curve = np.stack([np.asarray(row[2]) for row in rows], axis=0)
    except (IndexError, TypeError, ValueError) as e:
        raise ValueError(
            "capture_epoch_scores must contain (epoch, valid_primary, "
            "scores_valid) tuples with equal-length valid score arrays") from e
    if curve.ndim != 2:
        raise ValueError(f"captured score curve must be (epochs, rows); got {curve.shape}")

    def best_epoch(primaries, scores):
        return scores[int(np.argmax(primaries))]

    def mean_top_n(n):
        def _rule(primaries, scores):
            top = np.argsort(-primaries)[:min(n, len(scores))]
            return np.mean(scores[top], axis=0)
        return _rule

    rules = {"best_epoch": best_epoch}
    if len(curve) >= 2:
        rules["mean_top2"] = mean_top_n(2)
    if len(curve) >= 3:
        rules["mean_top3"] = mean_top_n(3)
    return selection_rule_test(np.asarray([curve]), users, labels, rules,
                               n_splits=n_splits, seed=seed)


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


# ---------------------------------------------------------------- contract ---
CONTRACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "capability_contract.json")


def contract(name: str | None = None) -> dict:
    """The capability contract, readable from inside a generated experiment.

    This is the EXECUTABLE half of the contract. A generated script can ask what
    a capability returns instead of guessing:

        from research_tools import contract
        contract("train_numpy_fm")["returns"]
        # {'kind': 'dict', 'keys': ['scores_valid', 'scores_test', ...]}

    Every Path B crash in the last recorded run was a call site disagreeing with
    one of these shapes -- unpacking a dict as a 2-tuple, or hunting for a test
    vector inside a per-epoch capture entry that only holds valid scores.

    Contains no data, no labels and no scores: it describes an API surface, so
    reading it cannot reach the hidden test.
    """
    if not os.path.exists(CONTRACT_PATH):
        return {}
    with open(CONTRACT_PATH) as fh:
        doc = json.load(fh)
    if name is None:
        return doc
    return (doc.get("capabilities") or {}).get(name, {})


def describe(name: str) -> str:
    """Human-readable usage for one capability, including a correct example."""
    c = contract(name)
    if not c:
        return f"no contract entry for {name!r}"
    L = [f"{name}: {c.get('purpose')}",
         f"  inputs:  {c.get('inputs')}",
         f"  returns: {c.get('outputs')}"]
    if c.get("example"):
        L.append("  example:")
        L += [f"    {ln}" for ln in str(c["example"]).splitlines()]
    return "\n".join(L)


# ------------------------------------------------------------ config builder ---
def incumbent_cfg(splits, meta, choices=None, **overrides):
    """A COMPLETE, valid training config for the incumbent, plus its encoding.

    Returns `(cfg, enc)` ready to pass straight to `train_lib.train_numpy_fm`.
    Any keyword argument overrides one key, so a one-axis experiment is:

        cfg, enc = incumbent_cfg(splits, meta, hist_tau_days=7.0)
        cfg["capture_epoch_scores"] = []          # if you want the epoch curve
        r = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)

    This exists because `train_numpy_fm` requires FIFTEEN keys and fails on the
    first missing one, so hand-building a config costs an iteration per key.
    That was measured: across three post-architecture runs, five of six real
    crashes were partial configs -- `KeyError: 'history'`, then `'dim'`, then
    `'bs'`, then `'seed'`, then `'k'`. The agent was correctly following the
    capability contract's advice to train directly; the contract simply had not
    told it what a complete config contains, and there was no way to obtain one.
    """
    import train_lib
    ch = choices
    if ch is None:
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(here)
        with open(os.path.join(root, "logs", "ensemble_results.json")) as fh:
            ch = json.load(fh)["config"]
    enc, dim, _off, _dims = train_lib.encode_features(splits, meta, ch["temporal"])
    training = ch.get("training", "default")
    cfg = {
        "dim": dim, "k": 32 if training == "k32" else 16,
        "lr": 5e-4 if training == "lower_lr_longer" else 1e-3,
        "bs": 8192,
        "epochs": 60 if training == "lower_lr_longer" else 40,
        "patience": 6 if training == "lower_lr_longer" else 4,
        "seed": 0,
        "loss": ch["loss"], "history": ch["user_history"],
        "multitask": ch.get("multitask", "none"), "model": ch["model"],
        "training": training, "neg_sampling": ch["neg_sampling"],
        "sample_weighting": ch["sample_weighting"],
        "l2": {"l2_default": 1e-6, "l2_1e5": 1e-5, "l2_1e4": 1e-4,
               "l2_1e3": 1e-3}.get(ch.get("regularization", "l2_default"), 1e-6),
        "snapshot_ensemble": 0, "bootstrap_seed": None,
        "aux_weight": 0.2, "device": "cpu",
    }
    cfg["aux_tasks"] = train_lib.AUX_MAP[cfg["multitask"]]
    cfg.update(overrides)
    if "multitask" in overrides:
        cfg["aux_tasks"] = train_lib.AUX_MAP[cfg["multitask"]]
    return cfg, enc


__all__ = ["NOISE", "expected_max_of_n", "selection_pressure",
           "convergence_epsilon", "audit_comparison", "selection_rule_test",
           "free_recombination", "redundancy", "incumbent_cfg"]
