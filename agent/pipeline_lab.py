"""Research capabilities distilled from the Opus research run.

Each of these exists because it was the thing that actually moved the research,
not because it sounded like a good tool. The provenance is recorded with each
one so the agent's prompt can explain WHY the capability is there.

  1. training_dynamics()   -- what the epoch curve does when nothing stops it.
     Found the finding that redirected the whole phase: the incumbent peaks at
     epoch 14 and decays to -29.6 sigma by epoch 60. Every menu-level lever was
     null while the real binding constraint was the stopping rule, and nobody
     had looked.

  2. hardcoded_constants() -- modelling constants baked into the training
     library that the menu cannot reach. Found tau_days=3.0 inside History, an
     untuned decay over a 14-day window sitting inside the incumbent.

  3. override_experiment() -- controlled paired experiments specified as direct
     cfg overrides rather than menu options, because k, tau and the stopping
     rule are not menu axes and never could be.

  4. selection_rule_test() -- THE important one. Distinguishes "this number is
     higher on validation" from "this RULE generalises". Choose on one half of
     validation, score on the other, both directions, several splits. This is
     what overturned a previously REJECTED idea: snapshot ensembling had been
     rejected by a guard that compared it against the best single checkpoint on
     the SAME validation set that chose that checkpoint -- a biased comparison.
     Tested honestly, averaging the top-5 checkpoints beats argmax by +0.00069
     (+0.87 sigma), t=5.54, winning 22/24 held-out evaluations.

  5. free_recombination() -- questions answerable from stored predictions with
     no training at all. Refuted a +0.46 sigma "improvement" from median
     aggregation in seconds: resampled over 24 member subsets it won 10/24 at
     -0.06 sigma.

The unifying lesson, and the reason these belong in an autonomous agent: when a
score is computed on the same data that selected it, the number is not
evidence. Three separate results in this run turned on that.
"""
from __future__ import annotations

import os
import re
import statistics
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for _p in (ROOT, os.path.join(ROOT, "runtime"),
           os.path.join(ROOT, "kuairand-starter-kit")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NOISE = 0.0008

# Overrides the agent may set directly on the training config. Deliberately
# small: each entry earned its place by being investigated in the research run,
# and a large list would just be the menu again under a new name.
# Descriptions are deliberately NEUTRAL: they say what a knob does, never
# whether it helps. An earlier version stated the teacher's findings and effect
# sizes here, which made a "discovery" by the agent unfalsifiable -- see
# logs/opus_research/AUTONOMY_AUDIT.md.
SAFE_OVERRIDES = {
    "k": "embedding dimension per field",
    "lr": "learning rate",
    "epochs": "maximum training epochs",
    "patience": "early-stopping patience, in epochs",
    "l2": "L2 penalty on embeddings and linear weights",
    "bs": "batch size",
    "hist_tau_days": "recency decay of the pooled user history, in days",
    "aux_weight": "weight applied to auxiliary-task gradients",
    "n_checkpoints": "how many epoch checkpoints to combine for the final "
                     "prediction (1 = use the single best epoch)",
    "checkpoint_combine": "combine the chosen checkpoints unconditionally "
                          "(otherwise the built-in guard decides)",
}


def training_dynamics(seeds=(0,), max_epochs: int = 60) -> dict:
    """The epoch curve with early stopping DISABLED.

    Answers 'is this model over- or under-fitting, and where does it peak?' --
    a question no score-level experiment can answer, and the one that redirected
    this project's research.
    """
    import train_lib
    from agent import research_run as RR

    splits, meta = train_lib.load_cache()
    base, enc = RR.incumbent_cfg(splits, meta)
    out = {"seeds": {}, "max_epochs": max_epochs}
    for s in seeds:
        cfg = dict(base)
        cfg.update(seed=s, epochs=max_epochs, patience=max_epochs)
        lines: list = []
        train_lib.train_numpy_fm(cfg, enc, splits, meta, lambda m: lines.append(m))
        curve = [(int(a), float(b)) for a, b in
                 re.findall(r"epoch +(\d+) \| valid primary ([0-9.]+)",
                            "\n".join(lines))]
        if not curve:
            continue
        peak = max(curve, key=lambda t: t[1])
        out["seeds"][s] = {
            "curve": curve, "peak_epoch": peak[0], "peak_primary": peak[1],
            "final_primary": curve[-1][1],
            "decline_sigma": round((curve[-1][1] - peak[1]) / NOISE, 2)}
    if out["seeds"]:
        d = statistics.mean(v["decline_sigma"] for v in out["seeds"].values())
        p = statistics.mean(v["peak_epoch"] for v in out["seeds"].values())
        out["verdict"] = (
            f"OVERFITS: peaks around epoch {p:.0f} then declines {d:.1f} sigma by "
            f"epoch {max_epochs}. Early stopping is the binding regulariser, so "
            f"the stopping RULE matters more than capacity or regularisation "
            f"strength." if d < -3 else
            f"no strong overfit: peak around epoch {p:.0f}, end-of-run change "
            f"{d:.1f} sigma. More capacity or more epochs may be affordable.")
    return out


def hardcoded_constants(path: str | None = None) -> list:
    """Modelling constants baked into the training library that no menu axis can
    reach. Found tau_days=3.0 in History -- an untuned decay over a 14-day
    window, sitting inside the incumbent and invisible to the search."""
    path = path or os.path.join(ROOT, "runtime", "train_lib.py")
    src = open(path).read()
    out = []
    for m in re.finditer(r"^(?!\s*#)\s*(?:def\s+\w+\([^)]*?(\w+)\s*:\s*"
                         r"(?:float|int)\s*=\s*([\d.eE+-]+))", src, re.M):
        out.append({"name": m.group(1), "default": m.group(2),
                    "line": src[:m.start()].count("\n") + 1,
                    "kind": "function default"})
    for m in re.finditer(r"^([A-Z_][A-Z0-9_]*)\s*=\s*([\d.eE+-]+)\s*(?:#(.*))?$",
                         src, re.M):
        out.append({"name": m.group(1), "default": m.group(2),
                    "line": src[:m.start()].count("\n") + 1,
                    "kind": "module constant",
                    "comment": (m.group(3) or "").strip()[:80]})
    # A constant can be exposed under a different cfg name than its parameter
    # name (tau_days is reachable as hist_tau_days), so match on both.
    aliases = {"tau_days": "hist_tau_days"}
    exposed = set(SAFE_OVERRIDES)
    for c in out:
        c["override_key"] = aliases.get(c["name"], c["name"])
        c["reachable_by_agent"] = c["override_key"] in exposed
    return out


def selection_rule_test(per_epoch_scores, users, labels, rules: dict,
                        n_splits: int = 4, seed: int = 7) -> dict:
    """Does a SELECTION RULE generalise, or is it fitting the data that scored it?

    Choose on one half of the users, score on the other, both directions, over
    several independent splits. `rules` maps name -> fn(per_epoch_primaries,
    per_epoch_scores) -> chosen score vector.

    This is the capability that overturned a rejected idea. Snapshot ensembling
    had been refused by a guard comparing it against the best single checkpoint
    on the SAME validation set that selected that checkpoint. Measured this way
    instead, averaging the top-5 checkpoints beat argmax by +0.87 sigma with
    t=5.54 over 22/24 wins.
    """
    from evaluate import evaluate
    users = np.asarray(users)
    labels = np.asarray(labels)
    uniq = np.unique(users)
    E = np.asarray(per_epoch_scores)          # (seeds, epochs, rows)

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
    Refuted a +0.46 sigma median-aggregation 'improvement' in seconds: over 24
    subsets it won 10/24 at -0.06 sigma, i.e. it was the best of five rules on
    one validation set.
    """
    from evaluate import evaluate
    M = np.asarray(member_scores)
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
        out["rules"][n] = {"mean_delta": round(m, 5), "sigma": round(m / NOISE, 2),
                           "t": round(t, 2),
                           "wins": sum(1 for x in v if x > 0), "n": len(v),
                           "beats_reference": bool(m >= NOISE / 2 and t > 2.0)}
    return out


def render_for_prompt(reveal_findings: bool = False) -> str:
    """The capabilities available, described by WHAT THEY DO.

    `reveal_findings` is False by default and must stay that way for any
    autonomy claim. An earlier version stated the teacher's conclusions and
    effect sizes here -- the finding, the exact parameters, and the caveat --
    which turned an apparent discovery into replay. See
    logs/opus_research/AUTONOMY_AUDIT.md.

    What is still transferred, deliberately, is METHOD rather than ANSWER: the
    capabilities themselves, and the principle that a score computed on the data
    that selected it is not evidence. A method the agent must still decide when
    to apply is capability transfer; a stated result is not.
    """
    L = [
        "## PIPELINE RESEARCH CAPABILITIES (beyond the menu)",
        "The menu names options. These reach parts of the pipeline no option "
        "can express -- the numbers inside the training library. Use them when "
        "the evidence suggests the bottleneck is not which option is selected.",
        "",
        "- training_dynamics(): trains with early stopping disabled and returns "
        "the whole epoch curve, its peak, and how far it moves afterwards. "
        "Answers 'is this model over- or under-fitting, and where does it "
        "peak?' -- which no single score can tell you.",
        "- hardcoded_constants(): lists modelling constants written into the "
        "training library that no menu axis can reach, and whether you can "
        "currently set each one.",
        "- selection_rule_test(): given per-epoch predictions, compares "
        "CHOOSING rules by selecting on one half of the validation users and "
        "scoring on the other, both directions, several splits. Use it whenever "
        "a choice is being made using the same data it will be judged on.",
        "- free_recombination(): compares ways of combining stored predictions "
        "by resampling member subsets. No training, so a rule must win "
        "repeatedly rather than once.",
        "",
        "HOW TO USE THEM: put any of these keys directly in menu_choices "
        "alongside the normal axes. They are validated and range-checked, and "
        "they change the actual training pipeline:",
    ]
    for k in sorted(SAFE_OVERRIDES):
        L.append(f"  {k}: {SAFE_OVERRIDES[k]}")
    L += [
        "",
        "METHODOLOGICAL PRINCIPLE: a score computed on the same data that "
        "selected it is not evidence of generalisation. If a number was picked "
        "as the best of several on the validation set -- a best epoch, a best "
        "rule, a best subset -- prefer a held-out or resampled comparison "
        "before believing it.",
    ]
    if reveal_findings:
        L += ["", "PRIOR RESULTS (teacher findings -- NOT for autonomy tests):",
              "  k=8 -0.03 sigma, k=32 -0.18 sigma; hist_tau_days 1/7/14 all "
              "null; l2 1e-5/1e-4 null; checkpoint averaging +0.87 sigma for a "
              "single model, -0.01 sigma once 16 seeds are ensembled."]
    return "\n".join(L)
