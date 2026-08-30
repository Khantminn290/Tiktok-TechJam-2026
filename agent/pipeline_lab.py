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


# selection_rule_test and free_recombination are NOT defined here. They live in
# runtime/research_tools.py, which is the module generated experiment code can
# import, and are re-exported so orchestrator callers keep the same name. A
# second copy here is exactly how the agent's tools and the agent's scripts
# would drift apart.
from research_tools import (  # noqa: E402
    free_recombination, redundancy, selection_rule_test,
)


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
