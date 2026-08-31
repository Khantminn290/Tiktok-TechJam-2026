"""Explicit run profiles, so the default does not undersell the system.

Every advanced capability in this agent is opt-in: research state, data tools,
feature discovery, multi-candidate planning. That is the right default for a
library, and the wrong default for a competition entry -- a judge running
`python3 run_agent.py` gets a plain menu search and sees none of the autonomy
the system is actually capable of.

`--competition` is one flag that turns the whole thing on, with conservative
resource caps chosen so a run finishes rather than being killed halfway.

Three rules this module keeps:

  * **Explicit CLI wins.** A profile fills in what you did NOT ask for. If you
    passed `--max-iterations 4`, you get 4, and the resolved configuration says
    the value came from the command line.
  * **Nothing unsafe is enabled quietly.** The profile never unlocks
    leakage-sensitive or locked menu options; that still requires its own
    explicit flag, and asking for both is refused rather than silently resolved.
  * **The resolved configuration is printed in full** before any spend, with the
    source of every value, so what ran is never in doubt afterwards.
"""
from __future__ import annotations

import sys

COMPETITION = "competition"
DEFAULT = "default"

# Conservative on purpose. A run that dies at 80% of its budget produces no
# submission; one that finishes at 60% of its budget produces evidence.
COMPETITION_PROFILE = {
    # capabilities -- the point of the profile
    "data_tools": True,
    "research_state": True,
    "feature_discovery": True,
    "n_candidates": 4,
    # The organizer rule, not a demonstration quota, decides when a scored run
    # stops. Branching remains available after the eligible artifact exists but
    # may never defer official convergence.
    "min_branching_iterations": 0,
    # resource caps
    "max_iterations": 50,
    "wall_clock_limit_h": 6.0,
    "max_spend_usd": 6.0,
    "exec_timeout": 1800,
    "seed": 0,
    "draft_count": 5,
    # training-run budget: an outer iteration is NOT one training run. A paired
    # 3-seed confirmation is six. See agent.budget.
    "max_training_runs": 90,
}

_WHY = {
    "data_tools": "measure the data before hypothesising about it",
    "research_state": "carry evidence between iterations",
    "feature_discovery": "Path B: invent and probe features",
    "n_candidates": "score several proposals instead of taking the first",
    "min_branching_iterations": "let the search branch before it may converge",
    "max_iterations": "outer-loop decisions",
    "wall_clock_limit_h": "hard stop so a run cannot hang overnight",
    "max_spend_usd": "LLM ceiling",
    "exec_timeout": "per-training-run timeout",
    "seed": "base seed; confirmations use a fixed multi-seed set",
    "draft_count": "drafts before the policy may branch",
    "max_training_runs": "TOTAL training executions, the real compute budget",
}


def explicit_flags(argv=None) -> set:
    """Which options the user actually typed.

    argparse cannot distinguish "defaulted to 8" from "you asked for 8", and a
    profile that overrode an explicit request would be a bug, so this reads the
    command line directly.
    """
    argv = sys.argv[1:] if argv is None else argv
    out = set()
    for tok in argv:
        if tok.startswith("--"):
            out.add(tok[2:].split("=", 1)[0].replace("-", "_"))
    return out


def resolve(args, argv=None) -> dict:
    """Apply the profile to `args` in place; return the resolved settings.

    Returns {name: (value, source)} where source is 'cli', 'profile' or
    'default'.
    """
    given = explicit_flags(argv)
    resolved: dict = {}
    active = bool(getattr(args, "competition", False))

    for key, prof_value in COMPETITION_PROFILE.items():
        cur = getattr(args, key, None)
        if key in given:
            resolved[key] = (cur, "cli")
            continue
        if active:
            setattr(args, key, prof_value)
            resolved[key] = (prof_value, "profile")
        else:
            resolved[key] = (cur, "default")
    return resolved


def validate(args, resolved: dict) -> list:
    """Reasons to refuse this configuration BEFORE any LLM or training spend."""
    problems = []
    active = bool(getattr(args, "competition", False))

    if active and getattr(args, "allow_locked_options", False):
        problems.append(
            "--competition with --allow-locked-options: the competition profile "
            "must not run leakage-sensitive or locked menu options. Choose one.")
    if active and getattr(args, "smoke", False):
        problems.append(
            "--competition with --smoke: smoke caps the run to a plumbing check, "
            "which is not a scored run. Choose one.")
    if active and getattr(args, "inject_error_at", None) is not None:
        problems.append(
            "--competition with --inject-error-at: that deliberately breaks an "
            "iteration to exercise the debug path, so it must not be part of a "
            "scored run.")
    if active and getattr(args, "max_iterations", None) != 50:
        problems.append(
            "--competition requires --max-iterations 50: 50 is the organizer "
            "cap. Use the default/research profile for a shorter diagnostic run.")
    if active and getattr(args, "wall_clock_limit_h", None) != 6.0:
        problems.append(
            "--competition requires --wall-clock-limit-h 6.0: 6h is the "
            "organizer ceiling. Use the default/research profile for a shorter "
            "diagnostic run.")

    n_iter = getattr(args, "max_iterations", 0) or 0
    n_train = getattr(args, "max_training_runs", 0) or 0
    if n_train and n_train < n_iter:
        problems.append(
            f"--max-training-runs ({n_train}) is below --max-iterations "
            f"({n_iter}). Every iteration needs at least one training run, and "
            f"a paired confirmation needs six.")
    if (getattr(args, "max_spend_usd", 0) or 0) <= 0:
        problems.append("--max-spend-usd must be positive.")
    if (getattr(args, "wall_clock_limit_h", 0) or 0) <= 0:
        problems.append("--wall-clock-limit-h must be positive.")
    return problems


def render(args, resolved: dict) -> str:
    active = bool(getattr(args, "competition", False))
    L = ["=" * 74,
         f"RESOLVED CONFIGURATION — profile: "
         f"{COMPETITION if active else DEFAULT}",
         "=" * 74]
    caps = ("data_tools", "research_state", "feature_discovery", "n_candidates",
            "min_branching_iterations")
    L.append("  capabilities")
    for k in caps:
        v, src = resolved.get(k, (getattr(args, k, None), "default"))
        L.append(f"    {k:<28}{str(v):<10}[{src}]   {_WHY.get(k, '')}")
    L.append("  resource caps")
    for k in ("max_iterations", "max_training_runs", "wall_clock_limit_h",
              "max_spend_usd", "exec_timeout", "draft_count", "seed"):
        v, src = resolved.get(k, (getattr(args, k, None), "default"))
        L.append(f"    {k:<28}{str(v):<10}[{src}]   {_WHY.get(k, '')}")
    L.append("  safety")
    L.append(f"    {'convergence':<28}{'official':<10}[profile] "
             f"epsilon=0.002, N=3; no research gate may defer it")
    L.append(f"    {'allow_locked_options':<28}"
             f"{str(getattr(args, 'allow_locked_options', False)):<10}"
             f"[cli]     locked/leakage-sensitive options stay OFF in this profile")
    L.append(f"    {'fresh':<28}{str(getattr(args, 'fresh', False)):<10}[cli]"
             f"     archives previous search logs; submission artifacts survive")
    L.append("=" * 74)
    return "\n".join(L)
