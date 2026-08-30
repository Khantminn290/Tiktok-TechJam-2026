"""Is this experiment measuring what it claims to measure?

Distilled from the question the teacher research run kept having to ask, and
kept getting wrong until it was asked explicitly. Three separate results turned
on it:

  * a "+0.46 sigma" aggregation improvement was the best of FIVE rules compared
    on one validation set; resampling put it at -0.06 sigma (10/24 wins).
  * the stopping epoch is an argmax over ~40 validation evaluations, so the
    number it produces is fitted to the set that scored it.
  * a working method sat REJECTED in the codebase because its guard compared it
    against a checkpoint chosen on the same validation set.

None of those are detectable from a score. They are properties of how the
comparison was set up, so they need their own check.

This module is deliberately advisory, not blocking. It computes what selection
pressure a comparison was under and how much evidence it actually carries, and
returns findings the agent can act on. It cannot know intent, so it never
refuses an experiment -- it tells the agent what its number is worth.
"""
from __future__ import annotations

import os
import sys

# One implementation, two access paths. These are defined in
# runtime/research_tools.py because that is the module GENERATED CODE can
# import; keeping a second copy here is how the orchestrator and the
# experiment scripts would quietly drift apart.
_RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runtime")
if _RUNTIME not in sys.path:
    sys.path.insert(0, _RUNTIME)

from research_tools import (  # noqa: E402
    FATAL, NOISE, NOTE, WARN, audit_comparison, convergence_epsilon,
    expected_max_of_n, free_recombination, redundancy, selection_pressure,
    selection_rule_test,
)

# Kept under its original private name: existing callers and tests reference it.
_expected_max_of_n = expected_max_of_n

__all__ = ["NOISE", "FATAL", "WARN", "NOTE", "audit_comparison",
           "selection_pressure", "convergence_epsilon", "expected_max_of_n",
           "selection_rule_test", "free_recombination", "redundancy",
           "render", "render_for_prompt"]


def render(a: dict) -> str:
    L = [f"## VALIDITY AUDIT — {a['delta']:+.5f} ({a['sigma']:+.2f} sigma): "
         f"{a['severity']}", a["verdict"]]
    for f in a["findings"]:
        L.append(f"  [{f['level']}] {f['message']}")
    return "\n".join(L)


def render_for_prompt() -> str:
    """How the agent is told to use this."""
    return "\n".join([
        "## CHECK YOUR OWN METHODOLOGY (validity.audit_comparison)",
        "Before believing any improvement -- including your own -- ask what the "
        "number is worth given how it was measured:",
        "  - how many seeds does it rest on, and were the arms paired?",
        "  - how many variants were compared before this one won? Picking the "
        "best of several noisy comparisons produces an apparent gain on its own.",
        "  - was the winner CHOSEN using the same data it is now scored on? An "
        "argmax over validation evaluations is fitted to validation.",
        "  - does the effect survive out-of-sample or resampled confirmation?",
        "  - does it still hold after the step that comes later in the pipeline "
        "(for example, does a single-model gain survive ensembling)?",
        "A number that fails these is not a small result; it is not evidence.",
    ])
