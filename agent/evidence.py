"""How much is this result allowed to count for?

Written because of a specific, documented mistake. In clean run 2 the agent
derived a good hypothesis from data it measured itself, specified the right way
to test it -- a paired sweep over four values -- then ran ONE seed and carried
the value forward as though it were settled. When the sweep it had specified was
actually run at 5 paired seeds, the effect was -0.01 sigma. The hypothesis was
false, and a single seed had been enough to make it look true.

The failure is not arithmetic. It is that nothing in the system distinguished
"a number I have seen once" from "a thing I know", so the two were treated
identically the moment the number looked good.

So evidence gets an explicit state, and the state is computed from HOW the
number was obtained rather than from how large it is:

    UNTESTED      no measurement exists
    HYPOTHESIS    stated and falsifiable, not yet measured
    PROBED        measured cheaply/indirectly (no training run)
    PRELIMINARY   measured once; a single draw. The ceiling for one seed.
    UNCONFIRMED   repeated but still not separable from noise
    CONFIRMED     survives paired repetition and selection pressure
    REJECTED      measured and does not hold
    REDUNDANT     real alone, but adds nothing on top of what is already there

The invariant that matters, enforced in `classify` and tested:

    A single-seed result can NEVER become CONFIRMED. Its ceiling is PRELIMINARY.

REDUNDANT is a state, not a failure. An intervention can be real and still
worth nothing in combination, so "it works but adds nothing here" has to be
expressible -- otherwise it gets recorded as either a win or a dead end, and
both are wrong.
"""
from __future__ import annotations

import math
import os
import sys

_RUNTIME = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "runtime")
if _RUNTIME not in sys.path:
    sys.path.insert(0, _RUNTIME)

from research_tools import (  # noqa: E402
    NOISE, audit_comparison, expected_max_of_n,
)

UNTESTED = "UNTESTED"
HYPOTHESIS = "HYPOTHESIS"
PROBED = "PROBED"
PRELIMINARY = "PRELIMINARY"
UNCONFIRMED = "UNCONFIRMED"
CONFIRMED = "CONFIRMED"
REJECTED = "REJECTED"
REDUNDANT = "REDUNDANT"

STATES = (UNTESTED, HYPOTHESIS, PROBED, PRELIMINARY, UNCONFIRMED, CONFIRMED,
          REJECTED, REDUNDANT)

# States that justify changing what gets submitted.
ACTIONABLE = (CONFIRMED,)

# How strong the paired evidence must look before an effect is CONFIRMED.
# Not a magic number: t=2.5 on paired seeds is roughly the point at which the
# effect exceeds the drift a running maximum shows by luck, which is the
# comparison that actually matters in a search loop.
T_CONFIRM = 2.5


def seeds_needed(delta: float, sd: float | None = None,
                 t_target: float = T_CONFIRM, cap: int = 24) -> int:
    """Paired seeds required to resolve an effect of this size.

    Derived from the benchmark's own noise, not chosen: to get |t| >= t_target
    we need n such that |delta| / (sd/sqrt(n)) >= t_target.
    """
    sd = NOISE if sd is None else sd
    if not delta:
        return cap
    n = math.ceil((t_target * sd / abs(delta)) ** 2)
    # Floor of 3, not 2: with two points the spread estimate rests on a single
    # difference, so a "confirmation" at n=2 confirms almost nothing however
    # large the effect looks.
    return max(3, min(int(n), cap))


def classify(delta: float | None, n_seeds: int = 0, paired: bool = False,
             n_candidates_compared: int = 1, selected_on_eval_data: bool = False,
             confirmed_out_of_sample: bool = False, trained: bool = True,
             redundant_with: str | None = None,
             seed_sd: float | None = None) -> dict:
    """Assign an evidence state, and say what would move it forward."""
    if redundant_with:
        return _out(REDUNDANT, delta, n_seeds,
                    f"real on its own but adds nothing on top of "
                    f"{redundant_with}",
                    "Measure it in the configuration you would actually ship, "
                    "not in isolation.")
    if delta is None:
        return _out(UNTESTED if n_seeds == 0 else HYPOTHESIS, delta, n_seeds,
                    "no measurement yet",
                    "State the measurement that would distinguish your "
                    "hypotheses, then run it.")
    if not trained:
        return _out(PROBED, delta, n_seeds,
                    "measured without a training run",
                    "A cheap probe can rule things out. It cannot confirm a "
                    "pipeline change; that needs paired training runs.")

    sd = NOISE if seed_sd is None else seed_sd
    need = seeds_needed(delta, sd)
    floor = expected_max_of_n(n_candidates_compared, sd)

    # THE INVARIANT: one draw is one draw, whatever it says.
    if n_seeds <= 1:
        return _out(PRELIMINARY, delta, n_seeds,
                    f"a single seed; seed noise is {sd}, so this cannot be "
                    f"separated from luck",
                    f"Repeat with ~{need} PAIRED seeds (same seeds in both arms) "
                    f"before believing it.")

    if selected_on_eval_data and not confirmed_out_of_sample:
        return _out(UNCONFIRMED, delta, n_seeds,
                    "the winner was chosen on the same data now scoring it, so "
                    "this measures fit to that split",
                    "Re-check on held-out users or by resampling.")

    if n_candidates_compared > 1 and abs(delta) <= floor:
        return _out(UNCONFIRMED, delta, n_seeds,
                    f"best of {n_candidates_compared}: selection alone yields "
                    f"about {floor:+.5f} ({floor / NOISE:+.2f} sigma), which this "
                    f"{delta:+.5f} does not clear",
                    "Test the single pre-stated candidate on fresh seeds, not "
                    "the winner of the comparison.")

    t = abs(delta) / (sd / math.sqrt(n_seeds))

    # Two DIFFERENT questions, which this used to conflate:
    #
    #   is it real?        -> t against the variance actually measured
    #   is it worth much?  -> the effect against the benchmark's noise floor
    #
    # The old rule answered only the second and applied it to everything, so a
    # deterministic post-process measured paired on identical predictions was
    # REJECTED at t=9.60 with 16/16 wins, because its +0.00017 sat under half
    # the SEED-noise floor. But seed noise is not that comparison's variance --
    # its measured spread was 0.000055, fourteen times smaller. An effect can be
    # statistically certain and practically small, and saying so is more useful
    # than calling it absent.
    measured = seed_sd is not None and seed_sd > 0
    if abs(delta) < NOISE / 2:
        if measured and t >= T_CONFIRM:
            return _out(CONFIRMED, delta, n_seeds,
                        f"{delta:+.5f} is small against the seed-noise floor "
                        f"({NOISE}), but this comparison's own spread is "
                        f"{sd:.6f}, giving t={t:.2f} over {n_seeds} paired "
                        f"measurements -- the effect is real, just small",
                        "Worth taking if it is free; do not expect it to move "
                        "the headline much.")
        return _out(REJECTED, delta, n_seeds,
                    f"{delta:+.5f} is under half the noise floor and is not "
                    f"separable from it",
                    "Stop spending runs here; the effect is absent, not small.")
    if t < T_CONFIRM:
        return _out(UNCONFIRMED, delta, n_seeds,
                    f"t={t:.2f} over {n_seeds} seeds is below the {T_CONFIRM} "
                    f"needed",
                    f"Add seeds (about {need} total) or accept it is not "
                    f"separable.")
    if delta < 0:
        return _out(REJECTED, delta, n_seeds,
                    f"repeatably worse ({delta:+.5f}, t={t:.2f})",
                    "Record the scope in which it lost; do not ban it globally.")
    return _out(CONFIRMED, delta, n_seeds,
                f"{delta:+.5f} ({delta / NOISE:+.2f} sigma) over {n_seeds} "
                f"{'paired ' if paired else ''}seeds, t={t:.2f}",
                "Check it still holds in the configuration you would ship "
                "(a solo gain can vanish after ensembling).")


def _out(state: str, delta, n_seeds: int, why: str, next_step: str) -> dict:
    return {"state": state, "delta": delta, "n_seeds": n_seeds,
            "sigma": round(delta / NOISE, 2) if delta is not None else None,
            "why": why, "next_step": next_step,
            "actionable": state in ACTIONABLE}


def confirmation_plan(delta: float, n_seeds: int = 1,
                      seed_sd: float | None = None) -> dict:
    """What it would take to promote this result, concretely."""
    need = seeds_needed(delta, seed_sd)
    return {"current_seeds": n_seeds, "seeds_required": need,
            "additional_seeds": max(0, need - n_seeds), "paired": True,
            "instruction": (f"Run both arms at the SAME {need} seeds and compare "
                            f"per-seed differences. Pairing removes the seed "
                            f"variance that otherwise dominates an effect this "
                            f"size.")}


def render(e: dict) -> str:
    L = [f"EVIDENCE: {e['state']}"
         + (f"  ({e['delta']:+.5f}, {e['sigma']:+.2f} sigma, "
            f"{e['n_seeds']} seed(s))" if e.get("delta") is not None else ""),
         f"  why:  {e['why']}",
         f"  next: {e['next_step']}"]
    if not e["actionable"]:
        L.append("  This does NOT justify changing the submitted system.")
    return "\n".join(L)


def render_for_prompt() -> str:
    return "\n".join([
        "## EVIDENCE STATES — what a result is allowed to count for",
        "Every measurement you make gets one of these, and the state is decided "
        "by HOW you measured, not by how good the number looks:",
        "  UNTESTED / HYPOTHESIS  nothing measured yet",
        "  PROBED                 measured without a training run",
        "  PRELIMINARY            measured ONCE. This is the ceiling for a single "
        "seed, no matter how large the gain.",
        "  UNCONFIRMED            repeated, still not separable from noise",
        "  CONFIRMED              survives paired repetition and selection "
        "pressure. Only this justifies changing what we submit.",
        "  REJECTED               measured; does not hold",
        "  REDUNDANT              real alone, adds nothing on top of what is "
        "already in the pipeline",
        "",
        "A single-seed improvement is PRELIMINARY. It is not a discovery, it is "
        "a reason to run the confirmation. If you adopt a value on one seed you "
        "have not learned anything -- you have guessed, and the record will say "
        "so.",
    ])
