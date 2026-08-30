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

import math
import statistics

NOISE = 0.0008          # the official baseline's own 5-seed std

# Severity levels, in the order a reader should care about them.
FATAL, WARN, NOTE = "FATAL", "WARN", "NOTE"


def _expected_max_of_n(n: int, sd: float) -> float:
    """Expected maximum of n independent zero-mean draws, in score units.

    This is the number that matters when someone reports "the best of n". Even
    with NO real effect, the best of five noisy comparisons sits about 1.16
    standard deviations above zero -- which at this project's noise floor is
    +0.00093, comfortably large enough to look like a discovery.
    """
    if n <= 1:
        return 0.0
    # Blom's approximation to the expected value of the largest order statistic
    return sd * (math.sqrt(2 * math.log(n)) -
                 (math.log(math.log(max(n, 3))) + math.log(4 * math.pi))
                 / (2 * math.sqrt(2 * math.log(n))))


def audit_comparison(delta: float, n_seeds: int = 1, paired: bool = False,
                     n_candidates_compared: int = 1,
                     selected_on_eval_data: bool = False,
                     seed_sd: float | None = None,
                     confirmed_out_of_sample: bool = False) -> dict:
    """Grade one claimed improvement by how it was MEASURED, not by its size.

    delta                    the claimed improvement in primary
    n_seeds                  how many seeds it rests on
    paired                   whether arms shared seeds (halves the variance)
    n_candidates_compared    how many variants were compared before this won
    selected_on_eval_data    was the winner chosen using the same data it is
                             now being scored on?
    confirmed_out_of_sample  was it re-checked on held-out or resampled data?
    """
    sd = seed_sd if seed_sd is not None else NOISE
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
        infl = _expected_max_of_n(n_candidates_compared, sd)
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


def convergence_epsilon(n_iterations: int, seed_sd: float | None = None) -> float:
    """How much the running BEST drifts upward over n iterations by luck alone.

    A search loop that stops when "the best has not improved by more than eps"
    is making a claim about noise, so eps has to be calibrated to noise. Set it
    too high and the loop quits while real effects are still findable; set it
    below the drift of a running maximum and the loop can never converge at all,
    because the best-so-far climbs on its own.

    The right threshold is exactly the selection drift: stop once the best has
    gained no more than picking the max of n noisy draws would have given you
    anyway. That is the same order statistic `selection_pressure` reports.

    This matters concretely here. The original hand-picked eps=0.002 is 2.5
    sigma, while the drift over 3 iterations is 0.60 sigma -- so the loop
    demanded four times more progress than the calibrated bar, and quit on
    differences larger than any effect this benchmark still has to offer.
    """
    return _expected_max_of_n(max(n_iterations, 2),
                              seed_sd if seed_sd is not None else NOISE)


def selection_pressure(n_candidates_compared: int,
                       seed_sd: float | None = None) -> dict:
    """How large an apparent gain does picking the best of n produce by itself?

    Use before believing any "we tried n things and this was best" number.
    """
    sd = seed_sd if seed_sd is not None else NOISE
    e = _expected_max_of_n(n_candidates_compared, sd)
    return {"n": n_candidates_compared, "expected_max_delta": round(e, 5),
            "expected_max_sigma": round(e / NOISE, 2),
            "reading": (f"choosing the best of {n_candidates_compared} noisy "
                        f"comparisons yields about {e / NOISE:+.2f} sigma even "
                        f"when nothing works; a real effect must clear this "
                        f"before it means anything")}


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
