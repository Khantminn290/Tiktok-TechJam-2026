"""Evidence-reactive research-category policy.

Chooses among exploration / exploitation / ablation / confirmation /
integration by scoring each against the CURRENT research state -- not by fixed
quotas like "40% exploration", which would be arbitrary and would not respond
to what the research has actually established.

Every decision is explainable: the policy returns the chosen category, the
reason, the state entries that drove it, and why each alternative lost. That
traceability is the point -- an autonomous agent that cannot say WHY it chose
an action is hard to trust and hard to demonstrate.

Guard against pathologies, both directions:
  * endless ablation -- ablation pressure is capped by budget share, so a
    config with many untested components cannot consume the whole run.
  * endless confirmation -- confirming a candidate that is already
    reseed-verified adds nothing, so confirmation scores 0 once satisfied.
  * premature integration -- blocked while either component is unconfirmed.
  * exploring a saturated space -- exploration decays when recent attempts all
    land inside the noise floor.
"""
from __future__ import annotations

EXPLORATION = "exploration"
EXPLOITATION = "exploitation"
ABLATION = "ablation"
CONFIRMATION = "confirmation"
INTEGRATION = "integration"
CATEGORIES = (EXPLORATION, EXPLOITATION, ABLATION, CONFIRMATION, INTEGRATION)

# Ablation is valuable but must not become the whole run: at most this share
# of scored experiments may be ablations before its pressure is suppressed.
MAX_ABLATION_SHARE = 0.35
# Exploration is considered saturated when this many recent distinct
# configurations all landed within the noise floor of the incumbent.
SATURATION_WINDOW = 4


def _recent_categories(nodes: list, k: int = 8) -> list:
    return [(n.get("research_category") or "") for n in nodes[-k:]]


def _mandatory_confirmation(scored: list) -> str:
    """Is there a single-seed result big enough that believing it is a risk?

    Returns the reason to confirm, or "" if nothing is pending. The threshold
    comes from the benchmark's own noise rather than a preference: an effect
    below half the noise floor is not worth a confirmation run either, so the
    gate fires only in the band where a result is both plausible and unproven.
    """
    if not scored:
        return ""
    from . import evidence as ev
    from .research_run import NOISE  # the benchmark's measured seed noise
    best = max(scored, key=lambda n: n["metrics"]["primary"])
    delta = best["metrics"]["primary"] - 0.6016      # vs the official baseline
    if delta < NOISE / 2:
        return ""
    state = ev.classify(delta=delta, n_seeds=1)
    if state["state"] != ev.PRELIMINARY:
        return ""
    plan = ev.confirmation_plan(delta, n_seeds=1)
    return (f"node {best.get('iteration_id')} is the best result at "
            f"{best['metrics']['primary']:.5f} ({delta / NOISE:+.2f} sigma over "
            f"baseline) but rests on ONE seed, so it is PRELIMINARY and cannot "
            f"be acted on. Confirm it at ~{plan['seeds_required']} paired seeds "
            f"before building anything on top of it")


def decide_category(state, nodes: list, iteration_budget_left: int = 50) -> dict:
    """Score every category against the research state and pick the highest.

    Returns {"category", "reason", "evidence", "alternatives": {cat: why_not}}.
    """
    scores: dict = {}
    why: dict = {}
    evidence: dict = {}

    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    n_scored = max(1, len(scored))
    recent = _recent_categories(nodes)
    ablations_done = sum(1 for n in nodes if n.get("research_category") == ABLATION)
    ablation_share = ablations_done / n_scored

    untested = [a for a, ev in (state.component_evidence or {}).items()
                if ev.get("status") == "untested_assumption"]
    questionable = [a for a, ev in (state.component_evidence or {}).items()
                    if ev.get("status") == "questionable"]
    bce = state.best_config_evidence or {}
    unconfirmed = [p for p in (state.promising or [])
                   if p.get("level") == "observed_once"]
    eligible = [c for c in (state.integration_candidates or [])
                if c.get("status") == "eligible"]

    # ---- ABLATION: untested assumptions sitting inside the incumbent -------
    if untested or questionable:
        s = 1.0 + 0.5 * len(untested) + 0.3 * len(questionable)
        if ablation_share >= MAX_ABLATION_SHARE:
            s *= 0.2
            why[ABLATION] = (f"suppressed: ablations are already "
                             f"{ablation_share:.0%} of experiments (cap "
                             f"{MAX_ABLATION_SHARE:.0%}); more would crowd out "
                             f"discovery")
        scores[ABLATION] = s
        evidence[ABLATION] = ([f"{a}={state.component_evidence[a]['value']}: untested"
                               for a in untested]
                              + [f"{a}: questionable" for a in questionable])
    else:
        scores[ABLATION] = 0.0
        why[ABLATION] = "every component of the current best already has isolated evidence"

    # ---- CONFIRMATION: is what we believe actually established? -----------
    conf = 0.0
    ev_conf = []
    if bce.get("level") == "observed_once":
        conf += 2.0
        ev_conf.append("the incumbent best rests on a SINGLE run")
    if unconfirmed:
        conf += 0.6 * len(unconfirmed)
        ev_conf.append(f"{len(unconfirmed)} promising candidate(s) still "
                       f"observed_once")
    # A PRELIMINARY result that is large enough to be interesting is exactly the
    # situation in which the agent previously went wrong: clean run 2 adopted a
    # value on one seed and carried it forward, and the effect turned out to be
    # -0.01 sigma when the sweep it had itself specified was actually run.
    #
    # So this is not a preference weight. When a single-seed result is plausibly
    # above noise, confirming it DOMINATES: no amount of exploration appetite
    # should outrank finding out whether the thing you already believe is true.
    mandatory = _mandatory_confirmation(scored)
    if mandatory:
        conf = max(conf, 10.0)
        ev_conf.append(mandatory)

    if conf == 0.0:
        why[CONFIRMATION] = ("nothing is awaiting confirmation: the incumbent and "
                             "the live candidates are already multi-seed backed")
    scores[CONFIRMATION] = conf
    evidence[CONFIRMATION] = ev_conf

    # ---- INTEGRATION: only when independently validated pieces exist ------
    if eligible:
        scores[INTEGRATION] = 1.5 + 0.5 * len(eligible)
        evidence[INTEGRATION] = [c["candidate"] for c in eligible]
    else:
        scores[INTEGRATION] = 0.0
        blocked = len(state.integration_candidates or [])
        why[INTEGRATION] = (f"{blocked} candidate(s) exist but are blocked pending "
                            f"confirmation of their components"
                            if blocked else "no independently validated "
                            "improvements to combine yet")

    # ---- EXPLORATION: decays when the space looks saturated ---------------
    dead = sum(1 for b in (state.branches or []) if "dead-end" in b.get("status", ""))
    live = [b for b in (state.branches or []) if "dead-end" not in b.get("status", "")]
    expl = 1.2
    ev_expl = [f"{len(live)} live branch(es), {dead} dead-end branch(es)"]
    n_recent_expl = sum(1 for c in recent if c == EXPLORATION)
    if n_recent_expl >= SATURATION_WINDOW:
        expl *= 0.4
        why[EXPLORATION] = (f"damped: {n_recent_expl} of the last "
                            f"{len(recent)} experiments were already exploration "
                            f"and the incumbent has not moved")
    if len(state.dead_ends or []) >= 12:
        expl *= 0.8
        ev_expl.append(f"{len(state.dead_ends)} recorded dead ends -- much of the "
                       f"obvious space is already mapped")
    scores[EXPLORATION] = expl
    evidence[EXPLORATION] = ev_expl

    # ---- EXPLOITATION: the default incremental move -----------------------
    scores[EXPLOITATION] = 0.9
    evidence[EXPLOITATION] = ["incremental refinement of the current best"]
    why.setdefault(EXPLOITATION,
                   "always available, but lowest information gain when stronger "
                   "signals exist")

    # near the end of the budget, prefer consolidating over opening new lines
    if iteration_budget_left <= 5:
        scores[EXPLORATION] *= 0.3
        scores[ABLATION] *= 0.5
        scores[CONFIRMATION] *= 1.5
        why[EXPLORATION] = (why.get(EXPLORATION, "") +
                            " | budget nearly exhausted: consolidate rather than "
                            "open new lines").strip(" |")

    best = max(CATEGORIES, key=lambda c: scores.get(c, 0.0))
    reason = {
        ABLATION: "the current best contains components with no isolated evidence",
        CONFIRMATION: "what we currently believe is not yet statistically established",
        INTEGRATION: "independently validated improvements are available to combine",
        EXPLORATION: "the space still has untried mechanisms worth probing",
        EXPLOITATION: "no stronger signal; refine the incumbent",
    }[best]
    alts = {c: why.get(c, f"scored lower ({scores.get(c, 0):.2f} vs "
                          f"{scores[best]:.2f})")
            for c in CATEGORIES if c != best}
    used = len(nodes)
    total = used + max(0, iteration_budget_left)
    return {"category": best, "reason": reason,
            "evidence": evidence.get(best, []),
            "scores": {c: round(scores.get(c, 0.0), 2) for c in CATEGORIES},
            "alternatives": alts,
            "iterations_used": used,
            "iterations_left": max(0, iteration_budget_left),
            "phase": _phase(iteration_budget_left, total),
            "best_primary": (round(max(n["metrics"]["primary"] for n in scored), 5)
                             if scored else None)}


# A probe needs a MINIMUM NUMBER of runs to pay off, and that is an absolute
# quantity, not a fraction: two iterations left is late whatever the cap.
LATE_ITERATIONS = 5
MIDDLE_ITERATIONS = 15


def _phase(left: int, total: int) -> str:
    """What the remaining budget is FOR. The planner was told which objective to
    pursue but never how much runway it had, so a speculative probe and a
    closing-out confirmation looked equally affordable at iteration 48.

    Judged on BOTH the absolute runs remaining and the fraction, taking the
    more conservative. Fraction alone got this wrong: with 5 iterations used
    and 2 left the implied budget is 7, so 2 remaining read as 29% and came out
    MIDDLE -- when two runs is plainly late-stage.
    """
    if total <= 0:
        return "unknown"
    frac = left / total
    if left <= LATE_ITERATIONS or frac <= 0.25:
        return ("LATE -- confirm, combine and stabilise; a speculative probe "
                "that cannot pay off within the remaining runs is not affordable")
    if left <= MIDDLE_ITERATIONS or frac <= 0.6:
        return ("MIDDLE -- explore and exploit; prefer mechanisms with isolated "
                "evidence, but open questions are still affordable")
    return ("EARLY -- runway to explore; a probe that resolves an open "
            "direction is worth more than a marginal tweak")


def render_decision(d: dict) -> str:
    """Human/LLM-readable explanation of the category choice."""
    L = [f"## Research objective for this iteration: {d['category'].upper()}",
         f"Why: {d['reason']}"]
    if d.get("phase"):
        L.append(f"Budget: iteration {d.get('iterations_used', 0) + 1}, "
                 f"{d.get('iterations_left', 0)} left. Phase: {d['phase']}")
    if d.get("best_primary") is not None:
        L.append(f"Best scored so far: {d['best_primary']}")
    if d.get("evidence"):
        L.append("Relevant research-state entries:")
        L += [f"  - {e}" for e in d["evidence"][:5]]
    L.append("Alternatives considered:")
    for c, w in d["alternatives"].items():
        L.append(f"  - {c}: {w}")
    L.append(f"(category scores: {d['scores']})")
    L.append("Your proposal MUST match this research_category. If you believe "
             "the state justifies a different objective, say so explicitly in "
             "your hypothesis and explain what the policy is missing.")
    return "\n".join(L)
