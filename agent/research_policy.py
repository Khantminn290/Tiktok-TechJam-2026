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
    return {"category": best, "reason": reason,
            "evidence": evidence.get(best, []),
            "scores": {c: round(scores.get(c, 0.0), 2) for c in CATEGORIES},
            "alternatives": alts}


def render_decision(d: dict) -> str:
    """Human/LLM-readable explanation of the category choice."""
    L = [f"## Research objective for this iteration: {d['category'].upper()}",
         f"Why: {d['reason']}"]
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
