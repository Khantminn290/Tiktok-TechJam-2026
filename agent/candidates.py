"""Candidate generation, deterministic scoring, and the decision trace.

AUDIT FINDING that motivated this module: Path B usage was 0 not because the
policy rejected it but because it was never GENERATED. The planner made ONE
structured_call returning ONE proposal; research_policy chose only a
*category*. There was no candidate set, no scoring, and no rejection step --
so "why was Path B rejected?" had no answer, and forcing Path B through prompt
pressure would have been treating a symptom.

Fix: the planner now emits K candidates in a SINGLE call (cheap: more output
tokens, not K× the calls), and selection is DETERMINISTIC here. That makes the
Path A/B decision visible, scoreable and auditable, and it is what makes a
counterfactual replay possible at all.

Scoring, deliberately simple and inspectable:

    utility = expected_gain x P(success) x novelty x redundancy x info / cost

where `info` is the VALUE OF INFORMATION: an experiment earns an iteration
either because it may raise the score, or because it resolves an uncertainty
that decides where the remaining budget goes. Pure expected-gain scoring cannot
express the second, so a cheap experiment settling whether a whole direction is
viable used to lose to a marginally better tweak -- even though it redirects
every iteration after it. It is graded from the research frontier's own status
(UNEXPLORED / UNCERTAIN / CONTRADICTORY) and decays as the budget closes out.

with hard gates applied BEFORE scoring, because a high utility computed from a
fabricated premise is worse than no score:
  * plausibility  -- novel+random is rejected; novel+plausible is not
  * duplication   -- semantically equivalent to a prior experiment
  * dead ends     -- overlaps a recorded negative finding
  * saturation    -- branch whose recent returns have collapsed

Everything here is deterministic. Per the efficiency rule, the LLM proposes;
arithmetic, duplicate detection, saturation and convergence stay in code.
"""
from __future__ import annotations

import json
import re

# Effect sizes below this are indistinguishable from seed noise on this task.
NOISE_FLOOR = 0.0008
# Cost model in "training-run equivalents" -- relative, not seconds.
COST = {"A": 1.0, "B": 1.6}          # custom code costs more (write + debug risk)
CATEGORY_COST = {"confirmation": 5.0, "integration": 1.2, "ablation": 1.0,
                 "exploration": 1.0, "exploitation": 1.0}

_STOP = {"the", "a", "an", "of", "to", "and", "or", "for", "with", "on", "in",
         "by", "is", "are", "be", "this", "that", "it", "as", "at", "from",
         "we", "our", "should", "could", "may", "will", "can", "add", "adding",
         "use", "using", "test", "testing", "try", "trying", "improve", "model"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z_]+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


def semantic_similarity(a: str, b: str) -> float:
    """Blend of Jaccard and overlap coefficient over content words.

    Measured calibration on real examples from this project:
        reworded duplicate  jaccard 0.60 / overlap 0.75
        reworded duplicate  jaccard 0.29 / overlap 0.50
        dead-end restatement jaccard 0.21 / overlap 0.50
        unrelated idea       0.00 / 0.00
        LEGITIMATE follow-up jaccard 0.60 / overlap 1.00   <-- highest of all

    That last row is why similarity alone must NEVER hard-block: an honest
    extension ("+ temporal decay") looks MORE similar than a true duplicate,
    because it contains the parent idea entirely. Lexical similarity cannot
    distinguish "reworded" from "extended" -- that is a semantic judgement.
    So this score drives a graded PENALTY, and only an exact configuration
    repeat (unambiguous) is a hard gate.
    """
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    jac = len(ta & tb) / len(ta | tb)
    ovl = len(ta & tb) / min(len(ta), len(tb))
    return max(jac, 0.75 * ovl)


class Candidate:
    def __init__(self, raw: dict, index: int):
        self.index = index
        self.raw = raw or {}
        self.hypothesis = str(self.raw.get("hypothesis", ""))
        self.path = str(self.raw.get("implementation_path", "A")).upper()
        self.category = str(self.raw.get("research_category", "exploration")).lower()
        self.menu_choices = self.raw.get("menu_choices") or {}
        self.mechanism = str(self.raw.get("mechanism")
                             or (self.raw.get("rationale") or {}).get(
                                 "why_expected_to_help", ""))
        self.grounded_in = str((self.raw.get("rationale") or {}).get("grounded_in", ""))
        # self-reported, then calibrated against measured history
        try:
            self.claimed_gain = float(self.raw.get("expected_gain", 0.0) or 0.0)
        except (TypeError, ValueError):
            self.claimed_gain = 0.0
        self.text = f"{self.hypothesis} {self.mechanism} {json.dumps(self.menu_choices)}"
        # The CLAIM, without the elaboration. Comparing on full text alone lets a
        # verbose mechanism paragraph dilute an otherwise obvious repeat
        # (measured: the same dead-end overlap scores 0.375 on the claim but
        # 0.273 once a long mechanism is appended), so similarity is taken as
        # the max over both views.
        self.claim = f"{self.hypothesis} {json.dumps(self.menu_choices)}"
        # filled by score_candidates
        self.utility = 0.0
        self.gates: list = []
        self.parts: dict = {}

    @property
    def rejected(self) -> bool:
        return bool(self.gates)

    def as_dict(self) -> dict:
        return {"index": self.index, "path": self.path, "category": self.category,
                "hypothesis": self.hypothesis[:160],
                "menu_choices": self.menu_choices,
                "utility": round(self.utility, 5), "parts": self.parts,
                "rejected_by": self.gates}


# ---------------------------------------------------------------- gates ----
def _gate_plausibility(c: Candidate) -> str | None:
    """Novel + plausible is welcome. Novel + random is not: a hypothesis with
    no stated mechanism and no grounding is unfalsifiable noise."""
    if len(c.mechanism.strip()) < 25 and len(c.grounded_in.strip()) < 25:
        return ("no stated mechanism and no grounding -- novelty alone is not a "
                "reason to spend an iteration")
    return None


def _gate_duplicate(c: Candidate, history: list) -> str | None:
    """HARD gate only for an unambiguous exact-configuration repeat."""
    for h in history:
        if c.menu_choices and c.menu_choices == (h.get("menu_choices") or {}):
            return f"exact configuration already run (node {h.get('iteration_id')})"
    return None


def _sim_both_views(c: Candidate, other: str) -> float:
    """Max over three progressively wider views of the proposal.

    A repeat can hide at any granularity: the bare claim, the claim plus its
    configuration, or the full write-up. Taking the max means added prose can
    never launder a proposal past the gate, which is the failure mode that
    matters -- a missed duplicate costs a whole iteration, while a slightly
    over-eager similarity score only costs a graded penalty.
    """
    return max(semantic_similarity(c.hypothesis, other),
               semantic_similarity(c.claim, other),
               semantic_similarity(c.text, other))


def max_similarity(c: Candidate, history: list) -> tuple:
    """Highest similarity to any prior experiment, and which node."""
    best, node = 0.0, None
    for h in history:
        sim = _sim_both_views(c, f"{h.get('hypothesis','')} "
                                 f"{json.dumps(h.get('menu_choices') or {})}")
        if sim > best:
            best, node = sim, h.get("iteration_id")
    return best, node


DEAD_END_HARD = 0.36     # calibrated: a real dead-end restatement measures 0.375


def _gate_dead_end(c: Candidate, dead_ends: list) -> str | None:
    """Dead ends ARE hard-gated: re-deriving a measured negative is pure waste,
    and the dead-end text is specific enough that overlap is meaningful."""
    for d in dead_ends or []:
        if _sim_both_views(c, d) >= DEAD_END_HARD:
            return f"overlaps a recorded dead end: {d[:90]}"
    return None


def _gate_saturation(c: Candidate, saturated: set) -> str | None:
    key = (c.menu_choices.get("loss"), c.menu_choices.get("model"))
    if key in saturated:
        return f"branch {key} is saturated (recent returns collapsed)"
    return None


# --------------------------------------------------------------- scoring ---
def branch_stats(history: list) -> dict:
    """Per-branch return history. Deterministic; drives saturation."""
    br: dict = {}
    for h in history:
        if not h.get("metrics"):
            continue
        mc = h.get("menu_choices") or {}
        k = (mc.get("loss"), mc.get("model"))
        br.setdefault(k, []).append(h["metrics"]["primary"])
    out = {}
    for k, vals in br.items():
        running, gains = -1.0, []
        for v in vals:
            gains.append(max(0.0, v - running) if running > 0 else 0.0)
            running = max(running, v)
        recent = gains[-4:]
        out[k] = {"n": len(vals), "best": max(vals),
                  "recent_mean_gain": sum(recent) / max(1, len(recent)),
                  "total_gain": sum(gains)}
    return out


def saturated_branches(stats: dict, min_experiments: int = 4) -> set:
    """A branch is saturated when it has had a fair trial and its recent
    marginal returns have fallen below half the noise floor."""
    return {k for k, s in stats.items()
            if s["n"] >= min_experiments
            and s["recent_mean_gain"] < NOISE_FLOOR / 2}


INFO_UNEXPLORED = 0.60      # never run: any outcome is new information
INFO_UNCERTAIN = 0.35       # tried, evidence does not separate
INFO_CONTRADICTORY = 0.45   # experiments disagree beyond seed noise


def _information_value(c, frontier) -> float:
    """How much does running this RESOLVE, independent of what it scores?

    Reads the frontier status of every axis-option the candidate would set.
    Takes the MAXIMUM rather than the sum: one genuinely open question is what
    makes an experiment informative, and adding up several half-open ones would
    let a candidate that changes many axes at once outrank a clean single-axis
    test -- which is exactly the uncontrolled experiment we do not want.
    """
    if frontier is None or not c.menu_choices:
        return 0.0
    try:
        from .frontier import (UNEXPLORED, UNCERTAIN, CONTRADICTORY)
    except ImportError:
        return 0.0
    by = {d["direction"]: d for d in getattr(frontier, "directions", [])}
    best = 0.0
    for axis, value in c.menu_choices.items():
        d = by.get(f"{axis}={value}")
        if not d:
            continue
        best = max(best, {UNEXPLORED: INFO_UNEXPLORED,
                          UNCERTAIN: INFO_UNCERTAIN,
                          CONTRADICTORY: INFO_CONTRADICTORY}.get(d["status"], 0.0))
    return best


def score_candidates(cands: list, *, history: list, dead_ends: list,
                     state=None, budget_left: int = 50,
                     objective: str | None = None, frontier=None) -> list:
    """Deterministically score every candidate; attach gates and components."""
    stats = branch_stats(history)
    sat = saturated_branches(stats)
    n_hist = max(1, len(history))
    fail_rate_b = _path_failure_rate(history, "B")
    fail_rate_a = _path_failure_rate(history, "A")

    for c in cands:
        c.gates = [g for g in (
            _gate_plausibility(c),
            _gate_duplicate(c, history),
            _gate_dead_end(c, dead_ends),
            _gate_saturation(c, sat)) if g]
        sim_max, sim_node = max_similarity(c, history)

        # --- expected gain: self-report, shrunk toward the branch's measured
        # recent return so an optimistic claim cannot dominate scoring.
        key = (c.menu_choices.get("loss"), c.menu_choices.get("model"))
        measured = stats.get(key, {}).get("recent_mean_gain", NOISE_FLOOR)
        claimed = min(max(c.claimed_gain, 0.0), 0.01)      # cap wild claims
        gain = 0.5 * claimed + 0.5 * measured
        if key not in stats:                                # never-tried branch
            gain = max(gain, NOISE_FLOOR)                   # genuine option value

        # --- success probability from MEASURED path failure rates, not priors
        p_succ = 1.0 - (fail_rate_b if c.path == "B" else fail_rate_a)
        p_succ = min(0.98, max(0.25, p_succ))

        # --- novelty: distance from everything already tried
        sim = max([semantic_similarity(c.text,
                                       f"{h.get('hypothesis','')} "
                                       f"{json.dumps(h.get('menu_choices') or {})}")
                   for h in history] or [0.0])
        novelty = 1.0 - sim
        # exploration values novelty; exploitation/confirmation do not
        nov_w = {"exploration": 1.0, "ablation": 0.3, "integration": 0.3,
                 "exploitation": 0.2, "confirmation": 0.0}.get(c.category, 0.5)
        novelty_factor = 1.0 + nov_w * novelty

        cost = COST.get(c.path, 1.0) * CATEGORY_COST.get(c.category, 1.0)
        # late in the run, expensive experiments are worse value
        if budget_left <= 8:
            cost *= 1.5

        # --- value of INFORMATION, not just of score ---------------------
        # An experiment can be worth an iteration because it may raise the
        # score, OR because it resolves an uncertainty that decides where the
        # remaining budget goes. Pure expected-gain scoring cannot express the
        # second: a cheap experiment that settles whether an entire direction
        # is viable outranks a marginally better tweak, because it redirects
        # every iteration after it.
        #
        # Graded by the frontier's own status, so this is evidence-derived
        # rather than a hand-set exploration bonus:
        #   UNEXPLORED -- never run, so ANY result is new information
        #   UNCERTAIN  -- tried, but the evidence does not separate; a repeat
        #                 with a decisive design is what breaks the tie
        #   settled    -- KNOWN_GOOD/KNOWN_BAD/SATURATED already answer it
        # Deliberately capped: information is worth less than a real gain when
        # the budget is nearly spent, so it decays as the run closes out.
        info = _information_value(c, frontier)
        budget_frac = min(1.0, budget_left / 20.0)
        info_factor = 1.0 + info * budget_frac

        # Graded redundancy penalty instead of a hard block: an idea that
        # merely restates a previous one loses most of its value, while a
        # genuine extension of it (which looks just as similar lexically) is
        # only mildly discounted and can still win on its own merits.
        # Ramp starts at 0.25 (measured: unrelated ideas score 0.15, genuine
        # near-duplicates 0.375-0.60) and is floored at 0.55, because the score
        # cannot tell a reworded repeat from a real extension. A duplicate loses
        # enough utility to lose a close race; a high-value follow-up keeps
        # enough to win one on merit.
        redundancy = max(0.55, 1.0 / (1.0 + 1.5 * max(0.0, sim_max - 0.25)))
        c.parts = {"gain": round(gain, 5), "p_success": round(p_succ, 3),
                   "max_similarity": round(sim_max, 3),
                   "similar_to_node": sim_node,
                   "redundancy_factor": round(redundancy, 3),
                   "novelty": round(novelty, 3),
                   "novelty_factor": round(novelty_factor, 3),
                   "cost": round(cost, 2),
                   "information_value": round(info, 3),
                   "info_factor": round(info_factor, 3),
                   "branch_seen": key in stats,
                   "objective_match": (objective is None or c.category == objective)}
        u = (gain * p_succ * novelty_factor * redundancy * info_factor) / cost
        # honour the research objective without making it absolute
        if objective and c.category != objective:
            u *= 0.55
        c.utility = 0.0 if c.gates else u
    return sorted(cands, key=lambda c: -c.utility)


def _path_failure_rate(history: list, path: str) -> float:
    runs = [h for h in history
            if str(h.get("implementation_path", "")).upper() == path]
    if len(runs) < 3:                 # not enough evidence -> mild prior
        return 0.15 if path == "A" else 0.35
    fails = sum(1 for h in runs if h.get("status") != "success")
    return fails / len(runs)


def select(cands: list) -> tuple:
    """Highest-utility surviving candidate. Returns (winner, all_sorted)."""
    ranked = sorted(cands, key=lambda c: -c.utility)
    live = [c for c in ranked if not c.rejected]
    return (live[0] if live else None), ranked


def render_trace(winner, ranked, objective, state=None, budget_left=None) -> str:
    """The per-iteration decision trace -- makes the agent's reasoning
    auditable after the fact, which is the point of the whole exercise."""
    L = ["## RESEARCH DECISION TRACE",
         f"objective: {objective} | budget left: {budget_left}"]
    if state is not None:
        f = getattr(state, "facts", {}) or {}
        L.append(f"current best: {f.get('best_observed_single_run')} "
                 f"| reseed-verified: {f.get('reseed_verified_mean')} "
                 f"| ensemble: {f.get('expected_ensemble_mean')}")
    L.append(f"candidates generated: {len(ranked)} "
             f"(Path A: {sum(1 for c in ranked if c.path == 'A')}, "
             f"Path B: {sum(1 for c in ranked if c.path == 'B')})")
    for c in ranked:
        mark = "SELECTED" if (winner is not None and c is winner) else \
               ("REJECTED" if c.rejected else "ranked")
        L.append(f"  [{mark}] #{c.index} path={c.path} cat={c.category} "
                 f"utility={c.utility:.5f}")
        L.append(f"      {c.hypothesis[:120]}")
        L.append(f"      parts={c.parts}")
        if c.gates:
            for g in c.gates:
                L.append(f"      rejected: {g}")
    return "\n".join(L)
