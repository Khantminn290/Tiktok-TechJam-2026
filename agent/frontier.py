"""The research frontier: what is known, what is not, and what is worth trying.

Replaces the previous notion of a "branch", which was the pair (loss, model).
That grouping could not represent the questions this project actually faces --
"is temporal information useful?", "has multitask saturated?" -- because those
live on other axes entirely, and it silently treated every other axis as noise.

A DIRECTION here is one axis-option: a concrete, testable mechanism such as
`user_history=recency_weighted_pool`. For each one the frontier derives, from
the journal alone:

    experiments, failures, best/mean primary, recent trend, variance,
    isolated ablation evidence, confirmation count, GAUC vs nDCG@5 split,
    confidence, status

Two distinctions the old code collapsed, both of which change what to do next:

  * UNEXPLORED is not KNOWN_BAD. An option never tried has no evidence against
    it; treating silence as refutation is how a search convinces itself the
    space is exhausted. Menu options with zero experiments are listed
    explicitly, as candidates rather than as failures.
  * GAUC and nDCG@5 are reported SEPARATELY. The primary is their mean, so a
    change that adds +0.004 GAUC while costing -0.004 nDCG@5 is invisible in
    the primary and looks like a null result -- when in fact it is a strong
    mechanism pointed at only half the objective. Nothing in this codebase
    made that distinction before; every module read `primary` alone.

Everything is deterministic and derived on construction. No LLM, no state.
"""
from __future__ import annotations

import json
import os
import statistics

from agent.research_state import (BASELINE_SEED_STD, grade_evidence,
                                  INCONCLUSIVE, REJECTED, STRONG, MODERATE)

# Status vocabulary -- evidence-based, never asserted.
KNOWN_GOOD = "KNOWN_GOOD"        # isolated evidence for, beyond the noise floor
KNOWN_BAD = "KNOWN_BAD"          # isolated evidence against, or a recorded dead end
PROMISING = "PROMISING"          # positive but not yet beyond the noise floor
UNCERTAIN = "UNCERTAIN"          # tried, but the evidence does not separate
UNEXPLORED = "UNEXPLORED"        # never tried -- absence of evidence, not evidence
SATURATED = "SATURATED"          # repeatedly tried, recent returns collapsed
CONTRADICTORY = "CONTRADICTORY"  # experiments disagree beyond seed noise

HIGH, MEDIUM, LOW = "HIGH", "MEDIUM", "LOW"
SATURATION_MIN_EXPERIMENTS = 4


def _sig(c) -> str:
    return json.dumps(c or {}, sort_keys=True)


class Frontier:
    def __init__(self, nodes: list, menu: dict, best_config: dict | None = None):
        self.nodes = nodes or []
        self.scored = [n for n in self.nodes
                       if n.get("status") == "success" and n.get("metrics")]
        self.menu = menu or {}
        self.axes = (self.menu.get("axes") or {})
        self.dead_ends = ((self.menu.get("notes") or {}).get("tested_dead_ends") or [])
        self.best_config = best_config or self._best_config()
        self.directions = self._build()

    def _best_config(self) -> dict:
        if not self.scored:
            return {}
        return dict(max(self.scored,
                        key=lambda n: n["metrics"]["primary"]).get("menu_choices") or {})

    # ------------------------------------------------------------------
    def _options(self, axis: str) -> list:
        spec = self.axes.get(axis)
        if isinstance(spec, dict):
            opts = spec.get("options")
        else:
            opts = spec
        if isinstance(opts, dict):
            return list(opts.keys())
        return list(opts or [])

    def _isolated(self, axis: str, value: str) -> dict | None:
        """Compare configs differing ONLY on `axis`, holding everything else at
        the best config. This is the only comparison that attributes an effect
        to a mechanism rather than to a bundle of simultaneous changes."""
        best = self.best_config
        if not best or axis not in best:
            return None
        ref = [n for n in self.scored if _sig(n.get("menu_choices")) == _sig(best)]
        if not ref:
            return None
        ref_m = ref[0]["metrics"]
        # Two cases, and conflating them found no evidence at all for the very
        # options the best config uses:
        #   value IS best[axis]  -> the comparison is against ANY other value on
        #                           this axis (what does removing it cost?)
        #   value is NOT in best -> the comparison is against that value itself
        #                           (what would swapping to it cost?)
        in_best = best.get(axis) == value
        alt = []
        for n in self.scored:
            c = n.get("menu_choices") or {}
            if not c or set(c) != set(best):
                continue
            diff = [a for a in best if c.get(a) != best.get(a)]
            if diff != [axis]:
                continue
            if in_best or c.get(axis) == value:
                alt.append(n)
        if not alt:
            return None
        a = max(alt, key=lambda n: n["metrics"]["primary"])
        am = a["metrics"]
        return {"vs_node": a["iteration_id"],
                "d_primary": ref_m["primary"] - am["primary"],
                "d_gauc": ref_m.get("GAUC", 0) - am.get("GAUC", 0),
                "d_ndcg": ref_m.get("nDCG@5", 0) - am.get("nDCG@5", 0),
                "n_alt": len(alt)}

    # A dead end names its SUBJECT first ("LambdaRank ...: MEASURED HERE,
    # decisively worse than bpr_pairwise"). Matching anywhere in the text made
    # every dead end that cites the winning config as its COMPARISON BASELINE
    # mark that config's own components KNOWN_BAD -- loss=bpr_pairwise and
    # model=fm_numpy, the two components carrying the best result, were both
    # falsely condemned. Only the subject line can implicate an option.
    SUBJECT_CHARS = 70
    # Words that introduce a COMPARISON BASELINE rather than the subject.
    # "gru4rec_seq as an ensemble member WITH fm_numpy" is a finding about
    # gru4rec; fm_numpy is the thing it was measured against.
    _BASELINE_MARKERS = (" with ", " than ", " vs ", " versus ", " compared ",
                         " against ", " beats ", " over ")

    def _dead_end_hit(self, axis: str, value: str) -> str | None:
        if value == "none":                      # "none" is a default, not a mechanism
            return None
        # The incumbent is never a dead end. Lexical matching cannot reliably
        # tell a finding's SUBJECT from the baseline it was measured against --
        # "none beat the uniform_1 default" and "decisively worse than
        # bpr_pairwise" both name a working option inside a negative finding.
        # Rather than keep patching the patterns, assert the invariant: an
        # option the current best configuration actually uses demonstrably
        # works, whatever some other experiment's write-up mentions.
        if self.best_config.get(axis) == value:
            return None
        import re
        # Whole-token match: plain substring made multitask=aux_click_like_forward
        # match the dead end for multitask=aux_click_like_forward_WATCH, wrongly
        # condemning a different mechanism by shared prefix.
        pat = re.compile(rf"\b{re.escape(axis)}={re.escape(value)}(?![\w])")
        for d in self.dead_ends:
            if pat.search(d):
                return d[:120]
            # A finding scoped to a whole AXIS covers its option values even
            # when they are named far into the text: "Negative-sampling variants
            # (neg_sampling axis): ... uniform_2 0.60356 and uniform_4 0.60316"
            # left both showing UNEXPLORED, inviting the agent to re-run an
            # experiment already measured over 5 paired seeds.
            if re.search(rf"\b{re.escape(axis)}\b", d) and \
                    re.search(rf"\b{re.escape(value)}(?![\w])", d):
                return d[:120]
            subject = d[:self.SUBJECT_CHARS].lower()
            for mark in self._BASELINE_MARKERS:
                if mark in subject:
                    subject = subject.split(mark)[0]
                    break
            if re.search(rf"\b{re.escape(value.lower())}(?![\w])", subject):
                return d[:120]
        return None

    def _build(self) -> list:
        out = []
        for axis in sorted(self.axes):
            for value in self._options(axis):
                runs = [n for n in self.scored
                        if (n.get("menu_choices") or {}).get(axis) == value]
                fails = [n for n in self.nodes
                         if (n.get("menu_choices") or {}).get(axis) == value
                         and n.get("status") != "success"]
                d = {"direction": f"{axis}={value}", "axis": axis, "value": value,
                     "experiments": len(runs), "failures": len(fails),
                     "failure_rate": round(len(fails) / max(1, len(runs) + len(fails)), 2),
                     "in_best_config": self.best_config.get(axis) == value}
                prim = [n["metrics"]["primary"] for n in runs]
                if prim:
                    d["best_primary"] = round(max(prim), 5)
                    d["mean_primary"] = round(statistics.mean(prim), 5)
                    d["variance_sigma"] = round(
                        (statistics.pstdev(prim) if len(prim) > 1 else 0.0)
                        / BASELINE_SEED_STD, 2)
                    d["best_gauc"] = round(max(n["metrics"].get("GAUC", 0) for n in runs), 5)
                    d["best_ndcg"] = round(max(n["metrics"].get("nDCG@5", 0) for n in runs), 5)
                    # A trend needs BOTH a recent window and an earlier one to
                    # compare it against. With exactly SATURATION_MIN_EXPERIMENTS
                    # runs there is no earlier window, and defaulting to 0.0 made
                    # four CONSECUTIVELY IMPROVING experiments read as saturated
                    # -- telling the planner to abandon a direction that was
                    # actively working. None means "not computable", not "flat".
                    recent = prim[-SATURATION_MIN_EXPERIMENTS:]
                    earlier = prim[:-SATURATION_MIN_EXPERIMENTS]
                    d["recent_trend"] = (round(max(recent) - max(earlier), 5)
                                         if earlier else None)

                iso = self._isolated(axis, value)
                if iso:
                    # d_primary is (best - alternative), so a POSITIVE value means
                    # the best config's own choice beats this one. Flip it so the
                    # grade always reads "evidence FOR this direction".
                    g = grade_evidence(-iso["d_primary"] if not d["in_best_config"]
                                       else iso["d_primary"])
                    d["ablation"] = {
                        "vs_node": iso["vs_node"], "n": iso["n_alt"],
                        "d_primary": round(iso["d_primary"], 5),
                        "d_GAUC": round(iso["d_gauc"], 5),
                        "d_nDCG@5": round(iso["d_ndcg"], 5),
                        "strength": g["strength"], "sigma": g["sigma"],
                        # the split the primary hides
                        "metric_conflict": bool(iso["d_gauc"] * iso["d_ndcg"] < 0
                                                and min(abs(iso["d_gauc"]),
                                                        abs(iso["d_ndcg"]))
                                                > BASELINE_SEED_STD)}
                d["dead_end"] = self._dead_end_hit(axis, value)
                d["status"], d["confidence"] = self._classify(d, prim)
                out.append(d)
        return sorted(out, key=lambda x: (_STATUS_ORDER.get(x["status"], 9),
                                          -(x.get("best_primary") or 0)))

    def _classify(self, d: dict, prim: list) -> tuple:
        if d["dead_end"]:
            return KNOWN_BAD, HIGH
        if not d["experiments"]:
            # No evidence is NOT evidence against.
            return UNEXPLORED, LOW
        ab = d.get("ablation")
        trend = d.get("recent_trend")
        if (trend is not None and trend <= 0
                and d["experiments"] > SATURATION_MIN_EXPERIMENTS
                and not d["in_best_config"]):
            return SATURATED, MEDIUM
        if ab:
            if ab["strength"] in (STRONG, MODERATE):
                good = ab["d_primary"] >= 0 if d["in_best_config"] else ab["d_primary"] <= 0
                conf = HIGH if ab["strength"] == STRONG else MEDIUM
                return (KNOWN_GOOD if good else KNOWN_BAD), conf
            if ab["strength"] == REJECTED:
                return KNOWN_BAD, MEDIUM
            if ab["strength"] == INCONCLUSIVE:
                return UNCERTAIN, LOW
        if len(prim) > 1 and (max(prim) - min(prim)) > 3 * BASELINE_SEED_STD:
            return CONTRADICTORY, LOW
        if d["in_best_config"]:
            return PROMISING, LOW
        return UNCERTAIN, LOW

    # ------------------------------------------------------------------
    def unexplored(self) -> list:
        return [d for d in self.directions if d["status"] == UNEXPLORED]

    def metric_conflicts(self) -> list:
        """Directions that help one metric and hurt the other. The primary is
        their mean, so these look like null results and get discarded."""
        return [d for d in self.directions
                if (d.get("ablation") or {}).get("metric_conflict")]

    def render(self, limit: int = 26) -> str:
        L = ["## RESEARCH FRONTIER (derived from the journal; no LLM)",
             f"{'direction':<40}{'status':<15}{'conf':<7}{'n':>3} {'best':>8} "
             f"{'GAUC':>8} {'nDCG@5':>8}"]
        for d in self.directions[:limit]:
            L.append(f"{d['direction']:<40}{d['status']:<15}{d['confidence']:<7}"
                     f"{d['experiments']:>3} "
                     f"{(d.get('best_primary') or 0):>8.5f} "
                     f"{(d.get('best_gauc') or 0):>8.5f} "
                     f"{(d.get('best_ndcg') or 0):>8.5f}")
        un = self.unexplored()
        if un:
            L.append(f"\nUNEXPLORED ({len(un)}) -- no evidence AGAINST these, they have "
                     f"simply never been run:")
            L.append("  " + ", ".join(d["direction"] for d in un[:14]))
        mc = self.metric_conflicts()
        if mc:
            L.append("\nMETRIC CONFLICTS (help one metric, hurt the other -- the primary "
                     "hides these):")
            for d in mc:
                a = d["ablation"]
                L.append(f"  {d['direction']}: GAUC {a['d_GAUC']:+.5f} "
                         f"nDCG@5 {a['d_nDCG@5']:+.5f}")
        return "\n".join(L)


_STATUS_ORDER = {KNOWN_GOOD: 0, PROMISING: 1, CONTRADICTORY: 2, UNCERTAIN: 3,
                 UNEXPLORED: 4, SATURATED: 5, KNOWN_BAD: 6}


def from_root(root: str) -> Frontier:
    nodes = []
    jp = os.path.join(root, "logs", "journal.jsonl")
    if os.path.exists(jp):
        with open(jp) as fh:
            for ln in fh:
                if ln.strip():
                    try:
                        nodes.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    menu = {}
    mp = os.path.join(root, "config", "modification_menu.json")
    if os.path.exists(mp):
        with open(mp) as fh:
            menu = json.load(fh)
    return Frontier(nodes, menu)


if __name__ == "__main__":
    _R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(from_root(_R).render())
