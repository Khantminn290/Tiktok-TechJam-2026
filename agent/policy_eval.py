"""Offline replay: what would a DIFFERENT policy have done with the same
candidates, and how much of that can we honestly claim to know?

Every decision the multi-candidate planner makes is journalled in full -- the
whole candidate set, each candidate's utility parts, the gates that fired, and
which one was selected. That makes counterfactual replay possible without
spending a single training run: rescore the recorded candidates under an
alternative policy and see where it diverges.

The hard part is not the replay, it is refusing to overclaim from it. A
candidate that was never implemented has no score, and no amount of arithmetic
will produce one. So every replayed decision is labelled by what is actually
knowable:

    OBSERVED              the alternative picks what was really run --
                          the outcome is measured, not inferred
    COUNTERFACTUAL_KNOWN  it picks something different that happens to have
                          been run at another node (matched on the exact
                          configuration signature) -- a real measurement,
                          borrowed, and flagged as borrowed
    COUNTERFACTUAL_UNKNOWN it picks something never implemented -- the
                          outcome is UNKNOWN and is never scored, never
                          imputed, and never averaged into a headline

Consequently this module reports NO "policy X would have scored Y" number
unless coverage is complete, which it essentially never is. What it does
report is decision quality that needs no outcome at all:

  * agreement rate with the policy that actually ran
  * how often a policy would have selected a GATED candidate -- a duplicate,
    a recorded dead end, or an unfalsifiable proposal. Spending an iteration
    on one of those is a mistake that is visible without running anything.
  * how much of the space each policy would have opened (Path B share,
    distinct branches, category mix)

Usage:
    python3 -m agent.policy_eval                       # live logs/
    python3 -m agent.policy_eval --journal <path.jsonl>
    python3 -m agent.policy_eval --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OBSERVED = "OBSERVED"
CF_KNOWN = "COUNTERFACTUAL_KNOWN"
CF_UNKNOWN = "COUNTERFACTUAL_UNKNOWN"


def _sig(choices) -> str:
    return json.dumps(choices or {}, sort_keys=True)


def load_decisions(journal_path: str) -> tuple:
    """Recorded decision points, plus every configuration whose score is known.

    The outcome index is what makes any counterfactual answerable at all: if
    an alternative policy picks a configuration that some OTHER node happened
    to run, that node's measured score is a real observation of it.
    """
    decisions, outcomes = [], {}
    if not os.path.exists(journal_path):
        return decisions, outcomes
    with open(journal_path) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            try:
                n = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if n.get("metrics") and n.get("status") == "success":
                outcomes[_sig(n.get("menu_choices"))] = {
                    "primary": n["metrics"]["primary"],
                    "node": n.get("iteration_id")}
            for e in (n.get("events") or []):
                if e.get("type") == "candidate_selection" and e.get("all"):
                    decisions.append({"node": n.get("iteration_id"),
                                      "candidates": e["all"],
                                      "actual": e.get("selected")})
    return decisions, outcomes


# --------------------------------------------------------------- policies ---
# Each takes the recorded candidate dicts and returns the chosen one (or None).
# They read ONLY fields that were recorded at decision time -- nothing that
# became known afterwards -- or the replay would be leaking hindsight.

def policy_deployed(cands: list):
    """What actually shipped: highest utility among ungated candidates."""
    ok = [c for c in cands if not c.get("rejected_by")]
    return max(ok, key=lambda c: c["utility"]) if ok else None


def policy_first(cands: list):
    """Baseline: take the model's FIRST proposal, gates ignored -- the
    single-proposal behaviour arm A actually had, and the thing multi-candidate
    scoring has to beat to be worth its tokens.

    Must sort by `index`, not take cands[0]: the journal stores the list already
    SORTED BY UTILITY (candidates.select returns `ranked`), so reading position
    0 would return the winner by construction and score a meaningless 100%
    agreement. `index` is the only field preserving the order the model emitted.
    """
    return min(cands, key=lambda c: c.get("index", 0)) if cands else None


def policy_greedy_gain(cands: list):
    """Ablation: chase the largest claimed gain, ignoring cost, novelty and
    probability of success. Isolates what the full utility adds."""
    ok = [c for c in cands if not c.get("rejected_by")]
    return max(ok, key=lambda c: (c.get("parts") or {}).get("gain", 0.0)) if ok else None


def policy_cheapest(cands: list):
    """Ablation: always the cheapest ungated option -- maximum iterations, no
    regard for value. Path B essentially never wins under this."""
    ok = [c for c in cands if not c.get("rejected_by")]
    return min(ok, key=lambda c: (c.get("parts") or {}).get("cost", 1.0)) if ok else None


def policy_no_gates(cands: list):
    """Counterfactual for the gates themselves: pure utility, gates disabled.
    Measures how often the gates are the only thing preventing a wasted
    iteration on a duplicate or a known dead end."""
    return max(cands, key=lambda c: c["utility"]) if cands else None


POLICIES = {"deployed": policy_deployed, "first_proposal": policy_first,
            "greedy_gain": policy_greedy_gain, "cheapest": policy_cheapest,
            "no_gates": policy_no_gates}


def replay(decisions: list, outcomes: dict, policy) -> dict:
    r = {"decisions": 0, "agreed": 0, "picked_gated": 0, "no_pick": 0,
         "path_b": 0, "observed": 0, "cf_known": 0, "cf_unknown": 0,
         "branches": set(), "categories": {}, "detail": []}
    for d in decisions:
        r["decisions"] += 1
        pick = policy(d["candidates"])
        if pick is None:
            r["no_pick"] += 1
            continue
        actual = d.get("actual") or {}
        same = actual.get("index") == pick.get("index")
        r["agreed"] += int(same)
        if pick.get("rejected_by"):
            r["picked_gated"] += 1
        if pick.get("path") == "B":
            r["path_b"] += 1
        cat = pick.get("category", "?")
        r["categories"][cat] = r["categories"].get(cat, 0) + 1
        r["branches"].add(_sig(pick.get("menu_choices")))

        if same:
            status, known = OBSERVED, outcomes.get(_sig(actual.get("menu_choices")))
        else:
            known = outcomes.get(_sig(pick.get("menu_choices")))
            status = CF_KNOWN if known else CF_UNKNOWN
        r[{OBSERVED: "observed", CF_KNOWN: "cf_known",
           CF_UNKNOWN: "cf_unknown"}[status]] += 1
        r["detail"].append({"node": d["node"], "status": status,
                            "picked_index": pick.get("index"),
                            "path": pick.get("path"),
                            "gated": bool(pick.get("rejected_by")),
                            "primary": (known or {}).get("primary")})
    r["branches"] = len(r["branches"])
    n = max(1, r["decisions"])
    r["agreement_rate"] = round(r["agreed"] / n, 3)
    r["gated_pick_rate"] = round(r["picked_gated"] / n, 3)
    r["path_b_rate"] = round(r["path_b"] / n, 3)
    r["outcome_coverage"] = round((r["observed"] + r["cf_known"]) / n, 3)
    return r


def render(res: dict, decisions: list) -> str:
    L = ["=" * 78, "POLICY REPLAY — counterfactual decision analysis", "=" * 78,
         f"recorded decision points: {len(decisions)}"]
    if not decisions:
        L += ["", "No candidate_selection events found. Multi-candidate planning "
                  "records these only when run with --n-candidates >= 2;",
              "a single-proposal run has no alternatives to replay."]
        return "\n".join(L)
    L += ["", f"{'policy':<16}{'agree':>7}{'gated':>7}{'pathB':>7}"
              f"{'branch':>8}{'cover':>7}   outcome knowability",
          "-" * 78]
    for name, r in res.items():
        L.append(f"{name:<16}{r['agreement_rate']:>7.0%}{r['gated_pick_rate']:>7.0%}"
                 f"{r['path_b_rate']:>7.0%}{r['branches']:>8}"
                 f"{r['outcome_coverage']:>7.0%}   "
                 f"obs {r['observed']} / known-cf {r['cf_known']} / "
                 f"UNKNOWN {r['cf_unknown']}")
    L += ["", "Reading this table:",
          "  agree  = same choice as the policy that actually ran",
          "  gated  = picked a duplicate / dead end / unfalsifiable proposal.",
          "           This is a mistake visible WITHOUT running anything, which",
          "           is why it is the most trustworthy column here.",
          "  cover  = share of decisions whose outcome is actually knowable."]
    worst = max(res.items(), key=lambda kv: kv[1]["gated_pick_rate"])
    if worst[1]["gated_pick_rate"] > 0:
        L.append(f"\n'{worst[0]}' would have spent "
                 f"{worst[1]['picked_gated']}/{worst[1]['decisions']} iterations on "
                 f"gated candidates; 'deployed' spent "
                 f"{res['deployed']['picked_gated']}.")

    # The comparison the whole multi-candidate design has to justify: is
    # scoring K proposals better than just taking the model's first one?
    fp, dep = res.get("first_proposal"), res.get("deployed")
    if fp and dep and fp["decisions"]:
        L.append(
            f"\nWHY SCORE CANDIDATES AT ALL — 'first_proposal' is what a "
            f"single-proposal\nplanner does. Replayed over the same "
            f"{fp['decisions']} decisions it would have:\n"
            f"  * picked a GATED candidate {fp['picked_gated']}/{fp['decisions']} "
            f"times ({fp['gated_pick_rate']:.0%}) -- a duplicate, a recorded dead\n"
            f"    end, or an unfalsifiable proposal, each costing a full iteration\n"
            f"  * agreed with the deployed choice only {fp['agreement_rate']:.0%} "
            f"of the time\n"
            f"  * opened {fp['branches']} branches vs {dep['branches']}, and chosen "
            f"Path B {fp['path_b_rate']:.0%} of the time vs {dep['path_b_rate']:.0%}\n"
            f"None of that depends on an outcome, so none of it depends on luck.")
    unknown = res["deployed"]["cf_unknown"]
    L.append(f"\nHONESTY BOUND: {unknown} replayed decision(s) selected work that was "
             f"never implemented.\nTheir outcomes are UNKNOWN and are not scored, "
             f"imputed, or averaged. No\n'policy X would have scored Y' claim is "
             f"made from this table.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(ROOT, "logs", "journal.jsonl"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    decisions, outcomes = load_decisions(a.journal)
    res = {name: replay(decisions, outcomes, fn) for name, fn in POLICIES.items()}
    print(render(res, decisions))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"n_decisions": len(decisions),
                       "known_outcomes": len(outcomes), "policies": res},
                      fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
