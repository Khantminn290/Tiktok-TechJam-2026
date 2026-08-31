"""Two convergence rules, kept separate and never conflated.

The organizers define when a run officially stops and what gets scored:

    ORGANIZER RULE:  epsilon = 0.002, N = 3
    converged when validation primary has not improved by more than epsilon
    over the last 3 consecutive scored iterations, OR the 50-iteration cap,
    OR the 6-hour ceiling -- whichever comes first. The scored submission is
    the validation-best checkpoint at that point.

This project also runs a STRICTER internal controller (epsilon = 0.00048, the
upward drift a running maximum shows by luck at this benchmark's noise floor).
That is a research choice: at 0.002 = 2.5 sigma the loop stops on differences
larger than anything the benchmark still has to offer, so a search that wants
to keep looking needs a tighter bar.

The distinction matters and was previously blurred. Project documentation
described 0.002 as "the earlier hard-coded constant" that a calibrated value
replaced -- which reads as though an official rule were a bug that got fixed.
It is not. It is the rule the competition is scored under, and this module
reports it as such.

A CORRECTION, because this module previously argued the opposite. It said
"stricter is the safe direction: a tighter epsilon can only make the loop run
LONGER than the organizer rule would, never stop it earlier, so no scored
checkpoint is ever missed."

That is true and beside the point. The organizer rule does not only say when to
stop; it says WHAT IS SCORED -- "the scored submission is the validation-best
checkpoint at that point." So running longer does not protect a later artifact,
it produces an INELIGIBLE one. On the recorded journal the official rule fires
at node 3, where the validation-best checkpoint is 0.60497; the 0.60541 ensemble
is node 4, after the official stop.

The risk therefore runs the other way from what this module used to claim. A
stricter internal epsilon is safe for RESEARCH -- it keeps a search alive past
the point the organizer rule would kill it -- and unsafe for SUBMISSION, because
everything it discovers after the official stopping point is evidence, not a
scored checkpoint. Both numbers are reported below so the two are never
confused, and `official.converged_at_node` is the one that decides eligibility.

Usage:
    python3 -m agent.convergence_report
    python3 -m agent.convergence_report --journal <path.jsonl> --json out.json
"""
from __future__ import annotations

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

# Published by the organizers. Do not "improve" these.
ORGANIZER_EPSILON = 0.002
ORGANIZER_N = 3
ORGANIZER_ITERATION_CAP = 50
ORGANIZER_WALL_CLOCK_H = 6.0


def _load(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def _running_best(nodes: list) -> list:
    """(iteration_id, running-best primary) for each SCORED node, in order.

    Errored iterations have no validation score, so they cannot advance the
    window or trigger it -- the organizer rule is written over scored
    iterations.
    """
    out, cur = [], -1.0
    for n in sorted(nodes, key=lambda x: x.get("iteration_id", 0)):
        if n.get("status") == "success" and n.get("metrics"):
            cur = max(cur, n["metrics"]["primary"])
            out.append((n.get("iteration_id"), cur))
    return out


def organizer_convergence(nodes: list, epsilon: float = ORGANIZER_EPSILON,
                          n_window: int = ORGANIZER_N) -> dict:
    """Apply the published rule exactly, and say where it first became true."""
    bests = _running_best(nodes)
    result = {"rule": f"epsilon={epsilon}, N={n_window}",
              "source": "competition brief",
              "scored_iterations": len(bests),
              "converged": False, "converged_at_node": None,
              "gain_over_window": None,
              "best_primary": bests[-1][1] if bests else None}
    if len(bests) < n_window + 1:
        result["note"] = (f"only {len(bests)} scored iterations; the rule needs "
                          f"{n_window + 1} before it can be evaluated")
        return result
    for i in range(n_window, len(bests)):
        gain = bests[i][1] - bests[i - n_window][1]
        if gain <= epsilon:
            result.update(converged=True, converged_at_node=bests[i][0],
                          gain_over_window=round(gain, 6))
            break
    if not result["converged"]:
        result["gain_over_window"] = round(bests[-1][1] - bests[-1 - n_window][1], 6)
        result["note"] = "still improving faster than the rule's threshold"
    return result


def research_controller(nodes: list) -> dict:
    """The stricter internal rule, reported as internal -- never as official."""
    from .loop import EPSILON, N_CONVERGE
    from .validity import NOISE
    r = organizer_convergence(nodes, epsilon=EPSILON, n_window=N_CONVERGE)
    r["rule"] = f"epsilon={EPSILON:.5f}, N={N_CONVERGE}"
    r["source"] = "internal research controller (NOT the organizer rule)"
    r["epsilon_sigma"] = round(EPSILON / NOISE, 2)
    r["rationale"] = (
        "calibrated to the upward drift a running maximum shows by luck over N "
        "iterations. Stricter than the organizer rule, so it extends the "
        "SEARCH -- but anything it finds after the official stopping point is "
        "research evidence, not an eligible scored checkpoint.")
    return r


def report(nodes: list) -> dict:
    org = organizer_convergence(nodes)
    res = research_controller(nodes)
    return {
        "official": org,
        "internal": res,
        "caps": {"iterations": ORGANIZER_ITERATION_CAP,
                 "wall_clock_hours": ORGANIZER_WALL_CLOCK_H},
        "compliance_note": (
            "The organizer rule is the official definition of convergence AND "
            "of which checkpoint is scored: the validation-best checkpoint at "
            "the point it fires. The internal controller is stricter, so the "
            "loop keeps searching past that point -- which is useful for "
            "research and does NOT make a later artifact eligible. Check "
            "official.converged_at_node before treating any result as the "
            "submission."),
        "eligible_checkpoint": _eligible(nodes, org),
    }


def _eligible(nodes: list, org: dict) -> dict:
    """Which checkpoint the official rule would actually score.

    The one question the two-rule report has to answer and previously did not:
    if the organizer rule fired at node X, the submission is the best scored
    node at or before X -- not the best node overall.
    """
    if not org.get("converged"):
        bests = _running_best(nodes)
        return {"determined": False,
                "reason": "the official rule has not fired on this journal",
                "best_so_far": bests[-1][1] if bests else None}
    stop = org["converged_at_node"]
    scored = [(n.get("iteration_id"), n["metrics"]["primary"]) for n in nodes
              if n.get("status") == "success" and n.get("metrics")
              and n.get("iteration_id") is not None
              and n["iteration_id"] <= stop]
    if not scored:
        return {"determined": False, "reason": "no scored node at or before "
                                               f"the stop at node {stop}"}
    node, primary = max(scored, key=lambda x: x[1])
    later = [(n.get("iteration_id"), n["metrics"]["primary"], n.get("action"))
             for n in nodes
             if n.get("status") == "success" and n.get("metrics")
             and (n.get("iteration_id") or -1) > stop
             and n["metrics"]["primary"] > primary]
    return {"determined": True, "converged_at_node": stop,
            "eligible_node": node, "eligible_primary": round(primary, 5),
            "better_but_ineligible": [
                {"node": i, "primary": round(p, 5), "action": a}
                for i, p, a in later],
            "note": ("anything in better_but_ineligible was produced AFTER the "
                     "official stopping point and cannot be the scored "
                     "submission for this journal")}


def render(r: dict) -> str:
    o, i = r["official"], r["internal"]
    L = ["## Convergence", "",
         "**Official (organizer) rule** — this is the one the competition is "
         "scored under.", "",
         f"- rule: `{o['rule']}` ({o['source']})",
         f"- scored iterations: {o['scored_iterations']}",
         f"- converged: **{'YES' if o['converged'] else 'no'}**"
         + (f", first at node {o['converged_at_node']} "
            f"(gain {o['gain_over_window']:+.6f} over the window)"
            if o["converged"] else ""),
         f"- best validation primary: {o['best_primary']}",
         f"- hard caps: {r['caps']['iterations']} iterations, "
         f"{r['caps']['wall_clock_hours']}h wall-clock", ""]
    if o.get("note"):
        L += [f"  > {o['note']}", ""]
    L += ["**Internal research controller** — stricter, and *not* the official "
          "rule.", "",
          f"- rule: `{i['rule']}` = {i.get('epsilon_sigma')}σ",
          f"- converged: {'YES' if i['converged'] else 'no'}"
          + (f", first at node {i['converged_at_node']}"
             if i["converged"] else ""),
          f"  > {i['rationale']}", "",
          f"> {r['compliance_note']}"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(ROOT, "logs", "journal.jsonl"))
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    r = report(_load(a.journal))
    print(render(r))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(r, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
