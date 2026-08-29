"""Compare two agent runs as RESEARCH PROCESSES, not just as scores.

The question this answers is the one the audit raised: multi-candidate planning
was added so that Path B (writing custom code) becomes a scoreable option
rather than something the planner never generates. Did it?

Score is reported, but it is deliberately NOT the headline. Two runs of an
agent on a benchmark whose seed noise is 0.0008 will differ by more than their
policies do, and a single pair of runs cannot separate those. What a single
pair CAN show is process: whether Path B was ever generated and chosen, how
much of the space was opened, how many iterations went to gated or duplicated
work, and whether decisions were auditable at all.

Usage:
    python3 -m agent.ab_report --a logs/ab_test/arm_A_journal.jsonl \\
                               --b logs/journal.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

BASELINE = 0.6016
SIGMA = 0.0008


def load(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def summarise(nodes: list, label: str, journal_path: str = "") -> dict:
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    prim = [n["metrics"]["primary"] for n in scored]
    s = {"label": label, "iterations": len(nodes), "scored": len(scored),
         "failed": len(nodes) - len(scored),
         "best": round(max(prim), 5) if prim else None,
         "median": round(statistics.median(prim), 5) if prim else None,
         "path_b_nodes": sum(1 for n in nodes
                             if (n.get("implementation_path") or "").upper() == "B"),
         "distinct_configs": len({json.dumps(n.get("menu_choices") or {}, sort_keys=True)
                                  for n in scored}),
         "model_families": ",".join(sorted({(n.get("menu_choices") or {}).get("model")
                                            for n in scored
                                            if (n.get("menu_choices") or {}).get("model")})),
         "categories": {}, "tokens": 0}
    for n in nodes:
        c = (n.get("research_category") or "").lower()
        if c:
            s["categories"][c] = s["categories"].get(c, 0) + 1
        s["tokens"] += int(n.get("tokens_used") or 0)

    # decision-level evidence: only present when multi-candidate planning ran
    dec = cand = pb = gated = 0
    for n in nodes:
        for e in (n.get("events") or []):
            if e.get("type") == "candidate_selection":
                dec += 1
                cand += e.get("n_candidates", 0)
                pb += e.get("n_path_b", 0)
                gated += e.get("n_rejected", 0)
    s["decision_points"] = dec
    s["candidates_generated"] = cand
    s["path_b_candidates_generated"] = pb
    s["candidates_gated"] = gated
    s["auditable_decisions"] = dec > 0

    # A declared Path B is not necessarily a real one. capability_report reads
    # the scripts on disk and separates code that implements its own mechanism
    # from code that merely calls train_lib.run() while claiming Path B. Only
    # the first is evidence that the agent can modify the approach itself, so
    # counting declarations alone would overstate the fix to the audit finding.
    s["path_b_genuine"] = None
    s["path_b_fake"] = None
    if journal_path and os.path.exists(journal_path):
        try:
            from agent.capability_report import analyse
            cap = analyse(journal_path)
            s["path_b_genuine"] = cap.get("path_B_genuine")
            s["path_b_fake"] = cap.get("path_B_fake")
        except Exception:
            pass
    if prim:
        d = max(prim) - BASELINE
        s["best_delta_sigma"] = round(d / SIGMA, 2)
    return s


def render(a: dict, b: dict) -> str:
    def row(name, key, fmt="{}"):
        va, vb = a.get(key), b.get(key)
        fa = fmt.format(va) if va is not None else "-"
        fb = fmt.format(vb) if vb is not None else "-"
        return f"  {name:<32}{fa:>18}{fb:>18}"

    L = ["=" * 74, "A/B — single-proposal planner vs multi-candidate policy", "=" * 74,
         f"  {'':<32}{a['label']:>18}{b['label']:>18}", "-" * 74,
         "PROCESS (what a single run pair CAN show)",
         row("Path B nodes declared", "path_b_nodes"),
         row("  ...GENUINE (own mechanism)", "path_b_genuine"),
         row("  ...fake (delegates to lib)", "path_b_fake"),
         row("Path B candidates generated", "path_b_candidates_generated"),
         row("decision points recorded", "decision_points"),
         row("candidates generated", "candidates_generated"),
         row("candidates gated (waste avoided)", "candidates_gated"),
         row("decisions auditable", "auditable_decisions"),
         row("distinct configs explored", "distinct_configs"),
         row("model families", "model_families",
             "{}" ),
         "", "COST",
         row("iterations", "iterations"),
         row("scored / failed", "scored"),
         row("LLM tokens", "tokens", "{:,}"),
         "", "SCORE (secondary -- see caveat)",
         row("best primary", "best"),
         row("best vs baseline (sigma)", "best_delta_sigma"),
         row("median primary", "median"),
         "-" * 74]

    if not a["auditable_decisions"] and b["auditable_decisions"]:
        L.append("Arm A recorded NO decision points: its planner emitted one proposal,"
                 "\nso there were no alternatives to score and 'why not Path B?' had no"
                 "\nanswer. Arm B records the full candidate set for every decision.")
    if a["path_b_nodes"] == 0 and b["path_b_nodes"] > 0:
        L.append(f"\nPath B went from NEVER generated ({a['path_b_nodes']} nodes) to "
                 f"generated and selected\n({b['path_b_nodes']} declared, "
                 f"{b['path_b_candidates_generated']} candidates offered). The audit "
                 f"finding was that Path B\nwas never GENERATED; that part is fixed.")
        g, f_ = b.get("path_b_genuine"), b.get("path_b_fake")
        if g is not None and f_ is not None and b["path_b_nodes"]:
            if f_ > g:
                L.append(f"\nBUT ONLY {g} OF {g + f_} ARE GENUINE. The other {f_} declare "
                         f"Path B and then call\ntrain_lib.run() -- Path A wearing a "
                         f"Path B label. Selecting Path B is fixed;\nWRITING it is not, "
                         f"and reporting the declaration count alone would hide that.")
            else:
                L.append(f"\nOf these, {g} implement their own mechanism and {f_} merely "
                         f"delegate to\ntrain_lib.run().")

    L.append("\nCAVEAT ON SCORE: seed noise here is 0.0008, and these are two runs, not"
             "\ntwo samples of a policy. A score difference between single runs cannot"
             "\nseparate policy from luck, so no causal claim is made from it. The"
             "\nprocess rows above do not depend on luck.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=os.path.join(ROOT, "logs", "ab_test",
                                                "arm_A_journal.jsonl"))
    ap.add_argument("--b", default=os.path.join(ROOT, "logs", "journal.jsonl"))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    a = summarise(load(args.a), "A single-prop", args.a)
    b = summarise(load(args.b), "B multi-cand", args.b)
    print(render(a, b))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"arm_A": a, "arm_B": b}, fh, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
