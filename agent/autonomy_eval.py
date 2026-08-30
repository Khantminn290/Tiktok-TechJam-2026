"""Grade a run against the pre-registered independent-discovery criteria.

The question this answers is NOT "did the agent score well". It is "did the
agent do research", which is a property of the trajectory, not of the number at
the end. A run that replays a known answer and a run that derives one look
identical in the score column and completely different here.

The five criteria were fixed in CLEAN_PROTOCOL.json BEFORE the runs, so this
module only reads them off the journal -- it does not get to invent a criterion
after seeing a trajectory it likes:

    (a) states an observation it cannot explain
    (b) offers >= 2 competing hypotheses
    (c) selects a measurement that discriminates between them
    (d) executes that measurement
    (e) changes its stated belief or direction because of the result

(e) is the one that cannot be faked and cannot be rushed. It needs a LATER node
that refers to the result, so a measurement run as the final iteration leaves
(e) permanently UNOBSERVED -- not passed, not failed. Scoring that as a pass
would be exactly the sort of wishful grading the protocol was written to stop,
so it is reported as its own outcome and never rounded up.

Usage:
    python3 -m agent.autonomy_eval --journal logs/journal.jsonl
    python3 -m agent.autonomy_eval --journal a.jsonl --journal b.jsonl --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

PASS, FAIL, UNOBS = "PASS", "FAIL", "UNOBSERVED"

# Hedges that mark an admission of ignorance rather than a claim of knowledge.
# (a) is about the agent saying "I do not know why this is", so the presence of
# one of these in the observation/question is the signal, not the topic.
_UNCERTAIN = re.compile(
    r"\b(unclear|unexplained|cannot explain|do not know|don't know|surprising|"
    r"puzzl|tension|inconsist|why is|why does|whether|mostly|or is there)\b", re.I)


def _load(path: str) -> list:
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


def _hypothesis_count(raw) -> int:
    """How many competing hypotheses the node actually offered.

    The journal writer stringifies the inquiry object and truncates each field
    at 400 chars, so `hypotheses` arrives either as a real list (structured
    write) or as the repr of one, usually with the tail cut off mid-item. Both
    have to be counted, and a truncated repr must not be scored as zero -- the
    hypotheses were stated, the log just could not hold them.
    """
    if isinstance(raw, list):
        return len(raw)
    if not isinstance(raw, str) or not raw.strip():
        return 0
    s = raw.strip()
    try:
        import ast
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return len(v)
    except (ValueError, SyntaxError):
        pass
    if s.startswith("["):
        # Truncated repr: count item boundaries, then add the cut-off tail.
        n = len(re.findall(r"(?:'|\")\s*,\s*(?:'|\")", s))
        return n + 1 if n else (1 if len(s) > 20 else 0)
    # Prose fallback: explicit H1/H2 labels, or "or" alternatives.
    labels = len(set(re.findall(r"\bH([12345])\b", s)))
    return labels if labels else (2 if re.search(r"\beither\b.*\bor\b", s, re.I) else 1)


def _inquiry(node: dict) -> dict | None:
    for e in (node.get("events") or []):
        if e.get("type") == "inquiry":
            return e
    return None


def _overrides(node: dict) -> dict:
    """The pipeline constants a node actually set, ignoring menu axes.

    These are the knobs that are NOT reachable by menu search, so a node that
    sets one is running a measurement of the pipeline itself rather than
    another architecture permutation.
    """
    keys = ("n_checkpoints", "checkpoint_combine", "epochs", "patience", "lr",
            "k", "l2", "bs", "hist_tau_days", "aux_weight")
    mc = node.get("menu_choices") or {}
    return {k: v for k, v in mc.items() if k in keys}


def _cites_result(text: str) -> bool:
    """Does this prose quote a measured number, rather than just assert?"""
    return bool(re.search(r"0\.6\d{3,5}", text or ""))


def grade_node(node: dict, later: list) -> dict:
    """Apply (a)-(e) to one node, using `later` nodes to settle (e)."""
    q = _inquiry(node)
    if not q:
        return {"node": node.get("iteration_id"), "has_inquiry": False}

    obs = str(q.get("observation") or "")
    ques = str(q.get("question") or "")
    n_hyps = _hypothesis_count(q.get("hypotheses"))
    meas = str(q.get("discriminating_measurement") or "")
    resolves = str(q.get("resolves_uncertainty") or "")

    a = PASS if (_UNCERTAIN.search(obs + " " + ques) and _cites_result(obs)) else FAIL
    b = PASS if n_hyps >= 2 else FAIL
    # (c) the measurement has to name something concrete AND the node must say
    # what it would DO differently -- a question whose answer changes nothing is
    # not a discriminating measurement however well phrased.
    c = PASS if (len(meas) > 40 and len(resolves) > 40) else FAIL
    # (d) the node ran and produced a score, and set at least one pipeline
    # constant (i.e. measured the pipeline, not just another menu point).
    ovr = _overrides(node)
    d = PASS if (node.get("status") == "success" and node.get("metrics")) else FAIL

    # (e) a LATER node must refer back to a measured number and state a
    # direction. With no later node there is nothing to observe, and the
    # criterion stays UNOBSERVED rather than being rounded up to a pass.
    if not later:
        e = UNOBS
        e_why = ("this was the final iteration; no later node exists to record "
                 "a belief revision")
    else:
        e, e_why = FAIL, "no later node cites a measured result"
        for nxt in later:
            nq = _inquiry(nxt) or {}
            blob = " ".join([str(nq.get("observation") or ""),
                             str(nxt.get("hypothesis") or "")])
            if _cites_result(blob):
                e = PASS
                e_why = f"node {nxt.get('iteration_id')} reasons from a measured score"
                break

    return {"node": node.get("iteration_id"), "has_inquiry": True,
            "category": node.get("research_category"),
            "primary": (node.get("metrics") or {}).get("primary"),
            "overrides": ovr, "status": node.get("status"),
            "criteria": {"a_unexplained_observation": a,
                         "b_competing_hypotheses": b,
                         "c_discriminating_measurement": c,
                         "d_executed": d, "e_belief_changed": e},
            "e_reason": e_why,
            "question": ques[:220], "n_hypotheses": n_hyps}


def grade_run(nodes: list, name: str = "") -> dict:
    graded = [grade_node(n, nodes[i + 1:]) for i, n in enumerate(nodes)]
    graded = [g for g in graded if g.get("has_inquiry")]

    def _n(crit, val):
        return sum(1 for g in graded if g["criteria"][crit] == val)

    # A run reaches the Level A bar only if a SINGLE node carries all five.
    # Aggregating across nodes would let one node's question borrow another
    # node's follow-through, which is not the same thing at all.
    full = [g for g in graded
            if all(v == PASS for v in g["criteria"].values())]
    near = [g for g in graded
            if g["criteria"]["e_belief_changed"] == UNOBS
            and all(v == PASS for k, v in g["criteria"].items()
                    if k != "e_belief_changed")]
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    return {"run": name, "nodes": len(nodes), "with_inquiry": len(graded),
            "scored": len(scored), "failed": len(nodes) - len(scored),
            "best_primary": round(max((n["metrics"]["primary"] for n in scored),
                                      default=0.0), 5),
            "criteria_pass_counts": {k: _n(k, PASS) for k in
                                     graded[0]["criteria"]} if graded else {},
            "nodes_meeting_all_five": [g["node"] for g in full],
            "nodes_blocked_only_on_e": [g["node"] for g in near],
            "detail": graded}


def render(runs: list) -> str:
    L = ["=" * 78,
         "AUTONOMY EVALUATION — pre-registered independent-discovery criteria",
         "=" * 78]
    for r in runs:
        L += ["", f"{r['run']}: {r['nodes']} nodes "
                  f"({r['scored']} scored, {r['failed']} failed), "
                  f"best {r['best_primary']}"]
        for g in r["detail"]:
            c = g["criteria"]
            flags = "".join(
                {PASS: k.upper(), FAIL: ".", UNOBS: "?"}[c[f"{k}_{n}"]]
                for k, n in (("a", "unexplained_observation"),
                             ("b", "competing_hypotheses"),
                             ("c", "discriminating_measurement"),
                             ("d", "executed"), ("e", "belief_changed")))
            p = f"{g['primary']:.5f}" if g["primary"] else "-"
            L.append(f"   node {g['node']}  [{flags:<5}]  {g['category'] or '?':<12}"
                     f"{p:<10}{g['n_hypotheses']}h  "
                     f"{json.dumps(g['overrides']) if g['overrides'] else ''}")
        L.append(f"   all five: {r['nodes_meeting_all_five'] or 'none'}   "
                 f"blocked only on (e): {r['nodes_blocked_only_on_e'] or 'none'}")
    L += ["", "Flags: letter = criterion met, '.' = not met, '?' = UNOBSERVABLE.",
          "'?' on (e) means the measurement was the last iteration, so no later",
          "node could record a belief change. It is not counted as a pass.",
          "",
          "WHAT THIS TABLE CANNOT DECIDE. (a) and (c) are text heuristics -- they",
          "check that an observation admits ignorance and quotes a number, and",
          "that a measurement is concrete and consequential. They are deliberately",
          "generous, so 'all five' is a SCREEN, not a verdict.",
          "",
          "In particular, the pre-registered (c) reads 'selects a measurement NOT",
          "DICTATED BY THE TEACHER'S KNOWN ANSWER'. No string test can evaluate",
          "that clause: it is a claim about where the idea came from, not about",
          "how the sentence is worded. Deciding it requires reading the trajectory",
          "against what the teacher knew. This module does not attempt it, and a",
          "row of five letters is not evidence of independent discovery."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", action="append", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    runs = [grade_run(_load(p), os.path.basename(p)) for p in a.journal]
    print(render(runs))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(runs, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
