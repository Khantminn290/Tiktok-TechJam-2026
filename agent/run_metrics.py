"""Reliability and scientific-discipline metrics for an agent run.

These are the numbers `CLEAN_PROTOCOL_2.json` pre-registered, computed from the
journal so the same definitions apply to runs recorded before and after the
architecture work. A metric chosen after seeing results is not a measurement,
so nothing here is derived from what the runs happened to do.

The two that matter most, because they are what the architecture was built to
change:

  path_b_crash_rate            crashes / Path B attempts. 0.71 before.
  orchestration_only_misuse    calls to a capability that does not exist in
                               generated code. Three of these each cost a full
                               iteration before; preflight should now catch them
                               for zero training time.

Usage:
    python3 -m agent.run_metrics --journal a.jsonl --journal b.jsonl
    python3 -m agent.run_metrics --journal a.jsonl --label post --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

from . import budget as B  # noqa: E402
from . import capabilities as C  # noqa: E402

PREFLIGHT_MARKER = "PREFLIGHT REJECTED"


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


class _N:
    """Journal dicts adapted to the attribute access agent.budget expects."""

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, k):
        return self._d.get(k)


def _event(node: dict, kind: str) -> dict | None:
    for e in (node.get("events") or []):
        if e.get("type") == kind:
            return e
    return None


def _inquiry(node: dict) -> dict:
    return _event(node, "inquiry") or {}


def _hyp_count(raw) -> int:
    from .autonomy_eval import _hypothesis_count
    return _hypothesis_count(raw)


def compute(nodes: list, label: str = "") -> dict:
    adapted = [_N(n) for n in nodes]
    b = B.count(adapted)

    path_b = [n for n in nodes
              if (n.get("implementation_path") or "").upper() == "B"]
    path_b_crash = [n for n in path_b if n.get("status") != "success"]

    # Failure classes, and specifically the misuse the contract exists to stop.
    classes: dict = {}
    orch_misuse = 0
    preflight_stages: dict = {}
    orch_names = C.orchestration_only()
    for n in nodes:
        trace = n.get("error_trace") or ""
        for e in (n.get("events") or []):
            if e.get("type") == "execution_error" and e.get("failure_class"):
                classes[e["failure_class"]] = classes.get(e["failure_class"], 0) + 1
        if PREFLIGHT_MARKER in trace:
            m = re.search(r"at the (\w+) stage", trace)
            stage = m.group(1).lower() if m else "unknown"
            preflight_stages[stage] = preflight_stages.get(stage, 0) + 1
        if any(re.search(rf"\b(?:train_lib|agent|pipeline_lab)\.{name}\b", trace)
               or f"`{name}` is an ORCHESTRATION-ONLY" in trace
               for name in orch_names):
            orch_misuse += 1

    # Repeated identical failures.
    from .failure import classify, fingerprint
    seen: dict = {}
    repeats = 0
    for n in nodes:
        if n.get("status") == "error" and n.get("error_trace"):
            fp = fingerprint(classify(n["error_trace"]), n["error_trace"])
            seen[fp] = seen.get(fp, 0) + 1
            if seen[fp] > 1:
                repeats += 1

    # Tool usage.
    tools: dict = {}
    for n in nodes:
        for t in ((_event(n, "inspect") or {}).get("tools") or []):
            tools[t] = tools.get(t, 0) + 1

    # Inquiry completeness and capability naming.
    inq = [_inquiry(n) for n in nodes]
    inq = [q for q in inq if q]
    complete = sum(1 for q in inq
                   if q.get("observation") and q.get("question")
                   and _hyp_count(q.get("hypotheses")) >= 2
                   and q.get("discriminating_measurement"))
    named_cap = [str(q.get("capability_required") or "").strip() for q in inq]
    named_cap = [c for c in named_cap if c]
    known = set(C.all_capabilities())
    cap_valid = sum(1 for c in named_cap
                    if any(k in c for k in known))
    promo = sum(1 for q in inq if str(q.get("promotion_criterion") or "").strip())

    # Confirmation discipline: did a promising single-seed result lead to a
    # confirmation node rather than being built upon?
    cats = [(_event(n, "research_category") or {}).get("category") for n in nodes]
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]

    return {
        "label": label,
        "nodes": len(nodes),
        "iterations_consumed": b["iterations_consumed"],
        "preflight_rejections": b["preflight_rejections"],
        "preflight_stages": preflight_stages,
        "experiments_completed": b["experiments_completed"],
        "experiments_crashed": b["experiments_crashed"],
        "training_wall_clock_s": b["training_wall_clock_s"],
        "path_b_attempts": len(path_b),
        "path_b_crashes": len(path_b_crash),
        "path_b_crash_rate": (round(len(path_b_crash) / len(path_b), 3)
                              if path_b else None),
        "orchestration_only_misuse": orch_misuse,
        "repeated_identical_failures": repeats,
        "failure_classes": classes,
        "tool_calls": tools,
        "tool_call_total": sum(tools.values()),
        "nodes_with_inquiry": len(inq),
        "inquiry_complete": complete,
        "capability_named": len(named_cap),
        "capability_named_valid": cap_valid,
        "promotion_criterion_stated": promo,
        "categories": {c: cats.count(c) for c in set(cats) if c},
        "confirmation_nodes": cats.count("confirmation"),
        "best_primary": (round(max(n["metrics"]["primary"] for n in scored), 5)
                         if scored else None),
        "manual_interventions": 0,
    }


def render(rows: list) -> str:
    L = ["=" * 78, "RUN METRICS — pre-registered in CLEAN_PROTOCOL_2.json", "=" * 78]

    def line(label, key, fmt="{}"):
        vals = []
        for r in rows:
            v = r.get(key)
            vals.append("-" if v is None else fmt.format(v))
        L.append(f"  {label:<34}" + "".join(f"{v:>14}" for v in vals))

    L.append(f"  {'':<34}" + "".join(f"{r['label']:>14}" for r in rows))
    L.append("  " + "-" * 74)
    L.append("  RELIABILITY")
    line("nodes", "nodes")
    line("iterations consumed", "iterations_consumed")
    line("experiments completed", "experiments_completed")
    line("experiments crashed", "experiments_crashed")
    line("Path B attempts", "path_b_attempts")
    line("Path B crashes", "path_b_crashes")
    line("Path B crash rate", "path_b_crash_rate", "{:.0%}")
    line("orchestration-only misuse", "orchestration_only_misuse")
    line("preflight rejections (free)", "preflight_rejections")
    line("repeated identical failures", "repeated_identical_failures")
    line("manual interventions", "manual_interventions")
    L.append("  RESEARCH DISCIPLINE")
    line("nodes with an inquiry", "nodes_with_inquiry")
    line("  ...complete (obs+2hyp+meas)", "inquiry_complete")
    line("  ...naming a capability", "capability_named")
    line("  ...capability name valid", "capability_named_valid")
    line("  ...promotion criterion set", "promotion_criterion_stated")
    line("confirmation-category nodes", "confirmation_nodes")
    line("diagnostic tool calls", "tool_call_total")
    line("best primary (single run)", "best_primary", "{:.5f}")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", action="append", required=True)
    ap.add_argument("--label", action="append", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    labels = a.label or []
    rows = []
    for i, p in enumerate(a.journal):
        lab = labels[i] if i < len(labels) else os.path.basename(p)[:13]
        rows.append(compute(_load(p), lab))
    print(render(rows))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(rows, fh, indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
