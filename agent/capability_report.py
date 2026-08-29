"""Measure autonomous RESEARCH CAPABILITY from a journal -- not just score.

Stage F. The redesign must be judged on whether the agent actually behaves
differently, because "the schema field says B" is not evidence of custom code
and "the score went up" can be luck. Everything here is computed from the
journal and the generated scripts on disk.

The distinction that matters most:

    FAKE Path B   -- declares path B, then calls train_lib.run() with menu
                     choices anyway. Structurally identical to Path A.
    GENUINE Path B-- implements a mechanism: no bare train_lib.run() delegation,
                     real code of its own (functions/classes/training loop).

Usage:  python3 -m agent.capability_report [journal.jsonl ...]
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)


def _load(path):
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


def _script_shape(code_path: str) -> dict:
    """Structural read of a generated script. A seed_solution.py clone has one
    train_lib.run() call and essentially no logic of its own."""
    if not code_path or not os.path.exists(code_path):
        return {"exists": False}
    src = open(code_path).read()
    body = [ln for ln in src.splitlines()
            if ln.strip() and not ln.strip().startswith("#")]
    return {"exists": True,
            "lines": len(body),
            "run_calls": len(re.findall(r"train_lib\.run\s*\(", src)),
            "defs": len(re.findall(r"^\s*(def|class)\s", src, re.M)),
            "uses_primitives": bool(re.search(
                r"train_lib\.(load_cache|encode_features|RankFM|History|"
                r"train_numpy_fm|evaluate|build_causal_sequences)", src)),
            "has_loop": bool(re.search(r"for\s+\w+\s+in\s+range\(", src))}


def analyse(journal_path: str) -> dict:
    nodes = _load(journal_path)
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    rep = {"journal": journal_path, "nodes": len(nodes), "scored": len(scored),
           "failed": len(nodes) - len(scored)}

    # --- 1/2/3: path usage, genuine vs fake ---
    declared_b = [n for n in nodes if (n.get("implementation_path") or "").upper() == "B"]
    declared_a = [n for n in nodes if (n.get("implementation_path") or "").upper() == "A"]
    genuine, fake = [], []
    for n in declared_b:
        sh = _script_shape(n.get("code_path", ""))
        if not sh.get("exists"):
            continue
        # genuine = does not merely delegate to train_lib.run, and carries
        # real logic of its own
        if sh["run_calls"] == 0 and (sh["defs"] >= 1 or sh["has_loop"]) \
                and sh["uses_primitives"]:
            genuine.append(n["iteration_id"])
        else:
            fake.append(n["iteration_id"])
    rep["path_A"] = len(declared_a)
    rep["path_B_declared"] = len(declared_b)
    rep["path_B_genuine"] = len(genuine)
    rep["path_B_fake"] = len(fake)
    rep["path_B_genuine_ids"] = genuine

    shapes = [_script_shape(n.get("code_path", "")) for n in nodes]
    shapes = [s for s in shapes if s.get("exists")]
    rep["scripts_delegating_to_train_lib_run"] = sum(1 for s in shapes
                                                     if s["run_calls"] >= 1)
    rep["scripts_with_own_logic"] = sum(1 for s in shapes
                                        if s["run_calls"] == 0 and s["defs"] >= 1)
    rep["median_script_lines"] = (sorted(s["lines"] for s in shapes)[len(shapes) // 2]
                                  if shapes else 0)

    # --- 4/5: structural diversity ---
    combos = {(  (n.get("menu_choices") or {}).get("loss"),
                 (n.get("menu_choices") or {}).get("model"))
              for n in scored}
    rep["distinct_loss_model_combos"] = len({c for c in combos if any(c)})
    rep["distinct_configs"] = len({json.dumps(n.get("menu_choices") or {}, sort_keys=True)
                                   for n in scored})
    rep["mechanism_families"] = sorted({str(c[1]) for c in combos if c[1]})

    # --- 6: did the agent choose objectives, incl. ablation? ---
    cats = {}
    for n in nodes:
        c = (n.get("research_category") or "").lower()
        if c:
            cats[c] = cats.get(c, 0) + 1
    rep["research_categories"] = cats
    rep["ablation_fired"] = cats.get("ablation", 0) > 0

    # --- 7: dead-end avoidance ---
    rep["failure_classes"] = {}
    for n in nodes:
        for e in (n.get("events") or []):
            if e.get("type") == "execution_error" and e.get("failure_class"):
                fc = e["failure_class"]
                rep["failure_classes"][fc] = rep["failure_classes"].get(fc, 0) + 1
    # repetition = same exact config scored more than once
    sigs = {}
    for n in scored:
        s = json.dumps(n.get("menu_choices") or {}, sort_keys=True)
        sigs[s] = sigs.get(s, 0) + 1
    rep["duplicate_config_runs"] = sum(v - 1 for v in sigs.values() if v > 1)
    rep["duplicate_rate"] = round(rep["duplicate_config_runs"] / max(1, len(scored)), 3)

    # --- outcome ---
    if scored:
        best = max(scored, key=lambda n: n["metrics"]["primary"])
        rep["best_observed"] = round(best["metrics"]["primary"], 5)
        rep["best_node"] = best["iteration_id"]
    return rep


def render(reps: list) -> str:
    L = ["=" * 74, "AUTONOMOUS RESEARCH CAPABILITY REPORT", "=" * 74]
    keys = [("nodes", "experiments run"), ("scored", "scored"), ("failed", "failed"),
            ("path_A", "Path A experiments"),
            ("path_B_declared", "Path B declared"),
            ("path_B_genuine", "Path B GENUINE (own mechanism)"),
            ("path_B_fake", "Path B fake (delegates to train_lib.run)"),
            ("scripts_with_own_logic", "scripts with own logic"),
            ("median_script_lines", "median script lines"),
            ("distinct_loss_model_combos", "distinct (loss, model) combos"),
            ("distinct_configs", "distinct configurations"),
            ("duplicate_rate", "duplicate-config rate"),
            ("ablation_fired", "ablation objective fired"),
            ("best_observed", "best observed primary")]
    for r in reps:
        L.append(f"\n--- {r['journal']}")
        for k, label in keys:
            if k in r:
                L.append(f"  {label:42s} {r[k]}")
        if r.get("research_categories"):
            L.append(f"  {'research categories':42s} {r['research_categories']}")
        if r.get("mechanism_families"):
            L.append(f"  {'model families explored':42s} {r['mechanism_families']}")
        if r.get("failure_classes"):
            L.append(f"  {'failure classes seen':42s} {r['failure_classes']}")
    if len(reps) == 2:
        a, b = reps
        L.append("\n" + "-" * 74)
        L.append("A/B DELTA (second minus first)")
        for k, label in keys:
            if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)):
                L.append(f"  {label:42s} {b[k] - a[k]:+g}")
    return "\n".join(L)


if __name__ == "__main__":
    paths = sys.argv[1:] or [os.path.join(ROOT, "logs", "journal.jsonl")]
    print(render([analyse(p) for p in paths]))
