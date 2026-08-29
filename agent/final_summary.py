"""Automated end-of-run research summary -- a required competition deliverable.

Generated deterministically from the journal, reseed artifacts, override log
and interventions log, so the numbers a judge reads are the numbers that were
actually measured. Nothing here is LLM-authored.

Deliberately reports the three performance figures separately, because
collapsing them is how a lucky draw becomes "the result":
    best OBSERVED single run   -- one draw
    reseed-VERIFIED mean       -- expected performance of one model
    ENSEMBLE expectation       -- expected performance of the submitted system

Usage: python3 -m agent.final_summary [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
LOGS = os.path.join(ROOT, "logs")
BASELINE = 0.6016
BASELINE_TEST = 0.5946
SIGMA = 0.0008


def _load_jsonl(p):
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def _load(p):
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def build(root: str = ROOT) -> dict:
    logs = os.path.join(root, "logs")
    nodes = _load_jsonl(os.path.join(logs, "journal.jsonl"))
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    reseed = _load(os.path.join(logs, "reseed_results.json"))
    ens = _load(os.path.join(logs, "ensemble_results.json"))
    final = _load(os.path.join(logs, "final_summary.json"))
    interventions = _load_jsonl(os.path.join(logs, "interventions.jsonl"))
    overrides = _load_jsonl(os.path.join(logs, "best_override_log.jsonl"))
    menu = _load(os.path.join(root, "config", "modification_menu.json")) or {}

    s = {"iterations_used": len(nodes),
         "scored": len(scored), "failed": len(nodes) - len(scored),
         "manual_interventions": len(interventions),
         "baseline_valid_primary": BASELINE,
         "baseline_test_primary": BASELINE_TEST}

    if scored:
        b = max(scored, key=lambda n: n["metrics"]["primary"])
        s["best_observed"] = {
            "node": b["iteration_id"], "primary": round(b["metrics"]["primary"], 5),
            "GAUC": round(b["metrics"]["GAUC"], 5),
            "nDCG@5": round(b["metrics"]["nDCG@5"], 5),
            "menu_choices": b.get("menu_choices"),
            "caveat": "single draw -- NOT an expected value"}
    if reseed and reseed.get("nodes"):
        top = max((n for n in reseed["nodes"] if n.get("mean_primary")),
                  key=lambda n: n["mean_primary"], default=None)
        if top:
            d = top["mean_primary"] - BASELINE
            s["reseed_verified"] = {
                "node": top["iteration_id"],
                "mean_primary": round(top["mean_primary"], 5),
                "std": round(top.get("std_primary") or 0.0, 5),
                "n_seeds": top.get("n_samples"),
                "delta_vs_baseline": round(d, 5),
                "sigma": round(d / SIGMA, 2)}
    if ens:
        # "primary" is the current schema written by agent.final_ensemble;
        # "mean" is the older key, kept readable so archived runs still render.
        p = ens.get("primary", ens.get("mean")) or 0
        d = p - BASELINE
        s["submitted_ensemble"] = {
            "mean_primary": p, "std": ens.get("single_seed_std", ens.get("std")),
            "k": ens.get("k"), "delta_vs_baseline": round(d, 5),
            "sigma": round(d / SIGMA, 2), "config": ens.get("config"),
            "members_dir": ens.get("members_dir"),
            "reproduce": ens.get("reproduce")}

    # research process
    cats, paths, families, fails = {}, {}, set(), {}
    for n in nodes:
        c = (n.get("research_category") or "").lower()
        if c:
            cats[c] = cats.get(c, 0) + 1
        p = (n.get("implementation_path") or "").upper()
        if p:
            paths[p] = paths.get(p, 0) + 1
        m = (n.get("menu_choices") or {}).get("model")
        if m:
            families.add(m)
        for e in (n.get("events") or []):
            if e.get("type") == "execution_error" and e.get("failure_class"):
                fails[e["failure_class"]] = fails.get(e["failure_class"], 0) + 1
    s["research_categories"] = cats
    s["implementation_paths"] = paths
    s["model_families_explored"] = sorted(families)
    s["failure_classes"] = fails
    s["recovery_events"] = sum(1 for n in nodes
                               if n.get("action") == "debug" and n.get("status") == "success")
    s["dead_ends_recorded"] = len((menu.get("notes") or {}).get("tested_dead_ends", []))
    s["best_node_overrides"] = len(overrides)

    # resources
    tok = {}
    for n in nodes:
        for k, v in (n.get("token_breakdown") or {}).items():
            tok[k] = tok.get(k, 0) + v
    s["llm_tokens"] = tok
    s["llm_tokens_total"] = sum(tok.values())
    s["training_wall_clock_s"] = round(sum(n.get("wall_clock_seconds", 0.0)
                                           for n in nodes), 1)
    if final:
        s["agent_wall_clock_s"] = final.get("total_agent_wall_clock_s")
        s["gpu_hours"] = final.get("gpu_hours", 0.0)
        s["devices_used"] = final.get("devices_used", ["cpu"])
        s["stop_reason"] = final.get("stop_reason")
        s["llm_spend_usd"] = (final.get("spend") or {}).get("total_usd")
    s["token_caveat"] = ("agent inference only; excludes the human-driven "
                         "development session, which is not instrumented and is "
                         "typically far larger")
    s["hidden_test_used"] = os.path.exists(
        os.path.join(root, "results", "final_evaluation.lock"))
    return s


def render(s: dict) -> str:
    L = ["=" * 74, "FINAL RUN SUMMARY — Autonomous ML Research Agent (KuaiRand-Pure)",
         "=" * 74,
         f"Iterations:              {s['iterations_used']}  "
         f"({s['scored']} scored, {s['failed']} failed)",
         f"Manual interventions:    {s['manual_interventions']}",
         f"Baseline (valid):        {s['baseline_valid_primary']}"]
    if "best_observed" in s:
        b = s["best_observed"]
        L.append(f"Best OBSERVED run:       {b['primary']} (node {b['node']}, "
                 f"GAUC {b['GAUC']}, nDCG@5 {b['nDCG@5']}) — {b['caveat']}")
    if "reseed_verified" in s:
        r = s["reseed_verified"]
        L.append(f"Reseed-VERIFIED mean:    {r['mean_primary']} +/- {r['std']} "
                 f"over {r['n_seeds']} seeds = {r['sigma']} sigma vs baseline")
    if "submitted_ensemble" in s:
        e = s["submitted_ensemble"]
        L.append(f"SUBMITTED ensemble:      {e['mean_primary']} +/- {e['std']} "
                 f"({e['k']} ckpts) = {e['sigma']} sigma vs baseline")
    L += ["", "RESEARCH PROCESS",
          f"  research categories:   {s['research_categories'] or 'n/a'}",
          f"  implementation paths:  {s['implementation_paths'] or 'n/a'}",
          f"  model families:        {', '.join(s['model_families_explored']) or 'n/a'}",
          f"  failure classes:       {s['failure_classes'] or 'none'}",
          f"  debug recoveries:      {s['recovery_events']}",
          f"  dead ends recorded:    {s['dead_ends_recorded']}",
          f"  best-node overrides:   {s['best_node_overrides']}",
          "", "RESOURCES",
          f"  LLM tokens (agent):    {s['llm_tokens_total']:,d}",
          f"  training wall-clock:   {s['training_wall_clock_s']}s"]
    if "agent_wall_clock_s" in s:
        L.append(f"  agent wall-clock:      {s['agent_wall_clock_s']}s")
    if "gpu_hours" in s:
        L.append(f"  GPU-hours (measured):  {s['gpu_hours']} "
                 f"(devices: {', '.join(s.get('devices_used', ['cpu']))})")
    if s.get("llm_spend_usd") is not None:
        L.append(f"  LLM spend:             ${s['llm_spend_usd']:.4f}")
    L.append(f"  NOTE: {s['token_caveat']}")
    L += ["", f"Stop reason:             {s.get('stop_reason', 'n/a')}",
          f"Hidden test evaluated:   {'YES' if s['hidden_test_used'] else 'NO (never touched)'}"]
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    s = build()
    print(render(s))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(s, fh, indent=2)
        print(f"\nwrote {a.json}")
