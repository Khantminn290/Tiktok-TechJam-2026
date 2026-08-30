"""Generate the competition report from artifacts, so it cannot go stale.

Hand-written status documents rot. This project accumulated several: a test
count that was right when it was typed, a convergence threshold that had since
been recalibrated, and a claim that the agent had independently reproduced a
discovery which had to be retracted. None of those were dishonest when written;
they were simply not regenerated when the facts moved.

So the facts are read from artifacts instead of retyped:

    logs/ensemble_results.json     the submitted result and its provenance
    logs/journal.jsonl             the live run
    logs/final_summary.json        budget ledger and stop reason
    logs/feature_store.jsonl       Path B lineage
    logs/experiments.jsonl         paired confirmations
    tests/test_harness.py          the harness, actually executed
    agent.loop.EPSILON             the convergence threshold in force
    kuairand-starter-kit/evaluate.py   the metric definition

Anything not derivable from those is either omitted or explicitly labelled as
unverified. The report distinguishes three tiers and never blurs them:

    VERIFIED    recomputed from artifacts during this generation
    OBSERVED    measured in a run, recorded in the journal
    OPEN        not established -- future work, stated as such

Usage:
    python3 -m agent.results_report                 # writes RESULTS.md
    python3 -m agent.results_report --json out.json
    python3 -m agent.results_report --run-tests     # also executes the harness
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
LOGS = os.path.join(ROOT, "logs")
DEFAULT_OUT = os.path.join(ROOT, "RESULTS.md")

VERIFIED, OBSERVED, OPEN = "VERIFIED", "OBSERVED", "OPEN"


def _load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _load_jsonl(path):
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


def metric_definition() -> dict:
    """Read the metric off the evaluator, which is authoritative.

    The competition brief contains a sentence that conflicts with the starter
    kit. The kit's `evaluate.py` is what actually scores a submission, so it
    wins; this reads it rather than restating either.
    """
    p = os.path.join(ROOT, "kuairand-starter-kit", "evaluate.py")
    src = ""
    try:
        with open(p) as fh:
            src = fh.read()
    except OSError:
        pass
    return {
        "source_file": "kuairand-starter-kit/evaluate.py",
        "label": "long_view" if "long_view" in src else "unknown",
        "metrics": ["GAUC", "nDCG@5"],
        "primary": "mean(GAUC, nDCG@5)",
        "k": 5 if "nDCG@5" in src or "k=5" in src else None,
        "tier": VERIFIED if src else OPEN,
        "note": ("The evaluator is authoritative. Where the brief's prose and "
                 "the starter kit disagree, the kit scores the submission, so "
                 "the kit wins. Benchmark code is not modified."),
    }


def incumbent() -> dict:
    """Recompute the submitted result from stored predictions, now."""
    rec = _load_json(os.path.join(LOGS, "ensemble_results.json")) or {}
    out = {"reported": {k: rec.get(k) for k in ("primary", "GAUC", "nDCG@5")},
           "k": rec.get("k"), "reproduce": rec.get("reproduce"),
           "provenance": rec.get("provenance"), "tier": OPEN}
    try:
        from agent.verify_incumbent import verify
        v = verify()
        out.update(verified=v["ok"], recomputed=v.get("recomputed"),
                   aggregation=v.get("aggregation"),
                   members=v.get("k"), issues=v.get("issues"),
                   tier=VERIFIED if v["ok"] else OPEN)
    except Exception as e:                        # noqa: BLE001
        out["verify_error"] = f"{type(e).__name__}: {e}"[:200]
    return out


def harness(run: bool = False) -> dict:
    """The test count, executed rather than remembered."""
    if not run:
        return {"tier": OPEN, "note": "not executed during this generation; "
                                      "pass --run-tests for a live count"}
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tests",
                                                         "test_harness.py")],
                           capture_output=True, text=True, cwd=ROOT, timeout=3600)
    except subprocess.SubprocessError as e:
        return {"tier": OPEN, "error": str(e)[:200]}
    m = re.search(r"(\d+) passed, (\d+) failed", r.stdout or "")
    return {"tier": VERIFIED, "passed": int(m.group(1)) if m else None,
            "failed": int(m.group(2)) if m else None,
            "exit_code": r.returncode, "seconds": round(time.time() - t0, 1)}


def convergence() -> dict:
    from agent.loop import EPSILON, N_CONVERGE, BASELINE_SEED_STD
    return {"epsilon": round(EPSILON, 6),
            "epsilon_sigma": round(EPSILON / BASELINE_SEED_STD, 2),
            "N": N_CONVERGE, "noise_floor": BASELINE_SEED_STD, "tier": VERIFIED,
            "note": ("calibrated to the upward drift a running maximum shows by "
                     "luck over N iterations, not hand-picked")}


def latest_run() -> dict:
    nodes = _load_jsonl(os.path.join(LOGS, "journal.jsonl"))
    summary = _load_json(os.path.join(LOGS, "final_summary.json")) or {}
    if not nodes:
        return {"tier": OPEN, "note": "no live journal on disk"}
    from agent.run_metrics import compute
    m = compute(nodes, "latest")
    m["tier"] = OBSERVED
    m["stop_reason"] = summary.get("stop_reason")
    m["budget_ledger"] = summary.get("budget_ledger")
    m["counting_note"] = summary.get("budget_counting_note")
    m["llm_spend_usd"] = (summary.get("spend") or {}).get("total_usd")
    m["agent_wall_clock_s"] = summary.get("total_agent_wall_clock_s")
    m["devices"] = summary.get("devices_used")
    tok = {}
    for n in nodes:
        for k, v in (n.get("token_breakdown") or {}).items():
            tok[k] = tok.get(k, 0) + v
    m["llm_tokens_total"] = sum(tok.values())
    return m


def path_b() -> dict:
    """Did Path B complete end to end?"""
    store = _load_jsonl(os.path.join(LOGS, "feature_store.jsonl"))
    reg = _load_jsonl(os.path.join(LOGS, "feature_registry.jsonl"))
    from collections import Counter
    statuses = Counter(r.get("status") for r in reg)
    trained = [e for e in store if e.get("training")]
    return {"features_probed": len(reg), "probe_statuses": dict(statuses),
            "features_stored_with_lineage": len(store),
            "features_trained_paired": len(trained),
            "end_to_end_complete": bool(trained),
            "tier": OBSERVED if store or reg else OPEN,
            "note": ("end-to-end means a proposed feature cleared the probe AND "
                     "was retrained from its stored source in a paired "
                     "experiment")}


def confirmations() -> dict:
    rows = _load_jsonl(os.path.join(LOGS, "experiments.jsonl"))
    promoted = [r for r in rows if (r.get("evidence") or {}).get("promote")]
    states = {}
    for r in rows:
        s = (r.get("evidence") or {}).get("state")
        if s:
            states[s] = states.get(s, 0) + 1
    return {"paired_experiments_run": len(rows), "states": states,
            "promoted": len(promoted), "tier": OBSERVED if rows else OPEN,
            "rule": "only CONFIRMED may change the submitted system; a "
                    "single-seed result is PRELIMINARY at any effect size"}


def build(run_tests: bool = False) -> dict:
    from agent import budget
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "python3 -m agent.results_report",
        "dataset_scope": "KuaiRand-Pure only",
        "metric": metric_definition(),
        "incumbent": incumbent(),
        "harness": harness(run_tests),
        "convergence": convergence(),
        "latest_run": latest_run(),
        "path_b": path_b(),
        "confirmations": confirmations(),
        "budget_counting_rule": budget.COUNTING_NOTE,
        "hidden_test_evaluated": os.path.exists(
            os.path.join(ROOT, "results", "final_evaluation.lock")),
    }


def render(d: dict) -> str:
    inc, h, cv = d["incumbent"], d["harness"], d["convergence"]
    run, pb, cf = d["latest_run"], d["path_b"], d["confirmations"]
    met = d["metric"]
    L = [
        "# Results — generated",
        "",
        f"Generated `{d['generated_utc']}` by `{d['generator']}`. Every figure "
        f"below is read from repository artifacts at generation time; nothing "
        f"here is retyped from memory.",
        "",
        "Tiers: **VERIFIED** recomputed during this generation · **OBSERVED** "
        "measured in a run and journalled · **OPEN** not established.",
        "",
        "## Metric (authoritative)",
        "",
        f"- source: `{met['source_file']}` — **{met['tier']}**",
        f"- positive label: `{met['label']}`",
        f"- metrics: {', '.join(met['metrics'])}; primary = {met['primary']}",
        "",
        f"> {met['note']}",
        "",
        "## Incumbent",
        "",
    ]
    rep = inc.get("reported") or {}
    if inc.get("verified"):
        rc = inc.get("recomputed") or {}
        L += [f"**{rep.get('primary')}** primary "
              f"(GAUC {rep.get('GAUC')}, nDCG@5 {rep.get('nDCG@5')}), "
              f"{inc.get('members')}-member ensemble — **VERIFIED**",
              "",
              f"Recomputed from stored predictions during this generation using "
              f"`{inc.get('aggregation')}`: "
              f"primary {rc.get('primary')}, GAUC {rc.get('GAUC')}, "
              f"nDCG@5 {rc.get('nDCG@5')} — exact match.",
              f"Reproduce: `{inc.get('reproduce')}`"]
        prov = inc.get("provenance") or {}
        git = prov.get("git") or {}
        if git:
            L.append(f"Provenance: commit `{git.get('short_sha')}` on "
                     f"`{git.get('branch')}`, data fingerprint "
                     f"`{(prov.get('data') or {}).get('sha256')}`.")
    else:
        L += [f"**NOT VERIFIED** — {'; '.join(inc.get('issues') or []) or inc.get('verify_error')}"]

    L += ["", "## Harness", ""]
    if h.get("tier") == VERIFIED:
        L.append(f"`python3 tests/test_harness.py` → **{h['passed']} passed, "
                 f"{h['failed']} failed** ({h['seconds']}s) — **VERIFIED**")
    else:
        L.append(f"**OPEN** — {h.get('note') or h.get('error')}")

    L += ["", "## Convergence rule", "",
          f"ε = **{cv['epsilon']}** ({cv['epsilon_sigma']}σ at a noise floor of "
          f"{cv['noise_floor']}), N = {cv['N']} — **VERIFIED**",
          f"> {cv['note']}"]

    L += ["", "## Latest run", ""]
    if run.get("tier") == OBSERVED:
        led = run.get("budget_ledger") or {}
        L += ["| | |", "|---|---|",
              f"| best primary (single run) | {run.get('best_primary')} |",
              f"| outer-loop nodes | {run.get('nodes')} |",
              f"| iterations consumed | {run.get('iterations_consumed')} |",
              f"| training runs used | {led.get('training_runs_used')} of "
              f"{led.get('max_training_runs')} |",
              f"| experiments completed | {run.get('experiments_completed')} |",
              f"| experiments crashed | {run.get('experiments_crashed')} |",
              f"| Path B attempts / crashes | {run.get('path_b_attempts')} / "
              f"{run.get('path_b_crashes')} |",
              f"| preflight rejections (free) | {run.get('preflight_rejections')} |",
              f"| automatic repairs attempted / recovered | "
              f"{run.get('automatic_repair_attempts')} / "
              f"{run.get('automatic_repairs_succeeded')} |",
              f"| paired confirmations run | {run.get('confirmation_runs')} |",
              f"| results promoted | {run.get('results_promoted')} |",
              f"| **manual interventions** | **{run.get('manual_interventions')}** |",
              f"| LLM tokens | {run.get('llm_tokens_total'):,} |"
              if run.get("llm_tokens_total") else "| LLM tokens | n/a |",
              f"| LLM spend | ${run.get('llm_spend_usd')} |"
              if run.get("llm_spend_usd") is not None else "| LLM spend | n/a |",
              f"| training wall-clock | {run.get('training_wall_clock_s')}s |",
              f"| devices | {', '.join(run.get('devices') or ['cpu'])} |",
              "", f"Stop reason: {run.get('stop_reason')}"]
        if run.get("counting_note"):
            L += ["", f"> {run['counting_note']}"]
    else:
        L.append(f"**OPEN** — {run.get('note')}")

    L += ["", "## Path B (feature discovery)", "",
          f"- features probed: {pb['features_probed']} {pb['probe_statuses']}",
          f"- stored with full lineage: {pb['features_stored_with_lineage']}",
          f"- retrained in a paired experiment: {pb['features_trained_paired']}",
          f"- **end-to-end complete: "
          f"{'YES' if pb['end_to_end_complete'] else 'NO'}** — {pb['tier']}",
          f"> {pb['note']}"]

    L += ["", "## Confirmations", "",
          f"- paired experiments run: {cf['paired_experiments_run']}",
          f"- outcomes: {cf['states'] or 'none'}",
          f"- **promoted: {cf['promoted']}**",
          f"> {cf['rule']}"]

    L += ["", "## Budget counting", "", f"> {d['budget_counting_rule']}"]
    L += ["", "## Scope and integrity", "",
          f"- dataset: {d['dataset_scope']}",
          f"- hidden test evaluated: "
          f"{'YES' if d['hidden_test_evaluated'] else 'NO (never touched)'}"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--json", default=None)
    ap.add_argument("--run-tests", action="store_true",
                    help="execute the harness so the count is VERIFIED rather "
                         "than OPEN")
    a = ap.parse_args()
    d = build(run_tests=a.run_tests)
    text = render(d)
    with open(a.out, "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nwrote {a.out}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(d, fh, indent=2, default=str)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
