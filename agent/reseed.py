"""Statistical rigor pass: re-run the top-N journal nodes across multiple
validation seeds to check whether a claimed gain separates from seed noise.

No LLM calls happen here -- this re-executes ALREADY-GENERATED scripts
(logs/solutions/node_NNN.py) through the same agent.executor.run_solution()
sandbox from the Phase 1 hardening work, varying only --seed. It reuses the
seeding convention runtime/train_lib.py already has (a single --seed CLI int
threaded into numpy's default_rng / torch.manual_seed) rather than inventing
a new one.

Cost model -- read this before raising --reseed-top on a big journal:
  This is a separate, later, human-triggered operation on a run whose own
  final_summary.json is already frozen, so it draws NOTHING from
  agent/pricing.py's LLM $ ceiling (no LLM calls happen; there is nothing to
  spend). The real resource is wall-clock, and it's bounded three ways:
    1. Before running anything, total wall-clock is ESTIMATED from each
       node's own already-recorded wall_clock_seconds (same config -> ~seed-
       invariant training time) and the whole operation refuses to start if
       that estimate exceeds the ceiling -- mirrors SpendTracker.would_exceed()
       in agent/pricing.py, applied to time instead of dollars.
    2. The running total is re-checked after every individual rerun, so a
       misestimate still stops early with partial results kept, not a runaway.
    3. A node's ORIGINAL score is already a valid sample at whatever seed it
       was trained with; that's reused instead of rerun, so a K-seed reseed of
       an already-run node only costs K-1 fresh runs. Nodes from before this
       field existed don't carry their seed on the journal -- ASSUMED_SEED
       (0, the project's config default) is used for those and the assumption
       is written into the output, not silently trusted.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agent.contracts import ExperimentTree  # noqa: E402
from agent.executor import run_solution  # noqa: E402
from agent.experience import EXPERIENCE_PATH, append_entry  # noqa: E402

BASELINE_VALID_PRIMARY = 0.6016
BASELINE_SEED_STD = 0.0008        # the official FM baseline's own 5-seed std
ASSUMED_SEED = 0                  # config/agent_config.json's documented default


def load_top_n_nodes(journal_path: str, n: int) -> list[dict]:
    if not os.path.exists(journal_path):
        return []
    nodes = []
    with open(journal_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("status") == "success" and d.get("metrics"):
                nodes.append(d)
    nodes.sort(key=lambda d: -d["metrics"]["primary"])
    return nodes[:n]


def estimate_wall_clock_s(nodes: list[dict], n_seeds: int) -> float:
    """Fresh runs per node = n_seeds, minus 1 for the reused original sample
    (whenever the original seed falls inside the requested seed range)."""
    total = 0.0
    for n in nodes:
        original_seed = n.get("seed")
        if original_seed is None:
            original_seed = ASSUMED_SEED
        n_fresh = n_seeds - (1 if 0 <= original_seed < n_seeds else 0)
        total += n.get("wall_clock_seconds", 0.0) * n_fresh
    return total


def run_reseed(root: str, top_n: int, n_seeds: int = 5,
               wall_clock_limit_h: float = 6.0, exec_timeout_s: int = 1200) -> dict:
    if n_seeds < 2:
        sys.exit("--reseed-seeds must be at least 2 (need >=2 samples for a std)")

    journal_path = os.path.join(root, "logs", "journal.jsonl")
    nodes = load_top_n_nodes(journal_path, top_n)
    if not nodes:
        sys.exit(f"no successful, scored nodes found in {journal_path}")
    if len(nodes) < top_n:
        print(f"! only {len(nodes)} successful node(s) available "
             f"(requested top {top_n})")

    ceiling_s = wall_clock_limit_h * 3600
    est_s = estimate_wall_clock_s(nodes, n_seeds)
    print(f"reseed plan: top {len(nodes)} node(s) x {n_seeds} seed(s) "
         f"(reusing each node's original-seed sample where possible)")
    print(f"estimated wall-clock: {est_s/60:.1f} min "
         f"(ceiling: {wall_clock_limit_h:.2f} h = {ceiling_s/60:.1f} min, "
         f"this is a SEPARATE allowance from the original run's own budget)")
    if est_s > ceiling_s:
        sys.exit(f"REFUSED before starting: estimated {est_s/60:.1f} min exceeds "
                 f"the {wall_clock_limit_h:.2f} h reseed ceiling. Raise "
                 f"--reseed-wall-clock-limit-h, or lower --reseed-top / "
                 f"--reseed-seeds.")

    runs_dir = os.path.join(root, "logs", "reseed_runs")
    os.makedirs(runs_dir, exist_ok=True)

    t_start = time.time()
    stopped_early, stop_reason = False, None
    node_results = []

    for n in nodes:
        iid = n["iteration_id"]
        code_path = n.get("code_path", "")
        if not code_path or not os.path.exists(code_path):
            print(f"! node {iid}: code_path missing on disk, skipping")
            continue
        with open(code_path) as fh:
            code = fh.read()

        original_seed = n.get("seed")
        seed_assumed = original_seed is None
        if original_seed is None:
            original_seed = ASSUMED_SEED

        samples: dict[int, float] = {}
        seed_source: dict[int, str] = {}
        if 0 <= original_seed < n_seeds:
            samples[original_seed] = n["metrics"]["primary"]
            seed_source[original_seed] = "reused_original"

        for s in range(n_seeds):
            if s in samples:
                continue
            if stopped_early:
                seed_source[s] = "skipped (wall-clock ceiling reached)"
                continue
            elapsed = time.time() - t_start
            if elapsed >= ceiling_s:
                stopped_early = True
                stop_reason = (f"reseed wall-clock ceiling reached "
                               f"({wall_clock_limit_h:.2f} h) after node {iid}, "
                               f"seed {s}")
                seed_source[s] = "skipped (wall-clock ceiling reached)"
                continue
            run_dir = os.path.join(runs_dir, f"node_{iid:03d}", f"seed_{s}")
            code_copy_path = os.path.join(run_dir, "solution.py")
            res = run_solution(code, code_copy_path, n["menu_choices"], run_dir,
                              timeout_s=exec_timeout_s, seed=s)
            if res.ok:
                samples[s] = res.metrics["primary"]
                seed_source[s] = "fresh"
            else:
                seed_source[s] = f"FAILED: {(res.error_trace or '')[:120]}"

        scores = list(samples.values())
        result = {
            "iteration_id": iid,
            "menu_choices": n["menu_choices"],
            "original_seed": original_seed,
            "original_seed_assumed": seed_assumed,
            "seed_source": {str(k): v for k, v in seed_source.items()},
            "primary_scores": {str(k): v for k, v in samples.items()},
            "n_samples": len(scores),
            "n_requested": n_seeds,
            "mean_primary": statistics.fmean(scores) if scores else None,
            "std_primary": (statistics.stdev(scores)  # sample std, ddof=1
                           if len(scores) >= 2 else None),
            "original_single_seed_primary": n["metrics"]["primary"],
        }
        node_results.append(result)
        m = result["mean_primary"]
        sd = result["std_primary"]
        print(f"  node {iid}: {len(scores)}/{n_seeds} samples -- "
             f"mean {m:.4f}" + (f" +/- {sd:.4f}" if sd is not None else "") +
             f" (single-seed was {result['original_single_seed_primary']:.4f})")

    scored = [r for r in node_results if r["mean_primary"] is not None]
    new_best = max(scored, key=lambda r: r["mean_primary"], default=None)
    original_best = max(node_results, key=lambda r: r["original_single_seed_primary"],
                        default=None)

    summary = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "top_n_requested": top_n,
        "n_seeds_requested": n_seeds,
        "reseed_wall_clock_limit_h": wall_clock_limit_h,
        "total_wall_clock_seconds": round(time.time() - t_start, 1),
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "nodes": node_results,
        "original_best_node": original_best["iteration_id"] if original_best else None,
        "best_by_mean_node": new_best["iteration_id"] if new_best else None,
        "best_changed": (bool(original_best) and bool(new_best)
                        and original_best["iteration_id"] != new_best["iteration_id"]),
    }
    if new_best is not None:
        d = new_best["mean_primary"] - BASELINE_VALID_PRIMARY
        summary["best_by_mean_delta_over_baseline"] = round(d, 4)
        summary["best_by_mean_delta_in_baseline_seed_sigmas"] = round(d / BASELINE_SEED_STD, 2)

    out_path = os.path.join(root, "logs", "reseed_results.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {out_path}")
    if summary["best_changed"]:
        print(f"BEST NODE CHANGED under reseeding: node {summary['original_best_node']} "
             f"(single-seed) -> node {summary['best_by_mean_node']} (mean over seeds)")
        apply_best_override(root, summary)
    else:
        print(f"best node unchanged under reseeding: node {summary['best_by_mean_node']}")
    return summary


def apply_best_override(root: str, summary: dict,
                        experience_path: str = EXPERIENCE_PATH) -> None:
    """Point logs/best_solution.py / best_metrics.json at the mean-verified
    winner, if reseeding found one that differs from the single-seed pick the
    live run recorded. Idempotent: safe to call again against an already-saved
    reseed_results.json without re-running any training.

    Also records the switch itself, durably: best_metrics.json only holds the
    CURRENT state, so a second override later would silently erase all trace
    of this one. logs/best_override_log.jsonl is append-only (same pattern as
    logs/final_eval_override_attempts.jsonl from Phase 1) and is the permanent
    record; an agent/experience.md entry additionally makes the correction
    prompt-visible if this journal is ever resumed rather than started fresh.
    """
    if not summary.get("best_changed"):
        return
    new_id = summary["best_by_mean_node"]
    old_id = summary["original_best_node"]
    by_id = {n["iteration_id"]: n for n in summary["nodes"]}
    winner, old = by_id[new_id], by_id.get(old_id)
    old_single = old["original_single_seed_primary"] if old else None
    old_mean = old["mean_primary"] if old else None

    if old is not None:
        reason = (f"node {new_id} mean {winner['mean_primary']:.4f} over "
                 f"{winner['n_samples']} seeds beat node {old_id}'s mean "
                 f"({old_mean:.4f}) under the same reseed pass -- node {old_id}'s "
                 f"single-seed pick of {old_single:.4f} did not hold up as the "
                 f"true best.")
    else:
        reason = (f"node {new_id} mean {winner['mean_primary']:.4f} beat the "
                 f"recorded single-seed best (node {old_id}).")

    tree = ExperimentTree(os.path.join(root, "logs"))
    tree.override_best_artifacts(new_id, extra={
        "reseed_verified": True,
        "reseed_mean_primary": winner["mean_primary"],
        "reseed_std_primary": winner["std_primary"],
        "reseed_n_samples": winner["n_samples"],
        "superseded_single_seed_best_node": old_id,
        "superseded_single_seed_best_primary": old_single,
        "override_reason": reason,
    })

    log_path = os.path.join(root, "logs", "best_override_log.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(json.dumps({
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "old_best_node": old_id,
            "old_best_single_seed_primary": old_single,
            "old_best_reseed_mean_primary": old_mean,
            "new_best_node": new_id,
            "new_best_single_seed_primary": winner["original_single_seed_primary"],
            "new_best_reseed_mean_primary": winner["mean_primary"],
            "new_best_reseed_std_primary": winner["std_primary"],
            "n_seeds": winner["n_samples"],
            "reason": reason,
        }) + "\n")

    old_single_s = f"{old_single:.4f}" if old_single is not None else "unrecorded"
    append_entry(new_id, "CORRECTION", f"best node corrected: {old_id} -> {new_id}",
                f"A {winner['n_samples']}-seed reseed found node {old_id}'s single-seed "
                f"score ({old_single_s}) was seed-lucky; node {new_id}'s true mean "
                f"({winner['mean_primary']:.4f}) is actually higher. {reason} Don't "
                f"treat a single high score as decisive without checking its variance.",
                path=experience_path)

    print(f"updated logs/best_metrics.json + logs/best_solution.py to point at "
         f"node {new_id} (reseed-verified, was node {old_id})")
    print(f"wrote {log_path}")
