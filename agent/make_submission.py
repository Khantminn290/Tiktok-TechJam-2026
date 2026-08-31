"""Generate + validate a submission CSV from the current best node's saved scores.

The best node's training run already wrote row_id-aligned scores_{valid,test}.npy
(the executor validated shape and finiteness), so a submission is a deterministic
re-format, checked with the starter kit's own read_submission.

Usage:
  python3 -m agent.make_submission                       # test-split submission.csv
  python3 -m agent.make_submission --split valid --score # valid CSV + local score
  python3 -m agent.make_submission --final-test-eval     # THE one-time test eval

--final-test-eval is the single hidden-test evaluation of the whole run: it scores
the test submission once with the official evaluate.py and writes
results/final_results.json. Do not run it during development.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(_ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)

from data import load  # noqa: E402
from evaluate import evaluate  # noqa: E402
from submit import read_submission, write_submission  # noqa: E402

BASELINE_VALID = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
BASELINE_TEST = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}
BASELINE_SEED_STD = 0.0008

DEFAULT_LOCK_PATH = os.path.join(_ROOT, "results", "final_evaluation.lock")
DEFAULT_OVERRIDE_LOG = os.path.join(_ROOT, "logs", "final_eval_override_attempts.jsonl")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _log_override_attempt(lock_existed: bool, allowed: bool,
                          override_log_path: str = DEFAULT_OVERRIDE_LOG) -> None:
    os.makedirs(os.path.dirname(override_log_path), exist_ok=True)
    with open(override_log_path, "a") as fh:
        fh.write(json.dumps({
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "lock_existed": lock_existed,
            "allowed": allowed,
        }) + "\n")


def check_final_eval_guard(admin_override: bool, lock_path: str = DEFAULT_LOCK_PATH,
                           override_log_path: str = DEFAULT_OVERRIDE_LOG) -> None:
    """Refuse a second --final-test-eval unless a human explicitly overrides it.

    The organizers score a ONE-TIME hidden-test evaluation. A lock file makes
    that true by construction: a second attempt fails loudly (sys.exit) unless
    --admin-override is passed, and either way the attempt is logged, so the
    override log can't be flattered by the thing it's supposed to catch.
    """
    lock_existed = os.path.exists(lock_path)
    if lock_existed and not admin_override:
        _log_override_attempt(lock_existed=True, allowed=False,
                              override_log_path=override_log_path)
        sys.exit(
            f"REFUSED: {lock_path} already exists -- the one-time hidden-test "
            f"evaluation has already run. Pass --admin-override to force a "
            f"second evaluation (this gets logged to {override_log_path}).")
    if admin_override:
        _log_override_attempt(lock_existed=lock_existed, allowed=True,
                              override_log_path=override_log_path)
        print(f"! --admin-override used (lock already existed: {lock_existed}) "
             f"-- logged to {override_log_path}")


def write_final_eval_lock(submission_path: str, test_metrics: dict,
                          lock_path: str = DEFAULT_LOCK_PATH) -> None:
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as fh:
        json.dump({
            "submission_path": os.path.relpath(submission_path, _ROOT),
            "submission_sha256": _sha256_file(submission_path),
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": test_metrics,
        }, fh, indent=2)


def finalization_issues(manifest: dict) -> list[str]:
    """Pure checks that must hold before the one hidden-test evaluation."""
    issues = []
    conv = manifest.get("convergence") or {}
    official = conv.get("official") or {}
    eligible = conv.get("eligible_checkpoint") or {}
    submitted = manifest.get("submitted") or {}
    latest = manifest.get("latest_run") or {}
    hidden = manifest.get("hidden_test") or {}
    if not official.get("converged"):
        issues.append("the organizer convergence rule has not fired")
    if not eligible.get("determined"):
        issues.append("no official eligible checkpoint is determined")
    if submitted.get("source_node") != eligible.get("eligible_node"):
        issues.append(
            "canonical ensemble source node does not equal the official eligible "
            f"node ({submitted.get('source_node')} vs "
            f"{eligible.get('eligible_node')})")
    if submitted.get("official_eligible") is not True:
        issues.append("canonical ensemble is not stamped official_eligible=true")
    if not submitted.get("verified"):
        issues.append("canonical ensemble does not recompute exactly")
    if submitted.get("kind") != "ensemble" or not submitted.get("members"):
        issues.append("canonical submission is not a complete fixed ensemble")
    if latest.get("run_profile") != "competition":
        issues.append("latest run was not produced by the competition profile")
    if latest.get("manual_interventions") != 0:
        issues.append("competition run contains manual interventions")
    if not latest.get("llm_tokens_total"):
        issues.append("provider-level LLM token total is missing")
    if hidden.get("evaluated") or hidden.get("lock_present"):
        issues.append("hidden-test evaluation is already recorded")
    valid = (manifest.get("submission_artifacts") or {}).get("valid") or {}
    if not valid.get("available") or not valid.get("sha256"):
        issues.append("validated submission_valid.csv and its hash are missing")
    return issues


def check_finalization_readiness(root: str = _ROOT) -> dict:
    """Rebuild facts live and fail closed before hidden labels are available."""
    from agent import manifest as MF

    manifest_path = os.path.join(root, "results", "manifest.json")
    if not os.path.exists(manifest_path):
        sys.exit("FINALIZATION REFUSED: results/manifest.json is missing; run "
                 "python3 -m agent.manifest first")
    with open(manifest_path) as fh:
        recorded = json.load(fh)
    live = MF.build(run_tests=False)
    issues = finalization_issues(live)

    # Generated presentation files may legitimately change after the source
    # commit. Code/config/data changes may not: a dirty source tree means the
    # manifest's commit cannot identify what is being submitted.
    proc = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                          capture_output=True, text=True, timeout=20)
    dirty_source = []
    if proc.returncode != 0:
        issues.append("git status failed; source cleanliness is unknown")
    else:
        for line in proc.stdout.splitlines():
            path = line[3:].strip()
            from agent.provenance import is_generated_path
            if not is_generated_path(path):
                dirty_source.append(path)
        if dirty_source:
            issues.append("source worktree is dirty: " + ", ".join(dirty_source))

    # The disk manifest need not reproduce its own generating commit hash (that
    # is self-referential when tracked), but its decision-critical facts must
    # match a live rebuild.
    for path in (("submitted", "source_node"),
                 ("submitted", "reported", "primary"),
                 ("convergence", "eligible_checkpoint", "eligible_node"),
                 ("latest_run", "llm_tokens_total")):
        def get(doc):
            cur = doc
            for key in path:
                cur = (cur or {}).get(key)
            return cur
        if get(recorded) != get(live):
            issues.append("results/manifest.json is stale at " + ".".join(path))

    if issues:
        sys.exit("FINALIZATION REFUSED:\n" + "\n".join(f"  - {x}" for x in issues))
    return live


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Map scores to evenly spaced ranks in [0, 1].

    Different nodes produce scores on completely different scales (a logit from a
    pointwise model versus a BPR margin), so averaging raw values would silently
    let whichever model has the widest spread dominate. Ranks are scale-free and
    preserve each model's ordering, which is all the metric reads.
    """
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks / max(1, len(x) - 1)


def final_ensemble_members(root: str, split: str) -> tuple:
    """The SUBMITTED ensemble: every seed of ONE configuration, no selection.

    This is what logs/ensemble_results.json reports and what the deliverable
    quotes, so it is what a submission must be built from. The legacy
    top_k_distinct_nodes() path below does something materially different and
    contradicts two measured findings on this project -- see its docstring --
    which is why this is now the default source.
    """
    res_path = os.path.join(root, "logs", "ensemble_results.json")
    if not os.path.exists(res_path):
        return None, None
    with open(res_path) as fh:
        res = json.load(fh)
    seeds = res.get("seeds_used") or []
    mdir = os.path.join(root, res.get("members_dir") or
                        os.path.join("logs", "final_ensemble"))
    stacked, used = [], []
    for s in seeds:
        p = os.path.join(mdir, f"seed_{s:02d}", f"scores_{split}.npy")
        if os.path.exists(p):
            stacked.append(_rank_normalize(np.load(p)))
            used.append(s)
    if not stacked or len({len(x) for x in stacked}) != 1:
        return None, None
    # Refuse a partial rebuild: averaging a SUBSET of the recorded members is
    # exactly the selection this ensemble is designed not to do, and it would
    # silently produce a number that is not the one being reported.
    if len(used) != len(seeds):
        print(f"! final ensemble incomplete for split={split}: {len(used)}/"
              f"{len(seeds)} member arrays present. Refusing to average a "
              f"subset (that would be selection). Rebuild with: "
              f"python3 -m agent.final_ensemble --seeds {len(seeds)}")
        return None, None
    return np.mean(np.stack(stacked, axis=0), axis=0), res


def top_k_distinct_nodes(root: str, k: int) -> list[dict]:
    """LEGACY. Best k successful nodes with DISTINCT menu_choices, best first.

    Kept for inspection, but NOT the submission default, because it does two
    things this project measured and rejected:

      * it selects the top-k BY VALIDATION SCORE, which is the ensemble
        selection bias measured here at +0.00081 -- an optimistic estimate,
        not a better model;
      * it forces DISTINCT configurations, i.e. heterogeneous blending, which
        was measured and lost (gru4rec_seq + fm_numpy: genuinely decorrelated
        at 0.9338, but a 2.1 sigma quality gap cancelled the gain).

    It also reads logs/runs/, the SEARCH journal, so it would build a
    submission from whatever run happens to be on disk rather than from the
    reported result. Use --legacy-topk-ensemble to get it deliberately.
    """
    journal = os.path.join(root, "logs", "journal.jsonl")
    if not os.path.exists(journal):
        return []
    nodes = []
    with open(journal) as fh:
        for line in fh:
            if not line.strip():
                continue
            n = json.loads(line)
            if n.get("status") == "success" and n.get("metrics"):
                nodes.append(n)
    nodes.sort(key=lambda n: -n["metrics"]["primary"])
    picked, seen = [], set()
    for n in nodes:
        sig = json.dumps(n["menu_choices"], sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        picked.append(n)
        if len(picked) >= k:
            break
    return picked


def ensemble_scores(root: str, split: str, nodes: list[dict]) -> np.ndarray | None:
    """Rank-average the saved score arrays of the given nodes. No model calls."""
    stacked = []
    for n in nodes:
        p = os.path.join(root, "logs", "runs",
                         f"node_{n['iteration_id']:03d}", f"scores_{split}.npy")
        if not os.path.exists(p):
            continue
        arr = np.load(p)
        stacked.append(_rank_normalize(arr))
    if not stacked:
        return None
    if len({len(a) for a in stacked}) != 1:
        return None
    return np.mean(np.stack(stacked, axis=0), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--score", action="store_true",
                    help="also score locally (valid split only)")
    ap.add_argument("--final-test-eval", action="store_true",
                    help="the ONE final hidden-test evaluation; writes results/")
    ap.add_argument("--data_dir", default=os.path.join(KIT, "KuaiRand-Pure", "data"))
    ap.add_argument("--ensemble", action="store_true",
                    help="use the SUBMITTED ensemble (all seeds of the reported "
                         "configuration, from logs/final_ensemble/) instead of a "
                         "single node. This is the number the deliverable quotes.")
    ap.add_argument("--legacy-topk-ensemble", action="store_true",
                    help="DEPRECATED, biased. Rank-average the top-K nodes chosen "
                         "BY VALIDATION SCORE with distinct configs. Selecting on "
                         "validation was measured here at +0.00081 optimistic bias, "
                         "and heterogeneous blending was measured and lost. Produces "
                         "a number that is NOT the reported result.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="K for --ensemble (default: config/llm_config.json "
                         "ensemble_top_k, or 3)")
    ap.add_argument("--admin-override", action="store_true",
                    help="force a second --final-test-eval past the "
                         "results/final_evaluation.lock guard. Logged either way.")
    a = ap.parse_args()
    if a.final_test_eval:
        if not a.ensemble:
            sys.exit("FINALIZATION REFUSED: --final-test-eval requires --ensemble; "
                     "the canonical submitted result is the fixed ensemble")
        a.split = "test"
        check_final_eval_guard(a.admin_override)
        check_finalization_readiness()

    k = a.top_k
    if k is None:
        cfg_path = os.path.join(_ROOT, "config", "llm_config.json")
        k = 3
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                k = int(json.load(fh).get("ensemble_top_k", 3))

    best_meta_path = os.path.join(_ROOT, "logs", "best_metrics.json")
    if not os.path.exists(best_meta_path):
        sys.exit("no best node yet (logs/best_metrics.json missing) — run the agent first")
    with open(best_meta_path) as fh:
        best = json.load(fh)
    run_dir = os.path.join(_ROOT, "logs", "runs", f"node_{best['iteration_id']:03d}")
    single_path = os.path.join(run_dir, f"scores_{a.split}.npy")
    single_scores = np.load(single_path) if os.path.exists(single_path) else None

    ens_meta = None
    members = []
    if a.legacy_topk_ensemble:
        members = top_k_distinct_nodes(_ROOT, k)
        ens_scores = ensemble_scores(_ROOT, a.split, members) if len(members) > 1 else None
        ens_desc = f"LEGACY top-{k} distinct nodes {[m['iteration_id'] for m in members]}"
        print("! --legacy-topk-ensemble selects members BY VALIDATION SCORE "
              "(+0.00081 measured bias) and blends distinct configs (measured, "
              "lost). This is NOT the reported result.")
    else:
        ens_scores, ens_meta = final_ensemble_members(_ROOT, a.split)
        ens_desc = (f"submitted ensemble, k={ens_meta['k']} seeds "
                    f"{ens_meta['seeds_used']} of node {ens_meta.get('source_node')} "
                    f"(no selection)" if ens_meta else "")

    scores = single_scores
    which = f"best single node {best['iteration_id']}"
    if a.ensemble:
        if ens_scores is None:
            sys.exit("ensemble unavailable or incomplete; refusing to fall back "
                     "to a different single-node artifact. Rebuild with: "
                     "python3 -m agent.final_ensemble --seeds 16")
        else:
            scores = ens_scores
            which = ens_desc
    if scores is None:
        sys.exit(f"scores for best node {best['iteration_id']} are not stored in "
                 f"{run_dir}; use --ensemble for an ensemble checkpoint")

    out = a.out or os.path.join(_ROOT, f"submission_{a.split}.csv")
    print(f"loading official split rows ({a.data_dir}) ...")
    rows = load(a.data_dir)[a.split]
    if len(rows) != len(scores):
        sys.exit(f"row count mismatch: split has {len(rows)}, scores {len(scores)}")
    write_submission(out, rows, scores)
    # validate with the starter kit's own checker
    read_submission(out, rows)
    print(f"✓ wrote and validated {out}: {len(rows):,d} rows ({which})")

    # ---- comparison table: best-single vs best-ensemble ----
    def _score(arr):
        r = evaluate([x[1] for x in rows], [x[6] for x in rows], arr)
        # float() required — see the note in train_lib.run: numpy>=2 returns
        # np.float32 here, which json.dump cannot serialize.
        return {"GAUC": float(r["GAUC"]), "nDCG@5": float(r["nDCG@5"]),
                "primary": float(r["primary"])}

    comparison = None
    if a.split == "valid" and (a.score or a.ensemble):
        base = BASELINE_VALID
        comparison = {}
        if single_scores is not None:
            comparison["single"] = _score(single_scores)
        if ens_scores is not None:
            comparison["ensemble"] = _score(ens_scores)
            # `members` only exists on the legacy top-k path; the submitted
            # ensemble identifies its members by SEED, not by journal node.
            comparison["ensemble_members"] = (
                [m["iteration_id"] for m in members] if a.legacy_topk_ensemble
                else (ens_meta or {}).get("seeds_used"))
        print(f"\n  {'variant':<26} {'GAUC':>8} {'nDCG@5':>8} {'primary':>9} "
              f"{'Δ vs base':>10} {'Δ in σ':>8}")
        for name in ("single", "ensemble"):
            if name not in comparison:
                continue
            m = comparison[name]
            d = m["primary"] - base["primary"]
            label = (name if name == "single"
                     else (f"ensemble (top-{len(members)})"
                           if a.legacy_topk_ensemble
                           else f"ensemble (k={(ens_meta or {}).get('k')})"))
            print(f"  {label:<26} {m['GAUC']:>8.4f} {m['nDCG@5']:>8.4f} "
                  f"{m['primary']:>9.4f} {d:>+10.4f} "
                  f"{d / BASELINE_SEED_STD:>+7.1f}σ")
        print(f"  (σ = {BASELINE_SEED_STD}, the official baseline's own 5-seed std)")

    if a.final_test_eval:
        base = BASELINE_TEST
        r = _score(scores)
        results = {
            "submission_source": which,
            "best_node": best["iteration_id"],
            "menu_choices": best["menu_choices"],
            "ensemble_used": bool(a.ensemble and ens_scores is not None),
            "ensemble_members": (
                [m["iteration_id"] for m in members]
                if (a.legacy_topk_ensemble and a.ensemble
                    and ens_scores is not None)
                else (ens_meta or {}).get("seeds_used")
                if (a.ensemble and ens_scores is not None) else None),
            "valid": ({k: ens_meta[k]
                       for k in ("GAUC", "nDCG@5", "primary")}
                      if (a.ensemble and ens_meta)
                      else best["valid_metrics"]),
            "test": r,
            "baseline_test": base,
            "delta_test": {m: round(r[m] - base[m], 4) for m in
                           ("GAUC", "nDCG@5", "primary")},
            "delta_test_primary_in_baseline_seed_sigmas":
                round((r["primary"] - base["primary"]) / BASELINE_SEED_STD, 2),
        }
        os.makedirs(os.path.join(_ROOT, "results"), exist_ok=True)
        with open(os.path.join(_ROOT, "results", "final_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        write_final_eval_lock(out, r)
        print("\n=== FINAL (one-time) HIDDEN-TEST EVALUATION ===")
        print(json.dumps(results, indent=2))
        print(f"\nwrote {DEFAULT_LOCK_PATH} -- a second run now needs --admin-override")


if __name__ == "__main__":
    main()
