"""Generate and validate a submission CSV from saved score artifacts.

The best node's training run already wrote row_id-aligned scores_{valid,test}.npy
(the executor validated shape and finiteness), so a submission is a deterministic
re-format, checked with the starter kit's own read_submission.

Usage:
  python3 -m agent.make_submission --verified-ensemble
  python3 -m agent.make_submission --verified-ensemble --split valid --score
  python3 -m agent.make_submission --verified-ensemble --final-test-eval

--final-test-eval is the single hidden-test evaluation of the whole run: it scores
the test submission once with the official evaluate.py and writes
results/final_results.json. Do not run it during development.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(_ROOT, "kuairand-starter-kit")
sys.path.insert(0, KIT)

from data import load  # noqa: E402
from evaluate import evaluate  # noqa: E402
from submit import HEADER, read_submission  # noqa: E402

BASELINE_VALID = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
BASELINE_TEST = {"GAUC": 0.6610, "nDCG@5": 0.5282, "primary": 0.5946}
BASELINE_SEED_STD = 0.0008
OFFICIAL_DATA_DIR = Path(KIT) / "KuaiRand-Pure" / "data"
OFFICIAL_DATA_SHA256 = {
    "video_features_basic_pure.csv":
        "a6f7ee02684c5777422306cdc416e170302288aa89aca9dfea995edbd625bcc2",
    "log_standard_4_08_to_4_21_pure.csv":
        "5bb6eb0b3d9f47e5436cb5dc82ee1899b845ebf9750a5560b801e929e18bd41c",
    "log_standard_4_22_to_5_08_pure.csv":
        "429e3b948828942e572f2c3a5be5a25799ffe75591d22d18cf417b9b534d31fd",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_official_data() -> dict[str, str]:
    actual = {
        name: _sha256_file(OFFICIAL_DATA_DIR / name)
        for name in OFFICIAL_DATA_SHA256
    }
    if actual != OFFICIAL_DATA_SHA256:
        raise ValueError(
            "official KuaiRand source-file hashes do not match the pinned dataset")
    return actual


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _write_submission_precise(path: Path, rows, scores) -> None:
    """Atomically write official-format scores with float64 round-trip precision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for index, (row, score) in enumerate(zip(rows, scores)):
                writer.writerow([
                    index, row[1], row[2], format(float(score), ".17g")])
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def verified_ensemble_scores(root: str, split: str) -> tuple[np.ndarray, dict, Path]:
    """Load the atomically published verified bundle and recheck its hashes."""
    if split not in ("valid", "test"):
        raise ValueError(f"unknown split: {split}")
    verified_root = Path(root) / "results" / "verified_ensemble"
    latest_path = verified_root / "latest.json"
    with latest_path.open("r", encoding="utf-8") as handle:
        latest = json.load(handle)
    bundle = (verified_root / latest["bundle"]).resolve()
    bundles_root = (verified_root / "bundles").resolve()
    if os.path.commonpath([str(bundle), str(bundles_root)]) != str(bundles_root):
        raise ValueError("verified ensemble pointer escapes bundles directory")
    summary_path = bundle / "summary.json"
    if _sha256_file(summary_path) != latest.get("summary_sha256"):
        raise ValueError("verified ensemble summary hash mismatch")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("seeds") != list(range(5)):
        raise ValueError("verified ensemble is not the frozen five-seed set 0..4")
    from agent.executor import (_hash_protected, _protected_paths,
                                _runtime_fingerprint)
    if summary.get("runtime_fingerprint") != _runtime_fingerprint():
        raise ValueError("verified ensemble runtime fingerprint mismatch")
    current_protected = _hash_protected(_protected_paths())
    if summary.get("protected_sha256") != current_protected:
        raise ValueError(
            "verified ensemble is stale for the current runtime/cache row order")
    runner_path = Path(__file__).with_name("verified_ensemble.py")
    if summary.get("ensemble_runner_sha256") != _sha256_file(runner_path):
        raise ValueError("verified ensemble runner hash mismatch")
    score_name = f"scores_{split}.npy"
    score_path = bundle / score_name
    expected = summary.get("artifacts_sha256", {}).get(score_name)
    if expected != _sha256_file(score_path):
        raise ValueError(f"verified ensemble {score_name} hash mismatch")
    scores = np.load(score_path, allow_pickle=False)
    expected_rows = 124_909 if split == "valid" else 170_588
    if scores.shape != (expected_rows,) or not np.all(np.isfinite(scores)):
        raise ValueError(
            f"verified ensemble {score_name} failed shape/finiteness checks")
    return np.asarray(scores, dtype=np.float64), summary, bundle


def _claim_final_test_evaluation(root: str, source: str) -> Path:
    """Atomically enforce the promised one-time hidden-test evaluation."""
    result_dir = Path(root) / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "final_results.json"
    lock_path = result_dir / "final_test_eval.lock"
    if result_path.exists():
        raise RuntimeError(
            f"final test was already evaluated ({result_path}); refusing to rerun")
    payload = json.dumps({
        "claimed_unix_time": time.time(),
        "submission_source": source,
        "policy": "one-time hidden-test evaluation; retained even after failure",
    }, indent=2).encode("utf-8")
    try:
        descriptor = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(
            f"final test evaluation was already claimed ({lock_path}); "
            "refusing to access outcomes again") from error
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return lock_path


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


def top_k_distinct_nodes(root: str, k: int) -> list[dict]:
    """Best k successful nodes with DISTINCT menu_choices, best first.

    Distinctness matters: averaging three near-identical configurations adds no
    diversity and therefore cancels no noise.
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
        p = os.path.join(root, "logs", "nodes",
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
    source = ap.add_mutually_exclusive_group()
    source.add_argument(
        "--ensemble", action="store_true",
        help="rank-average top-K successful search nodes")
    source.add_argument(
        "--verified-ensemble", action="store_true",
        help="use the hash-checked bundle selected by "
             "results/verified_ensemble/latest.json")
    ap.add_argument("--top-k", type=int, default=None,
                    help="K for --ensemble (default: config/llm_config.json "
                         "ensemble_top_k, or 3)")
    a = ap.parse_args()
    if a.final_test_eval and not a.verified_ensemble:
        ap.error("--final-test-eval requires --verified-ensemble so the one-time "
                 "outcome access cannot be spent on an unverified search node")
    if a.final_test_eval:
        a.split = "test"

    k = a.top_k
    if k is None:
        cfg_path = os.path.join(_ROOT, "config", "llm_config.json")
        k = 3
        if os.path.exists(cfg_path):
            with open(cfg_path) as fh:
                k = int(json.load(fh).get("ensemble_top_k", 3))

    best = None
    members = []
    ens_scores = None
    single_scores = None
    verified_summary = None
    if a.verified_ensemble:
        try:
            scores, verified_summary, bundle = verified_ensemble_scores(
                _ROOT, a.split)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            sys.exit(f"verified ensemble unavailable or invalid: {error}")
        which = f"verified ensemble bundle {bundle.name}"
    else:
        best_meta_path = os.path.join(_ROOT, "logs", "best_metrics.json")
        if not os.path.exists(best_meta_path):
            sys.exit("no best node yet (logs/best_metrics.json missing) — run "
                     "the agent or pass --verified-ensemble")
        with open(best_meta_path) as fh:
            best = json.load(fh)
        run_dir = os.path.join(
            _ROOT, "logs", "nodes", f"node_{best['iteration_id']:03d}")
        single_scores = np.load(
            os.path.join(run_dir, f"scores_{a.split}.npy"))
        members = top_k_distinct_nodes(_ROOT, k)
        ens_scores = (ensemble_scores(_ROOT, a.split, members)
                      if len(members) > 1 else None)
        scores = single_scores
        which = f"best single node {best['iteration_id']}"
        if a.ensemble:
            if ens_scores is None:
                print(f"! ensemble unavailable ({len(members)} distinct successful "
                      "node(s) with usable score arrays) — falling back to the "
                      "single best node")
            else:
                scores = ens_scores
                which = (f"rank-averaged ensemble of nodes "
                         f"{[m['iteration_id'] for m in members]}")

    out = a.out or os.path.join(_ROOT, f"submission_{a.split}.csv")
    out_path = Path(out).resolve()
    if out_path.suffix.lower() != ".csv":
        sys.exit("submission output must use a .csv filename")
    verified_artifacts = (
        Path(_ROOT) / "results" / "verified_ensemble").resolve()
    try:
        overwrites_verified = os.path.commonpath(
            [str(out_path), str(verified_artifacts)]) == str(verified_artifacts)
    except ValueError:
        overwrites_verified = False
    if overwrites_verified:
        sys.exit("submission output cannot overwrite verified ensemble artifacts")
    # Submission construction uses the feature-only cache, so writing a blind
    # test CSV does not load raw test outcomes. Raw rows are opened only below
    # when --final-test-eval is explicitly requested.
    from runtime import train_lib
    cached_splits, _ = train_lib.load_cache()
    cached = cached_splits[a.split]
    rows = [(None, str(user), str(video))
            for user, video in zip(cached["user_raw"], cached["video_raw"])]
    if len(rows) != len(scores):
        sys.exit(f"row count mismatch: split has {len(rows)}, scores {len(scores)}")
    _write_submission_precise(out_path, rows, scores)
    # validate with the starter kit's own checker
    submitted_scores = np.asarray(
        read_submission(out_path, rows), dtype=np.float64)
    print(f"✓ wrote and validated {out}: {len(rows):,d} rows ({which})")

    # ---- comparison table: best-single vs best-ensemble ----
    def _score(arr, users, labels):
        r = evaluate(users, labels, arr)
        # float() required — see the note in train_lib.run: numpy>=2 returns
        # np.float32 here, which json.dump cannot serialize.
        return {"GAUC": float(r["GAUC"]), "nDCG@5": float(r["nDCG@5"]),
                "primary": float(r["primary"])}

    comparison = None
    if a.split == "valid" and (a.score or a.ensemble):
        base = BASELINE_VALID
        if a.verified_ensemble:
            comparison = {"verified_ensemble": _score(
                submitted_scores, cached["user_raw"], cached["long_view"])}
        else:
            comparison = {"single": _score(
                single_scores, cached["user_raw"], cached["long_view"])}
        if ens_scores is not None and not a.verified_ensemble:
            comparison["ensemble"] = _score(
                ens_scores, cached["user_raw"], cached["long_view"])
            comparison["ensemble_members"] = [m["iteration_id"] for m in members]
        print(f"\n  {'variant':<26} {'GAUC':>8} {'nDCG@5':>8} {'primary':>9} "
              f"{'Δ vs base':>10} {'Δ in σ':>8}")
        for name in ("verified_ensemble", "single", "ensemble"):
            if name not in comparison:
                continue
            m = comparison[name]
            d = m["primary"] - base["primary"]
            label = ("verified ensemble" if name == "verified_ensemble"
                     else (name if name == "single"
                           else f"ensemble (top-{len(members)})"))
            print(f"  {label:<26} {m['GAUC']:>8.4f} {m['nDCG@5']:>8.4f} "
                  f"{m['primary']:>9.4f} {d:>+10.4f} "
                  f"{d / BASELINE_SEED_STD:>+7.1f}σ")
        print(f"  (σ = {BASELINE_SEED_STD}, the official baseline's own 5-seed std)")

    if a.final_test_eval:
        try:
            official_data_sha256 = _verify_official_data()
            _claim_final_test_evaluation(_ROOT, which)
        except (OSError, RuntimeError, ValueError) as error:
            sys.exit(str(error))
        print(f"loading raw test outcomes for the explicit one-time evaluation "
              f"({OFFICIAL_DATA_DIR}) ...")
        raw_rows = load(str(OFFICIAL_DATA_DIR))["test"]
        # Revalidate user/video alignment against the official raw loader before
        # this one authorized outcome access, and score the actual six-digit CSV
        # values rather than the higher-precision in-memory array.
        submitted_scores = np.asarray(
            read_submission(out_path, raw_rows), dtype=np.float64)
        base = BASELINE_TEST
        r = _score(submitted_scores, [x[1] for x in raw_rows],
                   [x[6] for x in raw_rows])
        results = {
            "submission_source": which,
            "best_node": best["iteration_id"] if best else None,
            "menu_choices": (verified_summary["configuration"]
                             if verified_summary else best["menu_choices"]),
            "ensemble_used": bool(
                a.verified_ensemble or (a.ensemble and ens_scores is not None)),
            "ensemble_members": (
                verified_summary["seeds"] if verified_summary else
                ([m["iteration_id"] for m in members]
                 if (a.ensemble and ens_scores is not None) else None)),
            "valid": (verified_summary["ensemble_valid_metrics"]
                      if verified_summary else best["valid_metrics"]),
            "test": r,
            "baseline_test": base,
            "delta_test": {m: round(r[m] - base[m], 4) for m in
                           ("GAUC", "nDCG@5", "primary")},
            "delta_test_primary_in_baseline_seed_sigmas":
                round((r["primary"] - base["primary"]) / BASELINE_SEED_STD, 2),
            "submission_csv_sha256": _sha256_file(out_path),
            "official_data_sha256": official_data_sha256,
            "submission_score_format": ".17g float64 round-trip",
        }
        _atomic_write_json(Path(_ROOT) / "results" / "final_results.json",
                           results)
        print("\n=== FINAL (one-time) HIDDEN-TEST EVALUATION ===")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
