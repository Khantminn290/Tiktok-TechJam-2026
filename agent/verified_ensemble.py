"""Reproduce the adopted fixed-seed ensemble with parent-side verification.

This is deliberately not another search loop. It runs one frozen configuration
for seeds ``0..4``, verifies every member through :func:`run_solution`, and
rank-averages the resulting score arrays.  Validation labels are loaded only by
the trusted parent evaluator; hidden-test outcomes are never extracted into
runtime arrays or scored.

Usage::

    python -m agent.verified_ensemble
    python -m agent.verified_ensemble --seeds 5 --fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np

from .executor import (_hash_protected, _make_validation_evaluator,
                       _protected_paths, _runtime_fingerprint, run_solution)
from .menu import Menu


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "verified_ensemble"
EXPECTED_ROWS = {"valid": 124_909, "test": 170_588}
FINAL_SEEDS = list(range(5))
# Frozen after the independent search.  Changing this constant creates a new
# experiment and intentionally invalidates every reusable member artifact.
MENU_CHOICES = {
    "loss": "bpr_pairwise",
    "score_prior": "batch_repeat_fatigue",
    "user_history": "recency_weighted_pool",
    "multitask": "none",
    "model": "fm_numpy",
    "temporal": "hour_plus_dow",
    "training": "lower_lr_longer",
    "data_extras": "none",
}

# A small, fixed adapter keeps all actual training inside the shared, tracked
# runtime.  run_solution writes this source into each seed directory and records
# its digest in verification.json.
SOLUTION_CODE = '''\
import argparse
import json

import train_lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--menu-choices", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    choices = json.loads(args.menu_choices)
    metrics = train_lib.run(choices, args.output_dir, seed=args.seed)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
'''


class EnsembleError(RuntimeError):
    """A member failed execution or could not satisfy the audit contract."""


def _pin_canonical_environment() -> None:
    """Make the publishable runner independent of caller path overrides."""
    canonical = {
        "KUAIRAND_KIT": ROOT / "kuairand-starter-kit",
        "KUAIRAND_DATA": ROOT / "kuairand-starter-kit" / "KuaiRand-Pure" / "data",
        "KUAIRAND_CACHE": ROOT / "runtime" / "cache",
    }
    for key, path in canonical.items():
        os.environ[key] = str(path.resolve())

    # Fail rather than silently reuse a module imported under an earlier,
    # noncanonical environment in programmatic/in-notebook use.
    expected_train = {
        "KIT_DIR": canonical["KUAIRAND_KIT"],
        "DATA_DIR": canonical["KUAIRAND_DATA"],
        "CACHE_DIR": canonical["KUAIRAND_CACHE"],
    }
    for module_name in ("train_lib", "runtime.train_lib"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attribute, expected in expected_train.items():
            actual = Path(getattr(module, attribute, "")).resolve()
            if actual != expected.resolve():
                raise EnsembleError(
                    f"{module_name}.{attribute} was imported from noncanonical "
                    f"path {actual}; restart before publishing")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _metrics(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        raise ValueError("metrics must be a JSON object")
    result = {}
    for key in ("GAUC", "nDCG@5", "primary"):
        number = value.get(key)
        if not isinstance(number, (int, float)) or not np.isfinite(number):
            raise ValueError(f"metric {key!r} is missing or non-finite")
        result[key] = float(number)
    return result


def _metrics_match(left: dict, right: dict, tolerance: float = 1e-8) -> bool:
    return all(abs(float(left[key]) - float(right[key])) <= tolerance
               for key in ("GAUC", "nDCG@5", "primary"))


def _load_scores(path: Path, split: str) -> np.ndarray:
    scores = np.load(path, allow_pickle=False)
    expected = EXPECTED_ROWS[split]
    if scores.shape != (expected,):
        raise ValueError(
            f"{path.name} has shape {scores.shape}, expected ({expected},)")
    if not np.all(np.isfinite(scores)):
        raise ValueError(f"{path.name} contains NaN or infinity")
    return np.asarray(scores, dtype=np.float64)


def _rank_normalize(scores: np.ndarray, user_ids: np.ndarray,
                    time_ms: np.ndarray) -> np.ndarray:
    """Return metric-aligned per-user ranks with an explicit temporal tie rule.

    Later impressions rank above earlier ones only when a member's scores are
    exactly equal. Exact score-and-time ties receive neutral midranks. This
    avoids accidental row-index leakage while keeping each seed on the scale
    that the official within-user metrics actually consume.
    """
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    users_raw = np.asarray(user_ids).reshape(-1)
    times = np.asarray(time_ms, dtype=np.int64).reshape(-1)
    if not (len(values) == len(users_raw) == len(times)):
        raise ValueError("rank context arrays are not row-aligned")
    _, users = np.unique(users_raw, return_inverse=True)
    users = users.astype(np.int64, copy=False)
    ranks = np.empty(len(values), dtype=np.float64)
    if not len(values):
        return ranks
    order = np.lexsort((times, values, users))
    sorted_users = users[order]
    sorted_scores = values[order]
    sorted_times = times[order]
    user_counts = np.bincount(users).astype(np.int64)
    user_starts = np.cumsum(user_counts) - user_counts

    # Midrank only rows whose user, score, and timestamp are all identical.
    changes = np.r_[
        True,
        (sorted_users[1:] != sorted_users[:-1])
        | (sorted_scores[1:] != sorted_scores[:-1])
        | (sorted_times[1:] != sorted_times[:-1]),
    ]
    tie_group = np.cumsum(changes) - 1
    tie_counts = np.bincount(tie_group).astype(np.int64)
    tie_starts = np.cumsum(tie_counts) - tie_counts
    absolute_midrank = tie_starts + (tie_counts - 1) / 2.0
    within_user = absolute_midrank[tie_group] - user_starts[sorted_users]
    denominator = np.maximum(user_counts[sorted_users] - 1, 1)
    ranks[order] = within_user / denominator
    return ranks


def _expected_digests() -> tuple[str, str]:
    return (hashlib.sha256(SOLUTION_CODE.encode("utf-8")).hexdigest(),
            _sha256_json(MENU_CHOICES))


def _verified_member(seed_dir: Path, seed: int, validation_evaluator):
    """Return a reusable member, or ``None`` when any audit check fails."""
    required = [
        seed_dir / "solution.py",
        seed_dir / "verification.json",
        seed_dir / "metrics.json",
        seed_dir / "scores_valid.npy",
        seed_dir / "scores_test.npy",
    ]
    if not all(path.is_file() for path in required):
        return None

    try:
        verification = _read_json(seed_dir / "verification.json")
        solution_digest, choices_digest = _expected_digests()
        if verification.get("status") != "success":
            return None
        if verification.get("seed") != seed:
            return None
        if verification.get("solution_sha256") != solution_digest:
            return None
        if verification.get("solution_after_sha256") != solution_digest:
            return None
        if verification.get("menu_choices_sha256") != choices_digest:
            return None
        if verification.get("runtime_fingerprint") != _runtime_fingerprint():
            return None
        if verification.get("protected_changed") != []:
            return None
        if verification.get("protected_before") != verification.get("protected_after"):
            return None
        # A member is also stale when the trusted evaluator, training runtime,
        # menu, or validation cache has changed *since* that member ran.
        protected_after = verification.get("protected_after")
        if not isinstance(protected_after, dict):
            return None
        protected_now = _hash_protected(_protected_paths())
        if protected_after != protected_now:
            return None

        artifact_hashes = verification.get("artifact_sha256")
        if not isinstance(artifact_hashes, dict):
            return None
        for name in ("solution.py", "scores_valid.npy", "scores_test.npy"):
            if artifact_hashes.get(name) != _sha256_file(seed_dir / name):
                return None

        valid_scores = _load_scores(seed_dir / "scores_valid.npy", "valid")
        test_scores = _load_scores(seed_dir / "scores_test.npy", "test")
        recomputed = _metrics(validation_evaluator(valid_scores))
        recorded = _metrics(_read_json(seed_dir / "metrics.json"))
        verified_metrics = _metrics(
            verification.get("parent_recomputed_metrics"))
        if not (_metrics_match(recomputed, recorded)
                and _metrics_match(recomputed, verified_metrics)):
            return None
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    return {
        "seed": seed,
        "metrics": recomputed,
        "valid_scores": valid_scores,
        "test_scores": test_scores,
        "verification_sha256": _sha256_file(seed_dir / "verification.json"),
    }


def _promote(staging: Path, destination: Path) -> None:
    """Replace one member directory only after its staged run is verified."""
    backup = None
    if destination.exists():
        backup = destination.parent / (
            f".{destination.name}.previous-{uuid.uuid4().hex}")
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _recover_previous(seed_dir: Path) -> None:
    """Restore the last member if a hard stop interrupted promotion."""
    if seed_dir.exists():
        return
    backups = sorted(
        seed_dir.parent.glob(f".{seed_dir.name}.previous-*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if backups:
        os.replace(backups[0], seed_dir)


def _run_member(seed: int, validation_evaluator) -> dict:
    seed_name = f"seed_{seed:02d}"
    seed_dir = RESULTS_DIR / seed_name
    staging = RESULTS_DIR / f".{seed_name}.staging-{uuid.uuid4().hex}"
    result = run_solution(
        SOLUTION_CODE,
        str(staging / "solution.py"),
        MENU_CHOICES,
        str(staging),
        seed=seed,
        validation_evaluator=validation_evaluator,
    )
    if not result.ok:
        failed = RESULTS_DIR / (
            f".{seed_name}.failed-{int(time.time())}-{uuid.uuid4().hex[:8]}")
        if staging.exists():
            os.replace(staging, failed)
        raise EnsembleError(
            f"seed {seed} failed; audit retained at {failed}:\n"
            f"{result.error_trace or 'unknown executor failure'}")

    checked = _verified_member(staging, seed, validation_evaluator)
    if checked is None:
        failed = RESULTS_DIR / (
            f".{seed_name}.failed-{int(time.time())}-{uuid.uuid4().hex[:8]}")
        os.replace(staging, failed)
        raise EnsembleError(
            f"seed {seed} returned success but failed post-run verification; "
            f"audit retained at {failed}")
    _promote(staging, seed_dir)
    checked["wall_clock_seconds"] = float(result.wall_clock_seconds)
    return checked


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float64))
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def run_ensemble(seed_count: int = 5, fresh: bool = False) -> dict:
    if list(range(seed_count)) != FINAL_SEEDS:
        raise ValueError(
            "the publishable ensemble is frozen to --seeds 5 (seeds 0..4); "
            "use a separate research script for exploratory seed counts")

    _pin_canonical_environment()
    # Hash the official raw sources before loading/reusing a derived cache.
    from .make_submission import _verify_official_data
    _verify_official_data()

    # This catches menu drift before spending any compute.
    Menu(str(ROOT / "config" / "modification_menu.json")).validate_choices(
        MENU_CHOICES)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    validation_evaluator = _make_validation_evaluator()
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime import train_lib
    splits, _ = train_lib.load_cache()

    members = []
    for seed in range(seed_count):
        seed_dir = RESULTS_DIR / f"seed_{seed:02d}"
        _recover_previous(seed_dir)
        member = None if fresh else _verified_member(
            seed_dir, seed, validation_evaluator)
        reused = member is not None
        if member is None:
            print(f"[seed {seed:02d}] running verified member", flush=True)
            member = _run_member(seed, validation_evaluator)
        else:
            print(f"[seed {seed:02d}] reusing verified member", flush=True)
        member["reused"] = reused
        members.append(member)

    # Average per-user ranks, not raw logits: model score scales vary across
    # seeds while both official metrics consume within-user ordering only.
    ensemble_valid = np.mean(
        np.stack([
            _rank_normalize(m["valid_scores"], splits["valid"]["user_raw"],
                            splits["valid"]["time_ms"])
            for m in members
        ]), axis=0)
    ensemble_test = np.mean(
        np.stack([
            _rank_normalize(m["test_scores"], splits["test"]["user_raw"],
                            splits["test"]["time_ms"])
            for m in members
        ]), axis=0)
    ensemble_metrics = _metrics(validation_evaluator(ensemble_valid))

    # Publish one immutable bundle, then atomically advance a tiny pointer.
    # Consumers following latest.json can never observe mixed-version arrays.
    bundle_id = time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:12]
    bundles_dir = RESULTS_DIR / "bundles"
    bundles_dir.mkdir(parents=True, exist_ok=True)
    staging_bundle = RESULTS_DIR / f".bundle.staging-{uuid.uuid4().hex}"
    staging_bundle.mkdir()
    valid_path = staging_bundle / "scores_valid.npy"
    test_path = staging_bundle / "scores_test.npy"
    metrics_path = staging_bundle / "metrics.json"
    summary_path = staging_bundle / "summary.json"
    _atomic_save_array(valid_path, ensemble_valid)
    _atomic_save_array(test_path, ensemble_test)
    _atomic_write_json(metrics_path, ensemble_metrics)

    summary = {
        "schema_version": 1,
        "configuration": MENU_CHOICES,
        "seeds": list(range(seed_count)),
        "method": ("mean of per-user ordinal ranks; exact score ties prefer "
                   "later time_ms; exact score/time ties use midranks"),
        "ensemble_runner_sha256": _sha256_file(Path(__file__)),
        "runtime_fingerprint": _runtime_fingerprint(),
        # Consumers must bind the score arrays to the exact evaluator, runtime,
        # menu, and row-aligned split caches that produced them.
        "protected_sha256": _hash_protected(_protected_paths()),
        "members": [
            {
                "seed": m["seed"],
                "reused": m["reused"],
                "valid_metrics": m["metrics"],
                "verification_sha256": m["verification_sha256"],
                **({"wall_clock_seconds": m["wall_clock_seconds"]}
                   if "wall_clock_seconds" in m else {}),
            }
            for m in members
        ],
        "ensemble_valid_metrics": ensemble_metrics,
        "hidden_test_evaluated": False,
        "artifacts_sha256": {
            "scores_valid.npy": _sha256_file(valid_path),
            "scores_test.npy": _sha256_file(test_path),
            "metrics.json": _sha256_file(metrics_path),
        },
    }
    _atomic_write_json(summary_path, summary)
    bundle_dir = bundles_dir / bundle_id
    os.replace(staging_bundle, bundle_dir)
    latest = {
        "schema_version": 1,
        "bundle": f"bundles/{bundle_id}",
        "summary_sha256": _sha256_file(bundle_dir / "summary.json"),
    }
    _atomic_write_json(RESULTS_DIR / "latest.json", latest)
    print(
        "verified ensemble: "
        f"primary={ensemble_metrics['primary']:.6f} "
        f"GAUC={ensemble_metrics['GAUC']:.6f} "
        f"nDCG@5={ensemble_metrics['nDCG@5']:.6f} "
        f"bundle={latest['bundle']}",
        flush=True,
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and verify the frozen fixed-seed rank ensemble.")
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="frozen final seed count; must be 5 (seeds 0 through 4)")
    parser.add_argument(
        "--fresh", action="store_true",
        help="rerun every seed instead of reusing matching verified artifacts")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        run_ensemble(seed_count=args.seeds, fresh=args.fresh)
    except (EnsembleError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"verified ensemble failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
