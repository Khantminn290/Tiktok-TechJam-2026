"""Execution wrapper — the robustness layer.

Writes generated code to disk, runs it in a subprocess with a timeout, captures
stdout/stderr/exit code, parses metrics.json on success and returns a readable
error trace on any failure. The loop never crashes on a bad iteration: every
failure mode (syntax error, exception, timeout, malformed metrics, NaN scores)
becomes an error result that the next debug action feeds back to the LLM.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
KIT_DIR = os.path.join(ROOT, "kuairand-starter-kit")
RUNTIME_DIR = os.path.join(ROOT, "runtime")
CHILD_GUARD = os.path.join(_HERE, "child_guard.py")


class ExecResult:
    def __init__(self, ok: bool, metrics=None, error_trace=None,
                 wall_clock_seconds=0.0, run_dir=None, metric_audit=None,
                 verification=None):
        self.ok = ok
        self.metrics = metrics
        self.error_trace = error_trace
        self.wall_clock_seconds = wall_clock_seconds
        self.run_dir = run_dir
        self.metric_audit = metric_audit
        self.verification = verification or {}


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _safe_sha256_file(path: str) -> str | None:
    try:
        return _sha256_file(path)
    except OSError:
        return None


def _sha256_json(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_fingerprint() -> dict:
    """Runtime facts that can change deterministic NumPy model artifacts."""
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
    }


def _protected_paths() -> dict[str, str]:
    """Files that define scoring, training, and prediction row alignment."""
    cache_dir = os.environ.get(
        "KUAIRAND_CACHE", os.path.join(RUNTIME_DIR, "cache"))
    return {
        "official_data_loader": os.path.join(KIT_DIR, "data.py"),
        "official_evaluate": os.path.join(KIT_DIR, "evaluate.py"),
        "official_submit": os.path.join(KIT_DIR, "submit.py"),
        "executor": os.path.join(_HERE, "executor.py"),
        "child_guard": CHILD_GUARD,
        "ensemble_runner": os.path.join(_HERE, "verified_ensemble.py"),
        "submission_writer": os.path.join(_HERE, "make_submission.py"),
        "train_lib": os.path.join(RUNTIME_DIR, "train_lib.py"),
        "menu": os.path.join(ROOT, "config", "modification_menu.json"),
        "train_cache": os.path.join(cache_dir, "train.npz"),
        "validation_cache": os.path.join(cache_dir, "valid.npz"),
        "test_cache": os.path.join(cache_dir, "test.npz"),
        "cache_meta": os.path.join(cache_dir, "meta.json"),
        "cache_vocabs": os.path.join(cache_dir, "vocabs.json"),
        "cache_schema": os.path.join(cache_dir, "cache_schema.json"),
    }


def _hash_protected(paths: dict[str, str]) -> dict[str, str | None]:
    return {name: _safe_sha256_file(path) for name, path in paths.items()}


def _make_validation_evaluator():
    """Capture official validation labels and evaluator in the trusted parent."""
    if RUNTIME_DIR not in sys.path:
        sys.path.insert(0, RUNTIME_DIR)
    import train_lib
    users, labels = train_lib.load_validation_targets()
    official_evaluate = train_lib.evaluate

    def score(scores):
        result = official_evaluate(users, labels, np.asarray(scores))
        return {k: float(result[k]) for k in ("GAUC", "nDCG@5", "primary")}
    return score


def _write_verification(run_dir: str, verification: dict) -> None:
    with open(os.path.join(run_dir, "verification.json"), "w", encoding="utf-8") as fh:
        json.dump(verification, fh, indent=2, ensure_ascii=False)


def _env(run_dir: str | None = None) -> dict:
    env = dict(os.environ)
    # Generated training code never needs the parent agent's provider or service
    # credentials. Keep them out of the child even when the host has them set.
    secret_markers = ("API_KEY", "ACCESS_TOKEN", "AUTH_TOKEN", "SECRET", "PASSWORD")
    for key in list(env):
        if any(marker in key.upper() for marker in secret_markers):
            env.pop(key, None)
    env["PYTHONPATH"] = os.pathsep.join(
        [RUNTIME_DIR, KIT_DIR, env.get("PYTHONPATH", "")])
    env.setdefault("KUAIRAND_KIT", KIT_DIR)
    raw_data_dir = env.get(
        "KUAIRAND_DATA", os.path.join(KIT_DIR, "KuaiRand-Pure", "data"))
    env["KUAIRAND_DATA"] = os.path.join(
        RUNTIME_DIR, ".generated-code-has-no-raw-data")
    env.setdefault("KUAIRAND_CACHE", os.path.join(RUNTIME_DIR, "cache"))
    if run_dir is not None:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["KUAIRAND_GUARD_RUN_DIR"] = os.path.abspath(run_dir)
        env["TMP"] = os.path.abspath(run_dir)
        env["TEMP"] = os.path.abspath(run_dir)
        env["KUAIRAND_GUARD_BLOCKED_PATHS"] = json.dumps([
            raw_data_dir,
            env["KUAIRAND_DATA"],
            os.path.join(ROOT, ".env"),
            os.path.join(ROOT, ".git"),
            os.path.join(ROOT, "scratch"),
            os.path.join(ROOT, "results"),
            os.path.join(ROOT, "logs"),
        ])
        env["KUAIRAND_GUARD_PROTECTED_PATHS"] = json.dumps(
            list(_protected_paths().values()))
    return env


def _expected_rows() -> dict:
    return {"valid": 124909, "test": 170588}


def run_solution(code: str, code_path: str, menu_choices: dict, run_dir: str,
                 timeout_s: int = 1200, seed: int = 0,
                 validation_evaluator=None) -> ExecResult:
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    with open(code_path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(code)

    code_sha256 = _sha256_file(code_path)
    protected_paths = _protected_paths()
    verification = {
        "schema_version": 1,
        "status": "running",
        "seed": int(seed),
        "solution_sha256": code_sha256,
        "menu_choices_sha256": _sha256_json(menu_choices),
        "runtime_fingerprint": _runtime_fingerprint(),
        "generated_code_guard": {
            "mode": "python_audit_hook",
            "blocks": ["raw_data", "project_env", "git", "prior_runs",
                       "writes_outside_run", "protected_file_mutation",
                       "subprocess", "exec", "network", "late_ctypes"],
            "os_sandbox": False,
        },
    }

    def protected_changes():
        after = _hash_protected(protected_paths)
        verification["protected_after"] = after
        return sorted(name for name in protected_before
                      if after[name] != protected_before[name])

    def solution_change_problem() -> str | None:
        """Record source state on every exit path without ever raising."""
        try:
            after_sha256 = _sha256_file(code_path)
        except OSError as error:
            verification["solution_after_sha256"] = None
            return ("generated solution.py is missing or unreadable after "
                    f"execution: {error}")
        verification["solution_after_sha256"] = after_sha256
        if after_sha256 != code_sha256:
            return "generated code rewrote its own reviewed solution.py source"
        return None

    def fail(trace: str, wall: float) -> ExecResult:
        source_problem = solution_change_problem()
        if source_problem and source_problem not in trace:
            trace = f"{trace}\n{source_problem}"
        verification["status"] = "error"
        verification["error"] = trace[:2000]
        try:
            _write_verification(run_dir, verification)
        except OSError as error:
            verification["verification_write_error"] = str(error)
        return ExecResult(False, error_trace=trace, wall_clock_seconds=wall,
                          run_dir=run_dir, verification=verification)

    try:
        trusted_score = validation_evaluator or _make_validation_evaluator()
    except Exception as error:
        snapshot = _hash_protected(protected_paths)
        verification["protected_before"] = snapshot
        verification["protected_after"] = snapshot
        verification["protected_changed"] = []
        return fail(
            f"trusted validation evaluator could not initialize: {error}", 0.0)

    # Cache migrations happen inside trusted evaluator initialization, so bind
    # the generated run to the settled post-migration cache files.
    protected_before = _hash_protected(protected_paths)
    verification["protected_before"] = protected_before

    cmd = [sys.executable, CHILD_GUARD, code_path,
           "--menu-choices", json.dumps(menu_choices),
           "--output-dir", run_dir, "--seed", str(seed)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=_env(run_dir), cwd=ROOT)
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        tail = ((e.stdout or b"").decode(errors="replace")
                if isinstance(e.stdout, bytes) else (e.stdout or ""))[-2000:]
        changed = protected_changes()
        verification["protected_changed"] = changed
        return fail((
            f"TIMEOUT: training run exceeded {timeout_s}s and was killed.\n"
            f"Last stdout:\n{tail}"), wall)
    wall = time.time() - t0
    changed = protected_changes()
    verification["protected_changed"] = changed

    if proc.returncode != 0:
        return fail((
            f"exit code {proc.returncode}\n--- stderr (tail) ---\n"
            f"{proc.stderr[-6000:]}\n--- stdout (tail) ---\n{proc.stdout[-2000:]}"),
            wall)

    # ---- validate outputs (contract enforcement) ----
    problems = []
    source_problem = solution_change_problem()
    if source_problem:
        problems.append(source_problem)
    reported_metrics = None
    metric_audit = None
    score_arrays = {}
    mp = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mp):
        problems.append("metrics.json was not written to --output-dir")
    else:
        try:
            with open(mp) as fh:
                reported_metrics = json.load(fh)
            for k in ("GAUC", "nDCG@5", "primary"):
                v = reported_metrics.get(k)
                if not isinstance(v, (int, float)) or not np.isfinite(v):
                    problems.append(f"metrics.json['{k}'] missing or non-finite: {v!r}")
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"metrics.json unreadable: {e}")
    for split, n_expected in _expected_rows().items():
        sp = os.path.join(run_dir, f"scores_{split}.npy")
        if not os.path.exists(sp):
            problems.append(f"scores_{split}.npy was not written")
            continue
        try:
            arr = np.load(sp)
            if arr.shape != (n_expected,):
                problems.append(f"scores_{split}.npy shape {arr.shape}, "
                                f"expected ({n_expected},)")
            elif not np.all(np.isfinite(arr)):
                problems.append(f"scores_{split}.npy contains NaN/Inf")
            else:
                score_arrays[split] = np.asarray(arr).copy()
        except Exception as e:
            problems.append(f"scores_{split}.npy unreadable: {e}")

    metrics = None
    if not problems and "valid" in score_arrays:
        try:
            metrics = trusted_score(score_arrays["valid"])
            reported = {k: float(reported_metrics[k])
                        for k in ("GAUC", "nDCG@5", "primary")}
            delta = max(abs(reported[k] - metrics[k]) for k in metrics)
            metric_audit = {
                "reported": reported,
                "parent_recomputed": metrics,
                "max_abs_difference": float(delta),
                "matched": bool(delta <= 1e-8),
            }
            with open(os.path.join(run_dir, "metrics_reported.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(reported, fh, indent=2)
            with open(mp, "w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
        except Exception as e:
            problems.append(f"parent validation recomputation failed: {e}")

    if changed:
        problems.append("generated code modified protected file(s): " + ", ".join(changed))

    # plausibility: a 'success' below random usually means a broken score sign etc.
    if metrics and metrics["primary"] < 0.40:
        problems.append(f"valid primary {metrics['primary']:.4f} is far below random "
                        f"(0.4834) — output contract or scoring is likely broken")

    verification["reported_metrics"] = (
        metric_audit["reported"] if metric_audit else None)
    verification["parent_recomputed_metrics"] = metrics
    verification["metric_audit"] = metric_audit
    verification["artifact_sha256"] = {}
    for path in (code_path,
                 os.path.join(run_dir, "scores_valid.npy"),
                 os.path.join(run_dir, "scores_test.npy")):
        digest = _safe_sha256_file(path)
        if digest is None:
            problem = f"artifact missing or unreadable during hashing: {path}"
            if problem not in problems:
                problems.append(problem)
        else:
            verification["artifact_sha256"][os.path.basename(path)] = digest

    if problems:
        return fail((
            "script exited 0 but violated the output contract:\n- "
            + "\n- ".join(problems)
            + f"\n--- stdout (tail) ---\n{proc.stdout[-2000:]}"),
            wall)

    verification["status"] = "success"
    try:
        _write_verification(run_dir, verification)
    except OSError as error:
        return fail(f"could not write verification manifest: {error}", wall)
    return ExecResult(True, metrics=metrics, wall_clock_seconds=wall,
                      run_dir=run_dir, metric_audit=metric_audit,
                      verification=verification)
