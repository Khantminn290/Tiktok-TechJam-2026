"""Execution wrapper — the robustness layer.

Writes generated code to disk, runs it in a subprocess with a timeout, captures
stdout/stderr/exit code, parses metrics.json on success and returns a readable
error trace on any failure. The loop never crashes on a bad iteration: every
failure mode (syntax error, exception, timeout, malformed metrics, NaN scores)
becomes an error result that the next debug action feeds back to the LLM.

This module also owns the two hidden-test/integrity boundaries around that
subprocess:
  - a technical (not instruction-following) train/valid/test boundary: the
    generated script only ever sees a sandboxed data view (test split has no
    label columns) because the *real* dataset and cache directories are
    chmod'd unreadable for the exact duration of the subprocess -- see
    restricted_access(). This holds regardless of what path string (env var,
    relative, hardcoded absolute) the generated code uses to try to reach them.
  - a protected-files write-lock: the journal, config/, and the final-eval lock
    are chmod'd read-only for the same window, so generated code cannot open
    them for writing no matter what it tries.
"""
from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
import os
import stat
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
KIT_DIR = os.path.join(ROOT, "kuairand-starter-kit")
RUNTIME_DIR = os.path.join(ROOT, "runtime")

if RUNTIME_DIR not in sys.path:
    sys.path.insert(0, RUNTIME_DIR)
import data_boundary  # noqa: E402

# ---- the real, labeled data the generated subprocess must never reach ----
REAL_DATA_DIR = os.path.join(KIT_DIR, "KuaiRand-Pure", "data")
REAL_CACHE_DIR = os.path.join(RUNTIME_DIR, "cache")

# ---- files/dirs generated code must never be able to open for writing ----
PROTECTED_PATHS = [
    os.path.join(ROOT, "logs", "journal.jsonl"),
    os.path.join(ROOT, "config"),
    os.path.join(ROOT, "results", "final_evaluation.lock"),
    os.path.join(ROOT, "agent", "experience.md"),
]

SEED_SOLUTION_PATH = os.path.join(RUNTIME_DIR, "seed_solution.py")


class ExecResult:
    def __init__(self, ok: bool, metrics=None, error_trace=None,
                 wall_clock_seconds=0.0, run_dir=None):
        self.ok = ok
        self.metrics = metrics
        self.error_trace = error_trace
        self.wall_clock_seconds = wall_clock_seconds
        self.run_dir = run_dir


def _env(sandbox_cache_dir: str, sandbox_data_dir: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [RUNTIME_DIR, KIT_DIR, env.get("PYTHONPATH", "")])
    env["KUAIRAND_KIT"] = KIT_DIR
    # Direct assignment, not setdefault: an inherited shell env var pointing at
    # the real, labeled paths must never leak through to the sandboxed run.
    env["KUAIRAND_DATA"] = sandbox_data_dir
    env["KUAIRAND_CACHE"] = sandbox_cache_dir
    return env


def _expected_rows() -> dict:
    return {"valid": 124909, "test": 170588}


def assert_not_root() -> None:
    """The boundary in this module is enforced with chmod, which root ignores
    unconditionally -- as root, every restricted_access() call below would
    silently grant full access instead of blocking it, while still looking
    like it succeeded. Fail loudly rather than pretend to protect anything.
    """
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise RuntimeError(
            "refusing to proceed: the train/valid/test boundary and the "
            "protected-files lock (agent/executor.py) are enforced with POSIX "
            "chmod, which root bypasses unconditionally. Running this agent as "
            "root would silently provide none of the guarantees described in "
            "README.md/HANDOVER.md -- every 'blocked' path would actually "
            "succeed. Run as a non-root user.")


@contextlib.contextmanager
def restricted_access(unreadable_paths=(), read_only_paths=()):
    """Temporarily restrict filesystem access for exactly the wrapped block.

    Restrictions are OS permission bits, not a convention the wrapped code has
    to cooperate with: chmod'ing a directory to 0o000 removes the search
    permission needed to resolve ANY path through it, so a hardcoded absolute
    path fails identically to one built from an env var. Permissions are always
    restored, including when the wrapped code raises or times out.
    """
    assert_not_root()
    saved = {}

    def _lock_file(p, mode):
        if p not in saved and os.path.exists(p):
            saved[p] = stat.S_IMODE(os.stat(p).st_mode)
            os.chmod(p, mode)

    try:
        for p in unreadable_paths:
            _lock_file(p, 0o000)
        for p in read_only_paths:
            if not os.path.exists(p):
                continue
            if os.path.isdir(p):
                # Directory mode alone only governs creating/deleting/renaming
                # entries -- an existing file's own mode bits still control
                # whether it can be opened for writing, so every file already
                # inside has to be locked individually too.
                for root, _dirs, files in os.walk(p):
                    for name in files:
                        _lock_file(os.path.join(root, name), 0o444)
                _lock_file(p, 0o555)
            else:
                _lock_file(p, 0o444)
        yield
    finally:
        for p, mode in saved.items():
            try:
                os.chmod(p, mode)
            except OSError:
                pass


def save_diff(new_code_path: str, parent_code_path, diffs_dir: str,
             iteration_id: int) -> dict:
    """Unified diff of an accepted candidate against its parent (or the seed
    solution, for a parentless draft) -- the literal 'code diff applied'
    deliverable. Saving full scripts (already done under logs/solutions/)
    shows what a node IS, not what changed to produce it.
    """
    os.makedirs(diffs_dir, exist_ok=True)
    base_path = parent_code_path if (parent_code_path and os.path.exists(parent_code_path)) \
        else SEED_SOLUTION_PATH
    with open(new_code_path) as fh:
        new_lines = fh.readlines()
    old_lines = []
    base_label = "(no parent -- new file)"
    if os.path.exists(base_path):
        with open(base_path) as fh:
            old_lines = fh.readlines()
        base_label = os.path.relpath(base_path, ROOT)
    diff_text = "".join(difflib.unified_diff(
        old_lines, new_lines, fromfile=base_label,
        tofile=os.path.relpath(new_code_path, ROOT)))
    diff_path = os.path.join(diffs_dir, f"node_{iteration_id:03d}.diff")
    with open(diff_path, "w") as fh:
        fh.write(diff_text)
    return {"diff_path": os.path.relpath(diff_path, ROOT),
            "diff_sha256": hashlib.sha256(diff_text.encode("utf-8")).hexdigest()}


def run_solution(code: str, code_path: str, menu_choices: dict, run_dir: str,
                 timeout_s: int = 1200, seed: int = 0) -> ExecResult:
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    with open(code_path, "w") as fh:
        fh.write(code)

    cmd = [sys.executable, code_path,
           "--menu-choices", json.dumps(menu_choices),
           "--output-dir", run_dir, "--seed", str(seed)]

    # Build the sandboxed data view BEFORE locking the real paths down (this
    # step needs to read the real cache/data itself).
    sandbox_cache_dir = data_boundary.ensure_sandbox_cache(REAL_DATA_DIR, REAL_CACHE_DIR)
    sandbox_data_dir = data_boundary.sandbox_raw_data_view(REAL_DATA_DIR)
    env = _env(sandbox_cache_dir, sandbox_data_dir)

    t0 = time.time()
    try:
        with restricted_access(unreadable_paths=[REAL_DATA_DIR, REAL_CACHE_DIR],
                               read_only_paths=PROTECTED_PATHS):
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=timeout_s, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired as e:
        wall = time.time() - t0
        tail = ((e.stdout or b"").decode(errors="replace")
                if isinstance(e.stdout, bytes) else (e.stdout or ""))[-2000:]
        return ExecResult(False, error_trace=(
            f"TIMEOUT: training run exceeded {timeout_s}s and was killed.\n"
            f"Last stdout:\n{tail}"), wall_clock_seconds=wall, run_dir=run_dir)
    wall = time.time() - t0

    if proc.returncode != 0:
        return ExecResult(False, error_trace=(
            f"exit code {proc.returncode}\n--- stderr (tail) ---\n"
            f"{proc.stderr[-6000:]}\n--- stdout (tail) ---\n{proc.stdout[-2000:]}"),
            wall_clock_seconds=wall, run_dir=run_dir)

    # ---- validate outputs (contract enforcement) ----
    problems = []
    metrics = None
    mp = os.path.join(run_dir, "metrics.json")
    if not os.path.exists(mp):
        problems.append("metrics.json was not written to --output-dir")
    else:
        try:
            with open(mp) as fh:
                metrics = json.load(fh)
            for k in ("GAUC", "nDCG@5", "primary"):
                v = metrics.get(k)
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
        except Exception as e:
            problems.append(f"scores_{split}.npy unreadable: {e}")
    # plausibility: a 'success' below random usually means a broken score sign etc.
    if metrics and isinstance(metrics.get("primary"), (int, float)) \
            and metrics["primary"] < 0.40:
        problems.append(f"valid primary {metrics['primary']:.4f} is far below random "
                        f"(0.4834) — output contract or scoring is likely broken")

    if problems:
        return ExecResult(False, error_trace=(
            "script exited 0 but violated the output contract:\n- "
            + "\n- ".join(problems)
            + f"\n--- stdout (tail) ---\n{proc.stdout[-2000:]}"),
            wall_clock_seconds=wall, run_dir=run_dir)

    return ExecResult(True, metrics={k: float(metrics[k])
                                     for k in ("GAUC", "nDCG@5", "primary")},
                      wall_clock_seconds=wall, run_dir=run_dir)
