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
import signal
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

sys.path.insert(0, ROOT)
from agent import leakage_check  # noqa: E402
from agent import preflight  # noqa: E402

# ---- the real, labeled data the generated subprocess must never reach ----
REAL_DATA_DIR = os.path.join(KIT_DIR, "KuaiRand-Pure", "data")
REAL_CACHE_DIR = os.path.join(RUNTIME_DIR, "cache")

# ---- files/dirs generated code must never be able to open for writing ----
PROTECTED_PATHS = [
    os.path.join(ROOT, "logs", "journal.jsonl"),
    os.path.join(ROOT, "config"),
    os.path.join(ROOT, "results", "final_evaluation.lock"),
    os.path.join(ROOT, "agent", "experience.md"),
    os.path.join(ROOT, "logs", "best_override_log.jsonl"),
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
            "README.md -- every 'blocked' path would actually "
            "succeed. Run as a non-root user.")


@contextlib.contextmanager
def restricted_access(unreadable_paths=(), read_only_paths=()):
    """Temporarily restrict filesystem access for exactly the wrapped block.

    Restrictions are OS permission bits, not a convention the wrapped code has
    to cooperate with: chmod'ing a directory to 0o000 removes the search
    permission needed to resolve ANY path through it, so a hardcoded absolute
    path fails identically to one built from an env var. Permissions are always
    restored, including when the wrapped code raises or times out.

    ...and including when the process is TERMINATED. `finally` does not run on
    SIGTERM, so a plain `kill` of an agent run used to leave the dataset and the
    cache at mode 0o000 -- the data was intact but unreadable, and the next run
    failed with "dataset not found". Found by killing a run mid-experiment.
    A SIGTERM/SIGINT handler restores the saved modes before re-raising, so an
    interrupted run leaves the working tree usable.
    """
    assert_not_root()
    saved = {}

    def _lock_file(p, mode):
        if p not in saved and os.path.exists(p):
            saved[p] = stat.S_IMODE(os.stat(p).st_mode)
            os.chmod(p, mode)

    def _restore():
        for p, mode in saved.items():
            try:
                os.chmod(p, mode)
            except OSError:
                pass

    def _on_signal(signum, _frame):
        _restore()
        # Restore the default disposition and re-raise, so the process still
        # dies the way the sender intended -- this handler exists to clean up,
        # not to make the run unkillable.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    prev_handlers = {}
    # Only the main thread may install handlers; parallel workers run in threads
    # and simply rely on the coordinator's handler plus `finally`.
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            prev_handlers[sig] = signal.signal(sig, _on_signal)
    except ValueError:
        prev_handlers = {}

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
        _restore()
        for sig, prev in prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, TypeError):
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
                 timeout_s: int = 1200, seed: int = 0, cwd: str | None = None,
                 sandbox_paths: tuple[str, str] | None = None,
                 lock_real_dirs: bool = True) -> ExecResult:
    """cwd/sandbox_paths/lock_real_dirs exist for run_parallel_round(): a worker
    running inside its own worktree passes its own (worktree cwd, already-
    hardlinked sandbox paths) and lock_real_dirs=False, because the COORDINATOR
    holds one shared lock for the whole concurrent round -- see the module
    docstring for why per-subprocess locking breaks under concurrency. The
    sequential caller (agent.loop) uses none of these and behaves exactly as
    before.
    """
    # code_path/run_dir get passed as argv to a subprocess whose cwd may be a
    # worktree, not this process's cwd -- a RELATIVE path would then resolve
    # against the WRONG directory (found the hard way: a relative code_path
    # written correctly from here still produced "can't open file" inside a
    # worker's worktree, because the subprocess resolved it against cwd=
    # worktree_root instead). Absolute paths make this caller-independent.
    code_path = os.path.abspath(code_path)
    run_dir = os.path.abspath(run_dir)
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    with open(code_path, "w") as fh:
        fh.write(code)

    # --- preflight: is this script even runnable? ---
    # Cheapest-first validation of syntax, imports, capability references and
    # configuration. This is what stops the failure that dominated the clean
    # evaluation -- calling a capability that does not exist in generated code --
    # from costing a full training run to discover. Leakage is skipped here and
    # left to the dedicated review below, which is the enforcement path.
    pre = preflight.preflight(code_path, menu_choices=menu_choices, menu=None,
                              skip=(preflight.LEAKAGE,))
    if not pre["ok"]:
        return ExecResult(False, error_trace=(
            "PREFLIGHT REJECTED THIS SCRIPT — it was never executed, so no "
            "training time was spent and the hypothesis remains untested.\n\n"
            + pre["feedback"]),
            wall_clock_seconds=0.0, run_dir=run_dir)

    # --- pre-execution leakage review (static, never executes the code) ---
    # The sandbox stops a script READING test labels; it cannot stop a script
    # building a feature out of labels it is allowed to see. Only clear
    # violations block -- a checker that rejects legitimate work is worse than
    # none, so everything else is advisory and surfaced to the agent.
    leak = leakage_check.check_file(code_path)
    if leak["block"]:
        return ExecResult(False, error_trace=(
            "BLOCKED BEFORE EXECUTION by the leakage review — the experiment "
            "was not run, so the hypothesis is still untested.\n"
            + leakage_check.render_for_agent(leak)),
            wall_clock_seconds=0.0, run_dir=run_dir)

    cmd = [sys.executable, code_path,
           "--menu-choices", json.dumps(menu_choices),
           "--output-dir", run_dir, "--seed", str(seed)]

    if sandbox_paths is not None:
        sandbox_cache_dir, sandbox_data_dir = sandbox_paths
    else:
        # Build the sandboxed data view BEFORE locking the real paths down
        # (this step needs to read the real cache/data itself).
        sandbox_cache_dir = data_boundary.ensure_sandbox_cache(REAL_DATA_DIR, REAL_CACHE_DIR)
        sandbox_data_dir = data_boundary.sandbox_raw_data_view(REAL_DATA_DIR)
    env = _env(sandbox_cache_dir, sandbox_data_dir)
    run_cwd = cwd or ROOT

    def _launch():
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=env, cwd=run_cwd)

    t0 = time.time()
    try:
        if lock_real_dirs:
            with restricted_access(unreadable_paths=[REAL_DATA_DIR, REAL_CACHE_DIR],
                                   read_only_paths=PROTECTED_PATHS):
                proc = _launch()
        else:
            proc = _launch()
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


def run_parallel_round(jobs: list[dict], timeout_s: int = 1200) -> list[ExecResult]:
    """Runs len(jobs) subprocesses CONCURRENTLY, each inside its own git
    worktree with its own hardlinked sandbox (runtime.data_boundary), under
    ONE shared real-data lock for the whole round.

    This is the concurrency fix, not a convenience wrapper: the sequential
    run_solution() takes/releases the lock once per call, which is correct
    only when calls never overlap. Two overlapping calls on the same shared
    directory can either unlock it mid-run (the OTHER worker's subprocess is
    still executing when this one restores real permissions) or permanently
    relock it after both finish (whichever call entered second captured
    "already locked" as its own "original" state, and restores to that).
    Locking once, here, for the whole batch removes the shared mutable
    resource from the per-worker code path entirely.

    Each job: {"slot": int, "code": str, "code_path": str,
              "menu_choices": dict, "run_dir": str, "seed": int}.
    Returns results in the SAME order as `jobs` (not completion order).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from . import worktree

    prepared = []
    for job in jobs:
        wt_root = worktree.ensure_worktree(job["slot"])
        cache_dir, data_dir = data_boundary.build_worker_sandbox(REAL_DATA_DIR, wt_root)
        prepared.append((job, wt_root, cache_dir, data_dir))

    def _run_one(job, wt_root, cache_dir, data_dir):
        return run_solution(job["code"], job["code_path"], job["menu_choices"],
                           job["run_dir"], timeout_s=timeout_s,
                           seed=job.get("seed", 0), cwd=wt_root,
                           sandbox_paths=(cache_dir, data_dir), lock_real_dirs=False)

    results: list = [None] * len(jobs)
    with restricted_access(unreadable_paths=[REAL_DATA_DIR, REAL_CACHE_DIR],
                           read_only_paths=PROTECTED_PATHS):
        with ThreadPoolExecutor(max_workers=max(1, len(prepared))) as ex:
            future_to_i = {ex.submit(_run_one, *spec): i
                          for i, spec in enumerate(prepared)}
            for fut in as_completed(future_to_i):
                results[future_to_i[fut]] = fut.result()
    return results
