"""Execution wrapper — the robustness layer.

Writes generated code to disk, runs it in a subprocess with a timeout, captures
stdout/stderr/exit code, parses metrics.json on success and returns a readable
error trace on any failure. The loop never crashes on a bad iteration: every
failure mode (syntax error, exception, timeout, malformed metrics, NaN scores)
becomes an error result that the next debug action feeds back to the LLM.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
KIT_DIR = os.path.join(ROOT, "kuairand-starter-kit")
RUNTIME_DIR = os.path.join(ROOT, "runtime")


class ExecResult:
    def __init__(self, ok: bool, metrics=None, error_trace=None,
                 wall_clock_seconds=0.0, run_dir=None):
        self.ok = ok
        self.metrics = metrics
        self.error_trace = error_trace
        self.wall_clock_seconds = wall_clock_seconds
        self.run_dir = run_dir


def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [RUNTIME_DIR, KIT_DIR, env.get("PYTHONPATH", "")])
    env.setdefault("KUAIRAND_KIT", KIT_DIR)
    env.setdefault("KUAIRAND_DATA", os.path.join(KIT_DIR, "KuaiRand-Pure", "data"))
    env.setdefault("KUAIRAND_CACHE", os.path.join(RUNTIME_DIR, "cache"))
    return env


def _expected_rows() -> dict:
    return {"valid": 124909, "test": 170588}


def run_solution(code: str, code_path: str, menu_choices: dict, run_dir: str,
                 timeout_s: int = 1200, seed: int = 0) -> ExecResult:
    os.makedirs(os.path.dirname(code_path), exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    with open(code_path, "w") as fh:
        fh.write(code)

    cmd = [sys.executable, code_path,
           "--menu-choices", json.dumps(menu_choices),
           "--output-dir", run_dir, "--seed", str(seed)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout_s, env=_env(), cwd=ROOT)
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
