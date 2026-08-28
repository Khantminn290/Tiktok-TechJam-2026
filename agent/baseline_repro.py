"""One-shot capture of the official baseline reproduction as a durable artifact.

The run-log deliverable needs proof the baseline was actually reproduced, not
just the number 0.6016 asserted in prose. This runs the starter kit's own
baseline.py --model fm and freezes stdout, parsed metrics, the exact command,
and enough provenance (repo commit + script hashes) that a judge can verify
kuairand-starter-kit/{baseline,data,evaluate}.py weren't quietly edited.

This is NOT the one-time hidden-test rule (see agent/make_submission.py
--final-test-eval): the organizers publish baseline numbers and expect
baseline.py to be rerun freely for verification. Safe to run repeatedly.

Usage: python3 -m agent.baseline_repro [--seed 0]
Writes logs/baseline/{stdout.txt, metrics.json}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KIT = os.path.join(_ROOT, "kuairand-starter-kit")
OUT_DIR = os.path.join(_ROOT, "logs", "baseline")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        h.update(fh.read())
    return h.hexdigest()


def _git_commit(root: str = _ROOT) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unknown (git failed)"
    except (OSError, subprocess.SubprocessError):
        return "unknown (git unavailable)"


def _parse_stdout(text: str) -> dict:
    """Parses baseline.py's own print format: '  valid  GAUC 0.x | nDCG@5 0.x | primary 0.x'."""
    metrics = {}
    for line in text.splitlines():
        s = line.strip()
        for split in ("valid", "test"):
            if s.startswith(split) and "GAUC" in s:
                toks = s.replace("|", " ").split()
                d = {}
                for i, tok in enumerate(toks):
                    if tok in ("GAUC", "nDCG@5", "primary"):
                        d[tok] = float(toks[i + 1])
                if len(d) == 3:
                    metrics[split] = d
    return metrics


def run(seed: int = 0) -> dict:
    os.makedirs(OUT_DIR, exist_ok=True)
    cmd = [sys.executable, "baseline.py", "--model", "fm", "--seed", str(seed)]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=KIT, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0:
        sys.exit(f"baseline reproduction FAILED (exit {proc.returncode}):\n{proc.stderr}")

    metrics = _parse_stdout(proc.stdout)
    if "valid" not in metrics or "test" not in metrics:
        sys.exit("could not parse valid/test metrics from baseline.py stdout -- "
                 "its print format may have changed (evaluate.py itself must stay "
                 "untouched, but if baseline.py's prints changed, update the "
                 "parser in agent/baseline_repro.py). Raw stdout:\n" + proc.stdout)

    record = {
        "command": " ".join(cmd),
        "cwd": os.path.relpath(KIT, _ROOT),
        "seed": seed,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_seconds": round(wall, 2),
        "repo_commit": _git_commit(),
        "script_sha256": {
            name: _sha256_file(os.path.join(KIT, name))
            for name in ("baseline.py", "data.py", "evaluate.py")
        },
        "metrics": metrics,
    }
    with open(os.path.join(OUT_DIR, "stdout.txt"), "w") as fh:
        fh.write(proc.stdout)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as fh:
        json.dump(record, fh, indent=2)
    print(f"wrote {os.path.join(OUT_DIR, 'stdout.txt')}")
    print(f"wrote {os.path.join(OUT_DIR, 'metrics.json')}")
    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(seed=a.seed)
