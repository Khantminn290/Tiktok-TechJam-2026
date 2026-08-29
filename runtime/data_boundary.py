"""Builds the sandboxed, label-safe data view that generated scripts run against.

A generated script runs inside the SAME process/interpreter as whatever it
imports, so pointing env vars at safe paths is not by itself a boundary: a
script that hardcodes the (fully guessable, prompt-visible) relative path to
the raw CSVs would reach them regardless of what KUAIRAND_DATA/KUAIRAND_CACHE
say. This module only builds the safe *copies*; the actual enforcement (making
the real paths unreadable for the duration of the subprocess, regardless of
which path string reaches them) lives in agent/executor.py.

Three artifacts are produced, all derived and regenerated from the real cache/
data whenever they're stale (mtime-compared), so nothing here is hand-maintained:

  ensure_sandbox_cache()   -- a copy of runtime/cache/ where test.npz has the
                              outcome-revealing columns removed. train.npz and
                              valid.npz are untouched: training needs train
                              labels, model selection needs validation labels,
                              and neither split is the hidden set.
  sandbox_raw_data_view()  -- a copy of the raw data dir containing only what
                              the legitimate diagnostic path
                              (train_lib._unbiased_check) needs: the video/
                              author lookup (no labels) and the random-exposure
                              log truncated to the validation date window --
                              test-window rows are physically absent, not
                              filtered by convention.
  build_worker_sandbox()  -- a PER-WORKER-SLOT copy of the above two, for
                              concurrent execution (see agent/executor.py
                              run_parallel_round). The redacted content is
                              worker-invariant -- same source, same
                              deterministic redaction -- so files are
                              HARDLINKED from the one canonical sandbox built
                              by the two functions above, not re-copied: zero
                              extra disk, near-instant setup regardless of how
                              many worker slots exist. Falls back to a real
                              copy only if hardlinking isn't possible (e.g. the
                              worker's git worktree ends up on a different
                              filesystem/volume -- os.link raises OSError).
"""
from __future__ import annotations

import csv
import os
import shutil
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

REAL_CACHE_DIR = os.path.join(_HERE, "cache")
SANDBOX_CACHE_DIR = os.path.join(_HERE, "cache_sandbox")

# Columns that reveal the outcome of a test-split impression. A generated
# script must never see these for the test split (it may for train/valid).
TEST_LABEL_COLUMNS = ("long_view", "is_click", "is_like", "is_forward", "play_time_ms",
                      # auxiliary supervision signals: outcomes of the impression,
                      # so they reveal the test label exactly as the others do
                      "is_follow", "is_comment", "is_hate", "is_profile_enter")

_VALID_WINDOW = (20220422, 20220428)   # inclusive -- test window (20220429+) excluded


def redact_test_columns(npz_files: dict) -> dict:
    """Pure function: drop label columns from a {col: array} mapping.

    Split out from ensure_sandbox_cache so it's unit-testable without needing
    the real dataset or filesystem.
    """
    return {k: v for k, v in npz_files.items() if k not in TEST_LABEL_COLUMNS}


def ensure_sandbox_cache(real_data_dir: str, real_cache_dir: str = REAL_CACHE_DIR,
                         sandbox_dir: str = SANDBOX_CACHE_DIR) -> str:
    """Build (if missing/stale) a copy of the cache where test.npz has no labels."""
    real_meta = os.path.join(real_cache_dir, "meta.json")
    if not os.path.exists(real_meta):
        import train_lib
        train_lib.build_cache(data_dir=real_data_dir, cache_dir=real_cache_dir)

    sandbox_meta = os.path.join(sandbox_dir, "meta.json")
    if os.path.exists(sandbox_meta) and \
            os.path.getmtime(sandbox_meta) >= os.path.getmtime(real_meta):
        return sandbox_dir  # already built from the current real cache

    os.makedirs(sandbox_dir, exist_ok=True)
    for name in ("meta.json", "vocabs.json"):
        shutil.copyfile(os.path.join(real_cache_dir, name),
                        os.path.join(sandbox_dir, name))
    for split in ("train", "valid"):
        shutil.copyfile(os.path.join(real_cache_dir, f"{split}.npz"),
                        os.path.join(sandbox_dir, f"{split}.npz"))

    z = np.load(os.path.join(real_cache_dir, "test.npz"), allow_pickle=True)
    redacted = redact_test_columns({k: z[k] for k in z.files})
    np.savez(os.path.join(sandbox_dir, "test.npz"), **redacted)
    return sandbox_dir


def sandbox_raw_data_view(real_data_dir: str,
                          sandbox_dir: str = SANDBOX_CACHE_DIR) -> str:
    """A raw-data-dir-shaped view with no test-window rows and no label columns
    for anything in the test window. Safe to hand a generated subprocess.
    """
    view_dir = os.path.join(sandbox_dir, "raw_data_view")
    marker = os.path.join(view_dir, ".built")
    src_random = os.path.join(real_data_dir, "log_random_4_22_to_5_08_pure.csv")
    if os.path.exists(marker) and os.path.getmtime(marker) >= os.path.getmtime(src_random):
        return view_dir

    os.makedirs(view_dir, exist_ok=True)
    for name in ("video_features_basic_pure.csv", "video_features_statistic_pure.csv",
                "user_features_pure.csv"):
        src = os.path.join(real_data_dir, name)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(view_dir, name))

    dst_random = os.path.join(view_dir, "log_random_4_22_to_5_08_pure.csv")
    with open(src_random, newline="") as fh_in, open(dst_random, "w", newline="") as fh_out:
        reader = csv.reader(fh_in)
        writer = csv.writer(fh_out)
        header = next(reader)
        writer.writerow(header)
        date_i = header.index("date")
        for row in reader:
            d = int(row[date_i])
            if _VALID_WINDOW[0] <= d <= _VALID_WINDOW[1]:
                writer.writerow(row)

    with open(marker, "w") as fh:
        fh.write("built\n")
    return view_dir


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return   # already set up for this worker slot -- reused across rounds
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def build_worker_sandbox(real_data_dir: str, worktree_root: str,
                         real_cache_dir: str = REAL_CACHE_DIR,
                         canonical_sandbox_dir: str = SANDBOX_CACHE_DIR) -> tuple[str, str]:
    """Populate a per-worker (cache_dir, data_dir) pair inside worktree_root,
    hardlinked from the one canonical redacted sandbox. Safe to call from
    multiple workers concurrently: this never touches real_data_dir/
    real_cache_dir once the canonical sandbox exists (build/refresh happens
    here, before any round-level lock is taken -- see run_parallel_round),
    and each worker only ever writes into its OWN worktree's directory.
    """
    ensure_sandbox_cache(real_data_dir, real_cache_dir, canonical_sandbox_dir)
    sandbox_raw_data_view(real_data_dir, canonical_sandbox_dir)

    worker_cache_dir = os.path.join(worktree_root, "runtime", "cache_sandbox_worker")
    worker_data_dir = os.path.join(worker_cache_dir, "raw_data_view")
    os.makedirs(worker_data_dir, exist_ok=True)

    for name in ("meta.json", "vocabs.json", "train.npz", "valid.npz", "test.npz"):
        _link_or_copy(os.path.join(canonical_sandbox_dir, name),
                      os.path.join(worker_cache_dir, name))

    src_view = os.path.join(canonical_sandbox_dir, "raw_data_view")
    for name in os.listdir(src_view):
        _link_or_copy(os.path.join(src_view, name), os.path.join(worker_data_dir, name))

    return worker_cache_dir, worker_data_dir
