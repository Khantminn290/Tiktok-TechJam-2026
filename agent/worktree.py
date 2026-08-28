"""Isolated git worktrees for parallel worker execution.

Each worker slot gets a full, independent checkout of the repo -- cheap:
worktrees share the .git object store, so this is a lightweight linked
checkout, not a clone. Workers never commit anything (any accepted result is
copied into the main tree's logs/ by the coordinator, not merged via git), so
every worktree is created in DETACHED HEAD state at the current commit; that
also sidesteps `git worktree add` refusing to check out the same branch into
two worktrees at once, since there's no branch involved at all.

Because kuairand-starter-kit/KuaiRand-Pure/ and runtime/cache/ are gitignored,
`git worktree add` does NOT check them out -- a fresh worktree structurally
has no copy of the real, labeled dataset at all. That is a stronger guarantee
than a permission bit: a hardcoded path *relative to the worktree* hits
FileNotFoundError, not "present but denied." It does not, by itself, stop a
hardcoded ABSOLUTE path back to the main repo's real data -- that is still
handled by the round-level chmod lock in agent.executor.run_parallel_round.
"""
from __future__ import annotations

import os
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
WORKTREES_DIR = os.path.join(ROOT, ".worktrees")


def worktree_path(slot: int) -> str:
    return os.path.join(WORKTREES_DIR, f"worker_{slot}")


def ensure_worktree(slot: int) -> str:
    """Creates the worktree for this slot if it doesn't already exist.
    Reused across rounds -- not torn down and rebuilt every iteration.
    """
    path = worktree_path(slot)
    subprocess.run(["git", "worktree", "prune", "-q"], cwd=ROOT,
                   capture_output=True, text=True)
    if os.path.isdir(path):
        return path
    os.makedirs(WORKTREES_DIR, exist_ok=True)
    r = subprocess.run(["git", "worktree", "add", "--detach", "-q", path, "HEAD"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git worktree add failed for slot {slot}: {r.stderr}")
    return path


def remove_worktree(slot: int) -> None:
    path = worktree_path(slot)
    if not os.path.isdir(path):
        return
    subprocess.run(["git", "worktree", "remove", "--force", path],
                   cwd=ROOT, capture_output=True, text=True)


def remove_all_worktrees() -> None:
    if not os.path.isdir(WORKTREES_DIR):
        return
    for name in list(os.listdir(WORKTREES_DIR)):
        subprocess.run(["git", "worktree", "remove", "--force",
                        os.path.join(WORKTREES_DIR, name)],
                       cwd=ROOT, capture_output=True, text=True)
    subprocess.run(["git", "worktree", "prune", "-q"], cwd=ROOT,
                   capture_output=True, text=True)


def is_available() -> bool:
    try:
        r = subprocess.run(["git", "worktree", "list"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
