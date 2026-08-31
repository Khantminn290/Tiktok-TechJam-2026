"""What produced this number?

A result without provenance is a claim, not a measurement. A judge reading
`primary: 0.60541` has to be able to reconstruct exactly what made it: which
commit, which data, which config, which seeds, which evaluation path.

The gap this closes was real and embarrassing: `ensemble_results.json` reported
the submitted result with `git_sha: null` and `data_fingerprint: null`. The
ensemble was reproducible in practice -- the members were on disk and the config
was recorded -- but nothing tied it to a commit or to a specific state of the
data, so "reproducible" rested on nobody having changed anything in between.

Design notes:

  * Fingerprints are content hashes, not timestamps. mtime changes when a file
    is touched; a content hash changes when the file actually differs.
  * The data fingerprint hashes the CACHE METADATA and per-split row counts
    rather than the multi-GB arrays. Rebuilding the cache from different source
    data changes row counts and column sets, which is the failure this needs to
    catch; hashing gigabytes on every stamp is not affordable.
  * `dirty` is recorded honestly. A result produced from an uncommitted tree is
    still a result, but the SHA alone does not identify it, and pretending
    otherwise is worse than saying so.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
CACHE_DIR = os.path.join(ROOT, "runtime", "cache")

SCHEMA = "provenance/1"


def is_generated_path(path: str) -> bool:
    """Whether a dirty path is run evidence rather than executable source."""
    path = path.strip()
    return (path.startswith(("logs/", "results/", "submission_"))
            or path in ("RESULTS.md", "docs/DEVPOST_SUBMISSION.md",
                        "agent/experience.md"))


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                             text=True, timeout=15)
        # Preserve the first porcelain status column. ``strip()`` changed
        # " M logs/x" into "M logs/x", then the path parser dropped its first
        # character and falsely classified generated evidence as source code.
        return out.stdout.rstrip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_state() -> dict:
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    paths = [line[3:].strip() for line in (status or "").splitlines()]
    source = [p for p in paths if not is_generated_path(p)]
    generated = [p for p in paths if is_generated_path(p)]
    return {"sha": sha,
            "short_sha": sha[:12] if sha else None,
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            # A dirty tree means the SHA does not fully identify the code that
            # ran. Recorded, never hidden.
            "dirty": bool(status) if status is not None else None,
            "dirty_files": len(paths),
            # A run necessarily writes journals and result artifacts. Keep raw
            # Git dirtiness above, but distinguish it from changes to code,
            # config, or dependency definitions that make the SHA insufficient.
            "source_dirty": bool(source),
            "source_dirty_files": source,
            "generated_dirty_files": generated}


def data_fingerprint(cache_dir: str = CACHE_DIR) -> dict:
    """Identify the dataset state without hashing the whole cache.

    Row counts and column names per split are what actually change when the
    cache is rebuilt from different data, so they are the useful signal.
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        return {"available": False, "reason": "cache not built"}
    h = hashlib.sha256()
    with open(meta_path, "rb") as fh:
        meta_bytes = fh.read()
    h.update(meta_bytes)
    splits = {}
    try:
        import numpy as np
        for name in ("train", "valid", "test"):
            p = os.path.join(cache_dir, f"{name}.npz")
            if not os.path.exists(p):
                continue
            z = np.load(p, allow_pickle=True)
            cols = sorted(z.files)
            n = int(z[cols[0]].shape[0]) if cols else 0
            splits[name] = {"rows": n, "columns": cols}
            h.update(f"{name}:{n}:{','.join(cols)}".encode())
    except Exception as e:                      # noqa: BLE001 - advisory only
        return {"available": False, "reason": f"{type(e).__name__}: {e}"[:120]}
    return {"available": True, "sha256": h.hexdigest()[:16],
            "splits": splits, "dataset": "KuaiRand-Pure"}


def config_fingerprint(config: dict | None) -> str | None:
    if not config:
        return None
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def code_fingerprint(paths) -> dict:
    """Content hash of the specific files that implement a result."""
    out = {}
    for p in paths:
        ap = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(ap):
            out[p] = None
            continue
        with open(ap, "rb") as fh:
            out[p] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def stamp(config: dict | None = None, seeds=None, code_paths=(),
          evaluation: str | None = None, extra: dict | None = None) -> dict:
    """The provenance block to embed in any result artifact."""
    p = {"schema": SCHEMA,
         "stamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "git": git_state(),
         "data": data_fingerprint(),
         "config_sha": config_fingerprint(config),
         "seeds": list(seeds) if seeds is not None else None,
         "code": code_fingerprint(code_paths) if code_paths else {},
         "evaluation": evaluation,
         "dataset_scope": "KuaiRand-Pure only"}
    if extra:
        p.update(extra)
    return p


def apply_to_file(path: str, config_key: str = "config",
                  seeds=None, code_paths=(), evaluation: str | None = None,
                  overwrite: bool = False) -> dict:
    """Add a `provenance` block to an existing JSON artifact, in place.

    Refuses to overwrite an existing stamp unless asked: a stamp records the
    state that produced the numbers already in the file, so silently restamping
    a stale artifact would assert something false about it.
    """
    with open(path) as fh:
        doc = json.load(fh)
    if doc.get("provenance") and not overwrite:
        return {"changed": False, "reason": "already stamped",
                "provenance": doc["provenance"]}
    doc["provenance"] = stamp(config=doc.get(config_key), seeds=seeds,
                              code_paths=code_paths, evaluation=evaluation)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=2)
    os.replace(tmp, path)
    return {"changed": True, "provenance": doc["provenance"]}


def verify(path: str) -> dict:
    """Does this artifact's stamp still match the current tree and data?

    Reports drift rather than judging it -- a mismatch means "this was produced
    under different conditions", which is information, not necessarily an error.
    """
    with open(path) as fh:
        doc = json.load(fh)
    p = doc.get("provenance")
    if not p:
        return {"stamped": False, "matches": False,
                "issues": ["artifact carries no provenance block"]}
    issues = []
    now_git, now_data = git_state(), data_fingerprint()
    if p.get("git", {}).get("sha") != now_git.get("sha"):
        issues.append(f"code moved: stamped {p.get('git', {}).get('short_sha')}, "
                      f"now {now_git.get('short_sha')}")
    if p.get("data", {}).get("sha256") != now_data.get("sha256"):
        issues.append(f"data differs: stamped {p.get('data', {}).get('sha256')}, "
                      f"now {now_data.get('sha256')}")
    if now_git.get("dirty"):
        issues.append(f"working tree is dirty ({now_git.get('dirty_files')} files); "
                      f"the current SHA does not fully identify the code")
    return {"stamped": True, "matches": not issues, "issues": issues,
            "stamped_at": p.get("stamped_utc")}


def render(p: dict) -> str:
    g, d = p.get("git") or {}, p.get("data") or {}
    L = [f"  commit    {g.get('short_sha')} on {g.get('branch')}"
         + ("  [DIRTY TREE]" if g.get("dirty") else ""),
         f"  data      {d.get('dataset')} fp={d.get('sha256')}"]
    for name, s in (d.get("splits") or {}).items():
        L.append(f"              {name}: {s['rows']:,} rows")
    L += [f"  config    sha={p.get('config_sha')}",
          f"  seeds     {p.get('seeds')}",
          f"  evaluate  {p.get('evaluation')}",
          f"  stamped   {p.get('stamped_utc')}"]
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="stamp/verify artifact provenance")
    ap.add_argument("--verify", metavar="PATH")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()
    if a.verify:
        r = verify(a.verify)
        print(json.dumps(r, indent=2))
        raise SystemExit(0 if r.get("matches") else 1)
    print(render(stamp()))
