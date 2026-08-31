"""Does the submitted number still follow from the artifacts on disk?

This is the guard that stands between "we reported 0.60541" and "we can show
0.60541". It never trains: it reloads the stored member predictions, re-applies
the recorded aggregation rule, re-evaluates, and demands an exact match against
the reported metrics.

Not retraining is the point. Rebuilding the ensemble to check it would REPLACE
the very artifacts being checked, so a verification step would become a way to
silently mutate the incumbent -- the exact accident this guards against.

It caught a real gap on first use. `ensemble_results.json` recorded the config,
the members and the metrics, but not the AGGREGATION RULE. A plain mean of the
16 members gives 0.60528; the rank-normalised mean gives 0.60541. The reported
number was correct, but the file did not contain enough information to
reproduce it -- someone re-deriving the result from the recorded fields would
have got a different number and had no way to tell which was right.

Usage:
    python3 -m agent.verify_incumbent            # verify, exit 1 on mismatch
    python3 -m agent.verify_incumbent --stamp    # verify, then write provenance
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
for p in (ROOT, os.path.join(ROOT, "runtime"),
          os.path.join(ROOT, "kuairand-starter-kit")):
    if p not in sys.path:
        sys.path.insert(0, p)

RESULTS = os.path.join(ROOT, "logs", "ensemble_results.json")

# The rule that actually produced the reported number. Recorded here AND
# written into the artifact's provenance, because the artifact did not say.
AGGREGATION = "rank_normalise_then_mean"


def _aggregate(member_scores, rule: str = AGGREGATION):
    import numpy as np
    from agent.ensemble import rank_normalise
    if rule == "rank_normalise_then_mean":
        return np.mean([rank_normalise(s) for s in member_scores], axis=0)
    if rule == "mean":
        return np.mean(member_scores, axis=0)
    raise ValueError(f"unknown aggregation rule: {rule}")


def verify(results_path: str = RESULTS, tol: float = 0.0) -> dict:
    """Recompute the submitted metrics from stored members.

    tol=0.0 means exact agreement at the 5-decimal precision the result is
    reported to. A drifting ensemble should fail loudly, not round into
    agreement.
    """
    import numpy as np
    import train_lib
    from evaluate import evaluate

    with open(results_path) as fh:
        rec = json.load(fh)

    members_dir = os.path.join(ROOT, rec.get("members_dir", "logs/final_ensemble"))
    issues = []
    if not os.path.isdir(members_dir):
        return {"ok": False, "issues": [f"members_dir missing: {members_dir}"]}

    # Publication keeps the generated ``seed_XX.py`` scripts beside the member
    # directories for reproducibility. Only directories are ensemble members;
    # counting same-prefix sidecars doubled k from 16 to 32 after the agent
    # began publishing its own canonical artifact.
    seed_dirs = sorted(
        d for d in os.listdir(members_dir)
        if d.startswith("seed_") and os.path.isdir(os.path.join(members_dir, d)))
    expected_k = rec.get("k")
    if expected_k is not None and len(seed_dirs) != expected_k:
        issues.append(f"expected {expected_k} members, found {len(seed_dirs)}")

    scores, missing = [], []
    for d in seed_dirs:
        p = os.path.join(members_dir, d, "scores_valid.npy")
        (scores.append(np.load(p)) if os.path.exists(p) else missing.append(d))
    if missing:
        issues.append(f"members missing scores_valid.npy: {missing}")
    if not scores:
        return {"ok": False, "issues": issues + ["no member predictions on disk"]}

    splits, _meta = train_lib.load_cache()
    va = splits["valid"]
    rule = (rec.get("provenance") or {}).get("aggregation", AGGREGATION)
    got = evaluate(list(va["user_raw"]), va["long_view"], _aggregate(scores, rule))

    recomputed, reported, mismatches = {}, {}, []
    for key in ("primary", "GAUC", "nDCG@5"):
        r = round(float(got[key]), 5)
        recomputed[key] = r
        if key in rec:
            reported[key] = rec[key]
            if abs(r - rec[key]) > tol:
                mismatches.append(f"{key}: recomputed {r} vs reported {rec[key]}")
    issues += mismatches

    return {"ok": not issues, "issues": issues, "k": len(scores),
            "aggregation": rule, "recomputed": recomputed, "reported": reported,
            "members_dir": os.path.relpath(members_dir, ROOT)}


def render(v: dict) -> str:
    L = ["=" * 70, "INCUMBENT VERIFICATION — recomputed from stored predictions",
         "=" * 70,
         f"members      {v.get('k')} from {v.get('members_dir')}",
         f"aggregation  {v.get('aggregation')}"]
    for key in ("primary", "GAUC", "nDCG@5"):
        if key in v.get("recomputed", {}):
            rep = v.get("reported", {}).get(key)
            L.append(f"  {key:<9} recomputed {v['recomputed'][key]:.5f}"
                     + (f"   reported {rep:.5f}" if rep is not None else "")
                     + ("   MATCH" if rep == v["recomputed"][key] else
                        "   *** MISMATCH ***" if rep is not None else ""))
    if v["ok"]:
        L.append("\nVERIFIED: the reported result follows from the artifacts on disk.")
    else:
        L.append("\nFAILED:")
        L += [f"  - {i}" for i in v["issues"]]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stamp", action="store_true",
                    help="write a provenance block after a successful verify")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()

    v = verify(a.results)
    print(render(v))

    if a.stamp:
        if not v["ok"]:
            print("\nrefusing to stamp: verification failed")
            raise SystemExit(1)
        from agent import provenance
        with open(a.results) as fh:
            rec = json.load(fh)
        p = provenance.stamp(
            config=rec.get("config"),
            seeds=list(range(v["k"])),
            code_paths=["agent/final_ensemble.py", "agent/ensemble.py",
                        "runtime/train_lib.py"],
            evaluation="kuairand-starter-kit/evaluate.py on the valid split",
            extra={"aggregation": v["aggregation"],
                   "members_dir": rec.get("members_dir"),
                   "reproduce": rec.get("reproduce"),
                   "verified": "recomputed from stored member predictions; "
                               "metrics matched the reported values exactly"})
        rec["provenance"] = p
        tmp = a.results + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=2)
        os.replace(tmp, a.results)
        print("\nstamped provenance:")
        print(provenance.render(p))

    raise SystemExit(0 if v["ok"] else 1)


if __name__ == "__main__":
    main()
