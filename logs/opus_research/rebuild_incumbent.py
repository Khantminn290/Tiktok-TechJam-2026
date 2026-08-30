"""End-to-end rebuild of the submitted ensemble from the current repository.

Audit finding 7: the incumbent is verified at the PREDICTION-ARTIFACT level --
the stored member arrays recombine to 0.60541 exactly -- but that only proves
the arithmetic, not that the current code still produces those members. This
retrains all 16 members from scratch and compares.

Writes to a SCRATCH directory. It never touches logs/final_ensemble, because a
verification step that overwrites the thing it verifies is not a verification.
"""
import json
import os
import sys
import time

ROOT = "/Users/khantminn/Desktop/Tiktok-TechJam-2026"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rebuild")
for p in (ROOT, os.path.join(ROOT, "runtime"), os.path.join(ROOT, "kuairand-starter-kit")):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402
from agent.ensemble import rank_normalise  # noqa: E402
from agent.executor import run_solution  # noqa: E402
from evaluate import evaluate  # noqa: E402
import train_lib  # noqa: E402

os.makedirs(OUT, exist_ok=True)
rec = json.load(open(os.path.join(ROOT, "logs", "ensemble_results.json")))
choices, k = rec["config"], rec["k"]
print(f"rebuilding {k} members of the submitted config", flush=True)
print(f"target: primary {rec['primary']} GAUC {rec['GAUC']} nDCG@5 {rec['nDCG@5']}",
      flush=True)

with open(os.path.join(ROOT, "runtime", "seed_solution.py")) as fh:
    src = fh.read()

scores, singles = [], []
splits, meta = train_lib.load_cache()
va = splits["valid"]
for s in range(k):
    d = os.path.join(OUT, f"seed_{s:02d}")
    m_path = os.path.join(d, "metrics.json")
    sv_path = os.path.join(d, "scores_valid.npy")
    if os.path.exists(sv_path):
        sv = np.load(sv_path)
    else:
        t0 = time.time()
        res = run_solution(src, os.path.join(OUT, f"seed_{s:02d}.py"),
                           choices, d, timeout_s=1800, seed=s)
        if not res.ok:
            print(f"  seed {s} FAILED: {(res.error_trace or '')[:150]}", flush=True)
            continue
        sv = np.load(sv_path)
        print(f"  seed {s}  primary {res.metrics['primary']:.5f}  "
              f"{time.time() - t0:.0f}s", flush=True)
    scores.append(sv)
    if os.path.exists(m_path):
        singles.append(json.load(open(m_path))["primary"])

print(f"\nrebuilt {len(scores)}/{k} members", flush=True)
agg = np.mean([rank_normalise(x) for x in scores], axis=0)
got = evaluate(list(va["user_raw"]), va["long_view"], agg)
out = {
    "rebuilt_members": len(scores),
    "rebuilt": {kk: round(float(got[kk]), 5) for kk in ("primary", "GAUC", "nDCG@5")},
    "recorded": {kk: rec[kk] for kk in ("primary", "GAUC", "nDCG@5")},
    "member_mean": round(sum(singles) / len(singles), 5) if singles else None,
    "aggregation": "rank_normalise_then_mean",
}
out["exact_match"] = all(out["rebuilt"][kk] == out["recorded"][kk]
                         for kk in ("primary", "GAUC", "nDCG@5"))
out["primary_delta"] = round(out["rebuilt"]["primary"] - out["recorded"]["primary"], 6)
print(json.dumps(out, indent=2), flush=True)
json.dump(out, open(os.path.join(OUT, "rebuild_result.json"), "w"), indent=2)
