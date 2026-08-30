"""Prediction-level analysis and controlled ensemble search.

Ensembling is the single largest validated win on this project (+0.00074
primary, variance halved), but every ensemble so far averaged seeds of the
SAME configuration -- members differing only by initialisation. Heterogeneous
ensembling across genuinely different configurations has never been tested,
and theory says gain scales with member DECORRELATION, so it is the most
promising untested score lever.

Two measured lessons from this project constrain the design:

  * Members must be BOTH independent and comparably good. Snapshot ensembling
    failed with quality-but-no-independence; bagging failed with
    independence-but-no-quality (-0.0028 per member sank a +0.059 correlation
    gain). Selection here therefore screens on BOTH.
  * Rank-normalise before averaging. Different configurations produce scores
    on different scales; averaging raw values lets whichever model has the
    widest spread dominate.

Selection is greedy forward search on validation, which is deliberately
modest: exhaustively searching subsets of validation-scored models is an
excellent way to overfit validation, and the run already has 50-iteration and
noise-floor constraints that make tiny gains meaningless.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
NOISE_FLOOR = 0.0008        # official baseline's own 5-seed std


def rank_normalise(x: np.ndarray) -> np.ndarray:
    """Scale-free transform: preserves each model's ordering, which is all the
    metric reads, and prevents a wide-spread model from dominating the mean."""
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[o] = np.arange(len(x), dtype=np.float64)
    return r / max(1, len(x) - 1)


def _evaluator():
    sys.path.insert(0, os.path.join(ROOT, "kuairand-starter-kit"))
    sys.path.insert(0, os.path.join(ROOT, "runtime"))
    from evaluate import evaluate
    return evaluate


def load_valid_targets(cache_dir: str | None = None):
    """Users/labels for the validation split. Prefers the SANDBOX cache, which
    is never locked while a training subprocess is running."""
    sys.path.insert(0, os.path.join(ROOT, "runtime"))
    sandbox = os.path.join(ROOT, "runtime", "cache_sandbox")
    real = os.path.join(ROOT, "runtime", "cache")
    for d in ([cache_dir] if cache_dir else []) + [sandbox, real]:
        p = os.path.join(d or "", "valid.npz")
        if d and os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            return list(z["user_raw"]), z["long_view"]
    raise FileNotFoundError("no readable validation cache found")


class Candidate:
    def __init__(self, name, scores, primary, meta=None):
        self.name, self.scores, self.primary = name, scores, primary
        self.meta = meta or {}
        self.ranked = rank_normalise(scores)


def collect_candidates(paths_and_names, users, labels, min_primary=0.0) -> list:
    """Score every candidate array and keep those clearing a floor."""
    ev = _evaluator()
    out = []
    for name, path in paths_and_names:
        if not os.path.exists(path):
            continue
        s = np.load(path)
        if len(s) != len(labels):
            continue
        p = float(ev(users, labels, s)["primary"])
        if p >= min_primary:
            out.append(Candidate(name, s, p))
    return sorted(out, key=lambda c: -c.primary)


def diversity_matrix(cands: list) -> dict:
    """Pairwise rank correlation. Low correlation between two strong models is
    where ensemble value lives."""
    n = len(cands)
    m = np.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        c = float(np.corrcoef(cands[i].ranked, cands[j].ranked)[0, 1])
        m[i, j] = m[j, i] = c
    return {"names": [c.name for c in cands], "matrix": m.round(4).tolist(),
            "mean_offdiag": round(float(m[~np.eye(n, dtype=bool)].mean()), 4)
            if n > 1 else 1.0}


def ensemble_value(a: Candidate, b: Candidate) -> dict:
    """Heuristic screen: strong AND decorrelated is worth testing; strong AND
    near-identical is not."""
    corr = float(np.corrcoef(a.ranked, b.ranked)[0, 1])
    gap = abs(a.primary - b.primary)
    if corr > 0.99:
        verdict = "LOW (near-duplicate predictions)"
    elif gap > 4 * NOISE_FLOOR:
        verdict = "LOW (one member is materially weaker)"
    elif corr < 0.95:
        verdict = "HIGH (comparable strength, decorrelated)"
    else:
        verdict = "MEDIUM"
    return {"pair": (a.name, b.name), "corr": round(corr, 4),
            "primary_gap": round(gap, 5), "value": verdict}


def greedy_forward(cands: list, users, labels, max_k: int = 6,
                   min_gain: float = NOISE_FLOOR / 2) -> dict:
    """Greedily add the member that most improves validation primary.

    Stops when no addition gains more than half the noise floor -- an ensemble
    that improves by less than that is indistinguishable from luck, and
    chasing it is how validation gets overfitted.
    """
    ev = _evaluator()
    chosen, cur = [], None
    best_primary = -1.0
    history = []
    for _ in range(min(max_k, len(cands))):
        best_add, best_score = None, best_primary
        for c in cands:
            if c in chosen:
                continue
            trial = (c.ranked if cur is None
                     else (cur * len(chosen) + c.ranked) / (len(chosen) + 1))
            p = float(ev(users, labels, trial)["primary"])
            if p > best_score:
                best_score, best_add = p, c
        if best_add is None or (best_primary > 0 and
                                best_score - best_primary < min_gain):
            break
        cur = (best_add.ranked if cur is None
               else (cur * len(chosen) + best_add.ranked) / (len(chosen) + 1))
        chosen.append(best_add)
        gain = best_score - best_primary if best_primary > 0 else 0.0
        best_primary = best_score
        history.append({"added": best_add.name, "primary": round(best_primary, 5),
                        "gain": round(gain, 5)})
    return {"members": [c.name for c in chosen], "k": len(chosen),
            "primary": round(best_primary, 5), "history": history,
            "scores": cur}


def report(cands: list, users, labels, max_k: int = 6) -> str:
    if not cands:
        return "no ensemble candidates found"
    L = ["=" * 70, "PREDICTION ANALYSIS / ENSEMBLE SEARCH", "=" * 70,
         f"candidates: {len(cands)}"]
    for c in cands[:8]:
        L.append(f"  {c.name:38s} primary {c.primary:.5f}")
    d = diversity_matrix(cands[:8])
    L.append(f"\nmean pairwise rank-correlation: {d['mean_offdiag']}")
    L.append("pair screening (strong AND decorrelated = worth testing):")
    for a, b in itertools.combinations(cands[:5], 2):
        v = ensemble_value(a, b)
        L.append(f"  {v['pair'][0][:22]:22s} + {v['pair'][1][:22]:22s} "
                 f"corr {v['corr']:.3f}  {v['value']}")
    g = greedy_forward(cands, users, labels, max_k=max_k)
    L.append(f"\ngreedy forward selection -> k={g['k']}, primary {g['primary']:.5f}")
    for h in g["history"]:
        L.append(f"  + {h['added'][:34]:34s} -> {h['primary']:.5f} "
                 f"(gain {h['gain']:+.5f})")
    best_single = cands[0].primary
    L.append(f"\nbest single member : {best_single:.5f}")
    L.append(f"ensemble           : {g['primary']:.5f} "
             f"({g['primary'] - best_single:+.5f}, "
             f"{(g['primary'] - best_single) / NOISE_FLOOR:+.2f} sigma)")
    if g["primary"] - best_single < NOISE_FLOOR / 2:
        L.append("VERDICT: gain is inside the noise floor -- do NOT adopt.")
    else:
        L.append("VERDICT: gain exceeds the noise floor; confirm before adopting.")
    return "\n".join(L)
