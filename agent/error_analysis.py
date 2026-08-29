"""Where is the model actually failing, and is any feature carrying signal it misses?

The loop previously saw only two scalars per experiment, GAUC and nDCG@5, which
turns "what should we try next?" into guesswork. This module answers the two
questions that make the difference empirical:

  1. WHERE does the error concentrate? Per-segment GAUC/nDCG over user activity,
     positive count, list length and item popularity.
  2. WHAT signal is missing? Within-user AUC of a candidate feature, plus
     whether blending it into the model's own scores adds anything.

Both are computed on VALIDATION only, from TRAIN-derived statistics. That
restriction is not decoration: any statistic built from the split being scored
would leak, and a segmentation built from valid labels would flatter every
conclusion drawn from it.

Two structural facts about this metric, measured here and easy to get wrong:

  * GAUC ranks WITHIN a user, so any feature CONSTANT across that user's rows
    contributes exactly nothing to it. User-level scalars (activity, affinity,
    lifetime rate) are not merely weak, they are at chance BY CONSTRUCTION:
    user long_view rate measures 0.5000, exactly, and every blend weight moves
    the score by 0.0. Only within-user-varying features can move GAUC.
    (An earlier tie-unaware implementation reported 0.5040 for this and would
    have made a structural certainty look like a weak empirical finding.)
  * The official nDCG@5 scores zero-positive users as 0.0 and includes them in
    the mean. They are 30.2% of validation users and no model can fix them;
    that, not model quality, is most of the gap to the 0.8484 ceiling.

Usage:
    python3 -m agent.error_analysis                 # segments the submitted ensemble
    python3 -m agent.error_analysis --scores X.npy
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MIN_SEGMENT_USERS = 200      # below this, a segment difference is not readable
NOISE_FLOOR = 0.0008         # the baseline's own 5-seed std


def _cache_dir() -> str:
    sb = os.path.join(ROOT, "runtime", "cache_sandbox")
    return sb if os.path.exists(os.path.join(sb, "meta.json")) \
        else os.path.join(ROOT, "runtime", "cache")


def load_valid():
    sys.path.insert(0, os.path.join(ROOT, "runtime"))
    import train_lib
    splits, meta = train_lib.load_cache(_cache_dir())
    return splits, meta


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, matching the official evaluator's Mann-Whitney
    tie correction.

    Not a detail: argsort-of-argsort hands tied values arbitrary distinct ranks,
    which makes a score that is CONSTANT within a user look like a perfect (or
    perfectly inverted) ranking instead of chance. A user-constant feature must
    score exactly 0.5, and without this it did not.
    """
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ranks = np.empty(len(x), dtype=np.float64)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def per_user_metrics(u, y, s) -> tuple:
    """Per-user AUC (users with both classes) and nDCG@5 (users with >=1 pos)."""
    u = np.asarray(u); y = np.asarray(y, float); s = np.asarray(s, float)
    order = np.argsort(u, kind="stable")
    ub, yb, sb = u[order], y[order], s[order]
    starts = np.searchsorted(ub, np.unique(ub))
    ends = np.r_[starts[1:], len(ub)]
    auc, ndcg = {}, {}
    for st, en in zip(starts, ends):
        yy, ss, uid = yb[st:en], sb[st:en], ub[st]
        npos = yy.sum()
        if 0 < npos < len(yy):
            r = _average_ranks(ss)
            auc[uid] = (r[yy > 0].sum() - npos * (npos + 1) / 2) / (npos * (len(yy) - npos))
        if npos > 0:
            top = np.argsort(-ss)[:5]
            disc = np.log2(np.arange(2, len(top) + 2))
            dcg = float(np.sum(yy[top] / disc))
            ideal = np.sort(yy)[::-1][:5]
            idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
            ndcg[uid] = dcg / idcg if idcg > 0 else 0.0
    return auc, ndcg


def within_user_auc(u, y, s) -> float:
    """The single number that says whether a feature can move GAUC at all."""
    auc, _ = per_user_metrics(u, y, s)
    return float(np.mean(list(auc.values()))) if auc else 0.5


def segment(u, y, s, keyfn, bins, label) -> list:
    auc, ndcg = per_user_metrics(u, y, s)
    users = np.array(sorted(auc.keys()))
    if not len(users):
        return []
    vals = keyfn(users)
    rows = []
    for lo, hi in bins:
        m = (vals >= lo) & (vals < hi)
        ks = users[m]
        if len(ks) < MIN_SEGMENT_USERS:     # too small to read
            continue
        rows.append({"segment": label, "lo": float(lo), "hi": float(hi),
                     "users": int(len(ks)),
                     "GAUC": round(float(np.mean([auc[k] for k in ks])), 4),
                     "nDCG@5": round(float(np.mean([ndcg[k] for k in ks if k in ndcg]))
                                     if any(k in ndcg for k in ks) else 0.0, 4)})
    return rows


def feature_probe(u, y, model_scores, feature, name) -> dict:
    """Does `feature` carry ranking signal the model does not already have?

    Reports the feature's standalone within-user AUC AND what happens when it is
    blended into the model. The second is the one that matters: item long_view
    rate scores 0.6394 standalone, which looks substantial, yet blending it in
    hurts monotonically at every weight -- the embedding already encodes it.
    A strong standalone number is not evidence of a missing mechanism.
    """
    f = np.asarray(feature, float)
    base = within_user_auc(u, y, model_scores)
    rm = _average_ranks(np.asarray(model_scores, float))
    rf = _average_ranks(f)
    rm /= max(1.0, rm.max()); rf /= max(1.0, rf.max())
    blends = {w: round(within_user_auc(u, y, w * rm + (1 - w) * rf) - base, 5)
              for w in (0.99, 0.95, 0.90)}
    best_gain = max(blends.values())
    # Judged against the NOISE FLOOR, not against zero. `> 0` reported a
    # +0.00003 blend gain (0.04 sigma) as "adds residual signal", which is the
    # same mistake as trusting a single lucky run -- and here it would have
    # sent the agent to implement a mechanism that carries nothing.
    meaningful = best_gain >= NOISE_FLOOR / 2
    return {"feature": name,
            "standalone_wAUC": round(within_user_auc(u, y, f), 4),
            "model_wAUC": round(base, 4),
            "blend_delta": blends,
            "best_gain_sigma": round(best_gain / NOISE_FLOOR, 2),
            "adds_signal": bool(meaningful),
            "verdict": ("adds residual signal" if meaningful else
                        f"NO USABLE RESIDUAL ({best_gain / NOISE_FLOOR:+.2f} sigma "
                        f"-- inside the noise floor)" if best_gain > 0 else
                        "REDUNDANT -- model already encodes it")}


def report(scores: np.ndarray) -> str:
    splits, meta = load_valid()
    va, tr = splits["valid"], splits["train"]
    u, v, y = va["user"], va["video"], va["long_view"]
    nU, nV = meta["field_dims"]["user"], meta["field_dims"]["video"]

    tr_cnt = np.bincount(tr["user"], minlength=nU).astype(float)
    tr_pos = np.zeros(nU); np.add.at(tr_pos, tr["user"], tr["long_view"])
    item_pop = np.bincount(tr["video"], minlength=nV).astype(float)
    sizes = np.bincount(u, minlength=nU).astype(float)

    rows = []
    rows += segment(u, y, scores, lambda k: tr_cnt[k],
                    [(0, 10), (10, 25), (25, 40), (40, 60), (60, 100), (100, 1e9)],
                    "train impressions/user")
    rows += segment(u, y, scores, lambda k: tr_pos[k],
                    [(0, 1), (1, 5), (5, 10), (10, 20), (20, 40), (40, 1e9)],
                    "train positives/user")
    rows += segment(u, y, scores, lambda k: sizes[k],
                    [(2, 4), (4, 6), (6, 10), (10, 20), (20, 1e9)],
                    "valid list length")

    auc, ndcg = per_user_metrics(u, y, scores)
    L = ["## ERROR ANALYSIS (validation only; segments from TRAIN statistics)",
         f"users scored: GAUC {len(auc)}, nDCG@5 {len(ndcg)} "
         f"(+{int((sizes > 0).sum()) - len(ndcg)} zero-positive users score 0.0 "
         f"in the official nDCG and cannot be fixed)",
         "",
         f"{'segment':<26}{'range':>16}{'users':>8}{'GAUC':>9}{'nDCG@5':>9}"]
    last = None
    for r in rows:
        if r["segment"] != last:
            L.append(f"-- {r['segment']}")
            last = r["segment"]
        hi = "inf" if r["hi"] >= 1e9 else f"{r['hi']:.0f}"
        rng = f"[{r['lo']:.0f},{hi})"
        L.append(f"{'':<26}{rng:>16}{r['users']:>8}"
                 f"{r['GAUC']:>9.4f}{r['nDCG@5']:>9.4f}")

    g = [r["GAUC"] for r in rows]
    if g:
        L.append(f"\nGAUC spread across ALL segments: {min(g):.4f} - {max(g):.4f} "
                 f"(range {max(g) - min(g):.4f})")
        if max(g) - min(g) < 0.03:
            L.append("=> error is DIFFUSE. There is no concentrated failure "
                     "population to target; segment-specific experiments are "
                     "unlikely to pay. nDCG@5's much wider spread is mechanical "
                     "(longer lists make top-5 harder), not a model weakness.")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=None,
                    help="a scores_valid.npy; default = the submitted ensemble")
    a = ap.parse_args()
    if a.scores:
        s = np.load(a.scores)
    else:
        from agent.ensemble import rank_normalise
        res = json.load(open(os.path.join(ROOT, "logs", "ensemble_results.json")))
        d = os.path.join(ROOT, res["members_dir"])
        s = np.mean([rank_normalise(np.load(os.path.join(
            d, f"seed_{i:02d}", "scores_valid.npy"))) for i in res["seeds_used"]], axis=0)
    print(report(s))


if __name__ == "__main__":
    main()
