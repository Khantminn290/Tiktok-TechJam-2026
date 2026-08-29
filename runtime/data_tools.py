"""Read-only data-inspection tools the AGENT may call before hypothesizing.

Motivation, measured: every real diagnostic in this project came from a human
running measurements (the train/eval list-length mismatch that sank
LambdaRank, the play-time outlier tail, the recency_weighted_pool/category
confound). The agent could not reason about data it could not see -- observe()
only ever read the journal and experience.md. These tools close that gap.

SAFETY MODEL (mirrors the Phase 1 data boundary; verified, not assumed):
  * Everything reads the SANDBOXED cache, never the real one. The sandbox's
    test.npz has long_view/is_click/is_like/is_forward/play_time_ms physically
    removed by runtime/data_boundary.redact_test_columns.
  * SPLITS is restricted to train/valid at this layer too -- belt and braces.
    Even if a caller names "test", it is rejected before any array is touched.
  * FEATURES is a fixed allowlist. Arbitrary column names, dunder attributes
    and path-like strings cannot reach the arrays.
  * Every function returns a small JSON-serialisable summary. No function
    returns raw rows, so this cannot become a data-exfiltration channel that
    reconstructs labels row by row.
  * No function writes anything.
"""
from __future__ import annotations

import collections
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Feature allowlist. Deliberately excludes user_raw (identity strings) and
# anything not already used as a modelling signal.
FEATURES = ("duration_ms", "hourmin", "date", "time_ms", "long_view",
            "is_click", "is_like", "is_forward", "play_time_ms",
            "user", "video", "author", "tab")

# Columns observed only AFTER the impression happened. They are legitimate
# TRAINING TARGETS (that is what the multitask axis uses) but must NEVER be
# used as INPUT features at scoring time -- at ranking time they do not exist.
# This matters because the tools report them as spectacular predictors: a live
# measurement gives play_time_ms a within-user AUC of 0.992 and is_click 0.880,
# versus the FM baseline's 0.670. An agent shown those numbers without this
# warning could reasonably conclude they are the best features available and
# propose catastrophic leakage. Flagged loudly in every result instead.
POST_HOC_OUTCOMES = ("long_view", "is_click", "is_like", "is_forward",
                     "play_time_ms")
_LEAK_WARNING = (
    "*** POST-HOC OUTCOME -- NOT USABLE AS AN INPUT FEATURE. This column is "
    "recorded AFTER the user has already watched, so it does not exist at "
    "ranking time. A high score here is NOT an opportunity; using it as a "
    "model input would be label leakage and the result would be invalid. It "
    "may legitimately be used only as a TRAINING TARGET (see the multitask "
    "axis). ***")
LABEL = "long_view"
ALLOWED_SPLITS = ("train", "valid")     # never "test"
MAX_BINS = 20


class ToolError(ValueError):
    """Invalid tool request. Message is safe to show the LLM."""


def _load(cache_dir):
    import train_lib
    splits, meta = train_lib.load_cache(cache_dir) if cache_dir else train_lib.load_cache()
    return splits, meta


def _leak_note(feature):
    return {"WARNING": _LEAK_WARNING} if feature in POST_HOC_OUTCOMES else {}


def _check(feature, split):
    if feature not in FEATURES:
        raise ToolError(f"unknown feature {feature!r}; allowed: {list(FEATURES)}")
    if split not in ALLOWED_SPLITS:
        raise ToolError(f"split {split!r} is not inspectable; allowed: "
                        f"{list(ALLOWED_SPLITS)} (the test split is never readable)")


def get_feature_stats(feature: str, split: str = "train", cache_dir=None) -> dict:
    """Distribution summary for one feature: coverage, spread, percentiles."""
    _check(feature, split)
    splits, _ = _load(cache_dir)
    x = np.asarray(splits[split][feature], dtype=np.float64)
    return {**_leak_note(feature), "tool": "get_feature_stats", "feature": feature, "split": split,
            "n": int(len(x)), "distinct": int(len(np.unique(x))),
            "mean": round(float(x.mean()), 6), "std": round(float(x.std()), 6),
            "min": round(float(x.min()), 4),
            "p25": round(float(np.percentile(x, 25)), 4),
            "median": round(float(np.median(x)), 4),
            "p75": round(float(np.percentile(x, 75)), 4),
            "p90": round(float(np.percentile(x, 90)), 4),
            "p99": round(float(np.percentile(x, 99)), 4),
            "max": round(float(x.max()), 4),
            "zero_frac": round(float((x == 0).mean()), 4)}


def get_label_rate_by_segment(feature: str, n_bins: int = 10,
                              split: str = "train", cache_dir=None) -> dict:
    """long_view rate across quantile bins of a feature -- shows whether the
    feature separates the label at all, and monotonically or not."""
    _check(feature, split)
    n_bins = max(2, min(int(n_bins), MAX_BINS))
    splits, _ = _load(cache_dir)
    x = np.asarray(splits[split][feature], dtype=np.float64)
    y = np.asarray(splits[split][LABEL], dtype=np.float64)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(qs, x, side="right") - 1, 0, len(qs) - 2)
    out = []
    for b in range(len(qs) - 1):
        m = idx == b
        if m.sum() == 0:
            continue
        out.append({"bin": b, "lo": round(float(qs[b]), 4),
                    "hi": round(float(qs[b + 1]), 4), "n": int(m.sum()),
                    "label_rate": round(float(y[m].mean()), 4)})
    rates = [o["label_rate"] for o in out]
    return {**_leak_note(feature), "tool": "get_label_rate_by_segment", "feature": feature,
            "split": split, "overall_label_rate": round(float(y.mean()), 4),
            "bins": out,
            "spread": round(float(max(rates) - min(rates)), 4) if rates else 0.0,
            "note": "a large spread means the feature separates the label in "
                    "aggregate; it does NOT mean it helps WITHIN-user ranking "
                    "-- use get_within_user_auc for that."}


def get_within_user_auc(feature: str, split: str = "valid", cache_dir=None) -> dict:
    """The decisive diagnostic for this task: does the feature order items
    correctly WITHIN a user? GAUC/nDCG@5 read only within-user order, so a
    feature that separates the label in aggregate can still be worth exactly
    nothing here (measured repeatedly on this project: content categories
    looked strong in aggregate but scored 0.5228 within-user).

    0.5 == no signal. Reported alongside the same statistic computed against a
    user-mean-centred version, which removes the constant-within-user
    component that cannot affect ranking.
    """
    _check(feature, split)
    splits, _ = _load(cache_dir)
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "kuairand-starter-kit"))
    from evaluate import auc
    x = np.asarray(splits[split][feature], dtype=np.float64)
    y = np.asarray(splits[split][LABEL], dtype=np.float64)
    u = np.asarray(splits[split]["user"])
    byu = collections.defaultdict(list)
    for uu, xx, yy in zip(u, x, y):
        byu[int(uu)].append((xx, yy))
    raw, centred = [], []
    for rows in byu.values():
        labs = [r[1] for r in rows]
        if not (0 < sum(labs) < len(labs)):
            continue
        vals = [r[0] for r in rows]
        m = sum(vals) / len(vals)
        raw.append(auc(labs, vals))
        centred.append(auc(labs, [v - m for v in vals]))
    if not raw:
        raise ToolError("no users with both a positive and a negative label")
    return {**_leak_note(feature), "tool": "get_within_user_auc", "feature": feature, "split": split,
            "n_users_scored": len(raw),
            "within_user_auc": round(float(np.mean(raw)), 4),
            "within_user_auc_user_centred": round(float(np.mean(centred)), 4),
            "note": "0.5 = no within-user signal. For scale: the FM baseline's "
                    "GAUC is about 0.670. A feature near 0.5 will not help "
                    "ranking no matter how strong it looks in aggregate."}


def get_user_history_stats(split: str = "train", cache_dir=None) -> dict:
    """Per-user impression/positive counts, and how the TRAIN list-length
    distribution compares with the EVAL one -- the mismatch that was measured
    to sink a position-discounted loss (train averages ~43.5 impressions/user,
    valid ~5.6)."""
    splits, _ = _load(cache_dir)
    out = {"tool": "get_user_history_stats"}
    for sp in ALLOWED_SPLITS:
        u = np.asarray(splits[sp]["user"])
        y = np.asarray(splits[sp][LABEL], dtype=np.float64)
        n = int(u.max()) + 1
        cnt = np.bincount(u, minlength=n).astype(float)
        pos = np.bincount(u, weights=y, minlength=n)
        act = cnt > 0
        out[sp] = {"rows": int(len(u)), "users": int(act.sum()),
                   "impressions_per_user_mean": round(float(cnt[act].mean()), 2),
                   "impressions_per_user_median": int(np.median(cnt[act])),
                   "impressions_per_user_p90": int(np.percentile(cnt[act], 90)),
                   "impressions_per_user_max": int(cnt[act].max()),
                   "positives_per_user_mean": round(float(pos[act].mean()), 2),
                   "users_with_zero_positives_frac": round(float((pos[act] == 0).mean()), 4)}
    a = out["train"]["impressions_per_user_mean"]
    b = out["valid"]["impressions_per_user_mean"]
    out["train_vs_valid_list_length_ratio"] = round(a / b, 2) if b else None
    out["note"] = ("A large train/valid ratio means any loss or feature that "
                   "truncates or discounts by RANK POSITION behaves very "
                   "differently at train time than at eval time.")
    return out


TOOLS = {"get_feature_stats": get_feature_stats,
         "get_label_rate_by_segment": get_label_rate_by_segment,
         "get_within_user_auc": get_within_user_auc,
         "get_user_history_stats": get_user_history_stats}


def describe_tools() -> str:
    return (
        "get_feature_stats(feature)                    -> distribution summary\n"
        "get_label_rate_by_segment(feature, n_bins)    -> label rate per quantile bin\n"
        "get_within_user_auc(feature)                  -> DECISIVE: does it order WITHIN a user?\n"
        "get_user_history_stats()                      -> per-user counts, train-vs-valid list lengths\n"
        f"allowed features: {list(FEATURES)}\n"
        f"POST-HOC OUTCOMES (targets only, NEVER inputs): {list(POST_HOC_OUTCOMES)}\n"
        "  -- these score extremely well because they encode the answer; they do\n"
        "     not exist at ranking time and using them as inputs is leakage.\n"
        "All read-only, train/valid only; the test split is never inspectable.")


def run_tool(name: str, args: dict, cache_dir=None) -> dict:
    """Dispatch one validated tool call. Raises ToolError on anything invalid."""
    if name not in TOOLS:
        raise ToolError(f"unknown tool {name!r}; allowed: {list(TOOLS)}")
    args = dict(args or {})
    for k in list(args):
        if k not in ("feature", "n_bins", "split"):
            raise ToolError(f"tool {name!r} got unexpected argument {k!r}")
    if cache_dir is not None:
        args["cache_dir"] = cache_dir
    try:
        return TOOLS[name](**args)
    except ToolError:
        raise
    except TypeError as e:
        raise ToolError(f"bad arguments for {name!r}: {e}") from e
