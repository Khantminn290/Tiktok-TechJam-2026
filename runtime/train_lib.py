"""Shared training engine for generated solutions (KuaiRand-Pure).

Design notes
- Row order everywhere matches data.load(): file 4_08_to_4_21 then 4_22_to_5_08,
  date-filtered, original file order preserved — so scores_{valid,test}.npy written
  by index are row_id-aligned with submit.py.
- Scoring is delegated to the starter kit's evaluate.py (never reimplemented).
- Hidden-test discipline: this module computes test-split *scores* but never
  extracts test outcomes into runtime arrays or evaluates them; only the
  organizer-side harness may evaluate test at the end.
- The pooled-history feature is recomputed from current embeddings each epoch but
  treated as stop-gradient in the numpy engine (a feature, not a trained path);
  the torch DIN path trains attention end-to-end.

Env: KUAIRAND_KIT = path to kuairand-starter-kit (defaults to sibling of this file's
parent). Data dir defaults to $KUAIRAND_KIT/KuaiRand-Pure/data.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
KIT_DIR = os.environ.get(
    "KUAIRAND_KIT", os.path.join(os.path.dirname(_HERE), "kuairand-starter-kit"))
if KIT_DIR not in sys.path:
    sys.path.insert(0, KIT_DIR)

from evaluate import evaluate  # noqa: E402  (the official scorer — do not reimplement)

DATA_DIR = os.environ.get("KUAIRAND_DATA", os.path.join(KIT_DIR, "KuaiRand-Pure", "data"))
CACHE_DIR = os.environ.get("KUAIRAND_CACHE", os.path.join(_HERE, "cache"))
SPLITS = {"train": (20220408, 20220421),
          "valid": (20220422, 20220428),
          "test": (20220429, 20220508)}
LOG_FILES = ("log_standard_4_08_to_4_21_pure.csv", "log_standard_4_22_to_5_08_pure.csv")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def rank_normalize(scores):
    """Tie-aware, scale-free midranks in [0,1] for fixed ensembles."""
    scores = np.asarray(scores).reshape(-1)
    _, inverse, counts = np.unique(
        scores, return_inverse=True, return_counts=True)
    starts = np.cumsum(np.r_[0, counts[:-1]])
    midranks = starts + (counts - 1) / 2.0
    return midranks[inverse].astype(np.float64) / max(1, len(scores) - 1)


def batch_repeat_fatigue_scores(base_scores: np.ndarray, split: dict,
                                weight: float = 0.10) -> tuple[np.ndarray, dict]:
    """Label-free batch rerank that penalizes repeated user-video exposure.

    The complete evaluation feature batch is available when producing the
    required row-aligned submission.  Repetition counts use true `user_raw`
    identities (never the train-vocabulary UNK bucket) and no outcome column.
    Per-user standardization keeps the fixed coefficient comparable across
    models without changing the base ranking by itself.
    """
    scores = np.asarray(base_scores, dtype=np.float64).reshape(-1)
    users_raw = np.asarray(split["user_raw"])
    videos_raw = np.asarray(split["video_raw"])
    if len(scores) != len(users_raw) or len(scores) != len(videos_raw):
        raise ValueError("batch repeat rerank arrays are not row-aligned")

    _, users = np.unique(users_raw, return_inverse=True)
    users = users.astype(np.int64, copy=False)
    n_users = int(users.max()) + 1 if len(users) else 0
    _, videos = np.unique(videos_raw, return_inverse=True)
    videos = videos.astype(np.int64, copy=False)

    def user_z(values):
        values = np.asarray(values, dtype=np.float64)
        count = np.bincount(users, minlength=n_users).astype(np.float64)
        total = np.bincount(users, weights=values, minlength=n_users)
        mean = total / np.maximum(count, 1.0)
        centered = values - mean[users]
        variance = np.bincount(
            users, weights=centered * centered, minlength=n_users)
        std = np.sqrt(variance / np.maximum(count, 1.0))
        return centered / np.maximum(std[users], 1e-12)

    video_dim = int(videos.max()) + 1 if len(videos) else 1
    pair = users * video_dim + videos
    _, pair_code = np.unique(pair, return_inverse=True)
    pair_count = np.bincount(pair_code)
    repeats_excluding_self = pair_count[pair_code] - 1
    fatigue = np.log1p(repeats_excluding_self.astype(np.float64))
    adjusted = user_z(scores) - float(weight) * user_z(fatigue)
    info = {
        "method": "label_free_batch_repeat_fatigue",
        "weight": float(weight),
        "identity": "user_raw_x_video_raw",
        "uses_outcome_columns": False,
        "repeated_row_fraction": float(np.mean(repeats_excluding_self > 0)),
        "affected_user_fraction": float(
            len(np.unique(users[repeats_excluding_self > 0])) / max(n_users, 1)),
    }
    return adjusted, info


def bayesian_prior_scores(splits: dict, mode: str) -> dict:
    """Train-only smoothed item/author long-view logits for valid/test ranking."""
    if mode == "none":
        return {name: np.zeros(len(s["user"]), dtype=np.float32)
                for name, s in splits.items()}
    if mode not in ("bayesian_item_author", "recency_bayesian_item_author"):
        raise ValueError(f"unknown score_prior mode: {mode}")

    tr = splits["train"]
    y = tr["long_view"].astype(np.float64)
    if mode == "recency_bayesian_item_author":
        age = tr["time_ms"].max() - tr["time_ms"].astype(np.float64)
        w = np.exp(-age / (3.0 * 86400e3))
    else:
        w = np.ones(len(y), dtype=np.float64)
    global_rate = float(np.sum(w * y) / np.maximum(np.sum(w), 1.0))
    global_rate = float(np.clip(global_rate, 1e-5, 1.0 - 1e-5))
    global_logit = np.log(global_rate / (1.0 - global_rate))

    def table(field: str, strength: float) -> np.ndarray:
        n = int(max(np.max(s[field]) for s in splits.values())) + 1
        count = np.zeros(n, dtype=np.float64)
        positive = np.zeros(n, dtype=np.float64)
        np.add.at(count, tr[field], w)
        np.add.at(positive, tr[field], w * y)
        rate = (positive + strength * global_rate) / (count + strength)
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        return np.log(rate / (1.0 - rate)) - global_logit

    video = table("video", 20.0)
    author = table("author", 100.0)
    return {name: (0.7 * video[s["video"]] + 0.3 * author[s["author"]]).astype(np.float32)
            for name, s in splits.items()}


def oof_recency_bayesian_prior_scores(splits: dict) -> tuple[dict, dict]:
    """Recency item/author prior with leave-one-date-out TRAIN scores.

    Validation and test priors are fit on the complete official training window.
    A training row receives a table fit on every *other* training date, excluding
    its own label and all same-day correlated outcomes.  The fixed prior can then
    be used as an offset inside BPR without leaking the target into its margin.
    """
    tr = splits["train"]
    y = tr["long_view"].astype(np.float64)
    age = tr["time_ms"].max() - tr["time_ms"].astype(np.float64)
    w = np.exp(-age / (3.0 * 86400e3))
    dates, date_code = np.unique(tr["date"], return_inverse=True)
    n_dates = len(dates)
    total_w = float(np.sum(w))
    total_pos = float(np.sum(w * y))
    date_w = np.bincount(date_code, weights=w, minlength=n_dates)
    date_pos = np.bincount(date_code, weights=w * y, minlength=n_dates)

    def clipped_logit(rate):
        rate = np.clip(rate, 1e-5, 1.0 - 1e-5)
        return np.log(rate / (1.0 - rate))

    full_global = float(total_pos / max(total_w, 1.0))
    oof_global = ((total_pos - date_pos[date_code]) /
                  np.maximum(total_w - date_w[date_code], 1.0))

    def table(field: str, strength: float):
        n_entities = int(max(np.max(s[field]) for s in splits.values())) + 1
        count = np.bincount(tr[field], weights=w, minlength=n_entities)
        positive = np.bincount(tr[field], weights=w * y, minlength=n_entities)
        pair = tr[field].astype(np.int64) * n_dates + date_code
        count_by_date = np.bincount(
            pair, weights=w, minlength=n_entities * n_dates).reshape(n_entities, n_dates)
        pos_by_date = np.bincount(
            pair, weights=w * y,
            minlength=n_entities * n_dates).reshape(n_entities, n_dates)

        entity = tr[field]
        oof_count = count[entity] - count_by_date[entity, date_code]
        oof_pos = positive[entity] - pos_by_date[entity, date_code]
        oof_rate = ((oof_pos + strength * oof_global) /
                    (oof_count + strength))
        train_residual = clipped_logit(oof_rate) - clipped_logit(oof_global)

        full_rate = (positive + strength * full_global) / (count + strength)
        full_residual = clipped_logit(full_rate) - clipped_logit(full_global)
        query = {name: full_residual[s[field]] for name, s in splits.items()
                 if name != "train"}
        return train_residual, query

    video_train, video_query = table("video", 20.0)
    author_train, author_query = table("author", 100.0)
    scores = {
        "train": (0.7 * video_train + 0.3 * author_train).astype(np.float32),
    }
    for name in ("valid", "test"):
        scores[name] = (0.7 * video_query[name] +
                        0.3 * author_query[name]).astype(np.float32)
    info = {
        "method": "leave_one_date_out_recency_empirical_bayes",
        "dates": [int(d) for d in dates],
        "tau_days": 3.0,
        "video_strength": 20.0,
        "author_strength": 100.0,
        "video_weight": 0.7,
        "author_weight": 0.3,
        "offset_beta": 0.30,
        "label_sources": {"train": "long_view", "valid": None, "test": None},
    }
    return scores, info


GPU_DEVICES = ("cuda", "mps")


def select_device() -> str:
    """Device for the torch path. CPU unless a GPU is explicitly requested.

    KUAIRAND_DEVICE=auto picks cuda > mps > cpu. Default stays cpu because the
    reference pipeline needs no GPU — but when a GPU IS used, the time spent on it
    is measured and reported rather than assumed to be zero.
    """
    want = os.environ.get("KUAIRAND_DEVICE", "cpu").lower()
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if want == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if want == "cuda" and torch.cuda.is_available():
        return "cuda"
    if want == "mps" and getattr(torch.backends, "mps", None) \
            and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_resource_json(output_dir: str, device: str, train_seconds: float) -> None:
    """Record measured compute so the harness reports real GPU-hours."""
    with open(os.path.join(output_dir, "resource.json"), "w") as fh:
        json.dump({"device": device,
                   "train_seconds": round(float(train_seconds), 3),
                   "gpu_seconds": round(float(train_seconds), 3)
                   if device in GPU_DEVICES else 0.0}, fh, indent=2)


# --------------------------------------------------------------------------
# Data cache: parse CSVs once, store per-split column arrays as .npz
# --------------------------------------------------------------------------
_CACHE_COLS = ["user", "video", "author", "tab", "duration_ms", "hourmin", "date",
               "time_ms", "long_view", "is_click", "is_like", "is_forward",
               "play_time_ms"]
_TARGET_COLUMNS = frozenset(
    {"long_view", "is_click", "is_like", "is_forward", "play_time_ms"})
_CACHE_SCHEMA_VERSION = 2
_CACHE_SCHEMA_FILE = "cache_schema.json"


def build_cache(data_dir: str = DATA_DIR, cache_dir: str = CACHE_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    schema_path = os.path.join(cache_dir, _CACHE_SCHEMA_FILE)
    # Version 0 marks an interrupted/incomplete rebuild. The ready version is
    # written only after every split and metadata file has been replaced.
    with open(schema_path, "w") as fh:
        json.dump({"version": 0}, fh)
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]

    # Feature rows span train/valid/test. Target rows deliberately stop at
    # validation, so hidden-test outcomes never enter a Python list or NumPy
    # array used by the runtime.
    raw = {c: [] for c in _CACHE_COLS if c not in _TARGET_COLUMNS}
    raw_targets = {c: [] for c in _CACHE_COLS if c in _TARGET_COLUMNS}
    target_dates = []
    for f in LOG_FILES:
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                d = int(r["date"])
                if not (SPLITS["train"][0] <= d <= SPLITS["test"][1]):
                    continue
                raw["user"].append(r["user_id"])
                raw["video"].append(r["video_id"])
                raw["author"].append(vid2author.get(r["video_id"], "UNK"))
                raw["tab"].append(r["tab"])
                raw["duration_ms"].append(float(r["duration_ms"]))
                raw["hourmin"].append(int(r["hourmin"]))
                raw["date"].append(d)
                raw["time_ms"].append(int(r["time_ms"]))
                is_hidden_test = (
                    SPLITS["test"][0] <= d <= SPLITS["test"][1])
                if not is_hidden_test:
                    target_dates.append(d)
                    raw_targets["long_view"].append(
                        1.0 if r["long_view"] != "0" else 0.0)
                    raw_targets["is_click"].append(
                        1.0 if r["is_click"] != "0" else 0.0)
                    raw_targets["is_like"].append(
                        1.0 if r["is_like"] != "0" else 0.0)
                    raw_targets["is_forward"].append(
                        1.0 if r["is_forward"] != "0" else 0.0)
                    raw_targets["play_time_ms"].append(
                        float(r["play_time_ms"]))

    date = np.asarray(raw["date"], dtype=np.int32)
    train_mask = (date >= SPLITS["train"][0]) & (date <= SPLITS["train"][1])

    # train-only vocab per categorical field; unseen values -> UNK slot (= vocab size)
    meta = {"field_dims": {}}
    coded = {}
    vocabs = {}
    for col in ("user", "video", "author", "tab"):
        vals = raw[col]
        vocab = {}
        for v, in_tr in zip(vals, train_mask):
            if in_tr and v not in vocab:
                vocab[v] = len(vocab)
        unk = len(vocab)
        coded[col] = np.asarray([vocab.get(v, unk) for v in vals], dtype=np.int32)
        meta["field_dims"][col] = unk + 1
        vocabs[col] = vocab
    with open(os.path.join(cache_dir, "vocabs.json"), "w") as fh:
        json.dump(vocabs, fh)

    feature_arrays = {
        "user": coded["user"], "video": coded["video"],
        "author": coded["author"], "tab": coded["tab"],
        "duration_ms": np.asarray(raw["duration_ms"], dtype=np.float32),
        "hourmin": np.asarray(raw["hourmin"], dtype=np.int32),
        "date": date,
        "time_ms": np.asarray(raw["time_ms"], dtype=np.int64),
        # raw user strings are needed so GAUC groups match the official ids
        "user_raw": np.asarray(raw["user"], dtype=object),
        # Raw video identity is required for evaluation-batch repeat features;
        # encoded unseen videos all share one UNK slot and are not interchangeable.
        "video_raw": np.asarray(raw["video"], dtype=object),
    }
    target_date = np.asarray(target_dates, dtype=np.int32)
    target_arrays = {
        key: np.asarray(values, dtype=np.float32)
        for key, values in raw_targets.items()
    }
    for name, (lo, hi) in SPLITS.items():
        m = (date >= lo) & (date <= hi)
        payload = {k: v[m] for k, v in feature_arrays.items()}
        if name != "test":
            target_mask = (target_date >= lo) & (target_date <= hi)
            if int(target_mask.sum()) != int(m.sum()):
                raise RuntimeError(
                    f"{name} feature/target cache rows are not aligned")
            payload.update(
                {key: values[target_mask]
                 for key, values in target_arrays.items()})
        np.savez(os.path.join(cache_dir, f"{name}.npz"),
                 **payload)
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)
    ready_path = schema_path + ".ready"
    with open(ready_path, "w") as fh:
        json.dump({"version": _CACHE_SCHEMA_VERSION}, fh)
    os.replace(ready_path, schema_path)


def load_cache(cache_dir: str = CACHE_DIR) -> tuple[dict, dict]:
    """Returns ({split: {col: array}}, meta). Builds the cache on first use."""
    meta_path = os.path.join(cache_dir, "meta.json")
    schema_path = os.path.join(cache_dir, _CACHE_SCHEMA_FILE)
    try:
        with open(schema_path) as fh:
            cache_version = int(json.load(fh).get("version", -1))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        cache_version = -1
    required = [meta_path, os.path.join(cache_dir, "vocabs.json")]
    required.extend(os.path.join(cache_dir, f"{name}.npz") for name in SPLITS)
    if cache_version != _CACHE_SCHEMA_VERSION \
            or not all(os.path.isfile(path) for path in required):
        print("train_lib: building data cache (first run, ~1 min)...", flush=True)
        build_cache(cache_dir=cache_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)
    splits = {}
    for name in SPLITS:
        with np.load(os.path.join(cache_dir, f"{name}.npz"), allow_pickle=True) as z:
            splits[name] = {k: z[k] for k in z.files}
    # Old caches may predate the boundary. Never expose hidden-test outcomes
    # through the supported training API, even when such a cache is reused.
    had_test_targets = any(key in splits["test"] for key in _TARGET_COLUMNS)
    for key in _TARGET_COLUMNS:
        splits["test"].pop(key, None)
    if had_test_targets:
        test_path = os.path.join(cache_dir, "test.npz")
        migration_path = os.path.join(cache_dir, "test.boundary-migration.npz")
        np.savez(migration_path, **splits["test"])
        os.replace(migration_path, test_path)
    return splits, meta


def load_validation_targets(cache_dir: str = CACHE_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Trusted parent-only validation grouping/labels for authoritative scoring."""
    # load_cache also completes schema upgrades and old test-target migration
    # before the executor captures protected hashes.
    splits, _ = load_cache(cache_dir=cache_dir)
    return (splits["valid"]["user_raw"].copy(),
            splits["valid"]["long_view"].copy())


# --------------------------------------------------------------------------
# Feature encoding: X (N, F) int32 with per-field offsets
# --------------------------------------------------------------------------
def encode_features(splits: dict, meta: dict, temporal: str = "none",
                    feature_mode: str = "standard"):
    """Encode base fields and optional compact multi-scenario cross fields."""
    tr = splits["train"]
    edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 11)[1:-1])
    dims = [meta["field_dims"]["user"], meta["field_dims"]["video"],
            meta["field_dims"]["author"], meta["field_dims"]["tab"], 11]
    use_hour = temporal in ("hour_bucket", "hour_plus_dow")
    use_dow = temporal == "hour_plus_dow"
    if use_hour:
        dims.append(24)
    if use_dow:
        dims.append(7)

    composite = []
    if feature_mode == "scenario_crosses":
        dur_all = {}
        for name, s in splits.items():
            dur_all[name] = np.searchsorted(
                edges, s["duration_ms"]).astype(np.int64)
        pair_specs = [
            ("user_tab", "user", "tab", meta["field_dims"]["tab"]),
            ("video_tab", "video", "tab", meta["field_dims"]["tab"]),
            ("author_tab", "author", "tab", meta["field_dims"]["tab"]),
        ]
        for label, left, right, right_dim in pair_specs:
            train_key = (tr[left].astype(np.int64) * int(right_dim) +
                         tr[right].astype(np.int64))
            vocab = np.unique(train_key)
            codes = {}
            for name, s in splits.items():
                key = (s[left].astype(np.int64) * int(right_dim) +
                       s[right].astype(np.int64))
                pos = np.searchsorted(vocab, key)
                seen = pos < len(vocab)
                safe = np.minimum(pos, len(vocab) - 1)
                seen &= vocab[safe] == key
                codes[name] = np.where(seen, pos, len(vocab)).astype(np.int32)
            composite.append((label, codes, len(vocab) + 1))

        train_key = tr["tab"].astype(np.int64) * 11 + dur_all["train"]
        vocab = np.unique(train_key)
        codes = {}
        for name, s in splits.items():
            key = s["tab"].astype(np.int64) * 11 + dur_all[name]
            pos = np.searchsorted(vocab, key)
            seen = pos < len(vocab)
            safe = np.minimum(pos, len(vocab) - 1)
            seen &= vocab[safe] == key
            codes[name] = np.where(seen, pos, len(vocab)).astype(np.int32)
        composite.append(("tab_duration", codes, len(vocab) + 1))
        dims.extend(int(dim) for _, _, dim in composite)
    elif feature_mode != "standard":
        raise ValueError(f"unknown feature_mode: {feature_mode}")
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    enc = {}
    for name, s in splits.items():
        cols = [s["user"], s["video"], s["author"], s["tab"],
                np.searchsorted(edges, s["duration_ms"]).astype(np.int32)]
        if use_hour:
            cols.append((s["hourmin"] // 100).astype(np.int32))
        if use_dow:
            # date is yyyymmdd in April–May 2022; day-of-week via day offset from 20220408 (a Friday)
            day = (s["date"] % 100) + np.where(s["date"] >= 20220501, 30, 0)
            cols.append(((day - 8 + 4) % 7).astype(np.int32))
        for _, codes, _ in composite:
            cols.append(codes[name])
        X = np.stack(cols, axis=1).astype(np.int32) + offsets[None, :]
        enc[name] = X
    return enc, int(sum(dims)), offsets, dims


# --------------------------------------------------------------------------
# History pooling (train-period positives per user)
# --------------------------------------------------------------------------
class History:
    """Per-user train-positive video ids/times. Provides stop-grad pooled vectors.

    valid/test rows: mean over ALL train positives of that user (causal — train
    strictly precedes both windows). train rows: leave-one-out (a positive row's
    own video is removed from its user's pool) to avoid self-leakage.
    """

    def __init__(self, splits: dict, n_users: int, mode: str, tau_days: float = 3.0):
        tr = splits["train"]
        self.mode = mode
        pos = tr["long_view"] > 0
        self.n_users = n_users
        self.pos_user = tr["user"][pos]
        self.pos_vid = tr["video"][pos]          # cache-level codes (video field, no offset)
        if mode == "recency_weighted_pool":
            t = tr["time_ms"][pos].astype(np.float64)
            tmax = np.zeros(n_users)
            np.maximum.at(tmax, self.pos_user, t)
            self.w = np.exp(-(tmax[self.pos_user] - t) / (tau_days * 86400e3)).astype(np.float32)
        else:
            self.w = np.ones(len(self.pos_user), dtype=np.float32)
        self.cnt = np.zeros(n_users, dtype=np.float32)
        np.add.at(self.cnt, self.pos_user, self.w)
        # per-train-row own weight (for leave-one-out on positive rows)
        self.train_row_w = np.zeros(len(tr["user"]), dtype=np.float32)
        self.train_row_w[pos] = self.w

    def pooled(self, V_video: np.ndarray) -> np.ndarray:
        """(n_users, k) weighted-mean embedding of train positives; zeros if none."""
        k = V_video.shape[1]
        S = np.zeros((self.n_users, k), dtype=np.float32)
        np.add.at(S, self.pos_user, self.w[:, None] * V_video[self.pos_vid])
        return S  # unnormalized sum; divide at use-site with cnt

    def batch_vectors(self, S, users, split_is_train, V_video=None,
                      row_vid=None, row_w=None, row_y=None):
        """Pooled (B, k) vectors for a batch. Leave-one-out on train positives."""
        num = S[users].copy()
        den = self.cnt[users].copy()
        if split_is_train:
            loo = row_y > 0
            num[loo] -= row_w[loo, None] * V_video[row_vid[loo]]
            den[loo] -= row_w[loo]
        den = np.maximum(den, 1e-6)[:, None]
        return num / den


class MultiBehaviorHistory:
    """Four disjoint, recency-weighted train histories with row-wise LOO.

    Channels are: long-view positive; social engagement without long-view;
    click-only without long-view/social; and clear negative (none of those).
    Channel 0 exactly supplies the incumbent positive history.  Channels 1..3
    are consumed by trainable, bounded FISM-style affinity gates.
    """

    N_CHANNELS = 4

    def __init__(self, splits: dict, n_users: int, tau_days: float = 3.0):
        tr = splits["train"]
        self.n_users = n_users
        positive = tr["long_view"] > 0
        social = (~positive) & ((tr["is_like"] > 0) | (tr["is_forward"] > 0))
        click_only = (~positive) & (~social) & (tr["is_click"] > 0)
        channel = np.full(len(positive), 3, dtype=np.int8)
        channel[click_only] = 2
        channel[social] = 1
        channel[positive] = 0
        self.channel = channel
        self.user = tr["user"]
        self.video = tr["video"]

        t = tr["time_ms"].astype(np.float64)
        tmax = np.zeros(n_users, dtype=np.float64)
        np.maximum.at(tmax, self.user, t)
        self.row_weight = np.exp(
            -(tmax[self.user] - t) / (tau_days * 86400e3)).astype(np.float32)
        self.count = np.zeros((self.N_CHANNELS, n_users), dtype=np.float32)
        for c in range(self.N_CHANNELS):
            mask = channel == c
            np.add.at(self.count[c], self.user[mask], self.row_weight[mask])

    def pooled_many(self, V_video: np.ndarray) -> np.ndarray:
        k = V_video.shape[1]
        sums = np.zeros((self.N_CHANNELS, self.n_users, k), dtype=np.float32)
        for c in range(self.N_CHANNELS):
            mask = self.channel == c
            np.add.at(sums[c], self.user[mask],
                      self.row_weight[mask, None] * V_video[self.video[mask]])
        return sums

    def batch_vectors_many(self, sums, users, split_is_train,
                           V_video=None, row_index=None):
        """Return (B,4,k) means; remove a train row from its own channel."""
        num = np.transpose(sums[:, users, :], (1, 0, 2)).copy()
        den = self.count[:, users].T.copy()
        if split_is_train:
            if row_index is None or V_video is None:
                raise ValueError("V_video and row_index are required for train history LOO")
            row_index = np.asarray(row_index, dtype=np.int64)
            rows = np.arange(len(row_index))
            channels = self.channel[row_index]
            weights = self.row_weight[row_index]
            vids = self.video[row_index]
            num[rows, channels] -= weights[:, None] * V_video[vids]
            den[rows, channels] -= weights
        return num / np.maximum(den, 1e-6)[:, :, None]


# --------------------------------------------------------------------------
# Numpy FM engine with pluggable loss / history / multitask
# --------------------------------------------------------------------------
class RankFM:
    """FM over categorical fields + optional stop-grad pooled history vector.

    logits = b + W[X].sum + 0.5*((S_f + H)^2 - sum E^2 - H^2)  (pure cross terms)
    Aux heads (multitask): z_t = b_t + W_t[X].sum + u_t · (S_f + H).
    """

    def __init__(self, dim, k=16, lr=1e-3, l2=1e-6, seed=0, aux_tasks=(),
                 history_channels=0, n_fields=None, field_weighted=False):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.k, self.lr, self.l2 = k, lr, l2
        self.aux_tasks = list(aux_tasks)
        self.Wt = {t: np.zeros(dim, dtype=np.float32) for t in self.aux_tasks}
        self.ut = {t: rng.normal(0, 0.01, k).astype(np.float32) for t in self.aux_tasks}
        self.bt = {t: np.float32(0.0) for t in self.aux_tasks}
        self.history_gate_raw = np.zeros(history_channels, dtype=np.float32)
        self.field_weighted = bool(field_weighted)
        self.n_fields = int(n_fields or 0)
        if self.field_weighted:
            self.pair_i, self.pair_j = np.triu_indices(self.n_fields, 1)
            self.field_pair_raw = np.zeros(len(self.pair_i), dtype=np.float32)
            self.history_field_raw = np.zeros(self.n_fields, dtype=np.float32)
        else:
            self.pair_i = self.pair_j = np.zeros(0, dtype=np.int64)
            self.field_pair_raw = np.zeros(0, dtype=np.float32)
            self.history_field_raw = np.zeros(0, dtype=np.float32)
        self._adam = {}
        self.t = 0

    # -- adam over named params (dict id -> array) --
    def _step_param(self, name, P, G):
        m, v = self._adam.setdefault(name, (np.zeros_like(P), np.zeros_like(P)))
        b1, b2, eps = 0.9, 0.999, 1e-8
        m *= b1; m += (1 - b1) * G
        v *= b2; v += (1 - b2) * (G * G)
        P -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)

    def forward(self, X, H=None, H_extra=None):
        E = self.V[X]                     # (B,F,k)
        S = E.sum(1)                      # (B,k)
        St = S if H is None else S + H
        if self.field_weighted:
            pair_weights = 2.0 * sigmoid(self.field_pair_raw)
            R = np.zeros((self.n_fields, self.n_fields), dtype=np.float32)
            R[self.pair_i, self.pair_j] = pair_weights
            R[self.pair_j, self.pair_i] = pair_weights
            weighted_other = np.einsum("ij,bjk->bik", R, E, optimize=True)
            inter = 0.5 * np.sum(E * weighted_other, axis=(1, 2))
            if H is not None:
                history_weights = 2.0 * sigmoid(self.history_field_raw)
                inter += np.einsum("bik,bk,i->b", E, H, history_weights,
                                   optimize=True)
        else:
            inter = 0.5 * ((St ** 2).sum(1) - (E ** 2).sum((1, 2)))
            if H is not None:
                inter -= 0.5 * (H ** 2).sum(1)
        z = self.b + self.W[X].sum(1) + inter
        hmix = None
        if H_extra is not None and self.history_gate_raw.size:
            gates = np.tanh(self.history_gate_raw)
            hmix = np.einsum("c,bck->bk", gates, H_extra, optimize=True)
            # FISM-style candidate-to-history affinity.  The history vectors are
            # stop-gradient; candidate embeddings and bounded gates are trained.
            z = z + np.sum(E[:, 1, :] * hmix, axis=1)
        return z, (X, E, St, H_extra, hmix, H)

    def aux_forward(self, task, cache):
        X, _, St = cache[:3]
        return self.bt[task] + self.Wt[task][X].sum(1) + St @ self.ut[task]

    def apply_grads(self, contribs, aux_contribs=()):
        """contribs: [(cache, g)] for the main head; aux: [(task, cache, g_t)].
        g arrays are dLoss/dz already divided by batch size."""
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        gb = 0.0
        gate_grad = np.zeros_like(self.history_gate_raw)
        pair_grad = np.zeros_like(self.field_pair_raw)
        history_field_grad = np.zeros_like(self.history_field_raw)
        for cache, g in contribs:
            X, E, St, H_extra, hmix, H = cache
            np.add.at(gW, X, g[:, None])
            if self.field_weighted:
                pair_weights = 2.0 * sigmoid(self.field_pair_raw)
                R = np.zeros((self.n_fields, self.n_fields), dtype=np.float32)
                R[self.pair_i, self.pair_j] = pair_weights
                R[self.pair_j, self.pair_i] = pair_weights
                grad_e = np.einsum("ij,bjk->bik", R, E, optimize=True)
                pair_dot = np.sum(E[:, self.pair_i, :] * E[:, self.pair_j, :],
                                  axis=2)
                pair_dr = pair_weights * (1.0 - pair_weights / 2.0)
                pair_grad += np.sum(g[:, None] * pair_dot, axis=0) * pair_dr
                if H is not None:
                    history_weights = 2.0 * sigmoid(self.history_field_raw)
                    grad_e += history_weights[None, :, None] * H[:, None, :]
                    history_dot = np.einsum("bik,bk->bi", E, H, optimize=True)
                    hist_dr = history_weights * (1.0 - history_weights / 2.0)
                    history_field_grad += (
                        np.sum(g[:, None] * history_dot, axis=0) * hist_dr)
                np.add.at(gV, X, g[:, None, None] * grad_e)
            else:
                np.add.at(gV, X, g[:, None, None] * (St[:, None, :] - E))
            if H_extra is not None and hmix is not None:
                np.add.at(gV, X[:, 1], g[:, None] * hmix)
                affinity = np.einsum("bk,bck->bc", E[:, 1, :], H_extra,
                                     optimize=True)
                gates = np.tanh(self.history_gate_raw)
                gate_grad += np.sum(g[:, None] * affinity, axis=0) * (1.0 - gates ** 2)
            gb += g.sum()
        gWt = {t: np.zeros_like(self.W) for t in self.aux_tasks}
        gut = {t: np.zeros_like(self.ut[t]) for t in self.aux_tasks}
        gbt = {t: 0.0 for t in self.aux_tasks}
        for task, cache, g in aux_contribs:
            X, E, St = cache[:3]
            np.add.at(gWt[task], X, g[:, None])
            gut[task] += (g[:, None] * St).sum(0)
            gbt[task] += g.sum()
            np.add.at(gV, X, (g[:, None] * self.ut[task][None, :])[:, None, :]
                      * np.ones((1, E.shape[1], 1), dtype=np.float32))
        gV += self.l2 * self.V
        gW += self.l2 * self.W
        self.t += 1
        self._step_param("V", self.V, gV)
        self._step_param("W", self.W, gW)
        if self.history_gate_raw.size:
            self._step_param("history_gate_raw", self.history_gate_raw, gate_grad)
        if self.field_weighted:
            # Keep the light interaction reweighting near its exact FM start.
            pair_grad += 1e-4 * self.field_pair_raw
            history_field_grad += 1e-4 * self.history_field_raw
            self._step_param("field_pair_raw", self.field_pair_raw, pair_grad)
            self._step_param("history_field_raw", self.history_field_raw,
                             history_field_grad)
        self.b -= self.lr * np.float32(gb)
        for t_ in self.aux_tasks:
            self._step_param(f"Wt.{t_}", self.Wt[t_], gWt[t_])
            self._step_param(f"ut.{t_}", self.ut[t_], gut[t_])
            self.bt[t_] -= self.lr * np.float32(gbt[t_])

    def predict(self, X, H=None, H_extra=None, bs=200_000):
        out = []
        for i in range(0, len(X), bs):
            h = None if H is None else H[i:i + bs]
            hx = None if H_extra is None else H_extra[i:i + bs]
            out.append(self.forward(X[i:i + bs], h, hx)[0])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def state(self):
        s = (self.V.copy(), self.W.copy(), np.float32(self.b),
             {t: self.Wt[t].copy() for t in self.aux_tasks},
             {t: self.ut[t].copy() for t in self.aux_tasks},
             dict(self.bt), self.history_gate_raw.copy(),
             self.field_pair_raw.copy(), self.history_field_raw.copy())
        return s

    def load_state(self, s):
        self.V, self.W, self.b = s[0].copy(), s[1].copy(), s[2]
        for t in self.aux_tasks:
            self.Wt[t] = s[3][t].copy()
            self.ut[t] = s[4][t].copy()
            self.bt[t] = s[5][t]
        if self.history_gate_raw.size and len(s) > 6:
            self.history_gate_raw = s[6].copy()
        if self.field_weighted and len(s) > 8:
            self.field_pair_raw = s[7].copy()
            self.history_field_raw = s[8].copy()


# ---- loss-specific epoch runners (numpy engine) ----
def _aux_targets(split_cols, multitask):
    if multitask == "aux_click":
        return {"click": split_cols["is_click"]}
    if multitask == "aux_click_like_forward":
        return {"click": split_cols["is_click"], "like": split_cols["is_like"],
                "forward": split_cols["is_forward"]}
    if multitask == "censored_watch_time":
        r = split_cols["play_time_ms"] / np.maximum(split_cols["duration_ms"], 1.0)
        return {"watch": np.minimum(r, 3.0).astype(np.float32),
                "watch_censored": (r >= 0.97).astype(np.float32)}
    return {}


def _aux_grad_contribs(model, cache, aux, idx, lam):
    out = []
    B = len(idx)
    for task in model.aux_tasks:
        if task == "watch":
            p = model.aux_forward("watch", cache)
            r = aux["watch"][idx]
            cens = aux["watch_censored"][idx] > 0
            g = np.where(cens, -np.clip(r - p, 0, None) * (p < r), p - r)
            out.append(("watch", cache, (lam * g / B).astype(np.float32)))
        elif task in aux:
            z = model.aux_forward(task, cache)
            g = (sigmoid(z) - aux[task][idx]) / B
            out.append((task, cache, (lam * g).astype(np.float32)))
    return out


def train_numpy_fm(cfg, enc, splits, meta, log):
    """Returns (best_state_scores) dict with valid metrics + scores for all splits."""
    Xtr, Xva, Xte = enc["train"], enc["valid"], enc["test"]
    tr = splits["train"]
    ytr = tr["long_view"]
    uva_raw = list(splits["valid"]["user_raw"])
    yva = splits["valid"]["long_view"]
    fixed = cfg.get("fixed_score_offsets")
    if fixed is None:
        qtr = np.zeros(len(Xtr), dtype=np.float32)
        qva = np.zeros(len(Xva), dtype=np.float32)
        qte = np.zeros(len(Xte), dtype=np.float32)
    else:
        qtr = np.asarray(fixed["train"], dtype=np.float32)
        qva = np.asarray(fixed["valid"], dtype=np.float32)
        qte = np.asarray(fixed["test"], dtype=np.float32)

    hist = None
    Htr = Hva = Hte = None
    Htr_extra = Hva_extra = Hte_extra = None
    if cfg["history"] in ("mean_pool_positives", "recency_weighted_pool"):
        hist = History(splits, meta["field_dims"]["user"], cfg["history"])
    elif cfg["history"] == "signed_multibehavior_fism":
        hist = MultiBehaviorHistory(splits, meta["field_dims"]["user"])

    aux_map = {"aux_click": ["click"], "aux_click_like_forward": ["click", "like", "forward"],
               "censored_watch_time": ["watch"], "none": []}
    model = RankFM(dim=cfg["dim"], k=cfg["k"], lr=cfg["lr"], seed=cfg["seed"],
                   aux_tasks=aux_map[cfg["multitask"]],
                   history_channels=3 if cfg["history"] == "signed_multibehavior_fism"
                   else 0, n_fields=Xtr.shape[1],
                   field_weighted=cfg["model"] == "fwfm_numpy")
    aux = _aux_targets(tr, cfg["multitask"])
    lam = cfg.get("aux_weight", 0.2)
    rng = np.random.default_rng(cfg["seed"])
    bs = cfg["bs"]

    # per-user row indices on train (for pairwise/listwise)
    n_users = meta["field_dims"]["user"]
    user_tr = tr["user"]
    order = np.argsort(user_tr, kind="stable")
    bounds = np.searchsorted(user_tr[order], np.arange(n_users + 1))

    def refresh_pooled():
        nonlocal Htr, Hva, Hte, Htr_extra, Hva_extra, Hte_extra
        if hist is None:
            return
        # video embeddings live at offset of the video field (index 1)
        v_off = np.cumsum([0, meta["field_dims"]["user"]])[1]
        Vvid = model.V[v_off: v_off + meta["field_dims"]["video"]]
        if isinstance(hist, MultiBehaviorHistory):
            S = hist.pooled_many(Vvid)
            all_train = hist.batch_vectors_many(
                S, tr["user"], True, V_video=Vvid,
                row_index=np.arange(len(tr["user"])))
            all_valid = hist.batch_vectors_many(
                S, splits["valid"]["user"], False)
            all_test = hist.batch_vectors_many(
                S, splits["test"]["user"], False)
            Htr, Htr_extra = all_train[:, 0], all_train[:, 1:]
            Hva, Hva_extra = all_valid[:, 0], all_valid[:, 1:]
            Hte, Hte_extra = all_test[:, 0], all_test[:, 1:]
        else:
            S = hist.pooled(Vvid)
            Htr = hist.batch_vectors(S, tr["user"], True, Vvid, tr["video"],
                                     hist.train_row_w, ytr)
            Hva = hist.batch_vectors(S, splits["valid"]["user"], False)
            Hte = hist.batch_vectors(S, splits["test"]["user"], False)

    def epoch_pointwise():
        idx = rng.permutation(len(ytr))
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            h = None if Htr is None else Htr[b]
            hx = None if Htr_extra is None else Htr_extra[b]
            z, cache = model.forward(Xtr[b], h, hx)
            g = ((sigmoid(z + qtr[b]) - ytr[b]) / len(b)).astype(np.float32)
            model.apply_grads([(cache, g)],
                              _aux_grad_contribs(model, cache, aux, b, lam))

    def epoch_bpr():
        pos_rows = np.flatnonzero(ytr > 0)
        neg_pick = np.empty(len(pos_rows), dtype=np.int64)
        ok = np.zeros(len(pos_rows), dtype=bool)
        for j, r in enumerate(pos_rows):
            u = user_tr[r]
            lo, hi = bounds[u], bounds[u + 1]
            rows_u = order[lo:hi]
            negs = rows_u[ytr[rows_u] == 0]
            if len(negs):
                neg_pick[j] = negs[rng.integers(len(negs))]
                ok[j] = True
        pos_rows, neg_pick = pos_rows[ok], neg_pick[ok]
        perm = rng.permutation(len(pos_rows))
        for i in range(0, len(perm), bs):
            pb, nb = pos_rows[perm[i:i + bs]], neg_pick[perm[i:i + bs]]
            hp = None if Htr is None else Htr[pb]
            hn = None if Htr is None else Htr[nb]
            hxp = None if Htr_extra is None else Htr_extra[pb]
            hxn = None if Htr_extra is None else Htr_extra[nb]
            zp, cp = model.forward(Xtr[pb], hp, hxp)
            zn, cn = model.forward(Xtr[nb], hn, hxn)
            margin = (zp + qtr[pb]) - (zn + qtr[nb])
            g = (-sigmoid(-margin) / len(pb)).astype(np.float32)
            contribs = [(cp, g), (cn, -g)]
            contribs_aux = (_aux_grad_contribs(model, cp, aux, pb, lam)
                            + _aux_grad_contribs(model, cn, aux, nb, lam))
            model.apply_grads(contribs, contribs_aux)

    def epoch_listwise(pointwise_mix=0.0):
        users = rng.permutation(n_users)
        chunk, budget = [], 0
        for u in users:
            lo, hi = bounds[u], bounds[u + 1]
            if hi - lo < 2:
                continue
            rows_u = order[lo:hi]
            if ytr[rows_u].sum() == 0 or (ytr[rows_u] == 0).sum() == 0:
                continue
            chunk.append(rows_u)
            budget += len(rows_u)
            if budget >= bs:
                _listwise_step(chunk, pointwise_mix)
                chunk, budget = [], 0
        if chunk:
            _listwise_step(chunk, pointwise_mix)

    def _listwise_step(groups, pointwise_mix):
        rows = np.concatenate(groups)
        seg = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
        nseg = len(groups)
        h = None if Htr is None else Htr[rows]
        hx = None if Htr_extra is None else Htr_extra[rows]
        z, cache = model.forward(Xtr[rows], h, hx)
        z = z + qtr[rows]
        zmax = np.full(nseg, -np.inf, dtype=np.float64)
        np.maximum.at(zmax, seg, z)
        ez = np.exp(z - zmax[seg])
        denom = np.zeros(nseg)
        np.add.at(denom, seg, ez)
        sm = (ez / denom[seg]).astype(np.float32)
        y = ytr[rows]
        ysum = np.zeros(nseg, dtype=np.float32)
        np.add.at(ysum, seg, y)
        gl = (sm - y / ysum[seg]) / nseg
        g = gl.astype(np.float32)
        if pointwise_mix > 0:
            g = g + pointwise_mix * ((sigmoid(z) - y) / len(rows)).astype(np.float32)
        model.apply_grads([(cache, g)],
                          _aux_grad_contribs(model, cache, aux, rows, lam))

    runners = {"pointwise_logloss": epoch_pointwise,
               "bpr_pairwise": epoch_bpr,
               "listwise_softmax": epoch_listwise,
               "listwise_softmax_plus_pointwise": lambda: epoch_listwise(1.0)}

    stages = [(cfg["loss"], cfg["epochs"], cfg["lr"])]
    if cfg["training"] == "two_stage_finetune" and cfg["loss"] != "pointwise_logloss":
        stages = [("pointwise_logloss", min(12, cfg["epochs"]), cfg["lr"]),
                  (cfg["loss"], cfg["epochs"], cfg["lr"] * 0.3)]

    best, best_state, bad = -1.0, None, 0
    snapshot_mode = cfg["training"] == "cosine_snapshot_ensemble"
    snapshots = []
    for loss_name, n_ep, lr in stages:
        model.lr = lr
        bad = 0
        for ep in range(1, n_ep + 1):
            t0 = time.time()
            if snapshot_mode:
                cycle_len = 7
                phase = (ep - 1) % cycle_len
                lo, hi = 1e-4, 7.5e-4
                model.lr = lo + 0.5 * (hi - lo) * (
                    1.0 + np.cos(np.pi * phase / (cycle_len - 1)))
            refresh_pooled()
            runners[loss_name]()
            refresh_pooled()
            pred_valid = model.predict(Xva, Hva, Hva_extra) + qva
            va = evaluate(uva_raw, yva, pred_valid)
            log(f"  [{loss_name}] epoch {ep:2d} | valid primary {va['primary']:.4f} "
                f"(GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f})"
                + (f" lr {model.lr:.6f}" if snapshot_mode else "")
                + f" | {time.time()-t0:.1f}s")
            if snapshot_mode and ep >= 14 and ep % 7 == 0:
                pred_test = model.predict(Xte, Hte, Hte_extra) + qte
                snapshots.append({
                    "epoch": ep,
                    "primary": float(va["primary"]),
                    "valid": pred_valid.copy(),
                    "test": pred_test.copy(),
                })
                log(f"  snapshot captured at epoch {ep} "
                    f"(member primary {va['primary']:.4f})")
            if va["primary"] > best + 1e-5:
                best, bad, best_state = va["primary"], 0, model.state()
            else:
                bad += 1
                if not snapshot_mode and bad >= cfg["patience"]:
                    log(f"  early stop ({loss_name}) at epoch {ep}")
                    break

    model.load_state(best_state)
    refresh_pooled()
    snapshot_info = None
    if snapshot_mode and len(snapshots) >= 2:
        score_valid = np.mean(
            [rank_normalize(s["valid"]) for s in snapshots], axis=0)
        score_test = np.mean(
            [rank_normalize(s["test"]) for s in snapshots], axis=0)
        em = evaluate(uva_raw, yva, score_valid)
        snapshot_info = {
            "epochs": [s["epoch"] for s in snapshots],
            "member_primary": [s["primary"] for s in snapshots],
            "ensemble_primary": float(em["primary"]),
            "schedule": {"cycle_epochs": 7, "lr_max": 7.5e-4,
                         "lr_min": 1e-4, "warmup_cycle_excluded": True},
        }
        log(f"  fixed snapshot ensemble primary {em['primary']:.4f} "
            f"from epochs {snapshot_info['epochs']}")
    else:
        score_valid = model.predict(Xva, Hva, Hva_extra) + qva
        score_test = model.predict(Xte, Hte, Hte_extra) + qte
    return {
        "scores_valid": score_valid,
        "scores_test": score_test,
        "model": model,
        "hist": hist,
        "snapshot_info": snapshot_info,
    }


# --------------------------------------------------------------------------
# Torch engine (DeepFM / DCN-lite, optional DIN attention history)
# --------------------------------------------------------------------------
def train_torch(cfg, enc, splits, meta, log):
    import torch
    import torch.nn as nn
    torch.manual_seed(cfg["seed"])
    dev = cfg.get("device") or select_device()

    log(f"  torch device: {dev}")
    Xtr = torch.from_numpy(enc["train"]).long().to(dev)
    Xva = torch.from_numpy(enc["valid"]).long().to(dev)
    Xte = torch.from_numpy(enc["test"]).long().to(dev)
    tr = splits["train"]
    ytr = torch.from_numpy(tr["long_view"]).to(dev)
    uva_raw = list(splits["valid"]["user_raw"])
    yva = splits["valid"]["long_view"]
    n_users = meta["field_dims"]["user"]
    F = Xtr.shape[1]
    k = cfg["k"]

    # ---- history setup ----
    hist_mode = cfg["history"]
    HL = 30
    hist_pad = None
    v_off = meta["field_dims"]["user"]  # video field embedding offset

    if hist_mode != "none":
        pos = tr["long_view"] > 0
        pu, pv = tr["user"][pos], tr["video"][pos]
        ptime = tr["time_ms"][pos]
        ordidx = np.lexsort((ptime, pu))
        pu, pv = pu[ordidx], pv[ordidx]
        pad = np.zeros((n_users, HL), dtype=np.int64)
        cnt = np.zeros(n_users, dtype=np.int64)
        starts = np.searchsorted(pu, np.arange(n_users + 1))
        for u in range(n_users):
            vids = pv[starts[u]:starts[u + 1]][-HL:]
            pad[u, :len(vids)] = vids + v_off
            cnt[u] = len(vids)
        hist_pad = torch.from_numpy(pad).to(dev)
        hist_cnt = torch.from_numpy(cnt).to(dev)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(cfg["dim"], k)
            nn.init.normal_(self.emb.weight, 0, 0.01)
            self.lin = nn.Embedding(cfg["dim"], 1)
            nn.init.zeros_(self.lin.weight)
            self.bias = nn.Parameter(torch.zeros(1))
            in_dim = F * k + (k if hist_mode != "none" else 0)
            if cfg["model"] == "deepfm_mlp":
                self.mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(),
                                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
            else:  # dcn_lite: 2 cross layers + small MLP
                self.cross_w = nn.ParameterList(
                    [nn.Parameter(torch.zeros(in_dim)) for _ in range(2)])
                self.cross_b = nn.ParameterList(
                    [nn.Parameter(torch.zeros(in_dim)) for _ in range(2)])
                self.mlp = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))
            if hist_mode == "din_attention":
                self.att = nn.Sequential(nn.Linear(3 * k, 32), nn.ReLU(), nn.Linear(32, 1))
            self.aux_heads = nn.ModuleDict()
            for t in cfg["aux_tasks"]:
                self.aux_heads[t] = nn.Linear(in_dim, 1)

        def hist_vec(self, users, cand_vid_emb):
            hv = self.emb(hist_pad[users])                       # (B,HL,k)
            mask = (torch.arange(HL, device=hv.device)[None, :]
                    < hist_cnt[users][:, None])
            if hist_mode == "din_attention":
                c = cand_vid_emb[:, None, :].expand_as(hv)
                a = self.att(torch.cat([hv, c, hv * c], dim=-1)).squeeze(-1)
                a = a.masked_fill(~mask, -1e9).softmax(dim=1)
                return (a[:, :, None] * hv).sum(1)
            m = mask.float()
            return (hv * m[:, :, None]).sum(1) / m.sum(1).clamp(min=1)[:, None]

        def forward(self, X, users):
            E = self.emb(X)                                       # (B,F,k)
            flat = [E.flatten(1)]
            if hist_mode != "none":
                flat.append(self.hist_vec(users, E[:, 1, :]))
            v = torch.cat(flat, dim=1)
            S = E.sum(1)
            fm = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
            first = self.lin(X).sum((1, 2)) + self.bias
            if cfg["model"] == "deepfm_mlp":
                deep = self.mlp(v).squeeze(-1)
                z = first + fm + deep
            else:
                x0 = v
                xi = v
                for w, b in zip(self.cross_w, self.cross_b):
                    xi = x0 * (xi @ w)[:, None] + b + xi
                z = first + self.mlp(xi).squeeze(-1) + fm
            auxz = {t: self.aux_heads[t](v).squeeze(-1) for t in self.aux_heads}
            return z, auxz

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    users_tr = torch.from_numpy(tr["user"].astype(np.int64)).to(dev)
    users_va = torch.from_numpy(splits["valid"]["user"].astype(np.int64)).to(dev)
    users_te = torch.from_numpy(splits["test"]["user"].astype(np.int64)).to(dev)
    aux = {kk: torch.from_numpy(v).to(dev)
           for kk, v in _aux_targets(tr, cfg["multitask"]).items()
           if not kk.endswith("_censored")}
    lam = cfg.get("aux_weight", 0.2)
    bce = nn.BCEWithLogitsLoss()

    # listwise segment prep
    user_np = tr["user"]
    order = np.argsort(user_np, kind="stable")
    bounds = np.searchsorted(user_np[order], np.arange(n_users + 1))
    rng = np.random.default_rng(cfg["seed"])

    def predict(X, users, bs=100_000):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs):
                out.append(net(X[i:i + bs], users[i:i + bs])[0])
        net.train()
        return torch.cat(out).cpu().numpy()

    def aux_loss(auxz, idx):
        L = 0.0
        for t, target in aux.items():
            if t in auxz:
                L = L + lam * bce(auxz[t], target[idx])
        return L

    def epoch_pointwise():
        idx = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(idx), cfg["bs"]):
            b = idx[i:i + cfg["bs"]]
            z, auxz = net(Xtr[b], users_tr[b])
            loss = bce(z, ytr[b]) + aux_loss(auxz, b)
            opt.zero_grad(); loss.backward(); opt.step()

    def epoch_bpr():
        pos_rows = np.flatnonzero(tr["long_view"] > 0)
        pairs_p, pairs_n = [], []
        for r in pos_rows:
            u = user_np[r]
            rows_u = order[bounds[u]:bounds[u + 1]]
            negs = rows_u[tr["long_view"][rows_u] == 0]
            if len(negs):
                pairs_p.append(r)
                pairs_n.append(negs[rng.integers(len(negs))])
        P = torch.from_numpy(np.asarray(pairs_p)).to(dev)
        N = torch.from_numpy(np.asarray(pairs_n)).to(dev)
        perm = torch.randperm(len(P), device=dev)
        for i in range(0, len(perm), cfg["bs"]):
            pb, nb = P[perm[i:i + cfg["bs"]]], N[perm[i:i + cfg["bs"]]]
            zp, ap = net(Xtr[pb], users_tr[pb])
            zn, _ = net(Xtr[nb], users_tr[nb])
            loss = torch.nn.functional.softplus(-(zp - zn)).mean() + aux_loss(ap, pb)
            opt.zero_grad(); loss.backward(); opt.step()

    def epoch_listwise(mix=0.0):
        uperm = rng.permutation(n_users)
        chunk, budget = [], 0
        for u in uperm:
            rows_u = order[bounds[u]:bounds[u + 1]]
            yv = tr["long_view"][rows_u]
            if len(rows_u) < 2 or yv.sum() == 0 or (yv == 0).sum() == 0:
                continue
            chunk.append(rows_u)
            budget += len(rows_u)
            if budget >= cfg["bs"]:
                _lw_step(chunk, mix); chunk, budget = [], 0
        if chunk:
            _lw_step(chunk, mix)

    def _lw_step(groups, mix):
        rows = torch.from_numpy(np.concatenate(groups)).to(dev)
        seg = torch.from_numpy(np.concatenate(
            [np.full(len(g), i) for i, g in enumerate(groups)])).to(dev)
        z, auxz = net(Xtr[rows], users_tr[rows])
        y = ytr[rows]
        zmax = torch.full((len(groups),), -1e30, device=dev).scatter_reduce(
            0, seg, z.detach(), reduce="amax")
        ez = (z - zmax[seg]).exp()
        denom = torch.zeros(len(groups), device=dev).scatter_add(0, seg, ez)
        logsm = (z - zmax[seg]) - denom[seg].log()
        ysum = torch.zeros(len(groups), device=dev).scatter_add(0, seg, y)
        loss = -((y / ysum[seg]) * logsm).sum() / len(groups)
        if mix > 0:
            loss = loss + mix * bce(z, y)
        loss = loss + aux_loss(auxz, rows)
        opt.zero_grad(); loss.backward(); opt.step()

    runners = {"pointwise_logloss": epoch_pointwise,
               "bpr_pairwise": epoch_bpr,
               "listwise_softmax": epoch_listwise,
               "listwise_softmax_plus_pointwise": lambda: epoch_listwise(1.0)}

    best, best_sd, bad = -1.0, None, 0
    for ep in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        runners[cfg["loss"]]()
        va = evaluate(uva_raw, yva, predict(Xva, users_va))
        log(f"  [{cfg['model']}/{cfg['loss']}] epoch {ep:2d} | valid primary "
            f"{va['primary']:.4f} (GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f}) "
            f"| {time.time()-t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_sd = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                log(f"  early stop at epoch {ep}")
                break
    net.load_state_dict(best_sd)
    return {"scores_valid": predict(Xva, users_va),
            "scores_test": predict(Xte, users_te)}


# --------------------------------------------------------------------------
# Menu-driven entrypoint used by seed/generated solutions
# --------------------------------------------------------------------------
def run(menu_choices: dict, output_dir: str, seed: int = 0, verbose: bool = True) -> dict:
    os.makedirs(output_dir, exist_ok=True)

    def log(msg):
        if verbose:
            print(msg, flush=True)

    splits, meta = load_cache()
    model_choice = menu_choices.get("model", "fm_numpy")
    feature_mode = ("scenario_crosses" if model_choice == "scenario_fm_numpy"
                    else "standard")
    enc, dim, offsets, dims = encode_features(
        splits, meta, menu_choices.get("temporal", "none"), feature_mode)

    training = menu_choices.get("training", "default")
    snapshot_training = training == "cosine_snapshot_ensemble"
    cfg = {
        "dim": dim, "k": 32 if training == "k32" else 16,
        "lr": (7.5e-4 if snapshot_training else
               (5e-4 if training == "lower_lr_longer" else 1e-3)),
        "bs": 8192,
        "epochs": (42 if snapshot_training else
                   (60 if training == "lower_lr_longer" else 40)),
        "patience": (42 if snapshot_training else
                     (6 if training == "lower_lr_longer" else 4)),
        "seed": seed,
        "loss": menu_choices.get("loss", "pointwise_logloss"),
        "history": menu_choices.get("user_history", "none"),
        "multitask": menu_choices.get("multitask", "none"),
        "model": model_choice,
        "training": training,
    }
    aux_map = {"aux_click": ["click"], "aux_click_like_forward": ["click", "like", "forward"],
               "censored_watch_time": ["watch"], "none": []}
    cfg["aux_tasks"] = aux_map[cfg["multitask"]]
    prior_mode = menu_choices.get("score_prior", "none")
    prior_info = None
    if prior_mode == "recency_bayesian_offset_oof":
        raw_offsets, prior_info = oof_recency_bayesian_prior_scores(splits)
        beta = float(prior_info["offset_beta"])
        cfg["fixed_score_offsets"] = {
            name: beta * raw_offsets[name] for name in ("train", "valid", "test")
        }
        log("  score prior: leave-one-date-out recency Bayesian offset "
            f"inside training (beta={beta:.2f})")

    t_train = time.time()
    if cfg["model"] in ("fm_numpy", "fwfm_numpy", "scenario_fm_numpy"):
        device = "cpu"                       # the numpy engine never uses a GPU
        res = train_numpy_fm(cfg, enc, splits, meta, log)
    else:
        cfg["epochs"] = min(cfg["epochs"], 12)
        cfg["patience"] = min(cfg["patience"], 3)
        device = cfg.get("device") or select_device()
        cfg["device"] = device
        res = train_torch(cfg, enc, splits, meta, log)
    train_seconds = time.time() - t_train
    # Measured compute, read back by the harness for the GPU-hours report.
    write_resource_json(output_dir, device, train_seconds)

    history_info = None
    if cfg["history"] == "signed_multibehavior_fism":
        gates = np.tanh(res["model"].history_gate_raw).astype(np.float64)
        history_info = {
            "method": "signed_multibehavior_fism",
            "channels": ["social_without_long_view", "click_only", "clear_negative"],
            "gates": [float(x) for x in gates],
            "positive_history": "recency_weighted_long_view",
            "train_leakage_control": "row removed from its own disjoint channel",
        }
        log(f"  learned signed-history gates: {history_info['gates']}")

    field_weight_info = None
    if cfg["model"] == "fwfm_numpy":
        model = res["model"]
        names = ["user", "video", "author", "tab", "duration"]
        temporal = menu_choices.get("temporal", "none")
        if temporal in ("hour_bucket", "hour_plus_dow"):
            names.append("hour")
        if temporal == "hour_plus_dow":
            names.append("day_of_week")
        weights = 2.0 * sigmoid(model.field_pair_raw)
        field_weight_info = {
            "method": "field_weighted_factorization_machine",
            "initial_weight": 1.0,
            "pair_weights": {
                f"{names[i]}__{names[j]}": float(w)
                for i, j, w in zip(model.pair_i, model.pair_j, weights)
            },
        }
        if res.get("hist") is not None:
            field_weight_info["history_weights"] = {
                name: float(w) for name, w in zip(
                    names, 2.0 * sigmoid(model.history_field_raw))
            }

    sv, st = np.asarray(res["scores_valid"]), np.asarray(res["scores_test"])
    batch_context_info = None
    if prior_mode == "batch_repeat_fatigue":
        sv, valid_context = batch_repeat_fatigue_scores(sv, splits["valid"])
        st, test_context = batch_repeat_fatigue_scores(st, splits["test"])
        batch_context_info = {
            "method": "label_free_batch_repeat_fatigue",
            "validation": valid_context,
            "test": test_context,
            "disclosure": (
                "Uses the complete input feature batch at inference; no outcome "
                "columns are accessed."),
        }
        log("  score adjustment: label-free repeated-exposure fatigue (weight=0.10)")
    elif prior_mode not in ("none", "recency_bayesian_offset_oof"):
        priors = bayesian_prior_scores(splits, prior_mode)
        blend_weight = 0.25 if prior_mode == "bayesian_item_author" else 0.30
        sv = sv + blend_weight * priors["valid"]
        st = st + blend_weight * priors["test"]
        log(f"  score prior: {prior_mode} (weight={blend_weight:.2f}, train-only)")
    va = evaluate(list(splits["valid"]["user_raw"]), splits["valid"]["long_view"], sv)
    # float() is REQUIRED, not cosmetic: evaluate() propagates the label array's
    # dtype, so under numpy >= 2 (NEP 50 casting rules) these are np.float32 and
    # json.dump raises "Object of type float32 is not JSON serializable". numpy 1.x
    # happened to promote to float64, which is a float subclass and serialized fine
    # — so omitting this works on an old numpy and fails on every fresh install.
    metrics = {"GAUC": float(va["GAUC"]), "nDCG@5": float(va["nDCG@5"]),
               "primary": float(va["primary"])}

    np.save(os.path.join(output_dir, "scores_valid.npy"), sv.astype(np.float64))
    np.save(os.path.join(output_dir, "scores_test.npy"), st.astype(np.float64))
    with open(os.path.join(output_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    if prior_info is not None:
        with open(os.path.join(output_dir, "prior_info.json"), "w") as fh:
            json.dump(prior_info, fh, indent=2)
    if history_info is not None:
        with open(os.path.join(output_dir, "history_info.json"), "w") as fh:
            json.dump(history_info, fh, indent=2)
    if field_weight_info is not None:
        with open(os.path.join(output_dir, "field_weights.json"), "w") as fh:
            json.dump(field_weight_info, fh, indent=2)
    if res.get("snapshot_info") is not None:
        with open(os.path.join(output_dir, "snapshot_info.json"), "w") as fh:
            json.dump(res["snapshot_info"], fh, indent=2)
    if batch_context_info is not None:
        with open(os.path.join(output_dir, "batch_context_info.json"), "w") as fh:
            json.dump(batch_context_info, fh, indent=2)

    if menu_choices.get("data_extras") == "random_log_valid_unbiased_check":
        try:
            ub = _unbiased_check(splits, meta, cfg, res,
                                 menu_choices.get("temporal", "none"))
            with open(os.path.join(output_dir, "unbiased_metrics.json"), "w") as fh:
                json.dump(ub, fh, indent=2)
            log(f"  unbiased (random-exposure, valid window) diagnostic: {ub}")
        except Exception as e:  # diagnostic only — never fail the run for it
            log(f"  unbiased check skipped: {e}")

    log(f"FINAL valid metrics: {metrics}")
    return metrics


def _unbiased_check(splits, meta, cfg, res, temporal):
    """Score the random-exposure log restricted to the VALID window (20220422–28).

    Evaluation diagnostic ONLY (never training data, never used for model selection
    by the harness). Supported for the numpy FM engine.
    """
    model = res.get("model")
    if model is None:
        raise RuntimeError("supported for fm_numpy runs only")
    with open(os.path.join(CACHE_DIR, "vocabs.json")) as fh:
        vocabs = json.load(fh)
    vid2author = {}
    with open(os.path.join(DATA_DIR, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]
    rows = []
    with open(os.path.join(DATA_DIR, "log_random_4_22_to_5_08_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            if 20220422 <= int(r["date"]) <= 20220428:
                rows.append(r)
    if not rows:
        raise RuntimeError("no valid-window rows in random log")

    tr = splits["train"]
    edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 11)[1:-1])
    unk = {c: len(vocabs[c]) for c in vocabs}
    dims = [meta["field_dims"]["user"], meta["field_dims"]["video"],
            meta["field_dims"]["author"], meta["field_dims"]["tab"], 11]
    use_hour = temporal in ("hour_bucket", "hour_plus_dow")
    use_dow = temporal == "hour_plus_dow"
    if use_hour:
        dims.append(24)
    if use_dow:
        dims.append(7)
    offsets = np.cumsum([0] + dims[:-1]).astype(np.int32)

    ucode = np.asarray([vocabs["user"].get(r["user_id"], unk["user"]) for r in rows], np.int32)
    cols = [ucode,
            np.asarray([vocabs["video"].get(r["video_id"], unk["video"]) for r in rows], np.int32),
            np.asarray([vocabs["author"].get(vid2author.get(r["video_id"], "UNK"),
                                             unk["author"]) for r in rows], np.int32),
            np.asarray([vocabs["tab"].get(r["tab"], unk["tab"]) for r in rows], np.int32),
            np.searchsorted(edges, np.asarray([float(r["duration_ms"]) for r in rows])).astype(np.int32)]
    if use_hour:
        cols.append(np.asarray([int(r["hourmin"]) // 100 for r in rows], np.int32))
    if use_dow:
        date = np.asarray([int(r["date"]) for r in rows], np.int32)
        day = (date % 100) + np.where(date >= 20220501, 30, 0)
        cols.append(((day - 8 + 4) % 7).astype(np.int32))
    X = np.stack(cols, axis=1).astype(np.int32) + offsets[None, :]

    H = None
    hist = res.get("hist")
    if hist is not None:
        v_off = meta["field_dims"]["user"]
        Vvid = model.V[v_off: v_off + meta["field_dims"]["video"]]
        S = hist.pooled(Vvid)
        H = hist.batch_vectors(S, ucode, False)

    y = np.asarray([1.0 if r["long_view"] != "0" else 0.0 for r in rows], np.float32)
    r_ = evaluate([r["user_id"] for r in rows], y, model.predict(X, H))
    return {"GAUC": float(r_["GAUC"]), "nDCG@5": float(r_["nDCG@5"]),
            "primary": float(r_["primary"]), "rows": int(len(rows))}
