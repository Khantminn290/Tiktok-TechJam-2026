"""Shared training engine for generated solutions (KuaiRand-Pure).

Design notes
- Row order everywhere matches data.load(): file 4_08_to_4_21 then 4_22_to_5_08,
  date-filtered, original file order preserved — so scores_{valid,test}.npy written
  by index are row_id-aligned with submit.py.
- Scoring is delegated to the starter kit's evaluate.py (never reimplemented).
- Hidden-test discipline: this module computes test-split *scores* but never touches
  test labels; only the harness evaluates test, once, at the very end of the run.
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
# is_follow / is_comment / is_hate / is_profile_enter are cached as AUXILIARY
# TARGETS only. They are outcomes of the impression, exactly like is_click /
# is_like / is_forward, so they are legal supervision and illegal as features.
# They are added to data_boundary.TEST_LABEL_COLUMNS in the same change, so the
# sandbox test split never carries them.
_AUX_SOCIAL_COLS = ["is_follow", "is_comment", "is_hate", "is_profile_enter"]
_CACHE_COLS = ["user", "video", "author", "tab", "duration_ms", "hourmin", "date",
               "time_ms", "long_view", "is_click", "is_like", "is_forward",
               "play_time_ms"] + _AUX_SOCIAL_COLS


def build_cache(data_dir: str = DATA_DIR, cache_dir: str = CACHE_DIR) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    vid2author = {}
    with open(os.path.join(data_dir, "video_features_basic_pure.csv")) as fh:
        for r in csv.DictReader(fh):
            vid2author[r["video_id"]] = r["author_id"]

    raw = {c: [] for c in _CACHE_COLS}
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
                raw["long_view"].append(1.0 if r["long_view"] != "0" else 0.0)
                raw["is_click"].append(1.0 if r["is_click"] != "0" else 0.0)
                raw["is_like"].append(1.0 if r["is_like"] != "0" else 0.0)
                raw["is_forward"].append(1.0 if r["is_forward"] != "0" else 0.0)
                raw["play_time_ms"].append(float(r["play_time_ms"]))
                for _c in _AUX_SOCIAL_COLS:
                    raw[_c].append(1.0 if r.get(_c, "0") != "0" else 0.0)

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

    arrays = {
        "user": coded["user"], "video": coded["video"],
        "author": coded["author"], "tab": coded["tab"],
        "duration_ms": np.asarray(raw["duration_ms"], dtype=np.float32),
        "hourmin": np.asarray(raw["hourmin"], dtype=np.int32),
        "date": date,
        "time_ms": np.asarray(raw["time_ms"], dtype=np.int64),
        "long_view": np.asarray(raw["long_view"], dtype=np.float32),
        "is_click": np.asarray(raw["is_click"], dtype=np.float32),
        "is_like": np.asarray(raw["is_like"], dtype=np.float32),
        "is_forward": np.asarray(raw["is_forward"], dtype=np.float32),
        "play_time_ms": np.asarray(raw["play_time_ms"], dtype=np.float32),
        **{c: np.asarray(raw[c], dtype=np.float32) for c in _AUX_SOCIAL_COLS},
        # raw user strings are needed so GAUC groups match the official ids
        "user_raw": np.asarray(raw["user"], dtype=object),
    }
    for name, (lo, hi) in SPLITS.items():
        m = (date >= lo) & (date <= hi)
        np.savez(os.path.join(cache_dir, f"{name}.npz"),
                 **{k: v[m] for k, v in arrays.items()})
    with open(os.path.join(cache_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh)


def load_cache(cache_dir: str = CACHE_DIR) -> tuple[dict, dict]:
    """Returns ({split: {col: array}}, meta). Builds the cache on first use."""
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        print("train_lib: building data cache (first run, ~1 min)...", flush=True)
        build_cache(cache_dir=cache_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)
    splits = {}
    for name in SPLITS:
        z = np.load(os.path.join(cache_dir, f"{name}.npz"), allow_pickle=True)
        splits[name] = {k: z[k] for k in z.files}
    return splits, meta


# --------------------------------------------------------------------------
# Feature encoding: X (N, F) int32 with per-field offsets
# --------------------------------------------------------------------------
EXTRA_FEATURE_BINS = 16     # quantile buckets per agent-discovered feature
MAX_EXTRA_FEATURES = 6      # keeps one experiment interpretable


def build_extra_features(source: str, splits: dict, meta: dict) -> dict:
    """Execute an agent-written build_features(splits, meta) and validate it.

    The contract is deliberately narrow: return
        {feature_name: {"train": arr, "valid": arr, "test": arr}}
    with one float per row of that split. Everything else -- bucketing, offsets,
    the embedding table -- is handled by encode_features, so the agent writes
    only the part that is actually a research decision.

    Validated rather than trusted: a wrong length would otherwise misalign
    silently against the labels and produce a plausible, meaningless score.
    """
    ns: dict = {}
    exec(compile(source, "<agent_features>", "exec"), ns)
    fn = ns.get("build_features")
    if not callable(fn):
        raise ValueError("feature source must define build_features(splits, meta)")
    out = fn(splits, meta)
    if not isinstance(out, dict) or not out:
        raise ValueError("build_features must return a non-empty dict")
    if len(out) > MAX_EXTRA_FEATURES:
        raise ValueError(f"at most {MAX_EXTRA_FEATURES} features per experiment, "
                         f"got {len(out)}")
    clean = {}
    for name, per_split in out.items():
        if not isinstance(per_split, dict):
            raise ValueError(f"feature {name!r} must map split -> array")
        cols = {}
        for sp in splits:
            if sp not in per_split:
                raise ValueError(f"feature {name!r} is missing split {sp!r}")
            arr = np.asarray(per_split[sp], dtype=np.float64).ravel()
            n = len(splits[sp]["user"])
            if len(arr) != n:
                raise ValueError(f"feature {name!r} split {sp!r} has {len(arr)} "
                                 f"values but the split has {n} rows")
            cols[sp] = arr
        clean[str(name)] = cols
    return clean


def encode_features(splits: dict, meta: dict, temporal: str = "none",
                    extra: dict | None = None):
    """Fields: user, video, author, tab, dur_bucket [+ hour_bucket [+ dow]]
    [+ one field per agent-discovered feature].

    `extra` is the extension point for autonomously discovered features:
        {feature_name: {"train": arr, "valid": arr, "test": arr}}
    Each is a float array with one value per row of that split. It is bucketed
    into EXTRA_FEATURE_BINS quantiles and appended as another categorical FM
    field, which is how every existing feature is represented -- so a
    discovered feature is a first-class citizen, not a bolted-on side channel.

    LEAKAGE SAFETY, by construction: the bin edges are computed from the TRAIN
    split ONLY and then applied unchanged to valid and test. A feature builder
    cannot widen its own bins using evaluation data even by accident. (It could
    still leak inside its own body, which is why the generated builder is put
    through the AST leakage checker before it ever runs.)
    """
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

    # agent-discovered features: train-only quantile edges, NaN -> its own bucket
    extra_edges = {}
    for fname in sorted(extra or {}):
        col = np.asarray((extra or {})[fname].get("train"), dtype=np.float64)
        finite = col[np.isfinite(col)]
        e = (np.unique(np.quantile(finite,
                                   np.linspace(0, 1, EXTRA_FEATURE_BINS + 1)[1:-1]))
             if finite.size else np.asarray([0.0]))
        extra_edges[fname] = e
        dims.append(len(e) + 2)          # buckets + 1 overflow + 1 missing
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
        for fname in sorted(extra_edges):
            v = np.asarray((extra or {})[fname].get(name), dtype=np.float64)
            e = extra_edges[fname]
            b = np.searchsorted(e, v).astype(np.int32)
            b[~np.isfinite(v)] = len(e) + 1          # missing gets its own bucket
            cols.append(b)
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


# --------------------------------------------------------------------------
# Numpy FM engine with pluggable loss / history / multitask
# --------------------------------------------------------------------------
class RankFM:
    """FM over categorical fields + optional stop-grad pooled history vector.

    logits = b + W[X].sum + 0.5*((S_f + H)^2 - sum E^2 - H^2)  (pure cross terms)
    Aux heads (multitask): z_t = b_t + W_t[X].sum + u_t · (S_f + H).
    """

    def __init__(self, dim, k=16, lr=1e-3, l2=1e-6, seed=0, aux_tasks=()):
        rng = np.random.default_rng(seed)
        self.V = rng.normal(0, 0.01, (dim, k)).astype(np.float32)
        self.W = np.zeros(dim, dtype=np.float32)
        self.b = np.float32(0.0)
        self.k, self.lr, self.l2 = k, lr, l2
        self.aux_tasks = list(aux_tasks)
        self.Wt = {t: np.zeros(dim, dtype=np.float32) for t in self.aux_tasks}
        self.ut = {t: rng.normal(0, 0.01, k).astype(np.float32) for t in self.aux_tasks}
        self.bt = {t: np.float32(0.0) for t in self.aux_tasks}
        self._adam = {}
        self.t = 0

    # -- adam over named params (dict id -> array) --
    def _step_param(self, name, P, G):
        m, v = self._adam.setdefault(name, (np.zeros_like(P), np.zeros_like(P)))
        b1, b2, eps = 0.9, 0.999, 1e-8
        m *= b1; m += (1 - b1) * G
        v *= b2; v += (1 - b2) * (G * G)
        P -= self.lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)

    def forward(self, X, H=None):
        E = self.V[X]                     # (B,F,k)
        S = E.sum(1)                      # (B,k)
        St = S if H is None else S + H
        inter = 0.5 * ((St ** 2).sum(1) - (E ** 2).sum((1, 2)))
        if H is not None:
            inter -= 0.5 * (H ** 2).sum(1)
        z = self.b + self.W[X].sum(1) + inter
        return z, (X, E, St)

    def aux_forward(self, task, cache):
        X, _, St = cache
        return self.bt[task] + self.Wt[task][X].sum(1) + St @ self.ut[task]

    def apply_grads(self, contribs, aux_contribs=()):
        """contribs: [(cache, g)] for the main head; aux: [(task, cache, g_t)].
        g arrays are dLoss/dz already divided by batch size."""
        gV = np.zeros_like(self.V)
        gW = np.zeros_like(self.W)
        gb = 0.0
        for (X, E, St), g in contribs:
            np.add.at(gW, X, g[:, None])
            np.add.at(gV, X, g[:, None, None] * (St[:, None, :] - E))
            gb += g.sum()
        gWt = {t: np.zeros_like(self.W) for t in self.aux_tasks}
        gut = {t: np.zeros_like(self.ut[t]) for t in self.aux_tasks}
        gbt = {t: 0.0 for t in self.aux_tasks}
        for task, (X, E, St), g in aux_contribs:
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
        self.b -= self.lr * np.float32(gb)
        for t_ in self.aux_tasks:
            self._step_param(f"Wt.{t_}", self.Wt[t_], gWt[t_])
            self._step_param(f"ut.{t_}", self.ut[t_], gut[t_])
            self.bt[t_] -= self.lr * np.float32(gbt[t_])

    def predict(self, X, H=None, bs=200_000):
        out = []
        for i in range(0, len(X), bs):
            h = None if H is None else H[i:i + bs]
            out.append(self.forward(X[i:i + bs], h)[0])
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def state(self):
        s = (self.V.copy(), self.W.copy(), np.float32(self.b),
             {t: self.Wt[t].copy() for t in self.aux_tasks},
             {t: self.ut[t].copy() for t in self.aux_tasks},
             dict(self.bt))
        return s

    def load_state(self, s):
        self.V, self.W, self.b = s[0].copy(), s[1].copy(), s[2]
        for t in self.aux_tasks:
            self.Wt[t] = s[3][t].copy()
            self.ut[t] = s[4][t].copy()
            self.bt[t] = s[5][t]


# ---- loss-specific epoch runners (numpy engine) ----
# Single source of truth for which heads each multitask option trains. Keep in
# step with _aux_targets below -- the two are read together.
AUX_MAP = {
    "none": [],
    "aux_click": ["click"],
    "aux_click_like_forward": ["click", "like", "forward"],
    "censored_watch_time": ["watch"],
    "aux_click_like_forward_watch": ["click", "like", "forward", "watch"],
    "aux_social4": ["follow", "comment", "hate", "profile_enter"],
}


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
    if multitask == "aux_click_like_forward_watch":
        # Candidate #3, in the only form compatible with a pairwise primary
        # loss: the graded play-time signal as an ADDITIONAL auxiliary head
        # beside the binary click/like/forward heads, leaving long_view as the
        # primary target. Replacing the primary target with a graded one would
        # require either grade-weighted pairs (structurally identical to
        # lambdarank_ndcg, measured here at -13.7 sigma) or pointwise
        # regression (already measured worse than BPR).
        #
        # The ratio is CLEANED before use, for measured reasons:
        #  * max observed ratio is 617842x duration, and rows with ratio > 5
        #    (1.85% of train) have a long_view rate of only 13.7% versus 100%
        #    at ratio 1-5 -- that tail is broken instrumentation, not
        #    engagement, and the binary label already rejects it. Feeding it
        #    raw would teach the model the opposite of the truth.
        #  * clipping at 2.0 keeps the monotone, informative band (long_view
        #    rate rises 50% -> 68% -> 83% -> 100% across ratio 0.2 -> 1.0)
        #    while discarding the corrupted tail.
        r = split_cols["play_time_ms"] / np.maximum(split_cols["duration_ms"], 1.0)
        r_clean = np.where(r > 5.0, 0.0, np.minimum(r, 2.0))
        return {"click": split_cols["is_click"], "like": split_cols["is_like"],
                "forward": split_cols["is_forward"],
                "watch": r_clean.astype(np.float32),
                "watch_censored": ((r >= 0.97) & (r <= 5.0)).astype(np.float32)}
    if multitask == "aux_social4":
        # The four feedback columns the pipeline never cached. Used ONLY as
        # binary auxiliary targets sharing the FM's embeddings; they are never
        # inputs. is_hate is deliberately included despite being rare (480
        # train positives, 0.042%): it is the only signal here that is
        # NEGATIVELY associated with long_view (P(long_view|hate)=0.246 against
        # a 0.337 base rate), so it carries a direction no currently-used
        # signal does. Prevalences: is_profile_enter 2.539%, is_comment 0.257%,
        # is_follow 0.101%, is_hate 0.042% -- for scale, is_forward, already in
        # use, has 0.100%.
        return {"follow": split_cols["is_follow"],
                "comment": split_cols["is_comment"],
                "hate": split_cols["is_hate"],
                "profile_enter": split_cols["is_profile_enter"]}
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

    hist = None
    Htr = Hva = Hte = None
    if cfg["history"] in ("mean_pool_positives", "recency_weighted_pool"):
        # tau_days was hardcoded at 3.0 -- a strong recency emphasis over a
        # 14-day train window, never tuned, and invisible to the menu even
        # though recency_weighted_pool is part of the incumbent. Now a config
        # knob so it can be swept like any other modelling choice.
        hist = History(splits, meta["field_dims"]["user"], cfg["history"],
                       tau_days=float(cfg.get("hist_tau_days", 3.0)))

    # AUX_MAP is module-level: this table was duplicated here and in run(), and
    # adding an option to one copy silently broke the other with a KeyError at
    # training time (found the hard way -- five treatment seeds failed).
    model = RankFM(dim=cfg["dim"], k=cfg["k"], lr=cfg["lr"], seed=cfg["seed"],
                   l2=cfg.get("l2", 1e-6),
                   aux_tasks=AUX_MAP[cfg["multitask"]])
    aux = _aux_targets(tr, cfg["multitask"])
    lam = cfg.get("aux_weight", 0.2)
    rng = np.random.default_rng(cfg["seed"])
    bs = cfg["bs"]

    # per-user row indices on train (for pairwise/listwise)
    n_users = meta["field_dims"]["user"]
    user_tr = tr["user"]
    order = np.argsort(user_tr, kind="stable")
    bounds = np.searchsorted(user_tr[order], np.arange(n_users + 1))
    # train-split impression count per video, for popularity-biased negatives
    item_pop = np.bincount(tr["video"], minlength=meta["field_dims"]["video"])

    def refresh_pooled():
        nonlocal Htr, Hva, Hte
        if hist is None:
            return
        # video embeddings live at offset of the video field (index 1)
        v_off = np.cumsum([0, meta["field_dims"]["user"]])[1]
        Vvid = model.V[v_off: v_off + meta["field_dims"]["video"]]
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
            z, cache = model.forward(Xtr[b], h)
            g = ((sigmoid(z) - ytr[b]) / len(b)).astype(np.float32)
            model.apply_grads([(cache, g)],
                              _aux_grad_contribs(model, cache, aux, b, lam))

    def _sample_pairs(mode="uniform_1"):
        """Build the (positive, negative) training pairs for a pairwise epoch.

        Separable from the loss SHAPE: BPR's objective and BPR's sampler are
        independent choices, and only the objective had ever been varied here.
        Measured motivation: each train user averages 14.7 positives and 28.9
        negatives, but uniform_1 draws exactly one negative per positive per
        epoch -- touching ~1/29 of the available negatives. Refs: Rendle et
        al., BPR (UAI 2009) for the uniform baseline; Rendle & Freudenthaler,
        'Improving Pairwise Learning for Item Recommendation from Implicit
        Feedback' (WSDM 2014), which shows uniform sampling is the weak point
        rather than the objective; Zhang et al., 'Optimizing Top-N
        Collaborative Filtering via Dynamic Negative Item Sampling' (SIGIR
        2013) for score-aware (hard) negatives.

        Returns (pos_rows, neg_rows), aligned and equal length.
        """
        pos_all = np.flatnonzero(ytr > 0)
        # --- bagging (candidate #5): row-level Poisson bootstrap ------------
        # Row-level, NOT user-level, and deliberately so. Measured constraints:
        #  * #1 showed that flattening user frequency HURTS monotonically
        #    (per_user_sqrt -0.00103, per_user_inv -0.00380 vs BPR), so the
        #    resampler must not implicitly equalise users. Poisson(1)
        #    multiplicities are drawn PER ROW independently of user identity,
        #    so expected rows per user stays exactly n_u -- the natural
        #    imbalance is preserved.
        #  * #4 showed ensemble members must be BOTH independent and
        #    comparably good. A user-level bootstrap would drop each user
        #    entirely from ~1/e = 37% of members, and with 100% train/valid
        #    user overlap those members would have untrained embeddings for
        #    37% of evaluated users -- not comparably good. Row-level keeps
        #    every user present (~63% of their distinct rows in expectation).
        # Ref: Breiman, "Bagging Predictors" (1996); Oza & Russell / Chamandy
        # et al. for the Poisson formulation used at scale.
        bag_seed = cfg.get("bootstrap_seed")
        if bag_seed is not None:
            brng = np.random.default_rng(int(bag_seed))
            mult = brng.poisson(1.0, size=len(ytr))
            pos_all = np.repeat(pos_all, mult[pos_all])
            keep_neg = mult > 0          # negatives present in this resample
        else:
            keep_neg = None
        n_per = {"uniform_1": 1, "uniform_2": 2, "uniform_4": 4}.get(mode, 1)
        scores = None
        if mode == "hard_negatives":
            scores = model.predict(Xtr, Htr)   # current model -> hardest negatives

        P, N = [], []
        for r in pos_all:
            u = user_tr[r]
            lo, hi = bounds[u], bounds[u + 1]
            rows_u = order[lo:hi]
            negs = rows_u[ytr[rows_u] == 0]
            if keep_neg is not None:
                negs = negs[keep_neg[negs]]   # negatives this member actually saw
            if not len(negs):
                continue
            if mode == "popularity_biased":
                # Popular items the user did NOT long-view are more informative
                # negatives than obscure ones: the impression was served and
                # declined, rather than simply never surfacing.
                w = item_pop[tr["video"][negs]].astype(np.float64) + 1.0
                pick = negs[rng.choice(len(negs), size=1, p=w / w.sum())]
            elif mode == "hard_negatives":
                # Dynamic negative sampling: prefer the negatives the model
                # currently scores highest -- the ones it is actively wrong about.
                k = min(len(negs), 8)
                top = negs[np.argpartition(-scores[negs], k - 1)[:k]]
                pick = top[rng.integers(k, size=1)]
            else:
                pick = negs[rng.integers(len(negs), size=n_per)]
            for q in np.atleast_1d(pick):
                P.append(r)
                N.append(q)
        return np.asarray(P, dtype=np.int64), np.asarray(N, dtype=np.int64)

    def _user_weights(pos_rows):
        """Per-pair weight implementing the sample_weighting axis.

        The metric aggregates PER USER (nDCG@5 averages every user equally;
        GAUC weights users by positive count), but training averages PER ROW.
        Measured on this train split: max 809 rows for one user vs a median of
        31, and the top 10% of users supply 33.3% of all rows -- so heavy users
        dominate the gradient in a way the metric never rewards. Weighting each
        pair by 1/n_u (or 1/sqrt(n_u)) makes the training objective aggregate
        the way the metric does. Weights are renormalised to mean 1.0 so the
        effective learning rate is unchanged and this is a pure reweighting,
        not a hidden lr change.
        """
        mode = cfg.get("sample_weighting", "per_row")
        if mode == "per_row":
            return None
        npairs = np.bincount(user_tr[pos_rows], minlength=n_users).astype(np.float64)
        per_user = npairs[user_tr[pos_rows]]
        if mode == "per_user_inv":
            w = 1.0 / np.maximum(per_user, 1.0)
        elif mode == "per_user_sqrt":
            w = 1.0 / np.sqrt(np.maximum(per_user, 1.0))
        else:
            return None
        return (w / w.mean()).astype(np.float32)

    def epoch_bpr():
        pos_rows, neg_pick = _sample_pairs(cfg.get("neg_sampling", "uniform_1"))
        uw = _user_weights(pos_rows)
        perm = rng.permutation(len(pos_rows))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            pb, nb = pos_rows[sel], neg_pick[sel]
            hp = None if Htr is None else Htr[pb]
            hn = None if Htr is None else Htr[nb]
            zp, cp = model.forward(Xtr[pb], hp)
            zn, cn = model.forward(Xtr[nb], hn)
            g = (-sigmoid(-(zp - zn)) / len(pb))
            if uw is not None:
                g = g * uw[sel]
            g = g.astype(np.float32)
            contribs = [(cp, g), (cn, -g)]
            contribs_aux = (_aux_grad_contribs(model, cp, aux, pb, lam)
                            + _aux_grad_contribs(model, cn, aux, nb, lam))
            model.apply_grads(contribs, contribs_aux)

    def epoch_lambdarank(uniform_weights=False, train_k=5):
        """BPR pairs reweighted by |delta nDCG@5| (LambdaRank / LambdaLoss).

        Identical pair enumeration to epoch_bpr above -- one sampled negative
        per positive, same user -- so the ONLY difference is the per-pair
        weight. BPR weights every pair 1.0, which makes a swap at ranks 1<->2
        count exactly as much as one at 50<->51; the evaluation metric does
        not agree with that. Here each pair is weighted by how much swapping
        those two items would actually change that user's nDCG@5, which is
        the quantity being scored.

        uniform_weights=True forces every weight to 1.0, which must reproduce
        epoch_bpr EXACTLY -- that is the degenerate-case test (see
        tests/test_harness.py test_lambdarank), and it is what proves any
        measured difference comes from the weighting rather than from an
        accidentally different sampling scheme.

        Weight derivation (binary labels, gain = 2^rel - 1, discount
        log2(rank+2), truncated at K=5, matching evaluate.py exactly and
        verified against it pair-by-pair):
            w = |gain_p - gain_n| * |invdisc(rank_p) - invdisc(rank_n)| / IDCG
        with invdisc(r) = 1/log2(r+2) for r < K and 0 beyond the cutoff, so a
        pair buried outside the top 5 on both sides contributes ~nothing.
        """
        # train_k is the TRAINING-time truncation, deliberately separable from
        # the metric's K=5. Measured reason: the evaluation split averages 5.6
        # impressions/user, but the TRAIN split averages 43.5 (median 31) --
        # 7.8x more. Truncating training pairs at 5 therefore zeroes ~78.6% of
        # them and collapses effective gradient magnitude to ~0.04x BPR's,
        # which underfits badly (measured: -0.011 primary, paired t=-20.4).
        # train_k=None keeps the log2(rank+2) discount but never truncates, so
        # every pair still carries signal while top positions stay upweighted.
        K = train_k if train_k is not None else 10 ** 9
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

        if uniform_weights:
            pair_w = np.ones(len(pos_rows), dtype=np.float32)
        else:
            # Current ranking per user, from the model as it stands this epoch.
            # One extra scoring pass over the train split (predict() batches
            # internally); cheaper than the positive-row Python loop above.
            z_all = model.predict(Xtr, Htr)
            rank_of = np.zeros(len(ytr), dtype=np.int32)
            idcg_u = np.zeros(n_users, dtype=np.float64)
            inv5 = np.array([1.0 / np.log2(i + 2) for i in range(5)], dtype=np.float64)
            for u in range(n_users):
                lo, hi = bounds[u], bounds[u + 1]
                if hi <= lo:
                    continue
                rows_u = order[lo:hi]
                o = np.argsort(-z_all[rows_u], kind="stable")
                rank_of[rows_u[o]] = np.arange(hi - lo, dtype=np.int32)
                ideal = np.sort(ytr[rows_u])[::-1][:5]   # metric K, not train_k
                idcg_u[u] = float(np.sum(((2.0 ** ideal) - 1.0) * inv5[:len(ideal)]))

            def invdisc(ranks):
                out = np.zeros(len(ranks), dtype=np.float64)
                inside = ranks < K
                out[inside] = 1.0 / np.log2(ranks[inside].astype(np.float64) + 2.0)
                return out

            gain_diff = ((2.0 ** ytr[pos_rows]) - 1.0) - ((2.0 ** ytr[neg_pick]) - 1.0)
            disc_diff = invdisc(rank_of[pos_rows]) - invdisc(rank_of[neg_pick])
            denom = idcg_u[user_tr[pos_rows]]
            pair_w = np.abs(gain_diff * disc_diff) / np.maximum(denom, 1e-12)
            pair_w = pair_w.astype(np.float32)

        perm = rng.permutation(len(pos_rows))
        for i in range(0, len(perm), bs):
            sel = perm[i:i + bs]
            pb, nb, wb = pos_rows[sel], neg_pick[sel], pair_w[sel]
            hp = None if Htr is None else Htr[pb]
            hn = None if Htr is None else Htr[nb]
            zp, cp = model.forward(Xtr[pb], hp)
            zn, cn = model.forward(Xtr[nb], hn)
            g = (-sigmoid(-(zp - zn)) * wb / len(pb)).astype(np.float32)
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
        z, cache = model.forward(Xtr[rows], h)
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
               "listwise_softmax_plus_pointwise": lambda: epoch_listwise(1.0),
               "lambdarank_ndcg": epoch_lambdarank,
               "lambdarank_ndcg_fulllist": lambda: epoch_lambdarank(train_k=None),
               # test-only degenerate mode: must reproduce bpr_pairwise exactly
               "_lambdarank_uniform": lambda: epoch_lambdarank(uniform_weights=True)}

    stages = [(cfg["loss"], cfg["epochs"], cfg["lr"])]
    if cfg["training"] == "two_stage_finetune" and cfg["loss"] != "pointwise_logloss":
        stages = [("pointwise_logloss", min(12, cfg["epochs"]), cfg["lr"]),
                  (cfg["loss"], cfg["epochs"], cfg["lr"] * 0.3)]

    best, best_state, bad = -1.0, None, 0
    # --- snapshot ensembling (candidate #4) ---------------------------------
    # Not a menu axis: it is a post-hoc scoring technique applicable to ANY
    # trained model, not a per-node search choice. Keeps the top-N epoch
    # checkpoints by valid primary and rank-averages their scores at the end.
    # Rationale, measured on this project: seed ensembling is the largest
    # verified win so far (+0.00157, +1.96 sigma, all 252 five-seed subsets
    # beat the best single seed) but costs N full training runs. Checkpoints
    # from a single run target the same variance-reduction mechanism at ~1x
    # cost, because SGD leaves each epoch's parameters at a different point.
    # Ref: Huang et al., "Snapshot Ensembles: Train 1, Get M For Free"
    # (ICLR 2017).
    snap_n = int(cfg.get("snapshot_ensemble", 0) or 0)
    snapshots = []   # (valid_primary, scores_valid, scores_test)

    def _rank_norm(x):
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=np.float64)
        r[o] = np.arange(len(x), dtype=np.float64)
        return r / max(1, len(x) - 1)

    for loss_name, n_ep, lr in stages:
        model.lr = lr
        bad = 0
        for ep in range(1, n_ep + 1):
            t0 = time.time()
            refresh_pooled()
            runners[loss_name]()
            refresh_pooled()
            sv = model.predict(Xva, Hva)
            va = evaluate(uva_raw, yva, sv)
            if cfg.get("capture_epoch_scores") is not None:
                cfg["capture_epoch_scores"].append((ep, float(va["primary"]),
                                                    sv.copy()))
            log(f"  [{loss_name}] epoch {ep:2d} | valid primary {va['primary']:.4f} "
                f"(GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f}) | {time.time()-t0:.1f}s")
            if snap_n:
                # test scores are computed only for retained snapshots, so the
                # cost stays proportional to snap_n rather than to epoch count
                snapshots.append((float(va["primary"]), sv.copy(),
                                  model.predict(Xte, Hte)))
                snapshots.sort(key=lambda s: -s[0])
                del snapshots[snap_n:]
            if va["primary"] > best + 1e-5:
                best, bad, best_state = va["primary"], 0, model.state()
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log(f"  early stop ({loss_name}) at epoch {ep}")
                    break

    model.load_state(best_state)
    refresh_pooled()
    scores_valid = model.predict(Xva, Hva)
    scores_test = model.predict(Xte, Hte)
    if snap_n and len(snapshots) > 1:
        sv = np.mean([_rank_norm(s[1]) for s in snapshots], axis=0)
        st_ = np.mean([_rank_norm(s[2]) for s in snapshots], axis=0)
        snap_primary = evaluate(uva_raw, yva, sv)["primary"]
        log(f"  snapshot ensemble of {len(snapshots)} checkpoints: "
            f"valid primary {snap_primary:.4f} (best single epoch {best:.4f})")
        # The guard below compares the snapshot against the best single
        # checkpoint ON THE SAME VALIDATION SET that chose that checkpoint --
        # a biased comparison, because argmax over ~20 epoch evaluations is
        # itself fitted to that set. Measured honestly (choose the epoch on one
        # half of validation, score on the other, 4 splits x both directions x
        # 3 seeds = 24 evaluations), averaging the top-5 checkpoints beats
        # argmax by +0.00069 (+0.87 sigma), t=5.54, winning 22/24. So
        # snapshot_force adopts it on that evidence instead of re-running the
        # biased test per run.
        if cfg.get("snapshot_force") or snap_primary > best:
            scores_valid, scores_test = sv, st_
        else:
            log("  snapshot ensemble did NOT beat the best checkpoint — keeping single")
    return {
        "scores_valid": scores_valid,
        "scores_test": scores_test,
        "model": model,
        "hist": hist,
    }


# --------------------------------------------------------------------------
# Torch engine (DeepFM / DCN-lite, optional DIN attention history)
# --------------------------------------------------------------------------
SEQ_LEN = 50   # covers p90 of train users (43.5 mean / 31 median impressions)


def build_causal_sequences(splits, n_users, seq_len=SEQ_LEN):
    """Per-user chronological sequence of TRAIN-period long_view items.

    Returns (seq_items, seq_times, seq_len_per_user), each (n_users, seq_len),
    left-aligned oldest -> newest so a left-to-right RNN reads them in time
    order. seq_times is kept so callers can apply a STRICT causal mask per
    row (position included only if its time < the scored row's time).

    Why times are stored rather than just a length:
      * A validation/test row can safely see the user's entire train history
        (train 20220408-21 strictly precedes valid 20220422-28 and test
        20220429-0508), so nothing is masked there.
      * A TRAIN row must NOT see itself or anything after it. Without a time
        mask, scoring a positive train row would include that very row in its
        own history -- the label leaking into its own features. The numpy
        History class already does leave-one-out for exactly this reason; the
        existing torch history path (hist_pad) does NOT, which is a real
        pre-existing leak in the deepfm/dcn/DIN path (those are a documented
        dead end, so it is noted here rather than silently changed).

    Sandbox note: this reads ONLY train-split positives (long_view from
    train.npz). runtime/data_boundary strips label columns from test.npz
    alone, and leaves train/valid intact -- so sequences build correctly
    inside the sandbox without ever needing a test label.
    """
    tr = splits["train"]
    pos = tr["long_view"] > 0
    pu, pv, pt = tr["user"][pos], tr["video"][pos], tr["time_ms"][pos]
    ordidx = np.lexsort((pt, pu))          # by user, then chronological
    pu, pv, pt = pu[ordidx], pv[ordidx], pt[ordidx]
    items = np.zeros((n_users, seq_len), dtype=np.int64)
    times = np.full((n_users, seq_len), np.iinfo(np.int64).max, dtype=np.int64)
    lens = np.zeros(n_users, dtype=np.int64)
    starts = np.searchsorted(pu, np.arange(n_users + 1))
    for u in range(n_users):
        lo, hi = starts[u], starts[u + 1]
        if hi <= lo:
            continue
        v, t = pv[lo:hi][-seq_len:], pt[lo:hi][-seq_len:]   # most recent seq_len
        items[u, :len(v)] = v
        times[u, :len(t)] = t
        lens[u] = len(v)
    return items, times, lens


def train_torch_seq(cfg, enc, splits, meta, log):
    """GRU4Rec-style sequential model: FM backbone + a GRU over the user's
    chronologically-ordered train history.

    Deliberately a SEPARATE function from train_torch rather than another
    branch inside it: the deepfm/dcn path is a documented dead end and adding
    a third model to its Net would put the one untested idea at risk of the
    same generated-code fragility that crashed 4 of 8 iterations there.

    Architecture choice -- GRU (Hidasi et al., "Session-based Recommendations
    with Recurrent Neural Networks", ICLR 2016) rather than a causal
    Transformer (Kang & McAuley, "Self-Attentive Sequential Recommendation",
    ICDM 2018), because a GRU has no positional embeddings and no causal
    attention mask, which are the two classic sources of silent look-ahead
    leakage; given this project's track record of real bugs surfacing on
    first live use, the smaller correctness surface is worth more than
    SASRec's marginal capacity. Sequential modelling is the intervention the
    dataset's own design implies -- KuaiRand is published as "An Unbiased
    Sequential Recommendation Dataset" (Gao et al., CIKM 2022) -- and no model
    tried on this project has used ORDER at all (FM/DeepFM/DCN treat history
    as a set; DIN attends over it but is order-invariant).

    The sequence signal is ADDITIVE on top of the FM score, not a replacement:
    score = bias + linear + FM_interaction + <gru_state, candidate_video_emb>.
    That follows the pattern that has actually worked here (broad, additive
    signal) rather than swapping out the backbone that is known to work.
    """
    import torch
    import torch.nn as nn
    torch.manual_seed(cfg["seed"])
    dev = cfg.get("device") or select_device()
    log(f"  torch device: {dev} (gru4rec_seq)")

    n_users = meta["field_dims"]["user"]
    v_off = n_users                      # video field offset in the shared table
    k = cfg["k"]
    tr = splits["train"]
    Xtr = torch.from_numpy(enc["train"]).long().to(dev)
    Xva = torch.from_numpy(enc["valid"]).long().to(dev)
    Xte = torch.from_numpy(enc["test"]).long().to(dev)
    ytr = torch.from_numpy(tr["long_view"]).to(dev)
    uva_raw = list(splits["valid"]["user_raw"])
    yva = splits["valid"]["long_view"]

    items, times, lens = build_causal_sequences(splits, n_users)
    seq_items = torch.from_numpy(items + v_off).to(dev)      # (U, L) into emb table
    seq_times = torch.from_numpy(times).to(dev)
    seq_lens = torch.from_numpy(lens).to(dev)
    L = seq_items.shape[1]

    users_tr = torch.from_numpy(tr["user"].astype(np.int64)).to(dev)
    users_va = torch.from_numpy(splits["valid"]["user"].astype(np.int64)).to(dev)
    users_te = torch.from_numpy(splits["test"]["user"].astype(np.int64)).to(dev)
    times_tr = torch.from_numpy(tr["time_ms"].astype(np.int64)).to(dev)

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(cfg["dim"], k)
            nn.init.normal_(self.emb.weight, 0, 0.01)
            self.lin = nn.Embedding(cfg["dim"], 1)
            nn.init.zeros_(self.lin.weight)
            self.bias = nn.Parameter(torch.zeros(1))
            self.gru = nn.GRU(k, k, batch_first=True)

        def seq_state(self, users, row_times):
            """Causal GRU state for each row. row_times=None => eval rows, which
            may use the user's FULL train history (the split is defined by DATE
            and train strictly precedes valid/test by date; a raw time_ms
            comparison is NOT used because date and time_ms are ~1h skewed, so
            0.04% of valid rows carry a time_ms below the train maximum)."""
            s_it = seq_items[users]                       # (B, L)
            valid = (torch.arange(L, device=dev)[None, :] < seq_lens[users][:, None])
            if row_times is not None:                     # TRAIN rows: strict causal
                valid = valid & (seq_times[users] < row_times[:, None])
            e = self.emb(s_it) * valid.unsqueeze(-1).float()
            out, _ = self.gru(e)                          # (B, L, k)
            idx = valid.sum(1).long()                     # count of usable steps
            has = idx > 0
            gather = (idx - 1).clamp(min=0)
            st = out[torch.arange(len(idx), device=dev), gather]
            return st * has.unsqueeze(-1).float()         # zero when no history

        def forward(self, X, users, row_times):
            E = self.emb(X)
            S = E.sum(1)
            fm = 0.5 * ((S ** 2).sum(1) - (E ** 2).sum((1, 2)))
            first = self.lin(X).sum((1, 2)) + self.bias
            st = self.seq_state(users, row_times)
            cand = self.emb(X[:, 1])                      # candidate video embedding
            return first + fm + (st * cand).sum(1)

    net = SeqNet().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    bce = nn.BCEWithLogitsLoss()
    bs = min(cfg["bs"], 4096)                             # GRU is the memory driver
    rng = np.random.default_rng(cfg["seed"])
    user_np = tr["user"]
    order = np.argsort(user_np, kind="stable")
    bounds = np.searchsorted(user_np[order], np.arange(n_users + 1))

    def predict(X, users, bs_=4096):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X), bs_):
                out.append(net(X[i:i + bs_], users[i:i + bs_], None))
        net.train()
        return torch.cat(out).cpu().numpy()

    def epoch_pointwise():
        idx = torch.randperm(len(ytr), device=dev)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            loss = bce(net(Xtr[b], users_tr[b], times_tr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()

    def epoch_bpr():
        pos_rows = np.flatnonzero(tr["long_view"] > 0)
        P, N = [], []
        for r in pos_rows:
            rows_u = order[bounds[user_np[r]]:bounds[user_np[r] + 1]]
            negs = rows_u[tr["long_view"][rows_u] == 0]
            if len(negs):
                P.append(r); N.append(negs[rng.integers(len(negs))])
        P = torch.from_numpy(np.asarray(P)).to(dev)
        N = torch.from_numpy(np.asarray(N)).to(dev)
        perm = torch.randperm(len(P), device=dev)
        for i in range(0, len(perm), bs):
            pb, nb = P[perm[i:i + bs]], N[perm[i:i + bs]]
            zp = net(Xtr[pb], users_tr[pb], times_tr[pb])
            zn = net(Xtr[nb], users_tr[nb], times_tr[nb])
            loss = torch.nn.functional.softplus(-(zp - zn)).mean()
            opt.zero_grad(); loss.backward(); opt.step()

    runner = epoch_bpr if cfg["loss"] == "bpr_pairwise" else epoch_pointwise
    best, best_sd, bad = -1.0, None, 0
    for ep in range(1, min(cfg["epochs"], 12) + 1):
        t0 = time.time()
        runner()
        va = evaluate(uva_raw, yva, predict(Xva, users_va))
        log(f"  [gru4rec_seq/{cfg['loss']}] epoch {ep:2d} | valid primary "
            f"{va['primary']:.4f} (GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f})"
            f" | {time.time()-t0:.1f}s")
        if va["primary"] > best + 1e-5:
            best, bad = va["primary"], 0
            best_sd = {kk: v.detach().clone() for kk, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= min(cfg["patience"], 3):
                log(f"  early stop at epoch {ep}")
                break
    net.load_state_dict(best_sd)
    return {"scores_valid": predict(Xva, users_va),
            "scores_test": predict(Xte, users_te)}


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
    # Agent-discovered features arrive as SOURCE for a build_features() function
    # rather than as data, so the exact code that produced them is recorded with
    # the experiment and the result is reproducible from the journal alone.
    extra = None
    fsrc = menu_choices.get("feature_source")
    if fsrc:
        extra = build_extra_features(fsrc, splits, meta)
        log(f"agent features: {sorted(extra)}")
    enc, dim, offsets, dims = encode_features(splits, meta,
                                              menu_choices.get("temporal", "none"),
                                              extra=extra)

    training = menu_choices.get("training", "default")
    cfg = {
        "dim": dim, "k": 32 if training == "k32" else 16,
        "lr": 5e-4 if training == "lower_lr_longer" else 1e-3,
        "bs": 8192,
        "epochs": 60 if training == "lower_lr_longer" else 40,
        "patience": 6 if training == "lower_lr_longer" else 4,
        "seed": seed,
        "loss": menu_choices.get("loss", "pointwise_logloss"),
        "history": menu_choices.get("user_history", "none"),
        "multitask": menu_choices.get("multitask", "none"),
        "model": menu_choices.get("model", "fm_numpy"),
        "training": training,
        "neg_sampling": menu_choices.get("neg_sampling", "uniform_1"),
        "sample_weighting": menu_choices.get("sample_weighting", "per_row"),
        "l2": {"l2_default": 1e-6, "l2_1e5": 1e-5, "l2_1e4": 1e-4,
               "l2_1e3": 1e-3}.get(menu_choices.get("regularization", "l2_default"), 1e-6),
        # not a menu axis -- post-hoc scoring technique, set via config/CLI
        "snapshot_ensemble": menu_choices.get("snapshot_ensemble", 0),
        # not a menu axis -- bagging is an ensemble-construction mechanism,
        # meaningful only across MULTIPLE runs, not within one node
        "bootstrap_seed": menu_choices.get("bootstrap_seed"),
    }
    cfg["aux_tasks"] = AUX_MAP[cfg["multitask"]]
    # Agent-set pipeline overrides. These are parts of the pipeline the MENU
    # cannot express -- embedding size, the history decay constant, the
    # stopping rule -- and the Opus research run showed they are where the
    # untested assumptions live. Validated at the menu boundary
    # (agent/menu.py PIPELINE_OVERRIDES), applied here.
    for _k in ("k", "lr", "epochs", "patience", "l2", "bs", "hist_tau_days",
               "aux_weight", "snapshot_ensemble", "snapshot_force"):
        if _k in menu_choices:
            cfg[_k] = menu_choices[_k]
            log(f"  pipeline override: {_k}={menu_choices[_k]}")

    t_train = time.time()
    if cfg["model"] == "fm_numpy":
        device = "cpu"                       # the numpy engine never uses a GPU
        res = train_numpy_fm(cfg, enc, splits, meta, log)
    else:
        cfg["epochs"] = min(cfg["epochs"], 12)
        cfg["patience"] = min(cfg["patience"], 3)
        device = cfg.get("device") or select_device()
        cfg["device"] = device
        res = (train_torch_seq(cfg, enc, splits, meta, log)
               if cfg["model"] == "gru4rec_seq"
               else train_torch(cfg, enc, splits, meta, log))
    train_seconds = time.time() - t_train
    # Measured compute, read back by the harness for the GPU-hours report.
    write_resource_json(output_dir, device, train_seconds)

    sv, st = np.asarray(res["scores_valid"]), np.asarray(res["scores_test"])
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
