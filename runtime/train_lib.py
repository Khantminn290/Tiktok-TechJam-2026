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
_CACHE_COLS = ["user", "video", "author", "tab", "duration_ms", "hourmin", "date",
               "time_ms", "long_view", "is_click", "is_like", "is_forward",
               "play_time_ms"]


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
def encode_features(splits: dict, meta: dict, temporal: str = "none"):
    """Fields: user, video, author, tab, dur_bucket [+ hour_bucket [+ dow]]."""
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

    hist = None
    Htr = Hva = Hte = None
    if cfg["history"] in ("mean_pool_positives", "recency_weighted_pool"):
        hist = History(splits, meta["field_dims"]["user"], cfg["history"])

    aux_map = {"aux_click": ["click"], "aux_click_like_forward": ["click", "like", "forward"],
               "censored_watch_time": ["watch"], "none": []}
    model = RankFM(dim=cfg["dim"], k=cfg["k"], lr=cfg["lr"], seed=cfg["seed"],
                   aux_tasks=aux_map[cfg["multitask"]])
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
            zp, cp = model.forward(Xtr[pb], hp)
            zn, cn = model.forward(Xtr[nb], hn)
            g = (-sigmoid(-(zp - zn)) / len(pb)).astype(np.float32)
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
               "listwise_softmax_plus_pointwise": lambda: epoch_listwise(1.0)}

    stages = [(cfg["loss"], cfg["epochs"], cfg["lr"])]
    if cfg["training"] == "two_stage_finetune" and cfg["loss"] != "pointwise_logloss":
        stages = [("pointwise_logloss", min(12, cfg["epochs"]), cfg["lr"]),
                  (cfg["loss"], cfg["epochs"], cfg["lr"] * 0.3)]

    best, best_state, bad = -1.0, None, 0
    for loss_name, n_ep, lr in stages:
        model.lr = lr
        bad = 0
        for ep in range(1, n_ep + 1):
            t0 = time.time()
            refresh_pooled()
            runners[loss_name]()
            refresh_pooled()
            va = evaluate(uva_raw, yva, model.predict(Xva, Hva))
            log(f"  [{loss_name}] epoch {ep:2d} | valid primary {va['primary']:.4f} "
                f"(GAUC {va['GAUC']:.4f} nDCG@5 {va['nDCG@5']:.4f}) | {time.time()-t0:.1f}s")
            if va["primary"] > best + 1e-5:
                best, bad, best_state = va["primary"], 0, model.state()
            else:
                bad += 1
                if bad >= cfg["patience"]:
                    log(f"  early stop ({loss_name}) at epoch {ep}")
                    break

    model.load_state(best_state)
    refresh_pooled()
    return {
        "scores_valid": model.predict(Xva, Hva),
        "scores_test": model.predict(Xte, Hte),
        "model": model,
        "hist": hist,
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
    enc, dim, offsets, dims = encode_features(splits, meta,
                                              menu_choices.get("temporal", "none"))

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
    }
    aux_map = {"aux_click": ["click"], "aux_click_like_forward": ["click", "like", "forward"],
               "censored_watch_time": ["watch"], "none": []}
    cfg["aux_tasks"] = aux_map[cfg["multitask"]]

    t_train = time.time()
    if cfg["model"] == "fm_numpy":
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
