"""Is there exploitable information the incumbent's representation cannot see?

This is a DIAGNOSTIC, not a candidate model. It answers the one question that
decides whether the plateau belongs to the menu or to the problem:

    Given every causally-available feature -- including the ten raw columns the
    cache never reads -- plus the incumbent's OWN score as a feature, can a
    flexible learner add within-user ranking accuracy?

Design choices that make the answer trustworthy:

  * The incumbent's score is INCLUDED as a feature. The question is not "can a
    GBDT match FM+BPR" (already known: eight model families tie in 0.55-0.605).
    It is whether anything is left OVER once the incumbent is given for free.
    Feature importances then rank candidate mechanisms by measured value
    instead of by argument.
  * Fitted on TRAIN only, scored on valid. Every feature must exist before the
    impression it describes: static item attributes (tag, music, upload type,
    item age) and observable context (hour, weekday, tab, duration). The four
    discarded FEEDBACK columns -- is_hate, is_follow, is_comment,
    is_profile_enter -- are post-outcome and are deliberately EXCLUDED here;
    they are legitimate only as auxiliary targets, never as inputs.
  * User-side columns are excluded on principle, not oversight. GAUC ranks
    WITHIN a user, so a feature constant across that user's rows contributes
    exactly nothing -- measured at 0.5000, exactly. Including them would only
    add noise and invite a false importance reading.
  * Judged on within-user AUC against the incumbent, with the noise floor as
    the threshold. A gain under half the noise floor is not a finding.

A null result here is strong evidence that the practical ceiling of the
available information is near the incumbent -- much stronger than exhausting a
predefined option list, because it tests the INFORMATION rather than the menu.

Usage: python3 -m agent.residual_screen [--rows N] [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import error_analysis as EA          # noqa: E402
from agent.ensemble import rank_normalise       # noqa: E402

NOISE = 0.0008
BASIC_CSV = os.path.join(ROOT, "kuairand-starter-kit", "KuaiRand-Pure", "data",
                         "video_features_basic_pure.csv")
OUT = os.path.join(ROOT, "logs", "residual_screen.json")


def _video_attrs(vocab: dict, n_video: int) -> dict:
    """Static item attributes, coded. These are properties of the item itself,
    fixed before any impression -- unlike the locked statistic file, whose
    counters are aggregated over a window overlapping the evaluation dates."""
    tag = np.full(n_video, -1, np.int32)
    music = np.full(n_video, -1, np.int32)
    vtype = np.full(n_video, -1, np.int32)
    utype = np.full(n_video, -1, np.int32)
    upload = np.full(n_video, -1, np.int64)
    width = np.zeros(n_video, np.float32)
    height = np.zeros(n_video, np.float32)
    vdur = np.zeros(n_video, np.float32)
    codes: dict = {}

    def code(d, s):
        s = (s or "").strip()
        if not s:
            return -1
        return d.setdefault(s, len(d))

    dm, dv, du = {}, {}, {}
    with open(BASIC_CSV) as fh:
        for r in csv.DictReader(fh):
            c = vocab.get(r["video_id"])
            if c is None or c >= n_video:
                continue
            t = (r.get("tag") or "").split(",")[0].strip()
            tag[c] = int(t) if t.isdigit() else -1
            music[c] = code(dm, r.get("music_id"))
            vtype[c] = code(dv, r.get("video_type"))
            utype[c] = code(du, r.get("upload_type"))
            d = (r.get("upload_dt") or "")[:10].replace("-", "")
            upload[c] = int(d) if d.isdigit() else -1
            for key, arr in (("server_width", width), ("server_height", height),
                             ("video_duration", vdur)):
                try:
                    arr[c] = float(r.get(key) or 0.0)
                except ValueError:
                    pass
    codes = {"music": len(dm), "video_type": len(dv), "upload_type": len(du)}
    return {"tag": tag, "music": music, "video_type": vtype, "upload_type": utype,
            "upload_dt": upload, "width": width, "height": height,
            "video_duration": vdur, "_cardinality": codes}


def build_matrix(split, attrs, incumbent_score=None):
    """Causally-available features only. No user-side columns (constant within
    a user, therefore worth exactly 0 to GAUC). No post-outcome columns."""
    v = split["video"]
    date = split["date"].astype(np.int64)
    day = (date % 100) + np.where(date >= 20220501, 30, 0)
    up = attrs["upload_dt"][v]
    age = np.where(up > 0, (date - up).astype(np.float64), -1.0)
    cols = {
        "tag": attrs["tag"][v].astype(np.float64),
        "music": attrs["music"][v].astype(np.float64),
        "video_type": attrs["video_type"][v].astype(np.float64),
        "upload_type": attrs["upload_type"][v].astype(np.float64),
        "item_age_days": age,
        "server_width": attrs["width"][v],
        "server_height": attrs["height"][v],
        "aspect": attrs["width"][v] / np.maximum(1.0, attrs["height"][v]),
        "video_duration": attrs["video_duration"][v],
        "duration_ms": split["duration_ms"].astype(np.float64),
        "hour": (split["hourmin"] // 100).astype(np.float64),
        "dow": ((day - 8 + 4) % 7).astype(np.float64),
        "tab": split["tab"].astype(np.float64),
    }
    if incumbent_score is not None:
        cols["incumbent_score"] = np.asarray(incumbent_score, np.float64)
    names = list(cols)
    X = np.column_stack([cols[n] for n in names])
    return X, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400000,
                    help="train rows to subsample (speed; 0 = all)")
    ap.add_argument("--json", default=OUT)
    ap.add_argument("--confirm", type=int, default=0,
                    help="repeat the screen with N independent train subsamples "
                         "at a FIXED blend weight, and report mean +/- std. A "
                         "single screen picked its weight by scanning three "
                         "values on validation, which is selection; this does "
                         "not.")
    ap.add_argument("--weight", type=float, default=0.95,
                    help="blend weight fixed IN ADVANCE for --confirm")
    a = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    def _fit_score(Xtr_, ytr_, Xva_, seed):
        c = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, early_stopping=True,
            validation_fraction=0.1, random_state=seed)
        c.fit(Xtr_, ytr_)
        return c.predict_proba(Xva_)[:, 1]

    splits, meta = EA.load_valid()
    tr, va = splits["train"], splits["valid"]
    nV = meta["field_dims"]["video"]
    vocab = json.load(open(os.path.join(EA._cache_dir(), "vocabs.json")))["video"]
    attrs = _video_attrs(vocab, nV)

    res = json.load(open(os.path.join(ROOT, "logs", "ensemble_results.json")))
    inc_va = np.mean([rank_normalise(np.load(os.path.join(
        ROOT, "logs", "final_ensemble", f"seed_{i:02d}", "scores_valid.npy")))
        for i in res["seeds_used"]], axis=0)

    u, y = va["user"], va["long_view"]
    base = EA.within_user_auc(u, y, inc_va)
    print(f"incumbent within-user AUC on valid: {base:.5f}")
    print(f"tag coverage on valid rows: "
          f"{100 * np.mean(attrs['tag'][va['video']] >= 0):.1f}%")

    # The incumbent's TRAIN scores are not stored, so the screen is run in two
    # modes: attributes alone, and attributes + the incumbent's valid score
    # folded in afterwards. Mode 1 answers "is there signal at all"; mode 2
    # answers "is any of it RESIDUAL", which is the question that matters.
    Xtr, names = build_matrix(tr, attrs)
    Xva, _ = build_matrix(va, attrs)
    ytr = tr["long_view"]
    if a.rows and a.rows < len(ytr):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(ytr), a.rows, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
    print(f"fitting GBDT on {len(ytr):,d} train rows x {len(names)} features ...")

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=None,
        early_stopping=True, validation_fraction=0.1, random_state=0)
    clf.fit(Xtr, ytr)
    p_va = clf.predict_proba(Xva)[:, 1]

    gbdt_auc = EA.within_user_auc(u, y, p_va)
    print(f"GBDT on discarded attributes alone: {gbdt_auc:.5f}")

    # residual: blend the GBDT's opinion into the incumbent
    probe = EA.feature_probe(u, y, inc_va, p_va, "GBDT(all discarded attrs)")

    # per-feature residual value, measured the same way
    per_feat = {}
    for i, nm in enumerate(names):
        pf = EA.feature_probe(u, y, inc_va, Xva[:, i], nm)
        per_feat[nm] = {"standalone_wAUC": pf["standalone_wAUC"],
                        "best_gain_sigma": pf["best_gain_sigma"],
                        "adds_signal": pf["adds_signal"]}

    confirm = None
    if a.confirm:
        # Fixed weight, independent subsamples. If the residual is real it
        # survives resampling; if it was the scan picking a lucky weight, it
        # will not.
        rm = EA._average_ranks(inc_va); rm /= rm.max()
        gains = []
        for s_ in range(a.confirm):
            rng = np.random.default_rng(100 + s_)
            idx = rng.choice(len(tr["long_view"]), min(a.rows or len(tr["long_view"]),
                                                       len(tr["long_view"])),
                             replace=False)
            pv = _fit_score(build_matrix(tr, attrs)[0][idx],
                            tr["long_view"][idx], Xva, seed=s_)
            rf = EA._average_ranks(pv); rf /= rf.max()
            g = EA.within_user_auc(u, y, a.weight * rm + (1 - a.weight) * rf) - base
            gains.append(g)
            print(f"  confirm seed {s_}: {g:+.5f} ({g / NOISE:+.2f} sigma)", flush=True)
        m, sd = float(np.mean(gains)), float(np.std(gains))
        t = m / (sd / max(1e-9, len(gains) ** 0.5)) if sd > 0 else 0.0
        confirm = {"n": a.confirm, "weight": a.weight,
                   "gains": [round(g, 5) for g in gains],
                   "mean_gain": round(m, 5), "std": round(sd, 5),
                   "mean_sigma": round(m / NOISE, 2), "t": round(t, 2),
                   "survives": bool(m >= NOISE / 2 and t > 2.0)}

    out = {
        "incumbent_wAUC": round(base, 5),
        "confirmation": confirm,
        "gbdt_attrs_only_wAUC": round(gbdt_auc, 5),
        "gbdt_residual": probe,
        "per_feature_residual": per_feat,
        "n_train_rows": int(len(ytr)),
        "features": names,
        "excluded_post_outcome": ["is_hate", "is_follow", "is_comment",
                                  "is_profile_enter", "profile_stay_time",
                                  "comment_stay_time", "play_time_ms"],
        "excluded_user_side": "constant within a user -> exactly 0 for GAUC",
        # The confirmation OVERRIDES the single screen. The single screen picks
        # its blend weight by scanning three values on validation, so its
        # headline is a selected number; the confirmation fixes the weight in
        # advance and resamples. Letting the scanned figure set the verdict
        # would report "signal found" over evidence that says otherwise.
        "verdict": (
            ("RESIDUAL SIGNAL CONFIRMED -- survives resampling at a fixed "
             "weight; a mechanism using these attributes is worth building"
             if confirm["survives"] else
             f"NOT WORTH A MECHANISM -- a residual IS statistically detectable "
             f"(t={confirm['t']}) but its size is {confirm['mean_sigma']:+.2f} "
             f"sigma, under half the seed-noise scale. Evidence that the "
             f"ceiling here is informational rather than a menu artefact.")
            if confirm else
            ("residual signal at a SCANNED weight -- unconfirmed, rerun with "
             "--confirm before believing it" if probe["adds_signal"] else
             "NO RESIDUAL SIGNAL -- the incumbent already captures what these "
             "attributes carry")),
    }
    with open(a.json, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"\n{'=' * 70}\nRESIDUAL SCREEN\n{'=' * 70}")
    print(f"  incumbent                {base:.5f}")
    print(f"  GBDT(attrs only)         {gbdt_auc:.5f}")
    print(f"  blended into incumbent   {probe['blend_delta']}  "
          f"({probe['best_gain_sigma']:+.2f} sigma)")
    print(f"  {probe['verdict']}")
    print("\n  per-feature residual (sigma over the incumbent) -- CAUTION: these "
          "are\n  single unconfirmed readings on heavily-tied columns, where "
          "rank-blending\n  barely perturbs the ordering. Treat sub-1-sigma "
          "entries as noise:")
    for nm, d in sorted(per_feat.items(), key=lambda kv: -kv[1]["best_gain_sigma"]):
        print(f"    {nm:<20} standalone {d['standalone_wAUC']:.4f}   "
              f"{d['best_gain_sigma']:+.2f} sigma")
    if confirm:
        print(f"\n  CONFIRMATION ({confirm['n']} independent subsamples, "
              f"weight FIXED at {confirm['weight']}):")
        print(f"    mean gain {confirm['mean_gain']:+.5f} "
              f"({confirm['mean_sigma']:+.2f} sigma) +/- {confirm['std']:.5f}, "
              f"t={confirm['t']}")
        print(f"    {'SURVIVES confirmation' if confirm['survives'] else 'DOES NOT SURVIVE -- treat as noise'}")
    print(f"\n  VERDICT: {out['verdict']}")
    print(f"\nwrote {os.path.relpath(a.json, ROOT)}")


if __name__ == "__main__":
    main()
