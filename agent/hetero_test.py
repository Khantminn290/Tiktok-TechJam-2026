"""PRE-REGISTERED test: does ensembling two COMPARABLY GOOD configurations help?

The standing lesson on this project is that ensemble members must be both
independent AND comparably good. It was learned from two failures that each
satisfied only one half:

    snapshot ensembling  quality, no independence  -> lost
    bagged members       independence, no quality  -> lost (-0.0028/member)
    gru4rec_seq blend    independence (corr 0.9338 vs 0.983 for same-config
                         seeds), but 2.1 sigma weaker  -> +0.15 sigma, inside
                         the noise floor

Every heterogeneous test so far has therefore been a test of the WRONG
condition. The case the rule actually leaves open -- genuinely different
members that are ALSO statistically indistinguishable in quality -- has never
been run. multitask=aux_click_like_forward sits 0.43 sigma from the incumbent
(0.60463 vs 0.60497), which is inside seed noise, while auxiliary heads reshape
the learned representation rather than merely reseeding it.

DESIGN, FIXED BEFORE ANY RESULT WAS SEEN (this matters more than the outcome):

  * members  : ALL 16 base seeds + ALL 16 aux seeds
  * weights  : equal. No weight sweep -- a weight chosen on validation is
               selection, and the gru4rec test already showed a swept weight
               producing a +0.00012 "gain" that was really a fitted parameter.
  * k        : fixed, not searched. No best-subset selection anywhere.
  * decision : adopt only if the combined ensemble beats base-only by more than
               half the noise floor (0.0004). Anything less is not a result.

Exactly ONE comparison is made against validation. That is recorded here so the
researcher-overfitting ledger stays honest: this is comparison #1 for this
hypothesis, not the best of several.

Usage: python3 -m agent.hetero_test
"""
from __future__ import annotations

import json
import os
import statistics
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent.ensemble import load_valid_targets, rank_normalise, _evaluator  # noqa: E402

BASE_DIR = os.path.join(ROOT, "logs", "final_ensemble")
AUX_DIR = os.path.join(ROOT, "logs", "ensemble_aux")
NOISE = 0.0008
ADOPT_THRESHOLD = NOISE / 2          # fixed in advance
OUT = os.path.join(ROOT, "logs", "hetero_test.json")


def _members(d: str) -> list:
    out = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name, "scores_valid.npy")
        if os.path.exists(p):
            out.append((name, np.load(p)))
    return out


def main() -> None:
    ev = _evaluator()
    users, labels = load_valid_targets()
    base, aux = _members(BASE_DIR), _members(AUX_DIR)
    if len(base) < 2 or len(aux) < 2:
        raise SystemExit(f"need >=2 members each; have base={len(base)} aux={len(aux)}")

    def score(a):
        return float(ev(users, labels, a)["primary"])

    def full(a):
        m = ev(users, labels, a)
        return {k: round(float(v), 5) for k, v in m.items()}

    rb = [rank_normalise(a) for _, a in base]
    ra = [rank_normalise(a) for _, a in aux]
    Eb, Ea = np.mean(rb, axis=0), np.mean(ra, axis=0)
    # equal weight per MEMBER, so the arm with more members is not penalised
    Ec = np.mean(rb + ra, axis=0)

    sb, sa = [score(x) for x in rb], [score(x) for x in ra]
    corr_within = float(np.corrcoef(rb[0], rb[1])[0, 1])
    corr_across = float(np.corrcoef(Eb, Ea)[0, 1])

    base_m, comb_m = full(Eb), full(Ec)
    delta = comb_m["primary"] - base_m["primary"]

    r = {
        "hypothesis": ("ensembling two configurations that are decorrelated AND "
                       "statistically indistinguishable in quality beats seeds of "
                       "one configuration"),
        "design_fixed_before_result": True,
        "validation_comparisons_for_this_hypothesis": 1,
        "base": {"members": len(base), "ensemble": base_m,
                 "member_mean": round(statistics.mean(sb), 5),
                 "member_std": round(statistics.pstdev(sb), 5)},
        "aux": {"members": len(aux), "ensemble": full(Ea),
                "member_mean": round(statistics.mean(sa), 5),
                "member_std": round(statistics.pstdev(sa), 5)},
        "quality_gap_sigma": round((statistics.mean(sb) - statistics.mean(sa)) / NOISE, 2),
        "corr_within_config_seeds": round(corr_within, 4),
        "corr_across_configs": round(corr_across, 4),
        "combined": {"members": len(base) + len(aux), "ensemble": comb_m},
        "delta_vs_base": round(delta, 5),
        "delta_sigma": round(delta / NOISE, 2),
        "adopt_threshold": ADOPT_THRESHOLD,
        "adopt": bool(delta > ADOPT_THRESHOLD),
    }
    # Confound check, stated rather than assumed: combined has 32 members and
    # base has 16, so a naive reading could credit MEMBER COUNT for any gain.
    # The base k-curve is the control: it is already flat from k=5 to k=16
    # (0.60491-0.60563, no trend), so 16 more seeds of the SAME config add
    # approximately nothing. Any real gain is therefore attributable to the
    # second configuration, not to the extra members.
    r["member_count_confound"] = (
        "combined k=32 vs base k=16. The base k-curve is flat from k=5 to k=16 "
        "with no trend, so additional same-config seeds add ~0; a gain here is "
        "attributable to the second configuration, not to member count.")
    r["verdict"] = ("ADOPT -- combined beats base beyond the pre-set threshold"
                    if r["adopt"] else
                    "REJECT -- gain does not clear the pre-set threshold; the "
                    "incumbent 16-seed ensemble stands")
    with open(OUT, "w") as fh:
        json.dump(r, fh, indent=2)

    print("=" * 72)
    print("PRE-REGISTERED HETEROGENEOUS ENSEMBLE TEST")
    print("=" * 72)
    print(f"  base  k={r['base']['members']:2d}  members {r['base']['member_mean']} "
          f"+/- {r['base']['member_std']}   ensemble {base_m['primary']}")
    print(f"  aux   k={r['aux']['members']:2d}  members {r['aux']['member_mean']} "
          f"+/- {r['aux']['member_std']}   ensemble {r['aux']['ensemble']['primary']}")
    print(f"\n  quality gap            {r['quality_gap_sigma']:+.2f} sigma "
          f"(gru4rec, which failed, was 2.1)")
    print(f"  corr within-config     {r['corr_within_config_seeds']}")
    print(f"  corr across-config     {r['corr_across_configs']}  "
          f"(lower = more independent)")
    print(f"\n  COMBINED k={r['combined']['members']}          {comb_m['primary']}  "
          f"(GAUC {comb_m['GAUC']}, nDCG@5 {comb_m['nDCG@5']})")
    print(f"  vs base                {delta:+.5f}  ({r['delta_sigma']:+.2f} sigma)")
    print(f"\n  {r['verdict']}")
    print(f"  confound: {r['member_count_confound']}")
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
