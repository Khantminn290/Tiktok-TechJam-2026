# Opus research journal — hunting a legitimate improvement over 0.60541

Incumbent, never overwritten by an experiment:
  primary 0.60541  GAUC 0.67212  nDCG@5 0.53870  (k=16 seed ensemble, no selection)
  single-model mean 0.60446 +/- 0.00042 over 5 seeds

Noise floor sigma = 0.0008 (the official baseline's own 5-seed std).
Nothing is promoted on a single seed.

---

## E1 — Training-dynamics diagnosis (patience disabled)
**Observation motivating it:** every menu-level intervention on record is null, yet
the learning curve says the model is data-limited. Those are only consistent if
something else is binding. Nobody had ever looked at the epoch curve.
**Change:** train the incumbent with `patience=60` so the whole 60-epoch curve is
visible instead of stopping at ~20.
**Result:** peak **epoch 14 at 0.6050**, then a monotone decline to **0.5813** at
epoch 60 — **-29.6 sigma**.
**Interpretation:** the model is severely OVERFITTING, and early stopping is the
only thing regularising it. Capacity >> data.
**Decision:** DIAGNOSTIC KEEP. Redirects the search from "add signal" to
"control capacity / reduce variance".

## E2 — Embedding capacity (k = 8, 32 vs incumbent 16), 3 paired seeds
**Hypothesis:** given the severe overfit, capacity is above what 1.14M rows
support; halving k should raise the peak.
**Result:** k=8 **-0.03 sigma** (t=-0.29); k=32 **-0.18 sigma** (t=-0.85).
**Interpretation:** HYPOTHESIS REFUTED. The model is insensitive to embedding
dimension — effective capacity is set by the stopping time, not by k. Note this
also finally measures `training=k32` on THIS config; the standing dead end was
the organisers' number on a pointwise model at the 0.589 level.
**Decision:** REJECT.

## E3 — History recency decay (tau_days = 1, 7, 14 vs hardcoded 3), 3 paired seeds
**Observation motivating it:** `tau_days=3.0` was HARDCODED in `History` and never
passed from cfg, invisible to the menu, over a 14-day train window — an untuned
modelling choice inside the incumbent.
**Result:** tau=1 **-0.03 sigma**; tau=7 **+0.01 sigma**; tau=14 **-0.04 sigma**.
**Interpretation:** the pooled history is insensitive to its decay constant. The
hardcoded value was not a hidden mistake.
**Decision:** REJECT. (Kept the cfg hook — the constant should be tunable.)

## E4 — Ensemble combination rule (no training required)
**Hypothesis:** rank-normalising before averaging was justified because "different
configurations produce scores on different scales", but all 16 members are the
SAME configuration, so the rank transform discards magnitude for no reason. Raw
averaging should be better.
**Result:** raw mean **-0.17 sigma**, z-scored mean **-0.20 sigma**, raw median
**-0.43 sigma**. **HYPOTHESIS REFUTED** — rank-normalising is not lossy here.
Rank-*median* came out **+0.46 sigma**, the best of five rules.
**Confirmation (pre-registered, 24 random 8-member subsets, paired):** median
beats mean in only **10/24**, mean delta **-0.06 sigma**, t=-1.09.
**Interpretation:** the +0.46 sigma was the max of five validation comparisons,
not a property of median aggregation. Exactly the trap the project's own rules
warn about.
**Decision:** REJECT. Incumbent rank-mean stands.
