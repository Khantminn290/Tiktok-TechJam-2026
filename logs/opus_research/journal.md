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

## E5 — Does the STOPPING RULE generalise? (held-out halves)
**Observation:** the stopping epoch is chosen by argmax over ~40 validation
evaluations — selection on validation. So "which epoch scores highest" is the
wrong question.
**Change:** choose the epoch on one half of the validation USERS, score on the
other, both directions, 4 splits x 3 seeds = 24 held-out evaluations.
**Result:** averaging the top-5 checkpoints beats argmax by **+0.00069
(+0.87 sigma), sd 0.00061, t=5.54, wins 22/24**.
**Interpretation:** OVERTURNS a standing rejection. Snapshot ensembling was
already in the codebase and already refused — by a guard that adopted it only if
it beat the best single checkpoint ON THE SAME validation set that selected that
checkpoint. The rejection was an artefact of a biased comparison.
**Decision:** KEEP the finding; test whether it survives ensembling.

## E6 — 16-member ensemble of snapshot-averaged members (pre-registered)
**Result:** incumbent 0.60541 -> snapshot 0.60540, **-0.00001 (-0.01 sigma)**.
**Decision:** INCONCLUSIVE by the rule fixed in advance. NOT adopted.

## E7 — Why did a +0.87 sigma member gain vanish in the ensemble?
**Change:** same held-out protocol as E5, but comparing 3-member ENSEMBLES.
**Result:** the advantage decays monotonically with ensemble size —
  1 member  +0.87 sigma (t=5.54, 22/24)
  3 members +0.28 sigma (t=1.90, 10/12)
  16 members -0.01 sigma
**Interpretation:** checkpoint averaging and seed ensembling remove the SAME
variance — the noise in where the epoch argmax lands. They are SUBSTITUTES, not
complements. At k=16 the seed ensemble has already removed all of it.
**Decision:** REJECT for the submitted ensemble, and CLOSE the whole
variance-reduction family: any further technique of that kind is redundant with
16-seed averaging for the same reason.
**Still true and still useful:** for a SINGLE model, or a small ensemble,
checkpoint averaging is a real +0.87 sigma improvement. It is only redundant at
our ensemble size.

## SELF-TEST — does the upgraded agent reproduce the research behaviour?
Given only the baseline, the tools, the experiment history and a budget. It was
NOT told which experiments worked.

**Node 2, Path B, written by the agent:**
> "Validation-safe checkpoint averaging over the strongest single-run FM+BPR
> trajectory can confirm whether residual epoch-selection noise is still leaving
> a small amount of single-run performance on the table... only relevant for a
> single model (not multi-seed ensembles)"

It set `snapshot_ensemble: 5, snapshot_force: True` inside its own script,
predicted **+0.0005 to +0.0008**, and measured **0.60524** against the incumbent
single-model mean of 0.60446 — **+0.00078 (+0.97 sigma)**, inside its own
predicted range and consistent with the +0.87 sigma measured in E5. It also
scoped the finding correctly: "not multi-seed ensembles" is exactly E7's
conclusion, read from the evidence rather than paid for with a run.

**A correction to my own reporting.** I first announced the agent used the
overrides "0 times", from a journal scan of `menu_choices`. That scan cannot see
Path B usage, where `menu_choices` is `{}` and the overrides live inside the
generated script. The claim was wrong.

**Two bugs this exposed**
1. the mechanism audit called that node "MECHANISM NOT EVIDENCED" because the
   override was a dict literal in a Path B script rather than a source pattern —
   a working experiment must not be warned about. Fixed.
2. `--fresh` archived `logs/opus_research/` mid-phase, because the research
   journal was not in the preserved-artifact list. Fixed; the same class of bug
   as the orphaned ensemble members earlier in the project.
