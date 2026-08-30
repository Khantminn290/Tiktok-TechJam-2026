# Distillation: from the Opus research run to agent capabilities

Each capability below is here because it *changed a conclusion* during the
research run. Nothing was added because it sounded like a good tool.

---

## CAPABILITY: `training_dynamics()`

**Research observation that motivated it.** Every menu-level intervention on
record was null, yet the learning curve said the model was data-limited. Those
two facts are only consistent if something else is binding. Nobody had looked at
the epoch curve, because the harness always ran with early stopping on.

**The Opus decision that demonstrated it.** Run the incumbent with
`patience=60` so the whole curve is visible. It peaks at **epoch 14 (0.6050)**
and decays monotonically to **0.5813** by epoch 60 — **−29.6 sigma**. The model
is severely overfitting and *early stopping is the only regulariser*.

**Why it generalises.** A score-level experiment can only tell you whether a
configuration is better. It cannot tell you *which part of the pipeline is
binding*. Training dynamics is the cheapest diagnostic that distinguishes
"needs more signal" from "needs better regularisation" from "needs a better
stopping rule" — and those three lead to completely different research programs.

**How the agent invokes it.** `pipeline_lab.training_dynamics(seeds, max_epochs)`
returns the curve, peak epoch, and a verdict string.

**Evidence it works.** It reproduced the −29.6 sigma finding, and that finding
redirected the phase away from capacity and toward checkpoint selection — which
is where the only real improvement was found.

---

## CAPABILITY: `hardcoded_constants()`

**Research observation.** The incumbent uses `user_history=recency_weighted_pool`,
but the decay constant `tau_days=3.0` was hardcoded in `History.__init__`, never
passed from config, and invisible to the menu — an untuned modelling choice over
a 14-day training window, sitting inside the best-known configuration.

**The Opus decision.** Grep the training library for modelling constants that no
menu axis can reach, then sweep the one that looked substantive.

**Why it generalises.** A search space defined by a menu silently excludes every
constant someone typed into the library. Those constants are exactly where
untested assumptions accumulate, because nobody has to justify a default.

**How the agent invokes it.** `pipeline_lab.hardcoded_constants()` lists each
constant with its line, default, and whether the agent can currently reach it.

**Evidence it works.** It independently rediscovers `tau_days=3.0` — the constant
found by hand — and flags `SEQ_LEN`, `EXTRA_FEATURE_BINS`, `MAX_EXTRA_FEATURES`
as still unreachable. (The sweep itself came out null; the capability is
validated by finding the right target, not by that target paying off.)

---

## CAPABILITY: `selection_rule_test()` — the important one

**Research observation.** The stopping epoch is chosen by argmax over ~40
validation evaluations. That is *selection on validation*. So "which epoch scores
highest" is the wrong question; the right one is "does the stopping RULE
generalise to data it did not choose on".

**The Opus decision.** Choose the epoch on one half of the validation users,
score on the other half, both directions, across 4 independent splits and 3
seeds — 24 held-out evaluations. Averaging the top-5 checkpoints beats argmax by
**+0.00069 (+0.87 sigma), t=5.54, winning 22/24**.

**Why it generalises — and what it overturned.** Snapshot ensembling was already
in the codebase and already *rejected*. The rejection came from a guard inside
training that adopted the snapshot only if it beat the best single checkpoint
**on the same validation set that selected that checkpoint**. That comparison is
biased by construction, and the rejection was an artefact of how it was measured
rather than a property of the method. This capability is what distinguishes the
two.

**How the agent invokes it.** `pipeline_lab.selection_rule_test(per_epoch_scores,
users, labels, rules)` returns, per rule, the held-out delta, sigma, t, win
count, and a boolean `generalises`.

**Evidence it works.** It converted a standing REJECT into a measured, confirmed
improvement, and it is the only positive result of the phase.

---

## CAPABILITY: `free_recombination()`

**Research observation.** Comparing five ensemble aggregation rules on the full
validation set produced a `rank-median` result of **+0.46 sigma** — the best of
five. That is a selection over five comparisons, not a discovery.

**The Opus decision.** Resample: 24 random 8-member subsets, compare median
against mean paired on each. Median won **10/24** at **−0.06 sigma**, t=−1.09.
The +0.46 sigma was noise.

**Why it generalises.** Any question about how to *combine* stored predictions
can be answered without training, and therefore can be answered many times.
Repetition is what separates a real effect from the maximum of a handful of
noisy comparisons — and it costs seconds.

**How the agent invokes it.** `pipeline_lab.free_recombination(member_scores,
users, labels, rules, n_subsets, subset)`.

**Evidence it works.** It stopped a +0.46 sigma false positive from being
promoted, in seconds, with no training.

---

## CAPABILITY: `research_run.run_variant()` / `SAFE_OVERRIDES`

**Research observation.** Embedding size, the history decay constant, and the
stopping rule are not menu axes and never could be — the menu is a set of named
options, and these are numbers inside the training library.

**The Opus decision.** Specify experiments as direct config overrides against
the incumbent, paired by seed, everything else held.

**Why it generalises.** It moves the agent from "choose an option" to "change a
number in the pipeline", which is the difference between a menu executor and a
researcher. The override surface is deliberately **10 entries**, each earned by
being investigated: a 100-option list would be the same menu under a new name.

**Evidence it works.** It produced E2 (k=8/32, null), E3 (tau, null), and E5/E6
(checkpoint averaging, positive) — none of which the menu could express.

---

## The transferable lesson

Three separate results in this run turned on the same point:

> **A score computed on the same data that selected it is not evidence.**

- the epoch argmax is fitted to validation → test it on held-out halves (E5)
- the best of five aggregation rules is fitted to validation → resample (E4)
- the snapshot guard compares against a checkpoint chosen on the same set → the
  rejection was an artefact (E5)

That sentence is carried into the agent's planning prompt, because it is the
part most likely to generalise beyond this dataset.

---

## CAPABILITY: `validity.audit_comparison()` — added after the leakage audit

**Research observation that motivated it.** The teacher run got the same class of
question wrong three times, and each time the error was invisible in the score:
a "+0.46 sigma" aggregation win that was the best of five rules on one split; a
stopping epoch chosen by argmax over ~40 validation evaluations; and a *working*
method left rejected because its guard compared it against a checkpoint selected
on the same data. None of these are detectable by looking at a number.

**The Opus decision that demonstrated it.** Ask, before believing anything: how
many seeds, were the arms paired, how many variants were compared before this one
won, and was the winner chosen on the data now scoring it?

**Why it generalises.** It grades the *design* rather than the effect, so it
transfers to any task. The load-bearing part is quantifying selection pressure:
picking the best of five noisy comparisons yields about **+0.96 sigma** on this
benchmark with no real effect — larger than the +0.46 sigma that was briefly
believed. That one number kills the false positive on sight.

**How the agent invokes it.** `validity.audit_comparison(delta, n_seeds, paired,
n_candidates_compared, selected_on_eval_data, confirmed_out_of_sample)` →
severity, verdict, and per-finding messages. Advisory, never blocking: it cannot
know intent, so it reports what a number is worth rather than refusing an
experiment.

**Evidence it works.** Replayed against the three real cases: E4 → FATAL, E5 →
WARN only, single-seed-at-any-size → FATAL.

---

## Autonomy classification — corrected

| Demonstration | Level | Why |
|---|---|---|
| First self-test (checkpoint averaging) | **C — knowledge replay** | the prompt stated the finding, the effect size, the exact parameters and the caveat; every element of the "discovery" traces back to that text |
| Feature discovery (`user_author_recent_long_view_rate`) | **B — capability transfer** | the agent was given a probe and a builder contract, not the feature. It invented the feature, wrote 129 lines implementing it, and the probe rejected it. The *idea* was its own; the *method* was transferred |
| Feature declines citing the registry | **B — capability transfer** | it read accumulated negative knowledge and declined rather than re-running |

The first row is the one I originally overclaimed. It is corrected in
`AUTONOMY_AUDIT.md`, in the commit history, and here.

**What honest Level-B transfer looks like:** the agent receives the *capability*
and the *methodological principle*, never the answer. It must still decide the
principle applies, choose the tool, and interpret the result. A stated result is
replay; a method the agent must decide when to apply is transfer.
