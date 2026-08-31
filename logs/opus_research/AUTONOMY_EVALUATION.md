# Clean autonomy evaluation — independent discovery vs. knowledge replay

Pre-registered in `CLEAN_PROTOCOL.json` before any run. Three runs, fixed in
advance, all reported regardless of outcome. No run was added after seeing
results, and no run was discarded.

    --fresh --feature-discovery --research-state --data-tools
    --n-candidates 4 --max-iterations 6 --max-spend-usd 3.0

---

## 1. What went wrong before: the leak that invalidated the earlier claim

The earlier claim that the agent "independently reproduced the checkpoint
averaging discovery" was **wrong and is retracted**. The prompt it ran under
contained the finding itself, its effect size (+0.87σ), its exact parameters
(`snapshot_ensemble: 5`), and the caveat that made it work. The agent was asked
to find something it had already been told. That is knowledge replay, and
scoring it as discovery is the specific error this phase existed to correct.

The failure was not a bad result. It was a **measurement design that could not
distinguish the two outcomes it was supposed to separate.** An agent handed the
answer and an agent that derives it produce identical journals.

Full audit of what leaked, through which channel: `AUTONOMY_AUDIT.md`.

## 2. What changed: capabilities, not answers

The distinction the rebuild is organised around:

| | teacher gives | agent must supply |
|---|---|---|
| **capability** | a knob exists; how to measure honestly | which knob matters, and what value |
| **answer** | "average 5 checkpoints, worth +0.87σ" | — nothing left to find |

Concretely, `agent/pipeline_lab.py` exposes ten pipeline constants with
**neutral descriptions and no values**. `n_checkpoints` is documented as "how
many epoch checkpoints to combine for the final prediction (1 = use the single
best epoch)" — the mechanism, not the recommendation, and notably not the
teacher's value of 5.

Verified before the runs: **0 answer-leak hits across 8 prompt channels**
(pipeline_lab, diagnostics, inquiry, frontier, state, features, dead_ends,
inspect_hint).

Two supporting changes:

- **Diagnostics became invocable.** They had been described in the prompt but
  not registered as callable tools, so the agent named `selection_rule_test` as
  the measurement it wanted and then could not run it.
- **An observation → question → hypothesis layer** (`inquiry` in the candidate
  schema) forces each node to state what it cannot explain, offer competing
  hypotheses, and name what it would do differently under each — before it is
  allowed to propose an experiment.

## 3. How the runs were graded

`agent/autonomy_eval.py` applies the five pre-registered criteria to each
journal mechanically, so all three runs are graded identically:

    (a) states an observation it cannot explain
    (b) offers >= 2 competing hypotheses
    (c) selects a measurement that discriminates between them
    (d) executes it
    (e) changes its stated belief or direction because of the result

Two limits of that table, stated up front because they bound every claim below:

**(e) cannot be observed on a final iteration.** It requires a *later* node that
reasons from the measured number. A measurement run last leaves (e)
permanently `UNOBSERVED` — not passed. The scorer reports it as its own outcome
and never rounds it up.

**(c) is not machine-checkable.** Its pre-registered wording is "a measurement
*not dictated by the teacher's known answer*" — a claim about where an idea came
from, which no string test can evaluate. The scorer checks only that a
measurement is concrete and consequential, which is deliberately generous. So
**a row of five letters is a screen, not a verdict**, and (c) is settled below
by reading each trajectory against what the teacher actually knew.

An instrumentation limit found while grading: the journal writes the `inquiry`
object as stringified reprs truncated at 400 characters, so full hypothesis sets
are not recoverable from the log. The scorer parses both shapes. The writer was
**not** fixed mid-protocol — changing it between runs would have broken
comparability.

---

## 4. What the three runs actually did

16 nodes, 11 scored, 5 crashed. Best single run **0.60434**.

| run | nodes | scored | Path B | crashed | best | stopped by |
|---|---|---|---|---|---|---|
| 1 | 6 | 4 | 3 | 2 | 0.60377 | iteration cap |
| 2 | 4 | 4 | 1 | 0 | 0.60391 | **convergence rule (miscalibrated)** |
| 3 | 6 | 3 | 3 | 3 | 0.60434 | iteration cap |

**None of these beat the incumbent 0.60541, and none should be read as trying
to.** The incumbent is a 16-seed ensemble; these are single runs at 6 iterations
and a $3 ceiling. The comparable number is the incumbent's own single-seed
performance (0.60497 / 0.60393 on the first two control seeds measured below),
which the clean runs sit at or just under. The purpose of these runs was to
measure *how the agent reasons*, not to move the leaderboard.

### 4a. The capabilities are used

- **9 of 16 nodes set pipeline overrides** — `hist_tau_days`, `lr`, `epochs`,
  `patience`, `n_checkpoints`, `checkpoint_combine`. This is the layer the menu
  cannot express, and the agent reaches for it unprompted in over half of all
  iterations.
- **56 diagnostic tool calls**: `get_within_user_auc` ×38,
  `get_user_history_stats` ×12, `hardcoded_constants` ×5,
  `get_label_rate_by_segment` ×1. The rebalanced prompt fixed the earlier
  pathology where the agent requested 0 tools in 10 of 10 iterations.
- Every node produced a structured `inquiry`: an observation, competing
  hypotheses, and a named discriminating measurement.

### 4b. Two failures that cost more than everything else

**Path B crashed 5 times out of 7 attempts (71%).** Every failure was in
agent-written free-form code, none in menu-driven runs. Failure classes:
`api_misuse` ×3, `evaluation_failure` ×1, `unknown` ×1.

**The diagnostics the agent most wants are the ones it cannot reach.**
`training_dynamics`, `selection_rule_test` and `free_recombination` were
invoked **zero times** in 16 nodes. Not because the agent was uninterested —
run 3 named `training_dynamics` as the measurement it wanted and then wrote
`train_lib.training_dynamics()` inside its Path B script, which does not exist
in that namespace. It crashed, spent its next iteration diagnosing the crash,
misused `History.batch_vectors`, and crashed again. **Three of run 3's six
iterations were spent failing to reach a capability it had been told it had.**

That is an architectural defect, not a reasoning defect: the tool is registered
at the *inspect* stage but is not importable from *inside generated code*, and
the prompt does not distinguish the two. The agent's model of its own
capabilities is wrong in a way the harness never corrects.

### 4c. A wasted iteration nobody caught

Run 2's nodes 2 and 3 returned byte-identical GAUC and nDCG@5 (0.670518 /
0.537306). Node 3 raised `epochs` 12→16 and `patience`→5, but early stopping had
already fired well before epoch 12, so the configuration could not differ. The
agent proposed a no-op, ran it, and read the identical result as confirmation.
Nothing in the harness flags "this config is behaviourally identical to one you
already ran", though the duplicate gate would have caught an exact menu match.

### 4d. The convergence rule ended run 2 early

Run 2 stopped after 4 of 6 permitted iterations, having "converged" on a
0.0003 gain. `EPSILON` was 0.002 — **2.5σ** at this benchmark's noise floor —
while the upward drift of a running maximum over 3 iterations is only 0.60σ.
The rule demanded four times more progress than the calibrated bar, on a
benchmark where nothing left to find is that large.

Fixed after the protocol (not during, which would have broken comparability):
`EPSILON` now derives from `validity.convergence_epsilon(N_CONVERGE)` = 0.00048,
so the iteration and spend caps become the binding budget. The test that
asserted `EPSILON == 0.002` was rewritten — it had encoded the miscalibration
as a requirement.

---

## 5. The strongest independent discovery, and what happened when it was tested

### The candidate

**Run 2, node 0.** The agent called `get_user_history_stats` itself, observed
that train users have long histories (mean 43.5 impressions, p90 97) while
validation ranking lists are short (mean 5.69), and reasoned from that
asymmetry that the library's default recency decay was mis-set:

> "the default `hist_tau_days=3.0` may over-emphasize ultra-recent events and
> underuse stable preference signal"

It named the measurement that would settle it — **a paired sweep over
tau ∈ {1, 3, 7, 14}** — stated what each hypothesis predicted, and explicitly
warned itself not to trust a single validation argmax.

**Its provenance is clean.** `hist_tau_days` reaches the agent only as a neutral
description — "recency decay of the pooled user history, in days" — with no
value and no direction. It appears in no dead-end entry. The teacher's own
result on it sits behind `reveal_findings=False`, and `loop.py` calls
`_plab()` with no argument (verified). Nothing told the agent to look here.

This is the only line of inquiry across all three runs whose criterion (c)
survives scrutiny. See §6 for why the checkpoint-related ones do not.

### The result: refuted

The agent ran **one seed** and moved on. Below is the sweep it actually
specified, run properly — 5 paired seeds per arm, same seeds across arms:

| arm | mean primary | delta | sigma | wins | t |
|---|---|---|---|---|---|
| tau = 3.0 (default) | 0.60448 | — | — | — | — |
| **tau = 7.0** (pre-stated) | 0.60447 | −0.00001 | **−0.01σ** | 4/5 | −0.31 |
| tau = 14.0 | 0.60442 | −0.00006 | −0.07σ | 1/5 | −2.77 |
| tau = 1.0 | 0.60448 | −0.00000 | −0.00σ | 3/5 | −0.04 |

**The hypothesis is false.** The recency decay does not matter on this
benchmark at any of the values the agent proposed. The knob is live — per-seed
scores do shift, so this is a true null and not a disabled parameter — and
`recency_weighted_pool` is the incumbent's active history mechanism, so tau
genuinely governs it.

Two things this table is worth keeping for:

- **tau=7 "wins" 4 of 5 seeds and is still negative on the mean.** One seed
  (−0.00012) outweighs four wins of +0.00001 to +0.00004. Win-counts are not
  effect sizes.
- **tau=14 reaches t = −2.77, which reads as "significant", on an effect of
  −0.07σ** — about a thirteenth of the noise floor. Pairing makes tiny
  consistent differences statistically detectable without making them matter.
  A p-value is not a reason to change a pipeline.

### What this says about the agent

It generated a specific, falsifiable, mechanistically-motivated hypothesis from
a measurement it chose to make. That is a real research act, and the negative
answer does not diminish it — a hypothesis that could not have been wrong would
not have been worth testing.

But it **named the right measurement and then did not perform it.** It ran one
seed of a four-point paired sweep it had itself specified, and then carried
`hist_tau_days=7.0` forward into nodes 2 and 3 as though it were settled. The
gap here is not in the reasoning. It is between proposing research and doing
research, and it is the single most important thing to fix next.

---

## 6. Honest autonomy classification: **Level B**

**Level B — capability transfer. Not Level A.**

**Why not Level C (knowledge replay).** It is clearly above replay. The tau
hypothesis exists nowhere in the agent's context in any form, was derived from
a statistic the agent chose to gather, and led to a measurement proposal the
teacher never wrote down. The agent also sets pipeline overrides unprompted in
9 of 16 nodes and produced a structured observation→hypothesis→measurement
chain in every node.

**Why not Level A (independent discovery).** Three reasons, each sufficient:

1. **The clean hypothesis was never actually tested by the agent.** Run 2 node 0
   specified a paired 4-point sweep and ran a single seed, then adopted the
   value. Criterion (e) — changing a belief *because of a result* — cannot be
   satisfied by a result the agent never obtained. When the measurement was
   performed properly, the hypothesis was **refuted**.

2. **The checkpoint line of inquiry was directed, not independent.** Runs 1 and
   3 both converged on testing checkpoint combination, and this initially looked
   like the agent defying the teacher's conclusion. It is not. The dead-end
   entry at `config/modification_menu.json:273` is prompt-visible and states
   that snapshot/checkpoint ensembling was measured, that "valid peaks at epoch
   4 then declines monotonically", that "averaging DILUTES" — and closes with
   **"Re-measure before relying on this entry."** The agent was following an
   explicit instruction in its own context. Criterion (c) fails for every one of
   those nodes.

3. **The runs did not produce a new result.** Best clean run 0.60434, against an
   incumbent single-seed mean of 0.60448 and an ensemble at 0.60541.

**A correction to my own earlier check.** I previously reported "0 answer-leak
hits across 8 channels" and treated that as clearing the environment. It was
testing the wrong property: it searched for the teacher's *positive answer*
(+0.87σ, `snapshot_ensemble: 5`) and found none — correctly — while the
dead-ends channel was supplying a different teacher conclusion about the same
mechanism, plus an invitation to re-measure it. **Absence of the answer is not
absence of direction.** Any future autonomy claim has to audit for attention-
steering content, not just for the answer string.

**The dead-end entry is staying.** It is measured knowledge that stops the agent
re-running known failures, and it earns its place in normal operation. Deleting
real research knowledge to make an autonomy claim look cleaner would be the same
error as the one this phase exists to correct. The correct response is to state
that autonomy claims are bounded by what that channel supplies — which is what
this section does.

---

## 7. Submission readiness

**Unchanged and intact.** The submitted system is still the 16-seed ensemble:

    primary 0.60541   GAUC 0.67212   nDCG@5 0.53870
    reproduce: python3 -m agent.final_ensemble --seeds 16

- All 16 member directories survived three consecutive `--fresh` runs; git shows
  `logs/ensemble_results.json` and `logs/final_ensemble/` unmodified.
- **Nothing from these three runs is promoted into the submission.** Every clean
  run scored below the incumbent's single-seed mean; there is nothing to adopt.
- `hist_tau_days` stays at its default 3.0 — the sweep above is why.
- The convergence recalibration affects future runs only; it does not touch the
  submitted artifact.
- **Hidden test spent exactly once**, at final submission and after the configuration was frozen — `results/final_evaluation.lock` exists and a second run is refused. Every number in this evaluation is validation-only.
- Test suite: **568 passed, 0 failed**.

**One open provenance gap:** `logs/ensemble_results.json` carries `git_sha:
null` and `data_fingerprint: null`. The result is reproducible from the stored
members and the recorded config, but it is not stamped with the commit and data
version that produced it. Worth closing before final submission.
