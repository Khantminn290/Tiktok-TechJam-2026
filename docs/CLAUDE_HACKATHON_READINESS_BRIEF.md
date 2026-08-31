# Claude Code Brief: Make the Pure Agent Hackathon-Ready

> ## Codex implementation status - 2026-08-31
>
> The recommendations below are retained as the decision record. Current code
> now implements: organizer-rule competition execution; an executed baseline
> node; candidate -> paired confirmation -> fixed ensemble scheduling;
> canonical agent-ensemble publication and eligibility stamping; fail-closed
> finalization; provider-token and artifact hashes; pinned dependencies; a
> three-level robustness report with 3/3 full-loop recoveries; a generated
> three-minute judge route and Devpost narrative; and a validation-only epoch
> sensitivity diagnostic.
>
> The epoch diagnostic found peak epochs 10-14 and a mean local-window spread
> of 1.31 sigma, but correctly returns `do_not_promote_rolling_origin`: these
> captures are official-validation-only, so selecting a new rule from them now
> would be validation tuning. Path B remains an honest null outcome (lifecycle
> machinery tested, no Pure feature cleared the real paired gate).
>
> **Still intentionally open:** run the now-implemented competition profile from
> a frozen clean commit and regenerate its manifest. The existing legacy journal
> still makes node 1 at 0.60497 eligible and node 4 at 0.60541 ineligible. The
> hidden test remains untouched and must not run until finalization passes.

> ## Review note — 2026-08-31, after commit `4b4282c`
>
> I fact-checked this brief against the repository. **The central finding is
> correct and I had it wrong in my own documentation.** Summary of the review,
> with the detail attached to each section below as `> REVIEW:` blocks.
>
> **Confirmed by re-deriving from artifacts:**
>
> | Claim | Status |
> |---|---|
> | Official rule fires at node 3; ensemble is node 4 | **Confirmed.** Eligible checkpoint is 0.60497 (node 1), not 0.60541 |
> | A stricter internal epsilon cannot extend the official run | **Confirmed, and it contradicted our own docs** |
> | Manifest underreports tokens (137,446 vs 269,807) | **Confirmed.** Provider total is 269,807 over 24 calls |
> | `best_single_seed` reports the ensemble node | **Confirmed.** Was 0.60541 labelled "one draw" |
> | `+0.00410` at the last training-data doubling | **Confirmed.** 0.60036 → 0.60446, 3 seeds |
> | Path B has no completed paired retrain | **Confirmed.** 8 registry entries: 6 REJECTED, 2 PROBED |
> | No pinned environment file | **Confirmed.** No `requirements.txt` or `pyproject.toml` |
> | Live fault run did not complete a repaired result | **Confirmed.** Correctly labelled incomplete |
>
> **Already done since this brief was written** — do not redo (see per-section
> notes): `manifest.render()` `cache_hits` fix, legacy-journal observation
> reporting, provider-token fix, `best_single_seed` fix, official-eligibility
> reporting, the fault suite, and a manifest-generated judge packet.
>
> **Where I disagree:** the `+0.00410` evidence does not support the
> rolling-origin proposal it is attached to (P1), and P0's "ensemble
> immediately" would skip confirmation — a concrete sequence that satisfies
> both is given in the P0 notes.

## Executive verdict

The agent is already a credible autonomous research system, not a dashboard
mock-up. It has a reproducible KuaiRand-Pure validation result, an executable
research loop, strict hidden-test boundaries, paired evidence, autonomous
ensembling, provenance, cost accounting, and detailed research memory.

It is not yet safe to submit as a hackathon-winning run. The largest risk is
not model quality; it is **official checkpoint eligibility**:

- The organizer rule is epsilon `0.002`, N `3`, with the first of convergence,
  50 iterations, or 6 hours ending the run.
- On the current journal, the official rule first converges at node `3`.
- The submitted `0.60541` ensemble is node `4`.
- Therefore this journal cannot prove that `0.60541` was the validation-best
  checkpoint **at convergence**. An internal stricter epsilon cannot extend the
  official run, because the problem statement says the first stopping condition
  wins.

Fix this before further score work. A clean competition run should put an
official baseline observation first and schedule the fixed 16-seed ensemble
before the earliest no-progress window can close. Do not rewrite the old
journal or reinterpret the rule after the fact.

> **REVIEW: correct, and this was the most valuable thing in the brief.**
> Re-derived from `logs/journal.jsonl`: running best 0.60330 → 0.60497 →
> 0.60497 → 0.60497, so the window ending at node 3 gains +0.00167 ≤ 0.002 and
> the rule fires. Eligible checkpoint = **0.60497 (node 1)**. The 0.60541
> ensemble (node 4) and 0.60509 (node 6) are both after the stop.
>
> This contradicted three places in our own repository, all of which argued
> that a stricter internal epsilon was "the safe direction" because it "can
> only make the loop run longer... so no scored checkpoint is ever missed".
> That is true and irrelevant: the rule fixes *what is scored*, so running
> longer produces an **ineligible** artifact rather than protecting it. Fixed
> in `agent/convergence_report.py`, `agent/loop.py`, `README.md` and the
> generated judge packet.
>
> **Now machine-checkable, so Codex does not have to take this on trust:**
> `convergence_report.report()` returns an `eligible_checkpoint` block naming
> the eligible node, its score, and every higher-scoring node that came too
> late. It is surfaced in `manifest.render()`, in `results/manifest.json`, and
> in section 10 of `results/JUDGE_PACKET.md`. Pinned by
> `test_official_checkpoint_eligibility`.
>
> One qualification worth holding onto: with ε = 0.002 against a benchmark
> whose *total* headroom over baseline is about 0.004, the rule fires at the
> earliest legal moment in almost any run. That is a property of the rule, not
> of this agent. It makes the 50-iteration cap and 6-hour ceiling nearly
> unreachable, which is worth raising with the organizers — but plan for the
> literal reading, because that is the one that will be applied.

## Current evidence snapshot

| Item | Current evidence | Submission meaning |
|---|---:|---|
| Official validation baseline | `0.60160` | Organizer-provided comparator |
| Incumbent validation primary | `0.60541` | `+0.00381`, but hidden-test delta is unknown |
| Incumbent components | GAUC `0.67212`, nDCG@5 `0.53870` | Correct official metrics |
| Incumbent construction | 16 fixed seeds, rank-normalized mean | Recomputable from stored predictions |
| Strongest single seed | `0.60497` | Preliminary; do not call it the ensemble |
| Latest search trajectory | 8 nodes, 28 training executions | Continued beyond official node-3 convergence |
| LLM use | 269,807 input + output tokens | Provider total in `logs/final_summary.json` |
| Agent wall-clock | 2,545.4 seconds | About 42.4 minutes |
| Training wall-clock | 2,321.5 seconds | CPU, zero GPU-hours reported |
| Manual interventions | 0 | Strong autonomy evidence |
| Hidden test | Not evaluated | Preserve until the frozen final artifact |
| Path B | 8 probes, no real paired retrain | Plumbing exists; lifecycle is incomplete |
| Robustness | Component faults plus one incomplete live recovery | Do not claim full closed-loop recovery yet |

These are validation facts, not hidden-test claims. The public validation gain
is promising but does not guarantee that the hidden-test primary beats
`0.5946`.

> **REVIEW: every row verified. Three need updating.**
>
> - *Incumbent validation primary* — accurate, but should now carry the
>   eligibility caveat: `0.60541` is the best artifact, and `0.60497` is what
>   the official rule would score on this journal.
> - *LLM use* — `269,807` is correct and is now what the manifest reports; it
>   was 137,446 when this table was written. Add that it is **24 provider calls
>   for 8 decisions**, which is the figure the P2 token work should target.
> - *Robustness* — now 20 component faults at 100% detection and correct
>   routing, plus 2 live subprocess faults, plus the one incomplete loop
>   recovery. The conclusion is unchanged and still right.
>
> *Strongest single seed `0.60497`* is correct as written (seed 0 of the
> submitted configuration). Note it is **not** the same as the manifest's
> `best_single_seed`, which means the best single-seed node in the journal
> (`0.60509`, node 6). Keep the two apart in any judge-facing text.

## Authoritative interpretation

Claude Code must treat the pasted challenge statement and starter-kit code as
authoritative. Preserve these invariants:

1. Work on **KuaiRand-Pure only**. Do not spend this cycle on 1k or 27k.
2. The label is `long_view`; metrics are GAUC and nDCG@5; primary is their mean.
   The earlier contradictory NDCG@10/Recall@50 line is superseded by the detailed
   starter-kit definition and evaluator.
3. Never alter `kuairand-starter-kit/evaluate.py`, train on hidden-test labels,
   inspect hidden-test metrics during development, or run the final test twice.
4. Do not use transductive features computed from an entire validation/test
   impression batch and fed back into rows from that same batch.
5. A single seed may screen a hypothesis but cannot promote the submission.
6. Validation-selected configurations require predeclared paired confirmation.
7. Artifact reuse is historical evidence, not fresh compute and not a new
   independent observation.
8. Keep the honest autonomy label: the current agent demonstrates strong
   Level-B capability transfer, not independent Level-A discovery.
9. The agent must still write code for genuinely new pipeline stages. Typed
   configuration is appropriate for known Path-A operations, but eliminating
   all code generation would weaken alignment with the task statement.

## Rubric assessment

### Technical Execution - 35%

**Strong:** exact incumbent verification, official evaluator reuse, protected
test labels, paired evidence gates, deterministic ensemble aggregation, source
hashing, preflight checks, and a final-evaluation lock.

**Material gaps:** the current scored journal makes the ensemble ineligible
under the official convergence point; hidden performance is unknown; the live
fault run chose a debug action but did not complete a repaired training result;
and Path B has not carried a real feature from source to paired outcome.

**Winning evidence:** one clean official run with an eligible ensemble, exact
reproduction, a valid submission CSV, and at least three genuine closed-loop
recoveries in an isolated non-scored run.

### Innovation and Problem Insight - 20%

**Strong:** the agent reasoned across objective, negative sampling, temporal
features, history, multitask learning, architecture, regularization, variance,
residual signal, and metric structure. It also rejected attractive but invalid
ideas such as transductive repeat fatigue and random-log IPS training.

**Material gaps:** the menu is essentially exhausted, many findings came from
transferred research memory, and the current literature grounding is embedded
in prose rather than attached to each autonomous decision. The randomized log
is useful as a diagnostic, but evaluation is on the logged policy, so forcing
IPS into training would optimize the wrong distribution.

> **REVIEW: "the menu is essentially exhausted" is asserted, not shown.**
> `agent/frontier.py` tracks per-axis, per-option coverage and can answer this
> directly — quote its output rather than the claim, especially since a
> frontier bug previously mislabelled the highest-scoring temporal option as a
> dead end and cost about 0.007 of apparent headroom. An exhaustion claim from
> the same subsystem should be checked before it is used to justify skipping
> Path-A work.
>
> The IPS reasoning is correct and worth keeping: the evaluator scores the
> logged distribution, so reweighting training toward the randomized-exposure
> distribution optimizes something the metric does not reward.

**Winning evidence:** research cards generated at decision time, one new
leakage-safe mechanism outside the menu, and an honest negative result if it
fails. Novelty should come from the reasoning and experiment design, not from
claim inflation.

### Impact and Relevance - 20%

**Strong:** the latest run reports zero manual interventions and the loop can
plan, generate/validate code, train, evaluate, reflect, confirm, and ensemble.

**Material gaps:** the original headline ensemble was first produced by a
human-invoked command, although the agent later reproduced it. Recovery and
Path-B completion still need human-independent end-to-end evidence.

**Winning evidence:** a fresh run in which the agent itself reaches the final
eligible artifact, with intervention events mechanically counted rather than
stated in prose.

### Feasibility and Practicality - 15%

**Strong:** the result is CPU-only and finishes well inside six hours.

**Material gaps:** roughly 270k tokens for eight decisions is high; ordinary
menu actions generate too much repeated script text; and the generated manifest
currently underreports tokens by summing node-owned calls instead of the
provider total. Feasibility is scored only after the hidden baseline gate, so
never trade away model quality to make the run look cheap.

**Winning evidence:** preserve the same decision quality and confirmation rigor
while reducing calls and repeated context, with provider totals matching the
journal and report exactly.

### Presentation and Communication - 10%

**Strong:** the dashboard, experiment tree, journal, architecture notes, and
generated results provide unusually rich evidence.

**Material gaps:** README tab descriptions have drifted, there is no compact
judge route, the repository setup is not fully packaged, and current generated
artifacts contain accounting/labeling inconsistencies.

**Winning evidence:** a public-repo-ready three-minute judge guide, one canonical
manifest, a valid CSV beside it, and a Devpost narrative generated from those
same facts.

## Implementation order

Complete the phases in this order. Do not start hidden-test evaluation in any
phase.

## P0 - Add an official competition profile

### Goal

Guarantee that the artifact named final was available at the first official
stopping point.

### Implement

1. Add a `--competition` profile with organizer epsilon `0.002`, N `3`, 50
   iterations, and 6-hour ceiling. This profile must stop on the organizer rule;
   internal research gates may not defer it.
2. Journal an actual organizer FM baseline evaluation as experiment `0`. Do not
   hardcode a synthetic score-only node. Record code/config/data hashes and the
   official evaluator output.
3. Make the first above-baseline candidate eligible for autonomous ensembling
   immediately. Queue the predeclared 16-seed ensemble before lower-value
   exploration, while the official convergence window is still open.
4. Compute official convergence after every scored iteration. Persist
   `official_eligible: true/false` on each candidate and ensemble artifact.
5. When convergence fires, freeze the validation-best eligible artifact and
   reject any later score as post-convergence research evidence.
6. Keep the calibrated `0.00048` controller only in a separate `--research`
   profile. Label it non-official everywhere.
7. Never mutate the existing journal to make it pass. Produce a clean new run
   directory and preserve the old run as evidence of the issue found.

> **REVIEW: items 4-7 are done; 1-3 are the real work, and item 3 needs
> amending.**
>
> - **Item 4 (compute official convergence, persist eligibility): done.**
>   `eligible_checkpoint` is computed on every report and carried in the
>   manifest. What remains is enforcing it *during* a run rather than
>   reporting it after.
> - **Item 6 (keep 0.00048 labelled non-official): done and now correctly
>   reasoned.** The `--competition` profile already exists
>   (`agent/profiles.py`) but does **not** set the organizer epsilon — the loop
>   always runs `EPSILON = 0.00048`, and `min_branching_iterations: 3` plus the
>   pending-ensemble gate actively *block* convergence. That is the gap to
>   close for item 1.
> - **Item 7 (never mutate the journal): agreed, and nothing has been.**
>
> **Item 3 needs amending — as written it would trade one rigour problem for
> another.** "Make the first above-baseline candidate eligible for autonomous
> ensembling immediately" means ensembling a configuration backed by a single
> seed, which is exactly what the evidence layer exists to prevent.
>
> The two goals are compatible, because the ensemble node is itself a scored
> iteration and its jump keeps the window open. This sequence is verified in
> `test_official_checkpoint_eligibility`:
>
> | node | action | primary | running best | window |
> |---|---|---|---|---|
> | 0 | official baseline | 0.6016 | 0.6016 | — |
> | 1 | candidate | 0.60497 | 0.60497 | — |
> | 2 | paired confirm | 0.60327 | 0.60497 | — |
> | 3 | **ensemble** | **0.60541** | 0.60541 | +0.00381 > ε → open |
> | 4 | improve | 0.60480 | 0.60541 | +0.00044 ≤ ε → **stop** |
>
> Convergence fires at node 4 and the eligible checkpoint is the ensemble.
> Confirmation is preserved. The requirement is "baseline first, ensemble by
> node 3", not "ensemble without confirming".

### Required tests

- Reproduce the current score sequence and assert official convergence at node
  `3`, with node `4` marked ineligible.
- Use `baseline -> candidate -> ensemble -> no-progress` and assert the ensemble
  is eligible when official convergence eventually fires.
- Assert `--competition` cannot be delayed by branching, pending ensemble,
  debug, or the research epsilon.
- Assert errored/preflight-only attempts do not fake scored progress, while all
  relevant wall-clock still counts.

## P0 - Repair the canonical manifest and budget truth

### Goal

Make every displayed result derivable from one machine-readable artifact.

### Implement

1. Source total LLM input + output tokens from the provider-level run summary.
   Node token ownership is useful diagnostics but is not the rubric total.
2. Exclude `action == "ensemble"` when calculating `best_single_seed`; report
   the ensemble in its own field.
3. For legacy journals without execution events, report observation count as
   `unknown` or explicitly derived from scored nodes. Never show 28 runs beside
   zero observations without explaining missing instrumentation.
4. Fix `manifest.render()` to use `reused_artifacts`; remove the stale
   `cache_hits` key.
5. Add official convergence node, final artifact node, eligibility, manual
   interventions, test-lock state, CSV hash, dataset hash, evaluator hash, git
   commit, and dirty-worktree state.
6. Make README, dashboard, `RESULTS.md`, judge guide, and Devpost summary consume
   the manifest rather than retyping numbers.
7. Fail finalization if manifest totals disagree with provider summary, if the
   repository is dirty, or if the selected node is after official convergence.

> **REVIEW: items 1-4 are done. Items 5-7 remain.**
>
> All four diagnoses were correct and are fixed in
> `agent/manifest.py`, pinned by `test_manifest_accounting_matches_the_provider`:
>
> - **Item 1 (provider tokens): fixed.** Now 269,807 from
>   `final_summary.total_llm_tokens`, with `llm_calls: 24` and the node-owned
>   137,446 retained as `llm_tokens_node_owned` for diagnostics. The 2x gap is
>   planning, candidate scoring and repair retries — real billed calls that no
>   single node owns.
> - **Item 2 (`best_single_seed`): fixed.** It was reporting 0.60541 — the
>   16-member ensemble — labelled "one draw; cannot change the submission",
>   which both misattributed the number and understated the evidence. Now
>   excludes `action == "ensemble"`.
>   *One correction to the brief:* the right value is **0.60509** (node 6), not
>   0.60497. Those are different quantities — 0.60497 is seed 0 of the
>   *submitted configuration*, while this field means the best single-seed
>   *node in this journal*. The test pins both and says which is which.
> - **Item 3 (legacy journals): fixed.** `unique_observations_source` states
>   either "derived from scored nodes: this journal predates execution-event
>   instrumentation" or the live source, so 28 runs never sit beside 0
>   observations unexplained.
> - **Item 4 (`cache_hits`): fixed**, and it was a live `KeyError` in
>   `manifest.render()`, not just a stale key.
>
> **Still open: items 5, 6, 7.** Item 5 partly — git commit, dirty state,
> dataset and evaluator hashes are present; CSV hash, test-lock state and
> final-artifact node are not. Item 6 partly — README, dashboard and the judge
> packet consume the manifest; `RESULTS.md` and a Devpost summary do not.
> Item 7 (fail finalization on disagreement) is entirely open and is the one
> that makes the rest enforceable rather than aspirational.

### Required tests

- Pin the current true total at 269,807 tokens and catch the incorrect 137,446
  node-only total. *(done)*
- Pin single `0.60497` versus ensemble `0.60541` labeling. *(done, but the
  single-seed node figure is `0.60509`; see the review note above)*
- Render a legacy and a fully instrumented journal without `KeyError`. *(done)*
- Regenerate twice and assert stable hashes except for timestamp fields.
  *(still open)*

## P1 - Improve hidden-test generalization, not public-valid overfit

### Goal

Use the strongest measured clue: the model gains `+0.00410` from the final
50%-to-100% training-data doubling, while epoch selection is highly sensitive.
With bonus datasets excluded, better use of Pure's legal training period is the
highest-value score direction.

> **REVIEW: the number is right, the inference attached to it is not.**
>
> `+0.00410` is confirmed (`logs/archive_20260830_025425/learning_curve.json`:
> 0.60036 at 50% → 0.60446 at 100%, 3 seeds, +5.12σ). But it measures a **data
> volume** effect. Rolling-origin inner validation adds no data — it changes
> *when you stop training*. The learning curve is evidence that the model is
> data-limited, which argues for more rows; it is not evidence that epoch
> selection is leaving anything on the table.
>
> The actual premise for P1 is the second clause, "epoch selection is highly
> sensitive", and that one is asserted without a citation. **Codex: measure it
> first.** The cheap version is to take the existing per-epoch capture on the
> incumbent config and ask how much primary varies across the plausible stopping
> window. If that spread is under about 1σ, rolling-origin cannot pay for itself
> and this phase should be dropped rather than run.
>
> Note also that `docs/RESEARCH_LOG.md` cites this same +0.00410 as the argument
> for **1K/27K** — which the user has excluded from this cycle. That exclusion is
> a scope decision, not a technical finding, and it is worth being explicit that
> excluding it forgoes the largest measured lever in the project.

### Implement

1. Add rolling-origin inner validation **inside the official train dates**.
   Example: train through day d, select/freeze training duration on later
   train-only days, and repeat across several cut points.
2. Use those folds to choose one fixed epoch/schedule rule before evaluating on
   official validation. Official validation remains the outer selection set;
   do not tune the rule after seeing its score.
3. Compare the fixed rolling-origin rule against current valid-argmax early
   stopping over the same predeclared seeds. Promote only on paired evidence and
   retain the 16-seed aggregation rule fixed in advance.
4. Add one bounded out-of-menu experiment only if rolling-origin error analysis
   motivates it. The best candidate is a leakage-safe as-of sequence/history
   residual that varies within user and uses only earlier train interactions.
   It must clear residual and paired gates; do not reopen broad architecture,
   negative-sampling, reweighting, or heterogeneous-ensemble sweeps already
   closed by evidence.
5. Ask organizers in writing whether a final train+validation refit with frozen
   hyperparameters is eligible as the "validation-best checkpoint." Implement
   it only after affirmative confirmation. Without confirmation, submit the
   eligible train-only checkpoint; do not gamble on an unscored refit.

### Required evidence

- Fold dates, labels, seeds, and rule are pre-registered and hashed.
- No official-valid score is used to choose epochs within the candidate.
- Paired deltas include GAUC, nDCG@5, primary, mean, standard deviation, wins,
  and the predeclared promotion threshold.
- A null result closes the direction and leaves `0.60541` untouched.

## P1 - Make Path B one complete lifecycle

### Goal

Prove that generated feature code can become a trustworthy experiment without
manual source copying.

### Implement

Use one state machine:

`proposal -> static/leakage check -> sandbox probe -> immutable source hash ->`
`paired retrain queue -> paired outcome -> evidence state`

Store every valid proposal and rejection. The orchestrator, not the LLM, must
inject the exact stored source into treatment runs. Deduplicate by source hash,
block validation/test labels before execution, and expose lifecycle counts in
the manifest. Add a synthetic integration fixture that completes the lifecycle;
do not weaken the real residual threshold just to produce a positive demo.

Success is either a confirmed real feature or the honest statement: "the full
lifecycle worked, but no Pure feature cleared the evidence gate."

## P1 - Produce genuine closed-loop recovery evidence

### Goal

Demonstrate what the rubric asks: after a failure, the agent completes a safe
later action without a human.

### Implement

1. Preserve Claude's current fault-accounting work in `agent/faults.py`,
   `agent/failure.py`, `agent/budget.py`, and the associated tests. Do not
   overwrite these in-flight changes.
2. Report three evidence levels separately:
   - component simulation;
   - real subprocess detection/termination;
   - full agent-loop recovery with a later successful action.
3. The existing live run detected an injected runtime error and selected debug,
   but network failures prevented a repaired result. Label it incomplete rather
   than calling it successful closed-loop recovery.
4. Run an isolated, non-scored, pre-registered suite that reaches a later
   successful action for at least: generated syntax/runtime error, malformed
   artifact, and timeout/reroute. Record retries, compute spent, final status,
   and manual interventions.
5. Keep robustness logs separate from the competition journal. A controlled
   safe stop is valid robustness evidence; it is not a recovery.

> **REVIEW: agreed throughout, and item 3 is a fair correction of my work.**
>
> The live run detected the injected error, classified it, and chose `debug` —
> then the network dropped and two LLM calls failed, so no repaired training
> result exists. Calling that "closed-loop recovery" would be overclaiming, and
> the brief is right to insist on the distinction. `results/live_fault_run/`
> and the judge packet both describe what happened rather than what was hoped
> for, but neither yet uses the word *incomplete* — Codex should add it.
>
> The three-level split in item 2 is a better framing than what I built, which
> reports two levels (component, real subprocess) and one narrative live run.
> Current state against those levels:
>
> | level | status |
> |---|---|
> | component simulation | 20 faults, 100% detected and correctly routed (`agent/faults.py`) |
> | real subprocess detection/termination | 2 faults, live through the real executor |
> | full agent-loop recovery to a later successful action | **not achieved** |
>
> Two things the fault work already contributes to item 4, worth reusing rather
> than rebuilding: the ten-axis check (detection alone is not recovery) and the
> repair/skip/pivot/abort routing distinction, which is where the interesting
> judgement lives. A timeout that gets "repaired" by re-running the same work
> is a failure even though nothing crashed.
>
> Note the accounting bug the live run exposed, since it bears on item 4's
> "record retries, compute spent, final status": the ledger was crediting
> crashed runs as unique observations while `execution_events` correctly was
> not. A crash costs compute and earns no evidence. Fixed in `agent/budget.py`.

## P2 - Reduce tokens without removing agentic code generation

### Goal

Reach the same eight-decision quality with materially less than 269,807 tokens.

### Implement

1. Use typed `ExperimentSpec` dispatch for known Path-A menu operations and
   reserve generated code for Path B or a genuinely new mechanism.
2. Ask for implementation source only after a candidate wins planning. Do not
   generate full scripts for discarded candidates.
3. Merge redundant planning/research-state calls and send only the delta in
   journal state. Keep a stable prompt prefix so provider prompt caching works.
4. For generated code, provide a small validated scaffold and request a patch,
   while still storing the final agent-authored source and diff required by the
   challenge.
5. Cache immutable diagnostics by data/code hash. Reuse never increments seeds,
   observations, or fresh training runs.
6. Benchmark before/after on a fixed smoke protocol. Target no more than 16 LLM
   calls and under 190k total tokens for an equivalent eight-decision run,
   without reducing candidate diversity or paired confirmation.

Do not optimize cost before correctness. Feasibility receives value only if the
hidden score clears the baseline.

## P2 - Strengthen innovation evidence

For every nontrivial proposal, generate a compact research card containing:

- observed failure or residual motivating the experiment;
- mechanism and why it can reorder items within a user;
- cited published method/public solution;
- dataset-specific adaptation;
- leakage analysis and expected failure mode;
- control, treatment, seeds, cost, and promotion gate;
- result and belief update.

Use the random-exposure valid window as a clearly labeled diagnostic only. The
measured random long-view rate differs sharply from the logged policy, while the
official evaluator scores the logged distribution. Do not force IPS training or
let random-log diagnostics promote a submission directly.

## P2 - Ship a judge packet

Create and generate these from the canonical manifest:

1. `docs/JUDGE_GUIDE.md`: a three-minute route covering baseline, clean run,
   tree/journal, final artifact, robustness, cost, and reproduction.

> **REVIEW: items 1 and 5 are stale — following them literally would undo
> work.**
>
> **Item 1 — extend, do not duplicate.** `results/JUDGE_PACKET.md` already
> exists (`python3 -m agent.judge_packet`, 245 lines) and is generated from the
> manifest, covering problem, loop, action space, experiment choice,
> confirmation, ensembling, both results with attribution, run cost, the fault
> suite, both convergence rules with the eligibility gap, limitations, and
> reproduction commands. `test_judge_packet_is_generated` mutates the manifest
> and asserts the packet's numbers move with it, so it cannot go stale.
> What is genuinely missing from item 1 is the *three-minute route* framing — a
> short ordered path through the evidence. Add that to the existing generator;
> a second hand-written guide would be the exact drift this replaced.
>
> **Item 5 — the dashboard now has five tabs, and the README is correct.**
> This instruction ("current four-tab dashboard... no stale five-tab text")
> was right when written and is now inverted. `app.py` has **Overview / Watch
> it run / Iteration log / Robustness / Start a run**; the Robustness tab was
> added with the fault suite and renders from the manifest. README was updated
> to match in `4b4282c`. Do not "fix" it back to four.
> The rest of item 5 stands: there is still no public repository URL in the
> README, and no pinned environment file (confirmed — no `requirements.txt`,
> `pyproject.toml` or `setup.py`), which makes item 3 a real gap.
2. `docs/DEVPOST_SUBMISSION.md`: problem, full-stack agent loop, key research
   insight, score, autonomy, robust recovery, resources, limitations, tools,
   libraries, dataset, and solo contribution.
3. A pinned environment file (`requirements.txt` or `pyproject.toml`) and one
   tested setup command from a clean environment.
4. A valid `submission_test.csv` and adjacent manifest with row count, hashes,
   eligible node, aggregation, and generation command.
5. README fixes: real public repository URL, current four-tab dashboard, exact
   commands, expected outputs, and no stale five-tab text.
6. A short screen recording: start run, circular experiment tree grows, open an
   experiment's hypothesis/diff/metrics, show recovery, verify incumbent, and
   end on the one-page rubric mapping.

## Clean Pure benchmark run

Run only after P0 and P1 tests pass and the worktree is intentionally frozen:

1. Archive prior logs without deleting final ensemble evidence.
2. Record repository commit, dependency lock, dataset/evaluator hashes, model,
   provider, prompt version, and competition profile.
3. Run a smoke test with hidden-test access mechanically unavailable.
4. Run `--competition --fresh` once with the baseline-first and early-ensemble
   sequence.
5. Regenerate manifest, results, judge guide, and valid submission from that run.
6. Verify the official convergence node contains the selected artifact.
7. Recompute the ensemble exactly and run starter-kit submission checks.
8. Inspect the final diff and require a clean repository state.
9. Only after the artifact is frozen, invoke the one-shot hidden-test command.
10. Record the hidden result without triggering any retraining, reselection, or
    second evaluation.

Suggested valid-only checks before step 9:

```bash
python3 tests/test_harness.py
python3 -m agent.verify_incumbent
python3 -m agent.manifest --run-tests
python3 -m agent.results_report --run-tests
python3 -m agent.make_submission --split valid --score --ensemble
python3 kuairand-starter-kit/submit.py --check submission_valid.csv --split valid \
  --data_dir kuairand-starter-kit/KuaiRand-Pure/data
```

Verify each command and output path from the repository root; update filenames
to the actual CLI contract rather than documenting commands that were not run.

## Do not spend remaining time on

- KuaiRand-1k or KuaiRand-27k in this cycle.
- Any transductive validation/test aggregate.
- IPS or random-log training merely to look counterfactual.
- Another broad menu, architecture, negative-sampling, reweighting, or blend
  sweep whose mechanism is already closed in `agent/experience.md`.
- Lowering residual, paired-evidence, leakage, or provenance gates for a demo.
- Choosing ensemble size, members, or weights after seeing validation scores.
- Calling component simulations full autonomous recovery.
- Polishing the dashboard before checkpoint eligibility and manifest truth.

## Definition of hackathon-ready

The project is ready when a clean KuaiRand-Pure run autonomously produces an
artifact that exists by the first official convergence point; beats the public
baseline with paired, reproducible validation evidence; preserves the one-shot
hidden-test boundary; completes real recoveries with zero manual intervention;
reports exact provider tokens and total wall-clock; and can be verified by a
judge from one manifest and one three-minute guide.

If Claude Code runs out of context before completing this brief, update
`HANDOVER_FOR_CODEX.md` with completed commits, dirty files, exact commands and
outputs, unresolved failures, and the next unchecked acceptance test. Never
claim a phase complete merely because its code exists.

---

## Review addendum — state at handover

Verified state as of this review (`python3 tests/test_harness.py`: **1077
passed, 0 failed**):

| | |
|---|---|
| Incumbent, recomputed from stored predictions | `0.60541`, exact match |
| Eligible checkpoint under the official rule | `0.60497` (node 1) — **the open gap** |
| Hidden test | unspent, no lock file |
| Faults injected / detected / correctly routed | 20 / 20 / 20 |
| Live subprocess faults | 2, both detected and classified |
| Full agent-loop recovery to a later success | **not achieved** |
| Manual interventions, all runs | 0 |
| Provider tokens, last run | 269,807 over 24 calls |
| Pinned environment file | **none** |
| Path B completed paired retrains | **0** (6 rejected, 2 probed) |

**Suggested order for Codex,** which differs from the brief only in putting the
cheap enforcement work before the expensive research:

1. **P0 competition profile** — the profile exists but does not set the
   organizer epsilon, and `min_branching_iterations` plus the pending-ensemble
   gate actively block convergence. Use the verified baseline → candidate →
   confirm → ensemble sequence in the P0 notes.
2. **P0 item 7, fail finalization on disagreement** — small, and it is what
   makes every other manifest guarantee enforceable rather than aspirational.
3. **Pinned environment file** — a judge cannot reproduce anything without it,
   and it is fifteen minutes.
4. **P1 closed-loop recovery** — the one robustness level not yet reached.
5. **P1 rolling-origin** — but measure epoch sensitivity first and drop the
   phase if the spread is under ~1σ (see the P1 review note).

**Do not** reinterpret the recorded journal to make `0.60541` eligible. It is
the best artifact in the repository and it is honestly documented as produced
after the official stopping point; a clean run is the fix, and an argument is
not.
