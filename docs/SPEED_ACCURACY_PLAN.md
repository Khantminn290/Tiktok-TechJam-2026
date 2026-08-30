# Speed Without Sacrificing Evidence

## Purpose

This is the implementation brief for making the autonomous research workflow
meaningfully faster while preserving its scientific safeguards and its best
defensible validation result. It is deliberately not a plan to maximise a noisy
validation peak. The objective is to reach, validate, and reproduce a strong
submission more efficiently.

**Current verified incumbent:** `0.60541` primary, a fixed 16-seed
rank-normalised ensemble (GAUC `0.67212`, nDCG@5 `0.53870`). It must remain
reproducible throughout this work.

**Observed competition run:** 12 charged decisions, 27 training executions,
44 LLM calls, 493,573 LLM tokens, $1.71402 spend, and 2,033.2 seconds of
training wall-clock. The high-confidence speed target is LLM/orchestration,
not validation shortcuts.

## Non-Negotiable Evaluation Contract

The official evaluator is the source of truth:

- Dataset: KuaiRand-Pure unless an explicit auxiliary-data profile is selected.
- Positive label: `long_view`.
- Metrics: GAUC and nDCG@5; primary is their mean.
- Baseline: `0.6016` validation primary and `0.5946` test primary.
- The starter-kit evaluator, split construction, and scorer must never be
  modified to obtain speed or score.
- The hidden test may be evaluated exactly once, only at final submission.
- A one-seed result is always `PRELIMINARY`. It cannot change the submitted
  configuration. Promotion remains a paired multi-seed decision with the
  existing evidence threshold.
- The final ensemble remains pre-declared: all members of one configuration,
  not validation-selected top-K models.

Speed work that violates any item above is a regression, even if it reports a
higher validation number.

## What Judges Need To See

| Judging area | Current strength | Speed-and-quality addition |
|---|---|---|
| Technical execution (35%) | Frozen official metric, test boundary, provenance, preflight, 810 harness checks. | Deterministic Path A runner, exact-result cache, and performance telemetry make each action cheaper and reproducible. |
| Innovation and problem insight (20%) | Capability contract, evidence states, transparent allocator, feature lineage. | Cost-aware staged experimentation: the agent spends full code-generation effort only when evidence says it is warranted. |
| Impact and autonomy (20%) | Zero manual interventions in the reported run; recovery and preflight are visible. | The agent chooses fast versus deep work itself, reuses verified work, and explains why a costly branch was skipped. |
| Feasibility and practicality (15%) | CPU-only resource accounting and hard budget caps. | A measured latency/token budget, bounded concurrency, and cache hit-rate report turn feasibility into proof. |
| Presentation (10%) | Generated results, live view, static experiment tree. | A compact before/after performance card and a screen-recordable fast run make the improvement immediately legible. |

The highest-value demonstration is not "the agent made many model calls." It is
"the agent selected a useful experiment, ran it safely, refused weak evidence,
and completed the same scientific loop under a defined resource budget."

## Diagnosis: Where Time Is Going

1. Multi-candidate planning is already a single LLM call and deterministic
   selection, which is good. The selected candidate is then still sent through
   a second full-script generation path. For Path A menu experiments this is
   unnecessary: the runtime already has a known solution interface.
2. `max_output_tokens` is globally 32,000. Small JSON plans, feature probes,
   repairs, and full custom scripts therefore share an unnecessarily generous
   output allowance.
3. The reported run used 44 LLM calls for 12 decisions. Static prompt material
   and repeated summaries should be compacted, cached, and traced per call.
4. Confirmations correctly use paired multi-seed evidence, but identical
   deterministic control arms may be re-trained rather than reused from an
   exact, verified artifact.
5. Parallel execution exists, but it must not be enabled by default without a
   hardware benchmark. Parallel proposals can increase compute and contention,
   not merely reduce wall-clock.
6. Path B is important for innovation, but it is the expensive/high-risk path.
   It needs a stricter feasibility gate and one narrowly-scoped code path, not
   repeated open-ended script generation.

## Design Principles

- **Compress exploration, never confirmation.** Use cheap deterministic work
  to reject weak ideas; keep the paired multi-seed promotion rule unchanged.
- **Cache facts, not decisions.** An exact prior run may be reused. A cached
  score must never be treated as extra independent evidence.
- **The LLM proposes; code enforces.** Use the LLM for hypotheses and the rare
  custom mechanism. Let schemas, templates, static checks, and arithmetic make
  routine decisions deterministic.
- **Escalate quality based on risk.** A compact planner can handle menu choices.
  Use the strongest model and a larger output budget only for a new Path B
  mechanism that passed feasibility checks.
- **Measure every claimed acceleration.** No new "fast" profile is accepted
  without comparable run telemetry and unchanged evaluator output.

## Implementation Order

Implement and test each phase before proceeding. Do not start the optional
transfer work until the Pure workflow and artifacts are stable.

### Phase 1: Deterministic Path A Execution

**Goal:** A menu/template experiment should not require an LLM to write a full
Python script.

1. Add a versioned `PathASpec` schema. It contains only the hypothesis,
   rationale, expected effect, category, and validated `menu_choices`.
2. Change candidate planning so Path A candidates return `PathASpec` data, not
   code. Keep the full candidate set and deterministic selection trace in the
   journal exactly as today.
3. Add a template renderer that produces the canonical solution script from
   `runtime/seed_solution.py`, the selected `menu_choices`, and the template
   version. The rendered source must be stored in `logs/solutions/` so the
   existing reproducibility story remains intact.
4. Use the existing preflight, leakage boundary, executor, scorer, provenance,
   and journal paths. This is a different *source-production* route, not a
   different training or evaluation route.
5. Keep Path B on the existing full-code path. Never silently downgrade a
   proposed Path B mechanism into a template experiment.

**Acceptance tests**

- A valid Path A spec renders a complete runnable script with no LLM script
  generation call.
- Rendered Path A source produces the same metrics and prediction arrays as the
  canonical script for a fixed config, seed, data fingerprint, and template
  version.
- Invalid, locked, or incomplete menu choices fail before training with useful
  structured feedback.
- The journal visibly records `source_mode: template` versus `source_mode: llm`.
- Existing `tests/test_harness.py` remains fully passing.

**Expected benefit:** removes at least one large LLM completion from normal
Path A decisions while making the routine path more reliable.

### Phase 2: Compact, Typed LLM Stages

**Goal:** Give each LLM call the smallest prompt, output budget, and model tier
that can safely perform its task.

1. Split the current LLM use into explicit stages:

| Stage | Output | Default quality tier | Must not do |
|---|---|---|---|
| `plan` | JSON candidate set plus inquiry | compact/fast | emit Python code |
| `path_b_design` | typed custom-mechanism design | strong | train or score |
| `path_b_code` | one complete script from approved design | strong | change evaluator/boundary |
| `repair` | minimal patch or structured rejection reason | compact first, strong on escalation | repeat an identical failure |
| `summarise` | bounded state update | compact | make promotion decisions |

2. Give every stage an explicit maximum output size and timeout. Start
   conservatively, record truncation/repair rates, and tune from evidence. Do
   not retain one global 32k-output setting for all stages.
3. Build a compact prompt context from immutable, hashed fragments:
   evaluator contract, relevant menu axes, current frontier, relevant dead ends,
   recent failures, and the requested output schema. Store fragment hashes and
   the final prompt hash in the journal, not just an opaque prompt string.
4. Retrieve only dead ends and research findings relevant to the requested
   axis/path. Preserve the full research state on disk; do not delete history
   merely to shorten a prompt.
5. Add per-stage telemetry: request count, input/output tokens, latency,
   timeout, parse failure, repair count, model, and prompt/schema version.

**Safety rule:** a smaller/cheaper model may classify, summarise, or propose a
typed Path A candidate. It may not approve promotion, bypass preflight, alter
the benchmark contract, or generate an unreviewed high-risk Path B script.

**Acceptance tests**

- A replay fixture produces schema-valid plans with the compact context.
- Prompt hashes change when and only when the relevant source fragments change.
- A malformed compact response follows the bounded repair/reject path and does
  not reach training.
- The results report exposes aggregate LLM telemetry without exposing prompts,
  credentials, or labels.

### Phase 3: Exact Evaluation Cache

**Goal:** Never rerun an identical deterministic execution while preserving the
meaning of paired evidence.

1. Add an append-only execution cache keyed by a canonical SHA-256 of:
   rendered source hash, template version, resolved menu choices, seed, dataset
   fingerprint, sandbox/data-boundary version, evaluator hash, and runtime
   environment fingerprint.
2. Cache only a completed execution with valid metrics, prediction arrays,
   resource record, and provenance. Never cache a partial failure as success.
3. On a hit, materialise a read-only reference to the original artifact and log
   `cache_hit`, source run ID, and the complete cache key. Do not copy a score
   without its predictions and provenance.
4. Count a cache hit as **zero new training executions** and **zero new
   statistical observations**. It can supply an unchanged control arm, but it
   cannot increase the confirmation sample size.
5. Invalidate naturally whenever any key component changes. Add an explicit
   `--no-execution-cache` switch for diagnosis and a report of cache hits,
   misses, and saved wall-clock.

**Acceptance tests**

- Same source/config/seed/fingerprints hits and reproduces exact metric and
  prediction checksums.
- Changing one key component misses.
- A cache hit does not alter the paired result's `n`, standard error, or
  evidence state.
- Corrupt or incomplete cached artifacts are rejected and re-executed safely.

### Phase 4: Evidence-Gated Search Scheduling

**Goal:** Spend the finite run budget where it can still produce a confirmed
result.

Use an explicit, journalled state machine rather than a fixed amount of
planning at every iteration:

| Stage | Entry condition | Work allowed | Exit condition |
|---|---|---|---|
| Establish | no verified incumbent/repro check | reproduce baseline/incumbent, warm cache, inspect once | evaluator and provenance verified |
| Explore | confirmed budget remains and open axes exist | two compact candidates; one-seed screening; bounded Path B probe | promising result or no open mechanism |
| Validate | a result clears a pre-registered screening threshold | paired 3-seed control/treatment confirmation | confirmed or rejected |
| Exploit | limited budget remains | confirmation, fixed ensemble/reseed, report generation | final artifact complete |
| Closeout | submission window | verify incumbent and make artifacts | one-time final evaluation only when chosen |

Rules:

- Start with two candidates. Escalate to four only when the deterministic
  frontier shows genuinely unresolved, distinct axes. This retains diversity
  without treating every routine choice as a four-way LLM task.
- Do not open a branch unless enough training budget remains to both screen and
  confirm it. The allocator should expose this as a reason, not hide it.
- Queue confirmation immediately for a qualifying screened result; do not let
  later exploration starve a result that could change the submission.
- A one-seed screen uses one deterministic seed only to rank hypotheses. The
  paired confirmation remains at three or more fixed seeds and is the sole
  adoption mechanism.
- Reuse the exact cache for an identical control arm when valid. Run every new
  treatment seed normally.

**Acceptance tests**

- A replayed journal shows why each stage transition occurred.
- The allocator refuses an unconfirmable late branch.
- No amount of one-seed screening can produce `CONFIRMED` or a promotion.
- Rejected features and identical fault fingerprints cannot be requeued without
  an explicit, logged counterevidence reason.

### Phase 5: Path B as a Narrow, Reliable Capability

**Goal:** Preserve the innovation path without spending full training runs on
avoidable code or contract errors.

1. Treat a Path B proposal as a two-step object: a typed `FeatureDesign` or
   `MechanismDesign`, then a generated implementation only after design
   validation passes.
2. Validate required inputs, output shape/dtype, train-only provenance, and
   compatibility with `train_numpy_fm` before code generation where possible.
3. Add a tiny dry-run that invokes only the custom builder on a bounded,
   synthetic or redacted train slice. It must not call the evaluator or train a
   model. This supplements, rather than replaces, static preflight.
4. Give each capability a template/example for the minimum viable integration.
   The generator should extend that contract, not invent imports or incomplete
   configuration dictionaries.
5. Apply a per-run Path B quota until a probe passes. A useful starting policy
   is one Path B probe every three charged decisions and at most two unresolved
   Path B branches. Release the quota only after an end-to-end successful
   retraining with recorded feature lineage.
6. Map each failure fingerprint to one of: repair once with structured feedback,
   reject before training, or abandon. Never ask the model to "try again" with
   the same unclassified failure.

**Acceptance tests**

- Known invalid imports, missing config keys, invalid output shape, and target
  leakage are rejected before a full training execution.
- The dry-run cannot access the hidden test labels or write protected paths.
- A successful feature is stored with source, hashes, train-only provenance,
  probe result, retraining result, and paired confirmation result.
- The results report distinguishes preflight/dry-run rejection from a crash
  that consumed training compute.

### Phase 6: Safe Parallelism, Only After Measurement

**Goal:** Reduce wall-clock, not increase contention or invalidate comparisons.

The repository already has isolated worktrees and a shared-lock
`run_parallel_round`. Use that foundation, but do not make `--parallel-k` the
competition default until it passes the protocol below.

1. Add a `hardware_profile` benchmark: one warm sequential run, then controlled
   two-worker runs of independent Path A screens and paired confirmation arms.
   Record CPU model/count, peak RSS if available, per-run wall-clock, and metric
   checksums.
2. Start at two workers. Pick the concurrency cap from measured wall-clock
   improvement, memory headroom, and zero integrity failures; never from CPU
   count alone.
3. Parallelise independent screens or fixed paired arms only. Do not parallelise
   dependent planning, prompt mutation, promotion, journal writes, or final
   ensemble construction.
4. The coordinator is the only process allowed to append the journal, update
   budget accounting, or promote an incumbent. Results must be sorted by their
   allocated run ID, never completion order.
5. Fall back to sequential execution automatically after sandbox-lock, resource,
   or integrity failure. Log the fallback as an autonomous recovery event.

**Acceptance tests**

- Two-worker and sequential runs have identical result artifacts for identical
  jobs/seeds, apart from measured timestamps.
- The real data/cache permissions are restored after success, failure, timeout,
  and interrupt.
- Training-run accounting counts every launched subprocess exactly once.
- A hardware profile that shows contention keeps the safe sequential default.

## Performance Scorecard and Gates

Create `logs/performance_summary.json` and render it into `RESULTS.md` or a
generated companion report. Report before/after runs under the same Pure
profile, data fingerprint, model, seed policy, and training budget.

Required fields:

- total and per-stage LLM calls, input/output tokens, cost, latency, retries;
- planning, preflight, dry-run, training, confirmation, and reporting wall-clock;
- training executions launched, cache hits/misses, and wall-clock saved;
- Path A template versus LLM-generated source counts;
- Path B probe/retrain/confirmation counts and failure classifications;
- evaluator hash, data fingerprint, environment fingerprint, and test status;
- best single score, confirmed score, submitted-incumbent verification result.

Initial targets are **gates to measure against**, not claims to put in a demo
before they are achieved:

| Metric | Target | Protection |
|---|---|---|
| LLM tokens for a comparable 12-decision Pure run | at least 45% below the observed 493,573 | same model policy for high-risk Path B; report model mix |
| LLM script calls for Path A | zero after template rendering | source hash/metric parity test |
| Orchestration wall-clock | at least 40% below baseline, measured separately from training | no change to evaluator or confirmation seed count |
| Duplicate deterministic training executions | zero when an exact cache entry exists | cache key includes all evaluation-relevant fingerprints |
| Validation evidence | no weaker than current paired rules | no promotion from screens/cache alone |
| Regression safety | all harness checks pass, incumbent verifies exactly | hidden test remains untouched |

Do not set a total wall-clock target until Phase 6 measures real two-worker
contention. The recorded training alone is 2,033.2 seconds, so a dramatic
end-to-end claim without controlled parallelism would not be credible.

## Hackathon-Ready Runbook

1. Run `python3 tests/test_harness.py` before any scored session.
2. Regenerate and archive the incumbent verification and results report.
3. Run a cheap smoke test after changes to LLM transport, schemas, cache, or
   sandboxing. It is plumbing verification, not benchmark evidence.
4. Run one clean Pure competition profile with a declared time, LLM, training,
   and concurrency budget. Start the live view before the run for recording.
5. Regenerate the results report and performance scorecard from artifacts only.
6. Screen-record the live view and export the static experiment tree. Show an
   inquiry, a preflight/dry-run rejection or recovery, a weak result held as
   preliminary, and the budget counters.
7. Re-run `verify_incumbent` and the harness. Only after these checks and a
   deliberate team decision should `--final-test-eval` be invoked once.

## Optional 1K / 27K Transfer

This is a bonus lever, not a prerequisite for a strong Pure submission. Attempt
it only after the Pure performance scorecard is stable and the data is actually
available.

- Follow `docs/TRANSFER_PLAN.md` exactly: explicit profile routing, hard
  `date <= 2022-04-21` auxiliary cutoff, label redaction, embeddings-only
  pretraining, Pure-only fine-tuning/evaluation, and paired confirmation.
- Test 1K first. A confirmed null makes 27K hard to justify.
- Chunk/cache auxiliary data by dataset fingerprint. Never auto-discover a data
  directory and silently include it in a Pure run.
- Give transfer a separate resource budget and report it independently so it
  cannot obscure the Pure feasibility story.

## Explicitly Out of Scope Until This Plan Passes

- Adding more model families or random hyperparameter axes.
- Relaxing confirmation seeds, using top-K validation-selected ensembles, or
  selecting a cached best run as new evidence.
- Changing the evaluator, split dates, labels, or hidden-test boundary.
- Enabling high parallelism without the hardware profile.
- Claiming an unexecuted 1K/27K transfer result.

## Definition of Done

The work is ready to present when all of the following are true:

1. The Path A template path and exact cache are implemented with the acceptance
   tests above.
2. A clean benchmarked run publishes a generated performance scorecard showing
   measured reductions, not estimates.
3. The incumbent still verifies exactly at `0.60541`, and all harness checks
   pass.
4. No hidden test evaluation has occurred.
5. The live view/static tree/report explain the agent's autonomous decisions,
   recoveries, evidence states, budgets, and final reproducibility in a form a
   judge can understand in a few minutes.

