# Hackathon Readiness Brief for Claude Code

## Purpose and non-negotiables

This is an implementation brief for improving the **KuaiRand-Pure** agent for
TikTok TechJam. It is based on the current repository and its generated
evidence, not on a hypothetical redesign.

Current verified position:

- Incumbent validation primary: **0.60541** (GAUC `0.67212`, nDCG@5 `0.53870`),
  a fixed 16-member rank-normalised ensemble.
- Official validation baseline: `0.60160`; current margin: `+0.00381`.
- The incumbent can be recomputed exactly from stored member predictions.
- The hidden test has not been evaluated.
- The latest autonomous run used 8 decision nodes, 28 training executions,
  269,807 LLM tokens, 2,321.5 seconds of training wall-clock, and zero manual
  interventions. It converged without crashes, but did not recover from a
  fault because no fault occurred.
- Path B feature discovery probed 8 features but has not completed its
  stored-source -> paired-retrain lifecycle.

The task is **not** to chase a cosmetic dashboard change or inflate claims. The
task is to strengthen the agent against the actual rubric:

| Rubric area | Weight | What judges can verify |
|---|---:|---|
| Technical execution | 35% | final hidden-test delta, correct convergence, robust recovery |
| Innovation and problem insight | 20% | why the agent targets a meaningful full-stack mechanism |
| Impact and relevance | 20% | autonomous iteration and manual-intervention count |
| Feasibility and practicality | 15% | LLM tokens and wall-clock after clearing baseline |
| Presentation and communication | 10% | reproducible, legible submission evidence |

### Hard constraints

1. Work on **KuaiRand-Pure only**. Do not spend effort on 1k or 27k.
2. Never read hidden-test labels or use hidden-test feedback while developing.
3. Do not use transductive batch features. In particular, do **not** compute a
   feature from the whole validation/test impression batch and feed it back to
   rows in that same batch. The rejected repeat-fatigue idea falls in this
   category and must not return under a different name.
4. Do not alter `kuairand-starter-kit/evaluate.py` or redefine the score.
   The starter-kit evaluator is authoritative: `long_view`, GAUC and nDCG@5.
5. A single seed is never enough to promote a submitted configuration.
6. Keep the current 16-member incumbent immutable until a candidate has passed
   the predeclared paired-evidence gate.
7. Preserve existing logs and dirty work. Use `--fresh` for a new run; do not
   overwrite submitted artifacts.
8. Do not claim Level-A independent discovery. The present system is honestly
   a Level-B capability-transfer agent. Improve the evidence, not the label.

## Findings from the current pipeline

### What is already strong

- Baseline and incumbent provenance are unusually strong: the repository can
  rebuild and re-evaluate the reported ensemble from stored predictions.
- Train/validation/test separation is enforced technically by the execution
  sandbox, not merely by prompt text.
- The loop records hypotheses, diffs, metrics, budget, errors, and evidence
  states. Paired confirmation and ensembling are executable actions rather
  than manual post-processing.
- The capability contract, preflight checks, mechanism audit, source lineage,
  budget ledger, and generated `RESULTS.md` are excellent foundations for a
  credible agent demonstration.

### Gaps that materially affect judging

1. **Convergence reporting is not explicitly mapped to the organizer rule.**
   The agent's internal epsilon is `0.00048`, while the brief fixes organizer
   convergence at epsilon `0.002`, N `3`. The stricter research controller is
   defensible, but the final submission must explicitly show that the organizer
   rule was met and must not present a private rule as the official one.
2. **Robustness is not yet demonstrated on the final run.** Zero crashes is
   good operation, but the rubric assesses recovery after a real failure. The
   current one-off injected `RuntimeError` is not a pre-registered recovery
   benchmark and does not cover the failure classes the system says it handles.
3. **Path B is architecturally capable but unproven end-to-end.** The feature
   probe is careful, but no real feature has reached `stored -> queued paired
   confirmation -> outcome`. There are also two Path-B routes: the feature
   lab and generic generated implementation code. That makes lineage and
   recovery harder to explain.
4. **The randomized-exposure resource is only a passive diagnostic.** The
   dataset's distinctive research opportunity is counterfactual/randomized
   exposure. The current NumPy-only check writes metrics but cannot inform a
   bounded next experiment, so it earns little Innovation credit.
5. **Path-A work spends too many tokens generating boilerplate scripts.** A
   normal menu experiment is planned by an LLM and then rendered as a complete
   LLM-written script, even though the deterministic reference runner already
   executes `menu_choices` in confirmations. This raises cost and failure risk
   without adding research freedom.
6. **Judge-facing documentation has drifted from the app.** The README still
   describes five dashboard tabs, while the current UI groups the judging
   explanation into Overview. A judge should never have to reconcile docs with
   the live demo.

## Implementation sequence

Complete each phase, its tests, and its generated evidence before starting the
next one. Do not perform a hidden-test evaluation in any phase.

## Phase 0: Freeze compliance and final-selection evidence

### Goal

Make the result unambiguous to a judge: one official metric, one organizer
convergence status, one final-selection manifest, and one protected test step.

### Implement

1. Add a small `agent/convergence_report.py` (or extend the generated results
   report) with two separately named concepts:
   - `organizer_convergence`: epsilon `0.002`, N `3`, computed exactly from
     the journal according to the published rule.
   - `research_controller`: the internal epsilon and its rationale.
2. Add `results/final_selection_manifest.json`, generated only from existing
   artifacts. It must contain:
   - selected configuration and aggregation rule;
   - validation metrics and baseline delta;
   - data/config/code hashes and member paths;
   - organizer convergence result and the journal node at which it became true;
   - explicit `hidden_test_evaluated: false` before the one final command;
   - a statement that test evaluation is blocked by the existing lock.
3. Make `agent.make_submission --final-test-eval` refuse to run unless this
   manifest exists, verifies the incumbent, and names the organizer
   convergence result. It must never mutate the validation artifact while
   performing these checks.
4. Add a single command that regenerates the valid submission, validates it
   with starter-kit `submit.py --check`, verifies the incumbent, and writes the
   manifest. It must not invoke the hidden-test path.

### Acceptance tests

- Unit-test the published convergence boundary, including a case where the
  organizer rule is satisfied but the stricter controller is not.
- Unit-test that the final-test command fails closed without a valid manifest.
- Run the valid-only finalization command and confirm the output CSV passes
  starter-kit alignment/schema validation.
- Run `python3 -m agent.verify_incumbent`; it must still match exactly.

### Why this helps judging

It removes the most avoidable technical-execution risk: a judge can see that
the score, the convergence definition, and the one-shot hidden test follow the
organizer's process rather than a custom interpretation.

## Phase 1: Make Path B one reliable, testable state machine

### Goal

Prove that autonomous feature engineering is a real pipeline, while retaining
the strict leakage and residual-signal gates that protect score validity.

### Implement

1. Establish one canonical Path-B lifecycle:

   `proposal -> static/leakage validation -> probe -> immutable lineage record
   -> queued paired experiment -> paired result -> evidence state`

   Route every feature-producing Path-B action through this lifecycle. A generic
   Path-B script must not bypass the feature registry or retype a prior source.
2. Store **every** valid proposal with source hash and outcome, not only
   promising ones. A rejected feature should have a durable lineage record and
   an explicit reason; only a cleared probe may queue training.
3. Change the prompt/result wording that currently tells the model to reinsert
   feature source manually into `menu_choices`. A cleared feature should be
   referenced by immutable source hash, and the orchestrator should inject the
   exact stored source itself.
4. Add an integration fixture using tiny synthetic splits. It should create a
   safe, non-transductive feature that clears the fixture's probe, queues a
   paired comparison using the exact stored source, and records the result.
   This validates plumbing only; it is not a claimed KuaiRand improvement.
5. Add a `Path B lifecycle` section to the generated report:
   proposals, static refusals, probes, stored sources, queued confirmations,
   completed paired retrains, promotions, and any recovery after a bad builder.

### Acceptance tests

- Same source under a new name deduplicates by source hash.
- A feature that reads valid/test labels is refused before execution.
- The fixture proves that stored source bytes equal the source used by the
  paired treatment.
- A rejected real feature remains recorded and cannot be silently retried.
- `python3 tests/test_harness.py` remains green.

### Important rule

Do not lower the real residual-signal threshold merely to make Path B appear
successful. An honest result is: "the lifecycle works; no evaluated feature
cleared the evidence gate yet." The synthetic fixture is evidence of system
integrity, not model quality.

## Phase 2: Demonstrate robustness with a pre-registered fault suite

### Goal

Show the behavior the rubric asks for: the agent encounters a fault, explains
it, repairs/retries/routes around it, and continues without a human.

### Implement

1. Create `agent/robustness_eval.py` and a committed
   `config/robustness_protocol.json` **before** running the evaluation.
2. Define a small fixed corpus of non-scored faults, each with an expected safe
   route. Suggested cases:
   - nonexistent or wrong-context capability: free preflight refusal;
   - incomplete training config: actionable validation with all missing keys;
   - syntax/runtime error in a generated script: debug/retry path;
   - malformed or missing output artifact: fail closed, retry or abandon;
   - timeout: classify, shrink/reroute, and remain within the configured cap.
3. Run each case in a dedicated smoke workspace with an immutable fixture and
   no competition score claim. Record fault ID, fingerprint, original attempt,
   recovery action, retry count, compute spent, final status, and manual
   intervention count.
4. Generate `logs/robustness/RESULTS.md` from those journaled artifacts. Keep
   it visibly separate from the scored Pure run.

### Acceptance tests

- The protocol is hashed and loaded before fault execution.
- At least one preflight, one runtime, and one artifact/timeout route are
  exercised end-to-end.
- Every completed recovery has an automated next action and zero manual edits.
- A fault that cannot be safely repaired is recorded as a controlled stop, not
  as a fictitious recovery.

### Why this helps judging

It turns "zero crashes" into evidence of robust behavior rather than an
unverified claim. It directly maps to the rubric's definition of robustness.

## Phase 3: Convert randomized exposure from a report into a bounded research action

### Goal

Use the one distinctive KuaiRand resource without leakage or transduction:
the randomized-exposure validation log is a diagnostic/confirmation signal,
not extra training data or a test proxy.

### Implement

1. Extract the present `_unbiased_check` into a named capability with an
   explicit contract: it may read only the sandboxed randomized **validation**
   window; test-window rows must remain physically absent.
2. Add a structured counterfactual diagnostic artifact that reports, at minimum:
   - sample/date boundaries and row counts;
   - standard logged-policy validation metrics;
   - randomized-exposure validation metrics;
   - metric disagreement and uncertainty notes;
   - no promotion decision by itself.
3. Add a narrow, predeclared mapping from diagnostic patterns to **one** legal
   experiment family. Examples are sample-weighting, a duration/watch-time
   auxiliary objective, or a counterfactual loss already implementable from
   train-only information. The mapping must output an `ExperimentSpec` with
   control, treatment, seed set, and paired acceptance gate.
4. Require the final selection metric to remain the official logged validation
   GAUC/nDCG@5. The randomized log can motivate a hypothesis or reject a fragile
   candidate; it cannot silently replace the official objective.
5. First run a fixed diagnostic-only baseline/incumbent comparison. Only then
   allow one pre-registered paired candidate. Record a negative outcome as a
   valid research result.

### Acceptance tests

- The data-boundary test proves the random-log test dates are absent.
- The capability fails closed for a non-NumPy model until an equivalent safe
  implementation exists.
- A diagnostic cannot directly promote a submission.
- Mapping output is deterministic for a fixture diagnostic and is paired before
  promotion.

### Why this helps judging

This is a meaningful, domain-specific research direction beyond architecture
tweaks. It uses a dataset property the prompt calls out, while preserving the
official objective and hidden-test boundary.

## Phase 4: Reduce cost and failure surface without reducing statistical rigor

### Goal

Lower token and wall-clock use for the same or better research throughput. The
current 8-node run used roughly 270k LLM tokens; standard menu experiments
should not need a full generated Python program every time.

### Implement

1. For Path A/menu-only experiments, replace full-script generation with a
   typed, validated experiment object: hypothesis, evidence grounding,
   `menu_choices`, expected effect, and promotion criterion. Dispatch the
   deterministic reference runner already used by paired confirmation.
2. Reserve code generation for Path B or a genuinely new implementation that
   cannot be expressed by the menu. Keep preflight and mechanism audit for
   those cases.
3. Add a strict immutable execution cache keyed by:
   - data fingerprint;
   - reference-runner/source hash;
   - normalized menu choices;
   - seed;
   - evaluation-script hash.

   Cached observations may be reused as the same observation, never counted as
   a new independent seed or new evidence.
4. Cache static, dataset-fingerprinted diagnostics such as feature feasibility
   tables. Do not cache model outputs across differing code/config/data hashes.
5. Benchmark optional parallel workers only after deterministic parity and
   isolation tests pass. Do not enable them by default merely because the
   executor has a parallel path: the repository has a history of shared data
   locks, and CPU contention could make wall-clock worse.

### Acceptance tests

- For a representative fixed config and seed, legacy Path A and the typed
  runner produce matching predictions/metrics and equivalent provenance.
- Cache hits have a visible cache key and are excluded from training-run and
  seed-count increments.
- A code, data, evaluator, configuration, or seed change is a cache miss.
- Compare a fixed smoke workload before/after: report tokens, elapsed time,
  training executions, metrics, and failures. Adopt only if metric parity is
  exact and total cost is lower.
- If parallel mode is kept, prove sandbox isolation and output parity first.

### Why this helps judging

Feasibility is scored only among teams that clear the baseline. This improves
the relevant comparison without taking shortcuts on paired confirmation or
provenance.

## Phase 5: Add a clean autonomy evaluation, without overstating it

### Goal

Provide stronger evidence that the agent can conduct a cycle from observation
to revised action, while keeping the existing honest Level-B classification.

### Implement

1. Extend the existing journal grader with a separate `clean_autonomy` runner.
   It must use a versioned, hash-recorded prompt/context that excludes
   answer-shaped research memory, known dead ends, measured effect sizes, and
   exact winning configurations.
2. Pre-register three short, non-submission tasks before execution. Each should
   give data schema, capability contract, constraints, and budget, but not the
   answer. The desired behavior is observation -> competing hypotheses ->
   discriminating measurement -> execution -> later belief/direction change.
3. Record prompt hashes, allowed files, tool calls, generated code/config,
   journal, manual interventions, and evaluator verdict. Make the provenance
   review explicit rather than relying on a text heuristic alone.
4. Keep this evaluation separate from score search. It can demonstrate method;
   it must not be used to select the final benchmark model.

### Acceptance tests

- Forbidden answer-shaped paths cannot be read in the clean workspace.
- The existing `autonomy_eval` still reports `UNOBSERVED` when a final node has
  no later belief revision.
- The rendered report includes both passed and failed criteria without rounding
  up an incomplete trajectory.

## Phase 6: Ship a judge packet and repair presentation drift

### Goal

Let a judge verify the work in minutes, not by reading the entire repository.

### Implement

1. Add a generated `docs/JUDGE_GUIDE.md` with a 3-minute route:
   - baseline reproduction;
   - incumbent verification;
   - valid-only submission schema check;
   - generated results and resource summary;
   - one autonomous run tree/journal;
   - Path-B and robustness evidence.
2. Generate a compact `results/submission_manifest.json` beside the CSV with
   CSV hash, row count, config/ensemble provenance, valid metrics, and the
   exact command that generated it.
3. Update README dashboard documentation to match the current tabs and the
   Overview-embedded judging explanation. Remove stale claims rather than
   adding another competing description.
4. Add a Devpost-ready outline generated from the same artifacts: problem,
   architecture, score, autonomy/recovery evidence, resources, constraints,
   limitations, tools/libraries/dataset, and solo contribution.
5. Before final submission, regenerate all reports from artifacts and run all
   valid-only verification commands. The final hidden-test command remains a
   deliberate last action, once.

## Score-focused research order after reliability work

The current model-family sweep is broad and the documented residual blend is
under the noise scale. Do not spend the remaining budget on another unstructured
architecture lottery. Use this order instead:

1. Complete the counterfactual/randomized-exposure diagnostic and one
   pre-registered paired treatment if it yields a concrete mechanism.
2. Complete any legitimate Path-B candidate that clears the residual gate, with
   exact-source paired confirmation.
3. For a candidate that survives paired evidence, test the fixed ensemble
   integration using a predeclared member set and compare against mean-member
   performance, never the best sampled seed.
4. If no candidate clears the gate, retain the verified incumbent. A stable,
   reproducible `+0.00381` validation margin is better than sacrificing the
   submission to a single-seed claim.

## Do not build

- Do not implement repeat-fatigue, frequency, exposure, or other aggregates
  computed across the full validation/test batch for use on that same batch.
- Do not touch test labels, hidden test metrics, or the official evaluator.
- Do not lower Path-B probe/evidence thresholds simply to obtain a positive
  demo.
- Do not report internal epsilon `0.00048` as the organizer's epsilon `0.002`.
- Do not call a fault-free run proof of recovery.
- Do not add 1k/27k transfer work in this cycle.
- Do not use cache hits as new independent seeds.
- Do not run two training jobs concurrently until the existing isolation and
  parity tests prove it safe and faster on this machine.

## Required final verification order

1. `python3 tests/test_harness.py`
2. Regenerate the valid submission, schema-check it with the starter kit, and
   write the final-selection/submission manifests.
3. `python3 -m agent.verify_incumbent`
4. `python3 -m agent.results_report --run-tests`
5. Regenerate the judge guide, robustness report, and Devpost outline.
6. Inspect the exact artifacts that will be submitted.
7. Only when the team explicitly freezes the configuration, run the one-shot
   hidden-test evaluation.

## Definition of success

The strongest hackathon-ready result is not merely a higher validation number.
It is a baseline-beating, reproducible Pure submission whose official
convergence is clear; whose agent independently runs observation, planning,
implementation, evaluation, and reflection; whose faults are handled
automatically in a pre-registered test; whose novel counterfactual and feature
research paths are evidence-gated; and whose cost and provenance are visible in
one generated judge packet.
