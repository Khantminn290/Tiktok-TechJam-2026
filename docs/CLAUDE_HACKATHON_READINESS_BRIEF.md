# Claude Code Brief: Make the Pure Agent Hackathon-Ready

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

### Required tests

- Pin the current true total at 269,807 tokens and catch the incorrect 137,446
  node-only total.
- Pin single `0.60497` versus ensemble `0.60541` labeling.
- Render a legacy and a fully instrumented journal without `KeyError`.
- Regenerate twice and assert stable hashes except for timestamp fields.

## P1 - Improve hidden-test generalization, not public-valid overfit

### Goal

Use the strongest measured clue: the model gains `+0.00410` from the final
50%-to-100% training-data doubling, while epoch selection is highly sensitive.
With bonus datasets excluded, better use of Pure's legal training period is the
highest-value score direction.

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
python3 kuairand-starter-kit/submit.py --check submission_valid.csv --split valid
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
