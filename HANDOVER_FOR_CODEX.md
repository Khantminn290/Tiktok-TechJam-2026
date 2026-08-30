# Handover for Codex

Working branch: `opus-research-agent`. **Do not reset or checkout user changes.**
**Do not overwrite the verified incumbent.** **Do not bypass the one-time
hidden-test guard.**

## Status at a glance

| Phase | State | Evidence |
|---|---|---|
| 1 — Path B reliability | **complete** | 749→778 tests; live Path B smoke succeeded |
| 2 — Competition profile | **complete** | `--competition`, resolved config printed, unsafe combos refused |
| 3 — Budget accounting | **complete** | `budget.Ledger`, training-runs cap, confirmation affordability |
| 4 — Clean benchmark run | **complete** | `logs/opus_research/phase4_competition_run.jsonl`, `RESULTS.md` |
| 5 — Generated documentation | **complete** | `python3 -m agent.results_report` |
| 6 — 1K/27K transfer | **blocked, plan written** | `TRANSFER_PLAN.md` — datasets not on disk |

### Phase 4 note — the run predates two commits

The competition run was launched at `d7e97c1`. While it ran it exposed the next
failure layer (`selection_rule_test() missing 1 required positional argument`),
and the `call_arity` preflight stage that prevents it was committed *after* the
run started. So the Phase 4 journal reflects `d7e97c1`, not HEAD. Re-run to see
the arity stage in effect.

- Tests: **796 passed, 0 failed** (`python3 tests/test_harness.py`)
- Incumbent: **0.60541** verified (`python3 -m agent.verify_incumbent`)
- Hidden test: untouched, no `results/final_evaluation.lock`

## Phase 1 — Path B reliability (complete)

**Problem.** Every Path B crash in the last real run was a call site
disagreeing with a return shape: `train_numpy_fm` returns a **dict**, not a
2-tuple; each `capture_epoch_scores` entry is a **3-tuple, valid split only**,
with no per-epoch test vector. Four crashes cost 42s, 42s, 71s and 983s of
compute to learn facts that were already written down.

**Fix.**
- `Capability.returns` is now a machine-readable shape
  (`{"kind": "dict"|"tuple"|"list_of_tuple", ...}`) plus a copy-pasteable
  `example`. Prose was demonstrably not enough.
- New preflight stage **`return_shape`** (stages are now syntax → imports →
  capability → return_shape → config → leakage → smoke) rejects a call site that
  destructures a dict, uses the wrong tuple arity, or unpacks a capture entry
  with the wrong number of names — before any training.
- `agent/capabilities.export_contract()` writes
  `runtime/capability_contract.json`, so **generated code can read the contract
  at runtime**: `from research_tools import contract, describe`. It carries no
  data, labels or scores — a test asserts that.
- The prompt now shows return shapes and worked examples for the two
  capabilities that caused every failure.

**Verified by a live run, not just tests:** a Path B script that captures the
epoch curve and runs `selection_rule_test` on held-out users executed
successfully — `ok: True`, primary **0.60347**, 36.8s, producing
`metrics.json`, `scores_valid.npy`, `scores_test.npy` and
`selection_rule_test.json` (`mean_top3` +0.00011, 0.14σ, `generalises: false`
— correctly rejected).

## Phase 2 — Competition profile (complete)

`python3 run_agent.py --competition` enables research state, data tools,
feature discovery, multi-candidate planning (4) and branching (3), with
conservative caps: 12 iterations, **90 training runs**, 4h wall clock, $6 spend,
1800s exec timeout.

- Explicit CLI always wins; the resolved config is printed with the **source** of
  every value (`[cli]` / `[profile]` / `[default]`).
- Refused before any spend: `--competition` with `--allow-locked-options`,
  `--smoke`, or `--inject-error-at`; and a training-run cap below the iteration
  cap.
- Default mode is byte-for-byte unchanged.

## Phase 3 — Budget accounting (complete)

`agent.budget.Ledger` separates **outer-loop decisions** from **training
executions**. The counting rule, applied everywhere and stated in reports
(`budget.COUNTING_NOTE`):

> A paired 3-seed confirmation is **1 outer-loop node and 6 training
> executions**. A preflight rejection is **neither** — no compute spent, no
> decision consumed — though repeated rejections are capped.

- `_dequeue_confirmation` now checks `ledger.can_afford(spec.n_runs)`, not one
  free iteration slot. Starting a 6-run confirmation with 2 runs left produces
  unpaired arms and answers nothing, so it is deferred and stays queued.
- An exhausted training-run budget stops the run with an explicit reason.
- Crashed training executions are counted, not forgiven.
- `final_summary` carries the full ledger plus the counting note.

## Phase 5 — Generated documentation (complete)

`python3 -m agent.results_report [--run-tests]` writes `RESULTS.md` from
artifacts, never from memory: the incumbent is **recomputed** at generation
time, the convergence threshold is read from `agent.loop`, the metric is read
from `kuairand-starter-kit/evaluate.py`, and run counters come from the journal
and the budget ledger.

Every figure is tiered **VERIFIED / OBSERVED / OPEN** and the tiers are not
blurred: an unexecuted harness reports OPEN rather than assuming it passes, and
generating while a run holds the data lock degrades to OPEN rather than
reporting the quoted incumbent as though it had been verified.

Evaluator note included in the output: where the brief's prose and the starter
kit disagree, `evaluate.py` scores the submission, so `long_view` / GAUC /
nDCG@5 / their mean is authoritative. Benchmark code is untouched.

Corrected one stale README claim the generated evidence disproves (convergence
stated as a hard-coded 0.002; it is calibrated to 0.00048 = 0.60σ).

## Phase 6 — Transfer (blocked)

Only KuaiRand-Pure is on disk; 1K and 27K are not mounted anywhere. Per the
brief, `TRANSFER_PLAN.md` is a ready-to-run design, **not** a result. The
critical constraint recorded there: Pure's valid/test come from the
2022-04-22..05-08 window, so auxiliary pretraining must be cut at
`date <= 2022-04-21` or the model sees the interactions it is scored on.

## Phase 4 — Clean competition run (complete)

Ran `--competition --fresh --wall-clock-limit-h 2.0`. Converged legitimately at
iteration 10: the running best gained **0.00000** over 3 scored iterations, well
under the calibrated ε.

| | |
|---|---|
| best single run | **0.60497** (= the incumbent's own single-seed level) |
| outer-loop nodes / iterations consumed | 11 / 11 |
| **training runs used** | **16 of 90** (the confirmation was 6 of them) |
| experiments completed / crashed | 8 / 3 |
| **Path B crash rate** | **50%** (was 71% pre-architecture, 92% in the prior run) |
| orchestration-only misuse | 0 |
| paired confirmations run / promoted | 1 / **0** |
| automatic repairs attempted / recovered | 2 / **1** |
| **manual interventions** | **0** |
| LLM tokens / spend | 237,246 / **$1.57** of $6.00 |
| training wall-clock | 885s, cpu |

Behaviour worth noting: the profile's `min_branching_iterations` got the policy
out of the drafting phase for the first time in this project's history — nodes
7–10 were `improve` actions extending the best node, and node 7 reached 0.60497.
The confirmation returned UNCONFIRMED (t=1.86) and correctly promoted nothing.

**No score improvement.** 0.60497 is a single seed and equals the incumbent's
single-seed value; the submitted 16-seed ensemble remains 0.60541.

## Files changed

```
agent/capabilities.py      returns/example fields, export_contract, shapes in prompt
agent/preflight.py         new return_shape stage
agent/profiles.py          NEW — competition profile, resolution, validation
agent/budget.py            NEW Ledger + COUNTING_NOTE
agent/loop.py              ledger wiring, affordability check, training accounting
run_agent.py               --competition, --max-training-runs
runtime/research_tools.py  contract()/describe() for generated code
runtime/capability_contract.json  NEW — generated, machine-readable
tests/test_harness.py      +51 tests across the three phases
```

## Commands run

- `python3 tests/test_harness.py` → **778 passed, 0 failed**
- `python3 -m agent.verify_incumbent` → 0.60541 / 0.67212 / 0.53870, exact match
- `python3 -m agent.preflight <script>` → catches both historic failures
- Live Path B smoke via `agent.executor.run_solution` → `ok: True`, 0.60347

## Current action / next concrete step

Phase 4: a clean benchmark run under the competition profile.

```
python3 run_agent.py --competition --fresh --wall-clock-limit-h <fits your window>
```

`--fresh` archives previous search logs and **preserves submission artifacts**
(`logs/final_ensemble/`, `logs/ensemble_results.json`,
`logs/research_memory.jsonl`, `logs/opus_research/`).

After it: Phase 5 (generated documentation command) and Phase 6 (1K/27K —
likely blocked, datasets are not mounted; write the integration plan instead of
fabricating results).

## Known risks

1. **Score unchanged at 0.60541.** No confirmed improvement has been found. Two
   live confirmations both correctly declined to promote.
2. **Path B feature discovery has still never fired end-to-end in an agent run**
   — no proposed feature has ever cleared the probe (6 REJECTED, 2 PROBED). The
   discovery→confirmation pipeline is unit-tested and the Path B *execution*
   path is now demonstrated, but that specific trigger has not occurred.
3. The allocator's priors are hand-set; with ~12 iterations a family is observed
   1–3 times, so early allocations are mostly prior.
