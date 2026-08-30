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
| 4 — Clean benchmark run | in progress | see "current action" |
| 5 — Generated documentation | not started | |
| 6 — 1K/27K transfer | not started | datasets not mounted (see CODE_REVIEW.md) |

- Tests: **778 passed, 0 failed** (`python3 tests/test_harness.py`)
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
