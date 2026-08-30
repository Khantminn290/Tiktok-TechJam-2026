# Handover for Codex

Working branch: `opus-research-agent`, pushed at `e36a492`.

## Codex continuation — 2026-08-31

Codex ran the competition profile at `28142c1` and then added a narrow follow-up
reliability patch in commit `ab6c315`. The run artifacts are intentionally still
uncommitted alongside the live journal; preserve them and do not reset the
worktree.

### Latest live run

`python3 run_agent.py --competition --fresh --wall-clock-limit-h 2.0` stopped
at the charged 12-decision cap. It created 14 journal records because two
preflight rejections are deliberately journalled but do not consume a research
decision or training execution.

| | |
|---|---|
| best single-seed primary | `0.60527` (node 9; not promoted) |
| submitted incumbent | `0.60541`, unchanged |
| journal nodes / charged decisions | 14 / 12 |
| training executions | 27 of 90 |
| completed / crashed training experiments | 9 / 3 |
| Path B attempts / crashes | 7 / 5 |
| preflight rejections | 2, no training spent |
| paired confirmations / promoted | 3 / 0 (all REJECTED) |
| LLM tokens / cost | 493,573 / $1.71402 |
| training / agent wall time | 2,033.2s / 2,471.2s |
| manual intervention | 0 |

The run is valuable but does **not** establish score improvement. Node 9 is a
single validation-selected Path B result and the evidence layer correctly left
the submitted 16-seed ensemble alone.

### Patch after the run

Observed failures were all around the low-level `selection_rule_test` surface:

1. node 10 used unsupported `user_ids` / `candidate_scores` keywords after
   training;
2. nodes 12/13 supplied the wrong curve/rule container shapes.

The current uncommitted patch adds:

- `research_tools.capture_selection_rule_test(...)`, a narrow adapter from the
  exact `(epoch, valid_primary, scores_valid)` capture contract to the required
  3-D tensor and fixed rule mapping;
- a registered, machine-readable capability entry and generated runtime
  contract for that adapter;
- preflight rejection of unknown capability keywords and literal lists where
  `selection_rule_test` requires a dict of rule callables;
- honest final-summary accounting: `journal_nodes` is separate from charged
  `iterations_used`;
- results-report behavior that marks a malformed/non-zero test run **OPEN**
  rather than falsely VERIFIED.

Verified after the patch:

```text
python3 tests/test_harness.py        # 800 passed, 0 failed
python3 -m agent.verify_incumbent    # 0.60541 / 0.67212 / 0.53870 exact
python3 -m agent.results_report --run-tests
```

### Next recommended action

Do not immediately spend another full competition run. First add a narrowly
scoped integration test that makes generated Path B code use
`capture_selection_rule_test`, then run a short competition smoke (`--max-
iterations 4`) to confirm the planner chooses the adapter instead of manually
reconstructing `selection_rule_test`. Only then schedule another clean full run.

The dominant open reliability gap is semantic, well-formed Path B code. Static
preflight now catches name, arity, return-shape, keyword and direct rule-type
mistakes, but cannot prove a dynamically constructed argument has the intended
meaning. Keep claims bounded accordingly.


## Correction to the Path B crash-rate figure (2026-08-31)

`agent/run_metrics.py` was counting **preflight rejections as crashes**. That is
backwards: a preflight rejection is the system refusing to spend compute on a
script it can already see is broken, so the better preflight gets at catching
things early, the worse a conflated "crash rate" looks. Codex's run made this
visible — 2 of its 7 Path B "crashes" were 0-second rejections that never
reached training.

Fixed: `path_b_preflight_rejected` and `path_b_reached_training` are reported
separately, and the rate is taken over attempts that actually reached training.
Recomputed across every recorded run:

| run | attempts | preflight | trained | crashed | rate (of trained) |
|---|---|---|---|---|---|
| PRE-architecture (3 runs) | 7 | 0 | 7 | 5 | 71% |
| POST-architecture (3 runs) | 13 | 2 | 11 | 10 | 91% |
| Phase 4 (return shapes) | 6 | 0 | 6 | 3 | **50%** |
| Codex run (+arity/keyword) | 7 | 2 | 5 | 3 | **60%** |

**Do not over-read the last two rows.** Training crashes have been stuck at
exactly **3 per run** across both, while attempts varied. At n=5–6 the
difference between 50% and 60% is one attempt and is not a meaningful change in
either direction. What is defensible: the rate is clearly below the 91%
post-architecture regression, and preflight is now catching attempts that
previously would have burned a training run. What is not yet defensible: that
Path B reliability is trending in a particular direction run-to-run.

**Safety constraints — please preserve these.**
- Do not reset, checkout or revert user changes. `CODE_REVIEW.md` and
  `QUICK_IMPLEMENTATION.md` are untracked on purpose; leave them.
- Do not overwrite the verified incumbent (`logs/final_ensemble/`,
  `logs/ensemble_results.json`).
- Do not bypass the one-time hidden-test guard. No
  `results/final_evaluation.lock` exists and the hidden test has never been read.
- `main` is untouched (`b13d632`) and should stay that way.

## Status

| Phase | State | Evidence |
|---|---|---|
| 1 — Path B reliability | **partly effective** | live Path B smoke succeeded; crash rate 71%→50/60%, see caveat |
| 2 — Competition profile | **complete** | `--competition`; resolved config printed; unsafe combos refused |
| 3 — Budget accounting | **complete** | `budget.Ledger`; 11 nodes vs 16 training runs in the live run |
| 4 — Clean benchmark run | **complete** | `logs/opus_research/phase4_competition_run.jsonl`, `RESULTS.md` |
| 5 — Generated documentation | **complete** | `python3 -m agent.results_report --run-tests` |
| 6 — 1K/27K transfer | **blocked, plan written** | `TRANSFER_PLAN.md`; datasets not on disk |

Verified at handover time:

- `python3 tests/test_harness.py` → **796 passed, 0 failed**
- `python3 -m agent.verify_incumbent` → **0.60541** / 0.67212 / 0.53870, exact match
- Hidden test: never evaluated

**The submitted score did not improve.** It is still 0.60541. Read §"What did
not work" before anything else.

---

## Phase 1 — Path B reliability

**Problem.** Every Path B crash in the previous run was a call site disagreeing
with a return shape the contract already knew: `train_numpy_fm` returns a
**dict**, not a 2-tuple; each `capture_epoch_scores` entry is a **3-tuple,
valid split only**, with no per-epoch test vector. Those cost 42s, 42s, 71s and
983s of training to discover facts already written down.

**Fix.**
- `Capability.returns` and `Capability.params` are machine-readable
  (`{"kind": "dict"|"tuple"|"list_of_tuple"}`, `{"required": [...]}`), plus a
  copy-pasteable `example`. Prose was demonstrably not enough — the agent read
  `"{'scores_valid': ...}"` and still destructured it.
- Preflight stages are now: syntax → imports → capability → **call_arity** →
  **return_shape** → config → leakage → smoke.
- `runtime/capability_contract.json` is generated from the registry so
  **generated code can read the contract at runtime**
  (`from research_tools import contract, describe`). It carries no data, labels
  or scores; a test asserts that.

**Verified live:** a Path B script that captures its own epoch curve and runs
`selection_rule_test` on held-out users executed successfully — `ok: True`,
primary **0.60347**, 36.8s, producing `metrics.json`, both score vectors and
`selection_rule_test.json` (`mean_top3` +0.00011, 0.14σ, `generalises: false`).

## Phase 2 — Competition profile

`python3 run_agent.py --competition` enables research state, data tools, feature
discovery, 4-candidate planning and branching, with conservative caps: 12
iterations, **90 training runs**, 4h wall clock, $6 spend, 1800s exec timeout.

- Explicit CLI always wins; the resolved config prints the **source** of every
  value (`[cli]` / `[profile]` / `[default]`).
- Refused before any spend: `--competition` with `--allow-locked-options`,
  `--smoke`, or `--inject-error-at`; and a training-run cap below the iteration
  cap.
- Default mode is unchanged.

## Phase 3 — Budget accounting

`agent.budget.Ledger` separates decisions from executions:

> An outer-loop node is one decision; a training execution is one model actually
> trained. A paired 3-seed confirmation is **1 node and 6 training executions**.
> A preflight rejection is **neither** — no compute spent, no decision consumed —
> though repeated rejections are capped.

`_dequeue_confirmation` checks `ledger.can_afford(spec.n_runs)`, not one free
iteration slot: a 6-run confirmation started with 2 runs left produces unpaired
arms and answers nothing, so it defers and stays queued. An exhausted
training-run budget stops the run with a reason. Crashed executions are counted,
not forgiven.

## Phase 4 — Clean competition run

`--competition --fresh --wall-clock-limit-h 2.0`. Converged legitimately at
iteration 10: the running best gained **0.00000** over 3 scored iterations.

| | |
|---|---|
| best single run | **0.60497** (= the incumbent's own single-seed level) |
| outer-loop nodes / iterations consumed | 11 / 11 |
| **training runs used** | **16 of 90** (the confirmation was 6) |
| experiments completed / crashed | 8 / 3 |
| **Path B crash rate** | **50%** of attempts that reached training (see correction above) |
| orchestration-only misuse | 0 |
| paired confirmations run / promoted | 1 / **0** (UNCONFIRMED, t=1.86) |
| automatic repairs attempted / recovered | 2 / **1** |
| **manual interventions** | **0** |
| LLM tokens / spend | 237,246 / **$1.57** of $6.00 |
| training wall-clock | 885s, cpu |

The profile's `min_branching_iterations` got the policy out of the drafting
phase for the first time in this project — nodes 7–10 were `improve` actions
extending the best node.

**Caveat: this run predates two commits.** It launched at `d7e97c1`, then hit
`selection_rule_test() missing 1 required positional argument` **twice**. The
`call_arity` stage that prevents it was committed *after* the run started, so
the journal cannot show it working. That recurrence is the evidence the stage
targets a real class rather than a one-off.

## Phase 5 — Generated documentation

`python3 -m agent.results_report [--run-tests]` writes `RESULTS.md` from
artifacts: incumbent **recomputed** at generation time, convergence threshold
read from `agent.loop`, metric read from `kuairand-starter-kit/evaluate.py`,
counters from the journal and ledger.

Figures are tiered **VERIFIED / OBSERVED / OPEN** and the tiers are not blurred:
an unexecuted harness reports OPEN rather than assuming it passes, and
generating while a run holds the data lock degrades to OPEN rather than
reporting the quoted incumbent as though verified.

Evaluator note in the output: where the brief's prose and the starter kit
disagree, `evaluate.py` scores the submission, so `long_view` / GAUC / nDCG@5 /
their mean is authoritative. Benchmark code untouched.

Corrected the one stale README claim the evidence disproves (convergence stated
as a hard-coded 0.002; it is calibrated to 0.00048 = 0.60σ).

## Phase 6 — Transfer (blocked)

Only KuaiRand-Pure is on disk; 1K and 27K are not mounted anywhere
(`find / -maxdepth 4 -iname "KuaiRand-1K" -o -iname "KuaiRand-27K"` → nothing).
`TRANSFER_PLAN.md` is a ready-to-run design, **not a result**; no transfer
number exists and none is claimed.

Critical constraint recorded there: Pure's valid/test come from the
2022-04-22..05-08 window, and 1K/27K cover the same period and largely the same
catalogue, so auxiliary pretraining must be cut at `date <= 2022-04-21`. A
data-limited model — which the learning curve says this one is (+5.12σ on the
last doubling) — will absorb that leakage and report a large fake gain.

---

## What did not work

1. **No score improvement.** Priority 1 of the brief. The best run was 0.60497,
   which is a single seed and equals the incumbent's own single-seed value. The
   submitted 16-seed ensemble is still 0.60541. Two live paired confirmations
   both correctly declined to promote (UNCONFIRMED at t=1.86 and t=2.25).
2. **Path B feature discovery has still never fired end to end.** 6 REJECTED,
   2 PROBED, 0 PROMISING — no proposed feature has ever cleared the probe, so
   the discovery→confirmation pipeline remains structurally verified but
   untriggered. The Path B *execution* path is now demonstrated separately.
3. **The allocator's priors are hand-set.** With ~12 iterations a family is
   observed 1–3 times, so early allocations are mostly prior. Data overrides via
   a Beta posterior, but slowly at this scale.
4. **Preflight is static.** It now catches missing functions, wrong arity and
   wrong return shapes. It cannot catch a semantically wrong but well-formed
   call — the remaining Path B crash in Phase 4 was
   `TypeError: 'method' object is not subscriptable`.

## Files changed

```
agent/capabilities.py             returns/params/example, export_contract, shapes in prompt
agent/preflight.py                new call_arity + return_shape stages
agent/profiles.py            NEW  competition profile: resolve, validate, render
agent/budget.py                   Ledger + COUNTING_NOTE
agent/loop.py                     ledger wiring, affordability check, training accounting
agent/results_report.py      NEW  generated RESULTS.md from artifacts
run_agent.py                      --competition, --max-training-runs
runtime/research_tools.py         contract()/describe() readable by generated code
runtime/capability_contract.json  NEW, generated
tests/test_harness.py             +69 tests across the phases
README.md                         corrected stale convergence claim
RESULTS.md                   NEW  generated
TRANSFER_PLAN.md             NEW  Phase 6 plan (blocked)
logs/opus_research/phase4_*       run journal + metrics
```

## Commands run, with outcomes

| command | outcome |
|---|---|
| `python3 tests/test_harness.py` | **796 passed, 0 failed** |
| `python3 -m agent.verify_incumbent` | 0.60541 / 0.67212 / 0.53870, exact match |
| `python3 -m agent.preflight <script>` | catches unpack, capture-arity and missing-arg failures |
| Path B smoke via `executor.run_solution` | `ok: True`, primary 0.60347, 36.8s |
| `run_agent.py --competition --fresh` | converged at iter 10; table above |
| `python3 -m agent.results_report --run-tests` | wrote `RESULTS.md`, 796 VERIFIED |

## Next concrete action

Re-run the competition profile **at HEAD**:

```bash
python3 run_agent.py --competition --fresh --wall-clock-limit-h 2.0
```

Both remaining Path B crashes in Phase 4 were the arity error that HEAD now
catches statically, so this directly tests whether the crash rate drops below
50%. It is also the cheapest way to find the next failure layer — each fix so
far has exposed exactly one more, in order: *where does this live* → *what does
it return* → *what arguments does it take*.

`--fresh` archives previous search logs and **preserves** submission artifacts
(`logs/final_ensemble/`, `logs/ensemble_results.json`,
`logs/research_memory.jsonl`, `logs/feature_registry.jsonl`,
`logs/opus_research/`).

## Tests still worth adding

- An end-to-end Path B feature test that forces a probe to PROMISING, so the
  discovery→confirmation pipeline is exercised rather than only unit-tested.
- A test that a semantically-wrong-but-well-formed call is *not* falsely
  rejected by `call_arity` (currently covered only for `*args`/`**kwargs`).
- Coverage for `results_report` when `logs/journal.jsonl` is absent entirely.
