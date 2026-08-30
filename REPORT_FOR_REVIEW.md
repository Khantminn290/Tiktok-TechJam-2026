# Work report for external review

Branch `opus-research-agent`. Everything below is reproducible from the
repository; commands are given inline. Written for a reviewer who should assume
nothing and check the claims.

**Headline: the submitted score did not improve. It is still 0.60541.** What
changed is that the agent can now, for the first time, run an experiment capable
of establishing that something is better — and the reviewer should weigh that
against the fact that it has not yet found anything that is.

---

## 0. What a reviewer should be most sceptical about

Listed first, deliberately.

1. **No score improvement.** Priority 1 in the brief was a better converged
   score. It did not happen, and §6 argues the benchmark is genuinely plateaued
   rather than under-searched. A reviewer should test that argument, not accept
   it.
2. **The end-to-end incumbent rebuild is still outstanding** at the time of
   writing (audit finding 7). The incumbent is verified at artifact level —
   the 16 stored prediction arrays recombine to exactly 0.60541 — which proves
   the arithmetic, not that today's code still produces those arrays.
3. **Path B feature discovery has still never fired end to end.** No feature has
   ever cleared the probe (6 REJECTED, 2 PROBED, 0 PROMISING). The new
   discovery→confirmation pipeline is unit-tested and structurally verified, but
   the trigger condition has never occurred in a real run.
4. **The allocator's priors are hand-set.** Data overrides them via a Beta
   posterior, but with 8 iterations a family is observed 1–3 times, so early
   allocations are mostly prior. That is stated in the code and is a real limit.

---

## 1. Phase 1 — what verification actually found

Two defects that were structural, not cosmetic. Both were found by inspecting
behaviour rather than reading intentions.

### 1a. Confirmation was never executable

```
$ python3 -c "..."   # seeds across every recorded node
seeds seen across all recorded nodes: {0: 37}
```

**Every node in every recorded run used seed 0. All 37.** "Confirmation" was a
research category that changed the wording of a prompt. The only multi-seed
machinery in the repository, `agent/reseed.py`, was a post-hoc tool the loop
never called.

The consequence was structural and is the reason the agent could never beat its
own incumbent with evidence:

```
one seed -> evidence.PRELIMINARY -> never actionable
...and no code path existed that could produce a second seed.
```

The agent could form a hypothesis, measure it, correctly report that a single
seed proves nothing, and then do nothing about it. It had been disciplined into
permanent inaction.

### 1b. Path B discoveries were discarded

```
$ grep -rn "_pending_feature" agent/ runtime/ tests/
agent/loop.py:122:        self._pending_feature = None
agent/loop.py:464:        self._pending_feature = (obj if ... else None)
```

Two references: one to initialise, one to assign. **Nothing ever read it.** And
`feature_source` — the only channel by which a discovered feature could reach
training — has never once appeared in any journal across every recorded run.
The sole route in was the model noticing a line of prompt text and retyping the
builder source by hand.

---

## 2. Phase 3 — the experiment specification

`agent/experiment_spec.py`. An `ExperimentSpec` carries parent, control,
treatment, hypothesis, expected GAUC/nDCG/primary effects, seed set, type,
runtime budget, acceptance threshold, promotion rule, rollback, feature lineage
and evidence tier — and is **executed**, not described.

Nine types: exploration, improvement, branch, crossover, path_b_discovery,
path_b_confirmation, multi_seed_replication, ensemble_construction,
debug_recovery.

Two constraints worth checking:

- A confirmatory spec **refuses fewer than 3 seeds**. Two points estimate spread
  from a single difference.
- An unset acceptance threshold defaults to half the noise floor, not to zero.

## 3. Phase 2/3 — confirmation that actually runs

`agent/confirm.py` executes a spec: both arms run the *same* reference script
(`runtime/seed_solution.py`) at the *same* seeds, so the only difference is the
configuration under test. No LLM sits between the decision and the measurement.

**Verified by a live 6-run paired confirmation**, and it immediately earned its
place:

```
    [control]   seed 0  0.60497   seed 1  0.60393   seed 2  0.60449
    [treatment] seed 0  0.60499   seed 1  0.60393   seed 2  0.60450
    delta +0.00001 (+0.01 sigma)  t=2.88  wins 3/3
    EVIDENCE: REJECTED — under half the noise floor
    PROMOTE:  NO
```

**t = 2.88 with every seed "winning", and it is correctly rejected.** A naive
significance test would have promoted a +0.01σ effect. This is the single
clearest demonstration that the evidence layer is doing real work.

## 4. Phase 4 — Path B made cumulative

`agent/feature_store.py` persists a discovery with its exact source, a hash of
the *normalised* source, mechanism, input columns, probe numbers, parent node,
and — once measured — the paired training outcome and the configs it worked in.

The important change: **a cleared probe now automatically produces an executable
paired follow-up** whose treatment carries the exact stored source. The
incumbent is the control. No LLM step, and no retyping, between discovering a
feature and measuring whether it survives.

Guards a reviewer should check:

- The same mechanism proposed under a new name is recognised by source hash
  (comments and whitespace normalised away).
- Variation families are spawned **only** from a CONFIRMED feature. Generating
  follow-ups around a single-seed result is how a search burns a budget on noise.

## 5. Phase 5 — transparent allocation

`agent/allocator.py` scores experiment families on an explicit utility:

```
utility = expected_gain x P(success) x generality
          - runtime_cost - failure_cost - redundancy_penalty
```

Every term is printed. Success rates are Beta posterior means shrunk toward a
prior, so one attempt does not become a policy — the same standard this project
applies to its ML claims. Deliberately **not** reinforcement learning: 50
iterations against a 0.0008 noise floor will not support a learned value
function, and an unaccountable allocator is worse than a simple one here.

On a real journal it behaves sensibly, and says why:

```
family                        util   gain  p(ok)   gen    -rt  -fail  -redun
multi_seed_replication       0.872   1.20   0.92  1.00   0.20   0.03    0.00
...
path_b_discovery            -1.686   1.50   0.20  0.50   0.10   1.32    0.42

CHOICE: multi_seed_replication
  - nothing is CONFIRMED yet, so no result can be acted on until something is
```

Path B is correctly penalised at its *observed* 20% completion rate in that run.


---

## 6. Phase 2 — the verification run, and what it exposed

Run stopped by the operator at 6 nodes of 8 because it was re-demonstrating a
defect already diagnosed. Everything below is from `logs/opus_research/phase3_run.jsonl`.

| | |
|---|---|
| nodes / iterations consumed | 6 / 6 |
| experiments completed | 2 |
| experiments crashed | 4 |
| Path B attempts / crashes | 4 / 4 (**100%**) |
| orchestration-only misuse | **0** |
| **paired confirmation runs** | **1** (6 training runs, seeds 0-1-2) |
| single-seed exploratory runs | 5 |
| results promoted | **0** |
| automatic repair attempts / recovered | 2 / 0 |
| allocations recorded | 5 |
| manual interventions | **0** |
| inquiry complete / capability named+valid / promotion criterion | 5/5, 5/5, 5/5 |

### What is now verified BY A REAL RUN (not just unit tests)

1. **Confirmation executes.** Iteration 2 ran a paired `multi_seed_replication`:
   6 training runs across seeds [0,1,2], control vs treatment on identical
   seeds. This had never happened before in this project.
2. **Evidence-aware promotion holds.** The confirmation returned
   `UNCONFIRMED — t=2.25 over 3 seeds is below the 2.5 needed`, `promote=False`.
   A single lucky seed did not replace the incumbent.
3. **The allocator runs and steers.** 5 allocations recorded; it switched to
   `multi_seed_replication` as soon as a promising single-seed result existed,
   with the stated reason "nothing is CONFIRMED yet".
4. **The re-confirmation fix holds.** After node 0 came back UNCONFIRMED it was
   not re-queued, despite still being the highest single-seed score.
5. **`incumbent_cfg` is used correctly.** The agent wrote
   `cfg, enc = incumbent_cfg(...)` and `cfg['capture_epoch_scores'] = []`
   exactly as the contract instructs. Zero partial-config KeyErrors — the
   previous run's dominant failure is gone.

### The defect it exposed — mine, again

All 4 Path B crashes were the same class, and it is a **documentation defect in
my own capability contract**:

```
iter 1  ValueError: too many values to unpack (expected 2)
iter 3  ValueError: too many values to unpack (expected 2)
iter 4  RuntimeError: Unexpected prediction payload in capture_epoch_scores
iter 5  RuntimeError: Could not locate both valid/test score vectors in capture entry 0
```

The agent wrote `valid_scores, test_scores = train_lib.train_numpy_fm(...)`.
That function returns a **dict** (`scores_valid`, `scores_test`, `model`,
`hist`). And each `capture_epoch_scores` entry is
`(epoch, valid_primary, scores_valid)` — **valid only**, no test vector — while
my contract said only "(epoch, valid_primary, scores)", so the agent kept
looking for a test vector that is not there.

**The pattern across three evaluations is consistent and worth naming:** the
capability contract fixed *where a capability lives*; every failure since has
been *what exactly it accepts and returns*. Fixing one layer of the agent's
model of its own tools exposes the next.

### Honest status of the fixes for this

Precise output shapes and a preflight check for destructuring a dict-returning
capability are **identified and specified but NOT yet applied** — the run was
killed before I changed code, so that this run corresponds to exactly one code
state. They are the first thing to do next.

---

## 7. Where this leaves the score

**Unchanged: 0.60541.** Priority 1 of the brief did not move, and I am not going
to dress that up.

What changed is that the agent can now, for the first time, run the kind of
experiment capable of *establishing* an improvement — and in two live
confirmations it has correctly declined to promote anything (`+0.01 sigma` with
t=2.88 and 3/3 wins; `UNCONFIRMED` at t=2.25). Both are the machinery working.
Neither is a better score.

My honest read is that the benchmark is genuinely plateaued rather than
under-searched: the menu is exhausted, k=32 was already measured at +0.00011
(below threshold), heterogeneous ensembling is closed, and the model is
data-limited with no more data available. A reviewer should challenge that read
rather than take it.

---

## 8. What I would do next, in order

1. Apply the two contract fixes above (precise return shapes; preflight catches
   destructuring a dict-returning capability). This is the current dominant
   failure class.
2. Re-run the verification to see whether Path B crash rate finally drops.
3. Do the end-to-end incumbent rebuild (audit finding 7) — script is written at
   `logs/opus_research/rebuild_incumbent.py`, needs ~16 training runs.
4. Only then consider score work, and only via paired confirmation.

## 9. Commits

```
b4b9e03  Report confirmation, seed and repair metrics; start the review report
31714fc  Make confirmation executable, Path B cumulative, allocation transparent
824944b  Post-architecture evaluation: primary criterion failed, and why
```

Tests: **727 passing, 0 failing**. Incumbent verified at 0.60541 throughout.
