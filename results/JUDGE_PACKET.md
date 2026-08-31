# Autonomous ML Research Agent — KuaiRand-Pure

*Generated from `results/manifest.json` on 2026-08-31T06:51:01Z at commit `4b4282c1434b` (opus-research-agent).*

Every number in this document is read from that manifest, which is generated from artifacts on disk. Nothing is transcribed by hand.

---

## 1. The problem

Build an autonomous agent that improves a recommender pipeline on **KuaiRand-Pure** without a human in the loop. The label is `long_view`; the score is the mean of GAUC and nDCG@5.

- dataset scope: Pure only (fingerprint `60fa8bc44d1e3d59`)
- rows: train 1,141,112, valid 124,909, test 170,588
- official baseline, validation: **0.60160** (GAUC 0.6674, nDCG@5 0.5357)
- official baseline, hidden test: 0.59460
- seed noise floor, measured: sigma = 0.0008. This is the number that decides what counts as a result here: differences smaller than about 1 sigma are indistinguishable from which seed was drawn.

## 2. The research loop

One iteration is one decision. The agent:

1. **reads its own history** -- the journal, the research memory, the frontier of what is tried, what failed, and what is still open;
2. **states a hypothesis and competing alternatives**, plus the measurement that would distinguish them;
3. **proposes candidates** and scores them against that frontier before any compute is spent;
4. **writes the experiment** as a script, or selects a configuration from the modification menu;
5. **passes preflight** -- eight static stages, cheapest first (syntax, imports, capability, call arity, return shape, config, leakage, import smoke). A rejection here costs no training run;
6. **executes** in a sandbox that cannot read the evaluation labels;
7. **classifies the outcome** -- a crash is a code failure, a poor score is a RESULT, and the two are never conflated;
8. **grades the evidence** and records what it now believes.

The distinction in step 7 is load-bearing. Retrying a hypothesis that was correctly measured and found wanting is not persistence, it is waste; and repairing a crash does not make the untested hypothesis any more or less likely.

## 3. The unified action space

The agent has one action space, not two. Every move below is a first-class `ExperimentSpec` the loop can schedule, price in training runs, and journal:

| action | what it does | cost |
|---|---|---|
| `draft` | a fresh configuration or script | 1 training run |
| `improve` | extend the current best | 1 |
| `debug` | repair a crashed node from its trace | 1 |
| `crossover` | combine two lineages | 1 |
| `confirm` | paired multi-seed replication | 2n |
| `ensemble` | average k seeds of one configuration | k |

Path A (menu configuration) and Path B (agent-written code) are two ways of expressing an experiment, not two pipelines. Ensembling is on this list deliberately: it is the single largest measured gain available (about 1 sigma) and for most of this project's history it sat outside the agent's reach, performed by a human afterwards.

## 4. How the agent chooses experiments

Not by picking the highest-scoring untried option. The frontier tracks, per axis and per option, what has been measured, what crashed, and what is a known dead end -- and candidates are scored on expected information, not expected score.

Two guards matter enough to name:

- **selection pressure is priced in.** Comparing 40 candidates and keeping the best of them produces an inflated number by construction. The evidence layer knows the expected maximum of n draws at this noise floor and discounts accordingly.
- **the shipped configuration is never condemned by lexical match.** A dead-end note that merely *mentions* an option used to mark it known-bad. That bug hid the highest-scoring temporal setting from the agent's own frontier; fixing it moved a first draft from 0.59805 to 0.60493.

## 5. How confirmation works

A single seed is one draw. At sigma = 0.0008, a 3-sigma-looking single-seed result is routine luck, so **no single-seed measurement can change what gets submitted**, at any effect size.

Evidence states, in order: `UNTESTED`, `HYPOTHESIS`, `PROBED`, `PRELIMINARY`, `UNCONFIRMED`, `CONFIRMED`, `REJECTED`, `REDUNDANT`. Only `CONFIRMED` is actionable.

Confirmation is a **paired** multi-seed experiment: the same seeds in both arms, so the seed draw cancels instead of being averaged over. The number of seeds required is computed from the effect size and the measured noise, not fixed in advance.

This is not decoration. In the agent's own reproduction run, a configuration that scored 0.60497 on one seed was put through a paired confirmation and **rejected as a lucky draw** -- and it was still the right configuration to ensemble.

## 6. How ensembling works

Averaging k seeds does not make any model better. It removes the seed variance from the thing you submit. So the effect is measured against the **mean member**, never the best one -- the best of k draws beats the mean by construction, and comparing against it would report a gain even if ensembling did nothing at all.

- aggregation: **rank_normalise_then_mean**. Both metrics read only the *order* of scores, so averaging raw values would let whichever member has the widest spread dominate.
- k = 16, **fixed before any score was seen**. All seeds trained were kept; no subset was searched. The recorded k-curve is diagnostic only -- best-subset selection was measured to carry +0.00081 of optimistic bias, which is larger than the effect being claimed.
- members already on disk are **reused, not retrained**. Reuse is real historical evidence and costs no compute, and an observation is counted once, keyed by (configuration, seed), so re-requesting the same member cannot inflate the support behind a claim.

Steps, as recorded in the manifest:

1. one configuration was selected (see `config`)
2. 16 independent seeds were trained, all of them kept -- no member was chosen on validation
3. each member's valid predictions were rank-normalised
4. the normalised predictions were averaged
5. the average was scored with the starter-kit evaluator

## 7. Results

All scores are **validation**. The hidden test has not been evaluated; see section 11.

| result | primary | GAUC | nDCG@5 | vs baseline | evidence |
|---|---|---|---|---|---|
| official baseline | 0.60160 | 0.6674 | 0.5357 | — | given |
| agent-discovered single model, seed 0 | 0.60497 | — | — | +0.00337 | PRELIMINARY (one draw) |
| **submitted 16-seed ensemble** | **0.60541** | 0.67212 | 0.53870 | +0.00381 (4.76 sigma) | CONFIRMED |

The ensemble beats the **mean** of its own members by +0.00078 (members: 0.60463 +/- 0.00032), which is about 1 sigma. That is the honest size of what ensembling bought.

### Verification

- recomputed from the stored member predictions at packet-generation time: **exact match** (0.60541 vs reported 0.60541)
- member predictions on disk: 16 directories, listed in the manifest
- issues: none

### Who did what

This is the part most easily overstated, so it is stated plainly.

- The **configuration** was discovered by an agent run.
- The **submitted artifact** was originally produced by a human-invoked command (`agent.final_ensemble --seeds 16`) — a human typed that command, after the agent had stopped.
- The agent has **since reproduced the whole pipeline unaided**: from a cold `--fresh` start it found the same configuration, ran its own paired confirmation, queued its own 16-member ensemble, and produced 0.60541 with a byte-identical config. Evidence: `logs/opus_research/agent_reproduced_incumbent.jsonl`.
- It **matched** that number. It did not beat it, and no new improvement is claimed.

## 8. What the run cost

| | |
|---|---|
| experiments (outer-loop decisions) | 8 |
| training runs spent | 28 of 150 |
| fresh executions | 28 |
| reused artifacts | 0 (no compute) |
| duplicate-reuse attempts | 0 (no compute, no evidence) |
| unique observations | 8 |
| confirmations run | 1 |
| candidates rejected by confirmation | 1 |
| crashes | 0 |
| preflight rejections (free) | 0 |
| automatic recoveries | 0 |
| **manual interventions** | **0** |
| training wall-clock | 2321.5 s |
| total agent wall-clock | 2545.4 s |
| LLM tokens | 269,807 (in 250,713, out 19,094) |
| LLM spend | $0.9132 |
| devices | cpu |

> derived from scored nodes: this journal predates execution-event instrumentation

Stop reason: `converged: running-best valid primary improved only 0.00000 (≤ ε=0.00048, the 0.60σ upward drift a running max shows by luck alone) over the last 3 scored iterations`

## 9. Robustness

**20 faults injected**, each checked on ten axes: was it detected, named correctly, routed correctly (repair / retry / skip / pivot / abort), bounded so it cannot repeat forever, charged correctly to the compute budget, kept out of the evidence, journalled, survivable, free of human intervention, and -- when recovery is impossible -- terminated cleanly.

- detection rate: **100%**
- recovery rate: **100%**
- automatic repairs 9, skips 6, pivots 3, clean terminations 2
- failed retries: 0
- manual interventions: 0
- invalid candidate promoted at any point: **False**

Reproduce: `python3 -m agent.faults --live`

### The live run

Unit tests establish that components behave when handed a constructed input. This is the whole loop: real LLM, real training, a deliberate failure injected at iteration 1.

- the injected fault was detected: **True**, after 143s of training that was correctly charged as spent
- the agent's next move: **debug** — *last attempt (node 1) errored; debug chain length 0 < 2, feeding the error trace back to fix it*
- **2 unplanned faults also occurred.** the network dropped mid-run and two LLM calls failed. Nothing about this was staged; both were journalled, the agent correctly declined to debug a node that had produced no code, and the run continued to its cap.
- ledger afterwards: 2 training runs charged, 1 crashed, 1 observation credited — a crash costs compute and earns no evidence, and the books say both
- stopped on `iteration cap reached (4)`, not on a crash
- manual interventions: **0**

Artifacts: `results/live_fault_run/`  
Reproduce: `python3 run_agent.py --fresh --max-iterations 4 --inject-error-at 1 --max-training-runs 4`

Test harness: **1077 passed, 0 failed** (21.9s).

## 10. Convergence

Two rules, kept separate, because conflating them would be a compliance problem rather than a stylistic one.

**Official (organizers').** `epsilon=0.002, N=3` — converged when the validation primary has not improved by more than epsilon over the last N consecutive scored iterations, or at the 50-iteration cap, or the 6.0h ceiling, whichever comes first.

- scored iterations: 8
- converged: **yes**, first at node 3 (gain 0.001672 over the window)
- best validation primary: 0.60541

**Internal research controller.** `epsilon=0.00048, N=3` (0.6 sigma) — *not* the official rule. It is calibrated to the upward drift a running maximum shows by luck at this noise floor, and it is stricter, so the search keeps going past the point the organizers' rule would end it.

> The organizer rule is the official definition of convergence AND of which checkpoint is scored: the validation-best checkpoint at the point it fires. The internal controller is stricter, so the loop keeps searching past that point -- which is useful for research and does NOT make a later artifact eligible. Check official.converged_at_node before treating any result as the submission.

### Eligible checkpoint — an open compliance risk

The organizers' rule fixes *what is scored*, not just when to stop: the validation-best checkpoint **at the point it fires**. On the recorded journal it fires at **node 3**, where the validation-best checkpoint is **0.60497** (node 1).

Later nodes score higher, but were produced after that point:

- node 4 — 0.60541 (`ensemble`)
- node 6 — 0.60509 (`improve`)

**We are not claiming these are eligible for this journal.** A stricter internal epsilon keeps a search alive; it does not make a later artifact scoreable. The fix is a clean run under the organizers' rule that reaches the ensemble before the first no-progress window can close — not a reinterpretation of this one after the fact.

## 11. Limitations

Stated because they are true, not because they are small.

1. **The submitted ensemble may not be an eligible checkpoint for the recorded journal.** The organizers' rule fires at node 3 and the ensemble is later than that (section 10). This is a compliance gap, not a scoring one, and it is fixed by a clean run under the organizers' rule — not by reinterpreting this journal.
2. **The hidden test has not been evaluated** (`evaluated: False`). Every number in this packet is validation. The gap between validation and test on the official baseline is -0.0070, and there is no reason to expect this submission to be exempt from a gap of that order.
3. **The agent matched the incumbent; it has not beaten it.** From a cold start it reproduced 0.60541 unaided. No result in this repository exceeds it.
4. **The submitted artifact was originally human-invoked.** The reproduction is real and journalled, but the file that will be submitted was built by a person typing a command.
5. **One configuration family.** The ensemble is 16 seeds of a single configuration, not a diverse ensemble. Diversity across configurations is untested and is the most obvious place left to look.
6. **The effect being claimed is close to the noise floor.** +0.00078 over the mean member is about 1 sigma. It is real and reproducible by re-aggregating the stored predictions, but it is not large.
7. **Autonomy is Level B, not Level A.** The agent transfers capabilities and writes its own experiments, but the capability contract and the modification menu are human-authored; a new axis requires human approval before it becomes live.
8. **The search is short.** The recorded run is 8 outer iterations. The organizers allow 50.

## 12. Exact reproduction commands

```bash
# verify the submitted number recomputes from stored predictions
python3 -m agent.verify_incumbent

# run the full test harness
python3 tests/test_harness.py

# run the fault-injection suite, including live faults
python3 -m agent.faults --live

# reproduce the live injected-failure run
python3 run_agent.py --fresh --max-iterations 4 --inject-error-at 1 --max-training-runs 4

# rebuild the 16-member ensemble from scratch
python3 -m agent.final_ensemble --seeds 16

# regenerate the manifest every number here comes from
python3 -m agent.manifest --run-tests

# regenerate this packet
python3 -m agent.judge_packet

# score a validation submission
python3 -m agent.make_submission --split valid --score --ensemble

# watch the agent work
streamlit run app.py

# run the agent
python3 run_agent.py --competition --fresh --wall-clock-limit-h 2.0

```

The hidden test is evaluated **once**, at final submission:

```bash
python3 -m agent.make_submission --final-test-eval --ensemble
```

A lock file records that it has happened (`exactly one evaluation, at final submission only`).
