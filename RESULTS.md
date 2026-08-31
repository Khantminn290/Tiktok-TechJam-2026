# Results — generated

Generated `2026-08-31T15:13:23Z` by `python3 -m agent.results_report`. Every figure below is read from repository artifacts at generation time; nothing here is retyped from memory.

Tiers: **VERIFIED** recomputed during this generation · **OBSERVED** measured in a run and journalled · **OPEN** not established.

## Metric (authoritative)

- source: `kuairand-starter-kit/evaluate.py` — **VERIFIED**
- positive label: `long_view`
- metrics: GAUC, nDCG@5; primary = mean(GAUC, nDCG@5)

> The evaluator is authoritative. Where the brief's prose and the starter kit disagree, the kit scores the submission, so the kit wins. Benchmark code is not modified.

## Incumbent

**0.60541** primary (GAUC 0.67212, nDCG@5 0.5387), 16-member ensemble — **VERIFIED**

Recomputed from stored predictions during this generation using `rank_normalise_then_mean`: primary 0.60541, GAUC 0.67212, nDCG@5 0.5387 — exact match.
Reproduce: `python3 -m agent.final_ensemble --seeds 16`
Provenance: commit `6b1aa7fc6399` on `opus-research-agent`, data fingerprint `60fa8bc44d1e3d59`.

## Hidden test — scored once

| split | primary | GAUC | nDCG@5 |
|---|---|---|---|
| official baseline | 0.5946 | 0.661 | 0.5282 |
| **this submission** | **0.59810** | **0.66510** | **0.53110** |
| **absolute delta** | **+0.0035** | **+0.0041** | **+0.0029** |

Judged score = mean absolute delta over GAUC and nDCG@5 = **+0.0035** (4.37σ on the baseline's own seed noise). — **OBSERVED**

Validation-to-test drop: 0.60541 → 0.59810. The official baseline loses 0.0070 across the same two splits, so this is the expected generalisation gap, not a further edge.
Source: `results/final_results.json`, written by the evaluation itself. One-shot: `results/final_evaluation.lock` is present.

## Official baseline — reproduced here

`/Library/Frameworks/Python.framework/Versions/3.12/bin/python3 baseline.py --model fm --seed 0` (seed 0) — **VERIFIED**

- validation: 0.6015 (GAUC 0.6671, nDCG@5 0.5358)
- hidden test: 0.5953 (GAUC 0.6621, nDCG@5 0.5286)

> Reproduced from the unmodified starter kit. SHA256 of `baseline.py`, `data.py` and `evaluate.py` are recorded in `logs/baseline/metrics.json` so a judge can confirm the benchmark code was not edited. Single seed, so it sits within seed noise of the organizers' published 5-seed means (0.6016 valid / 0.5946 test), which remain the comparators used above.

## Harness

**OPEN** — not executed during this generation; pass --run-tests for a live count

## Convergence

**Official (organizer) rule** — `epsilon=0.002, N=3`. Converged: **YES**, first at node 8. Best validation primary 0.60541. Hard caps: 50 iterations, 6.0h. — **VERIFIED**

**Internal research controller** — `epsilon=0.00048, N=3` (0.6σ), stricter and NOT the official rule. Converged: YES.

> The organizer rule is the official definition of convergence AND of which checkpoint is scored: the validation-best checkpoint at the point it fires. The internal controller is stricter, so the loop keeps searching past that point -- which is useful for research and does NOT make a later artifact eligible. Check official.converged_at_node before treating any result as the submission.

## Latest run

| | |
|---|---|
| best primary (single run) | 0.60497 |
| outer-loop nodes | 9 |
| iterations consumed | 8 |
| training runs used | 27 of 90 |
| experiments completed | 5 |
| experiments crashed | 3 |
| Path B attempts / crashes | 3 / 2 |
| preflight rejections (free) | 1 |
| automatic repairs attempted / recovered | 2 / 0 |
| paired confirmations run | 2 |
| results promoted | 1 |
| **manual interventions** | **0** |
| LLM tokens | 203,602 |
| LLM spend | $0.718218 |
| training wall-clock | 1691.3s |
| devices | cpu |

Stop reason: converged: running-best valid primary improved only 0.00044 (<= epsilon=0.00200, organizer rule) over the last 3 scored iterations

> An outer-loop node is one decision; a training execution is one model actually trained. A paired 3-seed confirmation is 1 node and 6 training executions. A preflight rejection is neither: no compute was spent and no decision was consumed, though repeated rejections are capped.

## Resource usage (Feasibility & Practicality)

| measure | value |
|---|---|
| LLM tokens, input + output | 203,602 |
| agent wall-clock | 42.8 min (2,567s) |
| iterations used | 8 of 50 (cap) |
| GPU-hours | 0.0 (cpu only) |

Token figure is the provider ledger (`provider_final_summary`) — every call the agent made, including planning calls and calls spent on iterations that errored before scoring. It is not the sum of per-node attributions, which undercounts.

> This is the agent's own inference cost. It excludes tokens spent by human-driven development sessions that authored the harness, which are not instrumented and are far larger.

## Path B (feature discovery)

- features probed: 8 {'REJECTED': 6, 'PROBED': 2}
- stored with full lineage: 0
- retrained in a paired experiment: 0
- **end-to-end complete: NO** — OBSERVED
> end-to-end means a proposed feature cleared the probe AND was retrained from its stored source in a paired experiment

## Confirmations

- paired experiments run: 2
- outcomes: {'CONFIRMED': 2}
- **promoted: 2**
> only CONFIRMED may change the submitted system; a single-seed result is PRELIMINARY at any effect size

## Budget counting

> An outer-loop node is one decision; a training execution is one model actually trained. A paired 3-seed confirmation is 1 node and 6 training executions. A preflight rejection is neither: no compute was spent and no decision was consumed, though repeated rejections are capped.

## Scope and integrity

- dataset: KuaiRand-Pure only
- hidden test evaluated: YES
