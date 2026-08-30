# Results — generated

Generated `2026-08-30T16:18:02Z` by `python3 -m agent.results_report`. Every figure below is read from repository artifacts at generation time; nothing here is retyped from memory.

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
Provenance: commit `024238f2b9ec` on `opus-research-agent`, data fingerprint `60fa8bc44d1e3d59`.

## Harness

`python3 tests/test_harness.py` → **800 passed, 0 failed** (8.9s) — **VERIFIED**

## Convergence rule

ε = **0.000477** (0.6σ at a noise floor of 0.0008), N = 3 — **VERIFIED**
> calibrated to the upward drift a running maximum shows by luck over N iterations, not hand-picked

## Latest run

| | |
|---|---|
| best primary (single run) | 0.60527 |
| outer-loop nodes | 14 |
| iterations consumed | 12 |
| training runs used | 27 of 90 |
| experiments completed | 9 |
| experiments crashed | 3 |
| Path B attempts / crashes | 7 / 5 |
| preflight rejections (free) | 2 |
| automatic repairs attempted / recovered | 2 / 0 |
| paired confirmations run | 3 |
| results promoted | 0 |
| **manual interventions** | **0** |
| LLM tokens | 493,573 |
| LLM spend | $1.71402 |
| training wall-clock | 2033.2s |
| devices | cpu |

Stop reason: iteration cap reached (12)

> An outer-loop node is one decision; a training execution is one model actually trained. A paired 3-seed confirmation is 1 node and 6 training executions. A preflight rejection is neither: no compute was spent and no decision was consumed, though repeated rejections are capped.

## Path B (feature discovery)

- features probed: 8 {'REJECTED': 6, 'PROBED': 2}
- stored with full lineage: 0
- retrained in a paired experiment: 0
- **end-to-end complete: NO** — OBSERVED
> end-to-end means a proposed feature cleared the probe AND was retrained from its stored source in a paired experiment

## Confirmations

- paired experiments run: 3
- outcomes: {'REJECTED': 3}
- **promoted: 0**
> only CONFIRMED may change the submitted system; a single-seed result is PRELIMINARY at any effect size

## Budget counting

> An outer-loop node is one decision; a training execution is one model actually trained. A paired 3-seed confirmation is 1 node and 6 training executions. A preflight rejection is neither: no compute was spent and no decision was consumed, though repeated rejections are capped.

## Scope and integrity

- dataset: KuaiRand-Pure only
- hidden test evaluated: NO (never touched)
