# Results — generated

Generated `2026-08-31T14:26:52Z` by `python3 -m agent.results_report`. Every figure below is read from repository artifacts at generation time; nothing here is retyped from memory.

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

## Harness

`python3 tests/test_harness.py` → **1114 passed, 0 failed** (26.7s) — **VERIFIED**

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
- hidden test evaluated: NO (never touched)
