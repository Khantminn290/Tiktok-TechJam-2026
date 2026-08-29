# Codex independent research pass

This branch started from `main` and did not merge either teammate search branch.
All model selection below used the official validation split only. Hidden-test
outcomes were never evaluated.

## Adopted result: label-free repeat-fatigue reranking

The retained change is intentionally small. For each evaluation batch it counts
repeated `(true user_id, video_id)` exposures, standardizes the base score and
`log1p(repeat_count_excluding_self)` within each user, and applies one fixed
penalty:

```text
final_score = z_user(base_score) - 0.10 * z_user(log1p(repeats))
```

The hypothesis is repeated-exposure fatigue: showing the same video to the same
person several times is weak negative context not represented by the base FM.
This direction is consistent with research that explicitly models recommendation
fatigue, including [FRec (SIGIR 2024)](https://arxiv.org/abs/2405.11764).

The coefficient was frozen at `0.10`; it was not selected per seed. A critical
audit caught and fixed an early implementation that grouped unseen users through
the shared training-vocabulary `UNK` code. The production implementation uses
`user_raw` and `video_raw`, and the corrected result remained positive on every
paired seed:

| Seed | Incumbent | + repeat fatigue | Delta |
|---:|---:|---:|---:|
| 0 | 0.604967594 | 0.605060816 | +0.000093222 |
| 1 | 0.603929520 | 0.604097366 | +0.000167847 |
| 2 | 0.604490995 | 0.604678392 | +0.000187397 |
| 3 | 0.604237974 | 0.604375482 | +0.000137508 |
| 4 | 0.604786396 | 0.605009437 | +0.000223041 |

Mean paired gain is `+0.000161803` (sample SD `0.000049334`), with 5/5 wins;
the exact one-sided sign-test p-value is `0.03125`. With the final metric-aligned
per-user aggregation, the five-seed ensemble improves from `0.605452001` to
**`0.605660439`** (`+0.000208437`).
An end-to-end seed-0 replay through the hardened executor reproduces
`0.605060816`, with parent- and child-computed metrics matching exactly.
The complete clean five-seed replay also reproduces `0.605660439`; its atomic
bundle is selected by `results/verified_ensemble/latest.json`.

The ensemble converts each member to ranks **within each user**, matching the
official metric scope. When a member gives two impressions exactly equal scores,
the later `time_ms` ranks first; exact score-and-time ties receive neutral
midranks. An audit rejected the earlier global stable-rank result (`0.605617940`)
because its nominally stable ascending sort implicitly gave later row indices
higher downstream scores. The final rule uses an explicit input feature instead
of that accidental row-order effect.

This is transductive **input-batch** inference, not label leakage: it uses only
the row-aligned user/video inputs that must be scored, and test outcome columns
are physically absent from the runtime cache. It is disclosed because a live
online system would normally replace the full-batch count with past-only exposure
history; that safer online analogue was much weaker in this offline benchmark.

## Rejected directions

| Direction | Comparable result | Decision |
|---|---:|---|
| Date-OOF recency prior inside BPR | +0.000114, seed 0 | Below seed noise |
| Signed multi-behaviour FISM | -0.000045, seed 0 | Reject |
| Field-weighted FM | -0.000211, seed 0 | Reject |
| Scenario-cross FM | -0.002426, seed 0 | Reject |
| Cyclic snapshot ensemble | 0.6047 → 0.6044 → 0.6015 members | Stopped after deterioration |
| Personalized empirical-Bayes priors | all five candidates negative on ensemble | Reject |
| One-layer LightGCN residual | -0.000411 seed-0 blend; -0.000241 ensemble blend | Reject |

The disposable arrays and diagnostic scripts live under ignored `scratch/`.
`research/experiment_journal.jsonl` is the compact, reviewable record.

## Integrity changes made during the pass

- The supported API strips test outcome columns and migrates old caches to a
  feature-only test archive. A fresh cache does not extract test-date outcome
  cells into any target list or NumPy array.
- Raw `user_id` and `video_id` are preserved separately from train-vocabulary
  codes, so unseen entities never collapse into one false repeat identity.
- Generated training subprocesses do not inherit provider/API credentials.
- A Python audit-hook guard blocks ordinary generated-code access to raw data,
  `.env`, prior runs, subprocesses, and network sockets. This is defense in depth,
  not an OS security sandbox; use a container for hostile code.
- The parent captures validation labels before execution and recomputes every
  metric from `scores_valid.npy`; child-reported metrics cannot select a winner.
- Every completed run emits `verification.json` with code/config/artifact hashes
  and protected-file before/after hashes. Reuse additionally binds all three
  split caches, cache metadata/vocabularies, and the Python/NumPy/platform
  fingerprint.
- The publishable runner ignores caller path overrides, pins the canonical kit,
  data, and cache locations, verifies the official raw-file hashes, and requires
  cache schema v3 to record the same source digests.
- The final ensemble is published as one immutable bundle; an atomic
  `results/verified_ensemble/latest.json` pointer prevents mixed-version output
  files after an interrupted write.

Run `python -X utf8 tests/test_harness.py` for the fast contract suite. Use
`python -m agent.verified_ensemble --seeds 5` to rebuild the adopted ensemble;
it evaluates validation only and writes test predictions blind.
