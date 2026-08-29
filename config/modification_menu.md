# Modification Menu — rationale

Companion to `modification_menu.json`. The agent's search space is a cross-product of the
axes below; every iteration picks one option per axis. Priorities and dead-ends come
straight from the organizers' measured results in `kuairand-starter-kit/README.md`
("从哪里开始改") — they are not guesses.

## Why these axes, in this order

The current menu also includes **score_prior** at priority 2: a train-only,
empirical-Bayes blend of smoothed video and author long-view logits. Its recency
variant exponentially downweights older training rows to address temporal drift.
It is model-agnostic, cheap, and uses no validation or test labels. The original
organizer-derived priorities below remain the provenance for the other axes.

1. **loss** — the baseline trains pointwise logloss but is scored on within-user ranking
   metrics (GAUC / nDCG@5). Aligning the objective with the metric (BPR pairwise, listwise
   softmax) is the organizers' #1 unexplored recommendation.
3. **user_history** — the baseline uses zero behavior-sequence information despite hundreds
   of train interactions per user (DIN/SIM territory). #2 recommendation.
   Constraint: purely user-side terms are constant within a user and provably cannot change
   within-user ranking — history only helps through interaction with the candidate item.
4. **multitask** — 12 feedback signals exist on *every* impression (`is_click`, `is_like`,
   `play_time_ms`, …), so auxiliary tasks have no sample-selection bias here. #3.
5. **model** — DeepFM / DCN. Deliberately ranked below the signal/objective improvements because the organizers measured
   that raw capacity (k = 8/16/32) does not move the score; a new architecture only pays off
   when it consumes new signal (e.g. DIN attention needs an MLP).
6. **temporal** — hour / day-of-week and train→eval drift. Same user-side caveat applies.
7. **training** — schedules that serve the loss axis (two-stage finetune); capacity knobs
   are kept only for combination experiments.
8. **data_extras** — extra files beyond the two standard logs. **Leakage-sensitive; see
   safety gate.**

## Measured dead-ends (do not respend iterations here)

| Tried by organizers | Result |
|---|---|
| All 13 CWM static feature fields + user-side coarse buckets | primary 0.5940 vs 0.5950 — no gain |
| FM embedding dim k = 8 / 16 / 32 | 0.5895 / 0.5902 / 0.5887 — no gain |
| Pure user-side first-order terms | exactly zero effect on within-user ranking |

## Safety gate (enforced in code, `agent/menu.py`)

Options flagged `"locked": true` are **mechanically unselectable**: `validate_choices()`
rejects them regardless of what the LLM proposes, unless a human has set
`"allow_locked_options": true` in `config/agent_config.json` (doing so should itself be
logged as a manual intervention).

- `data_extras.video_stats_features` — `video_features_statistic_pure.csv` counters are
  aggregated over a window overlapping the validation/test dates → target-leakage risk.
- `data_extras.random_log_test_period` — the test-window slice of the random-exposure log
  is direct test-period exposure.
The organizer-suggested validation-window random-exposure diagnostic is deliberately
not a generated-code menu option. It reads an additional outcome-bearing file, so it
belongs in a future trusted parent-side evaluator rather than the guarded training child.

## Cross-axis constraints (enforced by `validate_choices()`)

- `user_history = din_attention` requires `model ∈ {deepfm_mlp, dcn_lite}` — attention
  pooling needs an MLP to consume the pooled vector; plain FM has nowhere to put it.
- `multitask = censored_watch_time` requires `loss ∈ {pointwise_logloss,
  listwise_softmax_plus_pointwise}` — the censored-regression head combines with a
  calibrated pointwise head, not a pure pairwise/listwise objective.
