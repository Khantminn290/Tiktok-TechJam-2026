# Handover — read this first

You're picking up an autonomous ML research agent competing on **KuaiRand-Pure**
(TikTok TechJam 2026, Track 2). This document is for a human starting cold. Ten
minutes here and you'll know where things stand and what to do next.

Branch: **`search-phase-complete`** (branched from
`hardening/deliverables-and-integrity`). Not merged to `main` yet.

---

## 1. Where things stand

**The search phase is finished and deliberately exhausted.** This is not a first
attempt — it's the end of a long campaign in which nine distinct interventions
were implemented, measured, and mostly ruled out. The number below survived
multi-seed verification; several earlier "wins" did not.

**Validated best configuration:**

```
loss:             bpr_pairwise
user_history:     recency_weighted_pool
multitask:        aux_click_like_forward
model:            fm_numpy
temporal:         hour_plus_dow
training:         default
neg_sampling:     uniform_1
sample_weighting: per_row
regularization:   l2_default
+ 5-seed rank-averaged ensemble
```

**Score: `0.60515 ± 0.00022` validation primary** — mean and std over all 252
possible 5-seed subsets of 10 trained seeds, not a single lucky run.

**Delta over the official baseline (0.6016): `+0.00355` = `+4.44σ`**, where
σ = 0.0008 is the baseline's own 5-seed standard deviation. Comfortably outside
seed noise.

A single (non-ensembled) model of this config scores **0.60367 ± 0.00027**, so
roughly a third of the total gain comes from ensembling and the rest from the
pointwise → BPR loss switch.

**Two numbers to keep in perspective:** the metric ceiling is 0.8484, not 1.0
(27% of users have zero positive labels, so nDCG can never reach 1 for them),
and the official baseline already captures ~31% of the attainable range. Judge
progress against the ceiling, not against 1.0.

---

## 2. What NOT to re-try

These are all recorded in `config/modification_menu.json` under
`notes.tested_dead_ends` (injected verbatim into every agent prompt), but they're
here in plain English so you don't burn a session rediscovering them. **Each was
measured on this dataset, not assumed.**

| Don't re-try | Why it lost |
|---|---|
| **LambdaRank / position-discounted loss** | −13.7σ. Training lists average 43.5 impressions/user but evaluation lists average 5.6 — an @5 discount zeroes 78.6% of training pairs and collapses gradient to 0.04× BPR's. |
| **Per-user gradient weighting** | Real harm, monotonic: −0.001 (sqrt) to −0.004 (full). The metric aggregates per-user, but the optimizer genuinely benefits from heavy users' extra impressions. |
| **Hard-negative sampling** | −0.032 (t = −27.7). Concentrates training on the model's current top errors. |
| **Popularity-biased negatives** | −0.037 (t = −118.7). Same failure mode, worse. |
| **More negatives per positive** (2×, 4×) | Null to slightly negative. Uniform 1-per-positive is already right. |
| **Snapshot / checkpoint ensembling** | Rejected by its own guard. Validation peaks at epoch 4 then declines, so top-N checkpoints are one peak plus worse neighbours — averaging dilutes. |
| **Bagged / bootstrap ensemble members** | −0.00084 vs plain seed ensembling. Bootstrap *did* decorrelate members (rank-corr 0.900 vs 0.959) but cost −0.0028 in per-member quality. |
| **Graded play-time targets** | Perfect null (t = −0.23). Adds nothing beyond binary `long_view` + click/like/forward. |
| **Neural architectures** (DeepFM / DCN / DIN) | Statistically tied with numpy FM (t = +0.57) *and* crashed on 4 of 8 generated iterations vs 0 of 8 for FM. |
| **L2 regularization** (1e-5, 1e-4) | True null, ±0.00013. |
| **Bigger embeddings** (k = 8/16/32) | Organizers already measured this; no gain. |
| **Exposure debiasing via the random log** | Withdrawn on analysis. long_view is 8.5% under random exposure vs 33.7% logged, but you're *scored on the logged distribution* — debiasing optimizes the wrong objective. The random log also only exists from 2022-04-22, i.e. the evaluation period. |

**The pattern behind almost all of these:** anything that **concentrates or
reweights** the training signal loses on this dataset, and any ensemble whose
members aren't **both independent and comparably good** loses. Broad, uniform,
unweighted signal keeps winning. Weigh a new idea against that before spending a
session on it.

---

## 3. What's genuinely still open

Not ruled out — never tested. These are the real remaining options:

1. **Content-side video features.** `video_features_basic_pure.csv` has `tag`
   (content category), `video_type`, `upload_type`, `music_id` — none currently
   used. Caveat: the organizers measured "adding static features" as a dead end,
   but that was on the weak pointwise baseline, and `tag` specifically is a
   content signal rather than a popularity counter.
2. **New engineered behavioural features.** The logs carry columns the cache
   never loads: `is_follow`, `is_comment`, `is_profile_enter` (2.5% of rows),
   `profile_stay_time`, `comment_stay_time`. Adding *breadth* of signal fits the
   pattern that keeps working here.
3. **Multi-config ensembling.** Every ensemble tested so far averaged members of
   the *same* config differing only by seed. Averaging genuinely *different*
   good configs (e.g. BPR+history vs BPR+multitask) could satisfy both ensemble
   properties at once — independent *and* comparably good.

`video_features_statistic_pure.csv` is **locked** in the menu and should stay
locked: its counters span the evaluation window and risk target leakage.

---

## 4. How to run things

Full detail is in `README.md` — this is just the map.

```bash
python3 tests/test_harness.py                      # 213 checks, no LLM, no training
cd kuairand-starter-kit && python3 baseline.py --model fm && cd ..   # reproduce 0.6016
python3 -m agent.baseline_repro                    # durable baseline artifact

python3 run_agent.py --smoke                       # cheap plumbing check (~$0.02)
python3 run_agent.py --fresh --max-spend-usd 3     # a real run
python3 run_agent.py --reseed-top 3 --reseed-seeds 5   # multi-seed verification, no LLM
python3 run_agent.py --fresh --parallel-k 3        # parallel worker mode

python3 -m agent.report                            # readable run summary
python3 -m agent.make_submission --split valid --score --ensemble
```

**Two traps that cost real time here:**

- **Resuming into an already-converged journal silently does nothing.** It exits
  0 and reports the *pre-existing* iteration count. Use `--fresh` (it archives,
  never deletes). There's now a loud warning, but know the shape of it.
- **Don't run two training jobs at once.** `--parallel-k` chmods the real data
  directory for the duration of a round; a concurrent reseed or A/B will die
  with `PermissionError`. There's no mutex yet — adding a lockfile guard is a
  good small task for whoever picks this up.

**Heads-up on `logs/`:** it currently holds the *neural exploration* run (8
nodes, all DeepFM/DCN) — itself a documented dead end, not the winning config.
The winning config's evidence is in `logs/experiments/` and section 1 above.
Reproducing the headline number needs a fresh run with that config.

---

## 5. What's left before submission

1. **Final clean end-to-end run** with the exact config from section 1, so
   `logs/` reflects the winning configuration rather than the neural detour.
2. **Re-run `make_submission.py --ensemble`** against that fresh journal — the
   current submission CSV is stale.
3. **The hidden-test evaluation — one shot, never yet used.**
   `python3 -m agent.make_submission --final-test-eval --ensemble`. A lockfile
   (`results/final_evaluation.lock`) enforces one-time use; a second attempt
   needs `--admin-override` and is logged. **Do not run this until the config is
   final.** Everything to date is validation-only.
4. **Devpost + README writeup.** The negative results are arguably the stronger
   story: a systematic, measured account of *why* this benchmark's headroom is
   narrow, backed by 11 documented dead-ends with mechanisms rather than
   assertions.

---

## 6. House rules

- **Review every change.** Read diffs before committing (`CLAUDE.md` convention).
- **Never trust a single-seed result.** This bit us three times: a "best" node
  at 0.6035 was seed-lucky and a *different* node was genuinely better. Anything
  claimed as a win gets `--reseed-top` with ≥5 seeds first. Quote gains in σ
  (σ = 0.0008), not raw deltas.
- **Never touch `kuairand-starter-kit/evaluate.py`.** It's the scoring ground
  truth. Score through it; never reimplement a metric.
- **Never train on test.** Generated code runs in a sandbox where the real data
  directory is unreadable and the test split has its label columns physically
  stripped. Don't work around it.
- **Report negative results honestly.** Most of this project's value is in
  well-measured failures. A clean null with a mechanism beats a marginal,
  unverified bump.
