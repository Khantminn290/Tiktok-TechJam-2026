# Handover — read this first

You're picking up an autonomous ML research agent competing on **KuaiRand-Pure**
(TikTok TechJam 2026, Track 2). This document is for a human starting cold. Ten
minutes here and you'll know where things stand and what to do next.

Branch: **`search-phase-complete`**. Not merged to `main` yet.

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
multitask:        none
model:            fm_numpy
temporal:         hour_plus_dow
training:         lower_lr_longer
neg_sampling:     uniform_1
sample_weighting: per_row
regularization:   l2_default
data_extras:      none
+ 16-seed rank-averaged ensemble
```

Note `multitask: none`. An earlier version of this recipe carried
`aux_click_like_forward` for a whole session on no isolated evidence; the
agent's own ablation removed it. That episode is why component evidence is now
graded against the noise floor (`agent/research_state.py: grade_evidence`).

**Score: `0.60541` validation primary** — a 16-seed rank-averaged ensemble of
ONE configuration. Individual members average `0.60463 ± 0.00032`.

**Delta over the official baseline (0.6016): `+0.00381` = `+4.76σ`**, where
σ = 0.0008 is the baseline's own 5-seed standard deviation. Comfortably outside
seed noise.

**It is reproducible, and that mattered.** A previously reported `0.60545`
turned out not to be: its member arrays had been archived away by `--fresh`
while the JSON quoting them stayed behind, and the pool mixed several distinct
configurations. Rebuild the current number end-to-end with:

```bash
python3 -m agent.final_ensemble --seeds 16
```

Members live in `logs/final_ensemble/seed_NN/`, and `--fresh` now refuses to
archive them (`run_agent.py: SUBMISSION_ARTIFACTS`, with a regression test).

**No selection bias.** `k=16` is *all* seeds trained, fixed before any score was
seen. The k-curve is emitted as diagnostics only and shows exactly why that
matters: it wanders 0.60491–0.60563 with no trend, so picking its argmax
(k=14, 0.60563) would be selection, not a result. Best-subset selection was
measured here at `+0.00081` of optimistic bias and rejected.

**Two numbers to keep in perspective:** the metric ceiling is 0.8484, not 1.0
(27% of users have zero positive labels, so nDCG can never reach 1 for them),
and eight model families (FM, DeepFM, DCN, DIN, GRU4Rec, ItemCF, GBDT,
item-popularity) all land in 0.55–0.605.

**Correction — do not repeat the old label-noise claim.** This project used to
say "20.6% of repeated (user, video) pairs disagree with themselves, so most of
the headroom is noise". The statistic is true; the conclusion does not follow.
Only **4.1%** of train pairs repeat at all, so the disagreeing ones are **1.72%
of rows**; in valid, 3.2% repeat and 10.2% of those disagree (1.09% of rows).
And only **0.177%** of within-user positive/negative pairs are feature-identical
and therefore unrankable by *any* model — a hard ceiling of **0.99911** on
unweighted pair-AUC against an observed **0.707**. There is **no measurement
establishing this task is noise-limited.** The evidence for a ceiling is
informational (see 1c), not statistical.

**Two numbers to keep in perspective:** the metric ceiling is 0.8484, not 1.0
(27% of users have zero positive labels, so nDCG can never reach 1 for them),
and the official baseline already captures ~31% of the attainable range. Judge
progress against the ceiling, not against 1.0.

---

## 1b. Does the agent's research policy actually work?

Measured, not asserted. `logs/ab_test/` holds both arms and the analysis.

**The audit finding was that Path B (writing custom code) was never *generated*
— not that it was rejected.** The planner made one call returning one proposal,
so there was no candidate set, no scoring, and no answer to "why not Path B?".
Multi-candidate planning (`--n-candidates 4`) fixes the generation problem:

| | A: single proposal | B: multi-candidate |
|---|---|---|
| Path B declared | 0 | 6 (3 genuine, 3 fake) |
| decision points recorded | 0 | 10 |
| candidates generated / gated | 0 / 0 | 40 / 18 |
| decisions auditable | no | yes |
| best primary | 0.60497 | 0.60367 |

**The strongest evidence needs no outcome at all.** Replaying the same 10
recorded decisions under a single-proposal policy (`python3 -m agent.policy_eval`):

- it would have picked a **gated** candidate — a duplicate, a recorded dead end,
  or an unfalsifiable proposal — **7 times out of 10**. The deployed policy
  picked 0.
- it agrees with the deployed choice only **10%** of the time.
- it opens 3 branches instead of 5, and picks Path B **0%** of the time —
  reproducing arm A's pathology exactly.

None of those rows depends on a training outcome, so none depends on luck.

**Two things this does NOT show, stated plainly:**

1. **It did not improve the score.** Arm B's best was 0.60367 vs arm A's 0.60497,
   and it spent 3 iterations on failed custom code. One run against one run,
   against seed noise of 0.0008, supports no causal claim in either direction.
2. **Only 3 of 6 Path B declarations are genuine.** The other 3 declare Path B
   and then call `train_lib.run()` — Path A wearing a Path B label
   (`python3 -m agent.capability_report` measures this from the scripts on
   disk). *Selecting* Path B is fixed; *writing* it is not.

The counterfactual harness refuses to score work that was never run: every
replayed decision is labelled OBSERVED / COUNTERFACTUAL_KNOWN /
COUNTERFACTUAL_UNKNOWN, and unknown outcomes are never imputed or averaged.

---

## 1c. Is the ceiling the menu, or the problem?

Tested directly rather than assumed (`python3 -m agent.residual_screen`).

`build_cache` reads **13 of 19** log columns, **1 of 12** video columns and
**0 of 31** user columns. `video_features_basic_pure.csv` — `tag` (46
categories, 99.2% coverage), `music_id`, `video_type`, `upload_type`,
`upload_dt`, server dimensions — is **not in the menu at all**: not an option,
not locked, simply absent. These are static item attributes and carry none of
the eval-window leakage that locked the *statistic* file. So real information
loss exists.

It is not exploitable. A gradient-boosted model given every causally-available
discarded attribute plus context reaches **0.64904** within-user AUC standalone
(incumbent: 0.67720). Blended into the incumbent at a weight **fixed in advance**
and resampled 5 times:

```
+0.00031 mean  (+0.39σ)  ± 0.00032   t = 2.14
```

A residual **is** statistically detectable and is still **under half the
seed-noise scale**. `tag` alone measures −0.09σ; user×tag affinity +0.04σ.

**This is the strongest evidence available that the ceiling here is
informational rather than an artefact of the 45-option menu** — stronger than
exhausting an option list, because it tests the information rather than the
menu. Note the honest limit: it bounds what *these* features carry, not what
some unrepresented interaction might.

Also measured while doing this: **per-user monotone score transforms are
structurally null.** Both metrics rank *within* a user, so `(s−mean)/std`,
per-user shifts and per-user shrinkage cannot change either metric. Two Path B
nodes did exactly this, returned **identical** metrics (0.60340 / 0.66981 /
0.53700), and were both recorded as successes.


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
3. ~~**Multi-config ensembling.**~~ **CLOSED — and the rule it rested on was
   wrong.** The standing lesson was "members must be independent AND comparably
   good", learned from three failures that each satisfied only one half
   (snapshot: quality, no independence; bagging: independence, no quality;
   gru4rec: independent but 2.1σ weaker). A **pre-registered** test finally ran
   the open case — `multitask=aux_click_like_forward`, quality gap **0.28σ**
   and cross-config correlation **0.9601 vs 0.9839** for same-config seeds, so
   both halves genuinely satisfied. All 16+16 seeds, equal weights, no sweep,
   no subset search, one validation comparison:

   ```
   base k=16   0.60541    combined k=32  0.60552   +0.00011 (+0.14σ)  REJECTED
   ```

   The corrected lesson: on this benchmark the residual error is **shared**, so
   decorrelating the *configuration* does not decorrelate the *errors* enough
   to matter. That closes the
   mechanism, not just one instance of it. (`agent/hetero_test.py`,
   `logs/hetero_test.json`.)

4. **The menu search space is essentially exhausted.** Of 45 axis-options, 23
   dead ends are recorded and the rest have been run. The only untried options
   are two locked data sources, `training=k32` (embedding capacity — an
   organiser-measured dead end), `multitask=censored_watch_time` (the
   watch-time-as-auxiliary mechanism is already a measured null, t=−0.23), and
   `temporal=hour_bucket` (a strict subset of the `hour_plus_dow` already in
   the best config). Further gains must come from **outside** the menu.
   `python3 -m agent.frontier` prints the current status of every option.

`video_features_statistic_pure.csv` is **locked** in the menu and should stay
locked: its counters span the evaluation window and risk target leakage.

---

## 4. How to run things

Full detail is in `README.md` — this is just the map.

```bash
python3 tests/test_harness.py                      # 468 checks, no LLM, no training
cd kuairand-starter-kit && python3 baseline.py --model fm && cd ..   # reproduce 0.6016
python3 -m agent.baseline_repro                    # durable baseline artifact

python3 run_agent.py --smoke                       # cheap plumbing check (~$0.02)
python3 run_agent.py --fresh --max-spend-usd 3     # a real run
python3 run_agent.py --reseed-top 3 --reseed-seeds 5   # multi-seed verification, no LLM
python3 run_agent.py --fresh --parallel-k 3        # parallel worker mode

python3 run_agent.py --fresh --n-candidates 4 --research-state --data-tools
                                                   # multi-candidate policy

python3 -m agent.report                            # readable run summary
python3 -m agent.final_summary                     # the competition deliverable
python3 -m agent.final_ensemble --seeds 16         # rebuild the submitted number
python3 -m agent.policy_eval                       # counterfactual decision replay
python3 -m agent.ab_report                         # A vs B research-process comparison
python3 -m agent.capability_report                 # genuine vs fake Path B, from disk
python3 -m agent.frontier                          # status of every axis-option
python3 -m agent.error_analysis                    # where the model fails, per segment
python3 -m agent.axis_sweep --axis X --values a,b  # controlled paired-seed comparison
python3 -m agent.hetero_test                       # the ensemble test above
python3 -m agent.residual_screen --confirm 5       # is there signal outside the menu?
python3 -m agent.make_submission --split valid --score --ensemble
```

**Two traps that cost real time here:**

- **Resuming into an already-converged journal silently does nothing.** It exits
  0 and reports the *pre-existing* iteration count. Use `--fresh` (it archives,
  never deletes). There's now a loud warning, but know the shape of it.
- **Don't run two training jobs at once, and don't run the test suite during a
  run either.** The executor chmods the real data directory for the duration of
  a subprocess; a concurrent reseed, A/B, *or* `tests/test_harness.py` (its data-
  boundary tests take the same lock) will die with `PermissionError` mid-run.
  This has bitten repeatedly, most recently killing an A/B arm at iteration 2.
  There's still no mutex — a lockfile guard is the single best small task for
  whoever picks this up. Permissions are restorable with
  `chmod -R u+rw kuairand-starter-kit/KuaiRand-Pure`.

**Heads-up on `logs/`:** `logs/journal.jsonl` holds whatever the most recent
*search* run produced, and `--fresh` archives it to `logs/archive_<ts>/` rather
than deleting it. The **submitted** result is deliberately separate and survives
`--fresh`: `logs/ensemble_results.json` plus `logs/final_ensemble/`. Don't
conflate the two — reading a search journal's best node as "the result" is how
the previous headline drifted from its evidence.

---

## 5. What's left before submission

1. ~~Final clean end-to-end run.~~ **DONE** — `logs/final_ensemble/` holds 16
   seeds of the section-1 config, and `logs/ensemble_results.json` records
   0.60541 with every member array on disk.
2. **Re-run `make_submission.py --ensemble`** against that ensemble — the
   current submission CSV is stale.
3. **The hidden-test evaluation — one shot, never yet used.**
   `python3 -m agent.make_submission --final-test-eval --ensemble`. A lockfile
   (`results/final_evaluation.lock`) enforces one-time use; a second attempt
   needs `--admin-override` and is logged. **Do not run this until the config is
   final.** Everything to date is validation-only.
4. **Devpost + README writeup.** The negative results are arguably the stronger
   story: a systematic, measured account of *why* this benchmark's headroom is
   narrow, backed by 15 documented dead-ends with mechanisms rather than
   assertions, and by the residual screen in 1c.

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
- **A number without its evidence on disk is not a result.** The previous
  headline had to be withdrawn because its member arrays were archived away.
  Anything quoted as the submitted score must be rebuildable by one command.
- **Grade evidence against the noise floor, not by sign.** A `>` comparison
  calls a 0.0001 difference "support" when σ = 0.0008. Use
  `research_state.grade_evidence`; INCONCLUSIVE and REJECTED are different
  answers, and only one of them invites another experiment.
- **Report negative results honestly.** Most of this project's value is in
  well-measured failures. A clean null with a mechanism beats a marginal,
  unverified bump.
