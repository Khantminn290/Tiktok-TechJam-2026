# Handover — read this first

You're picking up an **autonomous ML research agent** competing on
**KuaiRand-Pure** (TikTok TechJam 2026, Track 2). Fifteen minutes here and you'll
know where things stand, what's already been ruled out, and where to spend your
next hour.

---

## 0. TL;DR for a teammate in a hurry

```bash
python3 tests/test_harness.py          # 523 checks, no LLM, no training, ~2 min
python3 -m agent.final_summary         # the competition deliverable
python3 -m agent.frontier              # status of every option we've tried
python3 -m agent.feature_lab           # the agent's own feature research log
```

**Result: `0.60541` validation primary** (GAUC 0.67212 / nDCG@5 0.53870),
**+0.00381 over the 0.6016 baseline = +4.76σ**. Reproduce it end-to-end with:

```bash
python3 -m agent.final_ensemble --seeds 16
```

**The hidden test has never been evaluated.** That is a one-shot action; see §6.

**Before you spend a day on an idea, run `python3 -m agent.frontier` and read §2.**
Thirty-plus directions have been measured and ruled out *with mechanisms*. Most
"obvious" ideas here are already dead, and re-running them is the main way to
waste time on this project.

---

## 1. Where things stand

**Validated best configuration:**

```
loss:             bpr_pairwise          neg_sampling:     uniform_1
user_history:     recency_weighted_pool sample_weighting: per_row
multitask:        none                  regularization:   l2_default
model:            fm_numpy              data_extras:      none
temporal:         hour_plus_dow         training:         lower_lr_longer
+ 16-seed rank-averaged ensemble
```

Individual members average `0.60463 ± 0.00032`; the ensemble is `0.60541`.

**No selection bias.** `k=16` is *all* seeds trained, fixed before any score was
seen. The k-curve is emitted as diagnostics only and wanders 0.60491–0.60563 with
no trend — picking its argmax (k=14) would be selection, not a result.
Best-subset selection was measured here at **+0.00081** of optimistic bias and is
rejected.

**Two numbers for perspective.** The metric ceiling is **0.8484**, not 1.0 — 27%
of users have no positive label, so nDCG can never reach 1 for them. And eight
model families (FM, DeepFM, DCN, DIN, GRU4Rec, ItemCF, GBDT, item-popularity) all
land in 0.55–0.605. Judge progress against 0.8484.

**Do not claim the task is label-noise-limited.** An earlier version of this doc
did, from "20.6% of repeated (user,video) pairs disagree". Only 4.1% of train
pairs repeat, so that is 1.72% of rows, and just **0.177%** of within-user
positive/negative pairs are feature-identical (hard ceiling 0.99911 against an
observed 0.707). The claim was never supported by that evidence.

---

## 2. What NOT to re-try

All recorded in `config/modification_menu.json` under `notes.tested_dead_ends`
(injected into every planning prompt) — **28 entries**. The ones that cost the
most time to learn:

| Direction | Evidence |
|---|---|
| Any other model family | DeepFM/DCN/DIN tie with FM; torch path crashed 4/8 runs |
| Bigger embeddings | k=8 **−0.03σ**, k=32 **−0.18σ** (measured on *this* config) |
| L2 sweep | 1e-5 and 1e-4 both null |
| Negative-sampling variants | uniform_2/4, hard, popularity-biased — all lose |
| Listwise / LambdaRank losses | decisively worse (LambdaRank −13.7σ) |
| Per-user gradient weighting | real harm |
| Auxiliary feedback heads | is_follow/comment/hate/profile_enter: −0.53σ, 0/5 wins |
| **Dense item features** | item long_view rate scores 0.639 *standalone* yet **hurts at every blend weight** — the embedding already encodes it |
| **User-level features** | **structurally** worth exactly 0.5 AUC: GAUC ranks *within* a user, so anything constant per user contributes nothing |
| user×author / user×video features | only **3.4%** / **1.6%** of valid rows have that pair in train → 6× train/serve skew → **−16.5σ** |
| Heterogeneous ensembling | closed: members were both decorrelated *and* equal-quality (0.28σ gap, corr 0.9601) and it still gave **+0.14σ** |
| Checkpoint averaging | real (**+0.87σ**) for a *single* model, **redundant at k=16** (−0.01σ) |

**The unifying mechanism, worth internalising:** every within-user-varying
observable is *already an FM field* (video, author, tab, duration bucket, hour,
dow). A constructed feature is therefore a re-parameterisation of information the
model already has. The sharpest evidence: a 3-way `dur×hour×tab` rate scores
**0.6129 standalone** — close to the model's own 0.6772 — and adds **0.07σ**.

---

## 3. What's genuinely still open

1. **KuaiRand-1K / 27K.** Explicitly permitted training data ("training must rely
   only on the KuaiRand datasets listed below" — all three are listed). 27K has
   the same ~27k users with the standard exposure logs Pure omits: **322M rows vs
   1.4M** (9.9 GB). The learning curve says the model is **data-limited** —
   +0.00410 (**+5.12σ**) at the last doubling, decaying only ~0.8× per doubling.
   This is the largest untapped lever by a wide margin. Filter to dates
   ≤ 20220421 so the hidden test stays untouched, and pre-train then fine-tune
   rather than training on the mixed distribution (27K's extra impressions are
   algorithmically, not randomly, exposed).
2. **Train on train+valid for the final test model.** Legitimate and standard;
   valid is also temporally closer to test. Cannot be measured on validation, so
   it is a judgement call.
3. **The bonus benchmarks** (1K/27K) earn extra credit and may go unattempted by
   most teams. `KUAIRAND_DATA` / `KUAIRAND_CACHE` are already env-overridable and
   the split dates are shared, so it is mostly filename handling.

`video_features_statistic_pure.csv` stays **locked** — its counters span the
evaluation window.

---

## 4. The agent, in one page

```
research state + frontier + feature registry   (what we know)
        ↓
research policy picks an objective             (explore / ablate / confirm / …)
        ↓
K candidates in ONE LLM call, scored deterministically
   gain × P(success) × novelty × redundancy × information-value ÷ cost
        ↓
gates BEFORE execution: leakage, mechanism audit, duplicates, dead ends
        ↓
sandboxed training run  →  metrics  →  journal  →  memory
```

**Key modules** (all carry module docstrings explaining *why* they exist):

| Module | Purpose |
|---|---|
| `loop.py` | the iteration loop; every phase hangs off it |
| `frontier.py` | status of every axis-option, aggregated across **all** runs |
| `research_state.py` / `research_policy.py` | derived facts; objective selection |
| `candidates.py` | multi-candidate generation + deterministic scoring |
| `feature_lab.py` | autonomous feature research (probe → registry) |
| `pipeline_lab.py` | capabilities the menu can't express (see §5) |
| `error_analysis.py` | per-segment failure analysis, residual feature probing |
| `leakage_check.py` / `mechanism_audit.py` | pre-execution gates |
| `final_ensemble.py` | builds the **submitted** result reproducibly |

---

## 5. Capabilities beyond the menu

The menu is **prior knowledge, not the boundary**. The agent can also:

- **propose and implement its own features** — it writes a `build_features()`
  builder, which is probed for leakage, within-user variation, redundancy and
  *incremental* value before any training run. Rejections are remembered in
  `logs/feature_registry.jsonl` so nothing is re-proposed.
- **set pipeline overrides directly** in `menu_choices`: `k`, `lr`, `epochs`,
  `patience`, `l2`, `bs`, `hist_tau_days`, `aux_weight`, `snapshot_ensemble`,
  `snapshot_force`. Range-checked at the menu boundary.
- **diagnose training dynamics** (`pipeline_lab.training_dynamics`) — this is how
  we found the model peaks at epoch 14 and decays **−29.6σ** by epoch 60, which
  redirected a whole research phase.

**It demonstrably uses them.** In the self-test it independently proposed
checkpoint averaging, predicted +0.0005–0.0008, and measured **+0.00078**
(`logs/opus_research/selftest_run.log`).

---

## 6. How to run things

```bash
# free — no LLM, no training
python3 tests/test_harness.py                      # 523 checks
python3 -m agent.frontier                          # what's tried / open / dead
python3 -m agent.error_analysis                    # where the model fails
python3 -m agent.feature_lab                       # feature research audit trail
python3 -m agent.policy_eval                       # counterfactual decision replay

# costs money (LLM) — set the ceiling deliberately
python3 run_agent.py --smoke --fresh                        # ~$0.02 plumbing check
python3 run_agent.py --fresh --n-candidates 4 --research-state \
        --data-tools --feature-discovery --max-spend-usd 3  # a real run

# costs CPU only
python3 -m agent.final_ensemble --seeds 16         # rebuild the submitted result
python3 -m agent.axis_sweep --axis X --values a,b  # controlled paired comparison
python3 -m agent.learning_curve                    # data-limited or not?
```

**Three traps that cost real time here:**

1. **Never run two training jobs at once, and never run the test suite during a
   run.** The executor chmods the data directory for the duration of a
   subprocess; a concurrent job *or* `tests/test_harness.py` (its data-boundary
   tests take the same lock) dies with `PermissionError` mid-run. Recover with
   `chmod -R u+rw kuairand-starter-kit/KuaiRand-Pure`.
2. **Resuming an already-converged journal silently does nothing.** Use
   `--fresh` (it archives to `logs/archive_<ts>/`, never deletes).
3. **`logs/` holds two different things.** `journal.jsonl` is whatever *search*
   run is on disk. The **submitted** result is separate and survives `--fresh`:
   `logs/ensemble_results.json` + `logs/final_ensemble/`. Reading a search
   journal's best node as "the result" is how the headline once drifted from its
   evidence.

---

## 7. What's left before submission

1. **Devpost written description** — not started.
2. **Regenerate the submission CSV** and validate it:
   ```bash
   python3 -m agent.make_submission --split valid --score --ensemble
   cd kuairand-starter-kit && python3 submit.py --check --split valid ../submission_valid.csv
   ```
3. **The hidden-test evaluation — one shot, never used.**
   `python3 -m agent.make_submission --final-test-eval --ensemble`. A lockfile
   (`results/final_evaluation.lock`) enforces one-time use. **Do not run it until
   the configuration is final.**
4. **Ask the organisers about a contradiction in the problem statement.** §2.3
   says *"NDCG@10 / Recall@50, click = positive"*; everywhere else (§2.4, §2.6,
   the starter kit, Appendix A.4) says **GAUC / nDCG@5 on `long_view`**, and A.4
   states Recall is *"not scored here"*. We implement the latter. If §2.3 were
   live, the whole task definition would be wrong.

---

## 8. House rules

- **Grade everything in σ (σ = 0.0008), never in raw deltas.** A +0.0015 gain is
  under 2σ and is not distinguishable from seed noise.
- **A score computed on the same data that selected it is not evidence.** Three
  separate results here turned on this: the epoch argmax is fitted to validation;
  "best of five aggregation rules" was a +0.46σ mirage that resampling refuted
  (10/24 wins); and a *working* method sat rejected in the codebase because its
  guard compared it against a checkpoint chosen on the same set. Prefer a
  held-out or resampled comparison.
- **Never trust a single seed.** Use `--reseed-top` or `agent.axis_sweep` with
  ≥5 paired seeds.
- **A number without its evidence on disk is not a result.** An earlier headline
  had to be withdrawn because its member arrays were archived away. Anything
  quoted as the submitted score must be rebuildable by one command.
- **Never touch `kuairand-starter-kit/evaluate.py`.** It is the scoring ground
  truth.
- **Never train on test.** Generated code runs in a sandbox where the real data
  directory is unreadable and the test split has its label columns physically
  stripped. Don't work around it.
- **Report negative results honestly.** Most of this project's value is in
  well-measured failures; a clean null with a mechanism beats a marginal,
  unverified bump.

---

## 9. Where the detailed evidence lives

| Path | What's in it |
|---|---|
| `config/modification_menu.json` | the search space + **28 dead ends with mechanisms** |
| `logs/opus_research/journal.md` | expert research trajectory (E1–E7) with reasoning |
| `logs/opus_research/DISTILLATION.md` | how each research behaviour became an agent capability |
| `logs/feature_registry.jsonl` | every feature proposed, probed, accepted or rejected |
| `logs/ensemble_results.json` | the authoritative result + provenance (git sha, data fingerprint) |
| `logs/journal.jsonl` | per-iteration hypothesis, diff, metrics, errors (competition deliverable) |
