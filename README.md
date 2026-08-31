# Autonomous ML Research Agent — KuaiRand-Pure

TikTok TechJam 2026, Track 2. An LLM-driven agent that runs the full ML engineering
loop on its own: read the problem, form a hypothesis, write a complete training
script, run it, read the score, decide what to try next — and keep going until the
validation score stops improving.

The thing being searched over is **complete Python scripts, never diffs**. Every
attempt is a *node*: a full script plus the validation score it earned. A search
policy picks which node to extend, the model writes the next script, a sandboxed
runner scores it, and the result is appended to `logs/journal.jsonl`, which is both
the agent's memory and the competition's required run-log deliverable.

**Task (fixed by the organisers):** rank each user's logged impressions;
positive label is `long_view`; metrics are GAUC and nDCG@5; primary score is their
mean. Official baseline: **0.6016** validation / **0.5946** test. The metric ceiling
is **0.8484** / **0.8645**, not 1.0 — 27% of users have no positive label at all,
so judge progress against that, not against a perfect score.

## Result

Scored once on the hidden test set, at final submission:

| split | primary | GAUC | nDCG@5 |
|---|---|---|---|
| official baseline | 0.5946 | 0.6610 | 0.5282 |
| **this submission** | **0.59810** | **0.66510** | **0.53110** |
| **absolute delta** | **+0.0035** | **+0.0041** | **+0.0029** |

**Judged score = +0.0035** — the mean absolute delta across GAUC and nDCG@5, at
**4.37σ** on the baseline's own seed noise (σ = 0.0008).

The submitted artifact is a 16-seed rank-averaged ensemble of one configuration,
scoring `0.60541` on validation (`+0.00381` over the 0.6016 baseline). Members
average 0.60463 ± 0.00032. `k=16` is *all* seeds trained, fixed before any score
was seen, so the figure carries no selection bias.

The validation-to-test drop is `-0.0073`. The official baseline loses `-0.0070`
across the same two splits, so this is the ordinary generalisation gap rather
than validation overfitting.

The hidden test was evaluated **exactly once**, after the configuration was
frozen — enforced by `results/final_evaluation.lock`, not by discipline. All
development used train + validation only.

### Where the deliverables are

| Deliverable | File |
|---|---|
| Devpost project story | [`docs/DEVPOST_STORY.md`](docs/DEVPOST_STORY.md) |
| Results summary + resource usage | [`RESULTS.md`](RESULTS.md) |
| Per-iteration run log | [`logs/ITERATION_LOG.md`](logs/ITERATION_LOG.md) |
| Machine-readable journal | [`logs/journal.jsonl`](logs/journal.jsonl) |
| Full evidence packet for judges | [`results/JUDGE_PACKET.md`](results/JUDGE_PACKET.md) |
| Final submission CSV | `submission_test.csv` (rebuild: see below) |
| Official baseline, reproduced here | [`logs/baseline/`](logs/baseline/) |

---

## Setup

```bash
git clone https://github.com/Khantminn290/Tiktok-TechJam-2026.git
cd Tiktok-TechJam-2026

# 1. dependencies
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# 2. dataset (~45 MB download, ~194 MB extracted; not committed)
cd kuairand-starter-kit
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz && cd ..

# 3. credentials — copy the template and paste the shared team key
cp .env.example .env
$EDITOR .env
```

`.env` is gitignored and is the **only** place an API key lives. Everything else
(provider, model names) falls back to `config/llm_config.json`, which is committed
and contains no secrets. A teammate who clones the repo and fills in `.env` can run
the agent with no code changes.

| Variable | Purpose |
|---|---|
| `PROVIDER` | `openai` (default) or `anthropic` |
| `OPENAI_API_KEY` | the shared key |
| `OPENAI_MODEL` | model for real, scored runs (default `gpt-5.4`) |
| `TEST_MODEL` | cheap model used only by `--smoke` (default `gpt-5.4-mini`) |
| `ANTHROPIC_API_KEY` / `MODEL_NAME` | used when `PROVIDER=anthropic` |
| `DRAFT_COUNT` | diverse initial drafts before committing to a lineage |

---

## Running it — in this order

### 1. Harness self-test (free, no model calls, seconds)

```bash
python3 tests/test_harness.py        # prints the live count
```

Covers the safety gate, cross-axis validation, every search-policy branch
(including crossover), the convergence rule, all five executor failure modes, and
the spend ceiling. **Run this before every real session** — it is free and catches
the failures that would otherwise cost money to discover.

### 1b. Fault and recovery suite

```bash
python3 -m agent.faults --live       # writes results/fault_report.json
```

Injects faults across the whole surface — a config missing an axis, wrong call
arity, a capability's return shape misused, malformed model output, unparseable
code, a broken feature builder, a training timeout, a prediction array full of
NaN, a missing ensemble member, a duplicate member request, an exhausted repair
chain, an unaffordable experiment, an invalid spec, a single seed offered as a
discovery, crashes offered as convergence, and an unverifiable provenance stamp.

Each one is checked on ten axes: detected, **named correctly**, **routed
correctly**, bounded so it cannot repeat forever, charged to the compute budget
if and only if compute was spent, kept out of the evidence, journalled,
survivable, free of human intervention, and — where recovery is impossible —
terminated cleanly.

The routing distinction is the part worth reading. `repair` means the idea is
fine and the artifact is broken; `pivot` means the approach itself cannot work.
A timeout is a pivot, not a repair — re-running the same work would time out
again — and a mechanism that cannot move a within-user ranking metric is not
repairable either, because a fixed version of an inert idea is equally inert.

`--live` additionally spawns real subprocesses through the real executor: one
script that exits 0 while writing NaN predictions (a clean exit code is not
evidence), and one that never returns.

A full live agent run with a deliberate failure in it is recorded under
`results/live_fault_run/`:

```bash
python3 run_agent.py --fresh --max-iterations 4 --inject-error-at 1 --max-training-runs 4
```

That historical run is intentionally labelled **incomplete**: it selected the
right debug action, then two network failures prevented a later scored result.
The deterministic full-loop evaluation removes that network confounder while
still driving the real policy, preflight, sandbox, and executor:

```bash
python3 -m agent.recovery_eval       # three faults, each must reach later success
python3 -m agent.epoch_sensitivity   # diagnostic only; never promotes a model
```

### 2. Budget-capped smoke test (cheap, ~3 iterations)

```bash
python3 run_agent.py --smoke --fresh
```

`--smoke` forces 3 iterations, a **$1** ceiling, and `TEST_MODEL`, so plumbing
checks never spend at full-model rates. Use this after any change to the provider
layer. It is a plumbing check, not a scored run.

> **First run is slower.** The very first training run parses the raw CSVs into
> `runtime/cache/` (~1 minute, one time only). Later runs load the cache in
> seconds. Preflight also makes one free `models.list` call to prove the key
> authenticates before anything is spent — `--no-verify-key` skips it if you're
> offline.

### 3. A real research run (only after the smoke test passes)

```bash
python3 run_agent.py --fresh --max-spend-usd 15
```

Whoever kicks off a scored run sets `--max-spend-usd` deliberately — the default is
**$2**, low on purpose so nobody burns the shared key by accident. The run stops at
whichever comes first: convergence (ε over N = 3 scored iterations), the
iteration cap, the **training-run** cap, the wall-clock ceiling, or the spend
ceiling.

**Two convergence rules, and they are not the same thing.**

The **organizers'** rule is the official one — it defines when a run stops and
which checkpoint is scored: **ε = 0.002, N = 3**, or the 50-iteration cap, or
the 6-hour ceiling, whichever comes first.

This project additionally runs a **stricter internal controller** at
ε = 0.00048 (0.60σ — the upward drift a running maximum shows by luck at this
benchmark's noise floor; see `agent/validity.py::convergence_epsilon`). At
2.5σ the loop stops on differences larger than anything the benchmark still has
to offer, so a search that wants to keep looking needs a tighter bar.

The stricter controller is useful for research but **does not extend submission
eligibility**. Anything discovered after the organizer rule fires is later
research evidence, not a checkpoint the competition can score. Both rules are
reported, and never conflated:

```bash
python3 -m agent.convergence_report
```

> An earlier version of this README described 0.002 as "the earlier hard-coded
> constant" that a calibrated value replaced. That was wrong: 0.002 is the
> organizers' published rule, not a bug that got fixed.

Note that an outer iteration is **not** one training run: a paired 3-seed
confirmation is one node and six training executions, which is why there is a
separate `--max-training-runs` cap. For a run that demonstrates the full
system, use `--competition` (prints its fully resolved configuration before
spending anything).

The official competition command is deliberately separate:

```bash
python3 run_agent.py --competition --fresh
```

It runs an actual validation-only FM baseline as node 0, reproduces the verified
incumbent as agent-authored code, performs paired confirmation, and schedules the
fixed 16-seed ensemble before the organizer convergence window can close. In
this profile epsilon is exactly `0.002`, N is `3`, and no branching or pending-
ensemble gate may defer the official stop.

Current figures — test count, incumbent, convergence threshold, latest run —
are generated, not retyped:

```bash
python3 -m agent.results_report --run-tests    # writes RESULTS.md
```

Useful flags: `--fresh` (archive previous logs and start at iteration 0; without it
the agent *resumes* the journal, which is how a crashed run continues),
`--draft-count N`, `--llm-model NAME`, `--inject-error-at N` (robustness self-test).

### 4. Read the results

```bash
python3 -m agent.report              # per-iteration history, spend, tokens, GPU-hours
python3 -m agent.results_report --run-tests   # regenerate RESULTS.md from artifacts
python3 -m agent.manifest --run-tests         # the one canonical results/manifest.json
python3 -m agent.judge_packet                 # writes results/JUDGE_PACKET.md
python3 -m agent.devpost                      # writes docs/DEVPOST_SUBMISSION.md
```

`results/manifest.json` is the single source every other artifact reads —
scores are recomputed from the stored member predictions at generation time,
never quoted. `JUDGE_PACKET.md` is generated *from* it: the problem, the loop,
the action space, how experiments are chosen, how confirmation and ensembling
work, both results with their attribution, the run's cost, the fault suite, the
convergence rules, the limitations, and the exact commands to reproduce all of
it. A test asserts the packet's numbers move when the manifest moves, so it
cannot quietly go stale.

### The dashboard (easiest way to see all of this)

```bash
python3 -m pip install streamlit
streamlit run app.py                 # http://localhost:8501
```

Five tabs, in the order someone actually evaluates this: **Overview** (the
result, how the loop works, and each competition criterion mapped to auditable
evidence and its limits), **Watch it run** (the
search as it grows, as a real parent-linked tree, with decisions, errors and
recoveries), **Iteration log** (every node, filterable, with the raw journal
record and the script the agent wrote), **Robustness** (every injected fault
and what the agent did about it, with component, real-subprocess, and full-loop
evidence kept separate), and **Start a run** (the resolved configuration is
shown first).

Every headline figure in the dashboard is read from `results/manifest.json`,
so the dashboard and this README cannot disagree with the artifacts.

The dashboard is a window, not a decision-maker: evidence tiers are recomputed
there, so a single-seed result displays as PRELIMINARY however good it looks,
and the one-time hidden-test evaluation is deliberately **not** wired to a
button.

**Or watch it in a terminal-free page.** Start this *before* kicking off a run
and leave it open — it polls every 3s, so new decisions, errors and
recoveries appear as they happen. This is the view to screen-record:

```bash
python3 -m agent.live                # http://localhost:8000
python3 -m agent.live --port 8080    # if 8000 is taken
```

It shows, per node: the allocator's chosen experiment family, the question the
agent could not answer, its competing hypotheses, the experiment it ran, the
score with Δ and σ — and for failures, the error **and what happened next**
(recovered by the debug chain, abandoned, or rejected by preflight for no
compute). The header carries live counters: nodes, crashes, free preflight
rejections, confirmations, promotions, training runs against the cap, and spend.

**Inspect a finished run** as a static page you can share or archive:

```bash
python3 -m agent.viz --open          # writes viz/tree.html and opens it
python3 -m agent.viz --journal logs/opus_research/phase4_competition_run.jsonl \
                     --out viz/phase4.html
```

Indentation on that page is the real search tree: a child node branched from its
parent, so `improve` chains, debug chains and paired confirmations sit under
whatever they extended. Evidence states are recomputed from the current rules at
render time, so nothing can display as more established than it is.

The report prints delta over baseline **both raw and in units of the baseline's own
5-seed standard deviation (σ = 0.0008)**. That second number is the one that matters:
a +0.0015 gain is under 2σ and is not distinguishable from seed noise, while +0.004
is 5σ and is a real effect.

### 5. Final submission — runs exactly once, at the end

```bash
python3 -m agent.make_submission --split valid --score --ensemble   # inspect first
python3 -m agent.make_submission --final-test-eval --ensemble       # THE one-time eval
```

> **Already spent.** `results/final_evaluation.lock` is present; the result is in
> [`RESULTS.md`](RESULTS.md). A second run refuses unless `--admin-override` is
> passed, and logs the attempt either way. The commands below are documentation
> of what was run, not an invitation to rerun it.

`--final-test-eval` is the **single** hidden-test evaluation of the whole project.
Everything before it develops on train + validation only. It writes
`results/final_results.json` with test metrics, deltas over the baseline, and the
delta in σ units.

`--ensemble` rebuilds the **submitted** ensemble: every seed of the one reported
configuration, read from `logs/final_ensemble/` (see `agent/final_ensemble.py`).
`k` is fixed before any score is seen and uses *all* seeds trained, so no member
is ever chosen on validation. If members are missing it refuses to average a
subset rather than quietly reporting a different number. Scores are
rank-normalised before averaging because the metric reads only ordering, and
averaging raw values would let whichever model has the widest spread dominate.
It costs no model calls — it averages arrays already on disk.

`--legacy-topk-ensemble` is the older behaviour (top-K nodes by validation score,
distinct configs) and is kept only for inspection. It combines two effects
measured and rejected on this project — validation selection bias (+0.00081) and
heterogeneous blending — so it does **not** reproduce the reported result, and
says so at runtime.

### Logging manual interventions

```bash
python3 -m agent.interventions "restarted after laptop sleep"
python3 -m agent.interventions --list
```

Autonomy is scored by how few of these there are. The log is deliberately separate
from the agent's own journal — the agent cannot write to it, so the number can't be
flattered by the thing being measured.

---

## What the agent can do

**Modify the pipeline** — 10 axes, ~45 options: `loss`, `neg_sampling`,
`user_history`, `multitask`, `model`, `temporal`, `training`, `data_extras`,
`sample_weighting`, `regularization`. Plus pipeline constants no menu axis
reaches: embedding size, learning rate, epochs, patience, decay constants,
checkpoint rules.

**Run experiments** — one action space, not tiers:

| Action | What it is |
|---|---|
| **New idea** | a fresh configuration |
| **Refine best** | extend the leading result |
| **Fix failure** | read its own traceback and repair its script |
| **Implement** | write a mechanism the menu cannot express (a new objective, representation, or aggregation rule) |
| **Confirm** | a paired multi-seed experiment — the only thing that can make a result CONFIRMED |
| **Ensemble** | train k seeds of a confirmed configuration and average their rank-normalised predictions |

`implementation_path` in the journal records only *how* an experiment was
implemented — `A` for a menu configuration, `B` for an agent-written script. It
is not a hierarchy and neither is the default.

**On ensembling.** This was added because it was a real gap: the submitted
result is a 16-seed ensemble, but ensembling was not in the agent's action
space, so the largest measured gain available (**+0.00078**, about 1σ) sat
outside its reach and a human ran `agent.final_ensemble --seeds 16` afterwards.
The agent now schedules it itself once a configuration is confirmed or has
scored repeatedly. Its value is measured against the **mean** member, never the
best one — the best of k draws sits above the mean by construction, so that
comparison would report a gain even if ensembling did nothing.

## The search space

`config/modification_menu.json` defines everything the agent is allowed to change.
**Anything not in it is invisible to the search**, which makes it the
highest-leverage file in the repo. Ten axes, priority-ordered from the organisers'
own measured findings:

| Priority | Axis | What it changes |
|---|---|---|
| 1 | `loss` | pointwise → BPR pairwise / listwise softmax / hybrid |
| 2 | `user_history` | behaviour-sequence pooling, DIN-style attention |
| 3 | `multitask` | auxiliary heads on click / like / forward, censored watch-time |
| 4 | `model` | FM → DeepFM / DCN |
| 5 | `temporal` | hour-of-day, day-of-week |
| 6 | `training` | schedules, two-stage fine-tuning |
| 7 | `data_extras` | extra data sources — **two options here are locked** |
| 8 | `neg_sampling` | uniform / popularity-biased / hard negatives |
| 9 | `sample_weighting` | per-row vs per-user normalisation |
| 10 | `regularization` | L2 strength, dropout |

`config/modification_menu.md` records the reasoning, and
`notes.tested_dead_ends` holds **15 approaches measured and ruled out here or by
the organisers** — each with its mechanism, injected verbatim into every planning
prompt so iterations aren't spent rediscovering them.

**The safety gate:** the two leakage-sensitive `data_extras` options are stripped
from the prompt *and* rejected by the validator. They can only be enabled by a human
editing `config/agent_config.json`, and doing so should be logged as an intervention.

---

## Cost control

Spend is computed from **real token counts in the API response**, never estimated,
using rates in `config/model_rates.json` (data, not logic — update it when prices
change). Running spend prints after every iteration, and the loop aborts *before*
the call that would exceed the ceiling, journaling the reason into
`logs/final_summary.json`.

A model missing from the rate table is priced with a deliberately high fallback, so
an unpriced model over-estimates spend and stops the run early rather than quietly
overspending. `run_agent.py` warns loudly when that happens.

---

## Repo layout

```
run_agent.py                 entrypoint + preflight (fails fast on bad config)
agent/
  contracts.py               Node, ExperimentTree, journal persistence
  menu.py                    search space, validity checks, safety gate
  policy.py                  decide_action: draft / debug / improve / crossover
  prompts.py                 four-section prompt builder
  llm.py                     provider-agnostic client (OpenAI | Anthropic)
  pricing.py                 rate table + spend tracker
  executor.py                subprocess sandbox, timeout, contract checks
  loop.py                    orchestration, convergence, spend + GPU accounting
  interventions.py           manual-intervention log
  make_submission.py         submission CSV, ensembling, one-time test eval
  report.py                  journal -> human-readable report
  live.py                    live view for a running agent (localhost:8000)
app.py                       Streamlit dashboard (streamlit run app.py)
  viz.py                     journal -> static experiment-tree page
  results_report.py          artifacts -> RESULTS.md, tiered VERIFIED/OBSERVED/OPEN
  capabilities.py            the capability contract (what/where/cost/returns)
  preflight.py               8-stage script validation before any training
  profiles.py                --competition profile: resolve, validate, print
  budget.py                  decisions vs training executions, separately capped
  evidence.py                PRELIMINARY / CONFIRMED / REJECTED, from how it was measured
  confirm.py                 paired multi-seed experiments, executed
  ensemble_experiment.py     ensembling as an agent action, not a human step
  experiment_spec.py         an experiment the system runs, not describes
  knowledge.py               scoped claims with counterevidence
  allocator.py               transparent utility over experiment families
  verify_incumbent.py        recompute the submitted score from stored predictions
runtime/
  train_lib.py               training engine the generated scripts build on
  seed_solution.py           the solution-script interface, by example
  API.md                     docs injected into every prompt
config/
  modification_menu.json     the search space          (+ .md rationale)
  llm_config.json            provider/model defaults   (never keys)
  model_rates.json           $/token for the budget guard
  agent_config.json          caps, seed, safety-gate override
tests/test_harness.py        harness self-tests, no model calls, no training
docs/
  ARCHITECTURE.md            design, measured before/after, honest autonomy level
  RESEARCH_LOG.md            dead ends with evidence, and what is still open
RESULTS.md                   generated -- do not hand-edit
logs/                        journal.jsonl, solutions/, best_*, final_summary.json
```

---

## Reproducing the numbers

### Start here, in a fresh clone

The 16 ensemble members' prediction arrays are **not committed** — they are ~36 MB
of floats that rebuild exactly from the recorded seeds. So in a fresh clone, build
them first; everything else depends on them:

```bash
python3 -m agent.final_ensemble --seeds 16     # ~10 min CPU, writes logs/final_ensemble/
python3 -m agent.verify_incumbent              # recomputes 0.60541 from those arrays
```

`verify_incumbent` fails with missing `scores_valid.npy` if you skip the first
command. That is the intended order, not a bug.

Then the rest:

```bash
python3 tests/test_harness.py                             # 1116 passed, 0 failed
python3 -m agent.baseline_repro                           # reproduces the official baseline
python3 -m agent.make_submission --split valid --score --ensemble
python3 -m agent.iteration_log                            # regenerates the run log
python3 -m agent.results_report                           # regenerates RESULTS.md
```

### Sanity checks

These should match before trusting anything else:

| Check | Expected |
|---|---|
| `python3 kuairand-starter-kit/baseline.py --model random` | test primary ≈ 0.4753 |
| `python3 kuairand-starter-kit/baseline.py --model fm` | valid ≈ 0.6016, test ≈ 0.5946 |
| Split sizes | 1,141,112 / 124,909 / 170,588 |

### The hidden-test evaluation

Already spent, and guarded so it cannot be spent twice:

```bash
python3 -m agent.make_submission --final-test-eval --ensemble
```

`results/final_evaluation.lock` exists, so a second run refuses unless
`--admin-override` is passed — and logs the attempt either way.

## Limitations and what we'd improve

- **Headroom on this benchmark is genuinely narrow, and that is a measurement,
  not an excuse.** Eight model families — FM, DeepFM, DCN, DIN, GRU4Rec, ItemCF,
  GBDT, item-popularity — all land in 0.55–0.605, and a gradient-boosted model
  given every raw column the pipeline discards adds only **+0.39σ** over the
  incumbent when the blend weight is fixed in advance and resampled
  (`agent/residual_screen.py`) — under half the seed-noise scale. Interventions
  are judged in σ, and 26 are recorded as dead ends with mechanisms. We do *not*
  claim the task is label-noise-limited: only 0.177% of within-user
  positive/negative pairs are feature-identical, so that argument is unsupported
  by measurement.
- Crossover only combines menu choices, not code. Merging two scripts' actual
  implementations would be strictly more expressive.
- The unbiased random-exposure diagnostic is wired for the NumPy engine only, and
  the agent does not yet act on it automatically.
- Search is single-threaded; parallel drafts would cut wall-clock substantially.
- Bonus benchmarks (KuaiRand-1k / 27k) are not attempted; the data cache would need
  chunking to scale.

## Team

Solo participant. Every agent-authored script is journaled per iteration in
`logs/journal.jsonl` with its hypothesis, and the full source is kept in
`logs/solutions/`.
