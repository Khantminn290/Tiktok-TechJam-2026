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

---

## Setup

```bash
git clone <this repo> && cd Tiktok-TechJam-2026

# 1. dependencies
python3 -m pip install numpy torch openai        # add `anthropic` only if PROVIDER=anthropic

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
python3 tests/test_harness.py        # 55 checks
```

Covers the safety gate, cross-axis validation, every search-policy branch
(including crossover), the convergence rule, all five executor failure modes, and
the spend ceiling. **Run this before every real session** — it is free and catches
the failures that would otherwise cost money to discover.

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

### 3. A real run (only after the smoke test passes)

```bash
python3 run_agent.py --fresh --max-spend-usd 15
```

Whoever kicks off a scored run sets `--max-spend-usd` deliberately — the default is
**$2**, low on purpose so nobody burns the shared key by accident. The run stops at
whichever comes first: convergence (ε = 0.002 over N = 3 scored iterations), 50
iterations, 6 hours, or the spend ceiling.

Useful flags: `--fresh` (archive previous logs and start at iteration 0; without it
the agent *resumes* the journal, which is how a crashed run continues),
`--draft-count N`, `--llm-model NAME`, `--inject-error-at N` (robustness self-test).

### 4. Read the results

```bash
python3 -m agent.report              # per-iteration history, spend, tokens, GPU-hours
python3 -m agent.report --html       # search-tree view -> logs/tree.html
```

The report prints delta over baseline **both raw and in units of the baseline's own
5-seed standard deviation (σ = 0.0008)**. That second number is the one that matters:
a +0.0015 gain is under 2σ and is not distinguishable from seed noise, while +0.004
is 5σ and is a real effect.

### 5. Final submission — runs exactly once, at the end

```bash
python3 -m agent.make_submission --split valid --score --ensemble   # inspect first
python3 -m agent.make_submission --final-test-eval --ensemble       # THE one-time eval
```

`--final-test-eval` is the **single** hidden-test evaluation of the whole project.
Everything before it develops on train + validation only. It writes
`results/final_results.json` with test metrics, deltas over the baseline, and the
delta in σ units.

`--ensemble` rank-averages the top-K distinct successful nodes (K from
`config/llm_config.json`, default 3) instead of using the single best node. Scores
are rank-normalised before averaging because different nodes produce scores on
different scales; averaging raw values would let whichever model has the widest
spread dominate. It costs no extra model calls — it just averages score arrays that
are already on disk — and the comparison table prints best-single next to
best-ensemble so you can see whether it actually helped.

### Logging manual interventions

```bash
python3 -m agent.interventions "restarted after laptop sleep"
python3 -m agent.interventions --list
```

Autonomy is scored by how few of these there are. The log is deliberately separate
from the agent's own journal — the agent cannot write to it, so the number can't be
flattered by the thing being measured.

---

## The search space

`config/modification_menu.json` defines everything the agent is allowed to change.
**Anything not in it is invisible to the search**, which makes it the
highest-leverage file in the repo. Seven axes, priority-ordered from the organisers'
own measured findings:

| Priority | Axis | What it changes |
|---|---|---|
| 1 | `loss` | pointwise → BPR pairwise / listwise softmax / hybrid |
| 2 | `score_prior` | train-only Bayesian item/author propensity blend |
| 3 | `user_history` | behaviour-sequence pooling, DIN-style attention |
| 4 | `multitask` | auxiliary heads on click / like / forward, censored watch-time |
| 5 | `model` | FM → DeepFM / DCN |
| 6 | `temporal` | hour-of-day, day-of-week |
| 7 | `training` | schedules, two-stage fine-tuning |
| 8 | `data_extras` | extra data sources — **two options here are locked** |

`config/modification_menu.md` records the reasoning and the two approaches the
organisers already measured as dead ends (more static features; larger embedding
dimension), so iterations aren't spent rediscovering them.

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
  report.py                  journal -> human-readable report and tree view
runtime/
  train_lib.py               training engine the generated scripts build on
  seed_solution.py           the solution-script interface, by example
  API.md                     docs injected into every prompt
config/
  modification_menu.json     the search space          (+ .md rationale)
  llm_config.json            provider/model defaults   (never keys)
  model_rates.json           $/token for the budget guard
  agent_config.json          caps, seed, safety-gate override
tests/test_harness.py        55 checks, no model calls
logs/
  nodes/node_NNN/            one self-contained folder per research iteration
    solution.py              complete generated script
    record.json              hypothesis, choices, metrics, errors, usage
    metrics.json             validation metrics (successful nodes)
    resource.json            measured compute (when produced)
    scores_*.npy             prediction artifacts (local, Git-ignored)
  smoke/                     isolated smoke-test journal and nodes
  history/                   prior run-level summaries; nodes stay in nodes/
  journal.jsonl              compact index of all active research nodes
  best_* / final_summary     current run summary artifacts
```

---

## Reproducing the numbers

Harness sanity checks (these should match before trusting anything else):

| Check | Expected |
|---|---|
| `python3 kuairand-starter-kit/baseline.py --model random` | test primary ≈ 0.4753 |
| `python3 kuairand-starter-kit/baseline.py --model fm` | valid ≈ 0.6016, test ≈ 0.5946 |
| Split sizes | 1,141,112 / 124,909 / 170,588 |

## Limitations and what we'd improve

- The five-iteration verification run reached +0.0018 validation (about 2σ). It was
  capped at five to prove the machinery, not to find gains — treat that number as
  unproven rather than as a ceiling.
- Crossover only combines menu choices, not code. Merging two scripts' actual
  implementations would be strictly more expressive.
- The unbiased random-exposure diagnostic is wired for the NumPy engine only, and
  the agent does not yet act on it automatically.
- Search is single-threaded; parallel drafts would cut wall-clock substantially.
- Bonus benchmarks (KuaiRand-1k / 27k) are not attempted; the data cache would need
  chunking to scale.

## Team

Solo participant. Every agent-authored script is journaled per iteration in
`logs/journal.jsonl` with its hypothesis. Every node is self-contained under
`logs/nodes/node_NNN/`, so its source, record, metrics, errors, and local prediction
artifacts can be inspected without cross-referencing `solutions/` and `runs/`.
