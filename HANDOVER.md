# Handover

## What this is

An autonomous agent that improves a KuaiRand-Pure ranking model on its own. It searches
over **complete Python training scripts, not diffs**: every attempt is a *node* (a full
script plus the validation score it earned), a policy picks which node to extend next,
an LLM writes the next script, a guarded subprocess runner scores it, and the result is appended
to `logs/journal.jsonl`. A run stops when it **converges** — the best validation score
hasn't improved by more than 0.002 over the last 3 scored iterations — or hits the
iteration, time, or spend cap. Read [README.md](README.md) for the architecture and
[config/modification_menu.json](config/modification_menu.json) for the search space
before changing any search logic.

## Current strongest verified result

The independent branch's frozen five-seed candidate reaches **0.605660439** on
validation versus the official **0.6016** baseline. It combines the strong BPR
+ recency-history configuration with a fixed, label-free repeat-fatigue reranker.
All five paired seeds improved. Reproduce it with:

```bash
python3 -m agent.verified_ensemble --seeds 5
```

The command reuses a member only when its code, configuration, complete cache,
and runtime hashes match; otherwise it retrains. It evaluates validation only
and writes test predictions blind. Details and rejected ideas are in
[`research/README.md`](research/README.md).

## Get running in under 5 minutes

```bash
git clone <repo> && cd Tiktok-TechJam-2026
python3 -m venv .venv && source .venv/bin/activate
pip install numpy torch openai          # torch is the slow one, ~2 min

cp .env.example .env
```

**Ask KM for the shared OpenAI key** — it is not in this repo and must never be
committed. Paste it into the single `OPENAI_API_KEY=` line in `.env` (which is
gitignored). Nothing else in `.env` needs changing.

Download the dataset once (see README Setup), then:

```bash
python3 tests/test_harness.py     # 129 checks, no model calls, zero spend
python3 run_agent.py --smoke      # ~$0.015, 3 iterations, proves you reach the model
```

`--smoke` ends with a **SMOKE TEST VERDICT** block. Read that, not the individual
iteration errors: it tells you whether *your setup* works. Some smoke scripts may fail —
the cheap test model sometimes writes buggy code, which is model quality, not a broken
install. Only `PLUMBING BROKEN` means your setup is wrong.

## Before you run anything real

**We all share one $50 budget.** Always pass `--max-spend-usd` explicitly and start
small — the default is $2 on purpose.

```bash
python3 run_agent.py --fresh --max-spend-usd 5
```

The loop stops *before* the call that would breach your ceiling, and prints running
spend every iteration. Cost is computed from real API token counts against the rates in
[config/model_rates.json](config/model_rates.json); a model missing from that table gets
a deliberately expensive fallback rate so it stops early rather than overspending. See
README → *Cost control*. For reference: a converged 8-iteration run on `gpt-5.4` cost
**$0.12**.

## Extending the search (the thing you'll probably do)

Everything the agent is allowed to change lives in
[config/modification_menu.json](config/modification_menu.json). **Anything not in that
file is invisible to the agent** — it cannot try what it cannot see, so this is where
new ideas go. Add an option under an existing axis, or add a whole axis with a
`priority` (lower = tried earlier). Use `requires` for cross-axis constraints and
`"locked": true` for anything leakage-sensitive. Then implement the option in
[runtime/train_lib.py](runtime/train_lib.py) and document it in
[runtime/API.md](runtime/API.md), which is pasted into every prompt — if the agent
can't find out how to use your option, it will guess and waste iterations.
`config/modification_menu.md` records what the organisers already measured as dead ends;
don't respend iterations there.

## Where results live

- `logs/nodes/node_NNN/` — one self-contained folder per iteration containing
  `solution.py`, `record.json`, metrics/resources, and local score arrays.
- `logs/journal.jsonl` — compact one-line-per-iteration index and competition log.
  Use `python3 -m agent.report --node N` for a complete readable node view.
- `logs/smoke/` — isolated smoke-test nodes; smoke runs never replace research logs.
- `logs/best_solution.py` + `logs/best_metrics.json` — the current winning script.
- `python3 -m agent.report` — readable summary; `--html` draws the search tree.

The report shows delta over baseline twice: raw, and **in units of σ**, where σ = 0.0008
is the official baseline's own run-to-run noise. Plain version: under ~2σ the gain could
just be luck; above ~3σ it's probably real. Quote the σ number when you claim an
improvement.

Final submission runs **exactly once, at the end**:
`python3 -m agent.make_submission --verified-ensemble` writes the blind CSV without
opening test outcomes. If a one-time local test evaluation is explicitly wanted, run
`python3 -m agent.make_submission --verified-ensemble --final-test-eval` only at the end.

## Known rough edges

- **First training run builds a ~1 minute data cache.** Later runs load it in seconds.
- **`torch` is only needed for the neural menu options**, but it's a big install.
- **Smoke runs on the cheap model often produce failed scripts** — see the VERDICT note
  above. Not a setup problem.
- **`--ensemble` only helps with several comparably-good nodes.** On a 3-node run it
  made things *worse* (0.6024 vs 0.6037); on an 8-node run it helped (+0.0025 vs
  +0.0019). Always check the printed single-vs-ensemble table before trusting it.
- **The Anthropic provider works but has never had a full paid run** — it initializes,
  authenticates, and fails gracefully, that's all we've proven. OpenAI is the tested path.
- **Runs are single-threaded**; expect 40–120 s of training per iteration on CPU. GPU is
  used only if you set `KUAIRAND_DEVICE=auto` (or `mps`/`cuda`); default is CPU.
- **Anything you do by hand during a run must be logged**:
  `python3 -m agent.interventions "what you did"`. Autonomy is scored on that count.

## Who to ask
- **Code-level questions** — self-serve first: the policy is
  [agent/policy.py](agent/policy.py) (~80 lines), the loop is
  [agent/loop.py](agent/loop.py), the failure handling is
  [agent/executor.py](agent/executor.py). Each is short and commented.
