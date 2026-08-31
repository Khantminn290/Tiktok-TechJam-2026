# Devpost Submission: Autonomous ML Research Agent

## Result

Scored once on the hidden test set: primary **0.59810** against the official baseline's 0.5946, an absolute gain of **+0.0035** (GAUC +0.0041, nDCG@5 +0.0029). The judged score is the mean absolute delta across the two metrics: **+0.0035**.

## Inspiration

Recommender-system progress is easy to overstate when one lucky seed, repeated
validation selection, or a post-hoc ensemble is presented as research. We built
an agent that treats evidence quality, execution safety, and reproducibility as
part of the optimization problem rather than paperwork after it.

## What it does

The agent reads its append-only research history, forms a measurable hypothesis,
writes a complete Python experiment, checks it for contract and leakage failures,
runs it in a label-protected sandbox, scores it with the official KuaiRand-Pure
evaluator, and decides whether to repair, pivot, confirm, ensemble, or stop.

Its current validation artifact scores **0.60541**, a
**+0.00381** gain over the organizer baseline. It is a
fixed 16-seed rank-normalized ensemble; every seed was retained,
so no member subset or blend weight was selected on validation.

## How we built it

- Python orchestration and subprocess isolation implement the experiment loop.
- NumPy and PyTorch provide the recommendation training primitives.
- OpenAI or Anthropic supplies the research/code-generation model behind one
  validated response contract.
- Streamlit and Graphviz render the live parent-linked experiment tree.
- The starter-kit evaluator remains unchanged and hidden-test labels are
  mechanically unavailable during research.

## What makes it different

The action space includes paired confirmation and fixed ensembling, not only
single training scripts. One seed can screen an idea but cannot change the
submission. Failures are classified by consequence: broken artifacts are
repaired, exhausted or inert approaches are abandoned, and timeouts pivot to a
materially cheaper experiment rather than being retried unchanged.

## Autonomy and robustness

The latest recorded run reports **0 manual
interventions**. The isolated full-loop suite recovered **3/
3** runtime, malformed-artifact, and timeout scenarios to a later
scored action using the real loop and executor. Its model is deterministic in
that evaluation so network or sampling cannot be mistaken for orchestration
quality.

The canonical artifact attribution is: **autonomous competition run (`AgentLoop` ensemble action)**.
This sentence changes automatically when an agent-produced competition artifact
becomes canonical.

## Resources and feasibility

The latest run used 27 training executions,
203602 provider-reported tokens, $0.718218
in model spend, and 2567.3 seconds wall-clock. Training is
CPU-capable and all limits are explicit before a run starts.

## Built with

**Development tools.** VS Code as the editor; git for version control, with
`git worktree` used to give each parallel agent worker an isolated checkout
(`agent/worktree.py`); Streamlit for the live run dashboard (`app.py`); the
standard `python3` toolchain, Python 3.12.10. Development was assisted by
Claude Code and Codex working in the repository.

**APIs.** One LLM provider API drives the agent loop: **openai `gpt-5.4`**, called
through `agent/llm.py`. The module also supports Anthropic as an alternative
provider, selected by `.env`. No other external API is used — no search, no
retrieval service, no hosted feature store.

**Libraries and frameworks.** **NumPy** is the substrate: the submitted model is
a factorization machine implemented in NumPy alone, matching the starter kit's
reference engine. **PyTorch** backs the optional sequential/neural branches of
the search space (`runtime/train_lib.py`). **Streamlit** renders the dashboard,
and the **openai** / **anthropic** SDKs handle provider calls. Exact pins are in
`requirements.txt`. Deliberately absent: no AutoML framework, no
hyperparameter-search library, no recommender toolkit — the search policy is the
contribution, so importing one would have replaced the thing being built.

**Datasets and assets.** **KuaiRand-Pure** only, exactly as distributed in the
organizers' starter kit — 1,141,112 train / 124,909 validation / 170,588 hidden
test rows on the fixed date-based splits, with `long_view` as the positive label.
No external training data, no pretrained weights, no augmentation, no manual
labelling. The starter kit's `baseline.py`, `data.py` and `evaluate.py` are used
unmodified; their SHA256 hashes are recorded in `logs/baseline/metrics.json` so
a judge can verify they were not edited.

## Challenges and lessons

The most important lesson was that a stricter internal convergence rule cannot
extend official checkpoint eligibility. We therefore separated research and
competition profiles and enforce the organizer's epsilon=0.002, N=3 rule during
the run. We also learned that crashes spend compute but create no evidence, and
that ensemble gains must be measured against the mean member, never the best of
many draws.

## Limitations

All development numbers are validation-only until the one-shot final command.
The model family is intentionally narrow, Path-B feature discovery has not yet
produced a confirmed Pure improvement, and the current ensemble reduces seed
variance rather than adding model diversity. We claim strong capability-transfer
autonomy, not independent discovery of an entirely new learning algorithm.

## Reproduce

```bash
python3 tests/test_harness.py
python3 -m agent.verify_incumbent
python3 -m agent.recovery_eval
streamlit run app.py
```

Generated from `results/manifest.json`; do not edit score or resource figures by
hand.
