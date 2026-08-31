# Devpost Submission: Autonomous ML Research Agent

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

The canonical artifact attribution is: **human-invoked command (`agent.final_ensemble --seeds 16`)**.
This sentence changes automatically when an agent-produced competition artifact
becomes canonical.

## Resources and feasibility

The latest run used 28 training executions,
269807 provider-reported tokens, $0.913193
in model spend, and 2545.4 seconds wall-clock. Training is
CPU-capable and all limits are explicit before a run starts.

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
