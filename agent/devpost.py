"""Generate the Devpost submission narrative from the canonical manifest."""
from __future__ import annotations

import argparse
import os

from . import manifest as MF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "docs", "DEVPOST_SUBMISSION.md")


def build(d: dict) -> str:
    s = d.get("submitted") or {}
    score = (s.get("reported") or {}).get("primary")
    r = d.get("latest_run") or {}
    cl = ((d.get("robustness") or {}).get("closed_loop_recovery") or {})
    hp = s.get("how_produced") or {}
    ht = d.get("hidden_test") or {}
    htr = ht.get("result") or {}
    _t, _dl = (htr.get("test") or {}), (htr.get("delta_test") or {})
    # Deliverable 1 asks in as many words for tools, APIs, libraries and
    # datasets. Generate the result and the provider/model from the run's own
    # ledger so the writeup cannot quote a model the agent did not call.
    prov = (d.get("latest_run") or {}).get("llm_provider") or "(not recorded)"
    modl = (d.get("latest_run") or {}).get("llm_model") or "(not recorded)"
    headline = ("The hidden test has not been evaluated yet."
                if not _t else
                f"Scored once on the hidden test set: primary "
                f"**{_t.get('primary'):.5f}** against the official baseline's "
                f"{(htr.get('baseline_test') or {}).get('primary')}, an "
                f"absolute gain of **+{_dl.get('primary'):.4f}** "
                f"(GAUC +{_dl.get('GAUC'):.4f}, nDCG@5 "
                f"+{_dl.get('nDCG@5'):.4f}). The judged score is the mean "
                f"absolute delta across the two metrics: "
                f"**+{(_dl.get('GAUC', 0) + _dl.get('nDCG@5', 0)) / 2:.4f}**.")
    return f"""# Devpost Submission: Autonomous ML Research Agent

## Result

{headline}

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

Its current validation artifact scores **{score:.5f}**, a
**+{s.get('delta_vs_baseline'):.5f}** gain over the organizer baseline. It is a
fixed {s.get('members')}-seed rank-normalized ensemble; every seed was retained,
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

The latest recorded run reports **{r.get('manual_interventions')} manual
interventions**. The isolated full-loop suite recovered **{cl.get('recovered')}/
{cl.get('total')}** runtime, malformed-artifact, and timeout scenarios to a later
scored action using the real loop and executor. Its model is deterministic in
that evaluation so network or sampling cannot be mistaken for orchestration
quality.

The canonical artifact attribution is: **{hp.get('originally_built_by')}**.
This sentence changes automatically when an agent-produced competition artifact
becomes canonical.

## Resources and feasibility

The latest run used {r.get('training_runs_spent')} training executions,
{r.get('llm_tokens_total')} provider-reported tokens, ${r.get('llm_spend_usd')}
in model spend, and {r.get('runtime_agent_s')} seconds wall-clock. Training is
CPU-capable and all limits are explicit before a run starts.

## Built with

**Development tools.** VS Code as the editor; git for version control, with
`git worktree` used to give each parallel agent worker an isolated checkout
(`agent/worktree.py`); Streamlit for the live run dashboard (`app.py`); the
standard `python3` toolchain, Python 3.12.10. AI coding assistants were used
during development of the harness; the agent's own inference cost is
instrumented and reported separately in the resource figures.

**APIs.** One LLM provider API drives the agent loop: **{prov} `{modl}`**, called
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
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=MF.MANIFEST)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    d = MF.load(a.manifest)
    if d is None:
        raise SystemExit("manifest missing; run `python3 -m agent.manifest` first")
    text = build(d)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(text)
    print(f"wrote {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
