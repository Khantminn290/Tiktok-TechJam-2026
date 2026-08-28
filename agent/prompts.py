"""Prompt builder — four sections, history shaped per action (AIDE-style):

(a) static task context, (b) the menu with priorities, (c) the decided action +
target + reason, (d) history: drafts see {hypothesis, menu_choices, score} summaries
only; improve sees full code of ONE node (its target); debug sees full code + error
trace of ONE node. Full code is never sent for more than the node being acted on.
"""
from __future__ import annotations

import json
import os

from .contracts import ExperimentTree, Node, error_headline

_HERE = os.path.dirname(os.path.abspath(__file__))
_API_MD = os.path.join(os.path.dirname(_HERE), "runtime", "API.md")

STATIC_CONTEXT = """You are the modeling brain of an autonomous ML research agent
competing on KuaiRand-Pure (short-video recommendation, within-user ranking).

TASK (fixed, do not reinterpret):
- Rank each user's logged impressions in the evaluation split. Positive label = long_view.
- Metrics: GAUC and nDCG@5 on the VALIDATION split; primary = mean of the two.
- Official baseline to beat: FM k=16 lr=1e-3, valid primary 0.6016 (GAUC 0.6674,
  nDCG@5 0.5357); 5-seed std 0.0008. Random scoring: 0.4834 valid. Oracle ceiling:
  0.8484 valid primary (27% of valid users have zero positives — nDCG can't reach 1).
  Judge progress against the ceiling: baseline has ~31% of the attainable range.
- The hidden test set is scored ONCE at the very end. You only ever see valid metrics.
- No external training data. Locked menu options are mechanically unselectable.

YOUR OUTPUT — exactly ONE JSON object, nothing else (no prose before or after):
{
  "hypothesis": "<what you try and WHY it should raise valid GAUC/nDCG@5 — judged text>",
  "menu_choices": {"loss": "...", "score_prior": "...", "user_history": "...", "multitask": "...",
                    "model": "...", "temporal": "...", "training": "...",
                    "data_extras": "..."},
  "code": "<the COMPLETE runnable python solution script — full file, not a diff>",
  "expected_effect": "<your quantitative expectation, e.g. '+0.003 primary from ...'>"
}

SOLUTION SCRIPT CONTRACT (your "code" must satisfy all of this):
- CLI: accepts --menu-choices '<json>' and --output-dir <path> (and optional --seed).
- On success: writes metrics.json ({"GAUC","nDCG@5","primary"} on VALID),
  scores_valid.npy and scores_test.npy (row_id-aligned, one float per split row)
  into --output-dir, exits 0. On failure: non-zero exit, readable stderr.
- Score with the official evaluate (import from train_lib) — NEVER reimplement metrics.
- NEVER read test labels or compute test metrics. Early stopping uses valid only.
- Runtime must stay under 20 minutes (numpy FM baseline ≈ 1 min).
- The simplest valid script is seed_solution.py: parse args, call
  train_lib.run(menu_choices, output_dir, seed). Use custom code (train_lib Path B)
  only when your hypothesis needs something the menu-driven path can't express;
  custom code must still start from the documented train_lib building blocks.
"""


def _summarize_node(n: Node) -> str:
    if n.status == "success" and n.metrics:
        score = (f"valid primary {n.metrics['primary']:.4f} "
                 f"(GAUC {n.metrics['GAUC']:.4f}, nDCG@5 {n.metrics['nDCG@5']:.4f})")
    else:
        score = f"ERROR: {error_headline(n.error_trace)}"
    return (f"- node {n.iteration_id} [{n.action}"
            f"{'' if n.parent_id is None else f'<-{n.parent_id}'}] "
            f"choices={json.dumps(n.menu_choices)} | {score}\n"
            f"  hypothesis: {n.hypothesis[:220]}")


def _read_code(n: Node) -> str:
    try:
        with open(n.code_path) as fh:
            return fh.read()
    except OSError:
        return "<code file missing>"


def build_prompt(action: str, target: Node | None, reason: str,
                 tree: ExperimentTree, menu) -> str:
    parts = [STATIC_CONTEXT]

    with open(_API_MD) as fh:
        parts.append("## train_lib API available to your script\n" + fh.read())

    parts.append("## Modification menu (pick exactly one option per axis)\n"
                 + menu.render_for_prompt())

    parts.append(f"## Current action (decided by the search policy)\n"
                 f"action: {action}\nreason: {reason}")

    history = [_summarize_node(n) for n in tree.nodes]
    if history:
        parts.append("## History (all attempts so far — summaries only)\n"
                     + "\n".join(history))
    best = tree.best()
    if best is not None:
        parts.append(f"Current best: node {best.iteration_id}, valid primary "
                     f"{best.metrics['primary']:.4f}, choices "
                     f"{json.dumps(best.menu_choices)}")

    if action == "draft":
        parts.append(
            "## Instructions\nPropose a FRESH combination not attempted yet. "
            "Bias toward the highest-priority unexplored axes (loss, score_prior, "
            "and user_history). One clear "
            "hypothesis per draft — do not change every axis at once.")
    elif action == "improve":
        parts.append(
            f"## Target node {target.iteration_id} (the ONE node you are improving)\n"
            f"menu_choices: {json.dumps(target.menu_choices)}\n"
            f"valid metrics: {json.dumps(target.metrics)}\n"
            f"hypothesis was: {target.hypothesis}\n"
            f"### Its full code\n```python\n{_read_code(target)}\n```\n"
            "## Instructions\nPropose ONE focused improvement to this solution — a "
            "menu-axis change, a hyperparameter change, or custom code via train_lib "
            "Path B. Explain in the hypothesis why THIS change should add on top of "
            "what the target already does. Return the complete new script.")
    elif action == "crossover":
        from .policy import crossover_partner
        partner = crossover_partner(tree, target)
        parts.append(
            f"## Crossover — combine TWO successful lineages\n"
            f"Both single-lineage directions are exhausted, so this move builds a "
            f"configuration neither parent can reach by extension alone.\n\n"
            f"### Parent A — node {target.iteration_id} "
            f"(valid primary {target.metrics['primary']:.4f})\n"
            f"menu_choices: {json.dumps(target.menu_choices)}\n"
            f"hypothesis was: {target.hypothesis}\n"
            f"### Its full code\n```python\n{_read_code(target)}\n```\n")
        if partner is not None:
            parts.append(
                f"### Parent B — node {partner.iteration_id} "
                f"(valid primary {partner.metrics['primary']:.4f})\n"
                f"menu_choices: {json.dumps(partner.menu_choices)}\n"
                f"hypothesis was: {partner.hypothesis}\n"
                f"expected_effect was: {partner.expected_effect}\n"
                f"(Parent B's source is deliberately omitted — take its *ideas* and "
                f"menu choices, and write the combined script yourself.)")
        parts.append(
            "## Instructions\nProduce ONE new solution whose menu_choices take the "
            "winning elements of BOTH parents — not a copy of either. Say in the "
            "hypothesis which axis you took from which parent and why those two "
            "changes should compose rather than conflict (e.g. a ranking loss from "
            "one and a history feature from the other act on different parts of the "
            "model). If you believe two choices actively fight each other, say so "
            "and pick the one you expect to dominate. Return the complete script.")
    elif action == "debug":
        parts.append(
            f"## Target node {target.iteration_id} (the failed attempt to fix)\n"
            f"menu_choices: {json.dumps(target.menu_choices)}\n"
            f"hypothesis was: {target.hypothesis}\n"
            f"### Its full code\n```python\n{_read_code(target)}\n```\n"
            f"### Error trace\n```\n{(target.error_trace or '')[-4000:]}\n```\n"
            "## Instructions\nDiagnose the error and return a FIXED complete script. "
            "Keep the original intent (same menu_choices unless the choices "
            "themselves caused the failure). State the root cause in the hypothesis.")
    return "\n\n".join(parts)
