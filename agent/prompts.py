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
from .experience import render_for_prompt as render_experience

_HERE = os.path.dirname(os.path.abspath(__file__))
_API_MD = os.path.join(os.path.dirname(_HERE), "runtime", "API.md")

CANDIDATE_SECTION = (
    "## Propose SEVERAL candidate experiments, not one\n"
    "Return a JSON object with a `candidates` list of {n} entries. They are "
    "scored deterministically by the harness and ONE is selected; you are "
    "generating the option set, not the final answer, so genuinely different "
    "ideas are worth more than {n} variations of the same one.\n"
    "At least one candidate MUST use implementation_path 'B' -- IMPLEMENT: a mechanism the "
    "menu cannot express) IF you can state a real mechanism for it. If you "
    "honestly cannot, say so in that candidate's hypothesis and mark it path "
    "'A' -- a fabricated Path B idea is worse than an honest Path A one, and "
    "the scorer will reject an unfounded candidate anyway.\n"
    "Each candidate needs: hypothesis, implementation_path, research_category, "
    "mechanism (WHY this should move GAUC/nDCG@5 on THIS dataset), "
    "expected_gain (a number, in primary-score units -- the seed noise floor "
    "is 0.0008, so be honest about magnitude), falsification (what result "
    "would disprove it), menu_choices (path A) or code_summary (path B).\n"
    "\n"
    "## Start from what you do not understand, not from what you can run\n"
    "Before the candidate list, return an `inquiry` object. The point is to stop "
    "the reasoning 'I have a feature tool, so I should try a feature' and replace "
    "it with 'I observed X, I cannot explain Y, and measurement M would tell "
    "them apart'.\n"
    "  \"inquiry\": {\n"
    "    \"observation\": \"<something in the evidence that is surprising, "
    "unexplained, or inconsistent -- quote the numbers>\",\n"
    "    \"question\": \"<the thing you actually do not know>\",\n"
    "    \"hypotheses\": [\"<explanation A>\", \"<explanation B>\", "
    "\"<explanation C>\"],\n"
    "    \"discriminating_measurement\": \"<what would tell them apart, and "
    "which tool or experiment provides it>\",\n"
    "    \"expected_outcome_per_hypothesis\": \"<what each explanation predicts "
    "that measurement would show>\",\n"
    "    \"resolves_uncertainty\": \"<what you would DO differently depending on "
    "the answer -- if nothing, this is the wrong question>\",\n"
    "    \"capability_required\": \"<the exact capability from the CAPABILITY "
    "CONTRACT this measurement needs, by name>\",\n"
    "    \"why_this_capability\": \"<why that one rather than a cheaper or more "
    "expensive one>\",\n"
    "    \"cost\": \"<free | cheap | one_training_run -- from the contract>\",\n"
    "    \"promotion_criterion\": \"<the result that would make you act on this, "
    "stated BEFORE you see it: effect size, seeds, and what state it must reach. "
    "Remember a single seed can only ever be PRELIMINARY>\"\n"
    "  }\n"
    "Give at least two competing hypotheses. A single explanation you already "
    "believe is not a question. And prefer the measurement whose answer would "
    "most change what you do next, which is NOT always the experiment with the "
    "highest expected score -- a cheap diagnostic that rules out three "
    "explanations can be worth more than another configuration.\n"
    "Name the capability by its contract name and check where it can be "
    "invoked BEFORE you plan around it: an orchestration-only capability does "
    "not exist inside a generated script, and planning to call one there wastes "
    "the attempt. State the promotion criterion in advance -- deciding after "
    "the fact what counts as success is how a noisy draw becomes a discovery.\n"
    "Only the SELECTED candidate is implemented, so do not write code yet.")

AXIS_PROPOSAL_SECTION = (
    "## Proposing a new axis\n"
    "If the data measurements or the dead-end history point at a category of "
    "intervention this menu cannot express, you may attach a `proposed_axis` "
    "object to your response (schema in the task context above). It is "
    "recorded as PENDING for human approval and is NOT usable this iteration, "
    "so still make a normal, valid menu_choices proposal as well. Note the "
    "measured pattern on this dataset: interventions that CONCENTRATE or "
    "reweight the training signal have lost every time, and broad additive "
    "signal has won -- state which yours is, honestly.")

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
  "implementation_path": "A" | "B",
  "research_category": "exploration" | "exploitation" | "ablation"
                       | "confirmation" | "integration",
  // Path A ONLY -- one option per axis:
  "menu_choices": {"loss": "...", "user_history": "...", ...},
  // Path B ONLY -- instead of menu_choices:
  "code_summary": "<the mechanism you implemented, and why the existing menu
                    primitives cannot express it>",
  "code": "<the COMPLETE runnable python solution script — full file, not a diff>",
  "expected_effect": "<your quantitative expectation, e.g. '+0.003 primary from ...'>",
  "rationale": {
    "idea": "<one sentence: what this iteration changes, concretely>",
    "why_expected_to_help": "<the mechanism — why THIS change should move GAUC/nDCG@5>",
    "grounded_in": "<REQUIRED, specific: quote/name the menu axis+option description you
      are building on (e.g. 'loss axis: organizers rank ranking-aligned loss as the top
      unexplored direction'), a number from the Modification menu's tested_dead_ends /
      baseline_scores.json, or a named paper/method (e.g. 'BPR', 'DIN', 'CWM'). A generic
      'general ML intuition' answer will be rejected and cost you a retry.>"
  }
}

OPTIONAL: you may also propose a genuinely NEW menu axis by adding a
"proposed_axis" key. Use it only when the data or history suggests a category
of intervention the menu cannot currently express -- not to restate an
existing axis. It is recorded as PENDING and a human must approve it before it
becomes selectable, so proposing one does NOT let you use it this iteration.
  "proposed_axis": {"axis_name": "lower_snake_case", "description": "...",
     "options": {"baseline_noop": {"description": "..."},
                 "variant": {"description": "..."}},
     "mechanism": "why this should change GAUC/nDCG@5, concretely",
     "citation": "a real paper, or a measurement from this run",
     "signal_breadth": "broad" | "concentrated"}

SOLUTION SCRIPT CONTRACT (your "code" must satisfy all of this):
- CLI: accepts --menu-choices '<json>' and --output-dir <path> (and optional --seed).
- On success: writes metrics.json ({"GAUC","nDCG@5","primary"} on VALID),
  scores_valid.npy and scores_test.npy (row_id-aligned, one float per split row)
  into --output-dir, exits 0. On failure: non-zero exit, readable stderr.
- Score with the official evaluate (import from train_lib) — NEVER reimplement metrics.
- NEVER read test labels or compute test metrics. Early stopping uses valid only.

CHOOSING AN IMPLEMENTATION PATH — decide from your HYPOTHESIS, not from which
path looks safer or more normal. Neither is the default; they are different
tools.
WHAT YOU CAN DO. One action space, not two tiers. Every experiment is one of
these, and `implementation_path` only records HOW you implemented it:

- CONFIGURE (implementation_path "A") — the hypothesis is about a mechanism the
  pipeline ALREADY expresses: a loss, an architecture, a history mechanism, a
  hyperparameter, an ablation, or a combination of validated components. Give
  menu_choices; your script can simply call
  train_lib.run(menu_choices, output_dir, seed).

- IMPLEMENT (implementation_path "B") — the hypothesis needs a mechanism the
  menu CANNOT express: a new training objective, a new data representation, a
  new feature transformation, a different way of forming training examples, a
  new ranking or aggregation strategy. Give code_summary instead of
  menu_choices and build it from train_lib primitives (load_cache,
  encode_features, RankFM, evaluate, ...).

Neither is the default and neither is braver. Choosing IMPLEMENT for something
the menu already covers wastes an iteration; refusing to implement a mechanism
the menu cannot express means the question never gets asked. Always ask: what is
the SIMPLEST experiment that could test this hypothesis?

Two further actions are scheduled FOR you when the evidence calls for them, so
you do not implement them yourself -- but you should say in your hypothesis when
you think one is due:

- CONFIRM — a paired multi-seed experiment. A single-seed result is
  PRELIMINARY at any effect size, and this is the only thing that can make it
  CONFIRMED.
- ENSEMBLE — train k seeds of a confirmed configuration and average their
  rank-normalised predictions. This does not make any model better; it removes
  seed variance from the number we submit. Its value is measured against the
  MEAN member, never the best one.
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


def _compute_budget_section(exec_timeout_s: int) -> str:
    """Dynamic, not a hardcoded '20 minutes' string: reflects whatever
    --exec-timeout the run was actually started with, and names the specific
    failure mode a real run hit -- three independent workers all proposed
    reimplementing multi-seed ensembling inside one script and all three timed
    out, because nothing told them exec_timeout_s is a hard, no-partial-credit
    ceiling OR that agent.reseed already measures seed-variance properly,
    after the search, for free.
    """
    mins = exec_timeout_s / 60
    return (
        f"## Compute budget (hard constraint, not a suggestion)\n"
        f"Your script is KILLED if it runs longer than {exec_timeout_s}s "
        f"(~{mins:.0f} min) wall-clock, with NO partial credit -- a timeout is "
        f"scored exactly like a crash, not like a partial success. A single "
        f"training pass on the menu-driven path (Path A) typically takes well "
        f"under this (numpy FM: tens of seconds to a few minutes; torch models "
        f"are capped at 12 epochs). If your idea needs MULTIPLE full training "
        f"passes inside one script, you must budget explicitly: estimate one "
        f"pass's cost from the most similar past node's wall-clock time (see "
        f"History below) and confirm N passes still fits under {exec_timeout_s}s "
        f"BEFORE writing the script -- do not find out by timing out.\n\n"
        f"Do NOT propose multi-seed averaging/ensembling as your idea: that is "
        f"already measured, correctly and for free, by the harness's own "
        f"`--reseed-top` mechanism AFTER the search converges, run once against "
        f"the best few nodes -- not by any single iteration re-implementing it "
        f"under a fraction of that time budget. A node should train ONE "
        f"configuration ONCE. If your hypothesis is specifically about "
        f"robustness to seed variance, a single iteration cannot validate that "
        f"anyway -- propose a different, single-pass idea instead.")


def render_sibling_section(sibling_choices: list) -> str:
    """The K-way diversity mechanism for parallel rounds.

    Measured failure this was written for: with --parallel-k 3 against a
    well-constrained prompt (14 recorded dead-ends), all three workers
    independently proposed the IDENTICAL configuration and scored identically
    -- 3x the cost for 1x the information. iterate_parallel() issues K calls
    against the same prompt and relied on sampling temperature for diversity,
    which is not a mechanism once the prompt narrows the plausible choice set
    to one "best remaining" option.

    Conditioning each worker on its siblings' already-made proposals is
    preferred over assigning each worker a fixed axis, for two reasons: the K
    LLM calls in a round are ALREADY sequential (only training is
    parallelised), so this costs no wall-clock; and it preserves the agent's
    freedom to choose its own hypothesis, dictating only that it not duplicate
    a sibling -- which is the capability a parallel round is supposed to show.
    """
    if not sibling_choices:
        return ""
    lines = ["## Sibling proposals ALREADY MADE in this same round",
             "Other workers are exploring these configurations RIGHT NOW, in "
             "parallel with you:"]
    for i, ch in enumerate(sibling_choices):
        lines.append(f"- worker {i}: {json.dumps(ch, sort_keys=True)}")
    lines.append(
        "\nYou MUST propose something MEANINGFULLY DIFFERENT from every one of "
        "them. Returning the same menu_choices as a sibling wastes an entire "
        "training run on a duplicate experiment and tells us nothing new. "
        "Differ on at least one axis, and prefer differing on an axis none of "
        "them touched -- the point of running workers in parallel is to cover "
        "MORE of the search space per round, not to re-run one idea K times. "
        "If you believe every remaining option is exhausted, say so explicitly "
        "in your hypothesis and pick the least-explored axis anyway.")
    return "\n".join(lines)


def build_prompt(action: str, target: Node | None, reason: str,
                 tree: ExperimentTree, menu, exec_timeout_s: int = 1200,
                 sibling_choices: list | None = None,
                 data_block: str = "", compact_menu: bool = False) -> str:
    parts = [STATIC_CONTEXT, _compute_budget_section(exec_timeout_s)]
    if data_block:
        parts.append(data_block)
    try:
        from .propose_axis import render_for_prompt as _axes
        _prev = _axes()
    except Exception:
        _prev = ""
    parts.append(AXIS_PROPOSAL_SECTION + ("\n" + _prev if _prev else ""))
    if sibling_choices:
        parts.append(render_sibling_section(sibling_choices))

    with open(_API_MD) as fh:
        parts.append("## train_lib API available to your script\n" + fh.read())

    # Menu volume was measured at 20,863 chars vs 2,906 for Path B guidance
    # (7.2x). That imbalance is itself a cause of menu-shaped thinking, so the
    # full menu is sent only when a detailed Path A selection is actually
    # likely; exploration turns get a compact index plus the dead-ends (which
    # must never be dropped -- they are what prevents re-deriving known nulls).
    if compact_menu:
        parts.append("## Modification menu (COMPACT INDEX -- axes and options "
                     "only)\nIf your hypothesis needs the full option "
                     "descriptions to make a precise Path A selection, say so "
                     "in your hypothesis and choose a menu option you already "
                     "understand.\n" + menu.render_compact()
                     + "\n\n" + menu.render_dead_ends())
    else:
        parts.append("## Modification menu (pick exactly one option per axis)\n"
                     + menu.render_for_prompt())

    parts.append(
        "## Experience memory (curated lessons from past iterations -- read this "
        "before proposing something that already crashed or was a measured dead "
        "end; it is NOT the same as the full History section below)\n"
        + render_experience())

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
            "The organizers' own top-ranked directions are the loss function and "
            "user_history, and those are good priors — but priority order is a "
            "prior, not a restriction, and an axis nothing has actually TRIED yet "
            "is worth more than another small variation on an axis already "
            "explored. Check the History below for which axes have real measured "
            "results and which are still untouched; the model/architecture axis in "
            "particular is open and untested (the measured dead-end there is about "
            "embedding capacity, NOT about architecture — see its axis description). "
            "One clear hypothesis per draft — do not change every axis at once.")
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
        try:
            from .failure import classify, fingerprint, repair_brief, repeat_count
            _fc = classify(target.error_trace)
            # Has this EXACT fault already happened in this run? Each repair
            # prompt is built fresh, so without this the model cannot tell a
            # first attempt from a fourth identical one -- the loop that cost
            # run 3 three of its six iterations.
            _fp = fingerprint(_fc, target.error_trace)
            _prev = [(n.status == "error" and classify(n.error_trace)["class"],
                      n.error_trace)
                     for n in tree.nodes
                     if n.status == "error" and n.iteration_id != target.iteration_id]
            parts.append(repair_brief(_fc, attempt=1, max_attempts=2,
                                      repeats=repeat_count(_fp, _prev)))
        except Exception:
            pass
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


def build_candidate_prompt(action, target, reason, tree, menu, *, n=4,
                          exec_timeout_s=1200, data_block="", objective=None):
    """Phase 1 of the two-phase planner: generate a scoreable OPTION SET.

    Deliberately ONE call producing n candidates rather than n calls: the audit
    found Path B was never generated (not rejected), and the cheapest way to
    put it on the table is to ask for alternatives explicitly. Extra cost is
    output tokens for one call, not n× the calls.
    """
    parts = [STATIC_CONTEXT, _compute_budget_section(exec_timeout_s)]
    if data_block:
        parts.append(data_block)
    parts.append("## Modification menu (compact index)\n" + menu.render_compact()
                 + "\n\n" + menu.render_dead_ends())
    parts.append(f"## Decided action\naction: {action}\nreason: {reason}")
    if objective:
        parts.append(f"## Research objective for this iteration: {objective}\n"
                     f"Prefer candidates in this category; you may include one "
                     f"outside it if you justify why the objective is wrong.")
    hist = "\n".join(
        f"- node {n_.iteration_id} [{n_.action}] "
        f"{('%.4f' % n_.metrics['primary']) if n_.metrics else 'ERROR'} "
        f"{json.dumps(n_.menu_choices)}" for n_ in tree.nodes[-8:])
    parts.append("## Recent attempts\n" + (hist or "(none)"))
    parts.append(CANDIDATE_SECTION.replace("{n}", str(n)))
    return "\n\n".join(parts)


def build_merge_prompt(a: Node, b: Node, reason: str, menu,
                       exec_timeout_s: int = 1200) -> str:
    """Coordinator merge prompt (Phase 3 item 3 Part B): two SIBLING candidates
    from the same parallel round, both of which already beat the running best.

    Unlike "improve"/"crossover" (which withhold an older parent's code to
    encourage a fresh combination rather than a copy), BOTH candidates here
    get their full code: neither is a "boring, already-explored parent" --
    both are fresh, comparably-strong ideas, and a diff of either against
    their shared ancestor would force the model to mentally reconstruct two
    full scripts before it could even start reasoning about combining them.
    """
    parts = [STATIC_CONTEXT, _compute_budget_section(exec_timeout_s)]
    with open(_API_MD) as fh:
        parts.append("## train_lib API available to your script\n" + fh.read())
    parts.append("## Modification menu (pick exactly one option per axis)\n"
                 + menu.render_for_prompt())
    parts.append(
        "## Coordinator merge task\n"
        f"Two independent workers were given the SAME task this round ({reason}), "
        f"and each produced a DIFFERENT candidate that beat the running best. Your "
        f"job is to write ONE new script that combines their distinct, complementary "
        f"ideas into something that should score HIGHER than either alone -- not to "
        f"pick one, and not to average their outputs. If the two ideas fundamentally "
        f"conflict (e.g. both change the same axis incompatibly), say so in the "
        f"hypothesis, pick whichever you expect to dominate on that axis, and still "
        f"combine whatever from the other candidate doesn't conflict with it.\n\n"
        f"### Candidate A -- node {a.iteration_id} "
        f"(valid primary {a.metrics['primary']:.4f})\n"
        f"menu_choices: {json.dumps(a.menu_choices)}\n"
        f"hypothesis was: {a.hypothesis}\n"
        f"### Its full code\n```python\n{_read_code(a)}\n```\n\n"
        f"### Candidate B -- node {b.iteration_id} "
        f"(valid primary {b.metrics['primary']:.4f})\n"
        f"menu_choices: {json.dumps(b.menu_choices)}\n"
        f"hypothesis was: {b.hypothesis}\n"
        f"### Its full code\n```python\n{_read_code(b)}\n```\n\n"
        "## Instructions\nReturn the complete merged script. In the hypothesis, name "
        "exactly which element you took from A, which from B, and why they should "
        "compose rather than conflict. Your rationale.grounded_in should cite what "
        "A's and B's own grounding already established, not invent a new citation.")
    return "\n\n".join(parts)
