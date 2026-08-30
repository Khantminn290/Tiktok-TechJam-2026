"""Harness side of the agent's data-inspection phase.

Two-phase design: the agent is a single structured-JSON call, so it cannot
invoke tools mid-generation. Phase 1 asks it which measurements it wants,
the harness executes them behind the sandbox, and phase 2 gives it the
results to hypothesize from. Provider-agnostic -- no vendor tool-calling API.

Budgeted on purpose: an agent that profiles forever never proposes anything,
so MAX_TOOL_CALLS caps how much of an iteration can go to inspection.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
RUNTIME = os.path.join(ROOT, "runtime")
if RUNTIME not in sys.path:
    sys.path.insert(0, RUNTIME)

import data_tools  # noqa: E402

MAX_TOOL_CALLS = 4          # per iteration
SANDBOX_CACHE = os.path.join(RUNTIME, "cache_sandbox")

# The earlier wording said only that profiling is not free and to prefer an
# empty list. Measured consequence: across two clean autonomy runs the agent
# requested ZERO tools in 10 of 10 iterations -- it was doing what it was told,
# and so never ran the measurement that would show which part of the pipeline
# was binding. The guidance is now symmetric: cheap when you do not know what is
# binding, skip when you do.
INSPECT_SCHEMA_HINT = """Respond with exactly ONE JSON object:
{"requests": [{"tool": "<name>", "args": {...}}, ...]}
Ask for at most %d measurements.

WHEN TO MEASURE. A measurement is worth an iteration's fraction when you do not
know WHICH PART of the pipeline is limiting the score -- for example when many
different configurations keep landing within noise of each other, which means
the thing that matters is not the option you are choosing. Most of these
diagnostics are free (no training); only training_dynamics costs a run.

WHEN NOT TO. Skip if the history already answers your question, or if you are
confirming a specific hypothesis you can test directly. An empty list is a fine
answer then.""" % MAX_TOOL_CALLS


def build_inspect_prompt(menu, tree, experience_text: str) -> str:
    from .prompts import STATIC_CONTEXT
    recent = tree.nodes[-6:]
    hist = "\n".join(
        f"- node {n.iteration_id} [{n.action}] "
        f"{('primary %.4f' % n.metrics['primary']) if n.metrics else 'ERROR'} "
        f"{json.dumps(n.menu_choices)}" for n in recent) or "(no attempts yet)"
    return "\n\n".join([
        STATIC_CONTEXT,
        "## You may inspect the DATA before deciding what to try\n"
        "These read-only measurements run against the sandboxed train/valid "
        "splits. Use them to ground your next hypothesis in what the data "
        "actually looks like rather than in assumption.\n\n"
        + data_tools.describe_tools() + describe_diagnostics(),
        "## Recent attempts\n" + hist,
        "## Lessons already learned (do not re-derive these)\n" + experience_text,
        "## Measured dead ends\n" + menu.render_for_prompt()[-3000:],
        INSPECT_SCHEMA_HINT,
    ])


def parse_requests(obj) -> list:
    """Validate the phase-1 response into a bounded, safe request list."""
    if not isinstance(obj, dict):
        return []
    reqs = obj.get("requests") or []
    if not isinstance(reqs, list):
        return []
    # Deduplicate BEFORE applying the cap. Observed in real runs: the model
    # asks for get_within_user_auc with identical args three times in one
    # iteration, spending 3 of its 4 tool calls to receive the same number
    # three times. These tools are deterministic reads of a fixed cache, so a
    # repeat cannot return anything new -- dropping it costs no information and
    # frees the budget for a genuinely different measurement.
    out, seen = [], set()
    for r in reqs:
        if not isinstance(r, dict):
            continue
        name = r.get("tool")
        args = r.get("args") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            continue
        key = (name, json.dumps(args, sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append({"tool": name, "args": args})
        if len(out) >= MAX_TOOL_CALLS:
            break
    return out


# ---------------------------------------------------------------------------
# PIPELINE DIAGNOSTICS -- run in-process, not in the data sandbox.
#
# Added because the clean autonomy test failed for a diagnosable reason: the
# capabilities were DESCRIBED in the planning prompt but there was no way to
# CALL them. The agent could set configuration but could not run the
# measurement that would tell it whether configuration was even the problem.
# A capability the agent cannot invoke is documentation, not an action space.
# ---------------------------------------------------------------------------
def _tool_hardcoded_constants(**_kw):
    from .pipeline_lab import hardcoded_constants
    return {"constants": hardcoded_constants()}


def _tool_selection_pressure(n: int = 5, **_kw):
    from .validity import selection_pressure
    return selection_pressure(int(n))


def _tool_audit_comparison(**kw):
    from .validity import audit_comparison
    allowed = ("delta", "n_seeds", "paired", "n_candidates_compared",
               "selected_on_eval_data", "confirmed_out_of_sample")
    return audit_comparison(**{k: v for k, v in kw.items() if k in allowed})


def _tool_training_dynamics(max_epochs: int = 40, **_kw):
    """EXPENSIVE: one full training run with early stopping disabled.

    Capped at once per run by the caller. It is the only way to see whether a
    model is over- or under-fitting and where its epoch curve peaks, which no
    score can reveal.
    """
    from .pipeline_lab import training_dynamics
    r = training_dynamics(seeds=(0,), max_epochs=int(max_epochs))
    s = r.get("seeds", {}).get(0, {})
    return {"peak_epoch": s.get("peak_epoch"), "peak_primary": s.get("peak_primary"),
            "final_primary": s.get("final_primary"),
            "change_after_peak_sigma": s.get("decline_sigma"),
            "verdict": r.get("verdict"),
            "curve_tail": (s.get("curve") or [])[-8:]}


def _tool_selection_rule_test(max_epochs: int = 40, **_kw):
    """EXPENSIVE: does the way we CHOOSE a checkpoint generalise?

    Registered because a clean-run trace showed the agent forming exactly this
    question -- "the pipeline chooses the best checkpoint using the same
    validation users it reports on" -- naming this tool as the discriminating
    measurement, and then being unable to call it. It fell back to an ordinary
    training run, which cannot answer the question. The reasoning was sound;
    the plumbing was missing.

    Compares GENERAL choosing rules. Which rule wins is measured here, not
    assumed.
    """
    import numpy as np
    from .pipeline_lab import selection_rule_test
    from .ensemble import load_valid_targets
    from agent import research_run as RR
    import train_lib

    splits, meta = train_lib.load_cache()
    base, enc = RR.incumbent_cfg(splits, meta)
    cfg = dict(base)
    cfg.update(seed=0, epochs=int(max_epochs), patience=int(max_epochs))
    cap: list = []
    cfg["capture_epoch_scores"] = cap
    train_lib.train_numpy_fm(cfg, enc, splits, meta, lambda *a, **k: None)
    if len(cap) < 5:
        return {"error": "not enough epochs captured"}
    E = np.asarray([[sv for _e, _p, sv in cap]])          # 1 seed x epochs x rows
    users, labels = load_valid_targets()

    def _rank(x):
        o = np.argsort(x, kind="stable")
        r = np.empty(len(x), dtype=np.float64)
        r[o] = np.arange(len(x), dtype=np.float64)
        return r / max(1, len(x) - 1)

    rules = {
        "argmax": lambda p, e: e[int(np.argmax(p))],
        "avg_top3": lambda p, e: np.mean([_rank(e[i]) for i in np.argsort(-p)[:3]], axis=0),
        "avg_top5": lambda p, e: np.mean([_rank(e[i]) for i in np.argsort(-p)[:5]], axis=0),
        "median_top5": lambda p, e: np.median([_rank(e[i]) for i in np.argsort(-p)[:5]], axis=0),
    }
    r = selection_rule_test(E, users, labels, rules, n_splits=3)
    return {"reference_rule": r["reference_rule"],
            "held_out_evaluations": r["n_evaluations"],
            "rules": r["rules"],
            "note": "positive sigma means the rule generalises BETTER than the "
                    "reference on users it was not chosen on"}


def _tool_free_recombination(n_subsets: int = 12, **_kw):
    """FREE: compare ways of combining the stored ensemble members.

    No training. Resamples member subsets so a rule must win repeatedly.
    """
    import json as _json
    import numpy as np
    from .pipeline_lab import free_recombination
    from .ensemble import load_valid_targets, rank_normalise

    res_p = os.path.join(ROOT, "logs", "ensemble_results.json")
    if not os.path.exists(res_p):
        return {"error": "no stored ensemble to recombine"}
    res = _json.load(open(res_p))
    mdir = os.path.join(ROOT, res.get("members_dir", ""))
    M = []
    for i in res.get("seeds_used", []):
        f = os.path.join(mdir, f"seed_{i:02d}", "scores_valid.npy")
        if os.path.exists(f):
            M.append(rank_normalise(np.load(f)))
    if len(M) < 4:
        return {"error": f"only {len(M)} members available"}
    users, labels = load_valid_targets()
    return free_recombination(
        np.stack(M), users, labels,
        {"mean": lambda m: m.mean(axis=0),
         "median": lambda m: np.median(m, axis=0),
         "trimmed_mean": lambda m: np.sort(m, axis=0)[1:-1].mean(axis=0)},
        n_subsets=int(n_subsets), subset=min(8, len(M)))


DIAGNOSTIC_TOOLS = {
    "selection_rule_test": _tool_selection_rule_test,
    "free_recombination": _tool_free_recombination,
    "hardcoded_constants": _tool_hardcoded_constants,
    "selection_pressure": _tool_selection_pressure,
    "audit_comparison": _tool_audit_comparison,
    "training_dynamics": _tool_training_dynamics,
}
EXPENSIVE_TOOLS = {"training_dynamics", "selection_rule_test"}


def describe_diagnostics() -> str:
    return (
        "\nPIPELINE DIAGNOSTICS (about the MODEL, not the data):\n"
        "- training_dynamics(max_epochs=40): trains once with early stopping "
        "DISABLED and returns the epoch curve, its peak, and how far validation "
        "moves after it. The only way to tell over- from under-fitting. "
        "EXPENSIVE (~1 training run); allowed at most once per run.\n"
        "- hardcoded_constants(): modelling constants written into the training "
        "library that no menu option can reach, and whether you can set each.\n"
        "- selection_rule_test(max_epochs=40): trains once capturing every "
        "epoch, then compares CHOOSING rules by selecting on one half of the "
        "validation users and scoring on the other. Use it whenever a choice is "
        "being made using the same data it will be judged on. EXPENSIVE.\n"
        "- free_recombination(n_subsets=12): compares ways of combining the "
        "stored ensemble members by resampling subsets. FREE, no training.\n"
        "- selection_pressure(n): how large an apparent gain appears purely from "
        "picking the best of n noisy comparisons.\n"
        "- audit_comparison(delta, n_seeds, paired, n_candidates_compared, "
        "selected_on_eval_data, confirmed_out_of_sample): what a claimed "
        "improvement is worth given how it was measured.\n")


def execute(requests: list, cache_dir: str | None = None) -> list:
    """Run validated requests against the SANDBOXED cache. Never raises: a bad
    request becomes a readable error the agent can learn from, exactly like a
    rejected menu choice."""
    cache_dir = cache_dir or (SANDBOX_CACHE if os.path.exists(
        os.path.join(SANDBOX_CACHE, "meta.json")) else None)
    results = []
    used_expensive = False
    for r in requests[:MAX_TOOL_CALLS]:
        try:
            name = r["tool"]
            if name in DIAGNOSTIC_TOOLS:
                if name in EXPENSIVE_TOOLS:
                    if used_expensive:
                        results.append({"request": r,
                                        "error": f"{name} is expensive and is "
                                                 f"allowed at most once per "
                                                 f"iteration"})
                        continue
                    used_expensive = True
                results.append({"request": r,
                                "result": DIAGNOSTIC_TOOLS[name](**(r["args"] or {}))})
                continue
            results.append({"request": r,
                            "result": data_tools.run_tool(r["tool"], r["args"],
                                                          cache_dir=cache_dir)})
        except data_tools.ToolError as e:
            results.append({"request": r, "error": str(e)[:400]})
        except Exception as e:                      # never kill an iteration
            results.append({"request": r,
                            "error": f"{type(e).__name__}: {str(e)[:300]}"})
    return results


def render_results(results: list) -> str:
    if not results:
        return ""
    lines = ["## Data measurements you requested (real numbers from this dataset)"]
    for r in results:
        if "error" in r:
            lines.append(f"- {json.dumps(r['request'])} -> ERROR: {r['error']}")
        else:
            lines.append(f"- {json.dumps(r['request'])}\n  {json.dumps(r['result'])}")
    lines.append("Ground your hypothesis in these numbers where relevant, and "
                 "say so in rationale.grounded_in.")
    return "\n".join(lines)
