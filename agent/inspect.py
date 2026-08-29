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

INSPECT_SCHEMA_HINT = """Respond with exactly ONE JSON object:
{"requests": [{"tool": "<name>", "args": {...}}, ...]}
Ask for at most %d measurements. Ask for NOTHING (an empty list) if the
history already tells you what you need -- profiling is not free, and an
iteration spent measuring is an iteration not spent testing an idea.""" % MAX_TOOL_CALLS


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
        + data_tools.describe_tools(),
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


def execute(requests: list, cache_dir: str | None = None) -> list:
    """Run validated requests against the SANDBOXED cache. Never raises: a bad
    request becomes a readable error the agent can learn from, exactly like a
    rejected menu choice."""
    cache_dir = cache_dir or (SANDBOX_CACHE if os.path.exists(
        os.path.join(SANDBOX_CACHE, "meta.json")) else None)
    results = []
    for r in requests[:MAX_TOOL_CALLS]:
        try:
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
