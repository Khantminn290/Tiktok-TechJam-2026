"""What did this run actually spend, and on what?

A research budget is measured in experiments, not in nodes. Those are the same
thing only when every node runs an experiment, and after preflight they are
emphatically not: a script rejected in two seconds for calling a function that
does not exist has cost no compute and answered no question.

Treating that as a spent iteration is wrong in both directions. It overstates
what was spent, and -- worse -- it punishes the agent for a repairable mistake
by deleting a research iteration it never got to use. In the clean evaluation,
three of run 3's six iterations went to crashes of exactly this kind. Under this
accounting those cost repair attempts, not experiments.

The rule:

    An iteration consumes research budget when MEANINGFUL COMPUTE WAS SPENT.

A training run that crashes halfway still consumed budget -- the compute is gone
and the harness cannot get it back. A preflight rejection consumed nothing, so
it is charged to a separate, smaller allowance that exists purely to stop an
agent looping forever on a script it cannot fix.
"""
from __future__ import annotations

# A preflight rejection is free, but not unlimited: without a cap, an agent that
# cannot satisfy the contract would retry forever. Three attempts is enough to
# act on structured feedback and few enough to fail fast.
MAX_PREFLIGHT_RETRIES = 3

PREFLIGHT_MARKER = "PREFLIGHT REJECTED"


def was_preflight_rejection(node) -> bool:
    """Rejected before execution: no compute, no answer, no charge."""
    trace = (getattr(node, "error_trace", None) or "")
    return (getattr(node, "status", None) == "error"
            and PREFLIGHT_MARKER in trace
            and not getattr(node, "wall_clock_seconds", 0.0))


def consumes_budget(node) -> bool:
    return not was_preflight_rejection(node)


def count(nodes) -> dict:
    """Break a run's nodes into what was actually spent."""
    consumed = [n for n in nodes if consumes_budget(n)]
    preflight = [n for n in nodes if was_preflight_rejection(n)]
    scored = [n for n in consumed
              if getattr(n, "status", None) == "success" and getattr(n, "metrics", None)]
    crashed = [n for n in consumed if getattr(n, "status", None) == "error"]
    wall = sum(getattr(n, "wall_clock_seconds", 0.0) or 0.0 for n in nodes)
    return {"nodes_total": len(nodes),
            "iterations_consumed": len(consumed),
            "preflight_rejections": len(preflight),
            "experiments_completed": len(scored),
            "experiments_crashed": len(crashed),
            "training_wall_clock_s": round(wall, 1),
            "crash_rate_of_consumed": (round(len(crashed) / len(consumed), 3)
                                       if consumed else 0.0)}


def consecutive_preflight_failures(nodes) -> int:
    """How many preflight rejections in a row, counting back from the end."""
    n = 0
    for node in reversed(list(nodes)):
        if was_preflight_rejection(node):
            n += 1
        else:
            break
    return n


def render(c: dict) -> str:
    return "\n".join([
        "BUDGET",
        f"  iterations consumed    {c['iterations_consumed']} "
        f"(of {c['nodes_total']} nodes)",
        f"  preflight rejections   {c['preflight_rejections']}  "
        f"(no compute spent, not charged)",
        f"  experiments completed  {c['experiments_completed']}",
        f"  experiments crashed    {c['experiments_crashed']}  "
        f"(rate {c['crash_rate_of_consumed']:.0%} of consumed)",
        f"  training wall-clock    {c['training_wall_clock_s']}s",
    ])
