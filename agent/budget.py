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


class Ledger:
    """Separate counters for things that are not interchangeable.

    The ambiguity this removes is real and was flagged in review: a paired
    3-seed confirmation is ONE outer-loop node and SIX training executions.
    Reporting it as "one iteration" understates compute by 6x; reporting it as
    six iterations overstates the number of decisions the agent made. Both are
    wrong, so both are counted, separately, and the report says which is which.

    Counting rule, stated once and applied everywhere:

        outer-loop node      one decision the agent made
        training execution   one model actually trained
        preflight rejection  neither -- no decision consumed, no compute spent

    A paired 3-seed confirmation therefore counts as:
        +1 outer-loop node, +6 training executions, +1 completed experiment.
    """

    def __init__(self, max_iterations: int | None = None,
                 max_training_runs: int | None = None):
        self.max_iterations = max_iterations
        self.max_training_runs = max_training_runs
        self.training_runs = 0
        self.training_crashes = 0

    def record_training(self, n: int = 1, crashed: int = 0) -> None:
        self.training_runs += int(n)
        self.training_crashes += int(crashed)

    def training_runs_left(self) -> int | None:
        if self.max_training_runs is None:
            return None
        return max(0, self.max_training_runs - self.training_runs)

    def can_afford(self, n_runs: int) -> bool:
        """Is there budget to COMPLETE an experiment costing n_runs?

        Starting a 6-run confirmation with 2 runs left produces two arms that
        cannot be paired and answers nothing, which is strictly worse than not
        starting it.
        """
        left = self.training_runs_left()
        return True if left is None else left >= int(n_runs)

    def why_not(self, n_runs: int) -> str:
        left = self.training_runs_left()
        if left is None or left >= n_runs:
            return ""
        return (f"needs {n_runs} training runs, only {left} remain of "
                f"{self.max_training_runs}")

    def as_dict(self) -> dict:
        return {"max_iterations": self.max_iterations,
                "max_training_runs": self.max_training_runs,
                "training_runs_used": self.training_runs,
                "training_runs_left": self.training_runs_left(),
                "training_crashes": self.training_crashes}


COUNTING_NOTE = (
    "An outer-loop node is one decision; a training execution is one model "
    "actually trained. A paired 3-seed confirmation is 1 node and 6 training "
    "executions. A preflight rejection is neither: no compute was spent and no "
    "decision was consumed, though repeated rejections are capped."
)


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
