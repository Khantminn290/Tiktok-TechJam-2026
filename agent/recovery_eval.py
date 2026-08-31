"""Isolated, deterministic full-loop recovery evaluation.

Unlike the component fault suite, this module drives AgentLoop.iterate(), the
real policy, preflight, sandbox and executor. A scripted model removes network
availability and model sampling as confounders; it does not bypass any runtime
or output-contract check.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

from .contracts import ExperimentTree
from .loop import AgentLoop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "results", "recovery_eval.json")


class ScriptedLLM:
    provider = "openai"
    model = "scripted-offline-evaluator"

    def __init__(self, responses: list[dict]):
        self.responses = list(responses)

    def structured_call(self, _prompt, validate_choices=None):
        if not self.responses:
            raise RuntimeError("scripted evaluator exhausted its responses")
        obj = self.responses.pop(0)
        if validate_choices:
            obj["menu_choices"] = validate_choices(obj["menu_choices"])
        usage = {"input_tokens": 0, "output_tokens": 0,
                 "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": 0}
        return obj, usage, [{"type": "scripted_recovery_response"}]


def _response(code: str, hypothesis: str) -> dict:
    return {
        "hypothesis": hypothesis,
        "menu_choices": {},
        "code": code,
        "expected_effect": "recovery evaluation; no promotion",
        "rationale": {
            "idea": hypothesis,
            "why_expected_to_help": "exercise recovery routing",
            "grounded_in": "pre-registered recovery scenario",
        },
        "implementation_path": "A",
        "research_category": "exploration",
        "code_summary": "",
    }


def _seed_code() -> str:
    with open(os.path.join(ROOT, "runtime", "seed_solution.py")) as fh:
        return fh.read()


def _malformed_code() -> str:
    return """import argparse, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--menu-choices', required=True)
ap.add_argument('--output-dir', required=True)
ap.add_argument('--seed', type=int, default=0)
a = ap.parse_args()
os.makedirs(a.output_dir, exist_ok=True)
with open(os.path.join(a.output_dir, 'metrics.json'), 'w') as fh:
    json.dump({'GAUC': 0.66, 'nDCG@5': 0.53, 'primary': 0.595}, fh)
np.save(os.path.join(a.output_dir, 'scores_valid.npy'),
        np.full(124909, np.nan))
np.save(os.path.join(a.output_dir, 'scores_test.npy'),
        np.zeros(170588))
"""


def _runtime_code() -> str:
    return """import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--menu-choices', required=True)
ap.add_argument('--output-dir', required=True)
ap.add_argument('--seed', type=int, default=0)
ap.parse_args()
raise RuntimeError('pre-registered generated runtime failure')
"""


def _timeout_code() -> str:
    code = _seed_code()
    return code.replace(
        "choices = json.loads(a.menu_choices)",
        "choices = json.loads(a.menu_choices)\n    import time\n    time.sleep(2)")


def _make_loop(td: str, responses: list[dict], timeout_s: int,
               inject_error_at: int | None = None) -> AgentLoop:
    loop = AgentLoop(ROOT, max_iterations=2, wall_clock_limit_h=0.25,
                     exec_timeout_s=timeout_s, draft_count=1,
                     max_training_runs=2, max_spend_usd=1,
                     inject_error_at=inject_error_at)
    loop.log_dir = td
    loop.solutions_dir = os.path.join(td, "solutions")
    loop.runs_dir = os.path.join(td, "runs")
    loop.diffs_dir = os.path.join(td, "diffs")
    for path in (loop.solutions_dir, loop.runs_dir, loop.diffs_dir):
        os.makedirs(path, exist_ok=True)
    loop.tree = ExperimentTree(td)
    defaults = loop.menu.default_choices()
    for response in responses:
        response["menu_choices"] = dict(defaults)
    loop.llm = ScriptedLLM(responses)
    loop.spend.provider = loop.llm.provider
    loop.spend.model = loop.llm.model
    loop._record_experience = lambda *_args, **_kwargs: None
    return loop


def _scenario(name: str, first_code: str, timeout_s: int = 120,
              inject_error_at: int | None = None,
              recovery_timeout_s: int | None = None) -> dict:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"agent-recovery-{name}-") as td:
        responses = [_response(first_code, f"inject {name}"),
                     _response(_seed_code(), f"recover from {name}")]
        loop = _make_loop(td, responses, timeout_s, inject_error_at)
        first = loop.iterate()
        if recovery_timeout_s is not None:
            loop.exec_timeout_s = recovery_timeout_s
        second = loop.iterate()
        nodes = [{
            "iteration_id": n.iteration_id,
            "parent_id": n.parent_id,
            "action": n.action,
            "status": n.status,
            "primary": ((n.metrics or {}).get("primary")),
            "wall_clock_seconds": round(n.wall_clock_seconds, 3),
            "error_head": (((n.error_trace or "").splitlines() or [""])[0][:200]),
            "decide_reason": n.decide_reason,
        } for n in loop.tree.nodes]
        first_class = ""
        for event in first.events:
            if event.get("type") == "execution_error":
                first_class = event.get("failure_class", "")
        recovered = (first.status == "error" and second.status == "success"
                     and loop.ledger.training_runs == 2
                     and loop.ledger.unique_observations == 1)
        return {
            "name": name,
            "injected_failure_class": first_class,
            "first_status": first.status,
            "recovery_action": second.action,
            "later_success": second.status == "success",
            "later_primary": ((second.metrics or {}).get("primary")),
            "training_runs_spent": loop.ledger.training_runs,
            "crashed_runs": loop.ledger.training_crashes,
            "unique_observations": loop.ledger.unique_observations,
            "manual_interventions": 0,
            "recovered": recovered,
            "elapsed_s": round(time.time() - started, 2),
            "nodes": nodes,
        }


def run() -> dict:
    scenarios = [
        _scenario("runtime_error", _runtime_code()),
        _scenario("malformed_artifact", _malformed_code()),
        _scenario("timeout_reroute", _timeout_code(), timeout_s=1,
                  recovery_timeout_s=120),
    ]
    return {
        "schema": "closed_loop_recovery/1",
        "evaluation_mode": "real AgentLoop and executor; deterministic scripted LLM",
        "competition_evidence": False,
        "hidden_labels_available": False,
        "scenarios": scenarios,
        "recovered": sum(1 for s in scenarios if s["recovered"]),
        "total": len(scenarios),
        "manual_interventions": 0,
        "all_passed": all(s["recovered"] for s in scenarios),
    }


def write(out: str = DEFAULT_OUT) -> dict:
    result = run()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2)
    os.replace(tmp, out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    result = write(a.out)
    print(json.dumps(result, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
