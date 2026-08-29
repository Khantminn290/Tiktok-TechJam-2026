"""The outer loop: decide_action -> build_prompt -> LLM -> validate -> execute ->
persist Node -> update best -> convergence check -> repeat.

Convergence rule (as specified by the organizers, implemented verbatim): the run is
converged when the running-best validation primary has not improved by more than
epsilon = 0.002 over the last N = 3 consecutive *scored* iterations (an errored
iteration has no validation score, so it cannot advance or trigger the window; it
still counts against the iteration cap). Backstops: 50-iteration cap and a
wall-clock ceiling.
"""
from __future__ import annotations

import json
import os
import time

from .contracts import ExperimentTree, Node, error_headline, now
from .executor import run_solution
from .llm import LLMClient, LLMError
from .menu import Menu
from .policy import MIN_DRAFTS, decide_action
from .pricing import SpendTracker
from .prompts import build_prompt

EPSILON = 0.002
N_CONVERGE = 3
BASELINE_VALID_PRIMARY = 0.6016
BASELINE_SEED_STD = 0.0008     # the official FM baseline's own 5-seed std


class AgentLoop:
    def __init__(self, root: str, llm_model: str | None = None,
                 max_iterations: int = 50, wall_clock_limit_h: float = 6.0,
                 exec_timeout_s: int = 1200, seed: int = 0,
                 inject_error_at: int | None = None,
                 allow_locked_options: bool = False,
                 max_spend_usd: float = 2.0, draft_count: int | None = None,
                 test_model: bool = False, log_dir: str | None = None):
        self.root = root
        self.log_dir = log_dir or os.path.join(root, "logs")
        self.nodes_dir = os.path.join(self.log_dir, "nodes")
        os.makedirs(self.nodes_dir, exist_ok=True)
        self.menu = Menu(os.path.join(root, "config", "modification_menu.json"),
                         allow_locked_options=allow_locked_options)
        self.llm = LLMClient(model=llm_model, test=test_model)
        self.tree = ExperimentTree(self.log_dir, self.nodes_dir,
                                   project_root=self.root)
        self.max_iterations = max_iterations
        self.wall_clock_limit_s = wall_clock_limit_h * 3600
        self.exec_timeout_s = exec_timeout_s
        self.seed = seed
        self.inject_error_at = inject_error_at
        self.run_started = now()
        self.spend = SpendTracker(self.llm.provider, self.llm.model, max_spend_usd)
        cfg_draft = draft_count if draft_count is not None else MIN_DRAFTS
        self.draft_count = max(1, int(cfg_draft))
        self.gpu_seconds = 0.0
        self.device_seen = set()
        # consecutive LLM-stage failures: an auth/config error never fixes itself
        # by retrying, so the run aborts instead of burning the iteration cap.
        self.consecutive_llm_failures = 0
        self.last_llm_error = ""
        self.max_consecutive_llm_failures = 3

    # ---------- convergence ----------
    def converged(self) -> tuple[bool, str]:
        bests = []
        cur = -1.0
        for n in self.tree.nodes:
            if n.status == "success" and n.metrics:
                cur = max(cur, n.metrics["primary"])
                bests.append(cur)
        if len(bests) < N_CONVERGE + 1:
            return False, ""
        gain = bests[-1] - bests[-1 - N_CONVERGE]
        if gain <= EPSILON:
            return True, (f"converged: running-best valid primary improved only "
                          f"{gain:.4f} (≤ ε={EPSILON}) over the last "
                          f"{N_CONVERGE} scored iterations")
        return False, ""

    def stop_reason(self) -> str | None:
        if len(self.tree.nodes) >= self.max_iterations:
            return f"iteration cap reached ({self.max_iterations})"
        if now() - self.run_started > self.wall_clock_limit_s:
            return (f"wall-clock ceiling reached "
                    f"({self.wall_clock_limit_s / 3600:.1f} h)")
        over, msg = self.spend.would_exceed()
        if over:
            return msg
        fails = getattr(self, "consecutive_llm_failures", 0)
        if fails >= getattr(self, "max_consecutive_llm_failures", 3):
            return (f"aborted: {fails} consecutive LLM-stage "
                    f"failures with no successful call — this is a configuration "
                    f"problem, not something retrying will fix. Last error: "
                    f"{getattr(self, 'last_llm_error', '')[:200]}")
        conv, msg = self.converged()
        if conv:
            return msg
        return None

    # ---------- measured GPU time (written by the training script, if any) ----------
    def _collect_resources(self, run_dir: str) -> None:
        path = os.path.join(run_dir, "resource.json")
        if not os.path.exists(path):
            return
        try:
            with open(path) as fh:
                r = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return
        self.gpu_seconds += float(r.get("gpu_seconds", 0.0) or 0.0)
        dev = r.get("device")
        if dev:
            self.device_seen.add(str(dev))

    # ---------- one iteration ----------
    def iterate(self) -> Node:
        it = self.tree.next_id()
        action, target, reason = decide_action(self.tree,
                                               draft_count=self.draft_count)
        # a node that failed before any code existed (LLM failure) can't be debugged
        if action == "debug" and target is not None and not (
                target.code_path and os.path.exists(self.tree.resolve_code_path(target))):
            failed_id = target.iteration_id
            action, target = "draft", None
            reason = (f"node {failed_id} failed before any code was written "
                      f"(LLM-stage failure), so there is nothing to debug; "
                      f"drafting a fresh combination instead")
        print(f"[iter {it}] action={action}"
              f"{'' if target is None else f' target={target.iteration_id}'} — {reason}")

        node_dir = os.path.join(self.nodes_dir, f"node_{it:03d}")
        os.makedirs(node_dir, exist_ok=True)
        code_path = os.path.join(node_dir, "solution.py")
        run_dir = node_dir
        events = []
        prompt = build_prompt(action, target, reason, self.tree, self.menu)

        try:
            obj, usage, llm_events = self.llm.structured_call(
                prompt, validate_choices=self.menu.validate_choices)
            events.extend(llm_events)
        except LLMError as e:
            # recoverable: journal the failure; the policy will draft again next turn
            events.append({"type": "llm_failure", "error": str(e)[:1000]})
            self.consecutive_llm_failures += 1
            self.last_llm_error = str(e)
            self._print_spend(it)
            node = Node(iteration_id=it,
                        parent_id=None if target is None else target.iteration_id,
                        action=action, menu_choices={}, hypothesis="",
                        status="error", metrics=None,
                        error_trace=f"LLM stage failed: {e}",
                        tokens_used=0, wall_clock_seconds=0.0, timestamp=now(),
                        code_path="", decide_reason=reason, events=events)
            self.tree.add(node)
            print(f"[iter {it}] LLM stage failed (recovered, journaled): {e}", flush=True)
            return node

        self.consecutive_llm_failures = 0   # a successful call clears the abort counter
        code = obj["code"]
        if self.inject_error_at is not None and it == self.inject_error_at:
            code += "\nraise RuntimeError('injected failure (harness robustness test)')\n"
            events.append({"type": "injected_error_for_testing",
                           "note": "harness appended a raise to exercise the debug path"})
            print(f"[iter {it}] NOTE: injecting deliberate failure (robustness test)", flush=True)

        res = run_solution(code, code_path, obj["menu_choices"], run_dir,
                           timeout_s=self.exec_timeout_s, seed=self.seed)
        if not res.ok:
            events.append({"type": "execution_error",
                           "error_head": (res.error_trace or "")[:300]})
        self._collect_resources(run_dir)
        cost = self.spend.record(usage)
        events.append({"type": "spend", "iteration_usd": round(cost, 6),
                       "run_total_usd": round(self.spend.total_usd, 6),
                       "provider": self.llm.provider, "model": self.llm.model})
        if action == "crossover" and target is not None:
            from .policy import crossover_partner
            partner = crossover_partner(self.tree, target)
            events.append({"type": "crossover_parents",
                           "parent_a": target.iteration_id,
                           "parent_b": None if partner is None else partner.iteration_id})

        node = Node(iteration_id=it,
                    parent_id=None if target is None else target.iteration_id,
                    action=action, menu_choices=obj["menu_choices"],
                    hypothesis=obj["hypothesis"],
                    status="success" if res.ok else "error",
                    metrics=res.metrics, error_trace=res.error_trace,
                    tokens_used=sum(usage.values()),
                    wall_clock_seconds=res.wall_clock_seconds, timestamp=now(),
                    code_path=os.path.relpath(code_path, self.root),
                    expected_effect=obj["expected_effect"],
                    decide_reason=reason, token_breakdown=usage, events=events)
        self.tree.add(node)
        if res.ok:
            best = self.tree.best()
            print(f"[iter {it}] SUCCESS primary {res.metrics['primary']:.4f} "
                  f"(GAUC {res.metrics['GAUC']:.4f} nDCG@5 {res.metrics['nDCG@5']:.4f}) "
                  f"| best={best.metrics['primary']:.4f} (node {best.iteration_id}) "
                  f"| {res.wall_clock_seconds:.0f}s train, {node.tokens_used} tok")
        else:
            print(f"[iter {it}] ERROR ({res.wall_clock_seconds:.0f}s): "
                  f"{error_headline(res.error_trace)}")
        self._print_spend(it)
        return node

    def _print_spend(self, it: int) -> None:
        pct = (100.0 * self.spend.total_usd / self.spend.ceiling_usd
               if self.spend.ceiling_usd else 0.0)
        print(f"[iter {it}] spend ${self.spend.total_usd:.4f} / "
              f"${self.spend.ceiling_usd:.2f} ceiling ({pct:.0f}%) "
              f"| next est ${self.spend.estimated_next_call_usd():.4f} "
              f"| {self.llm.provider}:{self.llm.model}", flush=True)

    # ---------- full run ----------
    def run(self) -> dict:
        print(f"agent run started: max_iterations={self.max_iterations}, "
              f"wall_clock_limit={self.wall_clock_limit_s/3600:.1f}h, "
              f"llm={self.llm.provider}:{self.llm.model}, "
              f"draft_count={self.draft_count}, "
              f"spend_ceiling=${self.spend.ceiling_usd:.2f} "
              f"({self.spend.rates.describe(self.llm.provider, self.llm.model)}), "
              f"ε={EPSILON}, N={N_CONVERGE}")
        stop = self.stop_reason()
        while stop is None:
            self.iterate()
            stop = self.stop_reason()
        return self.finish(stop)

    def finish(self, stop: str) -> dict:
        best = self.tree.best()
        interventions = 0
        ipath = os.path.join(self.log_dir, "interventions.jsonl")
        if os.path.exists(ipath):
            with open(ipath) as fh:
                interventions = sum(1 for line in fh if line.strip())
        summary = {
            "stop_reason": stop,
            "iterations_used": len(self.tree.nodes),
            "iteration_cap": self.max_iterations,
            "best_node": None if best is None else best.iteration_id,
            "best_valid_metrics": None if best is None else best.metrics,
            "delta_valid_primary_over_baseline":
                None if best is None else round(
                    best.metrics["primary"] - BASELINE_VALID_PRIMARY, 4),
            "total_llm_tokens": self.llm.tokens_for_report(),
            "total_training_wall_clock_s": round(self.tree.total_wall_clock(), 1),
            "delta_in_baseline_seed_sigmas":
                None if best is None else round(
                    (best.metrics["primary"] - BASELINE_VALID_PRIMARY)
                    / BASELINE_SEED_STD, 2),
            "total_agent_wall_clock_s": round(now() - self.run_started, 1),
            # Measured, not assumed: summed from resource.json written by each
            # training run. Stays 0.0 only when every run really was CPU-only.
            "gpu_hours": round(self.gpu_seconds / 3600.0, 4),
            "gpu_seconds_measured": round(self.gpu_seconds, 1),
            "devices_used": sorted(self.device_seen) or ["cpu"],
            "spend": self.spend.summary(),
            "draft_count": self.draft_count,
            "manual_interventions": interventions,
            "convergence_rule": {"epsilon": EPSILON, "N": N_CONVERGE,
                                 "counted_iterations": "scored (successful) only"},
        }
        with open(os.path.join(self.log_dir, "final_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print("\n=== RUN FINISHED ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        if best is not None:
            print(f"\nbest solution: {os.path.join(self.log_dir, 'best_solution.py')}", flush=True)
            print("generate a submission with: python3 -m agent.make_submission", flush=True)
        return summary
