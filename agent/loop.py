"""The outer loop: decide_action -> build_prompt -> LLM -> validate -> execute ->
persist Node -> update best -> convergence check -> repeat.

Convergence rule: the run is converged when the running-best validation primary
has not improved by more than epsilon over the last N = 3 consecutive *scored*
iterations (an errored iteration has no validation score, so it cannot advance
or trigger the window). Epsilon is CALIBRATED to the benchmark's noise rather
than hand-picked -- it is the upward drift a running maximum shows over N
iterations by luck alone. See validity.convergence_epsilon and the note on
EPSILON below.

Budget: an iteration is charged when compute was actually spent. A script
rejected by preflight never ran, so it costs a repair attempt rather than a
research iteration. See agent.budget. Backstops: iteration cap, wall-clock
ceiling, spend ceiling.
"""
from __future__ import annotations

import json
import os
import shutil
import time

from .contracts import ExperimentTree, Node, error_headline, now
from .executor import run_parallel_round, run_solution, save_diff
from .experience import append_entry
from .llm import LLMClient, LLMError
from .menu import Menu
from .policy import MIN_DRAFTS, decide_action
from .pricing import SpendTracker
from .prompts import build_candidate_prompt, build_merge_prompt, build_prompt
from . import candidates as cand_mod
from . import inspect as inspect_tools
from . import propose_axis
from . import failure as failure_mod
from .research_policy import decide_category, render_decision
from .research_state import ResearchState
from .validity import convergence_epsilon
from . import budget

N_CONVERGE = 3
BASELINE_VALID_PRIMARY = 0.6016
BASELINE_SEED_STD = 0.0008     # the official FM baseline's own 5-seed std

# Stop when the running best has gained no more than selection drift alone
# would produce over N_CONVERGE iterations. See validity.convergence_epsilon.
#
# READ agent/convergence_report.py BEFORE CHANGING THIS. Two rules are in play
# and they are not interchangeable:
#
#   organizer rule   epsilon = 0.002, N = 3.  PUBLISHED BY THE ORGANIZERS.
#                    It defines when the run stops AND what is scored: the
#                    validation-best checkpoint at that point.
#   this constant    epsilon = 0.00048 (0.60 sigma). An INTERNAL research
#                    controller, not the official rule.
#
# An earlier version of this comment called 0.002 "the previous hand-picked"
# value and "a round number somebody liked". That was wrong: it is the
# competition's own rule, not a mistake that got fixed.
#
# The calibrated bar exists because at 2.5 sigma the loop declared convergence
# on differences it should have been investigating -- clean run 2 stopped at 4
# of 6 permitted iterations having gained 0.0003. It is the right bar for
# SEARCH. It does not extend eligibility: anything found after the organizer
# rule would have fired is research evidence, not a scored checkpoint.
EPSILON = convergence_epsilon(N_CONVERGE)     # 0.00048 = 0.60 sigma


class AgentLoop:
    def __init__(self, root: str, llm_model: str | None = None,
                 max_iterations: int = 50, wall_clock_limit_h: float = 6.0,
                 exec_timeout_s: int = 1200, seed: int = 0,
                 inject_error_at: int | None = None,
                 allow_locked_options: bool = False,
                 max_spend_usd: float = 2.0, draft_count: int | None = None,
                 test_model: bool = False, parallel_k: int | None = None,
                 min_branching_iterations: int = 0,
                 enable_data_tools: bool = False,
                 enable_research_state: bool = False,
                 enable_feature_discovery: bool = False,
                 n_candidates: int = 0,
                 max_training_runs: int | None = None,
                 competition_mode: bool = False):
        self.root = root
        self.log_dir = os.path.join(root, "logs")
        self.solutions_dir = os.path.join(self.log_dir, "solutions")
        self.runs_dir = os.path.join(self.log_dir, "runs")
        self.diffs_dir = os.path.join(self.log_dir, "diffs")
        os.makedirs(self.solutions_dir, exist_ok=True)
        os.makedirs(self.runs_dir, exist_ok=True)
        os.makedirs(self.diffs_dir, exist_ok=True)
        self.menu = Menu(os.path.join(root, "config", "modification_menu.json"),
                         allow_locked_options=allow_locked_options)
        self.llm = LLMClient(model=llm_model, test=test_model)
        self.tree = ExperimentTree(self.log_dir)
        self.max_iterations = max_iterations
        self.wall_clock_limit_s = wall_clock_limit_h * 3600
        self.competition_mode = bool(competition_mode)
        if self.competition_mode:
            from .convergence_report import ORGANIZER_EPSILON, ORGANIZER_N
            self.active_epsilon = ORGANIZER_EPSILON
            self.active_n_converge = ORGANIZER_N
        else:
            self.active_epsilon = EPSILON
            self.active_n_converge = N_CONVERGE
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
        # Opt-in parallel exploration (Phase 3 item 3 Part B). None/1 = today's
        # exact sequential behavior via iterate(), untouched. >=2 = K worker
        # proposals per round via iterate_parallel(). Default stays sequential.
        self.parallel_k = parallel_k if (parallel_k and parallel_k >= 2) else None
        self._round_counter = 0
        # Measured structural gap this exists to fix: across every run of this
        # project only `draft` (and one `merge`) ever fired on real LLM output.
        # The convergence rule is correct -- the starting conditions simply
        # never gave branching a chance, because the score plateaus before the
        # policy leaves the drafting phase. When set, convergence is BLOCKED
        # until the policy has actually executed an improve AND a debug (or
        # established that no errored node exists to debug). The iteration,
        # wall-clock and spend caps are NOT affected -- this can only delay a
        # convergence stop, never overrun a real budget.
        self.min_branching_iterations = int(min_branching_iterations or 0)
        self.enable_data_tools = bool(enable_data_tools)
        self.enable_research_state = bool(enable_research_state)
        # 0/1 disables multi-candidate planning (old single-proposal behaviour)
        self.n_candidates = int(n_candidates or 0)
        self._current_objective = None
        self.enable_feature_discovery = bool(enable_feature_discovery)
        # blocks the discovery prompt reuses, and the candidate it cleared
        self._last_state_block = ""
        self._last_error_block = ""
        self._pending_feature = None
        # Paired multi-seed experiments waiting to run. Populated by feature
        # discovery and by the promising-result gate; drained at the top of
        # iterate(). Before this existed, `_pending_feature` was written and
        # never read, and `feature_source` appeared in no journal ever recorded.
        self._confirmation_queue = []
        self._confirm_seeds = (0, 1, 2)
        # Parent nodes already put through a paired confirmation. A node
        # whose confirmation came back UNCONFIRMED is answered, not
        # unanswered -- asking again buys nothing and costs six runs.
        self._confirmed_nodes = set()
        # Ensembling is an ACTION the agent can take, not a human
        # post-process. Once per run: it is k training runs and the
        # result is deterministic, so repeating it buys nothing.
        self._ensemble_done = False
        # Fixed in advance, deliberately, and set to match the scale the
        # submitted ensemble uses. k is never chosen after seeing scores --
        # picking the k that happens to look best is selection on validation.
        self._ensemble_k = int(os.environ.get("AGENT_ENSEMBLE_K", "16"))
        # Training executions are a SEPARATE budget from outer-loop decisions.
        # A paired 3-seed confirmation is 1 node and 6 training runs; conflating
        # them under-counts compute six-fold. See agent.budget.COUNTING_NOTE.
        self.ledger = budget.Ledger(max_iterations=max_iterations,
                                    max_training_runs=max_training_runs)
        self._resume_work_dirs = {}
        self._restore_runtime_state()

    def _restore_runtime_state(self) -> None:
        """Restore volatile scheduler and accounting state from durable evidence."""
        if not self.tree.nodes:
            return
        from . import execution_events as EX
        from . import experiment_spec as XS
        from . import confirm as CF

        # Wall time, provider usage and spend remain cumulative across process
        # restarts. The journal is authoritative; no sidecar checkpoint is
        # required to resume safely.
        first = min(n.timestamp - (n.wall_clock_seconds or 0.0)
                    for n in self.tree.nodes)
        self.run_started = min(self.run_started, first)
        for node in self.tree.nodes:
            usage = node.token_breakdown or {}
            if usage:
                for key in ("input_tokens", "output_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens"):
                    self.llm.total_usage[key] += int(usage.get(key, 0) or 0)
                self.llm.total_usage["calls"] += 1
            for event in node.events or []:
                if event.get("type") == "spend":
                    cost = float(event.get("iteration_usd") or 0.0)
                    self.spend.total_usd += cost
                    self.spend.per_call.append(cost)

            execution = [e for e in (node.events or [])
                         if e.get("type") == "execution_event"]
            if execution:
                tally = EX.tally(execution)
                self.ledger.record_training(
                    tally["training_runs_spent"],
                    crashed=tally["by_kind"].get(EX.FAILED_EXECUTION, 0),
                    reused=tally["reused_artifacts"],
                    unique=tally["unique_observations"],
                    duplicates=tally["duplicate_reuse_attempts"])
            elif node.action in ("confirm", "ensemble"):
                # An interruption can occur after subprocesses finish but
                # before their execution events reach the journal. Count each
                # contract-valid artifact as compute already spent.
                work = os.path.join(
                    self.log_dir,
                    "confirm" if node.action == "confirm" else "ensemble_exp",
                    f"node_{node.iteration_id:03d}")
                complete = 0
                if os.path.isdir(work):
                    complete = sum(
                        CF._completed_artifact(os.path.join(work, name)) is not None
                        for name in os.listdir(work)
                        if os.path.isdir(os.path.join(work, name)))
                graded = any(e.get("type") in ("paired_result", "ensemble_result")
                             for e in (node.events or []))
                # Interrupted artifacts consumed compute, but they become
                # evidence only when the recovered action grades them. This
                # avoids counting the same six observations once from disk and
                # again when the retry journals their result.
                self.ledger.record_training(complete,
                                            unique=complete if graded else 0)
            elif (node.code_path and not budget.was_preflight_rejection(node)):
                self.ledger.record_training(1, crashed=int(node.status == "error"))

            if any(e.get("type") == "paired_result" for e in node.events or []):
                if node.parent_id is not None:
                    self._confirmed_nodes.add(node.parent_id)
            if node.action == "ensemble" and node.status == "success":
                self._ensemble_done = True

        # Retry the latest interrupted evidence action from its exact persisted
        # spec. Its old directory is retained so valid completed arms are reused.
        latest = self.tree.nodes[-1]
        if latest.action in ("confirm", "ensemble") and latest.status == "error":
            raw = next((e.get("spec") for e in latest.events or []
                        if e.get("type") == "experiment_spec"), None)
            if raw:
                spec = XS.ExperimentSpec.from_dict(raw)
                self._confirmation_queue.append(spec)
                old_work = os.path.join(
                    self.log_dir,
                    "confirm" if latest.action == "confirm" else "ensemble_exp",
                    f"node_{latest.iteration_id:03d}")
                self._resume_work_dirs[id(spec)] = old_work
                print(f"  [resume] restored interrupted {spec.experiment_type} "
                      f"from node {latest.iteration_id}", flush=True)
                return

        # Queue decisions are written on the producing node before persistence.
        # Rebuild an unanswered one when the process stopped between nodes.
        import dataclasses
        nodes = [dataclasses.asdict(n) for n in self.tree.nodes]
        pending_confirm = any(
            e.get("type") == "confirmation_queued"
            and e.get("parent_node") not in self._confirmed_nodes
            for n in self.tree.nodes for e in (n.events or []))
        if pending_confirm:
            self._maybe_queue_confirmation(
                nodes, {"category": "confirmation"}, [],
                max(0, self.max_iterations - len(self.tree.nodes)))
        elif any(e.get("type") == "ensemble_queued"
                 for n in self.tree.nodes for e in (n.events or [])) \
                and not self._ensemble_done:
            self._maybe_queue_ensemble(
                nodes, [], max(0, self.max_iterations - len(self.tree.nodes)))

    # ---------- convergence ----------
    def converged(self) -> tuple[bool, str]:
        # A search is not finished while a known-valuable action is untried.
        #
        # Convergence is measured on SINGLE-RUN scores, but ensembling is worth
        # about 1 sigma and no single run can ever show it -- so the rule
        # happily declares the search over with the most valuable remaining
        # action unattempted. Measured: a 20-iteration run stopped at 10 nodes
        # with 130 of 150 training runs and $11 of $12 unspent, having never
        # ensembled anything.
        #
        # Only blocks while it is actually affordable, so this can delay a
        # convergence stop but never overrun the iteration, training-run,
        # wall-clock or spend budgets.
        # getattr: tests build partial loops via __new__, where an absent
        # attribute means "no ensemble pending", not a crash.
        if (not getattr(self, "competition_mode", False)
                and not getattr(self, "_ensemble_done", True)):
            led = getattr(self, "ledger", None)
            scored = [n for n in self.tree.nodes
                      if n.status == "success" and n.metrics]
            k = getattr(self, "_ensemble_k", 16)
            if scored and (led is None or led.can_afford(k)):
                return False, ""

        bests = []
        cur = -1.0
        for n in self.tree.nodes:
            if n.status == "success" and n.metrics:
                cur = max(cur, n.metrics["primary"])
                bests.append(cur)
        epsilon = getattr(self, "active_epsilon", EPSILON)
        n_window = getattr(self, "active_n_converge", N_CONVERGE)
        if len(bests) < n_window + 1:
            return False, ""
        gain = bests[-1] - bests[-1 - n_window]
        if gain <= epsilon:
            source = ("organizer" if getattr(self, "competition_mode", False)
                      else "internal research")
            return True, (f"converged: running-best valid primary improved only "
                          f"{gain:.5f} (<= epsilon={epsilon:.5f}, "
                          f"{source} rule) over the last {n_window} scored "
                          f"iterations")
        return False, ""

    def stop_reason(self) -> str | None:
        if (getattr(self, "competition_mode", False) and self.tree.nodes
                and self.tree.nodes[0].action == "baseline"
                and self.tree.nodes[0].status != "success"):
            return "competition baseline failed; no scored run may proceed"
        # Nodes are not the unit of budget; EXPERIMENTS are. A script rejected
        # by preflight spent no compute and answered no question, so charging a
        # research iteration for it would delete an iteration the agent never
        # got to use. See agent.budget.
        consumed = sum(1 for n in self.tree.nodes if budget.consumes_budget(n))
        if consumed >= self.max_iterations:
            return f"iteration cap reached ({self.max_iterations})"
        # ...but a free retry cannot be unlimited, or an agent that never
        # satisfies the contract would loop forever.
        # getattr: tests and tools construct partial loops via __new__, and a
        # missing ledger means "no training cap", not a crash.
        led = getattr(self, "ledger", None)
        left = led.training_runs_left() if led else None
        if left is not None and left <= 0:
            return (f"training-run budget exhausted "
                    f"({led.training_runs} of "
                    f"{led.max_training_runs} used)")
        stuck = budget.consecutive_preflight_failures(self.tree.nodes)
        if stuck >= budget.MAX_PREFLIGHT_RETRIES:
            return (f"aborted: {stuck} consecutive preflight rejections — the "
                    f"agent is not acting on the structured feedback, which "
                    f"retrying will not fix")
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
            if getattr(self, "competition_mode", False):
                return msg
            blocked = self._branching_unfinished()
            if blocked:
                return None          # budget caps above still apply
            return msg
        return None

    def _branching_unfinished(self) -> str | None:
        """Reason branching is still owed, or None. Only gates convergence."""
        if not self.min_branching_iterations:
            return None
        acted = {n.action for n in self.tree.nodes}
        n_branch = sum(1 for n in self.tree.nodes
                       if n.action in ('improve', 'debug', 'crossover', 'merge'))
        need = []
        if 'improve' not in acted:
            need.append('improve')
        # debug is only meaningful if something actually failed; if nothing has
        # errored, the requirement is satisfied as 'not applicable' rather than
        # forcing the run to manufacture a failure.
        if 'debug' not in acted and any(n.status == 'error' for n in self.tree.nodes):
            need.append('debug')
        if n_branch < self.min_branching_iterations:
            need.append(f'{self.min_branching_iterations - n_branch} more branching iteration(s)')
        if not need:
            return None
        msg = 'branching not yet exercised: still owed ' + ', '.join(need)
        if not getattr(self, '_branch_msg_shown', None) == msg:
            print(f'[branching gate] convergence deferred -- {msg}', flush=True)
            self._branch_msg_shown = msg
        return msg

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

    # ---------- official competition bootstrap ----------
    def _competition_required_config(self) -> dict:
        """The verified incumbent configuration carried into a scored run.

        This is capability transfer, not a new discovery claim. The agent has
        already reproduced this configuration unaided; a competition run must
        now turn that accumulated knowledge into an eligible checkpoint before
        the organizer's short convergence window closes.
        """
        path = os.path.join(self.log_dir, "ensemble_results.json")
        if not os.path.exists(path):
            raise RuntimeError(
                "competition profile requires logs/ensemble_results.json so "
                "the verified incumbent configuration is explicit")
        with open(path) as fh:
            config = json.load(fh).get("config") or {}
        return self.menu.validate_choices(config)

    def _competition_bootstrap_pending(self) -> bool:
        if not getattr(self, "competition_mode", False):
            return False
        return not any(
            n.status == "success" and n.metrics and n.action != "baseline"
            for n in self.tree.nodes)

    def _competition_bootstrap_block(self) -> str:
        config = self._competition_required_config()
        return "\n".join([
            "## OFFICIAL COMPETITION BOOTSTRAP",
            "This is the first agent-authored checkpoint after the measured "
            "organizer baseline. Reproduce the verified incumbent configuration "
            "below exactly. The project already established it in prior research; "
            "this run is testing autonomous execution and official eligibility, "
            "not pretending to rediscover it from scratch.",
            f"required menu_choices: {json.dumps(config, sort_keys=True)}",
            "Write the complete executable solution and state that this is "
            "capability transfer from the accumulated research record.",
        ])

    def _ensure_competition_baseline(self) -> None:
        """Create node 0 by actually training the validation-only FM baseline."""
        if not getattr(self, "competition_mode", False):
            return
        if self.tree.nodes:
            if self.tree.nodes[0].action != "baseline":
                raise RuntimeError(
                    "competition mode cannot resume a journal that does not "
                    "start with the official baseline; use --fresh")
            return

        it = self.tree.next_id()
        code_src = os.path.join(self.root, "runtime", "seed_solution.py")
        code_path = os.path.join(self.solutions_dir, f"node_{it:03d}.py")
        shutil.copyfile(code_src, code_path)
        with open(code_path) as fh:
            code = fh.read()
        choices = self.menu.default_choices()
        run_dir = os.path.join(self.runs_dir, f"node_{it:03d}")
        print("[iter 0] action=baseline - validation-only organizer FM", flush=True)
        res = run_solution(code, code_path, choices, run_dir,
                           timeout_s=self.exec_timeout_s, seed=self.seed)
        if getattr(self, "ledger", None):
            self.ledger.record_training(1, crashed=0 if res.ok else 1)
        diff = save_diff(code_path, None, self.diffs_dir, it)
        expected = {"GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016}
        within_tolerance = bool(
            res.ok and res.metrics
            and abs(res.metrics["primary"] - expected["primary"]) <= 0.001)
        from . import provenance
        events = [{
            "type": "official_baseline",
            "published_validation_mean": expected,
            "observed_seed": self.seed,
            "observed_validation": res.metrics,
            "within_published_seed_tolerance": within_tolerance,
            "hidden_test_labels_available_to_subprocess": False,
            "note": ("executed through agent.executor; test outcome columns are "
                     "mechanically redacted and no test metric is computed"),
            "provenance": provenance.stamp(
                config=choices, seeds=[self.seed],
                code_paths=("runtime/seed_solution.py",
                            "kuairand-starter-kit/evaluate.py"),
                evaluation="starter-kit evaluate.py on validation only"),
        }]
        status = "success" if within_tolerance else "error"
        error = res.error_trace
        if res.ok and not within_tolerance:
            error = ("baseline reproduction drifted beyond tolerance: "
                     f"observed {res.metrics}, published {expected}")
        node = Node(
            iteration_id=it, parent_id=None, action="baseline",
            menu_choices=choices,
            hypothesis=("Reproduce the organizer FM baseline before any agent "
                        "improvement so convergence has a measured anchor."),
            status=status, metrics=res.metrics if status == "success" else None,
            error_trace=error, tokens_used=0,
            wall_clock_seconds=res.wall_clock_seconds, timestamp=now(),
            code_path=code_path, expected_effect="published primary 0.6016",
            decide_reason="mandatory competition baseline",
            token_breakdown={}, events=events,
            diff_path=diff["diff_path"], diff_sha256=diff["diff_sha256"],
            seed=self.seed,
            rationale={"idea": "measure the official starting point",
                       "why_expected_to_help": "anchors official convergence",
                       "grounded_in": "organizer baseline and evaluate.py"},
            implementation_path="A", research_category="baseline",
            code_summary="validation-only execution of the starter FM config")
        self.tree.add(node)
        if status != "success":
            print(f"[iter 0] BASELINE FAILED: {error_headline(error)}", flush=True)
        else:
            print(f"[iter 0] BASELINE primary {res.metrics['primary']:.5f}",
                  flush=True)

    def _publish_competition_ensemble(self, it: int, spec, out: dict,
                                      work_dir: str) -> dict:
        """Atomically make an eligible agent ensemble the canonical artifact."""
        if not getattr(self, "competition_mode", False) or not out.get("promote"):
            return {"published": False, "reason": "not an actionable competition ensemble"}
        members = out.get("members") or {}
        result = out.get("result") or {}
        metrics = result.get("ensemble") or {}
        if len(members) != len(spec.seeds) or not metrics:
            raise RuntimeError("refusing to publish an incomplete ensemble")

        final_dir = os.path.join(self.log_dir, "final_ensemble")
        tmp_dir = os.path.join(self.log_dir, ".final_ensemble.tmp")
        history = os.path.join(self.log_dir, "submission_history")
        os.makedirs(history, exist_ok=True)
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        shutil.copytree(work_dir, tmp_dir)

        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        if os.path.isdir(final_dir):
            prior_dir = os.path.join(history, f"{stamp}_final_ensemble")
            os.replace(final_dir, prior_dir)
        old_record = os.path.join(self.log_dir, "ensemble_results.json")
        if os.path.exists(old_record):
            shutil.copy2(old_record,
                         os.path.join(history, f"{stamp}_ensemble_results.json"))
        os.replace(tmp_dir, final_dir)

        per_seed = {str(s): round(float(members[s]["metrics"]["primary"]), 5)
                    for s in sorted(members)}
        from . import provenance
        prov = provenance.stamp(
            config=spec.treatment, seeds=spec.seeds,
            code_paths=("agent/loop.py", "agent/ensemble_experiment.py",
                        "agent/ensemble.py", "runtime/train_lib.py"),
            evaluation="starter-kit evaluate.py on validation only",
            extra={"aggregation": "rank_normalise_then_mean",
                   "members_dir": "logs/final_ensemble",
                   "source_node": it,
                   "agent_produced": True})
        data = prov.get("data") or {}
        record = {
            "primary": metrics["primary"],
            "GAUC": metrics["GAUC"],
            "nDCG@5": metrics["nDCG@5"],
            "k": len(spec.seeds),
            "seeds_used": list(spec.seeds),
            "single_seed_mean": result.get("mean_member"),
            "single_seed_std": result.get("sd_member"),
            "best_individual_seed": result.get("best_member"),
            "worst_individual_seed": min(per_seed.values()),
            "per_seed_primary": per_seed,
            "gain_over_mean_member": result.get("gain_over_mean_member"),
            "k_curve_diagnostic_only": result.get(
                "k_curve_diagnostic_only", {}),
            "duplicate_arrays_dropped": [],
            "delta_vs_baseline": round(
                metrics["primary"] - BASELINE_VALID_PRIMARY, 5),
            "sigma_vs_baseline": round(
                (metrics["primary"] - BASELINE_VALID_PRIMARY)
                / BASELINE_SEED_STD, 2),
            "config": dict(spec.treatment),
            "source_node": it,
            "members_dir": "logs/final_ensemble",
            "selection_bias": ("NONE -- all seeds were fixed before training; "
                               "no member, subset, or weight was selected on "
                               "validation. k_curve is diagnostic only."),
            "reproduce": ("python3 -m agent.final_ensemble --seeds "
                          f"{len(spec.seeds)}"),
            "timestamp_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "code_version": (prov.get("git") or {}).get("sha"),
            "data_version": {
                "fingerprint": data.get("sha256"),
                "valid_rows": ((data.get("splits") or {}).get("valid") or {})
                              .get("rows")},
            "evaluator": "kuairand-starter-kit/evaluate.py (never modified)",
            "hidden_test_used": False,
            "produced_by": "autonomous competition run",
            "official_candidate_node": it,
            "provenance": prov,
        }
        tmp_record = old_record + ".tmp"
        with open(tmp_record, "w") as fh:
            json.dump(record, fh, indent=2)
        os.replace(tmp_record, old_record)
        return {"published": True, "source_node": it,
                "members": len(spec.seeds), "members_dir": "logs/final_ensemble",
                "record": "logs/ensemble_results.json"}

    def _plan_candidates(self, action, target, reason, events, extra, objective):
        """Generate K candidates in ONE call, score them DETERMINISTICALLY,
        select the winner, and journal the full decision trace.

        Returns (winner_candidate | None, trace_text). Non-fatal: on any
        failure we fall back to the single-proposal path.
        """
        if not self.n_candidates or self.n_candidates < 2:
            return None, ""
        import dataclasses
        try:
            p = build_candidate_prompt(action, target, reason, self.tree,
                                       self.menu, n=self.n_candidates,
                                       exec_timeout_s=self.exec_timeout_s,
                                       data_block=extra, objective=objective)
            obj, usage = self.llm.json_call(p)
            self.spend.record(usage)
            raw = obj.get("candidates") or []
            if not isinstance(raw, list) or not raw:
                return None, ""
            cands = [cand_mod.Candidate(r, i) for i, r in enumerate(raw)]
            hist = [dataclasses.asdict(n) for n in self.tree.nodes]
            dead = (self.menu.raw.get("notes", {}) or {}).get("tested_dead_ends", [])
            st = None
            try:
                st = ResearchState(self.root)
            except Exception:
                pass
            left = max(0, self.max_iterations
                        - sum(1 for n in self.tree.nodes
                              if budget.consumes_budget(n)))
            try:
                from .frontier import from_root as _frontier
                fr = _frontier(self.root)
            except Exception:
                fr = None
            cand_mod.score_candidates(cands, history=hist, dead_ends=dead,
                                      state=st, budget_left=left,
                                      objective=objective, frontier=fr)
            winner, ranked = cand_mod.select(cands)
            trace = cand_mod.render_trace(winner, ranked, objective, st, left)
            if isinstance(obj.get("inquiry"), dict):
                events.append({"type": "inquiry", **{
                    k: str(v)[:400] for k, v in obj["inquiry"].items()}})
                _q = obj["inquiry"].get("question", "")
                if _q:
                    print(f"  [inquiry] {str(_q)[:100]}", flush=True)
            events.append({"type": "candidate_selection",
                           "n_candidates": len(ranked),
                           "n_path_b": sum(1 for c in ranked if c.path == "B"),
                           "n_rejected": sum(1 for c in ranked if c.rejected),
                           "selected": (winner.as_dict() if winner else None),
                           "all": [c.as_dict() for c in ranked]})
            nb = sum(1 for c in ranked if c.path == "B")
            print(f"  [planner] {len(ranked)} candidates ({nb} path-B), "
                  f"selected: "
                  f"{('#%d path=%s util=%.5f' % (winner.index, winner.path, winner.utility)) if winner else 'NONE (all gated)'}",
                  flush=True)
            return winner, trace
        except Exception as e:
            events.append({"type": "candidate_planning_skipped",
                           "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return None, ""

    def _research_block(self, events: list) -> str:
        """Stage C+D: compact research state + the explained category choice.
        Non-fatal -- if it cannot be built the iteration proceeds without it."""
        if not getattr(self, "enable_research_state", False):
            return ""
        try:
            st = ResearchState(self.root)
            nodes = [__import__("dataclasses").asdict(n) for n in self.tree.nodes]
            left = max(0, self.max_iterations
                        - sum(1 for n in self.tree.nodes
                              if budget.consumes_budget(n)))
            d = decide_category(st, nodes, iteration_budget_left=left)
            events.append({"type": "research_category", "category": d["category"],
                           "scores": d["scores"], "reason": d["reason"]})

            # Transparent utility allocation over experiment families. Recorded
            # as an event so the choice is auditable after the fact, and
            # rendered into the prompt so the agent sees the same reasoning.
            alloc_block = ""
            try:
                from . import allocator as AL
                alloc = AL.allocate(nodes, budget_left=left)
                events.append({"type": "allocation",
                               "choice": alloc["choice"],
                               "ranked": [{k: r[k] for k in
                                           ("family", "utility", "p_success",
                                            "expected_gain_sigma")}
                                          for r in alloc["ranked"][:4]]})
                alloc_block = AL.render(alloc)
                print(f"  [allocator] {alloc['choice']} "
                      f"(utility {alloc['ranked'][0]['utility']:.3f})", flush=True)
            except Exception as e:                  # noqa: BLE001
                events.append({"type": "allocation_skipped",
                               "error": f"{type(e).__name__}: {str(e)[:160]}"})

            # If what we believe rests on a single seed, schedule the paired
            # experiment that could actually settle it. This is the link that
            # turns "confirmation" from a prompt category into a run.
            if not getattr(self, "competition_mode", False):
                self._maybe_queue_confirmation(nodes, d, events, left)
                self._maybe_queue_ensemble(nodes, events, left)
            self._current_objective = d["category"]
            print(f"  [research policy] objective={d['category']} "
                  f"({d['reason']})", flush=True)
            # The frontier is what makes "what should we try next?" answerable
            # from evidence: it names every axis-option with a status, keeps
            # UNEXPLORED separate from KNOWN_BAD, and splits GAUC from nDCG@5.
            # Rendered after the state so the state's headline numbers still
            # lead, and capped so it cannot crowd out the decision itself.
            self._last_state_block = st.render()
            blocks = [self._last_state_block, render_decision(d)]
            if alloc_block:
                blocks.append(alloc_block)
            try:
                from .frontier import from_root as _frontier
                blocks.insert(1, _frontier(self.root).render(limit=22))
                # Capabilities the menu cannot express -- embedding size, decay
                # constants, the stopping rule. Each earned its place by
                # changing a conclusion during the Opus research run.
                from .pipeline_lab import render_for_prompt as _plab
                blocks.insert(2, _plab())
                from .validity import render_for_prompt as _valid
                blocks.insert(3, _valid())
                # The authoritative action space. Rendered from the same
                # registry that preflight enforces, so what the agent believes
                # it can call and what it is actually allowed to call cannot
                # drift apart -- which is precisely how 5 of 7 Path B nodes
                # crashed in the clean evaluation.
                from .capabilities import render_for_prompt as _caps
                blocks.insert(4, _caps())
                # What a result is allowed to count for. A single seed is
                # PRELIMINARY, never CONFIRMED.
                from .evidence import render_for_prompt as _ev
                blocks.insert(5, _ev())
                # Scoped beliefs, with their counterevidence attached.
                from .knowledge import render_for_prompt as _mem
                mem = _mem()
                if mem:
                    blocks.insert(6, mem)
            except Exception as e:
                events.append({"type": "frontier_skipped",
                               "error": f"{type(e).__name__}: {str(e)[:160]}"})
            return "\n\n".join(blocks)
        except Exception as e:
            events.append({"type": "research_state_skipped",
                           "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return ""

    # ---------- agent-driven data inspection (two-phase) ----------
    def _inspect_phase(self, events: list) -> str:
        """Phase 1: let the agent choose measurements; run them; return a text
        block for the hypothesis prompt. Failures are non-fatal -- inspection
        is an aid, never a prerequisite for making progress."""
        if not self.enable_data_tools:
            return ""
        from .experience import render_for_prompt as _exp
        try:
            iprompt = inspect_tools.build_inspect_prompt(self.menu, self.tree, _exp())
            obj, usage = self.llm.json_call(iprompt)
            reqs = inspect_tools.parse_requests(obj)
        except LLMError as e:
            events.append({"type": "inspect_skipped", "error": str(e)[:200]})
            return ""
        except Exception as e:
            events.append({"type": "inspect_skipped", "error": f"{type(e).__name__}"})
            return ""
        self.spend.record(usage)
        if not reqs:
            events.append({"type": "inspect", "requests": 0})
            return ""
        results = inspect_tools.execute(reqs)
        events.append({"type": "inspect", "requests": len(reqs),
                       "tools": [r["tool"] for r in reqs],
                       "errors": sum(1 for x in results if "error" in x)})
        print(f"  [data tools] agent requested {len(reqs)}: "
              f"{[r['tool'] for r in reqs]}", flush=True)
        return inspect_tools.render_results(results)

    def _feature_discovery_phase(self, events: list) -> str:
        """Autonomous feature research, one candidate per iteration.

            evidence -> hypothesis -> builder code -> PROBE -> decision

        The probe is what makes this affordable: a training run costs ~70s and
        answers one question badly against a 0.0008 noise floor, while the
        probe costs seconds and answers the question that actually gates the
        decision -- does this carry signal the incumbent lacks? Only a
        candidate that clears it reaches a training run.

        Every outcome is recorded, including refusals, so a failed feature is
        never rediscovered. Failures here are non-fatal: feature discovery is
        an addition to the loop, never a prerequisite for making progress.
        """
        if not getattr(self, "enable_feature_discovery", False):
            return ""
        from . import feature_lab as FL
        try:
            state = self._last_state_block or ""
            # Accumulated evidence, not just this run's: the frontier aggregates
            # every archived journal, so on a --fresh run (where the state block
            # is nearly empty) the agent still reasons from what the project has
            # actually measured instead of declining for lack of evidence.
            err = self._last_error_block or ""
            try:
                from .frontier import from_root as _frontier
                err = (err + "\n\n" + _frontier(self.root).render(limit=18)).strip()
            except Exception:
                pass
            key_block = ""
            try:
                from . import error_analysis as _EA
                _sp, _mt = _EA.load_valid()
                key_block = FL.key_diagnostics(_sp, _mt)
            except Exception:
                pass
            prompt = FL.build_feature_prompt(state, err, FL.render_for_prompt(),
                                             key_block)
            obj, usage = self.llm.json_call(prompt)
            self.spend.record(usage)
        except LLMError as e:
            events.append({"type": "feature_discovery_skipped", "error": str(e)[:200]})
            return ""
        except Exception as e:
            events.append({"type": "feature_discovery_skipped",
                           "error": f"{type(e).__name__}: {str(e)[:160]}"})
            return ""

        if not obj.get("propose"):
            events.append({"type": "feature_discovery", "proposed": False,
                           "reason": str(obj.get("decline_reason", ""))[:200]})
            print(f"  [feature discovery] declined: "
                  f"{str(obj.get('decline_reason', ''))[:90]}", flush=True)
            return ""

        problems = FL.validate_proposal(obj)
        if problems:
            events.append({"type": "feature_discovery", "proposed": True,
                           "rejected": "incomplete proposal", "problems": problems})
            print(f"  [feature discovery] incomplete proposal: {problems[:2]}",
                  flush=True)
            return ""

        seen = FL.already_tried(obj.get("name", ""), obj.get("source", ""))
        if seen:
            events.append({"type": "feature_discovery", "proposed": True,
                           "duplicate_of": seen.get("name"),
                           "prior_status": seen.get("status")})
            print(f"  [feature discovery] already tried: {seen.get('name')} "
                  f"({seen.get('status')})", flush=True)
            return ""

        try:
            from . import error_analysis as EA
            splits, meta = EA.load_valid()
            inc = self._incumbent_valid_scores()
            res = FL.probe(obj, splits, meta, incumbent_scores=inc)
        except Exception as e:
            events.append({"type": "feature_discovery_skipped",
                           "error": f"probe failed: {type(e).__name__}: {str(e)[:160]}"})
            return ""

        entry = {k: obj.get(k) for k in FL.REQUIRED_FIELDS}
        entry.update(status=res["status"], reason=res.get("reason", ""),
                     probe=res, iteration=len(self.tree.nodes))
        FL.record(entry)
        events.append({"type": "feature_discovery", "proposed": True,
                       "name": obj.get("name"), "status": res["status"],
                       "reason": res.get("reason", "")[:200],
                       "best_incremental_sigma": res.get("best_incremental_sigma")})
        print(f"  [FEATURE DISCOVERY] {obj.get('name')} -> {res['status']} "
              f"({res.get('reason', '')[:80]})", flush=True)
        self._pending_feature = (obj if res["status"] == FL.PROMISING else None)

        # A cleared probe becomes an EXECUTABLE follow-up, automatically. The
        # treatment carries the exact stored source, so what gets retrained is
        # provably what was probed rather than whatever the model retypes next
        # iteration -- which, on the record, it never did once.
        if self._pending_feature is not None:
            try:
                from . import confirm as CF
                from . import feature_store as FS
                stored = FS.record_discovery(obj, res, node_id=len(self.tree.nodes))
                control = CF.incumbent_choices() or self.menu.default_choices()
                spec = FS.followup_spec(stored, control, seeds=self._confirm_seeds,
                                        timeout_s=self.exec_timeout_s)
                self.queue_confirmation(spec)
                events.append({"type": "feature_followup_queued",
                               "sha": stored["sha"], "name": stored.get("name"),
                               "n_runs": spec.n_runs})
                print(f"  [FEATURE] queued paired confirmation for "
                      f"{stored.get('name')} (sha {stored['sha']}, "
                      f"{spec.n_runs} runs)", flush=True)
            except Exception as e:                  # noqa: BLE001
                events.append({"type": "feature_followup_failed",
                               "error": f"{type(e).__name__}: {str(e)[:200]}"})
        return FL.render_probe_for_prompt(res)

    # ---------- executable paired confirmation ----------
    def _maybe_queue_confirmation(self, nodes: list, decision: dict,
                                  events: list, budget_left: int) -> None:
        """Turn a promising single-seed result into a paired experiment.

        Fires only when the best result is genuinely worth the runs: above half
        the noise floor, still PRELIMINARY, not already queued, and with enough
        budget left to finish. Confirming noise wastes as much of the budget as
        believing it does.
        """
        if self._confirmation_queue or budget_left < 1:
            return
        scored = [n for n in nodes
                  if n.get("status") == "success" and n.get("metrics")]
        if not scored:
            return
        best = max(scored, key=lambda n: n["metrics"]["primary"])
        # A node produced BY a confirmation is already multi-seed; re-confirming
        # it would just spend runs re-measuring what is already measured.
        if (best.get("action") or "") == "confirm":
            return
        # ...and neither may a node that has ALREADY been confirmed once.
        # Found by running it: node 0 came back UNCONFIRMED, stayed the
        # highest-scoring node because its single lucky seed still topped the
        # paired mean, and was re-queued on the very next iteration. Left alone
        # that spends six training runs per iteration re-measuring the same
        # thing forever and never explores again.
        if best.get("iteration_id") in self._confirmed_nodes:
            return
        delta = best["metrics"]["primary"] - BASELINE_VALID_PRIMARY
        if delta < BASELINE_SEED_STD / 2:
            return
        try:
            from . import confirm as CF
            from . import experiment_spec as XS
            # In a scored competition run, confirm the transferred incumbent
            # against the organizer FM baseline. Using the incumbent itself as
            # control would make the required bootstrap treatment identical to
            # control and silently skip the confirmation node.
            control = (self.menu.default_choices()
                       if getattr(self, "competition_mode", False)
                       else CF.incumbent_choices() or self.menu.default_choices())
            treatment = dict(best.get("menu_choices") or {})
            if not treatment or treatment == control:
                return
            spec = XS.ExperimentSpec(
                hypothesis=(f"Node {best.get('iteration_id')} scored "
                            f"{best['metrics']['primary']:.5f} on ONE seed "
                            f"({delta / BASELINE_SEED_STD:+.2f} sigma over "
                            f"baseline). Does it hold against the incumbent "
                            f"when both arms run the same seeds?"),
                experiment_type=XS.MULTI_SEED_REPLICATION,
                control=control, treatment=treatment,
                seeds=self._confirm_seeds,
                parent_node=best.get("iteration_id"),
                expected_primary_effect=delta,
                runtime_budget_s=self.exec_timeout_s)
            self.queue_confirmation(spec)
            events.append({"type": "confirmation_queued",
                           "parent_node": best.get("iteration_id"),
                           "n_runs": spec.n_runs, "seeds": list(spec.seeds)})
            print(f"  [confirm] queued paired confirmation of node "
                  f"{best.get('iteration_id')} ({spec.n_runs} runs)", flush=True)
        except Exception as e:                      # noqa: BLE001
            events.append({"type": "confirmation_queue_failed",
                           "error": f"{type(e).__name__}: {str(e)[:200]}"})

    def _maybe_queue_ensemble(self, nodes: list, events: list,
                             budget_left: int) -> None:
        """Decide to ensemble, once there is something worth ensembling.

        Fires when a configuration has been CONFIRMED by a paired experiment --
        or, failing that, when the same configuration has scored repeatedly --
        because averaging seeds of a configuration that is not actually good
        just buys a precise estimate of a mediocre number.

        This is the step that used to be a human running
        `agent.final_ensemble --seeds 16` after the agent had stopped. It is the
        largest single measured gain available (about 1 sigma) and it now sits
        inside the agent's action space rather than outside it.
        """
        if self._confirmation_queue or self._ensemble_done:
            return
        from . import ensemble_experiment as EE
        k = self._ensemble_k
        led = getattr(self, "ledger", None)
        if budget_left < 1 or (led and not led.can_afford(k)):
            return

        scored = [n for n in nodes
                  if n.get("status") == "success" and n.get("metrics")]
        if not scored:
            return

        # Prefer a configuration a paired confirmation has actually backed.
        target, why = None, ""
        for n in nodes:
            for e in (n.get("events") or []):
                if (e.get("type") == "paired_result"
                        and (e.get("evidence") or {}).get("state") == "CONFIRMED"):
                    target = n.get("menu_choices")
                    why = f"node {n.get('iteration_id')} was CONFIRMED"
        if target is None:
            if getattr(self, "competition_mode", False):
                # Official mode preserves the evidence gate: the fixed ensemble
                # is scheduled only after a paired result confirms its config.
                return
            # Ensemble the best MENU-DRIVEN configuration that beats the
            # baseline. An agent-written script with no menu_choices cannot be
            # re-run at k seeds through the reference solution, so it is not a
            # candidate however well it scored.
            #
            # Requiring the config to have been reproduced first was too strict
            # in practice: the best configuration is usually found exactly once,
            # so the trigger never fired and runs ended having never ensembled.
            # The downside of ensembling a mediocre config is a precise estimate
            # of a mediocre number -- it costs k runs and simply does not
            # promote, which the evidence layer already handles.
            usable = [n for n in scored
                      if (n.get("menu_choices") or {}).get("model")
                      and n["metrics"]["primary"] > BASELINE_VALID_PRIMARY]
            if not usable:
                return
            best = max(usable, key=lambda n: n["metrics"]["primary"])
            target = best.get("menu_choices")
            why = (f"best menu configuration is node "
                   f"{best.get('iteration_id')} at "
                   f"{best['metrics']['primary']:.5f}")
        if not target:
            return

        spec = EE.spec_for(target, k=k, timeout_s=self.exec_timeout_s)
        self.queue_confirmation(spec)
        self._ensemble_done = True
        events.append({"type": "ensemble_queued", "k": k, "reason": why})
        print(f"  [ensemble] queued {k}-member ensemble — {why}", flush=True)

    def queue_confirmation(self, spec) -> None:
        """Schedule a paired multi-seed experiment for the next iteration."""
        self._confirmation_queue.append(spec)

    def _schedule_competition_followup(self, node: Node) -> None:
        """Queue the next evidence step before the current node is persisted.

        The original scheduler ran inside the next iteration's planning phase,
        so each follow-up arrived one node late: candidate at 1, confirmation at
        3, ensemble at 5. The organizer can converge by node 3. Scheduling from
        the completed node gives the required candidate -> confirm -> ensemble
        sequence while keeping every queue decision in that node's journal.
        """
        if not getattr(self, "competition_mode", False):
            return
        if node.status != "success" or not node.metrics:
            return
        import dataclasses
        nodes = [dataclasses.asdict(n) for n in self.tree.nodes]
        nodes.append(dataclasses.asdict(node))
        used = sum(1 for n in self.tree.nodes if budget.consumes_budget(n)) + 1
        left = max(0, self.max_iterations - used)
        decision = {"category": "confirmation"}
        self._maybe_queue_confirmation(nodes, decision, node.events, left)
        self._maybe_queue_ensemble(nodes, node.events, left)

    def _dequeue_confirmation(self):
        q = getattr(self, "_confirmation_queue", None)
        if not q:
            return None
        # Only run one if the remaining budget can actually pay for it: a
        # confirmation is n_runs training runs, and starting one it cannot
        # finish would spend the budget and answer nothing.
        spec = q[0]
        left = max(0, self.max_iterations
                   - sum(1 for n in self.tree.nodes if budget.consumes_budget(n)))
        if left < 1:
            return None
        # The real constraint is TRAINING RUNS, not one outer-loop slot. Starting
        # a 6-run paired confirmation with 2 runs left produces two arms that
        # cannot be paired and answers nothing.
        led = getattr(self, "ledger", None)
        if led and not led.can_afford(spec.n_runs):
            print(f"  [confirm] deferring {spec.experiment_type}: "
                  f"{led.why_not(spec.n_runs)}", flush=True)
            return None
        return q.pop(0)

    def _run_ensemble_node(self, it: int, spec) -> Node:
        """Train k members, combine them, and journal it as a real node.

        Ensembling used to be a human step run after the agent finished, which
        made the submitted number only partly the agent's. It is an action now.
        """
        from . import ensemble_experiment as EE

        events = [{"type": "research_category", "category": "integration",
                   "reason": "removing seed variance from the submitted score"},
                  {"type": "experiment_spec", "spec": spec.to_dict()}]
        k = len(spec.seeds)
        print(f"[iter {it}] action=ensemble — {k} members of the best "
              f"configuration", flush=True)
        t0 = now()
        work = os.path.join(self.log_dir, "ensemble_exp", f"node_{it:03d}")
        try:
            out = EE.run(spec.treatment, spec.seeds, work,
                         timeout_s=self.exec_timeout_s)
        except Exception as e:                      # noqa: BLE001
            events.append({"type": "execution_error", "failure_class": "unknown",
                           "error_head": f"{type(e).__name__}: {str(e)[:300]}"})
            node = Node(iteration_id=it, parent_id=spec.parent_node,
                        action="ensemble", menu_choices=spec.treatment,
                        hypothesis=spec.hypothesis, status="error", metrics=None,
                        error_trace=f"{type(e).__name__}: {e}", tokens_used=0,
                        wall_clock_seconds=now() - t0, timestamp=time.time(),
                        code_path="", events=events,
                        research_category="integration", implementation_path="A")
            self.tree.add(node)
            return node

        res, ev = out["result"], out["evidence"]
        tally = out.get("execution_tally") or {}
        fresh = tally.get("fresh_executions", len(out["members"]))
        reused = tally.get("reused_artifacts", 0)
        if getattr(self, "ledger", None):
            self.ledger.record_training(
                fresh, crashed=max(0, k - fresh - reused), reused=reused,
                unique=tally.get("unique_observations"),
                duplicates=tally.get("duplicate_reuse_attempts", 0))
        events.extend(out.get("events") or [])
        events.append({"type": "ensemble_result", "result": res,
                       "evidence": ev, "promote": out["promote"]})

        metrics = res.get("ensemble") if res.get("usable") else None
        publish_error = None
        if metrics and getattr(self, "competition_mode", False):
            try:
                publication = self._publish_competition_ensemble(
                    it, spec, out, work)
                events.append({"type": "competition_ensemble_publication",
                               **publication})
            except Exception as e:                  # noqa: BLE001
                publish_error = f"{type(e).__name__}: {e}"
                events.append({"type": "competition_ensemble_publication_failed",
                               "error": publish_error[:300]})
                metrics = None
        node = Node(iteration_id=it, parent_id=spec.parent_node,
                    action="ensemble", menu_choices=spec.treatment,
                    hypothesis=spec.hypothesis,
                    status="success" if metrics else "error", metrics=metrics,
                    error_trace=(None if metrics else publish_error
                                 or "too few members combined"),
                    tokens_used=0, wall_clock_seconds=now() - t0,
                    timestamp=time.time(), code_path="", events=events,
                    decide_reason="queued ensemble construction",
                    research_category="integration", implementation_path="A")
        self.tree.add(node)
        if metrics:
            print(f"[iter {it}] ENSEMBLE {ev['state']} primary "
                  f"{metrics['primary']:.5f} (gain "
                  f"{res['gain_over_mean_member']:+.5f} over the mean member)",
                  flush=True)
        return node

    def _run_confirmation_node(self, it: int, spec) -> Node:
        """Execute a paired spec and journal it as a first-class node."""
        from . import confirm as CF
        from . import experiment_spec as XS
        from . import feature_store as FS

        events = [{"type": "research_category", "category": "confirmation",
                   "reason": "executing a queued paired multi-seed experiment"},
                  {"type": "experiment_spec", "spec": spec.to_dict()}]
        print(f"[iter {it}] action=confirm — paired {spec.experiment_type}, "
              f"{spec.n_runs} runs over seeds {list(spec.seeds)}", flush=True)
        print(spec.render(), flush=True)

        t0 = now()
        work = self._resume_work_dirs.pop(
            id(spec), os.path.join(self.log_dir, "confirm", f"node_{it:03d}"))
        if os.path.basename(work) != f"node_{it:03d}":
            events.append({"type": "interrupted_work_recovered",
                           "source_work_dir": os.path.relpath(work, self.root),
                           "recovery_node": it})
        try:
            out = CF.run_spec(spec, work_dir=work, timeout_s=self.exec_timeout_s)
            tally = out.get("execution_tally") or {}
            fresh = tally.get("fresh_executions", 0)
            reused = tally.get("reused_artifacts", 0)
            if getattr(self, "ledger", None):
                self.ledger.record_training(
                    fresh,
                    crashed=max(0, spec.n_runs - fresh - reused),
                    reused=reused,
                    unique=tally.get("unique_observations"),
                    duplicates=tally.get("duplicate_reuse_attempts", 0))
        except Exception as e:                      # noqa: BLE001
            events.append({"type": "execution_error", "failure_class": "unknown",
                           "error_head": f"{type(e).__name__}: {str(e)[:300]}"})
            node = Node(iteration_id=it, parent_id=spec.parent_node,
                        action="confirm", menu_choices=spec.treatment,
                        hypothesis=spec.hypothesis, status="error",
                        metrics=None,
                        error_trace=f"{type(e).__name__}: {e}", tokens_used=0,
                        wall_clock_seconds=now() - t0, timestamp=time.time(),
                        code_path="", events=events,
                        research_category="confirmation",
                        implementation_path="A")
            if spec.parent_node is not None:
                self._confirmed_nodes.add(spec.parent_node)
            self.tree.add(node)
            return node

        res, ev = out["paired"], out["evidence"]
        events.extend(out.get("events") or [])
        events.append({"type": "paired_result", "result": res,
                       "evidence": {k: v for k, v in ev.items() if k != "paired"},
                       "promote": out["promote"]})

        # The treatment arm's own mean is the node's score. It is a real
        # measurement of the treatment, and unlike a single draw it is an
        # average over seeds -- so it is comparable to other nodes without
        # overstating what was learned.
        metrics = None
        if res.get("usable"):
            tre = out["treatment"]
            seeds = res["seeds"]
            metrics = {k: sum(tre[s][k] for s in seeds) / len(seeds)
                       for k in ("GAUC", "nDCG@5", "primary")}

        if spec.feature_lineage.get("sha"):
            FS.update_outcome(spec.feature_lineage["sha"], res, ev,
                              runtime_s=now() - t0, config=spec.treatment)

        node = Node(iteration_id=it, parent_id=spec.parent_node,
                    action="confirm", menu_choices=spec.treatment,
                    hypothesis=spec.hypothesis,
                    status="success" if metrics else "error",
                    metrics=metrics,
                    error_trace=None if metrics else "confirmation produced too "
                                                     "few completed seeds to pair",
                    tokens_used=0, wall_clock_seconds=now() - t0,
                    timestamp=time.time(), code_path="", events=events,
                    expected_effect=f"{spec.expected_primary_effect:+.5f}",
                    decide_reason="queued paired confirmation",
                    research_category="confirmation", implementation_path="A")
        if spec.parent_node is not None:
            self._confirmed_nodes.add(spec.parent_node)
        self._schedule_competition_followup(node)
        self.tree.add(node)
        print(f"[iter {it}] CONFIRM {ev['state']}"
              + (f" primary {metrics['primary']:.5f}" if metrics else "")
              + f" | promote={out['promote']}", flush=True)
        return node

    def _incumbent_valid_scores(self):
        """Rank-averaged validation scores of the submitted ensemble, if built.
        Without it the probe can still measure standalone signal, just not the
        residual -- which is the number that actually decides."""
        import numpy as np
        p = os.path.join(self.log_dir, "ensemble_results.json")
        if not os.path.exists(p):
            return None
        try:
            from .ensemble import rank_normalise
            with open(p) as fh:
                res = json.load(fh)
            arrs = []
            for i in res.get("seeds_used", []):
                q = os.path.join(self.root, res.get("members_dir", ""),
                                 f"seed_{i:02d}", "scores_valid.npy")
                if os.path.exists(q):
                    arrs.append(rank_normalise(np.load(q)))
            return np.mean(arrs, axis=0) if arrs else None
        except Exception:
            return None

    # ---------- one iteration ----------
    def iterate(self) -> Node:
        it = self.tree.next_id()

        # A queued paired confirmation outranks anything the planner would pick.
        # This is the only code path that can produce more than one seed, and
        # without it the agent can form a hypothesis, measure it, correctly
        # report that one seed proves nothing, and then do nothing about it
        # forever. Every one of the 37 nodes recorded before this existed ran
        # seed 0.
        queued = self._dequeue_confirmation()
        if queued is not None:
            from . import experiment_spec as _XS
            if queued.experiment_type == _XS.ENSEMBLE_CONSTRUCTION:
                return self._run_ensemble_node(it, queued)
            return self._run_confirmation_node(it, queued)

        action, target, reason = decide_action(self.tree,
                                               draft_count=self.draft_count)
        # a node that failed before any code existed (LLM failure) can't be debugged
        if action == "debug" and target is not None and not (
                target.code_path and os.path.exists(target.code_path)):
            failed_id = target.iteration_id
            action, target = "draft", None
            reason = (f"node {failed_id} failed before any code was written "
                      f"(LLM-stage failure), so there is nothing to debug; "
                      f"drafting a fresh combination instead")
        print(f"[iter {it}] action={action}"
              f"{'' if target is None else f' target={target.iteration_id}'} — {reason}")

        code_path = os.path.join(self.solutions_dir, f"node_{it:03d}.py")
        run_dir = os.path.join(self.runs_dir, f"node_{it:03d}")
        events = []
        competition_bootstrap = self._competition_bootstrap_pending()
        if competition_bootstrap:
            # Do not spend three planning calls rediscovering a configuration
            # the accumulated research record already established. The LLM
            # still authors the executable code and rationale; the exact config
            # is pinned so official eligibility does not depend on sampling.
            extra = self._competition_bootstrap_block()
            winner, trace = None, ""
            events.append({"type": "competition_bootstrap",
                           "mode": "capability_transfer",
                           "required_config": self._competition_required_config()})
        else:
            extra = (self._research_block(events) + "\n\n"
                     + self._inspect_phase(events))
            feature_block = self._feature_discovery_phase(events)
            if feature_block:
                extra += "\n\n" + feature_block
            winner, trace = self._plan_candidates(
                action, target, reason, events, extra, self._current_objective)
        if trace:
            extra = extra + "\n\n" + trace
        if winner is not None:
            extra += ("\n\n## IMPLEMENT THIS SELECTED CANDIDATE\n"
                      f"path={winner.path} category={winner.category}\n"
                      f"hypothesis: {winner.hypothesis}\n"
                      f"mechanism: {winner.mechanism}\n"
                      + (f"menu_choices: {json.dumps(winner.menu_choices)}\n"
                         if winner.path == "A" else "")
                      + "The option set has already been scored; implement THIS "
                        "candidate faithfully rather than substituting another "
                        "idea. Keep implementation_path and research_category "
                        "consistent with it.")
        prompt = build_prompt(action, target, reason, self.tree, self.menu,
                             exec_timeout_s=self.exec_timeout_s,
                             data_block=extra)

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
        if competition_bootstrap:
            required = self._competition_required_config()
            if obj.get("menu_choices") != required:
                cost = self.spend.record(usage)
                events.extend([
                    {"type": "competition_bootstrap_rejected",
                     "required_config": required,
                     "received_config": obj.get("menu_choices")},
                    {"type": "spend", "iteration_usd": round(cost, 6),
                     "run_total_usd": round(self.spend.total_usd, 6),
                     "provider": self.llm.provider, "model": self.llm.model},
                ])
                node = Node(
                    iteration_id=it, parent_id=None, action=action,
                    menu_choices=obj.get("menu_choices") or {},
                    hypothesis=obj.get("hypothesis", ""), status="error",
                    metrics=None,
                    error_trace=(budget.PREFLIGHT_MARKER + ": competition "
                                 "bootstrap must reproduce the exact verified "
                                 "incumbent configuration"),
                    tokens_used=sum(usage.values()), wall_clock_seconds=0.0,
                    timestamp=now(), code_path="", decide_reason=reason,
                    token_breakdown=usage, events=events,
                    seed=self.seed,
                    rationale=obj.get("rationale", {}),
                    implementation_path=str(
                        obj.get("implementation_path", "")).upper(),
                    research_category="confirmation",
                    code_summary=obj.get("code_summary", ""))
                self.tree.add(node)
                self._print_spend(it)
                return node
        self._maybe_record_axis_proposal(obj, it, events)
        code = obj["code"]

        # Verify the IMPLEMENTATION against the HYPOTHESIS before spending a
        # training run on it. A declaration is not evidence and a clean exit is
        # not evidence: this project has three nodes that "succeeded" by
        # applying a per-user monotone transform to the scores, which cannot
        # move a within-user ranking metric at all -- two of them returned
        # byte-identical metrics and both were recorded as successes.
        try:
            from . import mechanism_audit as mech
            ma = mech.audit(code, obj.get("hypothesis", ""),
                            str(obj.get("implementation_path", "A")),
                            menu_choices=obj.get("menu_choices") or {})
            events.append({"type": "mechanism_audit",
                           "verdict": ma["verdict"][:200],
                           "claimed": ma["mechanisms_claimed"],
                           "missing": ma["mechanisms_missing"],
                           "blocks": ma["blocks_scoring"]})
            if ma["blocks_scoring"]:
                print(f"  [mechanism audit] BLOCKED: {ma['verdict'][:110]}",
                      flush=True)
                node = Node(iteration_id=it,
                            parent_id=None if target is None else target.iteration_id,
                            action=action,
                            menu_choices=obj.get("menu_choices") or {},
                            hypothesis=obj.get("hypothesis", ""),
                            status="error", metrics=None,
                            error_trace=("BLOCKED BEFORE EXECUTION by the mechanism "
                                         "audit: " + ma["verdict"]),
                            tokens_used=0, wall_clock_seconds=0.0, timestamp=now(),
                            code_path="", decide_reason=reason, events=events)
                self.tree.add(node)
                return node
            if ma["mechanisms_missing"]:
                print(f"  [mechanism audit] warning: {ma['verdict'][:110]}",
                      flush=True)
        except Exception as e:            # never let the audit kill an iteration
            events.append({"type": "mechanism_audit_skipped",
                           "error": f"{type(e).__name__}: {str(e)[:160]}"})
        if self.inject_error_at is not None and it == self.inject_error_at:
            code += "\nraise RuntimeError('injected failure (harness robustness test)')\n"
            events.append({"type": "injected_error_for_testing",
                           "note": "harness appended a raise to exercise the debug path"})
            print(f"[iter {it}] NOTE: injecting deliberate failure (robustness test)", flush=True)

        res = run_solution(code, code_path, obj["menu_choices"], run_dir,
                           timeout_s=self.exec_timeout_s, seed=self.seed)
        # A preflight rejection never reached training, so it is not charged as
        # a training execution -- but a crash mid-training is, because that
        # compute is spent and unrecoverable.
        if not (res.error_trace and budget.PREFLIGHT_MARKER in res.error_trace):
            if getattr(self, "ledger", None):
                self.ledger.record_training(1, crashed=0 if res.ok else 1)
        if not res.ok:
            fc = failure_mod.classify(res.error_trace)
            events.append({"type": "execution_error",
                           "failure_class": fc["class"],
                           "retry_worthwhile": fc["retry_worthwhile"],
                           "needs_shrink": fc["needs_shrink"],
                           "likely_cause": fc["likely_cause"],
                           "error_head": (res.error_trace or "")[:300]})
            print(f"  [failure] class={fc['class']} "
                  f"retry_worthwhile={fc['retry_worthwhile']}", flush=True)
        diff_info = {"diff_path": "", "diff_sha256": ""}
        if os.path.exists(code_path):
            parent_code_path = target.code_path if target is not None else None
            diff_info = save_diff(code_path, parent_code_path, self.diffs_dir, it)
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

        best_before = self.tree.best()   # captured before add() can change it
        node = Node(iteration_id=it,
                    parent_id=None if target is None else target.iteration_id,
                    action=action, menu_choices=obj["menu_choices"],
                    hypothesis=obj["hypothesis"],
                    status="success" if res.ok else "error",
                    metrics=res.metrics, error_trace=res.error_trace,
                    tokens_used=sum(usage.values()),
                    wall_clock_seconds=res.wall_clock_seconds, timestamp=now(),
                    code_path=code_path, expected_effect=obj["expected_effect"],
                    decide_reason=reason, token_breakdown=usage, events=events,
                    diff_path=diff_info["diff_path"], diff_sha256=diff_info["diff_sha256"],
                    seed=self.seed, rationale=obj.get("rationale", {}),
                    implementation_path=str(obj.get("implementation_path","")).upper(),
                    research_category=str(obj.get("research_category","")).lower(),
                    code_summary=obj.get("code_summary",""))
        self._schedule_competition_followup(node)
        self.tree.add(node)
        self._record_experience(node, best_before)
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

    def _maybe_record_axis_proposal(self, obj: dict, it: int, events: list) -> None:
        """The agent may attach `proposed_axis` to any response. It is recorded
        as PENDING only -- never added to the live menu, which requires an
        explicit human `python3 -m agent.propose_axis --approve <id>`."""
        p = obj.get("proposed_axis")
        if not p:
            return
        try:
            pid = propose_axis.append_proposal(p, iteration_id=it)
            events.append({"type": "axis_proposed", "id": pid,
                           "axis_name": p.get("axis_name")})
            print(f"  [axis proposal #{pid}] '{p.get('axis_name')}' recorded as "
                  f"PENDING (needs human approval to become selectable)", flush=True)
        except Exception as e:
            events.append({"type": "axis_proposal_rejected", "error": str(e)[:300]})
            print(f"  [axis proposal REJECTED as malformed] {str(e)[:120]}", flush=True)

    def _record_experience(self, node: Node, best_before: Node | None) -> None:
        """One curated lesson per iteration, into agent/experience.md -- distinct
        from the full node already in the journal. Classifies by comparing this
        node's outcome to the running best captured just before it was added, so
        a future hypothesis prompt sees "this config crashed" / "this beat the
        best" / "this was a dead end" without re-reading the whole journal.
        """
        if not node.menu_choices:   # LLM-stage failure -- nothing config-specific
            return
        choices_s = json.dumps(node.menu_choices)
        if node.status != "success":
            # Stage E: record the CLASSIFIED fault, so the lesson carried
            # forward is "this class of bug, and how to avoid it" rather than
            # an opaque stack-trace line.
            fc = failure_mod.classify(node.error_trace)
            outcome, _title, body = failure_mod.as_knowledge(
                fc, node.iteration_id, node.menu_choices)
        else:
            # Graded against the NOISE FLOOR, not against zero. The thresholds
            # here were 1e-9 for HELPED and 1e-4 for DEAD_END while seed noise
            # on this benchmark is 0.0008 -- so a 0.00001 difference was written
            # into the agent's own memory as "HELPED", and a 0.125-sigma dip as
            # "DEAD_END". The memory was recording noise as findings, and every
            # later decision read it back as evidence.
            primary = node.metrics["primary"]
            if best_before is None:
                outcome = "HELPED"
                body = (f"menu_choices={choices_s} scored {primary:.4f} as the "
                        f"first scored node (no prior best to compare against).")
            else:
                prev = best_before.metrics["primary"]
                delta = primary - prev
                sigma = delta / BASELINE_SEED_STD
                if delta >= BASELINE_SEED_STD:
                    outcome = "HELPED"
                    body = (f"menu_choices={choices_s} raised valid primary to "
                            f"{primary:.4f} from {prev:.4f} "
                            f"({delta:+.5f} = {sigma:+.1f} sigma).")
                elif delta <= -BASELINE_SEED_STD:
                    outcome = "DEAD_END"
                    body = (f"menu_choices={choices_s} scored {primary:.4f} vs the "
                            f"then-best {prev:.4f} ({delta:+.5f} = {sigma:+.1f} "
                            f"sigma) -- worse beyond seed noise, not worth "
                            f"repeating as-is.")
                else:
                    outcome = "NEUTRAL"
                    body = (f"menu_choices={choices_s} scored {primary:.4f} vs the "
                            f"then-best {prev:.4f} ({delta:+.5f} = {sigma:+.1f} "
                            f"sigma) -- INSIDE the {BASELINE_SEED_STD} noise "
                            f"floor, so this says nothing either way. Treat as "
                            f"untested, not as evidence.")
        title = (node.hypothesis or f"{node.action} node {node.iteration_id}")[:100]
        append_entry(node.iteration_id, outcome, title, body)

    # ---------- parallel round (opt-in; sequential iterate() above is default) ----------
    def iterate_parallel(self) -> list[Node]:
        """K independent proposals for ONE decided action, dispatched
        concurrently (agent.executor.run_parallel_round, Part A's sandbox), then
        merged if >=2 beat the running best. decide_action() runs ONCE per
        round -- policy.py is untouched; the diversity across the K proposals
        comes from independent LLM completions of the SAME prompt, not from K
        different decided actions.
        """
        self._round_counter += 1
        round_id = f"round_{self._round_counter}"
        action, target, reason = decide_action(self.tree, draft_count=self.draft_count)
        if action == "debug" and target is not None and not (
                target.code_path and os.path.exists(target.code_path)):
            failed_id = target.iteration_id
            action, target = "draft", None
            reason = (f"node {failed_id} failed before any code was written "
                      f"(LLM-stage failure), so there is nothing to debug; "
                      f"drafting a fresh combination instead")
        print(f"[{round_id}] action={action}"
              f"{'' if target is None else f' target={target.iteration_id}'} "
              f"({self.parallel_k} workers) — {reason}", flush=True)

        best_before = self.tree.best()   # ALL K workers compare against this SAME
                                         # snapshot -- they're siblings generated
                                         # from one pre-round state, not a chain
        # ---- K LLM calls, each conditioned on its siblings' proposals ----
        # NOT K independent calls against one prompt: that was measured to
        # produce K identical proposals once the prompt is well-constrained
        # (see prompts.render_sibling_section). These calls were already
        # sequential -- only training is parallel -- so conditioning is free.
        proposals = []
        saw_success = False
        sibling_choices: list = []
        round_events: list = []
        extra = self._research_block(round_events) + "\n\n" + self._inspect_phase(round_events)
        for _w in range(self.parallel_k):
            w_prompt = build_prompt(action, target, reason, self.tree, self.menu,
                                    exec_timeout_s=self.exec_timeout_s,
                                    sibling_choices=sibling_choices,
                                    data_block=extra)
            try:
                obj, usage, llm_events = self.llm.structured_call(
                    w_prompt, validate_choices=self.menu.validate_choices)
                proposals.append({"ok": True, "obj": obj, "usage": usage,
                                  "events": llm_events})
                sibling_choices.append(obj["menu_choices"])
                saw_success = True
            except LLMError as e:
                proposals.append({"ok": False, "error": str(e), "usage": {},
                                  "events": [{"type": "llm_failure",
                                             "error": str(e)[:1000]}]})
        # Report whether the diversity mechanism actually worked this round --
        # a silently-degraded K-way round is exactly the failure that went
        # unnoticed before, so it is measured rather than assumed.
        _sigs = [json.dumps(c, sort_keys=True) for c in sibling_choices]
        n_distinct = len(set(_sigs))
        if _sigs:
            print(f"[{round_id}] worker diversity: {n_distinct}/{len(_sigs)} "
                  f"distinct proposals"
                  + ("" if n_distinct == len(_sigs)
                     else "  <-- DUPLICATES: workers wasted on the same experiment"),
                  flush=True)
        diversity_event = {"type": "worker_diversity", "round_id": round_id,
                           "distinct": n_distinct, "workers": len(_sigs)}
        if saw_success:
            self.consecutive_llm_failures = 0
        else:
            self.consecutive_llm_failures += 1
            self.last_llm_error = next(p["error"] for p in proposals if not p["ok"])

        # ---- journal LLM-stage failures immediately (nothing to execute) ----
        round_nodes: list[Node] = []
        exec_jobs = []
        for w, p in enumerate(proposals):
            if p["ok"]:
                continue
            it = self.tree.next_id()
            node = Node(iteration_id=it,
                       parent_id=None if target is None else target.iteration_id,
                       action=action, menu_choices={}, hypothesis="", status="error",
                       metrics=None, error_trace=f"LLM stage failed: {p['error']}",
                       tokens_used=0, wall_clock_seconds=0.0, timestamp=now(),
                       code_path="", decide_reason=reason, events=p["events"],
                       seed=self.seed, round_id=round_id)
            self.tree.add(node)
            round_nodes.append(node)
            print(f"[{round_id}] worker {w}: LLM stage failed (journaled as node {it})",
                 flush=True)

        # Allocate ids by hand, sequentially: next_id() reads off self.tree.nodes,
        # which doesn't change until add() runs, and none of these K nodes are
        # added until after run_parallel_round() -- calling next_id() again for
        # each one here would hand every successful worker the SAME id.
        next_it = self.tree.next_id()
        for w, p in enumerate(proposals):
            if not p["ok"]:
                continue
            it = next_it
            next_it += 1
            code_path = os.path.join(self.solutions_dir, f"node_{it:03d}.py")
            run_dir = os.path.join(self.runs_dir, f"node_{it:03d}")
            code = p["obj"]["code"]
            if self.inject_error_at is not None and it == self.inject_error_at:
                code += "\nraise RuntimeError('injected failure (harness robustness test)')\n"
                p["events"].append({"type": "injected_error_for_testing",
                                   "note": "harness appended a raise to exercise the debug path"})
            exec_jobs.append({"slot": w, "code": code, "code_path": code_path,
                             "menu_choices": p["obj"]["menu_choices"],
                             "run_dir": run_dir, "seed": self.seed,
                             "_iteration_id": it, "_proposal": p})

        # ---- dispatch the successful proposals concurrently, ONE shared lock ----
        results = run_parallel_round(
            [{"slot": j["slot"], "code": j["code"], "code_path": j["code_path"],
              "menu_choices": j["menu_choices"], "run_dir": j["run_dir"],
              "seed": j["seed"]} for j in exec_jobs],
            timeout_s=self.exec_timeout_s) if exec_jobs else []

        for j, res in zip(exec_jobs, results):
            it, p, obj = j["_iteration_id"], j["_proposal"], j["_proposal"]["obj"]
            events = list(p["events"])
            if not res.ok:
                fc = failure_mod.classify(res.error_trace)
                events.append({"type": "execution_error",
                              "failure_class": fc["class"],
                              "retry_worthwhile": fc["retry_worthwhile"],
                              "needs_shrink": fc["needs_shrink"],
                              "likely_cause": fc["likely_cause"],
                              "error_head": (res.error_trace or "")[:300]})
            diff_info = {"diff_path": "", "diff_sha256": ""}
            if os.path.exists(j["code_path"]):
                parent_code_path = target.code_path if target is not None else None
                diff_info = save_diff(j["code_path"], parent_code_path, self.diffs_dir, it)
            self._collect_resources(j["run_dir"])
            cost = self.spend.record(p["usage"])
            events.append({"type": "spend", "iteration_usd": round(cost, 6),
                          "run_total_usd": round(self.spend.total_usd, 6),
                          "provider": self.llm.provider, "model": self.llm.model,
                          "round_id": round_id, "worker": j["slot"]})
            events.append(diversity_event)
            events.extend(round_events)
            node = Node(iteration_id=it,
                       parent_id=None if target is None else target.iteration_id,
                       action=action, menu_choices=obj["menu_choices"],
                       hypothesis=obj["hypothesis"],
                       status="success" if res.ok else "error", metrics=res.metrics,
                       error_trace=res.error_trace, tokens_used=sum(p["usage"].values()),
                       wall_clock_seconds=res.wall_clock_seconds, timestamp=now(),
                       code_path=j["code_path"], expected_effect=obj["expected_effect"],
                       decide_reason=reason, token_breakdown=p["usage"], events=events,
                       diff_path=diff_info["diff_path"], diff_sha256=diff_info["diff_sha256"],
                       seed=self.seed, rationale=obj.get("rationale", {}),
                    implementation_path=str(obj.get("implementation_path","")).upper(),
                    research_category=str(obj.get("research_category","")).lower(),
                    code_summary=obj.get("code_summary",""),
                       round_id=round_id)
            self.tree.add(node)
            self._record_experience(node, best_before)
            round_nodes.append(node)
            outcome_s = (f"SUCCESS primary {res.metrics['primary']:.4f}" if res.ok
                        else f"ERROR {error_headline(res.error_trace)}")
            print(f"[{round_id}] worker {j['slot']} -> node {it}: {outcome_s}", flush=True)

        # ---- merge: only when >=2 candidates beat the PRE-round best ----
        beat_best = [n for n in round_nodes if n.status == "success" and
                    (best_before is None
                     or n.metrics["primary"] > best_before.metrics["primary"])]
        if len(beat_best) >= 2:
            top2 = sorted(beat_best, key=lambda n: -n.metrics["primary"])[:2]
            merge_node = self._attempt_merge(round_id, reason, top2)
            round_nodes.append(merge_node)
            best_individual = top2[0]
            if merge_node.status == "success" and \
                    merge_node.metrics["primary"] > best_individual.metrics["primary"]:
                print(f"[{round_id}] MERGE ACCEPTED: node {merge_node.iteration_id} "
                     f"(primary {merge_node.metrics['primary']:.4f}) beats best "
                     f"individual node {best_individual.iteration_id} "
                     f"({best_individual.metrics['primary']:.4f})", flush=True)
            else:
                why = ("merge execution failed" if merge_node.status != "success"
                      else "merge did not strictly beat the best individual candidate")
                print(f"[{round_id}] merge REJECTED ({why}); round falls back to "
                     f"best individual node {best_individual.iteration_id}", flush=True)

        self._print_spend(self._round_counter)
        return round_nodes

    def _attempt_merge(self, round_id: str, reason: str, top2: list[Node]) -> Node:
        """Coordinator LLM call over the round's top-2 candidates (both already
        beat the pre-round best). Executed via the plain SEQUENTIAL run_solution()
        -- by this point all K workers have already finished, so there is no
        concurrency left to manage for this one extra call.

        No special-casing for "the merge crashed": it's journaled like any other
        candidate, and ExperimentTree's existing best-tracking (a new node only
        becomes best if it STRICTLY exceeds the current best) already enforces
        "accept only if it strictly beats the best individual" for free, as long
        as the caller adds the round's individual nodes before this one.
        """
        merge_prompt = build_merge_prompt(top2[0], top2[1], reason, self.menu,
                                         exec_timeout_s=self.exec_timeout_s)
        it = self.tree.next_id()
        merged_from = [n.iteration_id for n in top2]
        try:
            obj, usage, llm_events = self.llm.structured_call(
                merge_prompt, validate_choices=self.menu.validate_choices)
        except LLMError as e:
            node = Node(iteration_id=it, parent_id=top2[0].iteration_id, action="merge",
                       menu_choices={}, hypothesis="", status="error", metrics=None,
                       error_trace=f"merge LLM stage failed: {e}", tokens_used=0,
                       wall_clock_seconds=0.0, timestamp=now(), code_path="",
                       decide_reason=f"merge of nodes {merged_from}",
                       events=[{"type": "llm_failure", "error": str(e)[:1000]}],
                       seed=self.seed, round_id=round_id, merged_from=merged_from)
            self.tree.add(node)
            return node

        code_path = os.path.join(self.solutions_dir, f"node_{it:03d}.py")
        run_dir = os.path.join(self.runs_dir, f"node_{it:03d}")
        res = run_solution(obj["code"], code_path, obj["menu_choices"], run_dir,
                          timeout_s=self.exec_timeout_s, seed=self.seed)
        events = list(llm_events)
        if not res.ok:
            events.append({"type": "execution_error",
                          "error_head": (res.error_trace or "")[:300]})
        diff_info = (save_diff(code_path, top2[0].code_path, self.diffs_dir, it)
                    if os.path.exists(code_path) else {"diff_path": "", "diff_sha256": ""})
        self._collect_resources(run_dir)
        cost = self.spend.record(usage)
        events.append({"type": "spend", "iteration_usd": round(cost, 6),
                      "run_total_usd": round(self.spend.total_usd, 6),
                      "provider": self.llm.provider, "model": self.llm.model})
        events.append({"type": "merge_attempt", "merged_from": merged_from})

        node = Node(iteration_id=it, parent_id=top2[0].iteration_id, action="merge",
                   menu_choices=obj["menu_choices"], hypothesis=obj["hypothesis"],
                   status="success" if res.ok else "error", metrics=res.metrics,
                   error_trace=res.error_trace, tokens_used=sum(usage.values()),
                   wall_clock_seconds=res.wall_clock_seconds, timestamp=now(),
                   code_path=code_path, expected_effect=obj["expected_effect"],
                   decide_reason=f"merge of nodes {merged_from}", token_breakdown=usage,
                   events=events, diff_path=diff_info["diff_path"],
                   diff_sha256=diff_info["diff_sha256"], seed=self.seed,
                   rationale=obj.get("rationale", {}), round_id=round_id,
                   merged_from=merged_from)
        self.tree.add(node)
        # compared against the best INDIVIDUAL (top2[0]), not the pre-round best --
        # this is what "did the merge itself succeed" means, distinct from
        # "did a worker beat the running best" (already recorded per-worker above)
        self._record_experience(node, top2[0])
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
        epsilon = getattr(self, "active_epsilon", EPSILON)
        n_window = getattr(self, "active_n_converge", N_CONVERGE)
        print(f"agent run started: max_iterations={self.max_iterations}, "
              f"wall_clock_limit={self.wall_clock_limit_s/3600:.1f}h, "
              f"llm={self.llm.provider}:{self.llm.model}, "
              f"draft_count={self.draft_count}, "
              f"spend_ceiling=${self.spend.ceiling_usd:.2f} "
              f"({self.spend.rates.describe(self.llm.provider, self.llm.model)}), "
              f"epsilon={epsilon:.5f} "
              f"({epsilon / BASELINE_SEED_STD:.2f} sigma), N={n_window}, "
              f"profile={'competition/official' if self.competition_mode else 'research/internal'}"
              + (f", PARALLEL MODE k={self.parallel_k}" if self.parallel_k else ""))
        nodes_at_start = len(self.tree.nodes)
        self._ensure_competition_baseline()
        stop = self.stop_reason()
        if stop is not None:
            # Loud, because the failure mode is silent and expensive in wasted
            # time: resuming into an ALREADY-converged journal makes run() a
            # no-op that still exits 0 and still writes a final_summary.json
            # reporting the pre-existing journal's iteration count, so it looks
            # like a successful run that simply had nothing to add. Hit twice
            # during development before this warning existed.
            print(f"\n!!! NO ITERATIONS WILL RUN: the existing journal "
                  f"({nodes_at_start} node(s)) already satisfies a stop "
                  f"condition before this run began.\n    reason: {stop}\n"
                  f"    Nothing was generated, trained, or spent. To start a new "
                  f"search from iteration 0, re-run with --fresh (archives "
                  f"logs/ to logs/archive_<ts>/, does not delete it).\n", flush=True)
        while stop is None:
            if self.parallel_k:
                self.iterate_parallel()
            else:
                self.iterate()
            stop = self.stop_reason()
        summary = self.finish(stop)
        summary["iterations_this_run"] = len(self.tree.nodes) - nodes_at_start
        return summary

    def finish(self, stop: str) -> dict:
        best = self.tree.best()
        interventions = 0
        ipath = os.path.join(self.log_dir, "interventions.jsonl")
        if os.path.exists(ipath):
            with open(ipath) as fh:
                interventions = sum(1 for line in fh if line.strip())
        summary = {
            "stop_reason": stop,
            # Keep raw journal records distinct from charged research decisions:
            # preflight rejections are journalled for auditability but consume
            # neither compute nor an outer-loop research decision.
            "journal_nodes": len(self.tree.nodes),
            "iterations_used": sum(1 for n in self.tree.nodes
                                   if budget.consumes_budget(n)),
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
            "budget_ledger": (self.ledger.as_dict()
                              if getattr(self, "ledger", None) else {}),
            "budget_counting_note": budget.COUNTING_NOTE,
            "run_profile": ("competition" if self.competition_mode
                            else "research"),
            "convergence_rule": {
                "epsilon": self.active_epsilon,
                "N": self.active_n_converge,
                "source": ("organizer" if self.competition_mode
                           else "internal research controller"),
                "counted_iterations": "scored (successful) only"},
        }
        if self.competition_mode:
            from . import convergence_report
            summary["official_convergence"] = convergence_report.report(
                [json.loads(n.to_json()) for n in self.tree.nodes])
            eligible = summary["official_convergence"]["eligible_checkpoint"]
            record_path = os.path.join(self.log_dir, "ensemble_results.json")
            if os.path.exists(record_path):
                with open(record_path) as fh:
                    record = json.load(fh)
                record["official_eligible"] = bool(
                    eligible.get("determined")
                    and record.get("source_node") == eligible.get("eligible_node"))
                record["official_convergence_node"] = eligible.get(
                    "converged_at_node")
                record["official_eligible_node"] = eligible.get("eligible_node")
                record["official_eligible_primary"] = eligible.get(
                    "eligible_primary")
                tmp = record_path + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(record, fh, indent=2)
                os.replace(tmp, record_path)
                summary["canonical_artifact_official_eligible"] = record[
                    "official_eligible"]
        with open(os.path.join(self.log_dir, "final_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        print("\n=== RUN FINISHED ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)
        if best is not None:
            print(f"\nbest solution: {os.path.join(self.log_dir, 'best_solution.py')}", flush=True)
            print("generate a submission with: python3 -m agent.make_submission", flush=True)
        return summary
