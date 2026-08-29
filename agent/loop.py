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
                 test_model: bool = False, parallel_k: int | None = None,
                 min_branching_iterations: int = 0,
                 enable_data_tools: bool = False,
                 enable_research_state: bool = False,
                 n_candidates: int = 0):
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
            left = max(0, self.max_iterations - len(self.tree.nodes))
            cand_mod.score_candidates(cands, history=hist, dead_ends=dead,
                                      state=st, budget_left=left,
                                      objective=objective)
            winner, ranked = cand_mod.select(cands)
            trace = cand_mod.render_trace(winner, ranked, objective, st, left)
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
            left = max(0, self.max_iterations - len(self.tree.nodes))
            d = decide_category(st, nodes, iteration_budget_left=left)
            events.append({"type": "research_category", "category": d["category"],
                           "scores": d["scores"], "reason": d["reason"]})
            self._current_objective = d["category"]
            print(f"  [research policy] objective={d['category']} "
                  f"({d['reason']})", flush=True)
            return st.render() + "\n\n" + render_decision(d)
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

    # ---------- one iteration ----------
    def iterate(self) -> Node:
        it = self.tree.next_id()
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
        extra = self._research_block(events) + "\n\n" + self._inspect_phase(events)
        winner, trace = self._plan_candidates(action, target, reason, events,
                                              extra, self._current_objective)
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
        self._maybe_record_axis_proposal(obj, it, events)
        code = obj["code"]
        if self.inject_error_at is not None and it == self.inject_error_at:
            code += "\nraise RuntimeError('injected failure (harness robustness test)')\n"
            events.append({"type": "injected_error_for_testing",
                           "note": "harness appended a raise to exercise the debug path"})
            print(f"[iter {it}] NOTE: injecting deliberate failure (robustness test)", flush=True)

        res = run_solution(code, code_path, obj["menu_choices"], run_dir,
                           timeout_s=self.exec_timeout_s, seed=self.seed)
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
            primary = node.metrics["primary"]
            if best_before is None or primary > best_before.metrics["primary"] + 1e-9:
                outcome = "HELPED"
                prev = f"{best_before.metrics['primary']:.4f}" if best_before else "n/a"
                body = (f"menu_choices={choices_s} raised valid primary to "
                       f"{primary:.4f} (previous best {prev}).")
            elif primary < best_before.metrics["primary"] - 1e-4:
                outcome = "DEAD_END"
                body = (f"menu_choices={choices_s} scored {primary:.4f}, below the "
                       f"then-current best {best_before.metrics['primary']:.4f} -- "
                       f"not worth repeating as-is.")
            else:
                outcome = "NEUTRAL"
                body = (f"menu_choices={choices_s} scored {primary:.4f}, no clear "
                       f"change vs the running best.")
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
        print(f"agent run started: max_iterations={self.max_iterations}, "
              f"wall_clock_limit={self.wall_clock_limit_s/3600:.1f}h, "
              f"llm={self.llm.provider}:{self.llm.model}, "
              f"draft_count={self.draft_count}, "
              f"spend_ceiling=${self.spend.ceiling_usd:.2f} "
              f"({self.spend.rates.describe(self.llm.provider, self.llm.model)}), "
              f"ε={EPSILON}, N={N_CONVERGE}"
              + (f", PARALLEL MODE k={self.parallel_k}" if self.parallel_k else ""))
        nodes_at_start = len(self.tree.nodes)
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
