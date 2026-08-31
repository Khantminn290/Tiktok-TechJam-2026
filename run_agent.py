"""Entrypoint for the autonomous ML research agent.

    python3 run_agent.py --smoke                 # cheap 3-iteration plumbing check
    python3 run_agent.py --max-spend-usd 15      # a real run (set the ceiling yourself)
    python3 run_agent.py --inject-error-at 2     # robustness test: break iteration 2
    python3 run_agent.py --reseed-top 3          # no LLM: reseed the top 3 nodes,
                                                  # report mean/std (see agent/reseed.py)

Provider, model and API key come from .env (see .env.example), falling back to
config/llm_config.json for provider and model name only — never the key.

The safety-gate override `allow_locked_options` can ONLY be set in
config/agent_config.json by a human — doing so should be logged:
    python3 -m agent.interventions "unlocked <option>: <why>"
"""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from agent.llm import LLMError, load_dotenv, load_llm_config, preflight  # noqa: E402
from agent.loop import AgentLoop  # noqa: E402
from agent.pricing import RateTable  # noqa: E402


# Artifacts backing the SUBMITTED result. These survive --fresh deliberately:
# a previous headline became unreproducible precisely because --fresh archived
# the member arrays while the summary JSON quoting them stayed behind, leaving
# a number on the deliverable that could no longer be recomputed. The submitted
# ensemble is a separate, later artifact from any single search run, so a new
# search must not carry it off.
SUBMISSION_ARTIFACTS = ("final_ensemble", "ensemble_results.json",
                        "submission_history",
                        "ab_test",              # A/B evidence, not a search product
                        "feature_registry.jsonl",  # accumulated feature research
                        # Research memory is KNOWLEDGE, not a search product. A
                        # --fresh run restarts the SEARCH; it must not give the
                        # agent amnesia, or every run re-derives what is already
                        # known and the memory subsystem is inert in exactly the
                        # runs that evaluate it.
                        "research_memory.jsonl",
                        "opus_research")           # the research journal itself


def archive_logs(log_dir: str) -> None:
    """Move a previous run's logs aside so the next run starts at iteration 0."""
    import shutil
    import time
    if not os.path.exists(os.path.join(log_dir, "journal.jsonl")):
        return
    dest = os.path.join(log_dir, f"archive_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dest, exist_ok=True)
    kept = []
    for name in os.listdir(log_dir):
        if name.startswith("archive_"):
            continue
        if name in SUBMISSION_ARTIFACTS:
            kept.append(name)
            continue
        shutil.move(os.path.join(log_dir, name), os.path.join(dest, name))
    print(f"archived previous run to {dest}")
    if kept:
        print(f"kept submission artifacts in place: {', '.join(sorted(kept))}")


def main():
    # keep progress visible when stdout is piped/tee'd (long autonomous runs)
    sys.stdout.reconfigure(line_buffering=True)
    load_dotenv()          # before any os.environ read below
    cfg_path = os.path.join(_ROOT, "config", "agent_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            cfg = json.load(fh)
    llm_cfg = load_llm_config()

    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iterations", type=int,
                    default=cfg.get("max_iterations", 50))
    ap.add_argument("--wall-clock-limit-h", type=float,
                    default=cfg.get("wall_clock_limit_h", 6.0))
    ap.add_argument("--llm-model", default=None,
                    help="override the model from .env / config/llm_config.json")
    ap.add_argument("--max-spend-usd", type=float, default=2.0,
                    help="hard LLM spend ceiling for this run (default 2.0). The "
                         "loop stops before the call that would exceed it.")
    # env (from .env) wins over config/llm_config.json; --draft-count wins over both
    _draft_default = llm_cfg.get("draft_count", 7)
    if str(os.environ.get("DRAFT_COUNT", "")).strip().isdigit():
        _draft_default = int(os.environ["DRAFT_COUNT"])
    ap.add_argument("--draft-count", type=int, default=_draft_default,
                    help="diverse initial drafts before committing to a lineage "
                         "(env DRAFT_COUNT, then config/llm_config.json)")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap plumbing check: 3 iterations, $1 ceiling, TEST_MODEL")
    ap.add_argument("--exec-timeout", type=int,
                    default=cfg.get("exec_timeout_s", 1200))
    ap.add_argument("--seed", type=int, default=cfg.get("seed", 0))
    ap.add_argument("--inject-error-at", type=int, default=None,
                    help="deliberately break the generated code at this iteration "
                         "(harness robustness test)")
    ap.add_argument("--no-verify-key", action="store_true",
                    help="skip the free auth probe in preflight (offline use)")
    ap.add_argument("--fresh", action="store_true",
                    help="archive any existing logs/ into logs/archive_<ts>/ and start "
                         "iteration 0 from scratch (default: resume the journal, which "
                         "is how a crashed run continues where it left off)")
    ap.add_argument("--reseed-top", type=int, default=None,
                    help="statistical rigor mode: re-run the top-N journal nodes "
                         "across --reseed-seeds different seeds and report mean/std "
                         "(no LLM calls; see agent/reseed.py). Runs instead of a "
                         "normal agent loop.")
    ap.add_argument("--reseed-seeds", type=int, default=5,
                    help="seeds per node for --reseed-top (default 5, min 2)")
    ap.add_argument("--reseed-wall-clock-limit-h", type=float,
                    default=cfg.get("wall_clock_limit_h", 6.0),
                    help="wall-clock ceiling for --reseed-top. This is its OWN "
                         "allowance, separate from a normal run's budget -- reseeding "
                         "happens after a run has already finished and costs no LLM "
                         "spend, only training time (default: config/agent_config.json "
                         "wall_clock_limit_h).")
    ap.add_argument("--data-tools", action="store_true",
                    help="let the agent request read-only data measurements "
                         "(get_feature_stats / get_label_rate_by_segment / "
                         "get_within_user_auc / get_user_history_stats) before "
                         "hypothesizing. Sandboxed, train/valid only, capped "
                         "per iteration. See agent/inspect.py.")
    ap.add_argument("--research-state", action="store_true",
                    help="give the agent a compact derived research state "
                         "(agent/research_state.py) plus an evidence-reactive "
                         "research objective (agent/research_policy.py) instead "
                         "of raw history. Objectives: exploration / exploitation "
                         "/ ablation / confirmation / integration.")
    ap.add_argument("--feature-discovery", action="store_true",
                    help="autonomous FEATURE RESEARCH: each iteration the agent "
                         "may propose a new feature from the evidence, write its "
                         "builder, and have it probed (leakage, within-user "
                         "variation, redundancy, incremental signal) BEFORE any "
                         "training run is spent. Results go to "
                         "logs/feature_registry.jsonl so a failed feature is "
                         "never reproposed. See agent/feature_lab.py.")
    ap.add_argument("--n-candidates", type=int, default=0,
                    help="multi-candidate planning: generate N candidate "
                         "experiments in ONE call, score them deterministically "
                         "(agent/candidates.py) and implement the winner. 0/1 "
                         "keeps the old single-proposal behaviour. This is what "
                         "makes Path B a scoreable option rather than something "
                         "the planner never generates.")
    ap.add_argument("--min-branching-iterations", type=int, default=0,
                    help="convergence cannot fire until the policy has actually "
                         "executed improve/debug/crossover at least this many "
                         "times (and an improve, plus a debug if any node "
                         "errored). Budget caps still apply. Targets the "
                         "measured gap that only `draft` ever fired.")
    ap.add_argument("--competition", action="store_true",
                    help="one explicit profile that turns on the capabilities a "
                         "competition run should demonstrate (research state, "
                         "data tools, feature discovery, multi-candidate "
                         "planning, branching) with conservative resource caps. "
                         "Explicit CLI flags always win; the fully resolved "
                         "configuration is printed before any spend.")
    ap.add_argument("--max-training-runs", type=int, default=None,
                    help="cap on TOTAL training executions. An outer iteration "
                         "is not one training run: a paired 3-seed confirmation "
                         "is six. Defaults to unlimited outside --competition.")
    ap.add_argument("--parallel-k", type=int, default=None,
                    help="opt-in parallel exploration: each iteration dispatches K "
                         "worker proposals simultaneously in isolated git worktrees "
                         "(agent/worktree.py), merging via a coordinator LLM call when "
                         "2+ beat the running best in the same round. Sequential "
                         "(K=1, today's behavior) remains the default when omitted.")
    a = ap.parse_args()

    # Profile resolution happens BEFORE anything expensive: an unsafe or
    # contradictory combination should cost nothing to discover.
    from agent import profiles
    resolved = profiles.resolve(a)
    problems = profiles.validate(a, resolved)
    if problems:
        sys.exit("refusing to start:\n" + "\n".join(f"  - {p}" for p in problems))
    if a.competition:
        print(profiles.render(a, resolved))

    if a.smoke:
        a.max_iterations = min(a.max_iterations, 3)
        a.max_spend_usd = min(a.max_spend_usd, 1.0)

    # ---- preflight: fail fast, before any expensive work ----
    data_dir = os.path.join(_ROOT, "kuairand-starter-kit", "KuaiRand-Pure", "data")
    if not os.path.exists(os.path.join(data_dir, "log_standard_4_08_to_4_21_pure.csv")):
        sys.exit(f"preflight failed: dataset not found at {data_dir}\n"
                 f"  download it first (see README Quickstart).")
    try:
        from agent.executor import assert_not_root
        assert_not_root()
    except RuntimeError as e:
        sys.exit(f"preflight failed: {e}")

    if a.reseed_top is not None:
        # No LLM involved -- re-executes already-generated scripts at different
        # seeds -- so the LLM preflight/auth probe and AgentLoop below are
        # entirely skipped for this mode.
        from agent.reseed import run_reseed
        run_reseed(root=_ROOT, top_n=a.reseed_top, n_seeds=a.reseed_seeds,
                  wall_clock_limit_h=a.reseed_wall_clock_limit_h,
                  exec_timeout_s=a.exec_timeout)
        return

    try:
        info = preflight(test=a.smoke, verify_key=not a.no_verify_key)
    except LLMError as e:
        sys.exit(f"preflight failed: {e}")

    model = a.llm_model or info["model"]
    rates = RateTable()
    rate, known = rates.lookup(info["provider"], model)
    print(f"preflight OK — provider={info['provider']} model={model} "
          f"(key from ${info['key_var']}"
          f"{', verified' if info.get('key_verified') else ', NOT verified'})")
    print(f"  rate card: {rates.describe(info['provider'], model)}")
    print(f"  spend ceiling: ${a.max_spend_usd:.2f} | draft_count={a.draft_count} "
          f"| max_iterations={a.max_iterations}")
    if not known:
        print(f"  WARNING: '{model}' is not in config/model_rates.json. The budget "
              f"guard is using the deliberately-high fallback rate, so the run will "
              f"stop early. Add the real rate to config/model_rates.json.")
    if a.smoke:
        print("  SMOKE TEST: plumbing check only — not a scored run.")

    if a.fresh:
        archive_logs(os.path.join(_ROOT, "logs"))

    loop = AgentLoop(
        root=_ROOT,
        llm_model=model,
        max_iterations=a.max_iterations,
        wall_clock_limit_h=a.wall_clock_limit_h,
        exec_timeout_s=a.exec_timeout,
        seed=a.seed,
        inject_error_at=a.inject_error_at,
        allow_locked_options=bool(cfg.get("allow_locked_options", False)),
        max_spend_usd=a.max_spend_usd,
        draft_count=a.draft_count,
        test_model=a.smoke,
        parallel_k=a.parallel_k,
        min_branching_iterations=a.min_branching_iterations,
        enable_data_tools=a.data_tools,
        enable_research_state=a.research_state,
        enable_feature_discovery=a.feature_discovery,
        n_candidates=a.n_candidates,
        max_training_runs=a.max_training_runs,
        competition_mode=a.competition,
    )
    summary = loop.run()

    if a.smoke:
        # A smoke test checks the PLUMBING, not whether the model's code happened
        # to work. Without this, three buggy-but-correctly-journaled scripts look
        # identical to a broken setup, and a teammate concludes wrongly.
        nodes = loop.tree.nodes
        llm_ok = sum(1 for n in nodes if n.menu_choices)
        ran = sum(1 for n in nodes if n.code_path and os.path.exists(n.code_path))
        scored = sum(1 for n in nodes if n.status == "success")
        plumbing_ok = llm_ok > 0 and ran > 0 and summary["spend"]["total_usd"] > 0
        print("\n=== SMOKE TEST VERDICT ===")
        print(f"  provider auth + model reachable : {'PASS' if llm_ok else 'FAIL'}"
              f"  ({llm_ok}/{len(nodes)} iterations got a valid model response)")
        print(f"  script generation + execution   : {'PASS' if ran else 'FAIL'}"
              f"  ({ran} script(s) written and run)")
        print(f"  journal + spend accounting      : "
              f"{'PASS' if summary['spend']['total_usd'] > 0 else 'FAIL'}"
              f"  (${summary['spend']['total_usd']:.4f} tracked)")
        print(f"  -> PLUMBING {'OK' if plumbing_ok else 'BROKEN'}")
        print(f"  ({scored}/{len(nodes)} generated scripts also produced valid "
              f"metrics — that is model-code quality, not setup, and a smoke test "
              f"on the cheap TEST_MODEL often has failures here. Errors above are "
              f"only a setup problem if PLUMBING is BROKEN.)")
        if not plumbing_ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
