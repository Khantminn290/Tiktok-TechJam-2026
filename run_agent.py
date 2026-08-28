"""Entrypoint for the autonomous ML research agent.

    python3 run_agent.py --smoke                 # cheap 3-iteration plumbing check
    python3 run_agent.py --max-spend-usd 15      # a real run (set the ceiling yourself)
    python3 run_agent.py --inject-error-at 2     # robustness test: break iteration 2

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


def archive_logs(log_dir: str) -> None:
    """Move a previous run's logs aside so the next run starts at iteration 0."""
    import shutil
    import time
    if not os.path.exists(os.path.join(log_dir, "journal.jsonl")):
        return
    dest = os.path.join(log_dir, f"archive_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dest, exist_ok=True)
    for name in os.listdir(log_dir):
        if name.startswith("archive_"):
            continue
        shutil.move(os.path.join(log_dir, name), os.path.join(dest, name))
    print(f"archived previous run to {dest}")


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
    a = ap.parse_args()

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
