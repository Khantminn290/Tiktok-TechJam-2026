"""Harness self-tests — no LLM calls, no training. Run: python3 tests/test_harness.py

Covers the pieces that must be right before an autonomous run is trusted:
the safety gate, cross-axis validity checks, the search policy's branching
decisions, the convergence rule, and the executor's contract enforcement.
"""
import json
import os
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agent.contracts import ExperimentTree, Node, error_headline  # noqa: E402
from agent.executor import run_solution  # noqa: E402
from agent.loop import EPSILON, N_CONVERGE, AgentLoop  # noqa: E402
from agent.menu import Menu, MenuError  # noqa: E402
from agent.policy import decide_action  # noqa: E402

MENU_PATH = os.path.join(_ROOT, "config", "modification_menu.json")
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if not detail else '  — ' + detail}")


def expect_menu_error(menu, choices, label):
    try:
        menu.validate_choices(choices)
        check(label, False, "accepted an invalid combination")
        return ""
    except MenuError as e:
        check(label, True, str(e)[:70])
        return str(e)


# ---------------------------------------------------------------- safety gate
def test_safety_gate():
    print("\n[safety gate]")
    m = Menu(MENU_PATH)
    base = m.default_choices()
    m.validate_choices(base)
    check("baseline choices validate", True)

    locked = [(axis, opt)
              for axis in m.axis_names()
              for opt in m.options(axis)
              if m.is_locked(axis, opt)]
    check("menu declares locked options", len(locked) >= 2, f"{locked}")
    for axis, opt in locked:
        msg = expect_menu_error(m, {**base, axis: opt},
                                f"locked option rejected: {axis}={opt}")
        check(f"  rejection names the gate ({opt})", "LOCKED" in msg)
        check(f"  {opt} absent from selectable options",
              opt not in m.selectable_options(axis))
        check(f"  {opt} marked LOCKED in the LLM prompt",
              f"{opt} [LOCKED" in m.render_for_prompt())

    unlocked = Menu(MENU_PATH, allow_locked_options=True)
    ok = True
    for axis, opt in locked:
        try:
            unlocked.validate_choices({**base, axis: opt})
        except MenuError:
            ok = False
    check("human override flag unlocks them", ok)


# ------------------------------------------------------- cross-axis validity
def test_validity():
    print("\n[cross-axis validity]")
    m = Menu(MENU_PATH)
    base = m.default_choices()
    expect_menu_error(m, {**base, "user_history": "din_attention"},
                      "din_attention without an MLP model rejected")
    m.validate_choices({**base, "user_history": "din_attention",
                        "model": "deepfm_mlp"})
    check("din_attention + deepfm_mlp accepted", True)
    expect_menu_error(m, {**base, "multitask": "censored_watch_time",
                          "loss": "bpr_pairwise"},
                      "censored watch-time with a pure pairwise loss rejected")
    expect_menu_error(m, {**base, "loss": "focal_loss"}, "unknown option rejected")
    expect_menu_error(m, {k: v for k, v in base.items() if k != "loss"},
                      "missing axis rejected")
    expect_menu_error(m, {**base, "nonsense_axis": "x"}, "unknown axis rejected")


# -------------------------------------------------------------- search policy
def _node(i, status, primary=None, action="draft", parent=None, choices=None):
    return Node(i, parent, action, choices or {"loss": "pointwise_logloss"},
                "hypothesis", status,
                None if primary is None else
                {"GAUC": primary, "nDCG@5": primary, "primary": primary},
                None if status == "success" else "boom\nValueError: broke",
                100, 1.0, time.time(), "")


def test_policy():
    print("\n[search policy]")
    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        check("empty tree drafts", decide_action(t)[0] == "draft")
        t.add(_node(0, "error"))
        check("error triggers debug", decide_action(t)[0] == "debug")
        t.add(_node(1, "error", action="debug", parent=0))
        t.add(_node(2, "error", action="debug", parent=1))
        act, _, reason = decide_action(t)
        check("hopeless debug chain is abandoned", act == "draft", reason[:60])

    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        t.add(_node(0, "success", 0.60))
        t.add(_node(1, "success", 0.61, choices={"loss": "bpr_pairwise"}))
        t.add(_node(2, "success", 0.605))
        t.add(_node(3, "success", 0.607))
        act, target, _ = decide_action(t)
        check("improves the global best once enough drafts exist",
              act == "improve" and target.iteration_id == 1)
        for i in (4, 5, 6):
            t.add(_node(i, "success", 0.606, action="improve", parent=1))
        act, target, reason = decide_action(t)
        check("branches away from an exhausted best",
              act == "improve" and target.iteration_id != 1,
              f"target={target.iteration_id}")
        check("branch decision is logged with a reason", "exhausted" in reason)


# ----------------------------------------------------------- convergence rule
def test_convergence():
    print("\n[convergence rule]")
    with tempfile.TemporaryDirectory() as td:
        loop = AgentLoop.__new__(AgentLoop)   # no LLM/menu construction needed
        loop.tree = ExperimentTree(td)
        loop.max_iterations = 50
        loop.wall_clock_limit_s = 6 * 3600
        loop.run_started = time.time()
        check(f"rule constants are the official ones (ε={EPSILON}, N={N_CONVERGE})",
              EPSILON == 0.002 and N_CONVERGE == 3)
        for i, p in enumerate([0.60, 0.62, 0.64, 0.66]):
            loop.tree.add(_node(i, "success", p))
        check("still improving -> not converged", not loop.converged()[0])
        for i, p in enumerate([0.6601, 0.6602, 0.6603], start=4):
            loop.tree.add(_node(i, "success", p))
        conv, msg = loop.converged()
        check("flat for N scored iterations -> converged", conv, msg[:60])

    with tempfile.TemporaryDirectory() as td:
        loop = AgentLoop.__new__(AgentLoop)
        loop.tree = ExperimentTree(td)
        loop.max_iterations = 50
        loop.wall_clock_limit_s = 6 * 3600
        loop.run_started = time.time()
        loop.tree.add(_node(0, "success", 0.60))
        for i in range(1, 5):
            loop.tree.add(_node(i, "error"))
        check("errors alone never trigger convergence", not loop.converged()[0])
        loop.max_iterations = 5
        check("iteration cap stops the run",
              "iteration cap" in (loop.stop_reason() or ""))


# ------------------------------------------------------- executor robustness
def test_executor():
    print("\n[executor contract enforcement]")
    with tempfile.TemporaryDirectory() as td:
        cases = {
            "syntax error": "def broken(:\n",
            "runtime exception": "raise ValueError('kaboom')\n",
            "exits 0 writing nothing": "import sys; sys.exit(0)\n",
            "writes malformed metrics": (
                "import argparse,os,json\n"
                "p=argparse.ArgumentParser();p.add_argument('--menu-choices');"
                "p.add_argument('--output-dir');p.add_argument('--seed');a=p.parse_args()\n"
                "os.makedirs(a.output_dir,exist_ok=True)\n"
                "json.dump({'GAUC':float('nan'),'nDCG@5':1,'primary':1},"
                "open(os.path.join(a.output_dir,'metrics.json'),'w'))\n"),
        }
        for label, code in cases.items():
            r = run_solution(code, os.path.join(td, "s.py"), {}, os.path.join(td, "r"),
                             timeout_s=60)
            check(f"{label} -> error, not a crash",
                  (not r.ok) and bool(r.error_trace), error_headline(r.error_trace, 60))
        r = run_solution("import time; time.sleep(30)\n", os.path.join(td, "s.py"),
                         {}, os.path.join(td, "r"), timeout_s=3)
        check("timeout -> killed and reported",
              (not r.ok) and "TIMEOUT" in (r.error_trace or ""))


# ------------------------------------------------------------- crossover move
def test_crossover():
    print("\n[crossover move]")
    from agent.policy import crossover_partner

    A = {"loss": "bpr_pairwise", "user_history": "none"}
    B = {"loss": "pointwise_logloss", "user_history": "mean_pool_positives"}

    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        # two distinct lineages, BOTH exhausted -> nothing left to extend
        t.add(_node(0, "success", 0.610, choices=dict(A)))
        t.add(_node(1, "success", 0.605, choices=dict(B)))
        for i in (2, 3, 4):
            t.add(_node(i, "success", 0.606, action="improve", parent=0, choices=dict(A)))
        for i in (5, 6, 7):
            t.add(_node(i, "success", 0.601, action="improve", parent=1, choices=dict(B)))
        act, target, reason = decide_action(t)
        check("both lineages exhausted -> crossover", act == "crossover",
              f"got {act}")
        check("crossover extends from the better parent",
              target is not None and target.iteration_id == 0)
        partner = crossover_partner(t, target)
        check("partner is a DISTINCT successful config",
              partner is not None and partner.menu_choices != target.menu_choices,
              f"partner={None if partner is None else partner.iteration_id}")
        check("crossover reason names both parents",
              "crossing node 0" in reason and "node 1" in reason)
        check("crossover is deterministic",
              crossover_partner(t, target).iteration_id == partner.iteration_id)
        act2, _, _ = decide_action(t, allow_crossover=False)
        check("crossover can be disabled", act2 != "crossover", f"got {act2}")

    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        # only ONE distinct config -> nothing to cross with, must not crossover
        t.add(_node(0, "success", 0.61, choices=dict(A)))
        for i in (1, 2, 3):
            t.add(_node(i, "success", 0.60, action="improve", parent=0, choices=dict(A)))
        act, _, _ = decide_action(t)
        check("no distinct partner -> never crossover", act != "crossover",
              f"got {act}")

    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        t.add(_node(0, "success", 0.61, choices=dict(A)))
        t.add(_node(1, "success", 0.60, choices=dict(B)))
        t.add(_node(2, "success", 0.59, choices=dict(A)))
        t.add(_node(3, "success", 0.58, choices=dict(B)))
        act, target, _ = decide_action(t)
        check("healthy best is still extended, not crossed",
              act == "improve" and target.iteration_id == 0, f"got {act}")

    # draft_count is configurable without changing default behaviour
    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        for i in range(5):
            t.add(_node(i, "success", 0.60 + i * 0.001))
        check("default draft_count keeps old behaviour (improve at 5 nodes)",
              decide_action(t)[0] == "improve")
        check("raised draft_count keeps drafting",
              decide_action(t, draft_count=8)[0] == "draft")


# --------------------------------------------------------- spend ceiling abort
def test_spend_ceiling():
    print("\n[spend ceiling]")
    from agent.pricing import RateTable, SpendTracker

    rates = RateTable()
    known, is_known = rates.lookup("openai", "gpt-5.4")
    check("known model is priced from the table", is_known and known["input"] > 0)
    _, unknown_ok = rates.lookup("openai", "totally-made-up-model")
    check("unknown model is flagged", not unknown_ok)
    fb, _ = rates.lookup("openai", "totally-made-up-model")
    check("unknown model prices HIGH (fails safe)",
          fb["input"] >= known["input"] * 4,
          f"fallback ${fb['input']}/1M vs known ${known['input']}/1M")
    check("dated snapshot resolves to its base model",
          rates.lookup("openai", "gpt-5.4-2026-03-05")[1])

    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    cost = rates.cost_usd("openai", "gpt-5.4", usage)
    check("cost = input rate + output rate for 1M each",
          abs(cost - (known["input"] + known["output"])) < 1e-9, f"${cost:.4f}")
    cached = rates.cost_usd("openai", "gpt-5.4",
                            {"input_tokens": 0, "output_tokens": 0,
                             "cache_creation_input_tokens": 0,
                             "cache_read_input_tokens": 1_000_000})
    check("cached input is discounted", cached < known["input"], f"${cached:.4f}")

    tr = SpendTracker("openai", "gpt-5.4", ceiling_usd=1.0, rates=rates)
    check("nothing spent yet -> may proceed", not tr.would_exceed()[0])
    tr.record({"input_tokens": 100_000, "output_tokens": 20_000,
               "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
    check("spend is recorded from real usage", tr.total_usd > 0,
          f"${tr.total_usd:.4f}")
    for _ in range(20):
        tr.record({"input_tokens": 100_000, "output_tokens": 20_000,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
    over, msg = tr.would_exceed()
    check("ceiling trips once spend approaches it", over, msg[:70])
    check("abort message is actionable", "--max-spend-usd" in msg)

    # the loop must surface it as a stop reason, before the next call
    with tempfile.TemporaryDirectory() as td:
        loop = AgentLoop.__new__(AgentLoop)
        loop.tree = ExperimentTree(td)
        loop.max_iterations = 50
        loop.wall_clock_limit_s = 6 * 3600
        loop.run_started = time.time()
        loop.spend = tr
        check("loop stops on the spend ceiling",
              "spend ceiling" in (loop.stop_reason() or ""))
        loop.spend = SpendTracker("openai", "gpt-5.4", ceiling_usd=1000.0, rates=rates)
        check("generous ceiling does not stop the loop",
              "spend ceiling" not in (loop.stop_reason() or ""))


# ------------------------------------------- regressions found in the 2026-08-28 audit
def test_audit_regressions():
    print("\n[audit regressions]")
    from agent.llm import looks_like_placeholder
    from agent.pricing import RateTable, SpendTracker

    # A. placeholder key must not pass preflight (it used to, then burned the cap)
    for bad in ("sk-proj-...", "sk-ant-...", "your-key-here", "changeme", "short",
                "<paste key>"):
        check(f"placeholder rejected: {bad!r}", looks_like_placeholder(bad))
    for good in ("sk-proj-" + "a" * 40, "sk-ant-" + "b" * 60):
        check(f"real-looking key accepted: {good[:12]}...",
              not looks_like_placeholder(good))

    # B. truncated journal line must not make a run unresumable
    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        t.add(_node(0, "success", 0.60))
        t.add(_node(1, "success", 0.61))
        with open(os.path.join(td, "journal.jsonl"), "a") as fh:
            fh.write('{"iteration_id": 2, "action": "dra')   # killed mid-write
        try:
            t2 = ExperimentTree(td)
            check("truncated journal line survives reload",
                  [n.iteration_id for n in t2.nodes] == [0, 1],
                  f"loaded {[n.iteration_id for n in t2.nodes]}")
            check("corruption is reported, not hidden", len(t2.corrupt_lines) == 1)
            check("best node still resolves after truncation",
                  t2.best() is not None and t2.best().iteration_id == 1)
        except Exception as e:
            check("truncated journal line survives reload", False,
                  f"{type(e).__name__}: {e}")

    # C. cold-start estimate must not refuse to start a modest ceiling
    rates = RateTable()
    tr = SpendTracker("openai", "gpt-5.4-mini", ceiling_usd=0.02, rates=rates)
    check("first iteration is always allowed", not tr.would_exceed()[0])
    check("cold-start estimate is calibrated (< $0.05 on the cheap model)",
          tr.estimated_next_call_usd() < 0.05,
          f"${tr.estimated_next_call_usd():.4f}")
    # measured usage from the audited run: ~3.2k in / ~500 out per iteration
    n = 0
    while not tr.would_exceed()[0] and n < 50:
        tr.record({"input_tokens": 3_200, "output_tokens": 500,
                   "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
        n += 1
    over, msg = tr.would_exceed()
    check("ceiling trips on measured cost within a sane iteration count",
          over and n < 50, f"tripped after {n} iterations: {msg[:60]}")
    check("did not breach while stopping", tr.total_usd <= tr.ceiling_usd,
          f"${tr.total_usd:.4f} <= ${tr.ceiling_usd:.2f}")

    # D. repeated LLM-stage failures abort instead of burning the iteration cap
    with tempfile.TemporaryDirectory() as td:
        loop = AgentLoop.__new__(AgentLoop)
        loop.tree = ExperimentTree(td)
        loop.max_iterations = 50
        loop.wall_clock_limit_s = 6 * 3600
        loop.run_started = time.time()
        loop.spend = SpendTracker("openai", "gpt-5.4", 100.0, rates)
        loop.consecutive_llm_failures = 0
        loop.last_llm_error = ""
        loop.max_consecutive_llm_failures = 3
        check("healthy loop keeps going", loop.stop_reason() is None)
        loop.consecutive_llm_failures = 3
        loop.last_llm_error = "AuthenticationError: 401"
        sr = loop.stop_reason() or ""
        check("3 consecutive LLM failures abort the run", "aborted" in sr)
        check("abort message names the cause", "401" in sr)


def test_numpy2_json_serialization():
    """Regression: metrics must serialize on numpy>=2, where evaluate() returns
    np.float32 (NEP 50) instead of the float64 numpy 1.x happened to produce.
    Without the float() casts this failed on EVERY iteration of a fresh install.
    """
    print("\n[numpy 2 json serialization]")
    import numpy as np
    sys.path.insert(0, os.path.join(_ROOT, "kuairand-starter-kit"))
    from evaluate import evaluate

    labels = np.array([1, 0, 1, 0], dtype=np.float32)
    scores = np.array([0.9, 0.1, 0.8, 0.2], dtype=np.float32)
    raw = evaluate(["u", "u", "v", "v"], labels, scores)
    check(f"evaluate() returns a numpy scalar under numpy {np.__version__} "
          f"(type={type(raw['primary']).__name__})", True)

    cast = {"GAUC": float(raw["GAUC"]), "nDCG@5": float(raw["nDCG@5"]),
            "primary": float(raw["primary"])}
    check("cast metrics are plain python floats",
          all(type(v) is float for v in cast.values()))
    try:
        json.dumps(cast)
        check("cast metrics are JSON-serializable", True)
    except TypeError as e:
        check("cast metrics are JSON-serializable", False, str(e))

    # the shipped code path must do the casting itself
    src = open(os.path.join(_ROOT, "runtime", "train_lib.py")).read()
    check("train_lib.run casts metrics before json.dump",
          'float(va["GAUC"])' in src)
    sub = open(os.path.join(_ROOT, "agent", "make_submission.py")).read()
    check("make_submission casts metrics before json.dump",
          'float(r["GAUC"])' in sub)
    api = open(os.path.join(_ROOT, "runtime", "API.md")).read()
    check("API.md warns generated code about the float32 trap",
          "JSON serializable" in api and "float(" in api)
    check("API.md documents where n_users comes from",
          'meta["field_dims"]["user"]' in api)


if __name__ == "__main__":
    for t in (test_safety_gate, test_validity, test_policy, test_convergence,
              test_executor, test_crossover, test_spend_ceiling,
              test_audit_regressions, test_numpy2_json_serialization):
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
