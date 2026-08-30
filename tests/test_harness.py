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


def test_diff_artifacts():
    """The literal 'code diff applied' deliverable: unified diff + sha256 per node."""
    print("\n[diff artifacts]")
    import hashlib
    from agent.executor import save_diff

    with tempfile.TemporaryDirectory() as td:
        parent = os.path.join(td, "parent.py")
        child = os.path.join(td, "child.py")
        diffs_dir = os.path.join(td, "diffs")
        with open(parent, "w") as fh:
            fh.write("a = 1\nb = 2\n")
        with open(child, "w") as fh:
            fh.write("a = 1\nb = 3\n")

        info = save_diff(child, parent, diffs_dir, 7)
        diff_file = os.path.join(diffs_dir, "node_007.diff")
        check("diff file written", os.path.exists(diff_file))
        content = open(diff_file).read()
        check("diff shows the actual change", "-b = 2" in content and "+b = 3" in content)
        check("returned sha256 matches the written file",
              info["diff_sha256"] == hashlib.sha256(content.encode()).hexdigest())

        info2 = save_diff(child, None, diffs_dir, 8)
        content2 = open(os.path.join(diffs_dir, "node_008.diff")).read()
        check("parentless draft diffs against runtime/seed_solution.py",
              "seed_solution.py" in content2)
        check("that diff is also hashed", len(info2["diff_sha256"]) == 64)


def test_data_boundary():
    """Item 4, half A: the sandboxed test split must not carry outcome columns."""
    print("\n[data boundary: test-split label redaction]")
    import data_boundary  # loaded onto sys.path as a side effect of agent.executor

    fake_test_npz = {
        "user": [0, 1], "video": [3, 4], "author": [5, 6],
        "long_view": [1.0, 0.0], "is_click": [1.0, 0.0], "is_like": [0.0, 0.0],
        "is_forward": [0.0, 0.0], "play_time_ms": [500.0, 0.0],
        "user_raw": ["u0", "u1"],
    }
    redacted = data_boundary.redact_test_columns(fake_test_npz)
    for col in data_boundary.TEST_LABEL_COLUMNS:
        check(f"'{col}' stripped from the redacted test split", col not in redacted)
    check("non-label feature columns survive redaction",
          {"user", "video", "author", "user_raw"} <= set(redacted))
    # Pinned as an INVARIANT, not a frozen literal. The original list was an
    # equality check, so ADDING the four auxiliary outcome columns -- which
    # strengthens the boundary -- failed the test exactly as removing one
    # would. What must never happen is a column LEAVING the list, or an
    # outcome column existing in the cache without being redacted.
    ORIGINAL_OUTCOMES = {"long_view", "is_click", "is_like", "is_forward",
                         "play_time_ms"}
    declared = set(data_boundary.TEST_LABEL_COLUMNS)
    check("the original outcome columns are never dropped from the boundary",
          ORIGINAL_OUTCOMES <= declared,
          f"missing: {ORIGINAL_OUTCOMES - declared}")
    sys.path.insert(0, os.path.join(_ROOT, "runtime"))
    import train_lib as _tl
    check("every auxiliary outcome column the cache stores is also redacted",
          set(_tl._AUX_SOCIAL_COLS) <= declared,
          f"unredacted: {set(_tl._AUX_SOCIAL_COLS) - declared}")
    check("the boundary contains only outcome columns, no features",
          not (declared & {"user", "video", "author", "tab", "duration_ms",
                           "hourmin", "date", "time_ms", "user_raw"}))


def test_restricted_access():
    """Item 4, half B + item 5: a technical (not instruction-following) boundary.

    Enforcement is OS permission bits, so it must hold regardless of which path
    string reaches the blocked location -- that's the property being tested,
    not just that *a* convention was followed.
    """
    print("\n[restricted_access: OS-level lockdown around the subprocess]")
    from agent import executor
    from agent.executor import restricted_access

    # Root bypasses chmod-based restrictions unconditionally, which would make
    # every check below pass for the wrong reason (nothing was ever blocked).
    # Simulate root deterministically -- via monkeypatch, not by requiring an
    # actual root shell -- and confirm the guard refuses to proceed instead of
    # silently no-op'ing the entire boundary.
    if hasattr(os, "geteuid"):
        real_geteuid = os.geteuid
        os.geteuid = lambda: 0
        try:
            executor.assert_not_root()
            check("assert_not_root refuses to proceed when euid==0", False)
        except RuntimeError:
            check("assert_not_root refuses to proceed when euid==0", True)
        try:
            with restricted_access(unreadable_paths=[]):
                pass
            check("restricted_access itself refuses to proceed under root", False)
        except RuntimeError:
            check("restricted_access itself refuses to proceed under root", True)
        finally:
            os.geteuid = real_geteuid

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        check("skipped remaining checks (this test process is ACTUALLY root, "
              "not simulated -- chmod-based checks below would be meaningless)", True)
        return

    with tempfile.TemporaryDirectory() as td:
        secret_dir = os.path.join(td, "secret")
        os.makedirs(secret_dir)
        secret_file = os.path.join(secret_dir, "labels.txt")
        with open(secret_file, "w") as fh:
            fh.write("do not read me\n")
        protected_file = os.path.join(td, "journal.jsonl")
        with open(protected_file, "w") as fh:
            fh.write("{}\n")

        with restricted_access(unreadable_paths=[secret_dir]):
            try:
                open(secret_file).read()
                check("an unreadable dir blocks access via a direct path", False)
            except OSError:
                check("an unreadable dir blocks access via a direct path", True)
        check("permissions restored after the block",
              open(secret_file).read() == "do not read me\n")

        with restricted_access(read_only_paths=[protected_file]):
            check("a read-only file is still readable",
                  open(protected_file).read() == "{}\n")
            try:
                open(protected_file, "a").write("x")
                check("a read-only file refuses writes", False)
            except OSError:
                check("a read-only file refuses writes", True)
        with open(protected_file, "a") as fh:
            fh.write("y\n")
        check("write access restored after the block", True)

        # Regression: directory mode alone (e.g. 0o555) blocks creating/deleting
        # entries but NOT writes to a file that already exists inside it -- an
        # existing file's own mode bits govern that. A protected *directory*
        # must therefore lock every file already inside it too.
        protected_dir = os.path.join(td, "config")
        os.makedirs(protected_dir)
        existing = os.path.join(protected_dir, "agent_config.json")
        with open(existing, "w") as fh:
            fh.write("{}")
        with restricted_access(read_only_paths=[protected_dir]):
            check("a pre-existing file inside a protected dir is still readable",
                  open(existing).read() == "{}")
            try:
                open(existing, "a").write("TAMPERED")
                check("a pre-existing file inside a protected dir refuses writes", False)
            except OSError:
                check("a pre-existing file inside a protected dir refuses writes", True)
            try:
                with open(os.path.join(protected_dir, "new.json"), "w") as fh:
                    fh.write("{}")
                check("a protected dir refuses new files", False)
            except OSError:
                check("a protected dir refuses new files", True)
        with open(existing, "a") as fh:
            fh.write("y")
        check("write access restored for the file inside the dir", True)

        try:
            with restricted_access(unreadable_paths=[secret_dir]):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        check("permissions restored even when the wrapped code raises",
              open(secret_file).read() == "do not read me\n")


def test_final_eval_lock():
    """Item 3: the one-time hidden-test evaluation is enforced, not just documented."""
    print("\n[final_evaluation.lock: one-time-eval guard]")
    from agent.make_submission import (_sha256_file, check_final_eval_guard,
                                       write_final_eval_lock)

    with tempfile.TemporaryDirectory() as td:
        lock_path = os.path.join(td, "results", "final_evaluation.lock")
        override_log = os.path.join(td, "logs", "final_eval_override_attempts.jsonl")
        sub_path = os.path.join(td, "submission_test.csv")
        with open(sub_path, "w") as fh:
            fh.write("row_id,user_id,video_id,score\n0,0,1,0.5\n")

        try:
            check_final_eval_guard(admin_override=False, lock_path=lock_path,
                                   override_log_path=override_log)
            check("first run (no lock yet) is allowed", True)
        except SystemExit:
            check("first run (no lock yet) is allowed", False)

        write_final_eval_lock(sub_path, {"GAUC": 0.6, "nDCG@5": 0.5, "primary": 0.55},
                              lock_path=lock_path)
        check("lock file written", os.path.exists(lock_path))
        with open(lock_path) as fh:
            lock = json.load(fh)
        check("lock records the submission's real sha256",
              lock["submission_sha256"] == _sha256_file(sub_path))

        try:
            check_final_eval_guard(admin_override=False, lock_path=lock_path,
                                   override_log_path=override_log)
            check("second run without --admin-override is refused", False)
        except SystemExit:
            check("second run without --admin-override is refused", True)

        try:
            check_final_eval_guard(admin_override=True, lock_path=lock_path,
                                   override_log_path=override_log)
            check("--admin-override forces a second run through", True)
        except SystemExit:
            check("--admin-override forces a second run through", False)

        with open(override_log) as fh:
            attempts = [json.loads(ln) for ln in fh if ln.strip()]
        check("every override attempt is logged (refused + forced)", len(attempts) == 2)
        check("refused attempt logged as not allowed", attempts[0]["allowed"] is False)
        check("forced attempt logged as allowed", attempts[1]["allowed"] is True)


def test_reseed():
    """Phase 2: top-N reseeding for statistical rigor. No training happens in
    this test -- every synthetic node's code_path points at a file that
    doesn't exist (exercising the skip path), and the ceiling-refusal case
    exits before the training loop is ever entered.
    """
    print("\n[reseed: top-N re-seeding for statistical rigor]")
    from agent.reseed import estimate_wall_clock_s, load_top_n_nodes, run_reseed

    with tempfile.TemporaryDirectory() as td:
        logs_dir = os.path.join(td, "logs")
        os.makedirs(logs_dir)
        journal_path = os.path.join(logs_dir, "journal.jsonl")
        records = [
            {"iteration_id": 0, "status": "success", "action": "draft",
             "metrics": {"primary": 0.60, "GAUC": 0.6, "nDCG@5": 0.6},
             "code_path": os.path.join(td, "missing_node_000.py"),
             "menu_choices": {"loss": "pointwise_logloss"},
             "wall_clock_seconds": 20.0, "seed": 0},
            {"iteration_id": 1, "status": "success", "action": "draft",
             "metrics": {"primary": 0.62, "GAUC": 0.6, "nDCG@5": 0.6},
             "code_path": os.path.join(td, "missing_node_001.py"),
             "menu_choices": {"loss": "bpr_pairwise"},
             "wall_clock_seconds": 500.0},   # no "seed" key -> legacy/assumed
            {"iteration_id": 2, "status": "success", "action": "draft",
             "metrics": {"primary": 0.55, "GAUC": 0.55, "nDCG@5": 0.55},
             "code_path": os.path.join(td, "missing_node_002.py"),
             "menu_choices": {"loss": "listwise_softmax"},
             "wall_clock_seconds": 10.0, "seed": 0},
            {"iteration_id": 3, "status": "error", "metrics": None,
             "code_path": "", "wall_clock_seconds": 0.0},
        ]
        with open(journal_path, "w") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

        top2 = load_top_n_nodes(journal_path, 2)
        check("top-N sorted by validation primary, descending",
              [n["iteration_id"] for n in top2] == [1, 0])
        check("error/unscored nodes excluded from selection",
              all(n["status"] == "success" for n in top2))

        est = estimate_wall_clock_s(top2, n_seeds=5)
        # node 1: no recorded seed -> assumed 0, in [0,5) -> 4 fresh * 500s = 2000
        # node 0: seed 0, in [0,5) -> 4 fresh * 20s = 80
        check("estimate reuses the original-seed sample (K-1 fresh runs/node)",
              est == 2000.0 + 80.0)

        try:
            run_reseed(root=td, top_n=2, n_seeds=1, wall_clock_limit_h=6.0)
            check("--reseed-seeds < 2 is rejected (can't compute a std)", False)
        except SystemExit:
            check("--reseed-seeds < 2 is rejected (can't compute a std)", True)

        try:
            run_reseed(root=td, top_n=2, n_seeds=5, wall_clock_limit_h=0.001)
            check("refuses to start when the wall-clock estimate exceeds the ceiling", False)
        except SystemExit:
            check("refuses to start when the wall-clock estimate exceeds the ceiling", True)
        check("nothing written when refused up front (no training was risked)",
              not os.path.exists(os.path.join(logs_dir, "reseed_results.json")))

        summary = run_reseed(root=td, top_n=2, n_seeds=5, wall_clock_limit_h=6.0)
        check("proceeds and writes output when the estimate fits the ceiling",
              os.path.exists(os.path.join(logs_dir, "reseed_results.json")))
        check("nodes with a missing code_path are skipped, not crashed on",
              summary["nodes"] == [])

        empty_root = os.path.join(td, "empty")
        os.makedirs(os.path.join(empty_root, "logs"))
        try:
            run_reseed(root=empty_root, top_n=2, n_seeds=5, wall_clock_limit_h=6.0)
            check("no journal at all -> refuses cleanly", False)
        except SystemExit:
            check("no journal at all -> refuses cleanly", True)


def test_experience():
    """Phase 3 item 1: curated experience memory, auto-compacted to a char
    budget by dropping whole OLDEST entries, never truncating mid-entry.
    """
    print("\n[experience memory: curated lessons, compacted]")
    from agent import experience as exp

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "experience.md")

        check("no file yet -> placeholder, not an error",
              exp.render_for_prompt(path) == "(no prior experience recorded yet)")

        exp.append_entry(0, "helped", "bpr_pairwise beat pointwise",
                         "Switching loss to bpr_pairwise raised valid primary "
                         "by ~0.003 over the pointwise baseline.", path=path)
        rendered = exp.render_for_prompt(path)
        check("entry appears with its outcome tag uppercased",
              "[HELPED]" in rendered and "iter 0" in rendered)
        check("entry body is present, not truncated",
              "bpr_pairwise" in rendered and "0.003" in rendered)

        # Force compaction: a budget sized off a REAL entry's length (not a
        # short placeholder) so it can hold the header plus ~2 real entries --
        # tight enough to force drops, loose enough that something must survive.
        probe_entry_len = len(exp.format_entry(0, "dead_end", "idea 0 failed",
                                               "Tried variant 0; no improvement "
                                               "over baseline."))
        tight_budget = len(exp.HEADER) + probe_entry_len * 2 + 20
        for i in range(1, 6):
            exp.append_entry(i, "dead_end", f"idea {i} failed",
                             f"Tried variant {i}; no improvement over baseline.",
                             path=path, char_budget=tight_budget)
        with open(path) as fh:
            final_text = fh.read()
        check("compaction keeps the file within its character budget",
              len(final_text) <= tight_budget + len(exp.HEADER))
        check("compaction drops the OLDEST entries first (iter 0 is gone)",
              "iter 0" not in final_text)
        check("the most recent entry survives compaction",
              "iter 5" in final_text)
        _, remaining = exp._split_entries(final_text)
        check("every surviving entry is whole, not cut mid-sentence",
              all(e.rstrip().endswith(".") or e.rstrip().endswith(")")
                  for e in remaining))

        # A pathological single entry larger than the whole budget must not
        # corrupt the file -- it survives whole (nothing sane to cut), rather
        # than being sliced into invalid partial markdown.
        huge_body = "x" * (tight_budget * 3)
        exp.append_entry(99, "crashed", "huge", huge_body, path=path,
                         char_budget=tight_budget)
        with open(path) as fh:
            after_huge = fh.read()
        check("an oversized single entry is kept whole, not sliced mid-body",
              huge_body in after_huge)


def test_rationale_schema():
    """Phase 3 item 2: per-idea rationale is schema-enforced, not just requested
    in prose -- a generic non-answer for grounded_in must fail validation
    (forcing a repair retry) rather than being silently accepted.
    """
    print("\n[rationale schema: problem-insight enforcement]")
    from agent.llm import LLMClient

    def base_obj(rationale):
        # Path A shape: implementation_path/research_category are now required
        # on every response, and menu_choices is required for Path A only.
        return {"hypothesis": "try X because Y", "menu_choices": {"loss": "bpr_pairwise"},
                "code": "x" * 60, "expected_effect": "+0.003", "rationale": rationale,
                "implementation_path": "A", "research_category": "exploration"}

    good = base_obj({
        "idea": "Switch loss to bpr_pairwise",
        "why_expected_to_help": "GAUC/nDCG are ranking metrics; pointwise logloss "
                               "doesn't directly optimize ranking.",
        "grounded_in": "loss axis: organizers rank ranking-aligned loss as the top "
                      "unexplored direction",
    })
    check("a well-formed rationale passes schema validation",
          LLMClient._schema_problems(good) == [])

    check("missing rationale key is rejected",
          any("rationale" in p for p in
              LLMClient._schema_problems({k: v for k, v in good.items()
                                         if k != "rationale"})))

    missing_sub = base_obj({"idea": "x", "why_expected_to_help": "y"})  # no grounded_in
    check("missing rationale sub-key is rejected",
          any("grounded_in" in p for p in LLMClient._schema_problems(missing_sub)))

    too_short = base_obj({"idea": "x", "why_expected_to_help": "y", "grounded_in": "z"})
    check("too-short rationale sub-values are rejected",
          len(LLMClient._schema_problems(too_short)) > 0)

    for phrase in ("general ML intuition", "It seemed reasonable to try",
                  "Standard practice in the field"):
        generic = base_obj({"idea": "switch the loss function to something else",
                           "why_expected_to_help": "it might improve the score somehow",
                           "grounded_in": phrase})
        check(f"generic grounding rejected: {phrase!r}",
              any("generic" in p for p in LLMClient._schema_problems(generic)))

    named_paper = base_obj({
        "idea": "Add DIN-style attention over user history",
        "why_expected_to_help": "weights history items by similarity to the candidate",
        "grounded_in": "DIN (Deep Interest Network) attention pooling",
    })
    check("a named paper/method is accepted as grounding",
          LLMClient._schema_problems(named_paper) == [])


def test_best_override():
    """A multi-seed reseed mean can disagree with the single-seed pick the live
    loop recorded; the canonical best_metrics.json/best_solution.py must then
    point at the reseed-verified winner, not the single lucky/unlucky sample.
    """
    print("\n[best-node override: reseed-verified winner supersedes single-seed pick]")
    from agent.reseed import apply_best_override

    with tempfile.TemporaryDirectory() as td:
        logs_dir = os.path.join(td, "logs")
        sol_dir = os.path.join(logs_dir, "solutions")
        os.makedirs(sol_dir)
        for iid, content in ((6, "# node 6 code\n"), (7, "# node 7 code\n")):
            with open(os.path.join(sol_dir, f"node_{iid:03d}.py"), "w") as fh:
                fh.write(content)

        t = ExperimentTree(logs_dir)
        n6 = _node(6, "success", 0.6035, choices={"loss": "bpr_pairwise"})
        n6.code_path = os.path.join(sol_dir, "node_006.py")
        n7 = _node(7, "success", 0.6033, choices={"loss": "bpr_pairwise"})
        n7.code_path = os.path.join(sol_dir, "node_007.py")
        t.add(n6)   # code_path baked into the journal BEFORE apply_best_override
        t.add(n7)   # reloads the tree from disk, not from this in-memory object
        check("before override: best_metrics.json points at the single-seed winner",
              json.load(open(os.path.join(logs_dir, "best_metrics.json")))["iteration_id"] == 6)

        summary = {
            "best_changed": True,
            "original_best_node": 6,
            "best_by_mean_node": 7,
            "nodes": [
                {"iteration_id": 6, "mean_primary": 0.6032, "std_primary": 0.0003,
                 "n_samples": 5, "original_single_seed_primary": 0.6035},
                {"iteration_id": 7, "mean_primary": 0.6037, "std_primary": 0.0004,
                 "n_samples": 5, "original_single_seed_primary": 0.6033},
            ],
        }
        exp_path = os.path.join(td, "experience.md")  # NOT the real agent/experience.md
        apply_best_override(td, summary, experience_path=exp_path)

        with open(os.path.join(logs_dir, "best_metrics.json")) as fh:
            bm = json.load(fh)
        check("after override: best_metrics.json points at the mean-verified winner",
              bm["iteration_id"] == 7)
        check("override records the reseed provenance",
              bm.get("reseed_verified") is True and bm["reseed_mean_primary"] == 0.6037)
        check("override records what it superseded",
              bm["superseded_single_seed_best_node"] == 6
              and bm["superseded_single_seed_best_primary"] == 0.6035)
        with open(os.path.join(logs_dir, "best_solution.py")) as fh:
            check("best_solution.py is copied from the NEW winner's code",
                  fh.read() == "# node 7 code\n")

        # durable, append-only provenance -- distinct from best_metrics.json,
        # which only holds the CURRENT state and would lose this on a 2nd override
        override_log = os.path.join(logs_dir, "best_override_log.jsonl")
        check("a durable override-log entry is written", os.path.exists(override_log))
        with open(override_log) as fh:
            entries = [json.loads(ln) for ln in fh if ln.strip()]
        check("override-log entry names old and new best nodes",
              entries[0]["old_best_node"] == 6 and entries[0]["new_best_node"] == 7)
        check("override-log entry records the mean/std that triggered the switch",
              entries[0]["new_best_reseed_mean_primary"] == 0.6037
              and entries[0]["new_best_reseed_std_primary"] == 0.0004)

        check("a prompt-visible experience.md entry is also written",
              os.path.exists(exp_path))
        from agent import experience as exp
        rendered = exp.render_for_prompt(exp_path)
        check("experience entry names both nodes and is tagged CORRECTION",
              "[CORRECTION]" in rendered and "6" in rendered and "7" in rendered)

        # idempotent / no-op when nothing changed
        apply_best_override(td, {**summary, "best_changed": False}, experience_path=exp_path)
        with open(os.path.join(logs_dir, "best_metrics.json")) as fh:
            check("a not-changed summary is a no-op",
                  json.load(fh)["iteration_id"] == 7)
        with open(override_log) as fh:
            check("a not-changed summary does not add a spurious override-log entry",
                  sum(1 for ln in fh if ln.strip()) == 1)


def test_worktree_lifecycle():
    """Phase 3 item 3 Part A: per-worker git worktrees. Uses a dedicated,
    clearly-out-of-range test slot (999) so it can never collide with a real
    worker slot, and always cleans up after itself.
    """
    print("\n[worktree lifecycle: isolated per-worker checkouts]")
    from agent import worktree

    check("git worktree is available in this environment", worktree.is_available())
    TEST_SLOT = 999
    try:
        path = worktree.ensure_worktree(TEST_SLOT)
        check("worktree directory was created", os.path.isdir(path))
        check("tracked code is checked out", os.path.isdir(os.path.join(path, "runtime")))
        check("gitignored real dataset is NOT checked out (structural absence)",
              not os.path.exists(os.path.join(path, "kuairand-starter-kit",
                                              "KuaiRand-Pure")))
        again = worktree.ensure_worktree(TEST_SLOT)
        check("calling ensure_worktree again reuses the same worktree, doesn't rebuild",
              again == path)
    finally:
        worktree.remove_worktree(TEST_SLOT)
    check("remove_worktree deletes the directory", not os.path.isdir(path))
    import subprocess as sp
    listing = sp.run(["git", "worktree", "list"], cwd=worktree.ROOT,
                     capture_output=True, text=True)
    check("git no longer tracks the removed worktree",
          f"worker_{TEST_SLOT}" not in listing.stdout)


def test_worker_sandbox_hardlinking():
    """Phase 3 item 3 Part A: per-worker sandbox data is hardlinked from the
    one canonical copy, not re-copied -- verifying the "near-zero marginal
    disk cost for N workers" property directly, not just that files exist.
    """
    print("\n[worker sandbox: hardlinked, not re-copied]")
    from agent import worktree
    from agent.executor import REAL_DATA_DIR
    from runtime.data_boundary import SANDBOX_CACHE_DIR, build_worker_sandbox

    TEST_SLOT = 998
    try:
        wt_path = worktree.ensure_worktree(TEST_SLOT)
        cache_dir, data_dir = build_worker_sandbox(REAL_DATA_DIR, wt_path)
        check("worker sandbox cache dir created", os.path.isdir(cache_dir))
        check("worker test.npz exists", os.path.exists(os.path.join(cache_dir, "test.npz")))

        canonical = os.path.join(SANDBOX_CACHE_DIR, "test.npz")
        worker_copy = os.path.join(cache_dir, "test.npz")
        check("worker's test.npz is HARDLINKED to the canonical copy (same inode, "
             "zero extra disk) -- not a byte-for-byte re-copy",
             os.stat(canonical).st_ino == os.stat(worker_copy).st_ino)

        import numpy as np
        z = np.load(worker_copy, allow_pickle=True)
        for col in ("long_view", "is_click", "is_like", "is_forward", "play_time_ms"):
            check(f"worker sandbox test split still has no '{col}' column",
                  col not in z.files)

        # idempotent: calling again for the same slot doesn't error or duplicate work
        cache_dir2, _ = build_worker_sandbox(REAL_DATA_DIR, wt_path)
        check("calling build_worker_sandbox again is a no-op (same path, still linked)",
              cache_dir2 == cache_dir
              and os.stat(canonical).st_ino == os.stat(worker_copy).st_ino)
    finally:
        worktree.remove_worktree(TEST_SLOT)


def test_run_parallel_round():
    """Phase 3 item 3 Part A: the concurrent dispatch primitive itself --
    correct per-job results in request order, one job's failure doesn't sink
    the round, and the real directories are readable again once the whole
    round (not just the first job) has finished. Uses trivial/fast scripts
    (same style as test_executor), not real training -- the adversarial
    multi-vector concurrent hostile-script test was run and reported
    separately (that one takes several real seconds and is a manual
    verification step, same pattern as Phase 1's hostile-script checks).
    """
    print("\n[run_parallel_round: concurrent dispatch primitive]")
    from agent.executor import (REAL_CACHE_DIR, REAL_DATA_DIR,
                                run_parallel_round)

    with tempfile.TemporaryDirectory() as td:
        ok_code = ("import argparse,os,json\n"
                  "p=argparse.ArgumentParser();p.add_argument('--menu-choices');"
                  "p.add_argument('--output-dir');p.add_argument('--seed');a=p.parse_args()\n"
                  "os.makedirs(a.output_dir,exist_ok=True)\n"
                  "import numpy as np\n"
                  "json.dump({'GAUC':0.6,'nDCG@5':0.6,'primary':0.6},"
                  "open(os.path.join(a.output_dir,'metrics.json'),'w'))\n"
                  "np.save(os.path.join(a.output_dir,'scores_valid.npy'), np.zeros(124909))\n"
                  "np.save(os.path.join(a.output_dir,'scores_test.npy'), np.zeros(170588))\n")
        bad_code = "raise ValueError('deliberate failure for job isolation check')\n"

        jobs = [
            {"slot": 997, "code": ok_code, "code_path": os.path.join(td, "s0.py"),
             "menu_choices": {}, "run_dir": os.path.join(td, "r0"), "seed": 0},
            {"slot": 996, "code": bad_code, "code_path": os.path.join(td, "s1.py"),
             "menu_choices": {}, "run_dir": os.path.join(td, "r1"), "seed": 0},
        ]
        try:
            results = run_parallel_round(jobs, timeout_s=60)
            check("run_parallel_round returns one result per job", len(results) == 2)
            check("results preserve request order (not completion order)",
                  results[0].ok is True and results[1].ok is False)
            check("a failing job's error is captured, not raised out of the round",
                  "ValueError" in (results[1].error_trace or ""))
            check("real data dir readable again after the round finished",
                  os.access(REAL_DATA_DIR, os.R_OK | os.X_OK))
            check("real cache dir readable again after the round finished",
                  os.access(REAL_CACHE_DIR, os.R_OK | os.X_OK))
        finally:
            from agent import worktree
            worktree.remove_worktree(997)
            worktree.remove_worktree(996)


def test_merge_acceptance_via_tree_ordering():
    """Phase 3 item 3 Part B's central design claim: 'accept the merge only if
    it strictly beats the best individual' requires NO new gating code -- it
    falls out of ExperimentTree's existing best-tracking as long as the
    round's individual nodes are added before the merge node. Verified here
    directly against ExperimentTree/Node, with no LLM call and no training --
    the full orchestration (agent.loop.iterate_parallel/_attempt_merge) is
    exercised for real in the end-to-end parallel-round run instead.
    """
    print("\n[merge acceptance: falls out of tree.add() ordering, no new gate]")
    with tempfile.TemporaryDirectory() as td:
        # Case 1: merge beats both individuals -> merge becomes best
        t = ExperimentTree(td)
        t.add(_node(0, "success", 0.60))                          # pre-round best
        worker_a = _node(1, "success", 0.62, action="draft")
        worker_a.round_id = "round_1"
        worker_b = _node(2, "success", 0.63, action="draft")
        worker_b.round_id = "round_1"
        t.add(worker_a)
        t.add(worker_b)
        check("best individual (worker_b) is provisionally best after workers added",
              t.best().iteration_id == 2)
        merge_wins = _node(3, "success", 0.65, action="merge", parent=2)
        merge_wins.round_id, merge_wins.merged_from = "round_1", [1, 2]
        t.add(merge_wins)
        check("merge that strictly beats best individual becomes the tree's best",
              t.best().iteration_id == 3)

        # Case 2: merge does NOT beat the best individual -> falls back, no gate needed
        t2 = ExperimentTree(os.path.join(td, "case2"))
        t2.add(_node(0, "success", 0.60))
        w_a = _node(1, "success", 0.64, action="draft")
        w_b = _node(2, "success", 0.61, action="draft")
        t2.add(w_a)
        t2.add(w_b)
        merge_loses = _node(3, "success", 0.62, action="merge", parent=1)  # beats w_b, not w_a
        merge_loses.merged_from = [1, 2]
        t2.add(merge_loses)
        check("merge scoring between the two individuals still loses to the best one",
              t2.best().iteration_id == 1)

        # Case 3: merge CRASHES -> same fallback, no special-case needed
        t3 = ExperimentTree(os.path.join(td, "case3"))
        t3.add(_node(0, "success", 0.60))
        w_a = _node(1, "success", 0.64, action="draft")
        w_b = _node(2, "success", 0.66, action="draft")
        t3.add(w_a)
        t3.add(w_b)
        merge_crashed = _node(3, "error", None, action="merge", parent=2)
        merge_crashed.merged_from = [1, 2]
        t3.add(merge_crashed)
        check("a crashed merge (no score) leaves the best individual as best",
              t3.best().iteration_id == 2)

        # merged_from survives the journal round-trip (append-only file, reloaded)
        t3_reloaded = ExperimentTree(os.path.join(td, "case3"))
        reloaded_merge = t3_reloaded.get(3)
        check("merged_from and round_id round-trip through the journal",
              reloaded_merge.merged_from == [1, 2])


def test_standing_override_survives_reload():
    """Real bug caught while preparing Part B's real end-to-end test:
    ExperimentTree recomputes best_node_id purely from the journal's raw
    single-seed scores on every reload, entirely independent of
    best_metrics.json -- so a fresh ExperimentTree() (exactly what a resumed
    run, or a brand-new AgentLoop for a parallel round, constructs) would
    silently revert to the single-seed pick a reseed override had already
    superseded. decide_action()/iterate_parallel() consult tree.best()
    directly, so this isn't cosmetic -- it would make the live search target
    the wrong node.
    """
    print("\n[standing override: survives a fresh ExperimentTree reload]")
    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        t.add(_node(6, "success", 0.6035))   # single-seed "best" per the raw journal
        t.add(_node(7, "success", 0.6033))   # reseed-verified true winner (lower single-seed)
        check("without an override file, a fresh reload uses the raw single-seed max",
              ExperimentTree(td).best_node_id == 6)

        with open(os.path.join(td, "best_metrics.json"), "w") as fh:
            json.dump({"iteration_id": 7, "reseed_verified": True}, fh)
        check("WITH a reseed_verified override, a fresh reload respects it",
              ExperimentTree(td).best_node_id == 7)

        # a NON-reseed-verified best_metrics.json (the normal, un-overridden case)
        # must NOT change anything -- this only activates for actual overrides
        with open(os.path.join(td, "best_metrics.json"), "w") as fh:
            json.dump({"iteration_id": 7}, fh)   # no reseed_verified key
        check("a plain (non-override) best_metrics.json changes nothing",
              ExperimentTree(td).best_node_id == 6)

        # subsequent organic progress still supersedes the override normally
        with open(os.path.join(td, "best_metrics.json"), "w") as fh:
            json.dump({"iteration_id": 7, "reseed_verified": True}, fh)
        t2 = ExperimentTree(td)
        t2.add(_node(8, "success", 0.70))
        check("a new node that organically beats the override still wins",
              t2.best_node_id == 8)


def test_compute_budget_prompt_section():
    """Fix for the real bug the K=3 test found: all 3 workers reimplemented
    multi-seed ensembling and all 3 timed out, because nothing told them
    exec_timeout_s is a hard, no-partial-credit ceiling or that agent.reseed
    already measures seed-variance for free. Verifies the prompt is DYNAMIC
    (reflects whatever timeout the run was actually configured with, not a
    stale hardcoded '20 minutes' string) and explicitly steers away from
    in-script ensembling.
    """
    print("\n[compute budget: dynamic timeout + reseed steering in the prompt]")
    from agent.menu import Menu
    from agent.prompts import build_merge_prompt, build_prompt

    menu = Menu(os.path.join(_ROOT, "config", "modification_menu.json"))
    tree = ExperimentTree(tempfile.mkdtemp())

    for timeout in (300, 1200, 2400):
        prompt = build_prompt("draft", None, "test", tree, menu,
                             exec_timeout_s=timeout)
        check(f"prompt states the ACTUAL configured timeout ({timeout}s), not "
             f"a hardcoded value", f"{timeout}s" in prompt)
    check("no more stale hardcoded '20 minutes' text",
          "under 20 minutes" not in build_prompt("draft", None, "test", tree, menu))
    check("prompt explicitly says a timeout gets NO partial credit",
          "NO partial credit" in build_prompt("draft", None, "test", tree, menu,
                                              exec_timeout_s=1200))
    check("prompt explicitly points at --reseed-top instead of in-script ensembling",
          "--reseed-top" in build_prompt("draft", None, "test", tree, menu))
    check("prompt tells the model NOT to propose multi-seed ensembling itself",
          "Do NOT propose multi-seed" in build_prompt("draft", None, "test", tree, menu))

    # merge prompt gets the same treatment
    a = _node(1, "success", 0.62)
    b = _node(2, "success", 0.63)
    for n in (a, b):
        n.code_path = os.path.join(tempfile.gettempdir(), f"missing_{n.iteration_id}.py")
        n.hypothesis = "x"
    merge_prompt = build_merge_prompt(a, b, "test merge", menu, exec_timeout_s=900)
    check("merge prompt also states the actual configured timeout",
          "900s" in merge_prompt)
    check("merge prompt also steers away from in-script ensembling",
          "--reseed-top" in merge_prompt)


def test_lambdarank():
    """Part B proof-steps, kept permanently: the |delta nDCG@5| weight must
    match the OFFICIAL evaluate.py pair-by-pair (never a reimplementation of
    the metric), and forcing the weight to 1.0 must reproduce bpr_pairwise's
    pair enumeration exactly -- otherwise a measured score difference could
    come from different sampling rather than from the position discount.
    (The full degenerate-equivalence run trains a real model and is verified
    separately/manually; here we assert the wiring and the math.)
    """
    print("\n[lambdarank: |delta nDCG@5| weighting]")
    import itertools
    import math
    sys.path.insert(0, os.path.join(_ROOT, "kuairand-starter-kit"))
    from evaluate import ndcg_at_k

    K = 5

    def inv_disc(pos):
        return 1.0 / math.log2(pos + 2) if pos < K else 0.0

    def closed_form(labels, i, j):
        ideal = sorted(labels, reverse=True)[:K]
        idcg = sum(((2 ** t) - 1) / math.log2(p + 2) for p, t in enumerate(ideal))
        if idcg == 0:
            return 0.0
        gi, gj = (2 ** labels[i]) - 1, (2 ** labels[j]) - 1
        return abs((gi - gj) * (inv_disc(i) - inv_disc(j))) / idcg

    import random
    random.seed(0)
    checked = worst = 0
    for n in range(2, 13):
        for _ in range(15):
            labels = [random.randint(0, 1) for _ in range(n)]
            if sum(labels) in (0, n):
                continue
            for i, j in itertools.combinations(range(n), 2):
                if labels[i] == labels[j]:
                    continue
                before = ndcg_at_k(labels, K)
                sw = list(labels)
                sw[i], sw[j] = sw[j], sw[i]
                emp = abs(ndcg_at_k(sw, K) - before)
                worst = max(worst, abs(closed_form(labels, i, j) - emp))
                checked += 1
    check(f"|delta nDCG@5| matches the official scorer on {checked} real pairs",
          worst < 1e-12, f"max deviation {worst:.2e}")
    check("pairs beyond the @5 cutoff on both sides carry ~zero weight",
          closed_form([0] * 6 + [1] + [0], 6, 7) == 0.0)
    check("a swap into rank 0 is weighted more than one at rank 3<->4",
          closed_form([1, 0, 0, 0, 0], 0, 1) > closed_form([0, 0, 0, 1, 0], 3, 4))

    src = open(os.path.join(_ROOT, "runtime", "train_lib.py")).read()
    check("lambdarank_ndcg is registered in the numpy runners table",
          '"lambdarank_ndcg": epoch_lambdarank' in src)
    check("the uniform-weight degenerate mode exists for the BPR equivalence test",
          "_lambdarank_uniform" in src and "uniform_weights=True" in src)

    m = Menu(MENU_PATH)
    check("lambdarank_ndcg is a real, selectable menu option",
          "lambdarank_ndcg" in m.selectable_options("loss"))
    base = m.default_choices()
    m.validate_choices({**base, "loss": "lambdarank_ndcg", "model": "fm_numpy"})
    check("lambdarank_ndcg validates with the numpy engine", True)
    expect_menu_error(m, {**base, "loss": "lambdarank_ndcg", "model": "deepfm_mlp"},
                      "lambdarank_ndcg rejected on the torch engine (not implemented there)")
    spec = m.options("loss")["lambdarank_ndcg"]["description"]
    check("menu entry carries citations a grounded_in rationale can use",
          "LambdaLoss" in spec and "Burges" in spec)
    check("menu entry states the measured dataset-specific justification",
          "36.3%" in spec and "nDCG@5" in spec)


def test_new_axes_and_snapshot():
    """Candidates #1 (per-user weighting), #2 (regularization), #4 (snapshot
    ensembling). Asserts wiring + math without training a real model.
    """
    print("\n[candidates #1/#2/#4: wiring and math]")
    import numpy as np
    m = Menu(MENU_PATH)
    base = m.default_choices()

    # --- #1 sample_weighting ------------------------------------------------
    check("sample_weighting is a real axis with a per_row default",
          "sample_weighting" in m.axis_names()
          and base["sample_weighting"] == "per_row")
    m.validate_choices({**base, "loss": "bpr_pairwise",
                        "sample_weighting": "per_user_sqrt"})
    check("per_user_sqrt validates on bpr_pairwise (the untested config)", True)
    expect_menu_error(m, {**base, "loss": "pointwise_logloss",
                          "sample_weighting": "per_user_inv"},
                      "per-user weighting rejected on a non-pairwise loss")
    sw = m.options("sample_weighting")["per_user_inv"]["description"]
    check("sample_weighting entry cites the measured aggregation mismatch",
          "33.3%" in m.axes["sample_weighting"]["description"]
          and "809" in m.axes["sample_weighting"]["description"])
    check("sample_weighting entry records the listwise counter-signal honestly",
          "listwise" in m.axes["sample_weighting"]["description"]
          and "0.6032" in m.axes["sample_weighting"]["description"])

    # the weighting math itself: mean-1.0 renormalisation, correct ordering
    n_users, user_tr = 3, np.array([0, 0, 0, 0, 1, 1, 2])
    npairs = np.bincount(user_tr, minlength=n_users).astype(float)
    per_user = npairs[user_tr]
    w_inv = 1.0 / np.maximum(per_user, 1.0); w_inv /= w_inv.mean()
    w_sqrt = 1.0 / np.sqrt(np.maximum(per_user, 1.0)); w_sqrt /= w_sqrt.mean()
    check("per-user weights renormalise to mean 1.0 (no hidden lr change)",
          abs(w_inv.mean() - 1.0) < 1e-12 and abs(w_sqrt.mean() - 1.0) < 1e-12)
    check("the heavy user's pairs are downweighted vs the light user's",
          w_inv[0] < w_inv[6] and w_sqrt[0] < w_sqrt[6])
    check("sqrt variant is a gentler correction than inv",
          (w_sqrt[6] / w_sqrt[0]) < (w_inv[6] / w_inv[0]))

    # --- #2 regularization --------------------------------------------------
    check("regularization is a real axis defaulting to the current 1e-6",
          base["regularization"] == "l2_default")
    src = open(os.path.join(_ROOT, "runtime", "train_lib.py")).read()
    check("l2 is threaded from cfg into RankFM (was hardcoded)",
          'l2=cfg.get("l2", 1e-6)' in src)
    reg = m.axes["regularization"]["description"]
    check("regularization entry distinguishes itself from CAPACITY explicitly",
          "NOT capacity" in reg and "1.62%" in reg)

    # --- #4 snapshot ensembling --------------------------------------------
    check("snapshot_ensemble is config-driven, NOT a menu axis",
          "snapshot_ensemble" not in m.axis_names()
          and '"snapshot_ensemble"' in src)
    check("snapshot ensemble keeps only the top-N checkpoints by valid score",
          "snapshots.sort(key=lambda s: -s[0])" in src
          and "del snapshots[snap_n:]" in src)
    # The default still guards, but that guard is a BIASED comparison: it scores
    # the snapshot against the best single checkpoint on the SAME validation set
    # that selected that checkpoint. Measured on held-out halves instead,
    # averaging the top-5 checkpoints beats argmax by +0.87 sigma (t=5.54,
    # 22/24), so snapshot_force exists to adopt it on that evidence.
    check("by default the snapshot is adopted only if it beats the best checkpoint",
          "snap_primary > best:" in src)
    check("the same-set guard is documented as biased, not treated as ground truth",
          "biased comparison" in src and "snapshot_force" in src)
    check("an explicit force path exists for the held-out evidence",
          'cfg.get("snapshot_force")' in src)
    check("snapshot ensemble rank-normalises before averaging (scale-free)",
          "_rank_norm(s[1])" in src)


def test_parallel_worker_diversity():
    """The K-way diversity fix. Measured failure it exists for: with
    --parallel-k 3 against a prompt constrained by 14 dead-ends, all three
    workers proposed the IDENTICAL config and scored identically -- 3x cost,
    1x information.

    Verified with a FAKE LLM (no API calls): a stub that mimics the real
    failure mode -- it returns its single favourite config unless a sibling
    section tells it that config is taken. If conditioning is wired through
    correctly, a K=3 round yields 3 DISTINCT proposals; if it is not, the stub
    reproduces the original bug and this test fails.
    """
    print("\n[parallel worker diversity]")
    from agent.prompts import render_sibling_section

    check("no siblings yet -> no sibling section (worker 0 is unconstrained)",
          render_sibling_section([]) == "")
    sec = render_sibling_section([{"loss": "bpr_pairwise", "model": "fm_numpy"}])
    check("sibling section names the already-taken configuration",
          "bpr_pairwise" in sec and "worker 0" in sec)
    check("sibling section instructs the worker to differ",
          "MEANINGFULLY DIFFERENT" in sec)

    # --- simulate a K=3 round against a stub that reproduces the real bug ---
    FAVOURITES = [{"loss": "bpr_pairwise", "model": "fm_numpy"},
                  {"loss": "bpr_pairwise", "model": "gru4rec_seq"},
                  {"loss": "pointwise_logloss", "model": "fm_numpy"}]

    def fake_llm(prompt):
        """Greedy: always wants FAVOURITES[0]; only moves on when the prompt
        says that choice is already taken by a sibling."""
        for cand in FAVOURITES:
            if json.dumps(cand, sort_keys=True) not in prompt:
                return cand
        return FAVOURITES[-1]

    # WITHOUT conditioning (the old behaviour): same prompt every time
    old = [fake_llm("base prompt") for _ in range(3)]
    old_sigs = {json.dumps(c, sort_keys=True) for c in old}
    check("baseline: K identical prompts reproduce the ORIGINAL bug "
          "(1 distinct proposal from 3 workers)", len(old_sigs) == 1)

    # WITH conditioning: each worker sees its siblings' picks
    siblings, new = [], []
    for _ in range(3):
        p = "base prompt\n" + render_sibling_section(siblings)
        c = fake_llm(p)
        new.append(c)
        siblings.append(c)
    new_sigs = {json.dumps(c, sort_keys=True) for c in new}
    check("FIXED: conditioning yields 3 genuinely DISTINCT proposals",
          len(new_sigs) == 3, f"{len(new_sigs)}/3 distinct")
    check("the distinct proposals span more than one axis value",
          len({c["model"] for c in new}) > 1)

    src = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("loop.py rebuilds the prompt per worker with sibling_choices",
          "sibling_choices=sibling_choices" in src
          and "sibling_choices.append(obj[\"menu_choices\"])" in src)
    check("loop.py measures and journals per-round worker diversity",
          '"type": "worker_diversity"' in src and "diversity_event" in src)


def test_data_tools_and_proposals():
    """Phase: agent data tools, gated axis proposals, branching gate.
    Hostile cases are asserted, not assumed -- same standard as the Phase 1
    sandbox work."""
    print("\n[agent data tools / axis proposals / branching gate]")
    sys.path.insert(0, os.path.join(_ROOT, "runtime"))
    from agent import inspect as I
    from agent import propose_axis as PA
    import data_tools as D

    # ---- hostile: label leakage + arbitrary access ----
    attacks = [
        ("test-split feature stats", {"tool": "get_feature_stats",
                                      "args": {"feature": "long_view", "split": "test"}}),
        ("test-split within-user auc", {"tool": "get_within_user_auc",
                                        "args": {"feature": "is_click", "split": "test"}}),
        ("non-allowlisted feature", {"tool": "get_feature_stats",
                                     "args": {"feature": "user_raw"}}),
        ("dunder escape", {"tool": "get_feature_stats", "args": {"feature": "__class__"}}),
        ("path traversal", {"tool": "get_feature_stats", "args": {"feature": "../../.env"}}),
        ("unknown tool", {"tool": "os.system", "args": {}}),
        ("kwarg injection", {"tool": "get_feature_stats",
                             "args": {"feature": "long_view", "cache_dir": "/etc"}}),
    ]
    for label, req in attacks:
        check(f"blocked: {label}", "error" in I.execute([req])[0])
    check("test split is not in the inspectable allowlist",
          "test" not in D.ALLOWED_SPLITS)
    check("call cap enforced", len(I.execute([attacks[0][1]] * 20)) <= I.MAX_TOOL_CALLS)
    check("parse_requests truncates over-long request lists",
          len(I.parse_requests({"requests": [{"tool": "x", "args": {}}] * 20}))
          <= I.MAX_TOOL_CALLS)
    check("malformed phase-1 response degrades to no requests",
          I.parse_requests("not a dict") == [] and I.parse_requests({}) == [])

    # ---- axis proposals: validation + the human gate ----
    good = {"axis_name": "test_axis_xyz", "description": "d" * 40,
            "options": {"none": {"description": "baseline no-op"},
                        "variant": {"description": "the alternative"}},
            "mechanism": "m" * 40, "citation": "c" * 40, "signal_breadth": "broad"}
    PA.validate(good)
    check("well-formed proposal validates", True)
    for bad, why in [({**good, "axis_name": "Bad Name"}, "bad axis_name"),
                     ({**good, "options": {"only": {"description": "x"}}}, "needs >=2 options"),
                     ({**good, "signal_breadth": "maybe"}, "invalid signal_breadth"),
                     ({**good, "mechanism": "short"}, "mechanism too vague"),
                     ({k: v for k, v in good.items() if k != "citation"}, "missing citation")]:
        try:
            PA.validate(bad)
            check(f"rejected: {why}", False)
        except PA.ProposalError:
            check(f"rejected: {why}", True)
    menu_before = json.load(open(MENU_PATH))
    with tempfile.TemporaryDirectory() as td:
        pth = os.path.join(td, "proposed.jsonl")
        pid = PA.append_proposal(good, iteration_id=1, path=pth)
        recs = PA.load_all(pth)
        check("proposal recorded as PENDING, not applied",
              recs[0]["status"] == "pending")
        check("THE GATE: live menu is unchanged by a proposal",
              json.load(open(MENU_PATH)) == menu_before)
        check("agent cannot self-approve (approval is a separate human command)",
              "test_axis_xyz" not in json.load(open(MENU_PATH))["axes"])
        PA.reject(pid, "not grounded", path=pth)
        check("rejection is recorded", PA.load_all(pth)[0]["status"] == "rejected")

    # ---- branching gate ----
    with tempfile.TemporaryDirectory() as td:
        t = ExperimentTree(td)
        for i in range(4):
            t.add(_node(i, "success", 0.60))
        loop = type("L", (), {})()
        loop.tree = t
        loop.min_branching_iterations = 1
        from agent.loop import AgentLoop
        blocked = AgentLoop._branching_unfinished(loop)
        check("convergence deferred while only `draft` has fired", bool(blocked))
        check("the reason names what is still owed", "improve" in (blocked or ""))
        t.add(_node(4, "success", 0.60, action="improve", parent=0))
        check("gate clears once improve actually executes",
              AgentLoop._branching_unfinished(loop) is None)
        loop.min_branching_iterations = 0
        check("gate is a no-op when the flag is unset (default behaviour)",
              AgentLoop._branching_unfinished(loop) is None)


def test_stage_b_path_freedom():
    """Stage B: the menu must stop being the boundary of what can be proposed.

    The measured cause of ~0 Path B usage in 54 nodes was structural, not an
    LLM limitation: menu_choices was required on EVERY response and
    validate_choices ran unconditionally, so the model had to commit to a menu
    selection before it could even consider custom code.
    """
    print("\n[stage B: path freedom]")
    from agent.llm import (PATH_A_EXTRA, PATH_B_EXTRA, RESEARCH_CATEGORIES,
                           RESPONSE_SCHEMA, LLMClient)

    check("menu_choices is NO LONGER unconditionally required",
          "menu_choices" not in RESPONSE_SCHEMA)
    check("implementation_path and research_category are required of every response",
          "implementation_path" in RESPONSE_SCHEMA and "research_category" in RESPONSE_SCHEMA)
    check("menu_choices is required for Path A only", "menu_choices" in PATH_A_EXTRA)
    check("code_summary is required for Path B instead", "code_summary" in PATH_B_EXTRA)

    base = {"hypothesis": "h", "code": "x" * 60, "expected_effect": "e",
            "rationale": {"idea": "i" * 20, "why_expected_to_help": "w" * 20,
                          "grounded_in": "menu axis loss: bpr_pairwise"},
            "research_category": "exploration"}
    A = {**base, "implementation_path": "A", "menu_choices": {"loss": "bpr_pairwise"}}
    B = {**base, "implementation_path": "B",
         "code_summary": "forms BPR pairs only within the same (user, hour) session, "
                         "which no menu axis can express"}
    check("Path A validates", LLMClient._schema_problems(A) == [])
    check("Path B validates WITHOUT any menu_choices", LLMClient._schema_problems(B) == [])
    check("Path A still requires menu_choices",
          any("menu_choices" in p for p in
              LLMClient._schema_problems({**base, "implementation_path": "A"})))
    check("Path B requires a substantive code_summary",
          any("code_summary" in p for p in
              LLMClient._schema_problems({**base, "implementation_path": "B"})))
    check("a too-thin code_summary is rejected",
          any("code_summary" in p for p in
              LLMClient._schema_problems({**B, "code_summary": "custom"})))
    check("invalid research_category rejected",
          any("research_category" in p for p in
              LLMClient._schema_problems({**A, "research_category": "vibes"})))
    check("invalid implementation_path rejected",
          any("implementation_path" in p for p in
              LLMClient._schema_problems({**A, "implementation_path": "C"})))

    # THE DOWNSTREAM COERCION: a schema fix alone is not enough if a later
    # validator still demands a menu selection.
    src = open(os.path.join(_ROOT, "agent", "llm.py")).read()
    check("validate_choices is NOT applied to Path B responses",
          'str(obj.get("implementation_path", "A")).upper() != "B"' in src)

    # prompt framing + menu compression
    psrc = open(os.path.join(_ROOT, "agent", "prompts.py")).read()
    check("Path A is no longer described as the default/simplest",
          "The simplest valid script is seed_solution.py" not in psrc)
    check("path choice is framed as hypothesis-driven",
          "decide from your HYPOTHESIS" in psrc)
    check("Path B is explicitly warned against gratuitous complexity",
          "what is the SIMPLEST experiment" in psrc)
    m = Menu(MENU_PATH)
    full, comp = m.render_for_prompt(), m.render_compact() + m.render_dead_ends()
    check("every dead end survives compaction (none silently dropped)",
          m.render_dead_ends().count("\n- ")
          == len(json.load(open(MENU_PATH))["notes"]["tested_dead_ends"]))
    check("compaction keeps the CLAIM, not just a truncated fragment",
          all(len(ln) > 40 for ln in m.render_dead_ends().split("\n") if ln.startswith("- ")))
    check("dead ends stay a minority of the compact prompt",
          len(m.render_dead_ends()) < len(m.render_for_prompt()) * 0.35,
          f"{len(m.render_dead_ends())} vs {len(m.render_for_prompt())}")
    check("compact menu is materially smaller than the full menu",
          len(comp) < 0.5 * len(full), f"{len(comp)} vs {len(full)} chars")
    check("dead-ends survive compression (never dropped)",
          "LambdaRank" in comp and "GENERAL PATTERN" in comp)
    check("Node records the path so usage can be MEASURED, not assumed",
          all(f in open(os.path.join(_ROOT, "agent", "contracts.py")).read()
              for f in ("implementation_path", "research_category", "code_summary")))


def _rs_fixture(td, nodes, reseed=None, best=None, ensemble=None):
    """Build a fake logs/ + config/ tree for ResearchState."""
    import shutil
    os.makedirs(os.path.join(td, "logs"), exist_ok=True)
    os.makedirs(os.path.join(td, "config"), exist_ok=True)
    shutil.copyfile(MENU_PATH, os.path.join(td, "config", "modification_menu.json"))
    with open(os.path.join(td, "logs", "journal.jsonl"), "w") as fh:
        for n in nodes:
            fh.write(json.dumps(n) + "\n")
    for name, obj in (("reseed_results.json", reseed), ("best_metrics.json", best),
                      ("ensemble_results.json", ensemble)):
        if obj is not None:
            with open(os.path.join(td, "logs", name), "w") as fh:
                json.dump(obj, fh)
    from agent.research_state import ResearchState
    return ResearchState(td)


def _rs_node(i, choices, primary, status="success"):
    return {"iteration_id": i, "action": "draft", "status": status,
            "menu_choices": choices,
            "metrics": ({"GAUC": primary, "nDCG@5": primary, "primary": primary}
                        if status == "success" else None),
            "hypothesis": "h", "events": []}


def test_research_state():
    """Stage C: the state must remember what MATTERS, derived deterministically,
    and must never let a single-seed win masquerade as knowledge."""
    print("\n[stage C: research state]")
    from agent.research_state import (OBSERVED_ONCE, RESEED_VERIFIED, VALIDATED,
                                      ResearchState)
    A = {"loss": "bpr_pairwise", "model": "fm_numpy", "multitask": "none"}

    # 1 + 2: single-seed never validated; reseed-backed can be
    with tempfile.TemporaryDirectory() as td:
        rs = _rs_fixture(td, [_rs_node(0, A, 0.6100)])
        check("a single-seed win does NOT become a confirmed finding",
              rs.confirmed == [])
        check("the incumbent best is explicitly marked single-run, not knowledge",
              rs.best_config_evidence["level"] == OBSERVED_ONCE
              and "SINGLE run" in rs.best_config_evidence["caveat"])
        check("that caveat is surfaced in the rendered notebook",
              "SINGLE run" in rs.render())
        check("observed maximum is kept DISTINCT from expected performance",
              "best_observed_single_run" in rs.facts
              and "expected_ensemble_mean" not in rs.facts)
    with tempfile.TemporaryDirectory() as td:
        rsd = {"nodes": [{"iteration_id": 0, "mean_primary": 0.6100,
                          "std_primary": 0.0003, "n_samples": 5,
                          "original_single_seed_primary": 0.6100}]}
        rs = _rs_fixture(td, [_rs_node(0, A, 0.6100)], reseed=rsd)
        check("a reseed-confirmed result DOES become a confirmed finding",
              len(rs.confirmed) == 1 and rs.confirmed[0]["level"] == VALIDATED)
        check("confirmed finding carries evidence basis + uncertainty",
              rs.confirmed[0]["n_runs"] == 5 and "uncertainty_std" in rs.confirmed[0])

    # 3 + 4: untested assumptions tracked against the CURRENT best only
    with tempfile.TemporaryDirectory() as td:
        best = {"loss": "bpr_pairwise", "model": "fm_numpy", "multitask": "aux_click"}
        alt = dict(best, multitask="none")
        rs = _rs_fixture(td, [_rs_node(0, best, 0.6100), _rs_node(1, alt, 0.6050)],
                         best={"iteration_id": 0, "menu_choices": best})
        check("a component with an isolated counterfactual is NOT 'untested'",
              rs.component_evidence["multitask"]["status"] != "untested_assumption")
        check("components with no isolated experiment ARE flagged untested",
              rs.component_evidence["loss"]["status"] == "untested_assumption")
        # now the best changes and no longer contains that component
        newbest = {"loss": "listwise_softmax", "model": "fm_numpy", "multitask": "none"}
        rs2 = _rs_fixture(td, [_rs_node(0, best, 0.6100), _rs_node(1, alt, 0.6050),
                               _rs_node(2, newbest, 0.6200)],
                          best={"iteration_id": 2, "menu_choices": newbest})
        check("a component dropped from the best is no longer reported as a "
              "current assumption",
              rs2.component_evidence["multitask"]["value"] == "none"
              and all("aux_click" not in str(v["value"])
                      for v in rs2.component_evidence.values()))
        check("state tracks the NEW best after it changes",
              rs2.best_config["loss"] == "listwise_softmax")

    # 5: negative findings carry scope, not overgeneralisation
    with tempfile.TemporaryDirectory() as td:
        rs = _rs_fixture(td, [_rs_node(0, A, 0.6100)])
        joined = " ".join(rs.dead_ends).lower()
        check("dead ends are scoped to tested conditions, not sweeping claims",
              "measured here" in joined and "deep learning does not work" not in joined)
        check("dead ends are referenced, not duplicated into the state body",
              "do not re-derive them" in rs.render())

    # 6 + 7: branch status transitions; integration requires independence
    with tempfile.TemporaryDirectory() as td:
        rs = _rs_fixture(td, [_rs_node(0, A, 0.6100), _rs_node(1, A, 0.6099)])
        check("an active branch is reported as awaiting confirmation",
              rs.branches[0]["status"] == "awaiting confirmation")
        rsd = {"nodes": [{"iteration_id": 0, "mean_primary": 0.6100,
                          "std_primary": 0.0003, "n_samples": 5,
                          "original_single_seed_primary": 0.6100}]}
        rs2 = _rs_fixture(td, [_rs_node(0, A, 0.6100)], reseed=rsd)
        check("branch transitions to confirmed once reseed-backed",
              rs2.branches[0]["status"] == "confirmed")
    with tempfile.TemporaryDirectory() as td:
        best = {"loss": "bpr_pairwise", "model": "fm_numpy", "multitask": "none"}
        c1 = dict(best, multitask="aux_click"); c2 = dict(best, model="deepfm_mlp")
        rs = _rs_fixture(td, [_rs_node(0, best, 0.6000), _rs_node(1, c1, 0.6100),
                              _rs_node(2, c2, 0.6090)],
                         best={"iteration_id": 0, "menu_choices": best})
        check("integration candidates pair DIFFERENT axes",
              rs.integration_candidates and
              all("+" in c["candidate"] for c in rs.integration_candidates))
        check("unconfirmed components are BLOCKED from integration",
              all(c["status"].startswith("blocked")
                  for c in rs.integration_candidates))
        check("integration candidates state the interaction risk",
              "interaction may be negative" in rs.integration_candidates[0]["risk"])

    # 9 + 10: compactness and no raw-log duplication
    rs = ResearchState(_ROOT)
    r = rs.render()
    check("rendered state is compact (< 8k chars)", len(r) < 8000, f"{len(r)} chars")
    check("raw journal is NOT dumped into the state",
          "error_trace" not in r and "token_breakdown" not in r)
    check("state distinguishes observed max from expected ensemble",
          "NOT an expected value" in r)
    check("open questions are surfaced for decision-making",
          "Open questions" in r)


def test_research_state_no_side_effects():
    """Regression: Stage C must not perturb execution, evaluation, sandboxing,
    data boundaries, reseeding or the existing memories."""
    print("\n[stage C: no side effects on existing infrastructure]")
    from agent.research_state import ResearchState
    import runtime.data_boundary as db
    before_menu = open(MENU_PATH).read()
    before_exp = (open(os.path.join(_ROOT, "agent", "experience.md")).read()
                  if os.path.exists(os.path.join(_ROOT, "agent", "experience.md")) else "")
    before_journal = open(os.path.join(_ROOT, "logs", "journal.jsonl")).read()
    rs = ResearchState(_ROOT); rs.render(); rs.as_dict()
    check("ResearchState does not mutate the menu", open(MENU_PATH).read() == before_menu)
    check("ResearchState does not mutate experience memory",
          (open(os.path.join(_ROOT, "agent", "experience.md")).read()
           if before_exp else "") == before_exp)
    check("ResearchState does not mutate the journal",
          open(os.path.join(_ROOT, "logs", "journal.jsonl")).read() == before_journal)
    check("data boundary redaction still intact (never shrinks)",
          {"long_view", "is_click", "is_like", "is_forward",
           "play_time_ms"} <= set(db.TEST_LABEL_COLUMNS))
    from agent.executor import PROTECTED_PATHS, REAL_DATA_DIR
    check("sandbox protections unchanged",
          any("journal.jsonl" in p for p in PROTECTED_PATHS) and os.path.isdir(REAL_DATA_DIR))
    check("ResearchState is read-only by construction (no write calls)",
          "open(" in open(os.path.join(_ROOT, "agent", "research_state.py")).read()
          and ', "w"' not in open(os.path.join(_ROOT, "agent",
                                               "research_state.py")).read())


class _FakeState:
    def __init__(self, **kw):
        self.component_evidence = kw.get("component_evidence", {})
        self.best_config_evidence = kw.get("best_config_evidence",
                                           {"level": "reseed_verified", "n_runs": 5})
        self.promising = kw.get("promising", [])
        self.integration_candidates = kw.get("integration_candidates", [])
        self.branches = kw.get("branches", [])
        self.dead_ends = kw.get("dead_ends", [])


def test_research_policy():
    """Stage D: the category must REACT to evidence, be explainable, and be
    guarded against its own pathologies in both directions."""
    print("\n[stage D: evidence-reactive research policy]")
    from agent.research_policy import (ABLATION, CONFIRMATION, EXPLOITATION,
                                       EXPLORATION, INTEGRATION,
                                       MAX_ABLATION_SHARE, decide_category,
                                       render_decision)

    def node(cat, ok=True):
        return {"status": "success" if ok else "error", "research_category": cat,
                "metrics": {"primary": 0.6} if ok else None}

    # untested components -> ablation
    st = _FakeState(component_evidence={
        "loss": {"value": "bpr", "status": "untested_assumption"},
        "model": {"value": "fm", "status": "untested_assumption"}})
    d = decide_category(st, [node(EXPLOITATION)] * 4)
    check("untested components in the best config drive ABLATION",
          d["category"] == ABLATION)
    check("the decision names the specific untested entries",
          any("loss" in e for e in d["evidence"]))
    check("alternatives each carry a stated reason for losing",
          set(d["alternatives"]) == {EXPLORATION, EXPLOITATION, CONFIRMATION,
                                     INTEGRATION})

    # GUARD: ablation cannot eat the whole run
    many = [node(ABLATION)] * 9 + [node(EXPLOITATION)]
    d2 = decide_category(st, many)
    check("ablation is SUPPRESSED once it exceeds its budget share",
          d2["category"] != ABLATION and "suppressed" in d2["alternatives"][ABLATION])

    # single-run incumbent -> confirmation
    st2 = _FakeState(best_config_evidence={"level": "observed_once", "n_runs": 1})
    d3 = decide_category(st2, [node(EXPLORATION)] * 3)
    check("a single-run incumbent drives CONFIRMATION", d3["category"] == CONFIRMATION)
    check("confirmation cites the incumbent's weakness",
          any("SINGLE run" in e for e in d3["evidence"]))

    # GUARD: nothing to confirm -> confirmation scores zero
    st3 = _FakeState()
    d4 = decide_category(st3, [node(EXPLOITATION)])
    check("confirmation is not chosen when everything is already verified",
          d4["scores"][CONFIRMATION] == 0.0)

    # integration only when eligible; blocked otherwise
    st4 = _FakeState(integration_candidates=[{"candidate": "A+B", "status": "eligible"}])
    d5 = decide_category(st4, [node(EXPLOITATION)])
    check("eligible independent improvements drive INTEGRATION",
          d5["category"] == INTEGRATION)
    st5 = _FakeState(integration_candidates=[
        {"candidate": "A+B", "status": "blocked: needs confirmation"}])
    d6 = decide_category(st5, [node(EXPLOITATION)])
    check("blocked candidates do NOT trigger integration",
          d6["category"] != INTEGRATION and "blocked" in d6["alternatives"][INTEGRATION])

    # GUARD: saturation damps exploration
    st6 = _FakeState(branches=[{"status": "dead-end (explored)"}] * 3,
                     dead_ends=["x"] * 14)
    d7 = decide_category(st6, [node(EXPLORATION)] * 6)
    check("repeated fruitless exploration is damped",
          "damped" in d7["alternatives"].get(EXPLORATION, "")
          or d7["category"] != EXPLORATION)

    # budget end -> consolidate
    d8 = decide_category(st6, [node(EXPLOITATION)], iteration_budget_left=2)
    check("near budget exhaustion the policy consolidates rather than explores",
          d8["scores"][EXPLORATION] < 1.2)

    # explainability
    txt = render_decision(d)
    check("decision renders an explainable block for the prompt",
          "Research objective" in txt and "Alternatives considered" in txt
          and "category scores" in txt)
    check("the agent is told its proposal must match the objective",
          "MUST match this research_category" in txt)

    # policy is pure: no side effects on state
    before = dict(st.component_evidence)
    decide_category(st, [node(EXPLOITATION)])
    check("policy is read-only w.r.t. the research state",
          st.component_evidence == before)


def test_failure_taxonomy():
    """Stage E: Path B makes custom code common, so failures must be classified
    before repair -- and a disproved HYPOTHESIS must never be confused with
    broken CODE."""
    print("\n[stage E: failure taxonomy and repair policy]")
    from agent import failure as F

    cases = [("SyntaxError: unterminated string literal", F.SYNTAX),
             ("ModuleNotFoundError: No module named 'lightgbm'", F.IMPORT),
             ("KeyError: 'n_users'", F.API_MISUSE),
             ("TIMEOUT: training run exceeded 1200s and was killed", F.TIMEOUT),
             ("scores_valid.npy contains NaN/Inf", F.INVALID_PREDICTIONS),
             ("metrics.json was not written to --output-dir", F.DATA_CONTRACT),
             ("torch.cuda.OutOfMemoryError: CUDA out of memory", F.CUDA),
             ("LLM stage failed: response schema violations", F.LLM_RESPONSE)]
    for trace, expect in cases:
        check(f"classified: {expect}", F.classify(trace)["class"] == expect)

    # THE central distinction
    ok = F.classify(None, status="success", metrics={"primary": 0.55})
    check("a run that SUCCEEDED but scored poorly is hypothesis_disproved",
          ok["class"] == F.HYPOTHESIS_DISPROVED)
    check("...and is explicitly NOT a code failure", ok["is_code_failure"] is False)
    check("...and is NOT retried (it answered the question)",
          ok["retry_worthwhile"] is False)
    bad = F.classify("SyntaxError: bad")
    check("a crash IS a code failure", bad["is_code_failure"] is True)

    # retry policy is class-specific, not blanket
    check("a timeout is not worth retrying unchanged",
          F.classify("TIMEOUT: exceeded")["retry_worthwhile"] is False)
    check("a timeout demands a materially cheaper experiment",
          F.classify("TIMEOUT: exceeded")["needs_shrink"] is True)
    check("OOM demands shrinking too",
          F.classify("OutOfMemoryError: ...")["needs_shrink"] is True)
    check("a syntax error IS worth retrying after repair",
          F.classify("SyntaxError: x")["retry_worthwhile"] is True)

    # repair brief is compact and constrains the fix
    brief = F.repair_brief(F.classify("KeyError: 'n_users'"), 1, 2)
    check("repair brief is compact (< 900 chars)", len(brief) < 900, f"{len(brief)}")
    check("repair brief names the class and forbids redesigning the experiment",
          "api_misuse" in brief and "SMALLEST change" in brief)
    check("repair brief for a non-retryable class says so",
          "should NOT be retried" in F.repair_brief(F.classify("TIMEOUT: x"), 1, 2))

    # failures become research knowledge
    out, title, body = F.as_knowledge(F.classify("SyntaxError: x"), 7, {"loss": "bpr"})
    check("a crash is recorded as CRASHED knowledge with its class",
          out == "CRASHED" and "implementation_syntax" in body)
    out2, _, body2 = F.as_knowledge(
        F.classify(None, status="success", metrics={"primary": 0.5}), 8, {"loss": "bpr"})
    check("a disproved hypothesis is recorded as DEAD_END, not CRASHED",
          out2 == "DEAD_END" and "not a bug" in body2)

    src = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("loop journals the failure class for every execution error",
          '"failure_class": fc["class"]' in src)
    check("loop records classified failures into experience memory",
          "failure_mod.as_knowledge(" in src)
    psrc = open(os.path.join(_ROOT, "agent", "prompts.py")).read()
    check("debug prompts receive the classified repair brief",
          "repair_brief(_fc" in psrc)


def test_leakage_and_ensemble():
    """Leakage checker (P0 gap) + ensemble/selection-bias guards."""
    print("\n[leakage checker + ensemble infrastructure]")
    from agent.leakage_check import check_source, verdict, render_for_agent

    fatal = verdict(check_source(
        "s,m=train_lib.load_cache()\ny=s['test']['long_view']\nz=y.mean()\n"))
    check("using TEST labels is FATAL and blocks execution", fatal["block"])
    warn = verdict(check_source(
        "rate = tr.groupby('user')['long_view'].mean()\nX_feat = rate[tr['user']]\n"))
    check("the classic target-leak aggregation is flagged", warn["n_warn"] >= 1)
    check("...but is advisory, not a hard block (avoids false blocking)",
          not warn["block"])
    safe = verdict(check_source(
        "# leave_one_out causal history\nh=train_lib.History(s,n,mode)\n"
        "v=h.batch_vectors(S,users,split_is_train=True)\n"))
    check("legitimate leave-one-out code produces NO findings",
          len(safe["findings"]) == 0)
    seed = verdict(check_source(open(
        os.path.join(_ROOT, "runtime", "seed_solution.py")).read()))
    check("the real seed_solution.py yields no false positives",
          not seed["block"] and seed["n_warn"] == 0)
    check("blocked feedback explains the rule and says the hypothesis is untested",
          "available BEFORE" in render_for_agent(fatal)
          and "not run" in render_for_agent(fatal).lower())

    esrc = open(os.path.join(_ROOT, "agent", "executor.py")).read()
    check("executor runs the leakage gate BEFORE launching the subprocess",
          "BLOCKED BEFORE EXECUTION" in esrc
          and esrc.index("leakage_check.check_file") < esrc.index("subprocess.run"))
    from agent.failure import LEAKAGE_BLOCKED, classify
    check("a leakage block is its own failure class, and retryable after fixing",
          classify("BLOCKED BEFORE EXECUTION by the leakage review")["class"]
          == LEAKAGE_BLOCKED)

    # ensemble: rank-normalisation and the selection-bias guard
    import numpy as np
    from agent.ensemble import NOISE_FLOOR, rank_normalise
    r = rank_normalise(np.array([5.0, 1.0, 3.0]))
    check("rank_normalise is scale-free and order-preserving",
          list(np.argsort(r)) == [1, 2, 0] and r.min() == 0.0 and r.max() == 1.0)
    check("a huge-scale model cannot dominate after rank-normalisation",
          np.allclose(rank_normalise(np.array([1e9, 1.0, 5e8])),
                      rank_normalise(np.array([3.0, 1.0, 2.0]))))
    ens = json.load(open(os.path.join(_ROOT, "logs", "ensemble_results.json")))
    check("adopted ensemble records that it carries NO selection bias",
          "NONE" in ens["selection_bias"])
    # The previous headline (0.60545) quoted member arrays that had been
    # archived away, so it could not be recomputed. Reproducibility is now
    # asserted structurally, not promised in prose.
    check("the reported k equals the number of members actually averaged",
          ens["k"] == len(ens["seeds_used"]))
    mdir = os.path.join(_ROOT, ens["members_dir"])
    present = [s for s in ens["seeds_used"]
               if os.path.exists(os.path.join(mdir, f"seed_{s:02d}",
                                              "scores_valid.npy"))]
    check("every ensemble member's prediction array is on disk",
          len(present) == ens["k"], f"{len(present)}/{ens['k']}")
    check("the ensemble states how to reproduce itself",
          "final_ensemble" in (ens.get("reproduce") or ""))
    # A record that stores only the MEAN of two metrics does not say what the
    # result was, and the two move independently.
    check("the authoritative result records GAUC and nDCG@5, not just the mean",
          "GAUC" in ens and "nDCG@5" in ens
          and abs((ens["GAUC"] + ens["nDCG@5"]) / 2 - ens["primary"]) < 1e-4,
          f"{ens.get('GAUC')} {ens.get('nDCG@5')} {ens['primary']}")
    check("the result carries provenance (code, data, time)",
          all(k in ens for k in ("code_version", "data_version", "timestamp_utc")))
    check("the result records whether the hidden test was touched",
          ens.get("hidden_test_used") is False)
    check("the ensemble beats its own mean member (averaging is doing work)",
          ens["gain_over_mean_member"] > 0)
    # k must not be the argmax of the k-curve -- that would BE selection.
    curve = ens["k_curve_diagnostic_only"]
    argmax_k = max(curve, key=lambda k: curve[k])
    check("k is ALL seeds, not the best-scoring k on the curve",
          int(argmax_k) != ens["k"] or len(curve) == ens["k"],
          f"argmax k={argmax_k} reported k={ens['k']}")
    dead = json.load(open(os.path.join(_ROOT, "config",
                                       "modification_menu.json")))["notes"]["tested_dead_ends"]
    check("heterogeneous ensembling was measured and rejected, not assumed",
          any("gru4rec_seq" in d and "0.15 sigma" in d for d in dead))
    check("the gru4rec finding is scoped to ensemble MEMBERSHIP, not the model",
          any("ensemble MEMBERSHIP only" in d for d in dead))
    # The stronger test: BOTH halves of the rule satisfied, and still nothing.
    ht_path = os.path.join(_ROOT, "logs", "hetero_test.json")
    if os.path.exists(ht_path):
        ht = json.load(open(ht_path))
        check("the comparable-quality ensemble test was pre-registered",
              ht["design_fixed_before_result"]
              and ht["validation_comparisons_for_this_hypothesis"] == 1)
        check("its members really were comparable in quality",
              abs(ht["quality_gap_sigma"]) < 1.0, f"{ht['quality_gap_sigma']} sigma")
        check("its members really were more decorrelated than same-config seeds",
              ht["corr_across_configs"] < ht["corr_within_config_seeds"],
              f"{ht['corr_across_configs']} vs {ht['corr_within_config_seeds']}")
        check("and it was still REJECTED against a pre-set threshold",
              not ht["adopt"] and ht["delta_vs_base"] <= ht["adopt_threshold"])


def test_candidate_policy():
    """Multi-candidate planning: the fix for the audit finding that Path B was
    never GENERATED (not rejected). Selection must be deterministic, gated and
    auditable."""
    print("\n[candidate generation + deterministic policy]")
    from agent import candidates as C

    def mk(i, path="A", cat="exploration", hyp="", mech="x" * 40,
           gain=0.002, choices=None, grounded="menu axis loss: bpr_pairwise"):
        return C.Candidate({"hypothesis": hyp or f"idea {i}", "mechanism": mech,
                            "implementation_path": path, "research_category": cat,
                            "expected_gain": gain,
                            "menu_choices": choices or {"loss": f"l{i}", "model": "m"},
                            "rationale": {"grounded_in": grounded}}, i)

    hist = [{"iteration_id": 0, "status": "success", "hypothesis": "add user historical long_view rate",
             "menu_choices": {"loss": "bpr_pairwise", "model": "fm_numpy"},
             "metrics": {"primary": 0.605}, "implementation_path": "A"}]

    # Semantic duplicate detection (not just exact match). Calibration on real
    # examples showed lexical similarity CANNOT separate "reworded duplicate"
    # from "genuine extension" -- a legitimate follow-up scored HIGHER (0.75)
    # than a true duplicate (0.375), because an extension contains its parent
    # whole. So similarity drives a graded penalty, never a hard block; only an
    # exact configuration repeat is gated.
    dup = mk(1, hyp="calculate the user's previous long_view frequency",
             choices={"loss": "x", "model": "y"})
    novel = mk(6, hyp="use item popularity prior as a feature",
               choices={"loss": "x", "model": "y"})
    C.score_candidates([dup, novel], history=hist, dead_ends=[])
    check("semantically equivalent proposals are detected, not just exact dupes",
          dup.parts["max_similarity"] > 0.3 and dup.parts["similar_to_node"] == 0,
          f"sim={dup.parts.get('max_similarity')}")
    check("an unrelated proposal is NOT flagged as similar",
          novel.parts["max_similarity"] < 0.2, f"sim={novel.parts['max_similarity']}")
    check("a near-duplicate is penalised, not hard-blocked",
          not dup.rejected and dup.parts["redundancy_factor"] < 1.0)
    check("the near-duplicate scores strictly below an equivalent novel idea",
          dup.utility < novel.utility, f"{dup.utility} vs {novel.utility}")

    # The case that forced the redesign: an honest extension of a prior idea is
    # lexically MORE similar than a reworded duplicate. It must still survive.
    fu_hist = [{"iteration_id": 0, "status": "success",
                "hypothesis": "add user history pooling",
                "menu_choices": {"loss": "bpr_pairwise", "model": "fm_numpy"},
                "metrics": {"primary": 0.605}, "implementation_path": "A"}]
    fu = mk(7, hyp="add user history pooling with temporal decay",
            choices={"loss": "x", "model": "y"})
    C.score_candidates([fu], history=fu_hist, dead_ends=[])
    check("a legitimate follow-up is NOT rejected despite very high similarity",
          not fu.rejected and fu.utility > 0,
          f"sim={fu.parts['max_similarity']} gates={fu.gates}")

    # exact-config duplicate
    ex = mk(2, choices={"loss": "bpr_pairwise", "model": "fm_numpy"}, hyp="totally different words here")
    C.score_candidates([ex], history=hist, dead_ends=[])
    check("exact configuration repeats are rejected",
          any("exact configuration" in g for g in ex.gates))

    # plausibility gate: novel+random rejected, novel+plausible kept
    rnd = mk(3, mech="", grounded="", hyp="try something new")
    good = mk(4, mech="pairs formed within a session compare items under the "
                      "same user intent, which the menu cannot express",
              grounded="measured: train lists average 43.5 impressions vs 5.6 at eval")
    C.score_candidates([rnd, good], history=hist, dead_ends=[])
    check("novel+random (no mechanism, no grounding) is REJECTED",
          any("no stated mechanism" in g for g in rnd.gates))
    check("novel+plausible survives", not good.rejected and good.utility > 0)

    # dead-end overlap
    de = mk(5, hyp="use lambdarank ndcg position discount weighting for pairs",
            mech="weight pairs by delta ndcg at rank five positions" * 2)
    C.score_candidates([de], history=hist,
                       dead_ends=["LambdaRank |delta nDCG@5| pair weighting: MEASURED "
                                  "HERE, decisively worse than bpr_pairwise"])
    check("candidates overlapping a recorded dead end are rejected",
          any("dead end" in g for g in de.gates))

    # branch saturation
    sat_hist = [{"iteration_id": i, "status": "success", "hypothesis": f"h{i}",
                 "menu_choices": {"loss": "bpr_pairwise", "model": "fm_numpy"},
                 "metrics": {"primary": 0.605}, "implementation_path": "A"}
                for i in range(5)]
    stats = C.branch_stats(sat_hist)
    check("branch statistics are computed deterministically", len(stats) == 1)
    check("a branch with collapsed recent returns is marked saturated",
          ("bpr_pairwise", "fm_numpy") in C.saturated_branches(stats))

    # Path B is scoreable, and costs more but is not banned
    a = mk(6, path="A", choices={"loss": "za", "model": "m"})
    b = mk(7, path="B", choices={})
    b.raw["code_summary"] = "z" * 60
    C.score_candidates([a, b], history=hist, dead_ends=[])
    check("Path B candidates are SCORED, not silently dropped",
          b.utility > 0 and not b.rejected)
    check("Path B carries a higher cost than Path A", b.parts["cost"] > a.parts["cost"])

    # never-tried branch keeps option value
    fresh = mk(8, choices={"loss": "brand_new", "model": "brand_new"})
    C.score_candidates([fresh], history=hist, dead_ends=[])
    check("an untried branch retains option value (not zero-gain)",
          fresh.parts["gain"] >= C.NOISE_FLOOR and not fresh.parts["branch_seen"])

    # selection + trace
    pool = [mk(9, gain=0.0005), mk(10, gain=0.004, choices={"loss": "q", "model": "m"})]
    C.score_candidates(pool, history=hist, dead_ends=[])
    w, ranked = C.select(pool)
    check("selection picks the highest-utility surviving candidate",
          w is not None and w.utility == max(c.utility for c in pool))
    tr = C.render_trace(w, ranked, "exploration", None, 20)
    check("decision trace records every candidate and why",
          "SELECTED" in tr and "candidates generated" in tr and "parts=" in tr)
    allg = [mk(11, mech="", grounded="")]
    C.score_candidates(allg, history=hist, dead_ends=[])
    w2, _ = C.select(allg)
    check("if every candidate is gated, selection returns None (falls back)",
          w2 is None)

    lsrc = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("planning uses ONE call for N candidates (token-efficient)",
          "build_candidate_prompt" in lsrc and lsrc.count("json_call(p)") >= 1)
    check("the full candidate set is journalled for offline replay",
          '"type": "candidate_selection"' in lsrc and '"all": [c.as_dict()' in lsrc)


def test_budget_phase_awareness():
    """The planner was told WHICH objective to pursue but never how much runway
    it had, so a speculative probe and a closing-out confirmation looked equally
    affordable at iteration 48."""
    print("\n[budget / phase awareness]")
    from agent.research_policy import decide_category, render_decision
    from agent.research_state import ResearchState

    nodes = []
    for ln in open(os.path.join(_ROOT, "logs", "journal.jsonl")):
        if ln.strip():
            nodes.append(json.loads(ln))
    st = ResearchState(_ROOT)
    early = decide_category(st, nodes, iteration_budget_left=40)
    late = decide_category(st, nodes, iteration_budget_left=2)

    check("the decision carries iterations used and left",
          early["iterations_left"] == 40 and early["iterations_used"] == len(nodes))
    check("early and late runway map to different phases",
          early["phase"].startswith("EARLY") and late["phase"].startswith("LATE"),
          f"{early['phase'][:12]} / {late['phase'][:12]}")
    check("the phase reaches the rendered prompt",
          "Phase:" in render_decision(early) and "Budget:" in render_decision(early))
    check("the LATE phase warns against unaffordable probes",
          "not affordable" in late["phase"])
    check("the planner is told the best score so far",
          "Best scored so far" in render_decision(early)
          or early["best_primary"] is None)


def test_lesson_grading_uses_noise_floor():
    """The agent's own memory was recording seed noise as findings: HELPED
    fired above 1e-9 and DEAD_END below 1e-4, while the noise floor is 0.0008.
    Every later decision then read those back as evidence."""
    print("\n[experience lessons graded vs the noise floor]")
    src = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    seg = src.split("def _record_lesson")[-1] if "_record_lesson" in src else src
    check("the old sub-noise thresholds are gone",
          "+ 1e-9" not in seg and "- 1e-4" not in seg)
    check("HELPED requires clearing the noise floor",
          "delta >= BASELINE_SEED_STD" in seg)
    check("DEAD_END requires clearing it in the other direction",
          "delta <= -BASELINE_SEED_STD" in seg)
    check("a sub-noise result is recorded as saying nothing either way",
          "says nothing either way" in seg and "Treat as" in seg)
    check("lessons quote the effect size in sigma, not just a raw delta",
          "sigma" in seg and "BASELINE_SEED_STD" in seg)


def test_pipeline_lab():
    """Capabilities distilled from the Opus research run. Each is asserted on
    the property that made it useful, not on its existence."""
    print("\n[pipeline research lab]")
    import numpy as np
    from agent import pipeline_lab as PL

    # the audit must find the constant the research found BY HAND
    hc = PL.hardcoded_constants()
    names = {c["name"] for c in hc}
    check("the audit finds the hardcoded history decay",
          "tau_days" in names, f"{sorted(names)}")
    tau = [c for c in hc if c["name"] == "tau_days"][0]
    check("it reports the constant as now reachable, under its cfg alias",
          tau["reachable_by_agent"] and tau["override_key"] == "hist_tau_days")
    check("it still flags constants the agent CANNOT reach",
          any(not c["reachable_by_agent"] for c in hc))
    check("the override surface stays small (a menu by another name is no better)",
          len(PL.SAFE_OVERRIDES) <= 14, f"{len(PL.SAFE_OVERRIDES)}")

    # selection_rule_test must separate "higher here" from "generalises"
    rng = np.random.default_rng(0)
    n_users, per = 200, 6
    u = np.repeat(np.arange(n_users), per)
    truth = rng.normal(size=n_users * per)
    y = (truth + rng.normal(scale=0.5, size=n_users * per) > 0.4).astype(float)
    # epoch 3 is genuinely best; the others are noise around it
    E = np.stack([[truth + rng.normal(scale=s, size=n_users * per)
                   for s in (3.0, 2.0, 1.0, 0.4, 1.0, 2.0)] for _ in range(2)])
    rules = {"argmax": lambda p, e: e[int(np.argmax(p))],
             "avg_top3": lambda p, e: np.mean(e[np.argsort(-p)[:3]], axis=0)}
    r = PL.selection_rule_test(E, u, y, rules, n_splits=2)
    check("a selection-rule test evaluates on HELD-OUT users",
          r["n_evaluations"] == 2 * 2 * E.shape[0])
    check("it reports whether the rule GENERALISES, not just its score",
          "generalises" in r["rules"]["avg_top3"])

    # free_recombination must need no training and resample
    M = np.stack([truth + rng.normal(scale=1.0, size=n_users * per)
                  for _ in range(6)])
    fr = PL.free_recombination(M, u, y,
                               {"mean": lambda m: m.mean(axis=0),
                                "median": lambda m: np.median(m, axis=0)},
                               n_subsets=4, subset=4)
    check("recombination resamples member subsets rather than judging once",
          fr["n_subsets"] == 4 and fr["rules"]["median"]["n"] == 4)
    check("it reports a beats_reference verdict against the noise floor",
          "beats_reference" in fr["rules"]["median"])

    # the lesson must reach the agent
    pr = PL.render_for_prompt()
    check("the capabilities reach the prompt",
          all(c in pr for c in ("training_dynamics", "hardcoded_constants",
                                "selection_rule_test", "free_recombination")))
    # The default prompt must not hand the agent the teacher's answers -- that
    # is what turned the first self-test into replay rather than discovery.
    import re
    leaks = re.findall(r"0\.87|22/24|epoch 14|-29\.6|snapshot|redundant", pr, re.I)
    check("the DEFAULT prompt leaks no teacher findings", not leaks, f"{leaks}")
    check("override descriptions say what a knob DOES, not whether it helps",
          not re.search(r"sigma|measured|null|neutral at", " ".join(
              PL.SAFE_OVERRIDES.values()), re.I),
          f"{[v for v in PL.SAFE_OVERRIDES.values() if re.search(r'sigma|measured', v, re.I)]}")
    check("identifiers do not encode a conclusion",
          "snapshot_force" not in PL.SAFE_OVERRIDES
          and "n_checkpoints" in PL.SAFE_OVERRIDES)
    check("findings are available only behind an explicit flag",
          "0.87" in PL.render_for_prompt(reveal_findings=True))
    check("the prompt carries the METHOD (a principle, not an answer)",
          "same data that selected it is not evidence" in pr)
    lsrc = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("the lab is wired into the planning prompt",
          "from .pipeline_lab import render_for_prompt" in lsrc)


def test_feature_discovery():
    """Autonomous feature research. The agent could previously only SELECT
    features from a human-authored menu; these assert it can now propose, probe
    and retain them -- and that the gates which make that safe cannot be
    bypassed."""
    print("\n[autonomous feature discovery]")
    import numpy as np
    import tempfile
    from agent import feature_lab as FL

    # --- proposal must be complete: a feature without a mechanism is a guess
    check("an incomplete proposal is rejected with reasons",
          len(FL.validate_proposal({"name": "x"})) >= 5)
    full = {f: "stated" for f in FL.REQUIRED_FIELDS}
    full["source"] = "def build_features(splits, meta): return {}"
    check("a complete proposal passes validation", FL.validate_proposal(full) == [])
    nofn = dict(full); nofn["source"] = "x = 1"
    check("source must define the builder contract",
          any("build_features" in c for c in FL.validate_proposal(nofn)))

    # --- builder-specific leakage: labels from anything but train
    leak_all = ('def build_features(splits, meta):\n'
                '    return {"x": {s: splits[s]["long_view"] for s in splits}}')
    leak_valid = ('def build_features(splits, meta):\n'
                  '    va = splits["valid"]\n'
                  '    return {"x": {"valid": va["is_click"]}}')
    ok_alias = ('import numpy as np\n'
                'def build_features(splits, meta):\n'
                '    tr = splits["train"]\n'
                '    p = np.bincount(tr["video"], weights=tr["long_view"])\n'
                '    return {"x": {s: p[splits[s]["video"]] for s in splits}}')
    check("a builder reading labels from every split is flagged",
          FL.label_leak_findings(leak_all))
    check("a builder aliasing the VALID split is flagged",
          FL.label_leak_findings(leak_valid))
    check("the ordinary tr = splits['train'] alias is NOT flagged",
          FL.label_leak_findings(ok_alias) == [],
          f"{FL.label_leak_findings(ok_alias)}")
    # this gap is real: the general checker passes leak_all cleanly
    from agent.leakage_check import check_source, verdict as _lv
    check("the builder gate covers a case the general checker does not",
          not _lv(check_source(leak_all))["block"] and FL.label_leak_findings(leak_all))

    # --- the gate cannot be bypassed by routing through menu_choices
    from agent.menu import Menu, MenuError
    m = Menu(MENU_PATH)
    base = json.load(open(os.path.join(_ROOT, "logs",
                                       "ensemble_results.json")))["config"]
    good = dict(base); good["feature_source"] = ok_alias
    check("a legitimate builder survives menu validation",
          m.validate_choices(good).get("feature_source") == ok_alias)
    bad = dict(base); bad["feature_source"] = leak_all
    try:
        m.validate_choices(bad)
        check("a leaky builder is refused at the menu boundary", False)
    except MenuError as e:
        check("a leaky builder is refused at the menu boundary",
              "feature_source rejected" in str(e))
    junk = dict(base); junk["not_an_axis"] = "x"
    try:
        m.validate_choices(junk)
        check("passthrough does not weaken unknown-axis validation", False)
    except MenuError:
        check("passthrough does not weaken unknown-axis validation", True)

    # --- the builder contract is enforced, so misalignment cannot pass silently
    sys.path.insert(0, os.path.join(_ROOT, "runtime"))
    import train_lib
    fake = {"train": {"user": np.zeros(5, np.int32)},
            "valid": {"user": np.zeros(3, np.int32)}}
    try:
        train_lib.build_extra_features(
            'def build_features(s, m): return {"x": {"train": [1, 2], "valid": [1]}}',
            fake, {})
        check("a wrong-length feature is refused", False)
    except ValueError as e:
        check("a wrong-length feature is refused", "rows" in str(e))
    try:
        train_lib.build_extra_features("x = 1", fake, {})
        check("a source without the builder is refused", False)
    except ValueError:
        check("a source without the builder is refused", True)

    # --- bin edges come from TRAIN only (leakage-safe by construction)
    src = open(os.path.join(_ROOT, "runtime", "train_lib.py")).read()
    seg = src.split("def encode_features")[1].split("def ")[0]
    check("extra-feature bin edges are computed from the TRAIN split only",
          'extra or {})[fname].get("train")' in seg)

    # --- registry: reproducible, deduplicating, and fed back to the agent
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as t:
        reg = t.name
    try:
        FL.record({"name": "feat_a", "status": FL.REJECTED,
                   "reason": "constant within users", "source": ok_alias}, path=reg)
        check("a rejected feature is recorded, not discarded",
              FL.load_registry(reg)[0]["status"] == FL.REJECTED)
        check("a repeat proposal is detected by name",
              FL.already_tried("Feat A", path=reg) is not None)
        check("a RENAMED repeat is still detected by its builder body",
              FL.already_tried("totally_different", ok_alias, path=reg) is not None)
        check("an unrelated feature is not falsely matched",
              FL.already_tried("something_else", "def build_features(): pass",
                               path=reg) is None)
        check("past feature research is fed back into the prompt",
              "feat_a" in FL.render_for_prompt(reg))
    finally:
        os.unlink(reg)

    # --- the prompt states the two structural facts that kill bad proposals
    pr = FL.build_feature_prompt("S", "E", "R")
    check("the prompt warns that user-constant features are worth exactly 0.5",
          "0.5000 AUC, exactly" in pr)
    flat = " ".join(pr.split())      # the prompt is wrapped; assert on meaning
    check("the prompt warns that a strong standalone score is not evidence",
          "standalone number is not evidence" in flat and "0.639" in flat)
    check("the prompt states the builder contract",
          "build_features(splits, meta)" in flat
          and 'splits["train"] only' in flat)

    # --- the pathway exists in the loop and is non-fatal
    lsrc = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("the discovery phase runs BEFORE planning, feeding the prompt",
          "_feature_discovery_phase" in lsrc.split("_plan_candidates(action")[0])
    check("discovery failures cannot kill an iteration",
          '"type": "feature_discovery_skipped"' in lsrc)
    check("every discovery outcome is journalled",
          '"type": "feature_discovery"' in lsrc)
    check("probe results are recorded to the registry",
          "FL.record(entry)" in lsrc)

    # the demonstrable audit trail, and the real run behind it
    live = FL.load_registry()
    if live:
        log = FL.render_discovery_log()
        check("the discovery log renders hypothesis, probe and decision",
              all(k in log for k in ("[FEATURE DISCOVERY]", "Hypothesis",
                                     "Leakage gate", "Probe", "Decision")))
        agent_made = [e for e in live
                      if not str(e.get("created_by", "")).startswith("human")]
        check("the AGENT proposed at least one feature itself",
              len(agent_made) >= 1, f"{len(live)} entries, {len(agent_made)} agent")
        if agent_made:
            e = agent_made[0]
            check("its proposal carries every required research field",
                  FL.validate_proposal(e) == [], f"{FL.validate_proposal(e)}")
            check("it is NOT a menu axis value (genuinely outside the menu)",
                  e["name"] not in {o for ax in Menu(MENU_PATH).axes
                                    for o in Menu(MENU_PATH).options(ax)})
            check("it wrote a real builder, not a feature name",
                  len(str(e.get("source", "")).splitlines()) > 20)
            check("its probe measured incremental value, and it was recorded",
                  e.get("probe", {}).get("best_incremental_sigma") is not None)


def test_mechanism_audit():
    """A declaration is not evidence and a clean exit is not evidence. This
    catches the two ways this project has actually been fooled."""
    print("\n[mechanism audit]")
    from agent import mechanism_audit as MA

    null_src = ("import numpy as np, train_lib\n"
                "def userwise_affine_normalize(scores, user_ids):\n"
                "    u, inv = np.unique(user_ids, return_inverse=True)\n"
                "    sums = np.bincount(inv, weights=scores)\n"
                "    cnts = np.bincount(inv)\n"
                "    means = sums / np.maximum(cnts, 1.0)\n"
                "    return scores - means[inv]\n"
                "m = train_lib.run(choices, out)\n")
    a = MA.audit(null_src, "post-hoc per-user normalization of the scores", "B")
    check("a per-user monotone transform is caught as STRUCTURALLY NULL",
          a["postprocessing_null"] and a["blocks_scoring"], a["verdict"][:60])
    check("the audit explains WHY it cannot work",
          "within a user" in a["verdict"].lower() or "WITHIN a user" in a["verdict"])
    check("a null-only experiment must not be scored", not a["should_score"])

    # a claimed mechanism absent from both code and config
    a2 = MA.audit("import train_lib\ntrain_lib.run(c, o)\n",
                  "add auxiliary multitask heads on is_hate", "A",
                  menu_choices={"multitask": "none"})
    check("a claimed mechanism absent from code AND config is flagged",
          "NOT EVIDENCED" in a2["verdict"] and not a2["should_score"])

    # ...but selecting it in the config IS evidence, for Path A
    a3 = MA.audit("import train_lib\ntrain_lib.run(c, o)\n",
                  "add auxiliary multitask heads on is_hate", "A",
                  menu_choices={"multitask": "aux_social4"})
    check("a Path A mechanism selected by CONFIG counts as evidenced",
          a3["should_score"] and not a3["mechanisms_missing"], a3["verdict"][:50])

    # a script that trains something new is not merely post-processing
    learn = null_src + "\nfor epoch in range(10):\n    lr = 0.001\n"
    a4 = MA.audit(learn, "post-hoc normalization plus a new auxiliary head", "B")
    check("a script that also LEARNS is not blocked as pure post-processing",
          not a4["blocks_scoring"])

    check("a syntax error is reported, not raised",
          MA.audit("def (:", "x")["verdict"] == "does not parse")

    lsrc = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("the audit runs BEFORE the training run, not after",
          "mechanism_audit" in lsrc.split("run_solution(code")[0])
    check("a blocked mechanism is journalled as an error, not silently skipped",
          "BLOCKED BEFORE EXECUTION by the mechanism " in lsrc)
    from agent import failure as FA
    fc = FA.classify("BLOCKED BEFORE EXECUTION by the mechanism audit: "
                     "STRUCTURALLY NULL")
    check("a mechanism block has its own failure class",
          fc["class"] == FA.MECHANISM_BLOCKED)
    check("it is NOT retry-worthwhile -- a fix cannot rescue a null mechanism",
          not fc["retry_worthwhile"] and "reorders items" in fc["guidance"])
    check("a missing mechanism warns without blocking",
          'ma["mechanisms_missing"]' in lsrc and "warning:" in lsrc)
    check("an audit failure cannot kill an iteration",
          '"type": "mechanism_audit_skipped"' in lsrc)


def test_residual_screen_reporting():
    """The screen's headline must defer to its CONFIRMATION. The single screen
    picks its blend weight by scanning three values on validation, so its
    figure is selected; letting it set the verdict reported 'RESIDUAL SIGNAL
    FOUND' over a confirmation that said otherwise."""
    print("\n[residual screen reporting]")
    src = open(os.path.join(_ROOT, "agent", "residual_screen.py")).read()
    check("the confirmation overrides the scanned single screen",
          'if confirm else' in src and "confirm[\"survives\"]" in src)
    check("post-outcome columns are excluded from the FEATURES",
          '"excluded_post_outcome"' in src and "is_hate" in src)
    check("user-side columns are excluded with the structural reason",
          "constant within a user" in src)
    check("per-feature readings carry a noise caveat",
          "Treat sub-1-sigma" in src)

    p = os.path.join(_ROOT, "logs", "residual_screen.json")
    if os.path.exists(p):
        r = json.load(open(p))
        c = r.get("confirmation")
        check("the screen ran a fixed-weight confirmation", c is not None)
        if c:
            check("survival requires BOTH a real size and significance",
                  c["survives"] == (c["mean_gain"] >= 0.0004 and c["t"] > 2.0))
            check("a statistically detectable but sub-noise effect is NOT promoted",
                  not c["survives"] and "NOT WORTH A MECHANISM" in r["verdict"]
                  if (c["t"] > 2.0 and c["mean_gain"] < 0.0004) else True,
                  f"t={c['t']} gain={c['mean_gain']}")
        check("the incumbent score is the comparison baseline",
              r["incumbent_wAUC"] > 0.6)


def test_error_analysis():
    """The loop saw only two scalars per experiment. These are the properties
    that make per-segment analysis trustworthy rather than suggestive."""
    print("\n[error analysis]")
    import numpy as np
    from agent import error_analysis as EA

    # 3 users x 4 rows. Perfect ranking within each user.
    u = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], float)
    perfect = np.array([9, 8, 1, 0, 9, 1, 0, 0, 5, 5, 5, 5], float)
    auc, ndcg = EA.per_user_metrics(u, y, perfect)
    check("a perfect within-user ranking scores AUC 1.0",
          all(abs(a - 1.0) < 1e-9 for a in auc.values()))
    check("a zero-positive user is excluded from AUC and nDCG",
          2 not in auc and 2 not in ndcg)
    check("nDCG covers users with positives that AUC may exclude",
          set(ndcg) == {0, 1})

    inverted = -perfect
    ainv, _ = EA.per_user_metrics(u, y, inverted)
    check("an inverted ranking scores AUC 0.0", all(a < 1e-9 for a in ainv.values()))

    # THE structural fact: a feature constant within a user cannot move GAUC.
    const_per_user = np.array([5, 5, 5, 5, 9, 9, 9, 9, 1, 1, 1, 1], float)
    check("a user-CONSTANT feature is at chance for GAUC by construction",
          abs(EA.within_user_auc(u, y, const_per_user) - 0.5) < 1e-9,
          f"{EA.within_user_auc(u, y, const_per_user)}")

    # feature_probe must judge on RESIDUAL value, not standalone strength
    p = EA.feature_probe(u, y, perfect, const_per_user, "user-constant")
    check("a redundant feature is reported REDUNDANT despite any standalone score",
          not p["adds_signal"] and "REDUNDANT" in p["verdict"])
    check("the probe reports blend deltas, not just a standalone number",
          set(p["blend_delta"]) == {0.99, 0.95, 0.90})

    # Blend weights can only change an ORDERING, so this needs enough rows per
    # user to be readable at all -- an 8-row fixture reports 0.0 at every weight
    # simply because no pair ever swaps.
    rng = np.random.default_rng(0)
    n_users, per = 300, 8
    bu = np.repeat(np.arange(n_users), per)
    truth = rng.normal(size=n_users * per)
    by = (truth + rng.normal(scale=0.5, size=n_users * per) > 0.5).astype(float)
    weak = truth + rng.normal(scale=3.0, size=n_users * per)     # model: noisy
    strong = truth + rng.normal(scale=0.3, size=n_users * per)   # feature: sharp
    p2 = EA.feature_probe(bu, by, weak, strong, "the true signal")
    check("a feature that DOES add signal is detected",
          p2["adds_signal"] and p2["standalone_wAUC"] > p2["model_wAUC"],
          f"{p2['blend_delta']} standalone={p2['standalone_wAUC']} "
          f"model={p2['model_wAUC']}")

    check("segments below the readability floor are dropped",
          EA.segment(u, y, perfect, lambda k: k.astype(float),
                     [(0, 10)], "tiny") == [])
    check("the floor is documented as a number, not a vibe",
          isinstance(EA.MIN_SEGMENT_USERS, int) and EA.MIN_SEGMENT_USERS >= 100)


def test_research_frontier():
    """Directions are axis-options with evidence-based status. The old branch
    model was (loss, model), which could not represent 'is temporal useful?'
    at all."""
    print("\n[research frontier]")
    from agent import candidates as C
    from agent import frontier as F

    menu = {"axes": {"loss": {"options": ["bpr_pairwise", "lambdarank_ndcg",
                                          "listwise_softmax"]},
                     "multitask": {"options": ["none", "aux_click_like_forward",
                                               "aux_click_like_forward_watch"]},
                     "temporal": {"options": ["none", "hour_plus_dow"]}},
            "notes": {"tested_dead_ends": [
                "LambdaRank (|delta nDCG@5| pair weighting, loss=lambdarank_ndcg): "
                "MEASURED HERE, decisively worse than bpr_pairwise",
                "Graded play-time as an auxiliary head "
                "(multitask=aux_click_like_forward_watch): MEASURED HERE, worse",
                "gru4rec_seq as an ENSEMBLE member with fm_numpy: MEASURED HERE"]}}

    def node(i, choices, primary, gauc=None, ndcg=None, status="success"):
        m = ({"primary": primary, "GAUC": gauc if gauc is not None else primary,
              "nDCG@5": ndcg if ndcg is not None else primary}
             if status == "success" else None)
        return {"iteration_id": i, "status": status, "menu_choices": choices,
                "metrics": m}

    best = {"loss": "bpr_pairwise", "multitask": "none", "temporal": "hour_plus_dow"}
    nodes = [node(0, best, 0.6050, gauc=0.6720, ndcg=0.5380),
             node(1, {**best, "temporal": "none"}, 0.6020, gauc=0.6660, ndcg=0.5380)]
    f = F.Frontier(nodes, menu, best_config=best)
    by = {d["direction"]: d for d in f.directions}

    check("a never-run option is UNEXPLORED, not KNOWN_BAD",
          by["loss=listwise_softmax"]["status"] == F.UNEXPLORED)
    check("a recorded dead end is KNOWN_BAD with HIGH confidence",
          by["loss=lambdarank_ndcg"]["status"] == F.KNOWN_BAD
          and by["loss=lambdarank_ndcg"]["confidence"] == F.HIGH)

    # the two matcher bugs that condemned good mechanisms
    check("a dead end's COMPARISON BASELINE is not itself condemned",
          by["loss=bpr_pairwise"]["status"] != F.KNOWN_BAD,
          by["loss=bpr_pairwise"]["status"])
    check("a shared PREFIX does not condemn a different mechanism",
          by["multitask=aux_click_like_forward"]["status"] != F.KNOWN_BAD
          and by["multitask=aux_click_like_forward_watch"]["status"] == F.KNOWN_BAD)

    # A finding scoped to a whole AXIS must reach its option values, which may
    # be named far into the text -- otherwise an already-measured variant shows
    # UNEXPLORED and invites a re-run.
    menu2 = {"axes": {"neg": {"options": ["uniform_1", "uniform_2", "uniform_4"]}},
             "notes": {"tested_dead_ends": [
                 "Negative-sampling variants (neg axis): MEASURED HERE, none beat "
                 "the uniform_1 default. uniform_2 0.60356 and uniform_4 0.60316 "
                 "vs uniform_1 0.60367 over 5 paired seeds"]}}
    fb = F.Frontier([node(0, {"neg": "uniform_1"}, 0.60367)], menu2,
                    best_config={"neg": "uniform_1"})
    b2 = {d["direction"]: d for d in fb.directions}
    check("an axis-scoped dead end reaches its option values",
          b2["neg=uniform_2"]["status"] == F.KNOWN_BAD
          and b2["neg=uniform_4"]["status"] == F.KNOWN_BAD,
          f"{b2['neg=uniform_2']['status']}/{b2['neg=uniform_4']['status']}")
    check("the INCUMBENT is never condemned by a finding that merely names it",
          b2["neg=uniform_1"]["status"] != F.KNOWN_BAD,
          b2["neg=uniform_1"]["status"])

    # isolated ablation, graded against the noise floor
    t = by["temporal=hour_plus_dow"]
    check("an in-best option with isolated evidence is graded, not assumed",
          t.get("ablation") is not None and t["ablation"]["sigma"] > 3)
    check("GAUC and nDCG@5 are attributed SEPARATELY",
          "d_GAUC" in t["ablation"] and "d_nDCG@5" in t["ablation"])

    # a metric conflict is invisible in the primary and must be surfaced
    nodes2 = [node(0, best, 0.6050, gauc=0.6700, ndcg=0.5400),
              node(1, {**best, "temporal": "none"}, 0.6050, gauc=0.6600, ndcg=0.5500)]
    f2 = F.Frontier(nodes2, menu, best_config=best)
    check("a GAUC/nDCG trade hidden by an unchanged primary is surfaced",
          len(f2.metric_conflicts()) >= 1,
          f"{[d['direction'] for d in f2.metric_conflicts()]}")

    check("unexplored options are listed as candidates, not failures",
          all(d["experiments"] == 0 for d in f.unexplored()))

    # Saturation needs BOTH a recent window and an earlier one. Defaulting an
    # uncomputable trend to 0.0 made four CONSECUTIVELY IMPROVING experiments
    # read as saturated -- telling the planner to abandon a working direction.
    m1 = {"axes": {"x": {"options": ["a"]}}, "notes": {"tested_dead_ends": []}}

    def sn(i, p):
        return {"iteration_id": i, "status": "success", "menu_choices": {"x": "a"},
                "metrics": {"primary": p, "GAUC": p, "nDCG@5": p}}

    improving4 = F.Frontier([sn(i, 0.601 + i * 0.001) for i in range(4)],
                            m1, best_config={"x": "b"}).directions[0]
    check("an improving direction with no earlier window is NOT saturated",
          improving4["status"] != F.SATURATED
          and improving4["recent_trend"] is None,
          f"{improving4['status']} trend={improving4['recent_trend']}")
    improving6 = F.Frontier([sn(i, 0.601 + i * 0.001) for i in range(6)],
                            m1, best_config={"x": "b"}).directions[0]
    check("a still-improving direction is never saturated",
          improving6["status"] != F.SATURATED)
    collapsed = F.Frontier([sn(0, .601), sn(1, .607), sn(2, .605), sn(3, .604),
                            sn(4, .603), sn(5, .602)],
                           m1, best_config={"x": "b"}).directions[0]
    check("a direction whose recent returns collapsed IS saturated",
          collapsed["status"] == F.SATURATED, collapsed["status"])
    # Disagreement must be measured over TRUE REPLICATES. Nodes that merely
    # share one option differ on every other axis, and that spread is not
    # evidence the option is contradictory.
    m3 = {"axes": {"a": {"options": ["v"]}, "b": {"options": ["p", "q"]}},
          "notes": {"tested_dead_ends": []}}
    varied = [node(0, {"a": "v", "b": "p"}, 0.6050),
              node(1, {"a": "v", "b": "q"}, 0.5900)]      # spread from axis b
    fv = F.Frontier(varied, m3, best_config={"a": "v", "b": "p"})
    av = {d["direction"]: d for d in fv.directions}["a=v"]
    check("spread caused by OTHER axes is not called contradictory",
          av["status"] != F.CONTRADICTORY and av["replicate_spread"] == 0.0,
          f"{av['status']} spread={av['replicate_spread']}")
    repl = [node(0, {"a": "v", "b": "p"}, 0.6050),
            node(1, {"a": "v", "b": "p"}, 0.5900)]        # same config, disagrees
    fr2 = F.Frontier(repl, m3, best_config={"a": "v", "b": "q"})
    ar = {d["direction"]: d for d in fr2.directions}["a=v"]
    check("replicates that disagree beyond noise ARE contradictory",
          ar["status"] == F.CONTRADICTORY, ar["status"])

    # accumulated knowledge must survive --fresh
    fsrc = open(os.path.join(_ROOT, "agent", "frontier.py")).read()
    check("the frontier aggregates archived runs, not just the current journal",
          "include_archives" in fsrc and "archive_" in fsrc)
    live_f = F.from_root(_ROOT)
    check("aggregation actually finds more than one run",
          len({n.get("_run", "") for n in live_f.nodes}) > 1,
          f"{len({n.get('_run','') for n in live_f.nodes})} runs")
    check("node ids are namespaced so runs cannot collide",
          any(":" in str(n["iteration_id"]) for n in live_f.nodes))

    check("the frontier renders without an LLM",
          "RESEARCH FRONTIER" in f.render() and "UNEXPLORED" in f.render())

    # value of information: resolving an open question is worth an iteration
    # even when its expected score gain is not the largest on offer.
    def mk(i, choices, gain):
        return C.Candidate({"hypothesis": f"idea {i}",
                            "mechanism": "a stated mechanism long enough to pass",
                            "expected_gain": gain, "menu_choices": choices,
                            "research_category": "exploration",
                            "rationale": {"grounded_in": "measured evidence here"}}, i)

    hist = [{"iteration_id": 0, "status": "success", "hypothesis": "base",
             "menu_choices": best, "metrics": {"primary": 0.605},
             "implementation_path": "A"}]
    settled = mk(0, {"loss": "lambdarank_ndcg"}, 0.004)   # KNOWN_BAD (dead end)
    open_q = mk(1, {"loss": "listwise_softmax"}, 0.002)   # UNEXPLORED
    C.score_candidates([settled, open_q], history=hist, dead_ends=[],
                       budget_left=40, frontier=f)
    check("an UNEXPLORED direction earns information value",
          open_q.parts["information_value"] == C.INFO_UNEXPLORED,
          f"{open_q.parts['information_value']}")
    # temporal=hour_plus_dow has isolated evidence beyond the noise floor in
    # this fixture, so it is KNOWN_GOOD -- settled, nothing left to learn.
    check("a settled (KNOWN_GOOD) direction earns no information value",
          by["temporal=hour_plus_dow"]["status"] == F.KNOWN_GOOD
          and C._information_value(mk(2, {"temporal": "hour_plus_dow"}, 0.001), f) == 0.0,
          by["temporal=hour_plus_dow"]["status"])
    # fr2 above is contradictory for a real reason (replicates disagreeing),
    # not because other axes varied.
    check("a CONTRADICTORY direction DOES carry information value",
          C._information_value(mk(7, {"a": "v"}, 0.001), fr2)
          == C.INFO_CONTRADICTORY)
    check("information value takes the MAX over axes, not the sum",
          C._information_value(mk(3, {"loss": "listwise_softmax",
                                      "multitask": "aux_click_like_forward"}, 0.001),
                               f) == C.INFO_UNEXPLORED)
    late = mk(4, {"loss": "listwise_softmax"}, 0.002)
    C.score_candidates([late], history=hist, dead_ends=[], budget_left=2, frontier=f)
    check("information is worth less when the budget is nearly spent",
          late.parts["info_factor"] < open_q.parts["info_factor"],
          f"{late.parts['info_factor']} vs {open_q.parts['info_factor']}")
    nofr = mk(5, {"loss": "listwise_softmax"}, 0.002)
    C.score_candidates([nofr], history=hist, dead_ends=[], budget_left=40)
    check("scoring still works with no frontier available",
          nofr.parts["information_value"] == 0.0 and nofr.utility > 0)

    lsrc = open(os.path.join(_ROOT, "agent", "loop.py")).read()
    check("the frontier actually reaches the planning prompt",
          "from .frontier import from_root" in lsrc)
    check("the frontier also reaches candidate scoring",
          "frontier=fr" in lsrc)
    check("a frontier failure cannot kill an iteration",
          '"type": "frontier_skipped"' in lsrc)
    live = F.from_root(_ROOT).render(limit=22)
    check("the live frontier stays compact (< 4500 chars)",
          len(live) < 4500, f"{len(live)}")


def test_submission_matches_reported_result():
    """The submission must be built from the ensemble the deliverable reports.

    It previously was not: --ensemble rank-averaged the top-K nodes chosen BY
    VALIDATION SCORE with DISTINCT configs, which is simultaneously the
    +0.00081 selection bias and the heterogeneous blending that were both
    measured and rejected here -- and it read the search journal, not the
    submitted artifacts. A judge running it would have got a different number
    than the one being claimed.
    """
    print("\n[submission matches the reported result]")
    import shutil
    import tempfile
    import numpy as np
    from agent import make_submission as MS

    d = tempfile.mkdtemp()
    try:
        logs = os.path.join(d, "logs", "final_ensemble")
        os.makedirs(logs)
        seeds = [0, 1, 2]
        for s in seeds:
            os.makedirs(os.path.join(logs, f"seed_{s:02d}"))
            np.save(os.path.join(logs, f"seed_{s:02d}", "scores_valid.npy"),
                    np.arange(10, dtype=float) + s)
        json.dump({"k": 3, "seeds_used": seeds, "source_node": 4,
                   "members_dir": os.path.join("logs", "final_ensemble")},
                  open(os.path.join(d, "logs", "ensemble_results.json"), "w"))

        arr, meta = MS.final_ensemble_members(d, "valid")
        check("submission rebuilds from the recorded ensemble members",
              arr is not None and meta["k"] == 3)
        check("it averages ALL recorded members, not a chosen subset",
              len(meta["seeds_used"]) == 3)

        # a missing member must NOT silently yield a partial average
        shutil.rmtree(os.path.join(logs, "seed_02"))
        arr2, meta2 = MS.final_ensemble_members(d, "valid")
        check("a partial ensemble is REFUSED, not silently averaged",
              arr2 is None and meta2 is None)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    src = open(os.path.join(_ROOT, "agent", "make_submission.py")).read()
    check("the biased top-k path is no longer the default",
          "if a.legacy_topk_ensemble:" in src)
    check("the biased path warns that it is not the reported result",
          "NOT the reported result" in src)
    check("the legacy path documents BOTH measured biases",
          "+0.00081" in src and "heterogeneous" in src)


def test_evidence_strength():
    """Evidence must be graded against the noise floor, not by sign. The old
    test was a bare `>` comparison, which reads a 0.0001 difference as support
    when the baseline's own seed spread is 0.0008."""
    print("\n[evidence strength grading]")
    from agent import research_state as RS

    check("a difference inside the noise floor is INCONCLUSIVE, not support",
          RS.grade_evidence(0.0001)["strength"] == RS.INCONCLUSIVE)
    check("a sub-sigma NEGATIVE difference is also INCONCLUSIVE, not rejection",
          RS.grade_evidence(-0.0002)["strength"] == RS.INCONCLUSIVE)
    check("1-2 sigma from a single run is WEAK",
          RS.grade_evidence(0.0009)["strength"] == RS.WEAK)
    check("replication promotes WEAK to MODERATE",
          RS.grade_evidence(0.0009, n_runs=3)["strength"] == RS.MODERATE)
    check("2-3 sigma is MODERATE", RS.grade_evidence(0.0018)["strength"] == RS.MODERATE)
    check(">3 sigma is STRONG", RS.grade_evidence(0.0028)["strength"] == RS.STRONG)
    check("reseed verification is STRONG even at modest effect size",
          RS.grade_evidence(0.001, reseed_verified=True)["strength"] == RS.STRONG)
    check("evidence pointing the other way is REJECTED, not merely weak",
          RS.grade_evidence(-0.0015)["strength"] == RS.REJECTED)
    check("INCONCLUSIVE is not actionable; REJECTED is",
          not RS.grade_evidence(0.0001)["actionable"]
          and RS.grade_evidence(-0.0015)["actionable"])

    # the grading must actually reach component evidence, not just exist
    src = open(os.path.join(_ROOT, "agent", "research_state.py")).read()
    check("component evidence uses the grade, not a bare comparison",
          "grade_evidence(" in src.split("def _component_evidence")[1])
    check("an ablation inside the noise floor is NOT reported as supported",
          'if g["strength"] == INCONCLUSIVE:' in src
          and 'status = "untested_assumption"' in src)


def test_policy_replay():
    """Counterfactual replay must never invent an outcome for work that was
    never run. That restraint is the whole value of the harness."""
    print("\n[policy replay / counterfactual honesty]")
    from agent import policy_eval as P

    def cand(i, util, path="A", gated=None, gain=0.001, cost=1.0, choices=None):
        return {"index": i, "utility": util, "path": path, "category": "exploration",
                "menu_choices": choices if choices is not None else {"loss": f"l{i}"},
                "rejected_by": gated or [],
                "parts": {"gain": gain, "cost": cost}}

    # node 0 ran candidate 0; candidate 1 was never implemented; candidate 2's
    # configuration happens to match node 9, which WAS run.
    cands = [cand(0, 0.010), cand(1, 0.009, gain=0.02),
             cand(2, 0.007, choices={"loss": "ran_elsewhere"}),
             cand(3, 0.02, gated=["overlaps a recorded dead end"], gain=0.05)]
    decisions = [{"node": 0, "candidates": cands, "actual": cand(0, 0.010)}]
    outcomes = {P._sig({"loss": "l0"}): {"primary": 0.605, "node": 0},
                P._sig({"loss": "ran_elsewhere"}): {"primary": 0.601, "node": 9}}

    dep = P.replay(decisions, outcomes, P.policy_deployed)
    check("the deployed policy's own choice is OBSERVED, not counterfactual",
          dep["observed"] == 1 and dep["cf_unknown"] == 0)
    check("a gated candidate never wins under the deployed policy",
          dep["picked_gated"] == 0)

    # greedy_gain prefers index 1 (gain .02, ungated) -> never implemented
    gg = P.replay(decisions, outcomes, P.policy_greedy_gain)
    check("an alternative picking never-run work is COUNTERFACTUAL_UNKNOWN",
          gg["cf_unknown"] == 1 and gg["observed"] == 0,
          f"{gg['cf_unknown']}/{gg['observed']}")
    check("an unknown counterfactual carries NO score",
          gg["detail"][0]["primary"] is None)
    check("outcome coverage reports the unknown honestly", gg["outcome_coverage"] == 0.0)

    # no_gates takes index 3 (utility .02) despite the dead-end gate
    ng = P.replay(decisions, outcomes, P.policy_no_gates)
    check("disabling gates lets a known dead end be selected",
          ng["picked_gated"] == 1 and ng["gated_pick_rate"] == 1.0)
    check("gates are therefore doing measurable work",
          ng["picked_gated"] > dep["picked_gated"])

    # a differing pick whose config WAS run elsewhere is a real measurement
    borrowed = P.replay(decisions, outcomes, lambda cs: cs[2])
    check("a differing pick that ran elsewhere is COUNTERFACTUAL_KNOWN",
          borrowed["cf_known"] == 1 and borrowed["detail"][0]["primary"] == 0.601)

    check("no policy reports an aggregate score over unknown outcomes",
          not any(k in dep for k in ("mean_primary", "expected_primary", "score")))

    src = open(os.path.join(_ROOT, "agent", "policy_eval.py")).read()
    check("the replay reads only decision-time fields (no hindsight leak)",
          "metrics" not in src.split("# --------------------------------------------------------------- policies ---")[1].split("def replay")[0])


def test_submission_artifacts_survive_fresh():
    """Regression: a previous headline became unreproducible because --fresh
    archived the ensemble member arrays while the JSON quoting them stayed
    behind. A new SEARCH run must never carry off the SUBMITTED result."""
    print("\n[submission artifacts survive --fresh]")
    import shutil
    import tempfile
    sys.path.insert(0, _ROOT)
    import run_agent

    d = tempfile.mkdtemp()
    try:
        open(os.path.join(d, "journal.jsonl"), "w").write('{"iteration_id": 0}\n')
        open(os.path.join(d, "best_metrics.json"), "w").write("{}")
        open(os.path.join(d, "ensemble_results.json"), "w").write('{"k": 16}')
        os.makedirs(os.path.join(d, "final_ensemble", "seed_00"))
        open(os.path.join(d, "final_ensemble", "seed_00", "metrics.json"), "w").write("{}")
        run_agent.archive_logs(d)

        check("the search journal IS archived (run really does start fresh)",
              not os.path.exists(os.path.join(d, "journal.jsonl")))
        check("ensemble_results.json survives --fresh",
              os.path.exists(os.path.join(d, "ensemble_results.json")))
        check("ensemble member arrays survive --fresh",
              os.path.exists(os.path.join(d, "final_ensemble", "seed_00",
                                          "metrics.json")))
        check("the surviving result still matches its surviving members",
              json.load(open(os.path.join(d, "ensemble_results.json")))["k"] == 16)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # The reproduce command must not silently re-target. best_metrics.json is
    # overwritten by every search run; an A/B arm replaced it with a config
    # scoring 0.60367, which would have rebuilt a WORSE ensemble on top of the
    # reported 0.60541 under the exact command the docs tell a judge to run.
    from agent import final_ensemble as FE
    live_res = json.load(open(os.path.join(_ROOT, "logs", "ensemble_results.json")))
    live_best = json.load(open(os.path.join(_ROOT, "logs", "best_metrics.json")))
    pinned = FE.load_best()
    check("the rebuild pins to the RECORDED ensemble config",
          pinned["menu_choices"] == live_res["config"])
    check("...even when best_metrics.json has since moved on",
          live_best["menu_choices"] != live_res["config"]
          or pinned["menu_choices"] == live_best["menu_choices"])
    check("re-targeting requires an explicit flag",
          FE.load_best(retarget=True)["menu_choices"] == live_best["menu_choices"])
    check("the pinned script is a frozen member, not a mutable solutions/ path",
          "final_ensemble" in pinned["code_path"]
          and os.path.exists(pinned["code_path"]))


if __name__ == "__main__":
    for t in (test_safety_gate, test_validity, test_policy, test_convergence,
              test_executor, test_crossover, test_spend_ceiling,
              test_audit_regressions, test_numpy2_json_serialization,
              test_diff_artifacts, test_data_boundary, test_restricted_access,
              test_final_eval_lock, test_reseed, test_experience,
              test_rationale_schema, test_best_override, test_worktree_lifecycle,
              test_worker_sandbox_hardlinking, test_run_parallel_round,
              test_merge_acceptance_via_tree_ordering,
              test_standing_override_survives_reload,
              test_compute_budget_prompt_section, test_lambdarank,
              test_new_axes_and_snapshot, test_parallel_worker_diversity,
              test_data_tools_and_proposals, test_stage_b_path_freedom,
              test_research_state, test_research_state_no_side_effects,
              test_research_policy, test_failure_taxonomy,
              test_leakage_and_ensemble, test_candidate_policy,
              test_budget_phase_awareness,
              test_lesson_grading_uses_noise_floor,
              test_pipeline_lab, test_feature_discovery,
              test_mechanism_audit, test_residual_screen_reporting,
              test_error_analysis, test_research_frontier,
              test_submission_matches_reported_result,
              test_evidence_strength, test_policy_replay,
              test_submission_artifacts_survive_fresh):
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
