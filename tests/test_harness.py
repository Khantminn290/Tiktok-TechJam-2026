"""Harness self-tests — no LLM calls, no training. Run: python3 tests/test_harness.py

Covers the pieces that must be right before an autonomous run is trusted:
the safety gate, cross-axis validity checks, the search policy's branching
decisions, the convergence rule, and the executor's contract enforcement.
"""
import hashlib
import json
import os
import sys
import tempfile
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from agent.contracts import ExperimentTree, Node, error_headline  # noqa: E402
from agent.executor import (_hash_protected, _protected_paths, _runtime_fingerprint,
                            run_solution)  # noqa: E402
from agent.loop import EPSILON, N_CONVERGE, AgentLoop  # noqa: E402
from agent.make_submission import verified_ensemble_scores  # noqa: E402
from agent.menu import Menu, MenuError  # noqa: E402
from agent.policy import decide_action  # noqa: E402
from agent.verified_ensemble import (MENU_CHOICES as ENSEMBLE_CHOICES,
                                     _rank_normalize,
                                     _recover_previous)  # noqa: E402

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
    from agent.executor import _env
    old_key = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "unit-test-secret"
    try:
        check("child environment strips provider credentials",
              "OPENAI_API_KEY" not in _env())
    finally:
        if old_key is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = old_key
    trusted = {"GAUC": 0.61, "nDCG@5": 0.53, "primary": 0.57}
    fake_evaluator = lambda scores: dict(trusted)
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
        for i, (label, code) in enumerate(cases.items()):
            r = run_solution(code, os.path.join(td, f"s{i}.py"), {},
                             os.path.join(td, f"r{i}"), timeout_s=60,
                             validation_evaluator=fake_evaluator)
            check(f"{label} -> error, not a crash",
                  (not r.ok) and bool(r.error_trace), error_headline(r.error_trace, 60))
            if label == "runtime exception":
                check("nonzero exit records unchanged solution hash",
                      bool(r.verification.get("solution_sha256"))
                      and r.verification.get("solution_sha256")
                      == r.verification.get("solution_after_sha256"))

        mutation_cases = {
            "rewrite": "open(__file__, 'w').write('changed')\n",
            "delete": "import os\nos.remove(__file__)\n",
            "rename": "import os\nos.rename(__file__, __file__ + '.moved')\n",
        }
        for action, code in mutation_cases.items():
            code_path = os.path.join(td, f"self_{action}.py")
            try:
                result = run_solution(
                    code, code_path, {}, os.path.join(td, f"self_{action}"),
                    timeout_s=60, validation_evaluator=fake_evaluator)
                with open(code_path, encoding="utf-8") as handle:
                    unchanged = handle.read() == code
                check(f"guard blocks generated-code self-{action}",
                      (not result.ok)
                      and "generated-code guard blocked" in
                      (result.error_trace or "")
                      and unchanged)
                check(f"self-{action} failure records unchanged solution hash",
                      bool(result.verification.get("solution_sha256"))
                      and result.verification.get("solution_sha256")
                      == result.verification.get("solution_after_sha256"))
            except Exception as error:
                check(f"guard blocks generated-code self-{action}", False,
                      f"executor crashed: {error}")
        r = run_solution("import time; time.sleep(30)\n", os.path.join(td, "s.py"),
                         {}, os.path.join(td, "timeout"), timeout_s=3,
                         validation_evaluator=fake_evaluator)
        check("timeout -> killed and reported",
              (not r.ok) and "TIMEOUT" in (r.error_trace or ""))

        def output_script(reported):
            return (
                "import argparse,json,os,numpy as np\n"
                "p=argparse.ArgumentParser();p.add_argument('--menu-choices');"
                "p.add_argument('--output-dir');p.add_argument('--seed');a=p.parse_args()\n"
                "os.makedirs(a.output_dir,exist_ok=True)\n"
                f"json.dump({reported!r},open(os.path.join(a.output_dir,'metrics.json'),'w'))\n"
                "np.save(os.path.join(a.output_dir,'scores_valid.npy'),np.zeros(124909))\n"
                "np.save(os.path.join(a.output_dir,'scores_test.npy'),np.zeros(170588))\n")

        good_dir = os.path.join(td, "good")
        r = run_solution(output_script(trusted), os.path.join(td, "good.py"), {},
                         good_dir, timeout_s=60,
                         validation_evaluator=fake_evaluator)
        check("parent-authoritative matching metrics succeed",
              r.ok and r.metrics == trusted and r.metric_audit["matched"])
        check("successful execution writes a verification manifest",
              os.path.exists(os.path.join(good_dir, "verification.json"))
              and bool(r.verification.get("solution_sha256")))
        check("successful execution records unchanged solution hash",
              r.verification.get("solution_sha256")
              == r.verification.get("solution_after_sha256"))
        check("successful execution records the generated-code guard",
              r.verification.get("generated_code_guard", {}).get("mode")
              == "python_audit_hook")

        blocked_code = (
            "import os\n"
            "p=os.path.join(os.environ['KUAIRAND_DATA'],"
            "'log_standard_4_22_to_5_08_pure.csv')\n"
            "open(p,'rb').read(1)\n")
        blocked = run_solution(
            blocked_code, os.path.join(td, "blocked.py"), {},
            os.path.join(td, "blocked"), timeout_s=60,
            validation_evaluator=fake_evaluator)
        check("generated code cannot directly read raw outcome files",
              not blocked.ok and "generated-code guard blocked file access"
              in (blocked.error_trace or ""))

        forged_dir = os.path.join(td, "forged")
        r = run_solution(output_script({"GAUC": .999, "nDCG@5": .999,
                                        "primary": .999}),
                         os.path.join(td, "forged.py"), {}, forged_dir,
                         timeout_s=60, validation_evaluator=fake_evaluator)
        check("forged child metrics cannot control model selection",
              r.ok and r.metrics == trusted and not r.metric_audit["matched"])
        with open(os.path.join(forged_dir, "metrics.json")) as fh:
            replaced = json.load(fh)
        with open(os.path.join(forged_dir, "metrics_reported.json")) as fh:
            preserved = json.load(fh)
        check("trusted metrics replace the child report", replaced == trusted)
        check("the mismatched child report remains auditable",
              preserved["primary"] == .999)


def test_hidden_test_boundary():
    print("\n[hidden-test data boundary]")
    from runtime.train_lib import (_CACHE_SCHEMA_FILE, _CACHE_SCHEMA_VERSION,
                                   _TARGET_COLUMNS, build_cache, load_cache,
                                   load_validation_targets)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "meta.json"), "w") as fh:
            json.dump({"field_dims": {}}, fh)
        with open(os.path.join(td, "vocabs.json"), "w") as fh:
            json.dump({}, fh)
        with open(os.path.join(td, _CACHE_SCHEMA_FILE), "w") as fh:
            json.dump({"version": _CACHE_SCHEMA_VERSION}, fh)
        base = {
            "user": np.array([0, 1], dtype=np.int32),
            "video": np.array([0, 1], dtype=np.int32),
            "user_raw": np.array(["u0", "u1"], dtype=object),
            "video_raw": np.array(["v0", "v1"], dtype=object),
        }
        outcomes = {key: np.array([0.0, 1.0], dtype=np.float32)
                    for key in _TARGET_COLUMNS}
        for split in ("train", "valid", "test"):
            np.savez(os.path.join(td, f"{split}.npz"), **base, **outcomes)
        splits, _ = load_cache(td)
        check("train and validation retain their outcomes",
              all(key in splits["train"] and key in splits["valid"]
                  for key in _TARGET_COLUMNS))
        check("normal API strips every hidden-test outcome from old caches",
              all(key not in splits["test"] for key in _TARGET_COLUMNS))
        with np.load(os.path.join(td, "test.npz"), allow_pickle=True) as z:
            persisted_keys = set(z.files)
        check("old cache is physically migrated to a feature-only test file",
              not (persisted_keys & set(_TARGET_COLUMNS)))
        users, labels = load_validation_targets(td)
        check("trusted parent can load copied validation targets",
              list(users) == ["u0", "u1"]
              and np.array_equal(labels, outcomes["long_view"]))

    # A fresh cache must not even access/convert target cells on test dates.
    # The deliberately invalid test outcome strings would crash the old path.
    with tempfile.TemporaryDirectory() as td:
        header = ("user_id,video_id,tab,duration_ms,hourmin,date,time_ms,"
                  "long_view,is_click,is_like,is_forward,play_time_ms\n")
        train_row = "u0,v0,1,1000,1200,20220408,1,1,1,0,0,900\n"
        valid_row = "u1,v1,1,2000,1300,20220422,2,0,1,1,0,400\n"
        test_row = ("u2,v2,1,3000,1400,20220429,3,DO_NOT_READ,"
                    "DO_NOT_READ,DO_NOT_READ,DO_NOT_READ,DO_NOT_READ\n")
        with open(os.path.join(td, "video_features_basic_pure.csv"), "w") as fh:
            fh.write("video_id,author_id\nv0,a0\nv1,a1\nv2,a2\n")
        with open(os.path.join(
                td, "log_standard_4_08_to_4_21_pure.csv"), "w") as fh:
            fh.write(header + train_row)
        with open(os.path.join(
                td, "log_standard_4_22_to_5_08_pure.csv"), "w") as fh:
            fh.write(header + valid_row + test_row)
        cache_dir = os.path.join(td, "cache")
        try:
            build_cache(data_dir=td, cache_dir=cache_dir)
            fresh, _ = load_cache(cache_dir)
            built = True
        except (TypeError, ValueError) as error:
            built = False
            fresh = {}
            fresh_error = str(error)
        check("fresh cache never converts hidden-test outcome cells", built,
              "" if built else fresh_error)
        if built:
            check("fresh test cache contains features only",
                  not (set(fresh["test"]) & set(_TARGET_COLUMNS)))
            check("fresh split row order remains exact",
                  [str(v) for v in fresh["train"]["user_raw"]] == ["u0"]
                  and [str(v) for v in fresh["valid"]["user_raw"]] == ["u1"]
                  and [str(v) for v in fresh["test"]["user_raw"]] == ["u2"])
            check("fresh cache preserves raw video identity",
                  [str(v) for v in fresh["train"]["video_raw"]] == ["v0"]
                  and [str(v) for v in fresh["valid"]["video_raw"]] == ["v1"]
                  and [str(v) for v in fresh["test"]["video_raw"]] == ["v2"])
            check("fresh train/valid targets remain aligned",
                  fresh["train"]["long_view"].tolist() == [1.0]
                  and fresh["valid"]["long_view"].tolist() == [0.0]
                  and fresh["valid"]["play_time_ms"].tolist() == [400.0])


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
        check("node records are colocated under nodes/node_NNN",
              os.path.exists(os.path.join(td, "nodes", "node_001", "record.json")))
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


def test_research_primitives():
    """Fast invariants for the two new, leakage-aware research directions."""
    print("\n[research primitives]")
    import copy
    import numpy as np
    from runtime.train_lib import (History, MultiBehaviorHistory, RankFM,
                                   batch_repeat_fatigue_scores,
                                   oof_recency_bayesian_prior_scores,
                                   rank_normalize)

    tr = {
        "user": np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        "video": np.array([0, 1, 2, 0, 2, 3], dtype=np.int32),
        "author": np.array([0, 1, 1, 0, 1, 2], dtype=np.int32),
        "date": np.array([1, 1, 2, 1, 2, 2], dtype=np.int32),
        "time_ms": np.array([1, 2, 3, 1, 2, 3], dtype=np.int64) * 86400_000,
        "long_view": np.array([1, 0, 1, 0, 1, 0], dtype=np.float32),
        "is_click": np.array([1, 1, 1, 0, 1, 1], dtype=np.float32),
        "is_like": np.array([0, 0, 1, 0, 0, 0], dtype=np.float32),
        "is_forward": np.zeros(6, dtype=np.float32),
    }
    query = {
        "user": np.array([0, 1], dtype=np.int32),
        "video": np.array([1, 3], dtype=np.int32),
        "author": np.array([1, 2], dtype=np.int32),
    }
    splits = {"train": tr, "valid": query, "test": query}
    rng = np.random.default_rng(7)
    V = rng.normal(size=(4, 3)).astype(np.float32)

    incumbent = History(splits, 2, "recency_weighted_pool")
    multi = MultiBehaviorHistory(splits, 2)
    old_sum = incumbent.pooled(V)
    new_sum = multi.pooled_many(V)
    old_valid = incumbent.batch_vectors(old_sum, query["user"], False)
    new_valid = multi.batch_vectors_many(new_sum, query["user"], False)[:, 0]
    check("multi-behavior positive channel reproduces recency pool",
          np.allclose(old_valid, new_valid, atol=1e-6))

    old_train = incumbent.batch_vectors(
        old_sum, tr["user"], True, V, tr["video"],
        incumbent.train_row_w, tr["long_view"])
    new_train = multi.batch_vectors_many(
        new_sum, tr["user"], True, V_video=V,
        row_index=np.arange(len(tr["user"])))[:, 0]
    check("multi-behavior train positive channel preserves leave-one-out",
          np.allclose(old_train, new_train, atol=1e-6))

    X = np.array([[0, 4, 8], [1, 5, 9]], dtype=np.int32)
    H = rng.normal(size=(2, 4)).astype(np.float32)
    Hx = rng.normal(size=(2, 3, 4)).astype(np.float32)
    control = RankFM(12, k=4, seed=11)
    signed = RankFM(12, k=4, seed=11, history_channels=3)
    z0, _ = control.forward(X, H)
    z1, cache = signed.forward(X, H, Hx)
    check("zero signed-history gates exactly match the control logits",
          np.array_equal(z0, z1))
    signed.apply_grads([(cache, np.array([0.5, -0.5], dtype=np.float32))])
    check("signed-history gates receive a training gradient",
          bool(np.any(np.abs(signed.history_gate_raw) > 0)))

    fwfm = RankFM(12, k=4, seed=11, n_fields=3, field_weighted=True)
    zf, fw_cache = fwfm.forward(X, H)
    check("field-weighted FM starts from ordinary FM logits",
          np.allclose(z0, zf, atol=1e-6))
    fwfm.apply_grads([(fw_cache, np.array([0.5, -0.5], dtype=np.float32))])
    check("field interaction weights receive a training gradient",
          bool(np.any(np.abs(fwfm.field_pair_raw) > 0)))

    before, _ = oof_recency_bayesian_prior_scores(splits)
    changed = copy.deepcopy(splits)
    changed["train"] = {k: v.copy() for k, v in tr.items()}
    changed["train"]["long_view"][tr["date"] == 1] = \
        1 - changed["train"]["long_view"][tr["date"] == 1]
    after, _ = oof_recency_bayesian_prior_scores(changed)
    held = tr["date"] == 1
    check("date-OOF prior excludes every label from the held-out date",
          np.allclose(before["train"][held], after["train"][held], atol=1e-6))
    check("date-OOF prior arrays are finite and aligned",
          all(v.shape == (len(splits[k]["user"]),) and np.all(np.isfinite(v))
              for k, v in before.items()))
    ranks = rank_normalize(np.array([4.0, -2.0, 1.5, 1.5], dtype=np.float32))
    check("snapshot rank normalization is deterministic and tie-aware",
          np.array_equal(ranks, np.array([1.0, 0.0, 0.5, 0.5])))

    batch_split = {
        "user": np.array([99, 99, 99, 99, 99], dtype=np.int32),
        "user_raw": np.array(["new-a", "new-a", "new-a", "new-b", "new-b"]),
        # Every encoded video is the shared UNK value; raw ids remain distinct.
        "video": np.array([99, 99, 99, 99, 99], dtype=np.int32),
        "video_raw": np.array(["v1", "v1", "v2", "v3", "v4"]),
    }
    adjusted, info = batch_repeat_fatigue_scores(
        np.array([3.0, 2.0, 1.0, 3.0, 2.0]), batch_split)
    base_a = np.array([3.0, 2.0, 1.0])
    base_a = (base_a - base_a.mean()) / base_a.std()
    check("batch fatigue uses true users instead of a shared UNK code",
          np.isclose(info["affected_user_fraction"], 0.5))
    check("batch fatigue uses raw videos instead of a shared UNK code",
          info["identity"] == "user_raw_x_video_raw")
    check("repeat penalty boosts a unique item relative to a repeated item",
          adjusted[2] - adjusted[1] > base_a[2] - base_a[1])
    check("batch fatigue is finite and explicitly label-free",
          np.all(np.isfinite(adjusted)) and not info["uses_outcome_columns"])


def test_verified_ensemble_contract():
    """Fast checks for the frozen, reproducible final-run entry point."""
    print("\n[verified ensemble contract]")
    try:
        Menu(MENU_PATH).validate_choices(ENSEMBLE_CHOICES)
        check("frozen ensemble configuration remains selectable", True)
    except MenuError as error:
        check("frozen ensemble configuration remains selectable", False,
              str(error))

    ranks = _rank_normalize(
        np.array([4.0, -2.0, 1.5, 1.5], dtype=np.float64),
        np.array(["u", "u", "u", "u"]),
        np.array([4, 1, 2, 3], dtype=np.int64))
    check("ensemble ranks within user and breaks score ties by time",
          np.array_equal(ranks, np.array([1.0, 0.0, 1 / 3, 2 / 3])))
    check("exact score-and-time ties receive neutral midranks",
          np.array_equal(
              _rank_normalize(np.ones(3), np.array(["u", "u", "u"]),
                              np.ones(3, dtype=np.int64)),
              np.full(3, 0.5)))
    check("ensemble rank scaling is per-user",
          np.array_equal(
              _rank_normalize(np.array([0.0, 5.0, 2.0, 9.0]),
                              np.array(["a", "a", "b", "b"]),
                              np.arange(4)),
              np.array([0.0, 1.0, 0.0, 1.0])))
    protected = _protected_paths()
    check("reuse binds every split cache and cache metadata",
          {"train_cache", "validation_cache", "test_cache", "cache_meta",
           "cache_vocabs", "cache_schema", "executor", "child_guard"}
          .issubset(protected))
    runtime = _runtime_fingerprint()
    check("reuse binds Python, NumPy, and platform versions",
          {"python_version", "numpy_version", "platform"}.issubset(runtime))
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        seed_dir = Path(td) / "seed_00"
        backup = Path(td) / ".seed_00.previous-test"
        backup.mkdir()
        (backup / "marker").write_text("previous", encoding="utf-8")
        _recover_previous(seed_dir)
        check("interrupted member promotion restores its previous artifact",
              (seed_dir / "marker").read_text(encoding="utf-8") == "previous")


def test_verified_submission_bundle():
    """Submission loading must trust only the hash-checked published bundle."""
    print("\n[verified submission bundle]")
    from pathlib import Path

    def digest(path):
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                value.update(block)
        return value.hexdigest()

    with tempfile.TemporaryDirectory() as td:
        verified_root = Path(td) / "results" / "verified_ensemble"
        bundle = verified_root / "bundles" / "fake-bundle"
        bundle.mkdir(parents=True)
        valid_path = bundle / "scores_valid.npy"
        test_path = bundle / "scores_test.npy"
        expected_valid = np.zeros(124_909, dtype=np.float32)
        expected_valid[:3] = [0.25, 0.5, 0.75]
        np.save(valid_path, expected_valid)
        np.save(test_path, np.zeros(170_588, dtype=np.float32))
        summary = {
            "seeds": list(range(5)),
            "ensemble_runner_sha256": digest(
                Path(_ROOT) / "agent" / "verified_ensemble.py"),
            "runtime_fingerprint": _runtime_fingerprint(),
            "protected_sha256": _hash_protected(_protected_paths()),
            "artifacts_sha256": {
                "scores_valid.npy": digest(valid_path),
                "scores_test.npy": digest(test_path),
            }
        }
        summary_path = bundle / "summary.json"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        latest_path = verified_root / "latest.json"
        latest = {
            "bundle": "bundles/fake-bundle",
            "summary_sha256": digest(summary_path),
        }
        latest_path.write_text(json.dumps(latest), encoding="utf-8")

        scores, loaded_summary, loaded_bundle = verified_ensemble_scores(
            td, "valid")
        check("submission loads a valid hash-checked ensemble bundle",
              np.array_equal(scores, expected_valid)
              and loaded_summary == summary
              and loaded_bundle == bundle.resolve())

        with valid_path.open("ab") as handle:
            handle.write(b"corrupt")
        try:
            verified_ensemble_scores(td, "valid")
            check("submission rejects a corrupt ensemble score artifact", False)
        except ValueError as error:
            check("submission rejects a corrupt ensemble score artifact",
                  "scores_valid.npy hash mismatch" in str(error), str(error))

        np.save(valid_path, expected_valid)
        latest["summary_sha256"] = "0" * 64
        latest_path.write_text(json.dumps(latest), encoding="utf-8")
        try:
            verified_ensemble_scores(td, "valid")
            check("submission rejects a corrupt ensemble summary hash", False)
        except ValueError as error:
            check("submission rejects a corrupt ensemble summary hash",
                  "summary hash mismatch" in str(error), str(error))


if __name__ == "__main__":
    for t in (test_safety_gate, test_validity, test_policy, test_convergence,
              test_executor, test_hidden_test_boundary, test_crossover,
              test_spend_ceiling,
              test_audit_regressions, test_numpy2_json_serialization,
              test_research_primitives, test_verified_ensemble_contract,
              test_verified_submission_bundle):
        t()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
    sys.exit(1 if FAIL else 0)
