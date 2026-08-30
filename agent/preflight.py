"""Catch a broken experiment before it costs a training run.

The clean autonomy evaluation measured the problem this solves: 5 of 7 Path B
nodes crashed, none of them for an ML reason. The dominant failure was calling
something that does not exist -- `train_lib.training_dynamics()` -- which is
knowable in about two seconds and was instead discovered after the harness had
committed a full iteration to it.

The principle:

    Do not spend an expensive research iteration discovering that a function
    does not exist.

Stages run cheapest-first and stop at the first failure, so a syntax error
never pays for an import resolution:

    1. SYNTAX        ast.parse
    2. IMPORTS       every imported module is on the contract's allow-list
    3. CAPABILITY    every `module.attr` the script calls actually EXISTS,
                     resolved by importing in a subprocess
    4. CALL_ARITY    every required argument is supplied
    5. RETURN_SHAPE  the call site agrees with the declared return shape
    6. CONFIG        menu_choices validate against the menu and its bounds
    7. LEAKAGE       the existing static label-leakage review
    8. SMOKE         the script's own import block executes cleanly

Stages 4 and 5 were added because fixing stage 3 exposed the layer beneath it.
Once the agent stopped calling functions that do not exist, it started calling
existing ones wrongly: destructuring a dict as a tuple, unpacking a 3-tuple
capture entry into 4 names, and omitting a required argument. Each cost a full
training run -- 42s, 71s, 73s, 983s -- to learn something already written down.

Everything returns STRUCTURED FEEDBACK. A preflight failure is not a dead end;
it is a message telling the agent precisely what is wrong and what exists
instead, so the repair is targeted rather than a guess.

Budget note: a preflight failure costs no training time and is accounted
separately from a completed experiment. See `agent.budget`.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
RUNTIME_DIR = os.path.join(ROOT, "runtime")
KIT_DIR = os.path.join(ROOT, "kuairand-starter-kit")

from . import capabilities as caps  # noqa: E402
from . import leakage_check  # noqa: E402

SYNTAX, IMPORTS, CAPABILITY = "syntax", "imports", "capability"
RETURN_SHAPE = "return_shape"
CALL_ARITY = "call_arity"
CONFIG, LEAKAGE, SMOKE = "config", "leakage", "smoke"
STAGES = (SYNTAX, IMPORTS, CAPABILITY, CALL_ARITY, RETURN_SHAPE,
          CONFIG, LEAKAGE, SMOKE)

# Modules generated code may import: the contract's own list, plus the ordinary
# scientific-Python surface any experiment needs.
# Generous on purpose. This list exists to catch "you imported something that
# does not exist in the experiment environment", NOT to sandbox the script --
# the data boundary is enforced with file permissions, not with an import list.
# A false rejection here costs the agent an attempt for nothing, which is
# exactly what happened when a script was rejected for importing `traceback`.
_STDLIB_OK = {
    "abc", "argparse", "bisect", "collections", "contextlib", "copy", "csv",
    "dataclasses", "datetime", "decimal", "enum", "functools", "gzip", "hashlib",
    "heapq", "io", "itertools", "json", "logging", "math", "operator", "os",
    "pathlib", "pprint", "random", "re", "shutil", "statistics", "string", "sys",
    "textwrap", "time", "traceback", "typing", "uuid", "warnings",
}
_THIRD_PARTY_OK = {"numpy", "np", "scipy", "pandas"}


def allowed_modules() -> set:
    return set(caps.modules_for_generated_code()) | _STDLIB_OK | _THIRD_PARTY_OK


class Issue:
    def __init__(self, stage: str, message: str, line: int | None = None,
                 fix: str = ""):
        self.stage, self.message, self.line, self.fix = stage, message, line, fix

    def as_dict(self) -> dict:
        return {"stage": self.stage, "message": self.message,
                "line": self.line, "fix": self.fix}


# ------------------------------------------------------------------ stage 1 ---
def check_syntax(src: str) -> tuple:
    try:
        return ast.parse(src), []
    except SyntaxError as e:
        return None, [Issue(SYNTAX, f"{e.msg}", e.lineno,
                            "The script never ran. Fix the syntax and resubmit; "
                            "no training time was spent.")]


# ------------------------------------------------------------------ stage 2 ---
def _imported_modules(tree: ast.AST) -> dict:
    """module root -> first line it was imported on."""
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.setdefault(a.name.split(".")[0], node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.setdefault(node.module.split(".")[0], node.lineno)
    return found


def check_imports(tree: ast.AST) -> list:
    ok = allowed_modules()
    issues = []
    for mod, line in _imported_modules(tree).items():
        if mod in ok:
            continue
        if mod == "agent":
            issues.append(Issue(
                IMPORTS, f"cannot import `{mod}` from generated code", line,
                "The agent/ package is NOT on the experiment subprocess's "
                "PYTHONPATH -- only runtime/ and the starter kit are. The "
                "research capabilities you want are re-exported for exactly "
                "this reason: `from research_tools import ...`."))
        else:
            issues.append(Issue(
                IMPORTS, f"module `{mod}` is not available to generated code", line,
                f"Importable modules are: {', '.join(sorted(ok))}."))
    return issues


# ------------------------------------------------------------------ stage 3 ---
def _attribute_calls(tree: ast.AST, modules: set) -> list:
    """(module, attr, lineno) for every `module.attr` referenced."""
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in modules):
            out.append((node.value.id, node.attr, node.lineno))
    return out


def _imported_names(tree: ast.AST) -> list:
    """(module, name, lineno) for every `from module import name`."""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for a in node.names:
                out.append((node.module.split(".")[0], a.name, node.lineno))
    return out


def _resolve_in_subprocess(wanted: list, timeout_s: int = 90) -> dict:
    """Import each module in the REAL experiment environment and report which
    attributes exist. Run out-of-process so a heavy import cannot pollute or
    crash the orchestrator."""
    if not wanted:
        return {}
    probe = (
        "import json,sys\n"
        f"wanted={json.dumps(wanted)}\n"
        "out={}\n"
        "for mod,attr in wanted:\n"
        "    try:\n"
        "        m=__import__(mod)\n"
        "    except Exception as e:\n"
        "        out[mod+'.'+attr]={'module_ok':False,'err':f'{type(e).__name__}: {e}'[:200]}\n"
        "        continue\n"
        "    ok=hasattr(m,attr)\n"
        "    cand=[]\n"
        "    if not ok:\n"
        "        import difflib\n"
        "        cand=difflib.get_close_matches(attr,[n for n in dir(m) if not n.startswith('_')],3,0.6)\n"
        "    out[mod+'.'+attr]={'module_ok':True,'exists':ok,'suggest':cand}\n"
        "print(json.dumps(out))\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [RUNTIME_DIR, KIT_DIR, env.get("PYTHONPATH", "")])
    try:
        r = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                           text=True, timeout=timeout_s, env=env, cwd=ROOT)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except (subprocess.SubprocessError, ValueError, json.JSONDecodeError, IndexError):
        return {}


def _target_arity(target) -> int | None:
    """How many names an assignment target destructures into, if it is a tuple."""
    if isinstance(target, (ast.Tuple, ast.List)):
        return len(target.elts)
    return None


def _called_capability(node: ast.AST) -> str | None:
    """Name of the contract capability this Call invokes, if any."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def check_call_arity(tree: ast.AST) -> list:
    """Is every required argument actually supplied?

    The layer under return shapes, and the next one the agent hit once shapes
    were fixed: `selection_rule_test() missing 1 required positional argument:
    'rules'` — 73 seconds of training spent to learn a signature the contract
    already knows.

    Deliberately conservative. It only complains when a call supplies FEWER
    arguments than there are required parameters and cannot be doing so through
    *args/**kwargs, because a false rejection costs the agent an attempt for no
    reason.
    """
    issues = []
    known = caps.all_capabilities()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_capability(node)
        cap = known.get(name) if name else None
        if cap is None or not cap.params:
            continue
        required = list(cap.params.get("required") or [])
        if not required:
            continue
        # Unpacking into the call could supply anything; say nothing.
        if any(isinstance(a, ast.Starred) for a in node.args) or \
           any(k.arg is None for k in node.keywords):
            continue
        supplied = len(node.args) + len({k.arg for k in node.keywords if k.arg})
        allowed = set(required) | set(cap.params.get("optional") or [])
        if not cap.params.get("var_keyword"):
            unexpected = [k.arg for k in node.keywords
                          if k.arg and k.arg not in allowed]
            if unexpected:
                issues.append(Issue(
                    CALL_ARITY,
                    f"`{name}` does not accept keyword argument(s): "
                    f"{', '.join(unexpected)}", node.lineno,
                    (cap.example or
                     f"Accepted parameters: {', '.join(required + list(cap.params.get('optional') or []))}.")))
                continue
        if supplied >= len(required):
            # A common failure mode was passing a list of candidate names where
            # selection_rule_test needs a mapping of name -> callable. Catch
            # direct literals and simple aliases without attempting brittle
            # whole-program type inference.
            if name == "selection_rule_test":
                rules_arg = None
                if len(node.args) >= 4:
                    rules_arg = node.args[3]
                else:
                    rules_arg = next((k.value for k in node.keywords
                                      if k.arg == "rules"), None)
                if isinstance(rules_arg, (ast.List, ast.Tuple, ast.Set)):
                    issues.append(Issue(
                        CALL_ARITY,
                        "`selection_rule_test` needs `rules` to be a dict of "
                        "name -> callable, not a list of rule names", node.lineno,
                        "Use `capture_selection_rule_test(capture, users, labels)` "
                        "for a captured epoch curve, or pass a dict such as "
                        "{'best_epoch': lambda p, e: e[int(np.argmax(p))]}."))
            continue
        given_kw = {k.arg for k in node.keywords if k.arg}
        missing = [p for p in required[len(node.args):] if p not in given_kw]
        if not missing:
            continue
        issues.append(Issue(
            CALL_ARITY,
            f"`{name}` requires {len(required)} argument(s) "
            f"({', '.join(required)}); this call supplies {supplied}, missing "
            f"{', '.join(missing)}", node.lineno,
            (cap.example or f"Required: {', '.join(required)}."
                            + (f" Optional: {', '.join(cap.params.get('optional') or [])}."
                               if cap.params.get("optional") else ""))))
    return issues


def check_return_shapes(tree: ast.AST) -> list:
    """Does each call site agree with the capability's DECLARED return shape?

    This is the stage that would have prevented every Path B crash in the last
    recorded run. All four were a call site disagreeing with a shape the
    contract already knew:

        valid, test = train_lib.train_numpy_fm(...)   # returns a DICT
        for a, b in cfg['capture_epoch_scores']       # entries are 3-tuples
        ... looking for a test vector inside a per-epoch entry that has none

    Each cost a full training run to discover -- 42s, 42s, 71s and 983s of
    compute to learn a fact that was already written down.
    """
    issues = []
    known = caps.all_capabilities()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or node.value is None:
            continue
        name = _called_capability(node.value)
        cap = known.get(name) if name else None
        if cap is None or not cap.returns:
            continue
        for tgt in node.targets:
            arity = _target_arity(tgt)
            if arity is None:
                continue                       # plain `x = f(...)` is always fine
            kind = cap.return_kind
            if kind == "dict":
                keys = ", ".join((cap.returns.get("keys") or [])[:4])
                issues.append(Issue(
                    CAPABILITY,
                    f"`{name}` returns a DICT, but this line unpacks it into "
                    f"{arity} names — that raises ValueError at runtime",
                    node.lineno,
                    f"Index the dict instead of destructuring it. Keys: {keys}."
                    + (f"\n{cap.example}" if cap.example else "")))
            elif kind == "tuple" and cap.return_arity not in (None, arity):
                issues.append(Issue(
                    CAPABILITY,
                    f"`{name}` returns {cap.return_arity} values, but this line "
                    f"unpacks {arity}", node.lineno,
                    (cap.example or
                     f"Expected names: {', '.join(cap.returns.get('names') or [])}")))

    # Iterating a list-of-tuple payload with the wrong arity. The capture list
    # is reached through a config KEY, not a call, so it needs its own check.
    for node in ast.walk(tree):
        if not isinstance(node, ast.For):
            continue
        if not _iterates_capture(node.iter):
            continue
        arity = _target_arity(node.target)
        cap = known.get("capture_epoch_scores")
        want = cap.return_arity if cap else 3
        if arity is not None and arity != want:
            names = ", ".join((cap.returns.get("names") or []) if cap else [])
            issues.append(Issue(
                CAPABILITY,
                f"each capture_epoch_scores entry has {want} elements, but this "
                f"loop unpacks {arity}", node.lineno,
                f"Entries are ({names}). The array is the VALID split only — "
                f"there is no per-epoch test vector. Take test predictions from "
                f"train_numpy_fm's returned dict: res['scores_test']."))
    return issues


def _is_capture_expr(node: ast.AST) -> bool:
    """Does this expression evaluate to the raw capture list?"""
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and sl.value == "capture_epoch_scores":
            return True
    if isinstance(node, ast.Name) and "capture" in node.id.lower():
        return True
    if isinstance(node, ast.Attribute) and "capture" in node.attr.lower():
        return True
    return False


def check_capture_misuse(tree: ast.AST) -> list:
    """Passing the raw capture list straight into selection_rule_test.

    Observed live, and it costs a full training run every time: the capture
    payload is a LIST of (epoch, primary, scores) tuples, while
    selection_rule_test wants a 3-D (seeds, epochs, rows) array. Handing one to
    the other raises deep inside numpy --

        ValueError: setting an array element with a sequence. The requested
        array has an inhomogeneous shape after 2 dimensions

    -- which is a long way from the actual mistake. `capture_selection_rule_test`
    exists precisely to bridge the two, so this points at it rather than asking
    the agent to reshape by hand.
    """
    issues = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if _called_capability(node) != "selection_rule_test":
            continue
        if not node.args or not _is_capture_expr(node.args[0]):
            continue
        issues.append(Issue(
            RETURN_SHAPE,
            "selection_rule_test's first argument must be a 3-D array of shape "
            "(seeds, epochs, rows); this passes the raw capture list, which is "
            "a list of (epoch, valid_primary, scores) tuples",
            node.lineno,
            "Use the adapter that already does this conversion:\n"
            "    from research_tools import capture_selection_rule_test\n"
            "    out = capture_selection_rule_test("
            "cfg['capture_epoch_scores'], users, labels)"))
    return issues


def _iterates_capture(node: ast.AST) -> bool:
    """Is this iterating cfg['capture_epoch_scores'] (or a plain alias)?"""
    if isinstance(node, ast.Subscript):
        sl = node.slice
        if isinstance(sl, ast.Constant) and sl.value == "capture_epoch_scores":
            return True
    if isinstance(node, ast.Name) and "capture" in node.id.lower():
        return True
    if isinstance(node, ast.Attribute) and "capture" in node.attr.lower():
        return True
    return False


def check_capabilities(tree: ast.AST) -> list:
    """The stage that catches the measured failure.

    Two distinct mistakes are reported differently, because they need different
    repairs: calling an ORCHESTRATION-ONLY capability (the contract knows what
    to do instead), and calling something that simply is not there.
    """
    issues = []
    mods = set(_imported_modules(tree)) & allowed_modules()
    refs = _attribute_calls(tree, mods)
    named = _imported_names(tree)

    orch_only = caps.orchestration_only()

    # (a) orchestration-only capabilities, wherever they are referenced from
    for mod, attr, line in refs + [(m, n, ln) for m, n, ln in named]:
        if attr in orch_only:
            c = caps.get(attr)
            issues.append(Issue(
                CAPABILITY,
                f"`{attr}` is an ORCHESTRATION-ONLY capability and does not "
                f"exist inside generated code (you referenced it as "
                f"`{mod}.{attr}`)", line,
                (c.instead if c and c.instead else
                 "Request it during the inspect phase instead of calling it here.")))

    # (b) anything else that must actually resolve
    wanted = sorted({(m, a) for m, a, _ in refs if a not in orch_only}
                    | {(m, n) for m, n, _ in named if n not in orch_only})
    resolved = _resolve_in_subprocess(wanted)
    line_of = {(m, a): ln for m, a, ln in refs}
    line_of.update({(m, n): ln for m, n, ln in named})
    for (mod, attr) in wanted:
        info = resolved.get(f"{mod}.{attr}")
        if not info:
            continue
        line = line_of.get((mod, attr))
        if not info.get("module_ok"):
            issues.append(Issue(CAPABILITY,
                                f"`import {mod}` failed: {info.get('err')}", line,
                                "This module is on the allow-list but did not "
                                "import in the experiment environment."))
        elif not info.get("exists"):
            sug = info.get("suggest") or []
            hint = (f"Did you mean: {', '.join(sug)}?" if sug else
                    "Check the capability contract for what this module provides.")
            issues.append(Issue(
                CAPABILITY, f"`{mod}.{attr}` does not exist", line, hint))
    return issues


# ------------------------------------------------------------------ stage 4 ---
def check_config(menu_choices: dict | None, menu=None) -> list:
    if menu is None or menu_choices is None:
        return []
    try:
        menu.validate_choices(menu_choices)
        return []
    except Exception as e:                       # MenuError and friends
        return [Issue(CONFIG, str(e)[:400], None,
                      "Fix the configuration; nothing was trained.")]


# ------------------------------------------------------------------ stage 5 ---
def check_leakage(code_path: str) -> list:
    try:
        leak = leakage_check.check_file(code_path)
    except Exception as e:                       # noqa: BLE001 - advisory stage
        return [Issue(LEAKAGE, f"leakage review could not run: "
                               f"{type(e).__name__}: {e}"[:200])]
    if not leak.get("block"):
        return []
    return [Issue(LEAKAGE, "label leakage review BLOCKED this script", None,
                  leakage_check.render_for_agent(leak))]


# ------------------------------------------------------------------ stage 6 ---
def check_smoke(code_path: str, timeout_s: int = 120) -> list:
    """Execute the script's import block only.

    Everything above is static. This is the cheapest possible dynamic check:
    it proves the file's imports and module-level code actually run in the real
    environment, without touching training.
    """
    src = open(code_path).read()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []                                # stage 1 already reported it
    keep = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    if not keep:
        return []
    mod = ast.Module(body=keep, type_ignores=[])
    try:
        snippet = ast.unparse(mod)
    except Exception:                            # noqa: BLE001
        return []
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.abspath(code_path)), RUNTIME_DIR, KIT_DIR,
         env.get("PYTHONPATH", "")])
    try:
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True,
                           text=True, timeout=timeout_s, env=env, cwd=ROOT)
    except subprocess.SubprocessError as e:
        return [Issue(SMOKE, f"import smoke test could not run: {e}"[:200])]
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()
        return [Issue(SMOKE, "the script's imports failed to execute", None,
                      "\n".join(tail[-4:])[:500])]
    return []


# -------------------------------------------------------------------- driver ---
def preflight(code_path: str, menu_choices: dict | None = None, menu=None,
              skip: tuple = ()) -> dict:
    """Run the stages cheapest-first, stopping at the first that fails."""
    with open(code_path) as fh:
        src = fh.read()

    ran, issues = [], []
    tree = None
    for stage in STAGES:
        if stage in skip:
            continue
        ran.append(stage)
        if stage == SYNTAX:
            tree, issues = check_syntax(src)
        elif stage == IMPORTS:
            issues = check_imports(tree)
        elif stage == CAPABILITY:
            issues = check_capabilities(tree)
        elif stage == CALL_ARITY:
            issues = check_call_arity(tree)
        elif stage == RETURN_SHAPE:
            issues = check_return_shapes(tree) + check_capture_misuse(tree)
        elif stage == CONFIG:
            issues = check_config(menu_choices, menu)
        elif stage == LEAKAGE:
            issues = check_leakage(code_path)
        elif stage == SMOKE:
            issues = check_smoke(code_path)
        if issues:
            break

    return {"ok": not issues, "failed_stage": ran[-1] if issues else None,
            "stages_run": ran,
            "issues": [i.as_dict() for i in issues],
            "spent_training_time": False,
            "feedback": render_feedback(ran[-1], issues) if issues else ""}


def render_feedback(stage: str, issues: list) -> str:
    """The message the agent receives. Specific, and it names the repair."""
    L = [f"PREFLIGHT FAILED at the {stage.upper()} stage. Your experiment was "
         f"NOT run, so no training time was spent and your hypothesis is still "
         f"untested.", ""]
    for i in issues:
        where = f" (line {i.line})" if i.line else ""
        L.append(f"  - {i.message}{where}")
        if i.fix:
            for ln in i.fix.splitlines():
                L.append(f"      {ln}")
    L += ["", "Fix exactly this and resubmit. Do not change your hypothesis: "
              "the experiment has not been tested yet."]
    return "\n".join(L)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    a = ap.parse_args()
    r = preflight(a.path)
    print(json.dumps({k: v for k, v in r.items() if k != "feedback"}, indent=2))
    if r["feedback"]:
        print("\n" + r["feedback"])
    raise SystemExit(0 if r["ok"] else 1)
