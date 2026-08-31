"""Failure taxonomy and repair policy for generated experiments.

Stage E exists because Path B makes custom code common, and custom code fails
in more ways than a menu selection can. The previous behaviour was a blind
debug retry capped at 2 attempts, with no notion of WHY something failed --
so a missing dependency and a diverged training run were treated identically.

Two principles:

1. A failed EXPERIMENT is not the same as a failed HYPOTHESIS. Code that
   crashes taught us nothing about the research question. Code that ran
   correctly and scored badly answered the question -- that is a RESULT, and
   retrying it would be both wasteful and scientifically wrong. The taxonomy
   keeps these strictly separate.

2. Never blindly retry. Each class carries its own repair strategy and an
   explicit `retry_worthwhile` flag; some classes (a disproved hypothesis, a
   hard resource ceiling) should not be retried at all, and saying so is more
   useful than burning two attempts to rediscover it.
"""
from __future__ import annotations

import re

# --- classes ---------------------------------------------------------------
SYNTAX = "implementation_syntax"
IMPORT = "dependency_import"
API_MISUSE = "api_misuse"
RUNTIME = "runtime_error"
DATA_CONTRACT = "data_contract"
NUMERICAL = "numerical_instability"
DIVERGENCE = "training_divergence"
RESOURCE = "resource_exhaustion"
CUDA = "cuda_device"
TIMEOUT = "timeout"
INVALID_PREDICTIONS = "invalid_predictions"
EVALUATION = "evaluation_failure"
LLM_RESPONSE = "invalid_llm_response"
LEAKAGE_BLOCKED = "leakage_blocked"
MECHANISM_BLOCKED = "mechanism_blocked"
HYPOTHESIS_DISPROVED = "hypothesis_disproved"   # NOT a failure of the code
UNKNOWN = "unknown"

# A few classes leave their evidence in STDOUT rather than in the traceback: an
# OOM kill and a CUDA device assert are printed by the kernel or the driver, not
# raised. Everything else must be diagnosed from the traceback.
#
# The distinction is not academic. Every training run in this project logs
# `valid primary ... (GAUC ... nDCG@5 ...)` once per epoch, so matching the
# generic patterns against stdout means the EVALUATION pattern (`ndcg|gauc`)
# fires on ordinary progress logging. A deliberately injected RuntimeError was
# classified `evaluation_failure` in a live run for exactly this reason, and the
# agent was handed "Scoring itself failed. Use train_lib's official evaluate" --
# guidance that points away from the actual fault.
_STDOUT_VISIBLE = ("resource_exhaustion", "cuda_device")

# Ordered: first match wins, so specific patterns precede generic ones.
_PATTERNS = [
    (LEAKAGE_BLOCKED, r"BLOCKED BEFORE EXECUTION by the leakage review"),
    (MECHANISM_BLOCKED, r"BLOCKED BEFORE EXECUTION by the mechanism audit"),
    (LLM_RESPONSE, r"LLM stage failed|response schema violations|no JSON object"),
    (TIMEOUT, r"\bTIMEOUT\b|exceeded \d+s and was killed"),
    (CUDA, r"CUDA|cuDNN|device-side assert|MPS backend|torch\.cuda"),
    (RESOURCE, r"OutOfMemoryError|MemoryError|Cannot allocate memory|Killed"),
    (SYNTAX, r"SyntaxError|IndentationError|TabError|unterminated"),
    (IMPORT, r"ModuleNotFoundError|ImportError|No module named"),
    (NUMERICAL, r"\bnan\b|NaN|infinity|\binf\b|overflow encountered|divide by zero",),
    (DATA_CONTRACT, r"scores_\w+\.npy|metrics\.json|shape \(|output contract|"
                    r"row count mismatch|was not written"),
    (INVALID_PREDICTIONS, r"far below random|contains NaN/Inf"),
    (EVALUATION, r"evaluate\(|ndcg|gauc", ),
    (API_MISUSE, r"AttributeError|KeyError|TypeError|ValueError|IndexError|"
                 r"NameError|UnboundLocalError"),
    (RUNTIME, r"RuntimeError|AssertionError"),
]

_GUIDANCE = {
    SYNTAX: ("The script did not parse. Fix the syntax only; do not redesign "
             "the experiment -- the hypothesis was never tested."),
    IMPORT: ("A dependency is unavailable. Only numpy, torch 2.3 (CPU) and the "
             "starter kit are guaranteed. Re-implement using those, or drop "
             "the dependency."),
    API_MISUSE: ("The code called a train_lib/numpy API incorrectly (wrong key, "
                 "wrong argument, wrong shape). Re-read the API section for the "
                  "exact signature; the hypothesis itself may still be sound."),
    RUNTIME: ("Generated code raised during execution. Use the captured "
              "exception and traceback to repair the implementation without "
              "changing the untested hypothesis."),
    DATA_CONTRACT: ("The script ran but violated the output contract "
                    "(metrics.json / scores_*.npy shape or presence). Fix the "
                    "output writing; the modelling idea is untouched."),
    NUMERICAL: ("Training produced NaN/Inf. Usually a learning rate that is too "
                "high, an unclipped exp/log, or division by a count that can be "
                "zero. Stabilise numerically before changing the idea."),
    DIVERGENCE: ("Training ran but the objective diverged. Lower the learning "
                 "rate or add clipping; the mechanism may still be worth testing."),
    RESOURCE: ("Ran out of memory. Reduce batch size or the size of any "
               "materialised matrix. Do not retry unchanged."),
    CUDA: ("A device error occurred. This project runs CPU-only by default; "
           "prefer the numpy engine and do not require a GPU."),
    TIMEOUT: ("Killed on the wall-clock ceiling, which scores as a crash with no "
              "partial credit. The experiment must be made cheaper -- fewer "
              "epochs, fewer passes, or a smaller mechanism -- or abandoned."),
    INVALID_PREDICTIONS: ("Scores were produced but are degenerate (NaN, or "
                          "below random). Usually a sign error or an untrained "
                          "path being scored."),
    EVALUATION: ("Scoring itself failed. Use train_lib's official evaluate on "
                 "(user_raw, labels, scores); never reimplement the metric."),
    LLM_RESPONSE: ("The response was malformed, not the experiment. Re-emit "
                   "valid JSON matching the required schema."),
    HYPOTHESIS_DISPROVED: ("The experiment RAN CORRECTLY and the result did not "
                           "support the hypothesis. This is a RESULT, not a bug. "
                           "Do not retry it -- record it and move to a different "
                           "mechanism."),
    MECHANISM_BLOCKED: ("The experiment was refused before running because the "
                        "implementation cannot affect the metric. Both metrics "
                        "rank WITHIN a user, so a monotone per-user transform of "
                        "the scores changes neither. Propose a mechanism that "
                        "reorders items INSIDE a single user's impression list."),
    LEAKAGE_BLOCKED: ("The experiment was refused before running because the "
                      "code appeared to use evaluation-split labels. Rebuild the "
                      "feature so it uses only information available BEFORE the "
                      "row being scored. The hypothesis is still untested."),
    UNKNOWN: ("Cause unclear. Reduce the experiment to the smallest version that "
              "still tests the hypothesis and re-run to localise the fault."),
}

# Classes where re-attempting the SAME idea (after a fix) is scientifically
# justified. A disproved hypothesis and a hard ceiling are excluded.
# MECHANISM_BLOCKED is deliberately NOT retry-worthwhile: the idea is
# arithmetically incapable of moving the metric, so a repaired version of the
# same idea is equally incapable. It needs a different mechanism, not a fix.
_RETRY_WORTHWHILE = {SYNTAX, IMPORT, API_MISUSE, RUNTIME, DATA_CONTRACT, NUMERICAL,
                     DIVERGENCE, INVALID_PREDICTIONS, EVALUATION, LLM_RESPONSE,
                     LEAKAGE_BLOCKED, UNKNOWN}
# Classes needing the experiment made materially cheaper/smaller, not just fixed.
_NEEDS_SHRINK = {TIMEOUT, RESOURCE}


def classify(error_trace: str | None, *, status: str = "error",
             metrics: dict | None = None) -> dict:
    """Classify one experiment outcome.

    A SUCCESSFUL run that merely scored poorly is classified as
    HYPOTHESIS_DISPROVED, never as a failure -- keeping "the code broke" and
    "the idea was wrong" separable is the whole point of the taxonomy.
    """
    if status == "success":
        return {"class": HYPOTHESIS_DISPROVED, "is_code_failure": False,
                "retry_worthwhile": False, "needs_shrink": False,
                "guidance": _GUIDANCE[HYPOTHESIS_DISPROVED],
                "likely_cause": "experiment executed correctly; the result simply "
                                "did not support the hypothesis",
                "metrics": metrics}
    t = error_trace or ""
    # Diagnose from the traceback. The executor appends the run's stdout after
    # a "--- stdout" marker, and that tail is progress logging, not a fault.
    stderr_part = t.split("--- stdout")[0]
    cls = UNKNOWN
    for name, pat in _PATTERNS:
        if re.search(pat, stderr_part, re.IGNORECASE):
            cls = name
            break
    if cls is UNKNOWN:
        # Second pass over the whole trace, but only for the faults that really
        # do announce themselves in stdout.
        for name, pat in _PATTERNS:
            if name in _STDOUT_VISIBLE and re.search(pat, t, re.IGNORECASE):
                cls = name
                break
    # A NaN reported by the contract checker is an invalid-prediction problem,
    # not generic numerical noise mid-training.
    if cls == NUMERICAL and re.search(r"scores_\w+\.npy contains", t):
        cls = INVALID_PREDICTIONS
    return {"class": cls, "is_code_failure": True,
            "retry_worthwhile": cls in _RETRY_WORTHWHILE,
            "needs_shrink": cls in _NEEDS_SHRINK,
            "guidance": _GUIDANCE[cls],
            "likely_cause": _first_error_line(t)}


def _first_error_line(trace: str) -> str:
    for line in reversed((trace or "").split("--- stdout")[0].splitlines()):
        s = line.strip()
        if s and not s.startswith(("File \"", "--- ", "exit code", "Traceback")):
            return s[:200]
    return (trace or "").strip()[:200] or "no trace captured"


def fingerprint(cls_info: dict, trace: str | None = None) -> str:
    """A stable identity for "this exact failure, again".

    Class alone is too coarse -- two unrelated api_misuse faults would collide
    and a genuine second problem would be dismissed as a repeat. The first error
    line is what actually distinguishes them, so the fingerprint is the pair.
    """
    line = _first_error_line(trace or "")
    # Numbers inside a message (indices, sizes, addresses) vary between runs of
    # the SAME fault, so they are normalised away before hashing.
    norm = re.sub(r"\d+", "#", line)
    return f"{cls_info.get('class', UNKNOWN)}::{norm[:160]}"


def repeat_count(fp: str, previous: list) -> int:
    """How many times this exact failure already happened in `previous`.

    `previous` is a list of (class, trace) pairs, oldest first.
    """
    n = 0
    for cls, trace in previous:
        if fingerprint({"class": cls}, trace) == fp:
            n += 1
    return n


def repair_brief(cls_info: dict, attempt: int, max_attempts: int,
                 repeats: int = 0) -> str:
    """The compact block handed to the model on a debug attempt.

    Deliberately much smaller than a planning prompt: repair needs the fault
    and the constraint, not the whole research state.
    """
    L = [f"## Repair attempt {attempt}/{max_attempts}",
         f"Failure class: {cls_info['class']}",
         f"Likely cause: {cls_info['likely_cause']}",
         f"Guidance: {cls_info['guidance']}"]
    if cls_info.get("needs_shrink"):
        L.append("This class is NOT fixed by retrying the same work -- the "
                 "experiment must be made materially cheaper or abandoned.")
    if not cls_info.get("retry_worthwhile"):
        L.append("This should NOT be retried as-is.")
    if repeats >= 1:
        # The run-3 pattern: crash, diagnose, apply a fix that does not address
        # the cause, crash identically. Saying so explicitly is the cheapest
        # intervention available, and it is information the model does not
        # otherwise have -- each repair prompt is built fresh.
        L.append(
            f"YOU HAVE ALREADY HIT THIS EXACT FAILURE {repeats} TIME(S) IN THIS "
            f"RUN. The previous repair did not address the cause, so repeating "
            f"that approach will fail again. Do not re-apply it. Either state a "
            f"DIFFERENT root cause and fix that, or abandon this implementation "
            f"path and achieve the same measurement a simpler way -- if the "
            f"capability you are reaching for is orchestration-only, the "
            f"capability contract names what to use instead.")
    L.append("Apply the SMALLEST change that addresses the fault. Do not "
             "redesign the experiment: the hypothesis has not been tested yet, "
             "and changing it now would waste the attempt already spent.")
    return "\n".join(L)


def as_knowledge(cls_info: dict, node_id: int, choices: dict) -> tuple:
    """Turn a failure into a recordable lesson (outcome, title, body) for the
    experience memory, so the same fault is not rediscovered later."""
    import json as _json
    if not cls_info["is_code_failure"]:
        return ("DEAD_END", f"node {node_id}: hypothesis not supported",
                f"menu_choices={_json.dumps(choices)} ran correctly but did not "
                f"support the hypothesis. This is a result, not a bug.")
    return ("CRASHED", f"node {node_id}: {cls_info['class']}",
            f"menu_choices={_json.dumps(choices)} failed with "
            f"{cls_info['class']}: {cls_info['likely_cause']} -- {cls_info['guidance']}")
