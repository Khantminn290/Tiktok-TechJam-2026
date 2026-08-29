"""Static temporal-leakage checker for generated experiment code.

The Phase 1 sandbox stops a script READING test labels: the test split has its
outcome columns physically stripped and the real data directory is unreadable
during execution. That is necessary and it is not sufficient.

It does not stop a script computing a feature from labels it IS allowed to
see, in a way that embeds information from the future of the row being
scored. The classic form:

    df.groupby("user_id")["long_view"].mean()

computed over ALL rows and then joined back is target leakage: every row's
feature contains that row's own label. It trains beautifully and the
validation score is meaningless. Nothing in the pipeline previously caught it.

This matters much more now that Path B makes custom feature code common and
the agent is explicitly encouraged toward historical/statistical features.

Design: conservative AST-based static analysis. It flags patterns for review
rather than trying to prove correctness -- a checker that blocks legitimate
work is worse than useless, so findings carry a severity and only the
clearest violations are fatal. Analysis is purely syntactic and never
executes the code.
"""
from __future__ import annotations

import ast
import re

# Columns that are OUTCOMES of an impression. Aggregating these across rows and
# feeding the result back as an input is the leak we are hunting.
TARGET_COLUMNS = ("long_view", "is_click", "is_like", "is_forward",
                  "is_follow", "is_comment", "is_hate", "play_time_ms")
AGGREGATIONS = ("mean", "sum", "count", "agg", "transform", "apply", "median",
                "std", "var", "cumsum", "expanding", "rolling", "bincount",
                "add", "average")
# Names that indicate the author was aware of causality.
SAFE_MARKERS = ("leave_one_out", "leaveoneout", "loo", "shift", "expanding",
                "causal", "prior", "past", "history", "before", "cutoff",
                "train_only", "train_mask", "exclude_self")

FATAL = "fatal"
WARN = "warn"
INFO = "info"


class Finding:
    def __init__(self, severity, line, message, snippet=""):
        self.severity, self.line = severity, line
        self.message, self.snippet = message, snippet

    def __repr__(self):
        return f"[{self.severity}] line {self.line}: {self.message}"

    def as_dict(self):
        return {"severity": self.severity, "line": self.line,
                "message": self.message, "snippet": self.snippet[:160]}


def _src_line(src_lines, node):
    i = getattr(node, "lineno", 1) - 1
    return src_lines[i].strip() if 0 <= i < len(src_lines) else ""


def _mentions_target(text: str) -> list:
    return [c for c in TARGET_COLUMNS if c in text]


def _has_safe_marker(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in SAFE_MARKERS)


def check_source(src: str) -> list:
    """Return a list of Findings. Never raises on bad input."""
    findings = []
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [Finding(INFO, e.lineno or 0,
                        "could not parse for leakage analysis (syntax error); "
                        "the executor will report the syntax fault itself")]

    for node in ast.walk(tree):
        # --- pattern 1: aggregation over a target column ------------------
        if isinstance(node, ast.Call):
            fname = ""
            if isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            elif isinstance(node.func, ast.Name):
                fname = node.func.id
            if fname in AGGREGATIONS:
                ctx = _src_line(lines, node)
                # widen context: aggregation and column often span lines
                start = max(0, (node.lineno or 1) - 3)
                window = "\n".join(lines[start:(node.lineno or 1) + 1])
                tgts = _mentions_target(window)
                if tgts:
                    if _has_safe_marker(window):
                        findings.append(Finding(
                            INFO, node.lineno,
                            f"aggregation '{fname}' over target column "
                            f"{tgts[0]!r}, but causality markers are present "
                            f"-- verify the cutoff is genuinely applied", ctx))
                    else:
                        findings.append(Finding(
                            WARN, node.lineno,
                            f"aggregation '{fname}' over target column "
                            f"{tgts[0]!r} with no visible causal guard. If this "
                            f"feature is joined back onto the SAME rows it "
                            f"aggregates, every row contains its own label. Use "
                            f"a strictly-earlier cutoff or leave-one-out.", ctx))

        # --- pattern 2: target column used to build a feature array -------
        if isinstance(node, ast.Assign):
            ctx = _src_line(lines, node)
            low = ctx.lower()
            tgts = _mentions_target(ctx)
            if tgts and re.search(r"feat|x_|X\[|cols|column|encode|input", low):
                if not _has_safe_marker(ctx):
                    findings.append(Finding(
                        WARN, node.lineno,
                        f"target column {tgts[0]!r} appears in what looks like "
                        f"FEATURE construction. Outcome columns are valid "
                        f"training TARGETS but never model inputs -- they do "
                        f"not exist at ranking time.", ctx))

        # --- pattern 3: fitting statistics on the evaluation split --------
        if isinstance(node, ast.Subscript):
            ctx = _src_line(lines, node)
            if re.search(r"\[(['\"])(valid|test)\1\]", ctx) and _mentions_target(ctx):
                sev = FATAL if "test" in ctx else WARN
                findings.append(Finding(
                    sev, node.lineno,
                    "labels of an EVALUATION split are referenced. Validation "
                    "labels may be used ONLY to score/early-stop, never to "
                    "build features; test labels may never be touched at all.",
                    ctx))
    return findings


def verdict(findings: list) -> dict:
    """Aggregate to a decision. Only clear violations block execution."""
    fatal = [f for f in findings if f.severity == FATAL]
    warn = [f for f in findings if f.severity == WARN]
    return {"block": bool(fatal),
            "n_fatal": len(fatal), "n_warn": len(warn),
            "findings": [f.as_dict() for f in findings]}


def render_for_agent(v: dict) -> str:
    """Structured explanation handed back so the agent can fix the feature
    rather than merely being told 'no'."""
    if not v["findings"]:
        return ""
    L = ["## Leakage review of your code"]
    if v["block"]:
        L.append("BLOCKED: this experiment was NOT run. Fix the issues below "
                 "and resubmit; the hypothesis has not been tested yet.")
    else:
        L.append("Advisory only -- the experiment ran. Confirm these are "
                 "intentional, because a leaked feature produces a validation "
                 "score that means nothing.")
    for f in v["findings"][:6]:
        L.append(f"- [{f['severity']}] line {f['line']}: {f['message']}")
        if f["snippet"]:
            L.append(f"    {f['snippet']}")
    L.append("Rule: a feature for a row at time t may use only information "
             "available BEFORE t. Outcome columns (long_view, is_click, ...) "
             "are legitimate training targets but never model inputs.")
    return "\n".join(L)


def check_file(path: str) -> dict:
    try:
        with open(path) as fh:
            return verdict(check_source(fh.read()))
    except OSError as e:
        return {"block": False, "n_fatal": 0, "n_warn": 0,
                "findings": [{"severity": INFO, "line": 0,
                              "message": f"unreadable: {e}", "snippet": ""}]}
