"""Compact, curated experience memory -- distinct from logs/journal.jsonl.

journal.jsonl is the complete, append-only, per-iteration record (the required
run-log deliverable; every node, never pruned). agent/experience.md is the
opposite: a short, hand-sized digest of *lessons*, not events -- "menu option
X crashed with error class Y" or "loss=bpr_pairwise reliably helped" -- so a
future hypothesis-generation prompt doesn't have to re-read (or re-discover)
the entire journal to avoid re-trying a dead end the run already paid for.

Format is Markdown, not JSON: this file's only consumer is a text prompt (see
agent/prompts.py), so keeping it as prose from the start avoids a
serialize-then-render step, and it matches the existing convention in
config/modification_menu.md, which records the organizers' own dead-ends in
the same prose-digest style for the same reason (fed into a prompt via
Menu.render_for_prompt()).

It is capped to a fixed character budget: appending an entry that would push
the file over the cap drops the OLDEST whole entries first (never truncates
one mid-sentence), so what remains is always well-formed prose an LLM can
read directly, and the most useful information (title + outcome tag) is never
missing its own body.

This file is written by the harness (agent/loop.py calls append_entry() once
wired in), never by generated code -- it belongs on Phase 1's protected-files
list (agent/executor.py PROTECTED_PATHS).
"""
from __future__ import annotations

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIENCE_PATH = os.path.join(_HERE, "experience.md")
DEFAULT_CHAR_BUDGET = 8000

HEADER = (
    "<!-- Curated experience memory: lessons, not events. Auto-compacted to a "
    "fixed character budget -- oldest whole entries are dropped first, never "
    "truncated mid-entry. Distinct from logs/journal.jsonl (the complete, "
    "unpruned run log). Written by the harness only; generated code cannot "
    "write here (see agent/executor.py PROTECTED_PATHS). -->\n\n"
)

_ENTRY_PREFIX = "### "
VALID_OUTCOMES = ("HELPED", "DEAD_END", "CRASHED", "NEUTRAL", "CORRECTION")

# Zero-width split, right before any line starting with "### ". Deliberately
# NOT a plain string.split() on "\n### ": that consumes the newline into the
# delimiter, so a header-only compaction round (all entries dropped) silently
# loses its trailing blank line -- and the NEXT append then glues onto a
# boundary that no longer matches "\n### " at all, permanently disabling
# compaction from that point on. A lookahead split consumes no characters, so
# the boundary can't erode no matter how many compaction rounds run.
_ENTRY_START_RE = re.compile(r"(?m)^(?=" + re.escape(_ENTRY_PREFIX) + r")")


def _split_entries(text: str) -> tuple[str, list[str]]:
    """Returns (header, entries); each entry string starts with '### ' and
    runs up to (not including) the next entry. Splitting never consumes a
    character, so header + "".join(entries) always reconstructs text exactly.
    """
    parts = _ENTRY_START_RE.split(text)
    return parts[0], parts[1:]


def format_entry(iteration_id: int, outcome: str, title: str, body: str) -> str:
    outcome = outcome.strip().upper()
    return (f"{_ENTRY_PREFIX}[{outcome}] iter {iteration_id} -- {title.strip()}\n"
           f"{body.strip()}\n\n")


def compact(text: str, char_budget: int = DEFAULT_CHAR_BUDGET) -> str:
    """Drops the OLDEST whole entries (from the front) until the text fits the
    budget. Never cuts an entry in half, and never drops the single newest
    entry even if it alone exceeds the budget -- the budget is best-effort in
    that unavoidable case, but silently ending up with NO memory right after
    recording one would be a worse failure than a one-time overshoot. The
    header is kept as-is regardless (nothing sane to drop there either).
    """
    header, entries = _split_entries(text)
    while len(entries) > 1 and len(header) + sum(len(e) for e in entries) > char_budget:
        entries.pop(0)
    return header + "".join(entries)


def append_entry(iteration_id: int, outcome: str, title: str, body: str,
                 path: str = EXPERIENCE_PATH, char_budget: int = DEFAULT_CHAR_BUDGET) -> None:
    text = HEADER
    if os.path.exists(path):
        with open(path) as fh:
            text = fh.read()
    text += format_entry(iteration_id, outcome, title, body)
    text = compact(text, char_budget)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w") as fh:
            fh.write(text)
    except OSError as e:
        # Losing a note is not a reason to lose the run. This file is a
        # convenience -- the journal is the authoritative record, and every
        # lesson here is derived from it. An earlier run died outright with
        # PermissionError here because an interrupted experiment had left the
        # file read-only, throwing away five usable iterations to protect a
        # cache of something already stored elsewhere.
        print(f"  [experience] WARNING: could not write {path} "
              f"({type(e).__name__}: {e}); continuing — the journal still has "
              f"this iteration", flush=True)


def render_for_prompt(path: str = EXPERIENCE_PATH) -> str:
    """What gets pasted into the hypothesis-generation prompt. Already within
    budget by construction (append_entry compacts on write), so this just
    reads it back -- no further trimming happens here.
    """
    if not os.path.exists(path):
        return "(no prior experience recorded yet)"
    with open(path) as fh:
        text = fh.read()
    _, entries = _split_entries(text)
    if not entries:
        return "(no prior experience recorded yet)"
    return "".join(entries).strip()
