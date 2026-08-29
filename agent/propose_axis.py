"""Agent-proposed menu axes -- proposed autonomously, approved by a human.

Every axis in config/modification_menu.json was human-authored. This lets the
agent propose a genuinely NEW axis from what it observed via the data tools,
rather than only picking among options a human already wrote.

WHY GATED (the agent proposes; a human approves before it is selectable):
  * modification_menu.json is the highest-leverage file in the repo -- what is
    not in it is invisible to the search, and what IS in it enters EVERY
    prompt. A malformed or unfounded axis silently degrades all future
    iterations, not just one node.
  * An unfounded axis that gets tested becomes an entry in tested_dead_ends,
    i.e. fabricated evidence future runs would trust.
  * menu.py's safety gate ("locked" leakage-sensitive options) assumes
    human-authored options; letting the agent write options would let it
    author its way around that gate.
So: proposals are appended here as PENDING. Nothing is added to the live menu
until a human runs `--approve`. Approval is a separate, deliberate act, and
every proposal (approved or not) stays on the record.

Usage:
    python3 -m agent.propose_axis --list
    python3 -m agent.propose_axis --show 3
    python3 -m agent.propose_axis --approve 3
    python3 -m agent.propose_axis --reject 3 --note "unfounded citation"
"""
from __future__ import annotations

import argparse
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
PROPOSALS = os.path.join(ROOT, "logs", "proposed_axes.jsonl")
MENU = os.path.join(ROOT, "config", "modification_menu.json")

REQUIRED = ("axis_name", "description", "options", "mechanism",
            "citation", "signal_breadth")
VALID_BREADTH = ("broad", "concentrated")


class ProposalError(ValueError):
    """Proposal is malformed. Message is LLM-readable."""


def validate(p: dict) -> dict:
    """Structural validation only -- it does NOT judge whether the idea is any
    good. That judgement is exactly what the human approval step is for."""
    if not isinstance(p, dict):
        raise ProposalError("proposal must be an object")
    for k in REQUIRED:
        if k not in p:
            raise ProposalError(f"missing required field '{k}'")
    name = str(p["axis_name"]).strip()
    if not name.replace("_", "").isalnum() or not name.islower():
        raise ProposalError("axis_name must be lower_snake_case alphanumeric")
    menu = json.load(open(MENU))
    if name in menu["axes"]:
        raise ProposalError(f"axis '{name}' already exists in the menu")
    opts = p["options"]
    if not isinstance(opts, dict) or len(opts) < 2:
        raise ProposalError("options must be an object with at least 2 entries "
                            "(a baseline/no-op plus at least one alternative)")
    for o, spec in opts.items():
        if not isinstance(spec, dict) or not str(spec.get("description", "")).strip():
            raise ProposalError(f"option '{o}' needs a non-empty description")
    if str(p["signal_breadth"]).lower() not in VALID_BREADTH:
        raise ProposalError(f"signal_breadth must be one of {VALID_BREADTH} -- "
                            "this project measured that CONCENTRATING the "
                            "training signal loses repeatedly, so a proposal "
                            "must state which kind it is")
    for k in ("mechanism", "citation"):
        if len(str(p[k]).strip()) < 30:
            raise ProposalError(f"'{k}' must be specific (>=30 chars)")
    return p


def append_proposal(p: dict, iteration_id=None, path: str = PROPOSALS) -> int:
    validate(p)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    if os.path.exists(path):
        with open(path) as fh:
            n = sum(1 for ln in fh if ln.strip())
    rec = {"id": n, "status": "pending", "iteration_id": iteration_id,
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "proposal": p}
    with open(path, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return n


def load_all(path: str = PROPOSALS) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _rewrite(recs, path):
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def approve(pid: int, path: str = PROPOSALS, menu_path: str = MENU) -> dict:
    """HUMAN-ONLY. Promotes a pending proposal into the live menu."""
    recs = load_all(path)
    match = [r for r in recs if r["id"] == pid]
    if not match:
        raise ProposalError(f"no proposal with id {pid}")
    rec = match[0]
    if rec["status"] != "pending":
        raise ProposalError(f"proposal {pid} is already {rec['status']}")
    p = validate(rec["proposal"])
    menu = json.load(open(menu_path))
    prio = max(s.get("priority", 0) for s in menu["axes"].values()) + 1
    menu["axes"][p["axis_name"]] = {
        "priority": prio,
        "description": (f"{p['description']} MECHANISM: {p['mechanism']} "
                        f"REFS: {p['citation']} "
                        f"SIGNAL BREADTH (self-assessed): {p['signal_breadth']}. "
                        f"[AGENT-PROPOSED, human-approved]"),
        "options": p["options"],
    }
    with open(menu_path, "w") as fh:
        json.dump(menu, fh, indent=2, ensure_ascii=False)
    rec["status"] = "approved"
    rec["approved_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _rewrite(recs, path)
    return rec


def reject(pid: int, note: str = "", path: str = PROPOSALS) -> dict:
    recs = load_all(path)
    match = [r for r in recs if r["id"] == pid]
    if not match:
        raise ProposalError(f"no proposal with id {pid}")
    match[0]["status"] = "rejected"
    match[0]["note"] = note
    _rewrite(recs, path)
    return match[0]


def render_for_prompt(path: str = PROPOSALS) -> str:
    """Show the agent what it already proposed, so it neither repeats a pending
    idea nor re-proposes something a human rejected."""
    recs = load_all(path)
    if not recs:
        return ""
    lines = ["## Axes you have already proposed (do not repeat these)"]
    for r in recs:
        p = r["proposal"]
        lines.append(f"- [{r['status']}] {p['axis_name']}: "
                     f"{str(p['description'])[:110]}"
                     + (f"  (rejected: {r.get('note','')})" if r["status"] == "rejected" else ""))
    lines.append("A PENDING proposal is not selectable yet -- a human must "
                 "approve it first. Do not assume it is available.")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--show", type=int)
    ap.add_argument("--approve", type=int)
    ap.add_argument("--reject", type=int)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    if a.approve is not None:
        r = approve(a.approve)
        print(f"APPROVED proposal {a.approve}: axis '{r['proposal']['axis_name']}' "
              f"is now live in the menu.")
    elif a.reject is not None:
        reject(a.reject, a.note)
        print(f"rejected proposal {a.reject}")
    elif a.show is not None:
        m = [r for r in load_all() if r["id"] == a.show]
        print(json.dumps(m[0] if m else {"error": "not found"}, indent=2, ensure_ascii=False))
    else:
        recs = load_all()
        if not recs:
            print("no axis proposals yet")
        for r in recs:
            print(f"[{r['id']}] {r['status']:8s} {r['proposal']['axis_name']:24s} "
                  f"{str(r['proposal']['description'])[:70]}")
