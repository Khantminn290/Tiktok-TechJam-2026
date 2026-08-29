"""Shared data contracts: Node, ExperimentTree, journal persistence.

Every iteration of the outer loop produces exactly one Node. Nodes are persisted to
logs/journal.jsonl the moment they are created, so a crash never loses history —
that file IS the run-log deliverable (hypothesis, menu choices, metrics,
error/recovery events, tokens, wall-clock, one JSON object per line).
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Node:
    iteration_id: int
    parent_id: Optional[int]          # node this one branched from (None for drafts)
    action: str                       # "draft" | "debug" | "improve"
    menu_choices: dict                # one option id per menu axis
    hypothesis: str                   # what this iteration tries and why (LLM-authored)
    status: str                       # "success" | "error"
    metrics: Optional[dict]           # {"GAUC":…, "nDCG@5":…, "primary":…} on valid, or None
    error_trace: Optional[str]        # readable stderr/exception on failure, else None
    tokens_used: int                  # input+output tokens of the LLM call(s) for this node
    wall_clock_seconds: float         # training-run wall clock (subprocess)
    timestamp: float
    code_path: str                    # nodes/node_XXX/solution.py (full script)
    # --- extra bookkeeping (kept in the journal for the judges) ---
    expected_effect: str = ""         # LLM's stated expectation before running
    decide_reason: str = ""           # why decide_action picked this action/target
    token_breakdown: dict = field(default_factory=dict)
    events: list = field(default_factory=list)  # error/recovery events, validation retries…

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


class ExperimentTree:
    """All nodes of one run + best tracking. Journal-backed (append-only jsonl)."""

    def __init__(self, log_dir: str, nodes_dir: str | None = None,
                 project_root: str | None = None):
        self.log_dir = log_dir
        self.project_root = project_root or os.path.dirname(log_dir)
        self.nodes_dir = nodes_dir or os.path.join(log_dir, "nodes")
        self.journal_path = os.path.join(log_dir, "journal.jsonl")
        self.nodes: list[Node] = []
        self.best_node_id: Optional[int] = None
        self.corrupt_lines: list = []
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(self.nodes_dir, exist_ok=True)
        self._load_existing()

    # ---------- persistence ----------
    def _load_existing(self):
        """Rebuild the tree from the journal.

        Tolerates a truncated final line: a process killed mid-append leaves a
        partial record, and crashing on it would make a resumable run permanently
        unresumable. Unparseable lines are skipped and reported, never silently
        dropped, and the good records before them still load.
        """
        if not os.path.exists(self.journal_path):
            return
        known = {f.name for f in dataclasses.fields(Node)}
        skipped = []
        with open(self.journal_path) as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    self.nodes.append(Node(**{k: v for k, v in d.items() if k in known}))
                except (json.JSONDecodeError, TypeError) as e:
                    skipped.append((lineno, str(e)[:80]))
        if skipped:
            self.corrupt_lines = skipped
            print(f"WARNING: journal has {len(skipped)} unreadable line(s) — skipped, "
                  f"the run will continue from the last good node "
                  f"(likely a process killed mid-write): "
                  + "; ".join(f"line {n}: {m}" for n, m in skipped), flush=True)
        for n in self.nodes:
            self._maybe_update_best(n, persist_artifacts=False)

    def add(self, node: Node) -> None:
        """Persist one self-contained node record and append the run journal."""
        self.nodes.append(node)
        with open(self.journal_path, "a") as fh:
            fh.write(node.to_json() + "\n")
        node_dir = os.path.join(self.nodes_dir, f"node_{node.iteration_id:03d}")
        os.makedirs(node_dir, exist_ok=True)
        with open(os.path.join(node_dir, "record.json"), "w") as fh:
            json.dump(dataclasses.asdict(node), fh, indent=2, ensure_ascii=False)
        self._maybe_update_best(node, persist_artifacts=True)

    def _maybe_update_best(self, node: Node, persist_artifacts: bool):
        if node.status != "success" or not node.metrics:
            return
        cur = self.get(self.best_node_id) if self.best_node_id is not None else None
        if cur is None or node.metrics["primary"] > cur.metrics["primary"]:
            self.best_node_id = node.iteration_id
            if persist_artifacts:
                self._write_best_artifacts(node)

    def _write_best_artifacts(self, node: Node):
        """logs/best_solution.py + logs/best_metrics.json, updated when best changes."""
        source = self.resolve_code_path(node)
        if source and os.path.exists(source):
            shutil.copyfile(source, os.path.join(self.log_dir, "best_solution.py"))
        with open(os.path.join(self.log_dir, "best_metrics.json"), "w") as fh:
            json.dump({
                "iteration_id": node.iteration_id,
                "menu_choices": node.menu_choices,
                "hypothesis": node.hypothesis,
                "valid_metrics": node.metrics,
                "code_path": node.code_path,
                "timestamp": node.timestamp,
            }, fh, indent=2, ensure_ascii=False)

    # ---------- queries ----------
    def get(self, iteration_id: Optional[int]) -> Optional[Node]:
        for n in self.nodes:
            if n.iteration_id == iteration_id:
                return n
        return None

    def resolve_code_path(self, node: Node) -> str:
        if not node.code_path or os.path.isabs(node.code_path):
            return node.code_path
        return os.path.join(self.project_root, node.code_path)

    def best(self) -> Optional[Node]:
        return self.get(self.best_node_id)

    def buggy_leaves(self) -> list[Node]:
        """Error nodes that no later node has already tried to debug."""
        debugged = {n.parent_id for n in self.nodes if n.action == "debug"}
        return [n for n in self.nodes if n.status == "error" and n.iteration_id not in debugged]

    def recent(self, k: int) -> list[Node]:
        return self.nodes[-k:]

    def successes(self) -> list[Node]:
        return [n for n in self.nodes if n.status == "success" and n.metrics]

    def next_id(self) -> int:
        ids = [n.iteration_id for n in self.nodes]
        if os.path.isdir(self.nodes_dir):
            for name in os.listdir(self.nodes_dir):
                if name.startswith("node_") and name[5:].isdigit():
                    ids.append(int(name[5:]))
        return max(ids, default=-1) + 1

    def total_tokens(self) -> int:
        return sum(n.tokens_used for n in self.nodes)

    def total_wall_clock(self) -> float:
        return sum(n.wall_clock_seconds for n in self.nodes)

    def children_of(self, iteration_id: int) -> list[Node]:
        return [n for n in self.nodes if n.parent_id == iteration_id]


def now() -> float:
    return time.time()


def error_headline(trace: Optional[str], width: int = 160) -> str:
    """One-line summary of a failure for logs/prompts.

    Prefers the exception line from the stderr section; the naive "last line"
    would pick up trailing stdout (training progress) instead of the error.
    """
    if not trace:
        return "?"
    if trace.lstrip().startswith("TIMEOUT"):
        return trace.strip().splitlines()[0][:width]
    head = trace.split("--- stdout (tail) ---")[0].strip()
    for line in reversed(head.splitlines()):
        s = line.strip()
        if s and not s.startswith(("File \"", "--- ", "exit code")) \
                and not s.startswith("Traceback"):
            return s[:width]
    return head.splitlines()[-1][:width] if head else trace.strip()[:width]
