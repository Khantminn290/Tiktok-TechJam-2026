"""Modification menu: loading, cross-axis validity checks, and the hard safety gate.

The safety gate is mechanical, not documentation: `validate_choices()` rejects any
option flagged `"locked": true` in modification_menu.json unless a human has set
`"allow_locked_options": true` in config/agent_config.json. The agent's decision
step can therefore never select a locked option on its own.
"""
from __future__ import annotations

import json
import os


class MenuError(ValueError):
    """A proposed menu_choices dict violates the menu. Message is LLM-readable."""


class Menu:
    def __init__(self, menu_path: str, allow_locked_options: bool = False):
        with open(menu_path) as fh:
            self.raw = json.load(fh)
        self.axes: dict = self.raw["axes"]
        self.allow_locked_options = allow_locked_options

    # ---------- queries ----------
    def axis_names(self) -> list[str]:
        return list(self.axes.keys())

    def options(self, axis: str) -> dict:
        return self.axes[axis]["options"]

    def priority_order(self) -> list[str]:
        return sorted(self.axes, key=lambda a: self.axes[a].get("priority", 99))

    def is_locked(self, axis: str, option: str) -> bool:
        return bool(self.options(axis).get(option, {}).get("locked", False))

    def selectable_options(self, axis: str) -> list[str]:
        return [o for o in self.options(axis)
                if self.allow_locked_options or not self.is_locked(axis, o)]

    def default_choices(self) -> dict:
        """The official-baseline configuration (first option of every axis)."""
        return {axis: next(iter(self.options(axis))) for axis in self.axes}

    # ---------- validation (call BEFORE any choice is executed) ----------
    def validate_choices(self, choices: dict) -> dict:
        """Returns normalized choices or raises MenuError with a readable message."""
        if not isinstance(choices, dict):
            raise MenuError(f"menu_choices must be an object, got {type(choices).__name__}")
        problems = []
        normalized = {}
        for axis in self.axes:
            if axis not in choices:
                problems.append(f"missing axis '{axis}' (pick one of {list(self.options(axis))})")
                continue
            opt = choices[axis]
            if opt not in self.options(axis):
                problems.append(f"axis '{axis}': unknown option '{opt}' "
                                f"(valid: {list(self.options(axis))})")
                continue
            if self.is_locked(axis, opt) and not self.allow_locked_options:
                problems.append(
                    f"axis '{axis}': option '{opt}' is LOCKED by the safety gate "
                    f"(leakage-sensitive). It cannot be selected without a human setting "
                    f"allow_locked_options in the agent config. Choose another option.")
                continue
            normalized[axis] = opt
        for axis in choices:
            if axis not in self.axes:
                problems.append(f"unknown axis '{axis}' (valid axes: {list(self.axes)})")
        # cross-axis constraints: option-level "requires": {other_axis: [allowed…]}
        if not problems:
            for axis, opt in normalized.items():
                req = self.options(axis)[opt].get("requires", {})
                for other_axis, allowed in req.items():
                    if normalized.get(other_axis) not in allowed:
                        problems.append(
                            f"'{axis}={opt}' requires {other_axis} in {allowed}, "
                            f"but got '{normalized.get(other_axis)}'")
        if problems:
            raise MenuError("invalid menu_choices: " + "; ".join(problems))
        return normalized

    # ---------- prompt rendering ----------
    def render_for_prompt(self) -> str:
        lines = []
        for axis in self.priority_order():
            spec = self.axes[axis]
            lines.append(f"### axis '{axis}' (priority {spec.get('priority')})")
            lines.append(spec.get("description", "").strip())
            for opt, ospec in spec["options"].items():
                if self.is_locked(axis, opt) and not self.allow_locked_options:
                    lines.append(f"- {opt} [LOCKED — NOT selectable: safety gate] "
                                 f"{ospec.get('description', '')}")
                    continue
                req = ospec.get("requires")
                req_s = f" [requires {req}]" if req else ""
                lines.append(f"- {opt}{req_s}: {ospec.get('description', '')}")
            lines.append("")
        notes = self.raw.get("notes", {})
        if notes.get("tested_dead_ends"):
            lines.append("### Measured dead-ends (organizers already tried these — do NOT respend iterations):")
            for d in notes["tested_dead_ends"]:
                lines.append(f"- {d}")
        return "\n".join(lines)


    def render_compact(self) -> str:
        """Axis/option index without the long descriptions. Used on exploration
        turns to stop the menu's sheer volume from dominating the prompt."""
        lines = []
        for axis in self.priority_order():
            opts = [o for o in self.options(axis)
                    if self.allow_locked_options or not self.is_locked(axis, o)]
            lines.append(f"- {axis} (priority {self.axes[axis].get('priority')}): "
                         + ", ".join(opts))
        return "\n".join(lines)

    def render_dead_ends(self) -> str:
        """The measured dead-ends, always sent in full regardless of menu
        compression -- dropping them would let the agent re-derive known nulls."""
        d = self.raw.get("notes", {}).get("tested_dead_ends", [])
        if not d:
            return ""
        return ("### Measured dead ends (do NOT respend iterations here):\n"
                + "\n".join(f"- {x}" for x in d))


def load_agent_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}
