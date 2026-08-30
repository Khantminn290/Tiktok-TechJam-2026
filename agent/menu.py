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
    # Keys that are NOT menu axes but are legitimate parts of an experiment.
    # feature_source carries an agent-written build_features() -- it is code,
    # not an option, so it is validated by rule rather than by membership.
    PASSTHROUGH_KEYS = ("feature_source",)
    # Pipeline overrides the agent may set directly. Each is a part of the
    # pipeline no menu axis can reach, and each was investigated in the Opus
    # research run. Bounded so a typo cannot silently produce a 500-epoch run.
    PIPELINE_OVERRIDES = {
        "k": (int, 4, 128), "lr": (float, 1e-5, 1e-2),
        "epochs": (int, 1, 120), "patience": (int, 1, 120),
        "l2": (float, 0.0, 1e-2), "bs": (int, 256, 65536),
        "hist_tau_days": (float, 0.25, 60.0),
        "aux_weight": (float, 0.0, 2.0),
        "n_checkpoints": (int, 0, 20),
        "checkpoint_combine": (bool, 0, 1),
        # legacy names, still accepted so older journals replay unchanged
        "snapshot_ensemble": (int, 0, 20),
        "snapshot_force": (bool, 0, 1),
    }

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
            if axis in self.PASSTHROUGH_KEYS:
                continue
            if axis in self.PIPELINE_OVERRIDES:
                typ, lo, hi = self.PIPELINE_OVERRIDES[axis]
                v = choices[axis]
                try:
                    v = bool(v) if typ is bool else typ(v)
                except (TypeError, ValueError):
                    problems.append(f"pipeline override '{axis}' must be "
                                    f"{typ.__name__}, got {choices[axis]!r}")
                    continue
                if typ is not bool and not (lo <= v <= hi):
                    problems.append(f"pipeline override '{axis}'={v} is outside "
                                    f"the allowed range [{lo}, {hi}]")
                    continue
                normalized[axis] = v
                continue
            if axis not in self.axes:
                problems.append(f"unknown axis '{axis}' (valid axes: {list(self.axes)})")
        # An agent-written feature builder travels inside menu_choices, so the
        # executor's leakage gate -- which scans the SCRIPT -- never sees it.
        # Gate it here, at the only point it can enter an experiment.
        fsrc = choices.get("feature_source")
        if fsrc:
            from .feature_lab import label_leak_findings
            leaks = label_leak_findings(str(fsrc))
            if leaks:
                problems.append("feature_source rejected: " + leaks[0])
            elif "build_features" not in str(fsrc):
                problems.append("feature_source must define "
                                "build_features(splits, meta)")
            else:
                normalized["feature_source"] = str(fsrc)
        # cross-axis constraints: option-level "requires": {other_axis: [allowed…]}
        if not problems:
            for axis, opt in normalized.items():
                if axis in self.PASSTHROUGH_KEYS or axis in self.PIPELINE_OVERRIDES:
                    continue          # not an option -- no cross-axis rules
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

    # Each dead end must survive into the prompt -- dropping one lets the agent
    # re-derive a known null -- but they accumulate without bound. At 26 entries
    # they reached 13.5k characters, comparable to the whole menu, crowding out
    # the reasoning they exist to inform. The CLAIM is what stops a repeat; the
    # supporting numbers live in config/modification_menu.json for anyone who
    # needs them.
    DEAD_END_HEAD = 240

    def render_dead_ends(self, compact: bool = True) -> str:
        """Measured dead ends. Compact keeps every entry but only its claim."""
        d = self.raw.get("notes", {}).get("tested_dead_ends", [])
        if not d:
            return ""
        items = []
        for x in d:
            x = " ".join(x.split())
            if compact and len(x) > self.DEAD_END_HEAD:
                cut = x.rfind(". ", 0, self.DEAD_END_HEAD)
                x = (x[:cut + 1] if cut > 80 else x[:self.DEAD_END_HEAD].rstrip() + " ...")
            items.append(f"- {x}")
        note = ("" if not compact else
                "\n(claims only; full measurements are in "
                "config/modification_menu.json)")
        return ("### Measured dead ends (do NOT respend iterations here):"
                + note + "\n" + "\n".join(items))


def load_agent_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}
