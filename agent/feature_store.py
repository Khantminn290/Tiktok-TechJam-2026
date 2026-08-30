"""Path B discoveries that accumulate instead of evaporating.

Path B -- the agent inventing and implementing its own feature -- is the most
genuinely research-like thing this system does. It was also, on inspection, a
dead end in the literal sense: a discovered feature went into `_pending_feature`
and was never read again. Two references in the whole repository, one to
initialise it and one to assign it. `feature_source` has never once appeared in
a journal across every recorded run, so the discovery-to-training path has never
executed end to end.

The only route a feature had into an experiment was the LLM noticing a line of
prompt text and re-typing the builder source into `menu_choices.feature_source`
by hand. That is not a pipeline; it is a hope.

What this module adds:

  * **Lineage.** A feature is stored with its exact source, a hash of that
    source, its mechanism, input columns, probe numbers, and the node it came
    from -- so the thing that gets retrained is provably the thing that was
    probed, not a paraphrase of it.
  * **A follow-up.** Clearing the probe automatically produces an executable
    paired ExperimentSpec: incumbent as control, incumbent + this exact feature
    as treatment, multi-seed. No LLM step between discovering a feature and
    measuring whether it survives.
  * **Not rediscovering it.** Features are keyed by a hash of their normalised
    source, so the same mechanism proposed under a new name is recognised.

A feature that improves one seed is NOT promoted. It becomes a follow-up
experiment, and the paired result decides.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
STORE = os.path.join(ROOT, "logs", "feature_store.jsonl")

from . import evidence as EV  # noqa: E402
from . import experiment_spec as XS  # noqa: E402


def source_sha(source: str) -> str:
    """Hash of the NORMALISED source, so cosmetic edits do not look novel.

    Comments, blank lines and indentation vary freely between two proposals of
    the same mechanism; the identifiers and structure do not.
    """
    s = str(source or "")
    s = re.sub(r"#.*", "", s)                     # comments
    s = re.sub(r"\s+", " ", s)                    # all whitespace runs
    return hashlib.sha256(s.strip().encode()).hexdigest()[:16]


def load(path: str = STORE) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def already_known(source: str, path: str = STORE) -> dict | None:
    """Has this exact mechanism been seen before, under any name?"""
    sha = source_sha(source)
    for e in load(path):
        if e.get("sha") == sha:
            return e
    return None


def record_discovery(feature: dict, probe: dict, node_id: int | None = None,
                     path: str = STORE) -> dict:
    """Persist everything needed to retrain this feature identically later."""
    src = str(feature.get("source") or "")
    entry = {
        "sha": source_sha(src),
        "name": feature.get("name"),
        "mechanism": feature.get("mechanism"),
        "hypothesis": feature.get("hypothesis"),
        "source": src,
        "source_columns": feature.get("source_columns"),
        "leakage_rationale": feature.get("leakage_check"),
        "temporal_cutoff": feature.get("temporal_cutoff"),
        "probe": {k: probe.get(k) for k in
                  ("status", "reason", "best_incremental_sigma", "per_feature")},
        "parent_node": node_id,
        "evidence_tier": EV.PROBED,
        # Filled in later, by a paired experiment. Left explicitly null so the
        # difference between "probed" and "measured in training" is visible.
        "training": None,
        "gauc_change": None, "ndcg_change": None, "primary_change": None,
        "seed_stability": None, "runtime_cost_s": None,
        "compatible_configs": [], "suggested_generalizations":
            feature.get("suggested_generalizations") or [],
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry


def update_outcome(sha: str, paired: dict, evidence: dict,
                   runtime_s: float | None = None, config: dict | None = None,
                   path: str = STORE) -> dict | None:
    """Attach the measured training outcome to a stored feature."""
    rows = load(path)
    hit = None
    for e in rows:
        if e.get("sha") == sha:
            hit = e
            e["training"] = paired
            e["evidence_tier"] = evidence.get("state", EV.UNTESTED)
            if paired.get("usable"):
                e["primary_change"] = paired.get("delta")
                e["seed_stability"] = paired.get("sd")
            e["runtime_cost_s"] = runtime_s
            if config and evidence.get("state") == EV.CONFIRMED:
                e["compatible_configs"] = (e.get("compatible_configs") or []) + [config]
    if hit is None:
        return None
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for e in rows:
            fh.write(json.dumps(e, default=str) + "\n")
    os.replace(tmp, path)
    return hit


def followup_spec(entry: dict, control: dict, seeds=(0, 1, 2),
                  timeout_s: int = 1800) -> XS.ExperimentSpec:
    """The experiment a cleared probe should automatically produce.

    Control is the incumbent; treatment is the incumbent plus THIS EXACT
    source. One variable, paired by seed, so the result is attributable to the
    feature and nothing else.
    """
    treatment = dict(control)
    treatment["feature_source"] = entry["source"]
    sigma = (entry.get("probe") or {}).get("best_incremental_sigma") or 0.0
    return XS.ExperimentSpec(
        hypothesis=(f"Feature {entry.get('name')!r} cleared the probe at "
                    f"{sigma} sigma incremental. Does it survive training, "
                    f"paired against the incumbent?"),
        experiment_type=XS.PATH_B_CONFIRMATION,
        control=control, treatment=treatment, seeds=seeds,
        parent_node=entry.get("parent_node"),
        expected_primary_effect=float(sigma) * EV.NOISE,
        feature_lineage={"name": entry.get("name"), "sha": entry["sha"],
                         "mechanism": entry.get("mechanism")},
        runtime_budget_s=timeout_s,
        notes="auto-generated from a cleared feature probe; the treatment "
              "carries the exact stored source, not a re-derivation")


def variations(entry: dict, control: dict, seeds=(0, 1, 2)) -> list:
    """Follow-ups that vary ONE thing about a confirmed feature.

    Only offered once a feature is CONFIRMED -- generating a family of
    follow-ups around a single-seed result is how a search wastes its budget on
    noise.
    """
    if entry.get("evidence_tier") != EV.CONFIRMED:
        return []
    out = []
    for gen in (entry.get("suggested_generalizations") or [])[:3]:
        t = dict(control)
        t["feature_source"] = entry["source"]
        out.append(XS.ExperimentSpec(
            hypothesis=f"Generalisation of confirmed feature "
                       f"{entry.get('name')!r}: {gen}",
            experiment_type=XS.IMPROVEMENT,
            control=control, treatment=t, seeds=seeds,
            feature_lineage={"name": entry.get("name"), "sha": entry["sha"]},
            notes=str(gen)))
    return out


def render_for_prompt(path: str = STORE, limit: int = 6) -> str:
    rows = load(path)
    if not rows:
        return ""
    L = ["## FEATURES YOU HAVE ALREADY BUILT",
         "Each was implemented and probed. The source is stored, so a follow-up "
         "reuses the exact implementation -- you do not need to retype it, and "
         "proposing the same mechanism under a new name will be recognised."]
    for e in rows[-limit:]:
        tier = e.get("evidence_tier")
        delta = e.get("primary_change")
        L.append(f"\n- {e.get('name')} [{tier}] sha={e['sha']}")
        L.append(f"    mechanism: {str(e.get('mechanism'))[:140]}")
        probe = e.get("probe") or {}
        L.append(f"    probe: {probe.get('status')} "
                 f"({probe.get('best_incremental_sigma')} sigma incremental)")
        if delta is not None:
            L.append(f"    trained: {delta:+.5f} primary, paired")
        elif tier == EV.PROBED:
            L.append("    trained: NOT YET — a paired confirmation is queued")
    return "\n".join(L)


if __name__ == "__main__":
    rows = load()
    print(f"{len(rows)} stored feature(s)")
    for e in rows:
        print(f"  {e['sha']}  {e.get('name')}  {e.get('evidence_tier')}  "
              f"primary {e.get('primary_change')}")
