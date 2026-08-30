"""The agent's compact scientific memory: what currently MATTERS.

Division of responsibility (deliberately not another copy of the others):
    logs/journal.jsonl   complete historical record, append-only
    tested_dead_ends     curated negative findings, injected into every prompt
    experience.md        short prose lessons
    research_state       CURRENT scientific understanding + decision context

Everything here is DERIVED DETERMINISTICALLY from experiment records. The LLM
interprets this state; it never authors it. Anything requiring semantic
judgement is either omitted or explicitly marked with its provenance, because
a state the model can write into is a state that can hallucinate evidence and
then be trusted as though it were measured.

Evidence ladder (a single-seed win can NEVER skip to the top):
    observed_once  -- one run, one seed. Not knowledge.
    replicated     -- same configuration scored 2+ times independently.
    reseed_verified-- a multi-seed reseed exists for this configuration.
    validated      -- reseed-verified AND the effect exceeds the noise floor
                      against a named comparison.
"""
from __future__ import annotations

import json
import os
import statistics as st

BASELINE_VALID = 0.6016
BASELINE_SEED_STD = 0.0008          # the official baseline's own 5-seed std

OBSERVED_ONCE = "observed_once"
REPLICATED = "replicated"
RESEED_VERIFIED = "reseed_verified"
VALIDATED = "validated"

# Strength of the evidence FOR a claim, graded against the noise floor rather
# than by sign alone. The previous test was a bare `>` comparison, which reads
# a 0.0001 difference as support when the baseline's own seed-to-seed spread is
# 0.0008 -- the same mistake as trusting a single lucky run, one level down.
INCONCLUSIVE = "INCONCLUSIVE"    # inside the noise floor: says nothing either way
WEAK = "WEAK"                    # 1-2 sigma, single observation
MODERATE = "MODERATE"            # 2-3 sigma, or 1-2 sigma replicated
STRONG = "STRONG"                # >3 sigma, or reseed-verified
REJECTED = "REJECTED"            # evidence points the OTHER way, beyond noise


def grade_evidence(delta: float, n_runs: int = 1,
                   reseed_verified: bool = False) -> dict:
    """Grade one comparison by effect size in units of the noise floor.

    `delta` is (claim - alternative) on primary, so a negative delta beyond the
    noise floor is evidence AGAINST the claim, which is reported as REJECTED
    rather than quietly as "questionable". Distinguishing "no evidence" from
    "evidence against" matters: the first invites an experiment, the second
    forbids one.
    """
    sigma = abs(delta) / BASELINE_SEED_STD
    if sigma < 1.0:
        strength = INCONCLUSIVE
    elif delta < 0:
        strength = REJECTED
    elif reseed_verified:
        strength = STRONG
    elif sigma >= 3.0:
        strength = STRONG
    elif sigma >= 2.0:
        strength = MODERATE
    else:
        strength = MODERATE if n_runs > 1 else WEAK
    return {"strength": strength, "delta": round(delta, 5),
            "sigma": round(delta / BASELINE_SEED_STD, 2),
            "actionable": strength in (MODERATE, STRONG, REJECTED)}


def _sig(choices: dict) -> str:
    return json.dumps(choices or {}, sort_keys=True)


def _load_journal(log_dir: str) -> list:
    p = os.path.join(log_dir, "journal.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
    return out


def _load_json(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


class ResearchState:
    """Derived, not authored. Rebuilt from disk on every construction so it can
    never drift from the record or retain a stale claim."""

    def __init__(self, root: str):
        self.root = root
        log_dir = os.path.join(root, "logs")
        self.log_dir = log_dir
        self.nodes = _load_journal(log_dir)
        self.scored = [n for n in self.nodes
                       if n.get("status") == "success" and n.get("metrics")]
        self.reseed = _load_json(os.path.join(log_dir, "reseed_results.json"))
        self.best_meta = _load_json(os.path.join(log_dir, "best_metrics.json"))
        self.ensemble = _load_json(os.path.join(log_dir, "ensemble_results.json"))
        menu = _load_json(os.path.join(root, "config", "modification_menu.json")) or {}
        self.dead_ends = (menu.get("notes", {}) or {}).get("tested_dead_ends", [])
        self.axes = list((menu.get("axes") or {}).keys())
        self._build()

    # ---------------- derivation ----------------
    def _build(self):
        self.facts = self._facts()
        self.best_config = self._best_config()
        self.component_evidence = self._component_evidence()
        self.best_config_evidence = self._best_config_evidence()
        self.confirmed = self._confirmed_findings()
        self.promising = self._promising()
        self.branches = self._branches()
        self.integration_candidates = self._integration()
        self.open_questions = self._open_questions()

    def _facts(self) -> dict:
        f = {"official_baseline_valid_primary": BASELINE_VALID,
             "noise_floor_seed_std": BASELINE_SEED_STD,
             "scored_experiments": len(self.scored),
             "failed_experiments": len(self.nodes) - len(self.scored)}
        if self.scored:
            # OBSERVED MAXIMUM -- a single draw, deliberately kept distinct
            # from expected performance. Collapsing these is how a lucky seed
            # becomes a headline number.
            bm = max(self.scored, key=lambda n: n["metrics"]["primary"])
            f["best_observed_single_run"] = round(bm["metrics"]["primary"], 5)
            f["best_observed_node"] = bm["iteration_id"]
        if self.reseed and self.reseed.get("nodes"):
            top = max((n for n in self.reseed["nodes"] if n.get("mean_primary")),
                      key=lambda n: n["mean_primary"], default=None)
            if top:
                f["reseed_verified_mean"] = round(top["mean_primary"], 5)
                f["reseed_verified_std"] = round(top.get("std_primary") or 0.0, 5)
                f["reseed_n_seeds"] = top.get("n_samples")
                d = top["mean_primary"] - BASELINE_VALID
                f["reseed_delta_vs_baseline"] = round(d, 5)
                f["reseed_delta_in_sigma"] = round(d / BASELINE_SEED_STD, 2)
        if self.ensemble:
            # Support both the older "mean/std" ensemble summary and the
            # current recorded result emitted by agent.final_ensemble.py.
            if self.ensemble.get("primary") is not None:
                f["recorded_ensemble_primary"] = round(
                    float(self.ensemble["primary"]), 5)
                f["recorded_ensemble_gauc"] = round(
                    float(self.ensemble.get("GAUC") or 0.0), 5)
                f["recorded_ensemble_ndcg"] = round(
                    float(self.ensemble.get("nDCG@5") or 0.0), 5)
                f["ensemble_n_checkpoints"] = self.ensemble.get("k")
                f["ensemble_single_seed_mean"] = self.ensemble.get("single_seed_mean")
                f["ensemble_single_seed_std"] = self.ensemble.get("single_seed_std")
            else:
                f["expected_ensemble_mean"] = self.ensemble.get("mean")
                f["expected_ensemble_std"] = self.ensemble.get("std")
                f["ensemble_n_checkpoints"] = self.ensemble.get("k")
        return f

    def _best_config(self) -> dict:
        if self.best_meta and self.best_meta.get("menu_choices"):
            return dict(self.best_meta["menu_choices"])
        if self.scored:
            return dict(max(self.scored,
                            key=lambda n: n["metrics"]["primary"]).get("menu_choices") or {})
        return {}

    def _component_evidence(self) -> dict:
        """For each axis value in the CURRENT best config, is there isolated
        evidence for it?

        Isolated evidence = a scored node identical to best on every other
        axis, differing ONLY on this one. That is precisely the comparison a
        human failed to make for aux_click_like_forward, which was carried for
        an entire session untested and turned out to be harmful.

        Recomputed from the CURRENT best every time, so a component that
        leaves the best configuration stops being reported as an assumption.
        """
        best = self.best_config
        if not best:
            return {}
        best_score = None
        for n in self.scored:
            if _sig(n.get("menu_choices")) == _sig(best):
                best_score = n["metrics"]["primary"]
                break
        out = {}
        for axis, val in best.items():
            counterfactuals = []
            for n in self.scored:
                c = n.get("menu_choices") or {}
                if not c or set(c) != set(best):
                    continue
                diff = [a for a in best if c.get(a) != best.get(a)]
                if diff == [axis]:
                    counterfactuals.append((c[axis], n["metrics"]["primary"],
                                            n["iteration_id"]))
            if not counterfactuals:
                out[axis] = {"value": val, "status": "untested_assumption",
                             "evidence": "no experiment differs from the current "
                                         "best on this axis alone"}
            else:
                best_alt = max(counterfactuals, key=lambda t: t[1])
                helps = (best_score is not None and best_score > best_alt[1])
                g = (grade_evidence(best_score - best_alt[1],
                                    n_runs=len(counterfactuals))
                     if best_score is not None else
                     {"strength": INCONCLUSIVE, "delta": None, "sigma": None,
                      "actionable": False})
                # An ablation inside the noise floor is not support for the
                # component -- it is an untested assumption that happens to have
                # been probed once. Saying "supported" there is how a component
                # gets carried for a whole session on no evidence.
                if g["strength"] == INCONCLUSIVE:
                    status = "untested_assumption"
                elif g["strength"] == REJECTED:
                    status = "questionable"
                else:
                    status = "supported" if helps else "questionable"
                out[axis] = {
                    "value": val,
                    "status": status,
                    "strength": g["strength"],
                    "sigma": g["sigma"],
                    "evidence": (f"ablation vs {axis}={best_alt[0]} "
                                 f"(node {best_alt[2]}): "
                                 f"{best_alt[1]:.5f}"
                                 + (f" vs best {best_score:.5f}" if best_score else "")
                                 + (f" = {g['sigma']:+.1f} sigma ({g['strength']})"
                                    if g["sigma"] is not None else "")),
                    "level": OBSERVED_ONCE,
                }
        return out

    def _best_config_evidence(self) -> dict:
        """Evidence level of the CURRENT BEST configuration itself.

        Without this, a best config observed exactly once shows up only as a
        headline number with no confidence attached -- which is precisely how a
        lucky seed becomes 'the result'. The best config is excluded from the
        `promising` list (it is not a candidate, it is the incumbent), so it
        needs its own explicit status.
        """
        sig = _sig(self.best_config)
        if not self.best_config:
            return {"level": None, "n_runs": 0}
        runs = [n for n in self.scored if _sig(n.get("menu_choices")) == sig]
        reseeded = False
        if self.reseed:
            ids = {n["iteration_id"] for n in self.reseed.get("nodes", [])
                   if n.get("mean_primary")}
            reseeded = any(n["iteration_id"] in ids for n in runs)
        if reseeded:
            level = RESEED_VERIFIED
        elif len(runs) > 1:
            level = REPLICATED
        else:
            level = OBSERVED_ONCE
        return {"level": level, "n_runs": len(runs),
                "caveat": ("the incumbent itself rests on a SINGLE run -- it is "
                           "not yet knowledge, and confirming it may be worth "
                           "more than another exploration"
                           if level == OBSERVED_ONCE else "")}

    def _confirmed_findings(self) -> list:
        """Only reseed-backed effects reach here. A single-seed win cannot."""
        out = []
        if not self.reseed or not self.reseed.get("nodes"):
            return out
        ns = [n for n in self.reseed["nodes"] if n.get("mean_primary")]
        for n in ns:
            lvl = RESEED_VERIFIED
            d = n["mean_primary"] - BASELINE_VALID
            if abs(d) > 2 * BASELINE_SEED_STD:
                lvl = VALIDATED
            out.append({
                "claim": f"node {n['iteration_id']} configuration performs at "
                         f"{n['mean_primary']:.5f}",
                "evidence_type": "multi-seed reseed",
                "n_runs": n.get("n_samples"),
                "effect_vs_baseline": round(d, 5),
                "uncertainty_std": round(n.get("std_primary") or 0.0, 5),
                "sigma": round(d / BASELINE_SEED_STD, 2),
                "level": lvl,
            })
        return out

    def _promising(self) -> list:
        """Scored better than baseline but WITHOUT reseed backing -- explicitly
        not knowledge yet, and labelled with the action that would make it so."""
        reseeded = set()
        if self.reseed:
            reseeded = {n["iteration_id"] for n in self.reseed.get("nodes", [])}
        best_conf = _sig(self.best_config)
        seen, out = set(), []
        for n in sorted(self.scored, key=lambda n: -n["metrics"]["primary"]):
            if n["iteration_id"] in reseeded:
                continue
            s = _sig(n.get("menu_choices"))
            if s in seen or s == best_conf:
                continue
            seen.add(s)
            p = n["metrics"]["primary"]
            if p <= BASELINE_VALID:
                continue
            reps = sum(1 for m in self.scored if _sig(m.get("menu_choices")) == s)
            out.append({
                "node": n["iteration_id"],
                "observed_primary": round(p, 5),
                "level": REPLICATED if reps > 1 else OBSERVED_ONCE,
                "n_runs": reps,
                "next_action": "confirmation (reseed >=5 seeds) before it may be "
                               "treated as a finding or integrated",
            })
            if len(out) >= 4:
                break
        return out

    def _branches(self) -> list:
        """Conceptual branches = configuration families sharing (loss, model).
        The journal has nodes; it has no notion of the branch they belong to."""
        fam = {}
        for n in self.scored:
            c = n.get("menu_choices") or {}
            key = (c.get("loss", "?"), c.get("model", "?"))
            f = fam.setdefault(key, {"experiments": 0, "best": -1.0, "best_node": None})
            f["experiments"] += 1
            if n["metrics"]["primary"] > f["best"]:
                f["best"] = n["metrics"]["primary"]
                f["best_node"] = n["iteration_id"]
        reseeded = set()
        if self.reseed:
            reseeded = {n["iteration_id"] for n in self.reseed.get("nodes", [])}
        out = []
        for (loss, model), f in sorted(fam.items(), key=lambda kv: -kv[1]["best"]):
            if f["best_node"] in reseeded:
                status = "confirmed"
            elif f["best"] <= BASELINE_VALID:
                status = "dead-end (never beat baseline)"
            elif f["experiments"] >= 3 and f["best"] < (self.facts.get(
                    "best_observed_single_run", 1.0) - 2 * BASELINE_SEED_STD):
                status = "dead-end (explored, well below best)"
            else:
                status = "awaiting confirmation"
            out.append({"branch": f"{loss} + {model}", "experiments": f["experiments"],
                        "best_observed": round(f["best"], 5),
                        "best_node": f["best_node"], "status": status})
        return out

    def _integration(self) -> list:
        """Only combinations that are scientifically justified: two findings
        touching DIFFERENT axes, each with at least replication behind it."""
        cands = []
        confirmed_axes = {}
        for p in self.promising:
            node = next((n for n in self.scored if n["iteration_id"] == p["node"]), None)
            if not node:
                continue
            c = node.get("menu_choices") or {}
            diff = [a for a in self.best_config if c.get(a) != self.best_config.get(a)]
            if len(diff) == 1:
                confirmed_axes[diff[0]] = (p["node"], c[diff[0]], p["level"])
        axes = list(confirmed_axes)
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                a, b = axes[i], axes[j]
                cands.append({
                    "candidate": f"{a}={confirmed_axes[a][1]} + {b}={confirmed_axes[b][1]}",
                    "reason": "different axes, so the mechanisms are plausibly "
                              "independent",
                    "risk": "interaction may be negative; must be measured, not assumed",
                    "status": ("eligible" if all(confirmed_axes[x][2] != OBSERVED_ONCE
                                                 for x in (a, b))
                               else "blocked: both components need confirmation first"),
                })
        return cands[:3]

    def _open_questions(self) -> list:
        q = []
        for axis, ev in self.component_evidence.items():
            if ev["status"] == "untested_assumption":
                q.append(f"Does {axis}={ev['value']} contribute anything? It is in "
                         f"the current best but has never been isolated.")
        for p in self.promising:
            q.append(f"Is node {p['node']} ({p['observed_primary']}) robust across "
                     f"seeds, or a lucky draw? Currently {p['level']}.")
        for c in self.integration_candidates:
            if c["status"] == "eligible":
                q.append(f"Do these compose? {c['candidate']}")
        return q[:6]

    # ---------------- rendering ----------------
    def render(self, max_chars: int = 6000) -> str:
        """Compact notebook. Optimised for decision-relevant information per
        token -- the point is NOT to replace a 20k menu with a 20k notebook."""
        f = self.facts
        L = ["## RESEARCH STATE (derived from experiment records, not authored)"]
        L.append("### Benchmark facts")
        L.append(f"- official baseline: {f['official_baseline_valid_primary']} "
                 f"| noise floor (1 sigma): {f['noise_floor_seed_std']}")
        if "best_observed_single_run" in f:
            L.append(f"- best OBSERVED single run: {f['best_observed_single_run']} "
                     f"(node {f['best_observed_node']}) -- one draw, NOT an "
                     f"expected value")
        if "reseed_verified_mean" in f:
            L.append(f"- reseed-VERIFIED mean: {f['reseed_verified_mean']} +/- "
                     f"{f['reseed_verified_std']} over {f['reseed_n_seeds']} seeds "
                     f"= {f['reseed_delta_in_sigma']} sigma vs baseline")
        if "recorded_ensemble_primary" in f:
            extra = ""
            if f.get("ensemble_single_seed_mean") is not None:
                extra = (f" | member mean {f['ensemble_single_seed_mean']}"
                         + (f" +/- {f['ensemble_single_seed_std']}"
                            if f.get("ensemble_single_seed_std") is not None else ""))
            L.append(f"- recorded ensemble result: {f['recorded_ensemble_primary']} "
                     f"(GAUC {f['recorded_ensemble_gauc']}, nDCG@5 "
                     f"{f['recorded_ensemble_ndcg']}, k={f.get('ensemble_n_checkpoints')})"
                     f"{extra}")
        if "expected_ensemble_mean" in f:
            L.append(f"- expected ENSEMBLE: {f['expected_ensemble_mean']} +/- "
                     f"{f['expected_ensemble_std']} ({f['ensemble_n_checkpoints']} ckpts)")
        if "best_observed_single_run" not in f:
            L.append("- no observed single run is currently loaded from the journal, "
                     "so there is nothing to treat as an observed maximum. A "
                     "recorded ensemble headline is also NOT an expected value "
                     "for any one seed.")
        L.append(f"- experiments: {f['scored_experiments']} scored, "
                 f"{f['failed_experiments']} failed")

        if self.best_config:
            bce = self.best_config_evidence
            L.append(f"### Current best configuration [evidence: {bce['level']}, "
                     f"{bce['n_runs']} run(s)]")
            if bce.get("caveat"):
                L.append(f"  !! {bce['caveat']}")
            L.append("component evidence:")
            for axis, ev in self.component_evidence.items():
                mark = {"untested_assumption": "UNTESTED",
                        "supported": "supported",
                        "questionable": "QUESTIONABLE"}[ev["status"]]
                L.append(f"- {axis}={ev['value']}: {mark} ({ev['evidence']})")

        if self.confirmed:
            L.append("### Confirmed findings (reseed-backed only)")
            for c in self.confirmed[:4]:
                L.append(f"- [{c['level']}] {c['claim']} ({c['n_runs']} seeds, "
                         f"{c['sigma']} sigma vs baseline)")
        if self.promising:
            L.append("### Promising but UNCONFIRMED (not knowledge yet)")
            for p in self.promising:
                L.append(f"- node {p['node']}: {p['observed_primary']} "
                         f"[{p['level']}, {p['n_runs']} run(s)] -> {p['next_action']}")
        if self.branches:
            L.append("### Research branches")
            for b in self.branches[:5]:
                L.append(f"- {b['branch']}: {b['experiments']} exp, best "
                         f"{b['best_observed']} -- {b['status']}")
        if self.integration_candidates:
            L.append("### Integration candidates")
            for c in self.integration_candidates:
                L.append(f"- {c['candidate']} -- {c['status']} ({c['reason']})")
        L.append("### Open questions (highest-value uncertainty)")
        if self.open_questions:
            for q in self.open_questions:
                L.append(f"- {q}")
        else:
            L.append("- none derived from the current on-disk journal; load or "
                     "rebuild experiment records before trusting the absence of "
                     "questions")
        L.append(f"### Negative findings: {len(self.dead_ends)} recorded dead ends "
                 f"are supplied separately in full -- do not re-derive them.")
        out = "\n".join(L)
        return out if len(out) <= max_chars else out[:max_chars] + "\n...[truncated]"

    def as_dict(self) -> dict:
        return {"facts": self.facts, "best_config": self.best_config,
                "component_evidence": self.component_evidence,
                "confirmed": self.confirmed, "promising": self.promising,
                "branches": self.branches,
                "integration_candidates": self.integration_candidates,
                "open_questions": self.open_questions}
