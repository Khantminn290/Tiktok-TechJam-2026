"""An experiment the system EXECUTES, not a paragraph it writes.

The gap this closes was measured, not suspected. Across every recorded run of
this project -- 37 nodes -- the seed was 0. Every single one. "Confirmation"
existed as a research category that changed the wording of a prompt, while the
only multi-seed machinery in the repository (`agent/reseed.py`) was a separate
post-hoc tool the loop never called.

The consequence is structural, and it is the reason the agent could never beat
its own incumbent with evidence:

    one seed  ->  evidence.PRELIMINARY  ->  never actionable
    ...and there was no code path that could produce a second seed.

So the agent could form a hypothesis, measure it, correctly report that a single
seed proves nothing, and then had no way to do anything about that. It was
disciplined into permanent inaction.

An ExperimentSpec fixes the shape of the problem: it names a control, a
treatment, a seed SET, and an acceptance rule, and something executes it.
Paired on seed, because pairing removes the seed variance that otherwise
dominates every effect at this benchmark's scale.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

from . import evidence as EV  # noqa: E402

# ------------------------------------------------------------------- types ---
EXPLORATION = "exploration"
IMPROVEMENT = "improvement"
BRANCH = "branch"
CROSSOVER = "crossover"
PATH_B_DISCOVERY = "path_b_discovery"
PATH_B_CONFIRMATION = "path_b_confirmation"
MULTI_SEED_REPLICATION = "multi_seed_replication"
ENSEMBLE_CONSTRUCTION = "ensemble_construction"
DEBUG_RECOVERY = "debug_recovery"

TYPES = (EXPLORATION, IMPROVEMENT, BRANCH, CROSSOVER, PATH_B_DISCOVERY,
         PATH_B_CONFIRMATION, MULTI_SEED_REPLICATION, ENSEMBLE_CONSTRUCTION,
         DEBUG_RECOVERY)

# Types that must be paired and multi-seed to mean anything.
CONFIRMATORY = (PATH_B_CONFIRMATION, MULTI_SEED_REPLICATION)

ROLLBACK_KEEP = "keep_incumbent"     # a failed treatment changes nothing
ROLLBACK_PROMOTE = "promote_treatment"


@dataclass
class ExperimentSpec:
    """A complete, executable description of one experiment."""
    hypothesis: str
    experiment_type: str
    control: dict                       # menu_choices for the control arm
    treatment: dict                     # menu_choices for the treatment arm
    seeds: tuple = (0, 1, 2)
    parent_node: int | None = None

    # What the proposer expects, stated BEFORE running.
    expected_primary_effect: float = 0.0
    expected_gauc_effect: float = 0.0
    expected_ndcg_effect: float = 0.0

    # When does this change what we submit?
    acceptance_threshold: float = 0.0   # 0 -> derived from the noise floor
    promotion_rule: str = "confirmed_only"
    rollback: str = ROLLBACK_KEEP

    runtime_budget_s: int = 1800
    feature_lineage: dict = field(default_factory=dict)
    evidence_tier: str = EV.UNTESTED
    notes: str = ""

    def __post_init__(self):
        if self.experiment_type not in TYPES:
            raise ValueError(f"unknown experiment_type {self.experiment_type!r}; "
                             f"expected one of {TYPES}")
        self.seeds = tuple(self.seeds)
        if self.experiment_type in CONFIRMATORY and len(self.seeds) < 3:
            # Two points give a spread estimate from a single difference, so a
            # "confirmation" at n=2 confirms almost nothing.
            raise ValueError(
                f"{self.experiment_type} needs >=3 seeds to be worth running; "
                f"got {self.seeds}")
        if not self.acceptance_threshold:
            # Half the noise floor: below this an effect is absent, not small.
            self.acceptance_threshold = EV.NOISE / 2

    @property
    def is_paired(self) -> bool:
        if self.experiment_type == ENSEMBLE_CONSTRUCTION:
            return False
        return bool(self.control) and bool(self.treatment)

    @property
    def n_runs(self) -> int:
        return len(self.seeds) * (2 if self.is_paired else 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["seeds"] = list(self.seeds)
        d["n_runs"] = self.n_runs
        d["is_paired"] = self.is_paired
        return d

    @classmethod
    def from_dict(cls, value: dict) -> "ExperimentSpec":
        """Rebuild a persisted spec, ignoring derived display fields."""
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in value.items() if k in fields})

    def render(self) -> str:
        L = [f"EXPERIMENT SPEC — {self.experiment_type}",
             f"  hypothesis   {self.hypothesis[:150]}",
             f"  seeds        {list(self.seeds)}  ({self.n_runs} training runs)",
             f"  paired       {self.is_paired}",
             f"  expects      primary {self.expected_primary_effect:+.5f} "
             f"(GAUC {self.expected_gauc_effect:+.5f}, "
             f"nDCG@5 {self.expected_ndcg_effect:+.5f})",
             f"  accepts if   |delta| >= {self.acceptance_threshold:.5f} "
             f"and state reaches {self.promotion_rule}",
             f"  on failure   {self.rollback}"]
        if self.feature_lineage:
            L.append(f"  feature      {self.feature_lineage.get('name')} "
                     f"(sha {str(self.feature_lineage.get('sha'))[:12]})")
        return "\n".join(L)


# ------------------------------------------------------------------ result ---
def paired_result(control: dict, treatment: dict, key: str = "primary") -> dict:
    """Per-seed differences, which is what a paired design is for.

    control/treatment map seed -> metrics dict.
    """
    shared = sorted(set(control) & set(treatment))
    if len(shared) < 2:
        return {"usable": False, "n": len(shared),
                "reason": "fewer than two seeds completed in BOTH arms"}
    d = [treatment[s][key] - control[s][key] for s in shared]
    m = statistics.mean(d)
    sd = statistics.pstdev(d)
    t = m / (sd / len(d) ** 0.5) if sd > 0 else 0.0
    return {"usable": True, "n": len(shared), "seeds": shared,
            "control_mean": round(statistics.mean(control[s][key] for s in shared), 5),
            "treatment_mean": round(statistics.mean(treatment[s][key] for s in shared), 5),
            "delta": round(m, 5), "sigma": round(m / EV.NOISE, 2),
            "sd": round(sd, 5), "t": round(t, 2),
            "wins": sum(1 for x in d if x > 0),
            "per_seed": {s: round(treatment[s][key] - control[s][key], 5)
                         for s in shared}}


def grade(spec: ExperimentSpec, res: dict) -> dict:
    """Evidence state for a completed paired experiment.

    The state comes from HOW it was measured -- seeds, pairing -- not from
    whether the number is pleasing.
    """
    if not res.get("usable"):
        return {"state": EV.UNTESTED, "actionable": False,
                "why": res.get("reason", "not enough completed runs"),
                "next_step": "re-run the arms that failed"}
    ev = EV.classify(delta=res["delta"], n_seeds=res["n"], paired=True,
                     n_candidates_compared=1, selected_on_eval_data=False)
    ev["promote"] = bool(ev["state"] == EV.CONFIRMED
                         and res["delta"] >= spec.acceptance_threshold)
    ev["paired"] = res
    return ev


def render_result(spec: ExperimentSpec, res: dict, ev: dict) -> str:
    L = [spec.render(), ""]
    if not res.get("usable"):
        L.append(f"  RESULT: unusable — {res.get('reason')}")
        return "\n".join(L)
    L += [f"  RESULT over {res['n']} paired seeds {res['seeds']}",
          f"    control    {res['control_mean']:.5f}",
          f"    treatment  {res['treatment_mean']:.5f}",
          f"    delta      {res['delta']:+.5f} ({res['sigma']:+.2f} sigma)  "
          f"t={res['t']}  wins {res['wins']}/{res['n']}",
          f"    per seed   {res['per_seed']}",
          f"  EVIDENCE: {ev['state']} — {ev['why']}",
          f"  PROMOTE:  {'YES' if ev.get('promote') else 'NO'}"]
    if not ev.get("promote"):
        L.append(f"    {spec.rollback}: the submitted system is unchanged.")
    return "\n".join(L)


def record(spec: ExperimentSpec, res: dict, ev: dict,
           path: str | None = None) -> dict:
    entry = {"recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             "spec": spec.to_dict(), "paired": res,
             "evidence": {k: v for k, v in ev.items() if k != "paired"}}
    path = path or os.path.join(ROOT, "logs", "experiments.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return entry
