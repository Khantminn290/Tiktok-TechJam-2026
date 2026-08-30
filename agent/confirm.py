"""Execute a paired, multi-seed experiment. Actually run it.

This is the missing half of the confirmation story. `agent/evidence.py` could
already say "one seed is PRELIMINARY, repeat it at N paired seeds" -- correctly,
and to no effect, because nothing in the loop could produce a second seed. Every
one of the 37 nodes on record ran seed 0.

What runs here is deliberately boring and LLM-free. Both arms execute the SAME
reference script (`runtime/seed_solution.py`), which takes `menu_choices` and a
seed as arguments, so the only difference between control and treatment is the
configuration under test. Nothing is regenerated between seeds, which is what
makes the comparison paired rather than merely repeated.

Cost is real and stated up front: a 3-seed paired confirmation is 6 training
runs. That is the price of being allowed to believe something, and it is far
cheaper than promoting a result that is not there.
"""
from __future__ import annotations

import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
RUNTIME_DIR = os.path.join(ROOT, "runtime")

from .executor import run_solution  # noqa: E402
from . import experiment_spec as XS  # noqa: E402

SEED_SOLUTION = os.path.join(RUNTIME_DIR, "seed_solution.py")


def _flushing_print(*a, **k):
    """A paired run is minutes long; buffered output makes it look hung."""
    k.setdefault("flush", True)
    print(*a, **k)


def _arm(choices: dict, seeds, tag: str, work_dir: str, timeout_s: int,
         code: str | None = None, log=_flushing_print) -> dict:
    """Run one arm at each seed. Returns {seed: metrics}.

    A seed that fails is simply absent from the result: pairing then drops that
    seed from BOTH arms, which is the honest thing to do -- comparing arms on
    different seed sets would reintroduce exactly the variance pairing exists to
    remove.
    """
    src = code
    if src is None:
        with open(SEED_SOLUTION) as fh:
            src = fh.read()
    out = {}
    for s in seeds:
        run_dir = os.path.join(work_dir, f"{tag}_seed_{s:02d}")
        code_path = os.path.join(work_dir, f"{tag}_seed_{s:02d}.py")
        t0 = time.time()
        res = run_solution(src, code_path, choices, run_dir,
                           timeout_s=timeout_s, seed=s)
        if res.ok and res.metrics:
            out[s] = {k: float(res.metrics[k])
                      for k in ("GAUC", "nDCG@5", "primary")}
            log(f"    [{tag}] seed {s}  primary {out[s]['primary']:.5f}  "
                f"{time.time() - t0:.0f}s")
        else:
            head = (res.error_trace or "")[:120].replace("\n", " ")
            log(f"    [{tag}] seed {s}  FAILED — {head}")
    return out


def run_spec(spec: XS.ExperimentSpec, work_dir: str | None = None,
             timeout_s: int = 1800, control_code: str | None = None,
             treatment_code: str | None = None, log=_flushing_print) -> dict:
    """Execute a paired spec and grade it. No LLM involved."""
    work_dir = work_dir or os.path.join(ROOT, "logs", "confirm")
    os.makedirs(work_dir, exist_ok=True)

    log(f"  [confirm] {spec.experiment_type}: {spec.n_runs} training runs "
        f"across seeds {list(spec.seeds)}")
    control = _arm(spec.control, spec.seeds, "control", work_dir, timeout_s,
                   control_code, log)
    treatment = _arm(spec.treatment, spec.seeds, "treatment", work_dir,
                     timeout_s, treatment_code, log)

    res = XS.paired_result(control, treatment)
    ev = XS.grade(spec, res)
    spec.evidence_tier = ev["state"]
    XS.record(spec, res, ev)
    log(XS.render_result(spec, res, ev))
    return {"spec": spec, "control": control, "treatment": treatment,
            "paired": res, "evidence": ev,
            "promote": bool(ev.get("promote"))}


def confirm_node(node_choices: dict, control_choices: dict, hypothesis: str,
                 seeds=(0, 1, 2), node_id: int | None = None,
                 feature_lineage: dict | None = None,
                 experiment_type: str = XS.MULTI_SEED_REPLICATION,
                 work_dir: str | None = None, timeout_s: int = 1800,
                 log=_flushing_print) -> dict:
    """Confirm one candidate configuration against a control, paired by seed."""
    spec = XS.ExperimentSpec(
        hypothesis=hypothesis, experiment_type=experiment_type,
        control=control_choices, treatment=node_choices, seeds=seeds,
        parent_node=node_id, feature_lineage=feature_lineage or {},
        runtime_budget_s=timeout_s)
    return run_spec(spec, work_dir=work_dir, timeout_s=timeout_s, log=log)


def incumbent_choices() -> dict | None:
    """The configuration currently submitted -- the control worth beating."""
    p = os.path.join(ROOT, "logs", "ensemble_results.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as fh:
            return json.load(fh).get("config")
    except (json.JSONDecodeError, OSError):
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="run a paired multi-seed confirmation")
    ap.add_argument("--treatment", required=True,
                    help="JSON menu_choices for the treatment arm")
    ap.add_argument("--control", default=None,
                    help="JSON menu_choices; defaults to the submitted config")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--hypothesis", default="manual confirmation")
    a = ap.parse_args()
    ctrl = json.loads(a.control) if a.control else incumbent_choices()
    if ctrl is None:
        raise SystemExit("no control available; pass --control")
    out = confirm_node(json.loads(a.treatment), ctrl, a.hypothesis,
                       seeds=tuple(int(s) for s in a.seeds.split(",")))
    raise SystemExit(0 if out["promote"] else 1)
