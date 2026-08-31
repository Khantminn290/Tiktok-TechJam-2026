"""Verify the agent's OWN discriminating measurement, run properly.

Run 2 node 0 observed that train histories are long (mean 43.5, p90 97) while
valid ranking lists are short (mean 5.69), hypothesised that the default
recency decay hist_tau_days=3.0 over-weights ultra-recent events, and named a
paired sweep over tau in {1,3,7,14} as the measurement that would settle it.

It then ran ONE seed at tau=7 and scored 0.60366. That is not an answer, and
the agent said so itself. This runs the sweep it asked for, paired by seed.

tau=7 is the PRE-STATED arm and is the test. tau in {1,14} are exploratory and
carry best-of-n selection pressure, reported separately.
"""
import json
import os
import sys

ROOT = "."
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "runtime"))
sys.path.insert(0, os.path.join(ROOT, "kuairand-starter-kit"))

from agent.research_run import paired, report, run_variant  # noqa: E402
from agent.validity import audit_comparison, render, selection_pressure  # noqa: E402

SEEDS = [0, 1, 2, 3, 4]
OUT = os.path.dirname(os.path.abspath(__file__))

print("control: incumbent, hist_tau_days at library default 3.0", flush=True)
control = run_variant({}, SEEDS, "tau3")

arms = {}
for tau in (7.0, 14.0, 1.0):
    print(f"\narm: hist_tau_days = {tau}", flush=True)
    arms[tau] = run_variant({"hist_tau_days": tau}, SEEDS, f"tau{tau:g}")

print("\n" + "=" * 74)
print("PAIRED RESULT — the agent's proposed sweep, 5 paired seeds")
print("=" * 74)
for tau, arm in arms.items():
    print(report(f"hist_tau_days {tau} vs 3.0 (default)", control, arm))

print("\n" + "=" * 74)
print("VALIDITY OF THE PRE-STATED ARM (tau=7)")
print("=" * 74)
p7 = paired(control, arms[7.0])
a = audit_comparison(p7["delta"], n_seeds=p7["n"], paired=True,
                     n_candidates_compared=1, selected_on_eval_data=False)
print(render(a))

print("\nIf instead the best of the three taus is quoted as the result:")
sp = selection_pressure(3)
print(f"  {sp['reading']}")
best = max(arms, key=lambda t: paired(control, arms[t])["delta"])
pb = paired(control, arms[best])
ab = audit_comparison(pb["delta"], n_seeds=pb["n"], paired=True,
                      n_candidates_compared=3, selected_on_eval_data=True)
print(f"  best arm is tau={best} at {pb['delta']:+.5f} ({pb['sigma']:+.2f} sigma)")
print(render(ab))

with open(os.path.join(OUT, "tau_verification.json"), "w") as fh:
    json.dump({"seeds": SEEDS, "control": control,
               "arms": {str(k): v for k, v in arms.items()},
               "paired": {str(k): paired(control, v) for k, v in arms.items()},
               "prestated_arm_audit": a}, fh, indent=2)
print(f"\nwrote {os.path.join(OUT, 'tau_verification.json')}")
