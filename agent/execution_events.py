"""One vocabulary for how a result came to exist.

Every number in this project traces back to an execution, and "an execution"
turns out to mean five different things that were previously counted as one:

    FRESH_EXECUTION     a model was trained now; compute was spent
    CACHE_REUSE         an identical prior result was reused; NO compute spent
    FAILED_EXECUTION    training started and crashed; compute IS spent
    REPAIRED_EXECUTION  a failure the agent diagnosed and re-ran successfully
    MANUAL_EXECUTION    a human ran it; never counts toward autonomy

The distinction is not bookkeeping pedantry. Two of the project's own claims
depend on getting it right:

  * **Compute cost.** The ensemble action reuses member predictions that
    already exist on disk and logs "(reusing)", but every member was charged to
    the training-run budget regardless. A 16-member ensemble over 14 existing
    members reported 16 training runs and spent 2.
  * **Independent seeds.** Evidence strength is a function of how many
    INDEPENDENT observations back a claim. A reused member is a real
    independent seed -- it was trained once, with that seed -- so reuse is
    legitimate evidence. But a reused result must never be counted as a NEW
    observation of the same thing, which is how a single measurement becomes a
    fake confirmation.

So the rule this module encodes:

    reuse counts as EVIDENCE (the observation is real and independent)
    reuse does NOT count as COMPUTE (nothing was spent)
    reuse is NEVER a new observation of an existing one
"""
from __future__ import annotations

FRESH_EXECUTION = "fresh_execution"
CACHE_REUSE = "cache_reuse"
FAILED_EXECUTION = "failed_execution"
REPAIRED_EXECUTION = "repaired_execution"
MANUAL_EXECUTION = "manual_execution"

EVENT_TYPES = (FRESH_EXECUTION, CACHE_REUSE, FAILED_EXECUTION,
               REPAIRED_EXECUTION, MANUAL_EXECUTION)

# Which kinds actually cost compute. A cache hit does not; a crash does,
# because that time is spent and unrecoverable.
COSTS_COMPUTE = (FRESH_EXECUTION, FAILED_EXECUTION, REPAIRED_EXECUTION,
                 MANUAL_EXECUTION)

# Which kinds contribute an independent observation to an evidence claim. A
# crash produced no measurement; everything else did.
IS_OBSERVATION = (FRESH_EXECUTION, CACHE_REUSE, REPAIRED_EXECUTION,
                  MANUAL_EXECUTION)

# Which kinds are the agent's own work. A human-run execution is excluded from
# any autonomy claim, whatever it measured.
IS_AUTONOMOUS = (FRESH_EXECUTION, CACHE_REUSE, FAILED_EXECUTION,
                 REPAIRED_EXECUTION)


def event(kind: str, seed: int | None = None, seconds: float = 0.0,
          detail: str = "", key: str | None = None) -> dict:
    """One execution event, in the shape the journal and manifest both read."""
    if kind not in EVENT_TYPES:
        raise ValueError(f"unknown execution event {kind!r}; "
                         f"expected one of {EVENT_TYPES}")
    return {"type": "execution_event", "kind": kind, "seed": seed,
            "seconds": round(float(seconds), 1), "detail": detail[:200],
            "cache_key": key,
            "costs_compute": kind in COSTS_COMPUTE,
            "is_observation": kind in IS_OBSERVATION,
            "is_autonomous": kind in IS_AUTONOMOUS}


def tally(events) -> dict:
    """Counts that keep compute and evidence separate."""
    out = {k: 0 for k in EVENT_TYPES}
    compute = observations = autonomous = 0
    seconds = 0.0
    seeds_fresh, seeds_reused = set(), set()
    for e in events or ():
        if e.get("type") != "execution_event":
            continue
        k = e.get("kind")
        if k not in out:
            continue
        out[k] += 1
        seconds += float(e.get("seconds") or 0.0)
        if e.get("costs_compute"):
            compute += 1
        if e.get("is_observation"):
            observations += 1
        if e.get("is_autonomous"):
            autonomous += 1
        s = e.get("seed")
        if s is not None:
            (seeds_fresh if k == FRESH_EXECUTION else
             seeds_reused if k == CACHE_REUSE else seeds_fresh).add(s)
    return {"by_kind": out,
            "training_runs_spent": compute,
            "cache_hits": out[CACHE_REUSE],
            "independent_observations": observations,
            "autonomous_executions": autonomous,
            "manual_executions": out[MANUAL_EXECUTION],
            "distinct_fresh_seeds": sorted(seeds_fresh),
            "distinct_reused_seeds": sorted(seeds_reused),
            "wall_clock_s": round(seconds, 1)}


def render(t: dict) -> str:
    b = t["by_kind"]
    return "\n".join([
        "EXECUTION EVENTS",
        f"  fresh            {b[FRESH_EXECUTION]:>4}   trained now, compute spent",
        f"  cache reuse      {b[CACHE_REUSE]:>4}   identical prior result, no compute",
        f"  failed           {b[FAILED_EXECUTION]:>4}   crashed; compute IS spent",
        f"  repaired         {b[REPAIRED_EXECUTION]:>4}   agent diagnosed and re-ran",
        f"  manual           {b[MANUAL_EXECUTION]:>4}   human-run; excluded from autonomy",
        f"  ---",
        f"  training runs spent      {t['training_runs_spent']}",
        f"  independent observations {t['independent_observations']}",
        f"  wall-clock               {t['wall_clock_s']}s",
        "  (reuse counts as evidence, never as compute, and never as a NEW "
        "observation)",
    ])
