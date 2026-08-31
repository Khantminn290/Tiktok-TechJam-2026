"""One vocabulary for how a result came to exist.

Every number in this project traces back to an execution, and "an execution"
turns out to mean five different things that were previously counted as one:

    FRESH_EXECUTION     a model was trained now; compute was spent
    REUSED_ARTIFACT     a member already on disk was reused; NO compute spent
    DUPLICATE_REUSE     the SAME artifact reused again; no compute AND no
                        additional evidence
    FAILED_EXECUTION    training started and crashed; compute IS spent
    REPAIRED_EXECUTION  a failure the agent diagnosed and re-ran successfully
    MANUAL_EXECUTION    a human ran it; never counts toward autonomy

There is no general execution cache in this repository, and this module does
not add one. The only reuse is a previously completed ensemble member sitting
on disk, so the vocabulary says "artifact reuse" and never "cache".

The distinction is not bookkeeping pedantry. Two of the project's own claims
depend on getting it right:

  * **Compute cost.** The ensemble action reuses member predictions that
    already exist on disk and logs "(reusing)", but every member was charged to
    the training-run budget regardless. A 16-member ensemble over 14 existing
    members reported 16 training runs and spent 2.
  * **Independent seeds.** Evidence strength is a function of how many
    INDEPENDENT observations back a claim. A reused member is a real
    independent seed -- it was trained once, with that seed -- so reuse is
    legitimate historical evidence. But reusing it AGAIN adds nothing, and
    counting it twice is how one measurement becomes a fake confirmation.

So the rules this module encodes:

    reuse counts as EVIDENCE          (the observation is real and independent)
    reuse does NOT count as COMPUTE   (nothing was spent)
    an observation counts ONCE, ever, keyed by (configuration, seed)

That last rule needs an identity, not a counter. Two ensembles over the same
configuration share members; without a stable key the second one would report
the same seeds as fresh evidence and double the apparent support for a claim.
`observation_id(config, seed)` is that key.
"""
from __future__ import annotations

FRESH_EXECUTION = "fresh_execution"
REUSED_ARTIFACT = "reused_artifact"
DUPLICATE_REUSE = "duplicate_reuse"
FAILED_EXECUTION = "failed_execution"
REPAIRED_EXECUTION = "repaired_execution"
MANUAL_EXECUTION = "manual_execution"

# Kept as an alias so older journals still parse; new code uses REUSED_ARTIFACT.
CACHE_REUSE = REUSED_ARTIFACT

EVENT_TYPES = (FRESH_EXECUTION, REUSED_ARTIFACT, DUPLICATE_REUSE,
               FAILED_EXECUTION, REPAIRED_EXECUTION, MANUAL_EXECUTION)

# Which kinds actually cost compute. A cache hit does not; a crash does,
# because that time is spent and unrecoverable.
COSTS_COMPUTE = (FRESH_EXECUTION, FAILED_EXECUTION, REPAIRED_EXECUTION,
                 MANUAL_EXECUTION)

# Which kinds contribute an independent observation to an evidence claim. A
# crash produced no measurement; everything else did.
# A duplicate is deliberately absent: the observation it refers to was already
# counted, and counting it again is exactly the failure this guards against.
IS_OBSERVATION = (FRESH_EXECUTION, REUSED_ARTIFACT, REPAIRED_EXECUTION,
                  MANUAL_EXECUTION)

# Which kinds are the agent's own work. A human-run execution is excluded from
# any autonomy claim, whatever it measured.
IS_AUTONOMOUS = (FRESH_EXECUTION, REUSED_ARTIFACT, DUPLICATE_REUSE,
                 FAILED_EXECUTION, REPAIRED_EXECUTION)


def observation_id(config: dict | None, seed: int | None) -> str:
    """Stable identity for one measurement: this configuration, at this seed.

    Evidence is counted per identity, not per event. Two ensembles over the same
    configuration share members, and without this key the second would report
    those seeds as fresh support and silently double the evidence behind a
    claim.
    """
    import hashlib
    import json as _json
    blob = _json.dumps(config or {}, sort_keys=True, default=str)
    h = hashlib.sha256(blob.encode()).hexdigest()[:12]
    return f"{h}:seed{seed}"


def event(kind: str, seed: int | None = None, seconds: float = 0.0,
          detail: str = "", config: dict | None = None,
          observation: str | None = None) -> dict:
    """One execution event, in the shape the journal and manifest both read."""
    if kind not in EVENT_TYPES:
        raise ValueError(f"unknown execution event {kind!r}; "
                         f"expected one of {EVENT_TYPES}")
    oid = observation or (observation_id(config, seed)
                          if config is not None or seed is not None else None)
    return {"type": "execution_event", "kind": kind, "seed": seed,
            "seconds": round(float(seconds), 1), "detail": detail[:200],
            "observation_id": oid,
            "costs_compute": kind in COSTS_COMPUTE,
            "is_observation": kind in IS_OBSERVATION,
            "is_autonomous": kind in IS_AUTONOMOUS}


def tally(events) -> dict:
    """Counts that keep compute, evidence and duplication separate.

    `unique_observations` is the number that may back an evidence claim. It is
    a count of DISTINCT observation ids, so re-reusing a member cannot inflate
    it however many times it appears.
    """
    out = {k: 0 for k in EVENT_TYPES}
    compute = autonomous = 0
    seconds = 0.0
    seeds_fresh, seeds_reused = set(), set()
    seen: set = set()
    unique, duplicates = set(), 0
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
        if e.get("is_autonomous"):
            autonomous += 1

        oid = e.get("observation_id")
        if e.get("is_observation"):
            if oid is None:
                unique.add(f"_anon{len(unique)}")
            elif oid in seen:
                # Already counted. Recorded as duplication, never as support.
                duplicates += 1
                out[DUPLICATE_REUSE] += 1
                out[k] -= 1
            else:
                seen.add(oid)
                unique.add(oid)
        elif k == DUPLICATE_REUSE:
            duplicates += 1

        s = e.get("seed")
        if s is not None:
            (seeds_fresh if k == FRESH_EXECUTION else
             seeds_reused if k == REUSED_ARTIFACT else seeds_fresh).add(s)
    return {"by_kind": out,
            "training_runs_spent": compute,
            "fresh_executions": out[FRESH_EXECUTION],
            "reused_artifacts": out[REUSED_ARTIFACT],
            "duplicate_reuse_attempts": duplicates,
            "unique_observations": len(unique),
            "historical_evidence": out[REUSED_ARTIFACT],
            "autonomous_executions": autonomous,
            "manual_executions": out[MANUAL_EXECUTION],
            "distinct_fresh_seeds": sorted(seeds_fresh),
            "distinct_reused_seeds": sorted(seeds_reused),
            "wall_clock_s": round(seconds, 1),
            # kept for older readers
            "cache_hits": out[REUSED_ARTIFACT],
            "independent_observations": len(unique)}


def render(t: dict) -> str:
    b = t["by_kind"]
    return "\n".join([
        "EXECUTION EVENTS",
        f"  fresh              {b[FRESH_EXECUTION]:>4}  trained now, compute spent",
        f"  reused artifact    {b[REUSED_ARTIFACT]:>4}  already on disk, no compute",
        f"  duplicate reuse    {b[DUPLICATE_REUSE]:>4}  already counted, adds nothing",
        f"  failed             {b[FAILED_EXECUTION]:>4}  crashed; compute IS spent",
        f"  repaired           {b[REPAIRED_EXECUTION]:>4}  agent diagnosed and re-ran",
        f"  manual             {b[MANUAL_EXECUTION]:>4}  human-run; not autonomy",
        "  ---",
        f"  fresh compute (runs)     {t['training_runs_spent']}",
        f"  historical evidence      {t['historical_evidence']}",
        f"  UNIQUE observations      {t['unique_observations']}   "
        f"(what evidence may rest on)",
        f"  duplicate attempts       {t['duplicate_reuse_attempts']}",
        f"  wall-clock               {t['wall_clock_s']}s",
    ])
