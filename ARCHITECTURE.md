# Autonomous ML Research Agent — architecture report

KuaiRand-Pure, TikTok TechJam 2026 Track 2. Branch `opus-research-agent`.

Every number in this document is reproducible from the repository. Metrics come
from `agent/run_metrics.py` over the journals in `logs/opus_research/`; the
submitted result is re-derivable with `python3 -m agent.verify_incumbent`.

---

## 1. Architecture: before and after

### The old agent

A competent search loop. It proposed a configuration from a menu, wrote a
script, ran it, scored it, and iterated. It also had an inquiry layer, data
tools and a validity auditor. What it did **not** have was any accurate model
of its own action space, any way to tell a measurement from a guess, or any way
to hold a belief provisionally.

Three defects, all measured rather than suspected, in a pre-registered
three-run evaluation:

| defect | evidence |
|---|---|
| **The agent's model of its own capabilities was wrong.** Diagnostics were described in the prompt but existed only in the orchestrator. | 5 of 7 Path B nodes crashed (71%). Run 3 wrote `train_lib.training_dynamics()`, crashed, spent the next iteration diagnosing it, misused a second API, crashed again — 3 of its 6 iterations gone. |
| **A single seed could become a belief.** | Run 2 derived a hypothesis from data it measured itself, specified a 4-point paired sweep, ran **one seed**, and carried the value forward as settled. Running the sweep it had specified gave −0.01σ. The hypothesis was false. |
| **The convergence rule was miscalibrated 4×.** ε = 0.002 = 2.5σ, against a running-max drift of 0.60σ. | Run 2 stopped at 4 of 6 permitted iterations having "converged" on a 0.0003 gain. |

Plus a reproducibility gap: the submitted result carried `git_sha: null` and
`data_fingerprint: null`.

### The new agent

```
                      agent/capabilities.py
                   THE CAPABILITY CONTRACT (one registry)
                                 |
        +------------------------+------------------------+
        |                        |                        |
    rendered into           read by                  read by
    the prompt          agent/preflight.py     runtime/research_tools.py
   (what the agent      (what is actually       (the real importable
    believes it has)      allowed)                    surface)
```

The three consumers are generated from the same registry, so the agent's belief
about its action space, the enforcement of that space, and the code it can
actually import cannot drift apart. That is the whole fix — the old failure was
not a bad model, it was two different action spaces described as one.

Around that:

| module | what failure it fixes | how it is tested |
|---|---|---|
| `agent/capabilities.py` | the agent guessing where a capability lives | contract completeness; every capability advertised as importable really imports; nothing orchestration-only is reachable from `research_tools` |
| `runtime/research_tools.py` | orchestrator and generated code running different implementations | implementations *moved* here, not copied; `agent.validity` and `agent.pipeline_lab` delegate |
| `agent/preflight.py` | spending an iteration to learn a function does not exist | catches the real run-3 script at the capability stage in seconds |
| `agent/budget.py` | charging a research iteration for a script that never ran | a rejection is free and capped; a crash that burned compute still counts |
| `agent/evidence.py` | a one-seed number becoming a discovery | single seed is PRELIMINARY at every effect size, swept |
| `agent/knowledge.py` | a scoped result hardening into a global ban | counterevidence weakens rather than deletes; out-of-scope claims are marked |
| `agent/provenance.py` | a result nobody can trace | commit, branch, data fingerprint, config sha, seeds, evaluation path |
| `agent/verify_incumbent.py` | trusting the submitted number | recomputes 0.60541 from stored members without retraining |
| `failure.fingerprint` | crash → bad fix → identical crash | identical faults collide, differing indices do not disguise them |

---

## 2. The capability contract

Every capability declares: purpose, when to use it, what uncertainty it
resolves, inputs, outputs, invocation context, whether it is an orchestrator
tool, whether it is importable from generated Python, cost, whether it mutates
the pipeline, validation requirements, and failure modes.
Full entry: `python3 -m agent.capabilities --name selection_rule_test`.

| capability | kind | where | invoke | cost |
|---|---|---|---|---|
| `get_within_user_auc` | MEASURE | both | `from data_tools import ...` | free |
| `get_user_history_stats` | MEASURE | both | `from data_tools import ...` | free |
| `get_label_rate_by_segment` | MEASURE | both | `from data_tools import ...` | free |
| `get_feature_stats` | MEASURE | both | `from data_tools import ...` | free |
| `selection_rule_test` | CONFIRM | both | `from research_tools import ...` | cheap |
| `free_recombination` | ENSEMBLE | both | `from research_tools import ...` | cheap |
| `audit_comparison` | CONFIRM | both | `from research_tools import ...` | free |
| `selection_pressure` | CONFIRM | both | `from research_tools import ...` | free |
| `evaluate` | EVALUATE | both | `from evaluate import evaluate` | free |
| `train_numpy_fm` | TRAIN | generated code | `from train_lib import ...` | one run |
| `pipeline_override` | MODIFY | both | a key in `menu_choices` | free |
| `hardcoded_constants` | INSPECT | **orchestrator only** | — | free |
| `training_dynamics` | TRAIN | **orchestrator only** | — | one run |

**The two orchestration-only entries name what to use instead**, which is the
part that makes this a contract rather than a refusal. `training_dynamics`
says: inside generated code do not re-train — your script is already training,
so pass `cfg['capture_epoch_scores'] = []` and read the epoch curve out of it
afterwards. Same data, no extra cost.

No API was invented to satisfy the model. `pipeline_override` is available in
generated code but is *not* invoked by import, and the contract distinguishes
those two things rather than implying a module that does not exist.

---

## 5. Reproducibility and provenance

`logs/ensemble_results.json` now carries a full provenance block: commit,
branch, dirty-tree flag, dataset fingerprint with per-split row counts, config
fingerprint, seeds, code hashes and the evaluation path.

**Stamping it found a second gap.** The artifact recorded the members, the
config and the metrics — but not the **aggregation rule**. A plain mean of the
16 members gives **0.60528**; the rank-normalised mean gives **0.60541**. The
reported number was correct, but the file did not contain enough information to
reproduce it, and anyone re-deriving the result would have got a different
number with no way to tell which was right. The rule is now recorded, and
`verify_incumbent` re-applies it and demands an exact match.

```
INCUMBENT VERIFICATION — recomputed from stored predictions
members      16 from logs/final_ensemble
aggregation  rank_normalise_then_mean
  primary   recomputed 0.60541   reported 0.60541   MATCH
  GAUC      recomputed 0.67212   reported 0.67212   MATCH
  nDCG@5    recomputed 0.53870   reported 0.53870   MATCH
VERIFIED: the reported result follows from the artifacts on disk.
```

A dirty working tree is recorded rather than hidden — a SHA from a dirty tree
does not identify the code that ran, and saying so is more useful than a stamp
that implies precision it does not have.

---

## 7. Honest autonomy classification: **Level B — capability transfer**

This is unchanged by the architecture work, and deliberately so.

**Why not Level C.** The agent is clearly above replay. In the pre-architecture
evaluation it derived a hypothesis about the history decay constant from a
statistic it chose to gather (`get_user_history_stats`), reasoned from a
train/eval asymmetry it noticed itself, specified the paired sweep that would
settle it, and named a value. That hypothesis exists nowhere in its context.
It sets pipeline overrides unprompted in 9 of 16 nodes and produces a
structured observation → hypotheses → measurement chain in every node.

**Why not Level A.** Three reasons, each independently sufficient:

1. **The hypothesis with clean provenance was never actually tested by the
   agent.** It specified a four-point paired sweep and ran one seed. When the
   sweep was run properly at 5 paired seeds per arm, the effect was −0.01σ:
   **the hypothesis was false.** A belief cannot be changed by a result that
   was never obtained.
2. **The other promising line of inquiry was directed, not independent.** Runs
   1 and 3 both converged on testing checkpoint combination, which initially
   looked like the agent contradicting the teacher. It was not. The dead-end
   entry at `config/modification_menu.json:273` is prompt-visible, states that
   checkpoint averaging "DILUTES", and ends with *"Re-measure before relying on
   this entry."* The agent was following an instruction in its own context.
3. **The environment is knowledge-rich by design.** An audit before the
   post-architecture runs found 5 answer-shaped statements in research memory
   and 21 in the dead-end notes. That content is legitimate — it is what stops
   the agent re-running known failures — but it steers attention, so no run in
   this environment can support an independent-discovery claim.

**The knowledge is staying.** Removing measured results to make an autonomy
claim look cleaner would be the same error this project already had to retract
once. The honest position is that autonomy claims are bounded by what the
memory and dead-end channels supply, and to state what those contain.

A correction to an earlier claim of my own: I previously reported "0 answer-leak
hits across 8 channels" and treated the environment as clean. That check
searched for the teacher's *positive answer* and found none — correctly — while
a different teacher conclusion about the same mechanism was being supplied
elsewhere. **Absence of the answer is not absence of direction.**

---

## 8. Submission readiness

| | |
|---|---|
| **Incumbent** | `primary 0.60541`, GAUC 0.67212, nDCG@5 0.53870, k=16 seed ensemble |
| **Verified** | recomputed exactly from 16 stored member predictions, no retraining |
| **Reproduce** | `python3 -m agent.final_ensemble --seeds 16` |
| **Provenance** | commit, branch, data fingerprint, config sha, seeds, aggregation rule, evaluation path |
| **Dataset scope** | KuaiRand-Pure only. No 1K, no 27K, no external data, no transfer learning. |
| **Hidden test** | never touched — no `results/final_evaluation.lock` exists |
| **Main branch** | untouched |
| **Research branch** | `opus-research-agent` |

### Known limitations, stated plainly

- **The benchmark has plateaued.** Nothing in the post-architecture work was
  aimed at the score, and the score did not move. The architecture work is the
  deliverable; claiming otherwise would be dishonest.
- **Preflight is static.** It catches a function that does not exist. It does
  **not** catch a function called with the wrong arguments — two of run 3's
  three Path B crashes were signature misuse of a real API, which only a smoke
  execution of the full script would catch.
- **Autonomy is Level B and bounded by a knowledge-rich environment.**
- **The evidence states are computed from reported metadata.** If a node
  understates how many variants it really compared, the audit understates the
  selection risk with it.
