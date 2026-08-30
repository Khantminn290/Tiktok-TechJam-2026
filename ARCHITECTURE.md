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
| `incumbent_cfg` | MODIFY | both | `from research_tools import ...` | free |
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

## 3. Reliability metrics — pre-registered, and the primary criterion FAILED

Three runs before the architecture work, three after, identical settings
(`--fresh --feature-discovery --research-state --data-tools --n-candidates 4
--max-iterations 6 --max-spend-usd 3.0`). Definitions fixed in
`CLEAN_PROTOCOL_2.json` before any run and computed by `agent/run_metrics.py`,
which applies the same definitions to both sets of journals.

| metric | PRE | POST | |
|---|---|---|---|
| nodes | 16 | 21 | |
| iterations consumed | 16 | 18 | |
| experiments completed | 11 | 8 | worse |
| experiments crashed | 5 | 10 | worse |
| Path B attempts | 7 | 13 | |
| Path B crashes | 5 | 12 | |
| **Path B crash rate** | **0.714** | **0.923** | **worse** |
| **orchestration-only misuse** | **1** | **0** | **fixed** |
| preflight rejections (free) | 0 | 3 | new |
| repeated identical failures | 1 | 0 | fixed |
| manual interventions | 0 | 0 | held |

**The pre-registered primary criterion was "Path B crash rate strictly below
0.71 AND zero training time spent on orchestration-only misuse". Only the
second half holds. The criterion failed.** Reporting it as a partial success
would be reframing the target around the half that worked.

### What actually happened

The targeted defect is gone. Orchestration-only misuse went to zero, preflight
caught three bad scripts before execution for no training time, and no failure
repeated identically. Those were the things the architecture was built to fix
and they are fixed.

But fixing the capability boundary made the agent *do the right thing* — follow
the contract's advice to train directly and capture its own epoch curve — and
that exposed the next defect immediately underneath:

```
failure modes across the three post-architecture runs
  preflight rejection (free, 0s)                3
  partial cfg KeyError: history, dim, bs, seed, k   5
  other (NameError 'false', unknown model)      2
```

**Five of the six real crashes were one root cause, and it was mine.**
`train_numpy_fm` requires thirteen config keys and failed on the first missing
one, so the agent burned an iteration per key: `'history'`, then `'dim'`, then
`'bs'`, then `'seed'`, then `'k'`. The contract told it to train directly
without saying what a complete config contains, and gave it no way to obtain
one — `incumbent_cfg` existed but lived in `agent/`, which generated code
cannot import.

### The fix, and its status

Applied after the protocol closed, never during:

1. `incumbent_cfg` moved into `runtime/research_tools.py` and registered in the
   contract, so generated code can get a complete config and override one key:
   `cfg, enc = incumbent_cfg(splits, meta, hist_tau_days=7.0)`. The orchestrator
   now delegates to the same implementation.
2. `train_numpy_fm` validates its config up front and reports **every** missing
   key at once, naming the builder — turning five sequential repair iterations
   into at most one.

**This fix is verified at unit level (12 keys reported together, builder
resolves, orchestrator and generated code share one implementation) and is NOT
verified at run level.** Running a fourth evaluation to show the improved number
would be adding runs after seeing results, which this protocol forbids. The
honest status is: the diagnosis is solid, the fix is tested, its effect on the
crash rate is unmeasured.

### The deeper lesson

Both evaluations found the same shape of defect: **the agent's model of its
tools was incomplete, and it paid for that in iterations.** First it did not
know *where* a capability lived; now it did not know *what a call requires*.
Preflight catches a function that does not exist; it does not catch a function
called with an incomplete argument, because that is a runtime property of a
dict. That is the honest boundary of the current design.

---

## 4. Scientific rigor metrics

| metric | PRE | POST |
|---|---|---|
| nodes stating a complete inquiry (observation + ≥2 hypotheses + measurement) | 16/16 | 21/21 |
| nodes naming the capability they need | **0/16** | **21/21** |
| ...naming one that exists and is valid in that context | 0/16 | **20/21** |
| nodes stating a promotion criterion in advance | **0/16** | **21/21** |
| confirmation-category nodes | 10/16 | 17/21 |
| diagnostic tool calls | 56 | 76 |
| **single-seed results promoted to confirmed** | — | **0** |

This is where the architecture paid off unambiguously.

**Every node now names the capability it needs and states in advance what
result would make it act.** Both were absent before. Deciding after the fact
what counts as success is how a noisy draw becomes a discovery, so requiring
the criterion up front is the cheapest available protection.

**No single-seed result was promoted.** The evidence state is computed from how
a number was obtained, and one seed is `PRELIMINARY` at every effect size —
including +3.20σ, which is what the best post-architecture run actually
produced. The agent internalised the vocabulary rather than merely being graded
by it; run 2 node 0 wrote its own promotion criterion as *"that would still be
only PRELIMINARY and would require paired repetition before adoption."*

**Methodology self-audit is live.** `audit_comparison` and `selection_pressure`
are in the contract and the validity block is in the prompt. The agent named
`selection_rule_test` — the held-out selection-rule check — as the measurement
it wanted, which is the correct tool for the question it asked.

**Leakage protection.** Before the runs, an audit of every prompt-facing channel
found 0 answer-shaped statements in the capability contract, evidence block and
validity block; 5 in research memory and 21 in the dead-end notes. Those two
channels are legitimate accumulated knowledge and were deliberately left in
place, which is precisely why §7 refuses to claim independent discovery.

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

## 6. Autonomous demonstration — one complete cycle

From `logs/opus_research/post_arch_run_2.jsonl`, node 0. Rendered by
`python3 -m agent.demo_cycle --journal logs/opus_research/post_arch_run_2.jsonl`,
which reads every line from the journal and prints `NOT PRESENT` for any step
that did not happen.

| step | what the agent did |
|---|---|
| **OBSERVATION** | Noticed its own research memory contradicted itself: *"checkpoint-combination claims are internally inconsistent: claim [2] says top-N snapshot averaging was initially rejected…"* |
| **QUESTION** | Is the remaining headroom in model *selection* along one trajectory, or in the one still-open history mechanism? |
| **HYPOTHESES** | Three, competing: H1 epoch-selection overfitting worth 0.0005–0.001; H2 DIN attention holds real under-realised signal; H3 both are noise. |
| **DIAGNOSTIC** | Capture per-epoch validation predictions and compare checkpoint rules on held-out user splits. |
| **CAPABILITY** | Named `selection_rule_test` — a real contract capability, valid in the chosen context. |
| **PROMOTION CRITERION** | Declared *before* the result: rule must beat single-best-epoch by ≥0.0005, then score ≥+0.0008 over baseline — *"and that would still be only PRELIMINARY."* |
| **TOOL CALL** | `get_within_user_auc` ×3, `hardcoded_constants`. 4 requested, 0 errors. |
| **EXPERIMENT** | DIN-attention + deepfm_mlp with `lr=0.0005`, `epochs=12` — pipeline overrides the menu cannot express. |
| **PREFLIGHT** | Passed; the experiment ran. |
| **EVALUATION** | primary 0.60416 (GAUC 0.67058, nDCG@5 0.53775). |
| **CONFIRM/REJECT** | **PRELIMINARY** (+0.00256, +3.20σ, 1 seed). *"This does NOT authorise changing the submitted system."* |
| **MEMORY** | 4 claims, 1 CONTESTED by counterevidence — the contradiction it opened on. |
| **NEXT DECISION** | Objective switched to **confirmation**: *"Is node 0 genuinely better, or is its gain mostly a validation-selection artifact from epoch/checkpoint choice and one favorable seed?"* |

The loop closes. The agent found a contradiction in its own memory, formed
competing explanations, named the right tool, pre-committed to what would count,
ran the experiment, was told a +3.20σ result was still only preliminary, and
responded by scheduling confirmation instead of building on it.

That last step is the one worth looking at. The pre-architecture agent, given a
result like this, adopted it on one seed and carried it forward. This one did
not.

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
  **not** catch one called with wrong or incomplete arguments — that is a
  runtime property. This is what let the partial-config failure through, and it
  is the honest boundary of the design.
- **The post-architecture crash-rate fix is unverified at run level.** The
  diagnosis is solid and the fix is unit-tested, but measuring its effect would
  mean adding a fourth evaluation run after seeing results, which the protocol
  forbids.
- **Two aborted protocol attempts are disclosed** in `CLEAN_PROTOCOL_2.json`,
  with what was examined before each stop. Neither was abandoned for
  unfavourable results.
- **Autonomy is Level B and bounded by a knowledge-rich environment.**
- **The evidence states are computed from reported metadata.** If a node
  understates how many variants it really compared, the audit understates the
  selection risk with it.
