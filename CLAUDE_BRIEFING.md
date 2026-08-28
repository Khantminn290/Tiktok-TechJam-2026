# Project briefing — for a fresh Claude session planning the next improvement pass

This is not the project README (see `README.md`/`HANDOVER.md` for human onboarding).
This document exists to give a **fresh Claude Code session, with no memory of this
work, enough grounded context to propose what to do next** — it should be pasted or
referenced at the start of that session. It states facts and open threads; it
deliberately does not pre-decide what the next pass should target.

Repo: `/Users/khantminn/Desktop/Tiktok-TechJam-2026`, branch
`hardening/deliverables-and-integrity` (6 commits ahead of `main`, not yet merged).

---

## 1. What this project is

TikTok TechJam 2026, Track 2: build an LLM-driven agent that autonomously runs the
full ML engineering loop (hypothesize → write a complete training script → run it →
read the score → decide what's next) on the **KuaiRand-Pure** short-video
recommendation dataset, without a human iterating by hand.

**Fixed task** (do not reinterpret): rank each user's logged impressions within that
user (not full-catalog retrieval); positive label = `long_view`. Metrics: **GAUC**
and **nDCG@5** on the validation split; **primary = mean of the two**. The hidden
test set is scored **once**, at the very end, by the harness — the agent only ever
sees validation metrics during search.

**Official baseline** (organizer-provided FM, k=16, lr=1e-3): valid primary
**0.6016** (GAUC 0.6674, nDCG@5 0.5357); test primary **0.5946**. Baseline's own
5-seed std: **0.0008** (test), used throughout this project as the "is this real or
noise" yardstick — a gain is reported in units of this σ, not just as a raw delta.
Random-scoring floor: 0.4834 valid. Oracle ceiling: 0.8484 valid primary (27% of
users have zero positives, so nDCG can't reach 1 for them — judge progress against
this ceiling, not against 1.0).

**Judging weights**: Technical Execution 35% (mean absolute delta over baseline on
hidden test, GAUC/nDCG@5 equal-weighted, + robustness), Innovation & Problem Insight
20%, Impact & Relevance 20% (autonomy — fewer manual interventions scores higher),
Feasibility & Practicality 15% (token + wall-clock cost, tiered), Presentation 10%
(final event only).

**Hard constraints that must never change**: `kuairand-starter-kit/evaluate.py` is
the untouched scoring ground truth. No external training data. The hidden test set
gets scored exactly once (see `results/final_evaluation.lock` below). Everything
else is open.

---

## 2. Architecture

```
run_agent.py                 entrypoint + preflight + CLI (--fresh, --parallel-k,
                              --reseed-top, --smoke, --inject-error-at, ...)
agent/
  contracts.py               Node dataclass, ExperimentTree (journal-backed best-
                              tracking, override_best_artifacts for reseed corrections)
  menu.py                    search space loader, cross-axis validity, safety gate
  policy.py                  decide_action: draft / debug / improve / crossover
                              (untouched throughout all hardening work)
  prompts.py                 4-section prompt builder + build_merge_prompt
                              (coordinator merge) + dynamic compute-budget section
  llm.py                     provider-agnostic client (OpenAI default | Anthropic),
                              structured-output schema + bounded repair retries
  pricing.py                 real-token-based $ spend tracking + ceiling
  executor.py                subprocess sandbox: run_solution (sequential) +
                              run_parallel_round (concurrent, one shared lock/round)
  worktree.py                per-worker git worktree lifecycle (detached HEAD)
  loop.py                    AgentLoop: iterate() sequential (default) /
                              iterate_parallel() opt-in (parallel_k>=2)
  experience.py              curated lessons memory (agent/experience.md),
                              auto-compacted, fed into every prompt
  reseed.py                  --reseed-top: multi-seed statistical verification,
                              no LLM calls, self-bounds its own wall-clock cost
  baseline_repro.py          durable baseline-reproduction artifact capture
  make_submission.py         submission CSV + ensembling + the one-time test eval
  interventions.py           HUMAN-only manual-intervention log (agent can't write it)
  report.py                  journal -> human-readable report + search-tree HTML
runtime/
  train_lib.py                the training engine every generated script imports
  data_boundary.py            sandboxed/redacted data views (shared + per-worker)
  seed_solution.py / API.md   the solution-script interface, documented for prompts
config/
  modification_menu.json      THE search space -- anything not here is invisible
                               to the agent. 7 axes: loss > user_history > multitask
                               > model > temporal > training > data_extras. Two
                               leakage-sensitive data_extras options are LOCKED
                               (mechanically unselectable without a human flag).
  agent_config.json            caps, seed, safety-gate override
tests/test_harness.py          188 checks, zero LLM calls except explicit real runs
```

`config/modification_menu.json` is the single highest-leverage file: it encodes the
organizers' own priority-ranked findings (which axis to try first, and two measured
dead ends — adding static features, raising FM embedding dim — already documented
so iterations don't respend effort rediscovering them).

---

## 3. What was built, by phase (all on this branch, none yet merged to `main`)

**Phase 1 — deliverable integrity + technical scoring boundaries.** Per-node unified
diffs with SHA-256 (`logs/diffs/`); a durable baseline-reproduction artifact
(`agent/baseline_repro.py` → `logs/baseline/`); `results/final_evaluation.lock`
refusing a second hidden-test eval without an explicit, logged
`--admin-override`; a **technical** (not instruction-following) train/valid/test
boundary — generated scripts run against a sandboxed, redacted data copy while the
real dataset/cache directories are `chmod`'d unreadable for the exact duration of
the subprocess; a protected-files write-lock (journal/config/eval-lock read-only
during generated-code execution); a root-bypass fail-fast guard (`chmod` enforcement
is meaningless for root). All verified with real hostile-script attacks, not just
unit tests — one real bug found and fixed (directory-mode chmod doesn't cover files
that already exist inside it).

**Phase 2 — statistical rigor.** `--reseed-top N`: re-runs the top-N journal nodes
across multiple seeds, no LLM calls, estimates and caps its own wall-clock cost
up front. Real finding on the original (now-archived) 8-node run: the node that
looked best on a single seed (0.6035) was seed-lucky (mean 0.6032 ± 0.0003 over 5
seeds); a different node's true mean (0.6037 ± 0.0004) was actually higher.

**Phase 3 item 1 — experience memory.** `agent/experience.py`/`.md`: short, curated
lessons (HELPED/DEAD_END/CRASHED/NEUTRAL/CORRECTION), auto-compacted to a character
budget by dropping the *oldest whole entries* (never truncating mid-entry), fed into
every hypothesis-generation prompt, write-protected from generated code.

**Phase 3 item 2 — per-idea rationale.** Every proposal's LLM response includes
`rationale: {idea, why_expected_to_help, grounded_in}`, generated in the *same* call
that proposes the change (not a separate justification call — reasoning: the
grounding sources, i.e. menu-axis descriptions, are already in-context in that same
call, so attribution there is lower-risk than recall in a decontextualized second
call). `grounded_in` rejects generic non-answers ("general ML intuition") and forces
a repair retry. Surfaced in `agent/report.py`.

**Best-node override mechanism.** When a reseed pass finds a mean-verified winner
that disagrees with the live single-seed pick, `agent.reseed.apply_best_override()`
updates `logs/best_metrics.json`/`best_solution.py` and durably logs the switch
(`logs/best_override_log.jsonl` + an `experience.md` entry) — plus a real bug fix:
`ExperimentTree` was recomputing its own `best_node_id` from the raw journal on
every reload, independent of that override, so a resumed run's own `decide_action()`
would have silently reverted to the superseded pick.

**Phase 3 item 3 Part A — concurrency-safe sandbox for parallel workers.**
`agent/worktree.py` (per-worker `git worktree`, detached HEAD — gitignored real data
is *structurally absent* from any worktree checkout) + hardlinked per-worker
sandboxes (near-zero disk/setup cost for K workers) + `run_parallel_round()` (ONE
shared lock for the whole concurrent round, not one per subprocess — the actual fix
for a real race the old per-subprocess-lock design had: two overlapping locks on one
shared directory could either unlock it mid-run for a still-running sibling, or
permanently relock it after both finish). Verified with two adversarial concurrent
tests (simultaneous multi-vector attack; an asymmetric-duration test proving a fast
worker's completion doesn't prematurely unlock for a slower one).

**Phase 3 item 3 Part B — parallel dispatch + coordinator merge.** Opt-in
`--parallel-k K` (sequential remains default). K independent proposals for ONE
decided action (`policy.py` untouched — diversity comes from independent LLM
completions of the same prompt, not K different decisions), dispatched
concurrently. If 2+ candidates in a round beat the running best, a coordinator LLM
call gets FULL code for the top 2 and tries to synthesize one script that beats
both; accepted only if it strictly beats the best individual, which required **no
new gating code** — `ExperimentTree`'s existing best-tracking already only replaces
the tracked best on a strict `>`. A real K=3 test found and fixed two bugs
(iteration-id collision under concurrent dispatch; a relative-path resolution bug
when a subprocess's cwd differs from the caller's) before producing a genuine,
unforced result: all 3 workers independently proposed reimplementing multi-seed
ensembling inside one script (citing the experience-memory CORRECTION entry as their
grounding — direct evidence that mechanism steers real reasoning) and all 3 timed
out at 1200s. No merge was attempted (correctly — 0 candidates beat the best).

**Compute-budget prompt fix.** Root-caused the timeout above: nothing told the model
`exec_timeout_s` is a hard, no-partial-credit ceiling, or that `--reseed-top`
already measures seed-variance correctly, for free, after the search — so a node
reimplementing it was both redundant and structurally too expensive. Fixed with a
dynamic (not hardcoded) compute-budget section in every prompt, naming the actual
configured timeout and explicitly steering away from in-script ensembling. Chosen
over just raising `exec_timeout_s`, which would only move the same failure to a
higher seed count and raises worst-case cost for every iteration (including
`--parallel-k` rounds, whose wall-clock is bounded by the slowest worker).

---

## 4. Results timeline (all valid-split primary, mean(GAUC, nDCG@5))

| Run | Best | Score | vs. baseline (σ=0.0008) | Notes |
|---|---|---|---|---|
| Official baseline | — | 0.6016 | — | organizer FM, k=16 |
| Original 8-node sequential run (pre-hardening; now archived) | node 6 | 0.6035 (single-seed) | +2.4σ | later found seed-lucky |
| ↳ reseed-verified (5 seeds) | node 6 | mean 0.6032 ± 0.0003 | +2.0σ | |
| ↳ reseed-verified (5 seeds) | **node 7** | mean **0.6037 ± 0.0004** | **+2.6σ** | true winner of that run |
| Real Part B parallel test (K=3, pre-timeout-fix) | — | no successes | — | all 3 workers timed out |
| **Current: fresh run, fully hardened system, `--fresh`, sequential** | **node 5** | **0.6039** (single-seed) | **+2.9σ** | **not yet reseed-verified** — converged in 6 iterations, all `draft` actions (never reached `improve`/`debug`/`crossover`) |

**The current best (0.6039, node 5) is a single-seed number that has not been
reseed-verified.** Given the immediately preceding finding (a single-seed "best" of
0.6035 turned out to be seed-lucky), the same caution applies here by the project's
own established discipline — this is exactly the kind of number `--reseed-top`
exists to check before it's trusted as final.

Hidden test set has never been touched by any of this (`results/final_evaluation.lock`
doesn't exist yet — the one-time eval hasn't run).

---

## 5. Current git/commit state

```
2e6758e Fix the exec_timeout_s-vs-ensembling mismatch with prompt content
0143977 Phase 3 item 3 Part B: parallel dispatch + coordinator merge
ff7d514 Phase 3 item 3 Part A: concurrency-safe sandbox for parallel workers
1512b9e Record best-node override provenance durably
17eb2b6 Phase 2 + Phase 3 items 1-2 + best-node override mechanism
9015b6b Phase 1 hardening: deliverable integrity + technical scoring boundaries
```

**Uncommitted right now**: the fresh end-to-end run's results (journal with 6 new
nodes, `experience.md` additions, `best_metrics.json`/`best_solution.py`) — the
previous 11-node history (pre-hardening + the Part B timeout test) is safely
archived under `logs/archive_20260828_224114/`, not deleted.

`tests/test_harness.py`: **188 checks, 0 failures**, zero LLM calls or training
except explicit, reported, real verification runs (baseline repro, reseed, hostile
sandbox attacks, the parallel K=3 test, this fresh end-to-end run).

---

## 6. Unresolved threads / things a next pass could look at

Stated as observations, not a prescribed plan:

- The current best (node 5, 0.6039) has not been reseed-verified. This run never
  exercised `improve`/`debug`/`crossover` — it converged during the initial
  diverse-drafting phase, so the iterative-refinement paths (and the
  experience/rationale mechanisms under an `improve` action specifically) remain
  comparatively under-exercised on real traffic.
- `--parallel-k` has exactly one full real-world exercise (the K=3 test), and that
  one didn't produce a successful candidate, so the merge accept/reject path has
  never fired on real LLM output — only on synthetic test data.
- Bonus benchmarks (KuaiRand-1k / KuaiRand-27k) are untouched; the data cache would
  need chunking to scale.
- `make_submission.py --ensemble` hasn't been re-run against the new node 5 /
  post-hardening journal.
- The search space itself (`config/modification_menu.json`, 7 axes) hasn't grown
  during any of this hardening work — everything above changed *how* the search
  runs (integrity, statistics, concurrency, cost-awareness), not *what* it's allowed
  to try.
- Crossover only combines menu choices, not code — merging two scripts' actual
  implementations would be strictly more expressive (a pre-existing, documented
  limitation, unrelated to this session's work).
- The branch is not yet merged to `main`.
