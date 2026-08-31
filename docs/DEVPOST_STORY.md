## Inspiration

We started by measuring the thing we were about to compete on, and the number stopped us cold.

The official baseline scores $0.5946$ on KuaiRand-Pure. A perfect ranking — using the true labels as the score — reaches only $0.8645$, because 27.1% of users have no positive label at all and 9.2% are all-positive. So the room above the baseline is real but narrow. And when we ran that same baseline five times, changing nothing but the random seed, the scores moved by

$$\sigma = 0.0008$$

That is the whole problem in one line. **The gap between a real improvement and a lucky seed is smaller than most people's reported gains.** We trained eight model families — FM, DeepFM, DCN, DIN, GRU4Rec, ItemCF, GBDT, item-popularity — and every one landed between 0.55 and 0.605. On a benchmark like this, an agent that simply tries things and keeps whatever scored highest is not doing research. It is doing a lottery, and then writing up the winning ticket as a finding.

So we stopped asking *how do we build an agent that gets a high score* and started asking **how do we build an agent that cannot fool itself**. That question turned out to be the whole project.

## What it does

The agent runs the full ML engineering loop with no human in it: read the research history, form a hypothesis, write a **complete training script**, check it for contract and leakage violations, run it in a sandbox that structurally cannot see the test labels, score it with the organizers' own evaluator, and decide what to do next — repair, pivot, confirm, ensemble, or stop.

The unit of search is a whole script, never a diff. Every attempt is a node: full source plus the score it earned, appended to `logs/journal.jsonl`, which is simultaneously the agent's memory and the competition's required run log.

What makes it different is what it does with a good result: **it distrusts it.**

- A single-seed score is marked `PRELIMINARY` and **cannot** promote the submission, no matter how large.
- Promotion requires a **paired confirmation** — same seeds, both arms — declared before the run.
- The final ensemble uses **all 16 seeds trained**, with $k$ fixed before any score was seen, so no member is ever picked on validation.
- The ensemble is measured against the **mean** member, not the best of $k$ draws, because "best of 16" is a statistic about luck.

## How we built it

Four pieces, each written because something went wrong without it.

**A journal that is the memory.** Every iteration records its hypothesis, why that branch was chosen, the expected effect *stated before running*, the diff, the metrics, and any failure. The agent reads its own history to decide what to try next, so the run log is not paperwork — it is the input.

**A structural data boundary, not a permission bit.** Each parallel worker runs in its own `git worktree`. Because the dataset is gitignored, a fresh worktree *has no copy of the labeled data at all*. A hardcoded relative path fails with `FileNotFoundError` — not "present but denied". A round-level lock closes the absolute-path hole. We chose the version of this that fails loudly over the version that fails quietly.

**A preflight that rejects before it spends.** Contract and leakage checks run before any training. A rejected script costs zero compute and zero budget, and the rejection is fed back so the next attempt is informed by it.

**An evidence ledger with tiers.** Every claim in our reports is `VERIFIED` (recomputed at generation time), `OBSERVED` (measured in a run), or `OPEN` (not established). Nothing is retyped by hand. If a number appears in our README, a command regenerates it.

## The moment that defined the project

Our agent produced a 16-seed ensemble scoring $0.60541$ on validation. It was our best number. We were ready to submit it.

Then a review of our own convergence handling found something ugly.

The organizers' rule ($\varepsilon = 0.002$, $N = 3$) does not only say *when a run stops*. It says **which checkpoint gets scored**: the validation-best checkpoint at the moment the rule fires. On our journal, that rule fired at **node 3**. Our ensemble was **node 4**.

It was one iteration late. It was ineligible.

We had a comfortable argument available: our internal controller used a stricter $\varepsilon$, so the search legitimately continued past that point. Three separate places in our own repository had already written down that reasoning, and it is wrong. Running longer than the official rule does not protect a later artifact — **it produces an ineligible one.** The first stopping condition wins, and you do not get to pick your interpretation after seeing which one pays better.

So we threw the run away.

We re-ran the whole thing under the organizers' rule, restructured so the official baseline is observed at node 0 and the ensemble is scheduled at node 4 — *before* the earliest no-progress window can close. Convergence now fires at node 8, and the eligible checkpoint is the ensemble itself, with nothing higher-scoring after the stop. Then we made it machine-checkable so no one has to take our word for it: `convergence_report.report()` returns an `eligible_checkpoint` block naming the eligible node and every higher-scoring node that arrived too late, surfaced in the manifest and pinned by a test.

The difference between the number we wanted and the number we were entitled to was $0.00044$ — **about half a standard deviation.** Statistically it was nothing. It was also the entire point of the project.

## What we learned

**A stricter internal rule does not make you safer.** We had assumed searching longer could only help, since it "can only find more". That is true and irrelevant. When a rule fixes *what is scored*, running past it manufactures results you are not allowed to use.

**Negative results are the deliverable.** We recorded **28 dead ends with mechanisms**, not just labels: LambdaRank decisively worse than BPR across 5 paired seeds; per-user gradient weighting actively harmful and monotonic in strength; L2 regularization a true null within $\pm0.00013$; neural architecture swaps statistically tied with a NumPy FM. We pre-registered a heterogeneous-ensemble test — members genuinely more decorrelated than same-config seeds — and it gained $0.15\sigma$, inside the noise. We wrote that down too.

**Resist the convenient explanation.** It would have been easy to declare the benchmark label-noise-limited and call 0.605 a ceiling. We measured it: only **0.177%** of within-user positive/negative pairs are feature-identical. That argument is unsupported, so we did not make it.

**Honesty has to be mechanical.** Every safeguard here exists because good intentions were not enough. The one-shot hidden-test rule is enforced by a lock file, not by discipline. The manual-intervention count comes from a ledger the agent cannot write to, so the number cannot be flattered by the thing being measured.

## Challenges we faced

**Distinguishing signal from noise at $0.0008$.** Solved with paired seeds, pre-declared comparisons, and a rule that no single-seed result can promote anything.

**Failures mid-run, with no human to catch them.** Four of nine iterations failed — a broken pipe, a preflight rejection, two API-misuse errors. The agent recovered from all four alone: it retried the crashed confirmation, fed tracebacks back into two debug attempts, and when the debug chain hit its cap it **abandoned the branch and routed around it**, returning to the best known node. **Zero manual interventions.**

**One shot at the test set.** No feedback, no second try, no way to check whether we had guessed right. Everything had to be settled on validation first.

**Our own tooling lying to us.** Late on, we found our run report summing per-node token attributions and reporting 115,099 tokens while the provider ledger said 203,602 — because only 5 of 9 nodes carry an attribution, silently dropping every call spent on failed iterations. We had been about to report a 43% undercount of the exact figure we are judged on. It is now read from the ledger everywhere.

## Results

Evaluated **once**, at final submission, after the configuration was frozen:

| split | primary | GAUC | nDCG@5 |
|---|---|---|---|
| official baseline | 0.5946 | 0.6610 | 0.5282 |
| **this submission** | **0.59810** | **0.66510** | **0.53110** |
| **absolute delta** | **+0.0035** | **+0.0041** | **+0.0029** |

**Judged score: $+0.0035$**, at $4.37\sigma$ on the baseline's own seed noise.

The validation-to-test drop was $-0.0073$. The official baseline loses $-0.0070$ across the same two splits — so this is the ordinary generalisation gap, not validation overfitting. We predicted that gap in writing *before* we spent the shot, and it landed within $0.0003$.

Reaching it cost **203,602 tokens** across 17 provider calls, **42.8 minutes** of wall-clock, **8 of the 50 permitted iterations**, **0 GPU-hours**, and **$0.72**.

## What's next

Crossover currently merges menu choices but not code; merging two scripts' actual implementations would be strictly more expressive. The search is single-threaded, and the worktree isolation to parallelise it is already built and tested. The unbiased random-exposure diagnostic is wired but the agent does not yet act on it. And the bonus benchmarks — KuaiRand-1k and 27k — need the data cache chunked before they are reachable.

But the thing we would keep is not any of that. On a benchmark this narrow, the machinery that decides **what counts as knowing something** mattered more than any model we tried. We would rather ship $+0.0035$ we can defend than $+0.006$ we cannot.
