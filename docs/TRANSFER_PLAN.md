# KuaiRand-1K / 27K transfer — integration plan (NOT executed)

**Status: blocked on data availability. No transfer result exists, and none is
claimed.**

Checked at the time of writing:

```
kuairand-starter-kit/          KuaiRand-Pure, KuaiRand-Pure.tar.gz    <- only Pure
find / -maxdepth 4 -iname "KuaiRand-1K" -o -iname "KuaiRand-27K"      <- nothing
~/Downloads                    KuaiRand-Pure.tar.gz                   <- only Pure
```

Neither larger dataset is mounted. Everything below is a design to run *if* they
are obtained under the hackathon's KuaiRand-only data rule. It is deliberately
specific enough to execute without re-deciding anything.

---

## Why this is the largest remaining lever

The one measured fact that points here: the learning-curve experiment
(`agent.learning_curve`, recorded in the menu's dead-end notes) found the model
is **data-limited, not information-limited** — the last doubling of training
data still gained **+0.00410 (+5.12σ)**, and the per-doubling gain decayed only
slowly. Every within-Pure lever has been screened and closed; more *rows* is the
only direction the evidence still supports.

That is also why it must be done carefully. A data-limited model will happily
absorb leakage and report a large, fake improvement.

---

## The leakage risk, stated precisely

Pure's evaluation window overlaps the period covered by the larger datasets.
Pure's files are:

```
log_standard_4_08_to_4_21_pure.csv     training window   2022-04-08 .. 2022-04-21
log_standard_4_22_to_5_08_pure.csv     target window     2022-04-22 .. 2022-05-08
log_random_4_22_to_5_08_pure.csv       target window (random exposure)
```

The valid/test splits are drawn from the **target window**. KuaiRand-1K/27K
cover the same calendar period and largely the same item catalogue, so
pretraining on any 1K/27K row dated **on or after 2022-04-22** would let the
model see interactions from the window it is scored on — for users and items it
will be ranking. That is not a subtle statistical leak; it is the answer.

**Hard cutoff: pretrain only on rows with `date <= 2022-04-21`.**

## Design

```
       1K / 27K rows, date <= 2022-04-21
                    |
                    v
         [1] pretrain representations only
             (item + user embeddings; no head, no labels from target window)
                    |
                    v
         [2] initialise the Pure model from those embeddings
                    |
                    v
         [3] fine-tune and evaluate on Pure ONLY, unchanged splits
                    |
         +----------+-----------+
         v                      v
   transfer arm            Pure-only control
   (identical config,      (current incumbent recipe)
    identical seeds)
                    |
                    v
    paired multi-seed confirmation -- agent/confirm.py
```

### Steps

1. **Dataset routing.** Add explicit profiles — `pure`, `pure+1k`, `pure+27k` —
   in `agent/profiles.py`. Routing must be by profile, never by an
   auto-discovered path, so an accidentally-present directory cannot silently
   change what a run trains on.
2. **Date-cutoff enforcement at load.** A single chokepoint in the loader that
   drops every auxiliary row with `date > 2022-04-21` and **records the row
   count dropped**. A cutoff that is not counted is a cutoff nobody can audit.
3. **Label redaction.** Auxiliary rows contribute representation-learning signal
   only. The Pure target column (`long_view`) must be redacted from auxiliary
   frames at load, by the same mechanism `runtime/data_boundary.py` already uses
   for the hidden test.
4. **Pretrain.** Embeddings only. No classifier head crosses over; the head is
   trained on Pure alone.
5. **Fine-tune and evaluate on Pure**, with splits untouched.
6. **Confirm, do not adopt.** Run the transfer arm and the Pure-only control at
   the same seeds through `agent/confirm.py`. Transfer replaces the incumbent
   only if it reaches `CONFIRMED` — a single-seed win is `PRELIMINARY` at any
   effect size, and this is exactly the situation where a large fake gain is
   most likely.
7. **Log provenance.** For every auxiliary file: path, sha256, row count before
   and after the cutoff, min/max date, and wall-clock cost. `agent/provenance.py`
   already stamps this shape.

## Tests to write before running anything

- **routing** — `pure` profile loads no auxiliary rows at all; `pure+1k` loads
  exactly the declared files.
- **date cutoff** — a synthetic frame spanning the cutoff keeps only
  `date <= 2022-04-21`, and the dropped count is reported.
- **cutoff cannot be disabled** — no flag, env var or config key turns it off.
- **label redaction** — `long_view` is absent from every auxiliary frame handed
  to the model.
- **provenance** — the run summary names every auxiliary file with its hash and
  row range.
- **no Pure split contamination** — Pure's valid/test row counts (124,909 and
  170,588) are unchanged by enabling transfer.

## Resource estimate

27K is roughly two orders of magnitude larger than Pure. On the current CPU
path (~30–60s per Pure training run) a 27K pretrain is plausibly hours, not
minutes. Budget it against `--max-training-runs` explicitly, and consider 1K
first: it is the cheaper test of whether transfer helps at all, and a null on 1K
would make a 27K run hard to justify.

## What would make me abandon this

A confirmed null on 1K, or evidence that the item catalogues overlap so little
that pretrained item embeddings do not transfer. Both are cheap to establish
before committing to 27K.

---

**Nothing in this document has been run.** No transfer number exists. If a
future report cites one, it did not come from here.
