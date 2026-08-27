# Tiktok TechJam 2026 — Autonomous ML Research Agent for Recommender Systems

Team repo for the "Autonomous Machine Learning Research Agent for Recommender Systems" challenge (KuaiRand-Pure benchmark).

## Quickstart — reproduce the official baseline

Requirements: **Python 3.9+ and numpy only.** No torch, pandas, or scikit-learn needed for this step.

```bash
# 1. Clone this repo
git clone https://github.com/Khantminn290/Tiktok-TechJam-2026.git
cd Tiktok-TechJam-2026/kuairand-starter-kit

# 2. Download the dataset (~45MB download, ~194MB extracted — not committed to this repo, see below)
wget https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
tar xzf KuaiRand-Pure.tar.gz
# → produces ./KuaiRand-Pure/data/*.csv

# 3. Sanity-check the evaluation harness (should print primary ≈ 0.475 ± 0.001)
python3 baseline.py --model random --seed 0

# 4. Run the official baseline (~40s on one CPU core)
python3 baseline.py --model fm --seed 0
```

Expected output (single seed will vary slightly — see "Verifying your numbers" below):

```
=== fm (seed=0) ===
  valid  GAUC 0.6671 | nDCG@5 0.5358 | primary 0.6015
  test   GAUC 0.6621 | nDCG@5 0.5286 | primary 0.5953
```

### Verifying your numbers match

A single seed wobbles a bit (the official baseline's own std across 5 seeds is 0.0008). To reproduce the exact published reference number:

```bash
for s in 0 1 2 3 4; do python3 baseline.py --model fm --seed $s; done
```

Average the 5 "test primary" values — it should land on **0.5946 ± 0.0008**, matching `kuairand-starter-kit/baseline_scores.json`. This is the number every future model/experiment needs to beat.

## Why the dataset isn't in this repo

`KuaiRand-Pure/` (~194MB of CSVs) is excluded via `.gitignore` — it's a large public research dataset (hosted on Zenodo, no registration needed), not something we should be re-hosting or bloating the repo with. Everyone on the team should download it locally with the commands above; it's the same file for everyone (same date-based train/valid/test split, deterministic).

## Repo layout

```
kuairand-starter-kit/
  data.py                  # loads raw logs, applies the fixed date-based train/valid/test split, encodes 5 categorical fields
  baseline.py               # 3 baselines: random (sanity floor), pop (item popularity), fm (the official baseline to beat)
  evaluate.py                # PINNED scoring code — GAUC + nDCG@5, primary = mean of both. Don't modify this.
  submit.py                 # generate/validate submission CSVs in the required row_id,user_id,video_id,score format
  ablation_features.py       # reproduces the organizers' own "adding more features/capacity doesn't help" findings
  baseline_scores.json       # published reference scores, seed variance, and convergence rule (ε=0.002, N=3)
  README.md                  # the kit's own detailed reference doc (task definition, submission format, what's been tried)
```

## Ground rules everyone on the team should know

- **Task**: rank items within each user's own logged impressions (not full-catalogue retrieval). Label = `long_view` (a native 0/1 column — pinned definition: `play_time_ms >= duration_ms` if `duration_ms <= 18s`, else `play_time_ms >= 18s`).
- **Metrics**: GAUC + nDCG@5, averaged into a single `primary` score. This is exactly what `evaluate.py` computes — never modify that file, it's the sole scoring authority everyone is compared against.
- **Data split is fixed by date**: train `20220408–20220421`, valid `20220422–20220428`, test `20220429–20220508`. Splitting logic lives in `data.py`.
- **Validation only during development.** The test split's labels are physically present in your local download, but using test-split feedback for *any* development decision (feature choice, hyperparameters, early stopping, "are we winning") defeats the point of the exercise. Only look at `valid` scores while iterating; test gets evaluated once, at the very end, on the submission you're actually turning in.
- **Two things are already known dead ends** (organizers tested them, no gain — see `ablation_features.py`): adding more static feature columns, and just increasing the FM's embedding dimension. Don't re-spend an iteration rediscovering these.

See `kuairand-starter-kit/README.md` for the full detailed reference (submission format, exact convergence rule, everything tried so far).
