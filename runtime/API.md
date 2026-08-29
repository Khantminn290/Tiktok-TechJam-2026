# train_lib API (for generated solutions)

Generated scripts run with `PYTHONPATH` containing both `runtime/` and
`kuairand-starter-kit/`, and env vars `KUAIRAND_KIT`, `KUAIRAND_DATA`,
`KUAIRAND_CACHE` already set. Two ways to write a solution:

## Path A — menu-driven (fastest, covers every menu combination)

```python
import train_lib
metrics = train_lib.run(menu_choices_dict, output_dir, seed=0)
```

`train_lib.run` handles everything: cached data loading, feature encoding,
the loss / user_history / multitask / model / temporal / training axes, early
stopping on valid primary, and writing `metrics.json`, `scores_valid.npy`,
`scores_test.npy` to `output_dir`. A solution that only picks menu options is
just `seed_solution.py`.

Loss options are `pointwise_logloss`, `bpr_pairwise`, `listwise_softmax`,
`listwise_softmax_plus_pointwise`, and `lambdarank_ndcg`. The last is BPR's
identical pair sampling with each pair weighted by |delta nDCG@5| instead of
uniformly, so gradient concentrates where the metric actually looks (top-5,
log2(rank+2) discount); it requires `model: fm_numpy` (numpy engine only) and
costs one extra scoring pass over the train split per epoch to compute the
current per-user ranking.

## Path B — custom code (for ideas beyond the menu)

Reusable pieces, all importable from `train_lib`:

- `splits, meta = train_lib.load_cache()` — per-split dict of numpy columns:
  `user, video, author, tab` (int codes; train vocab, UNK = last id;
  `meta["field_dims"][col]` = vocab size incl. UNK), `duration_ms, hourmin,
  date, time_ms` and labels `long_view, is_click, is_like, is_forward,
  play_time_ms`, plus `user_raw` (original string ids — use these as the
  user_ids passed to `evaluate`). Row order == official `data.load()` order,
  so per-row score arrays are submission-aligned by index.
  `meta` has exactly one key: `meta = {"field_dims": {"user": …, "video": …,
  "author": …, "tab": …}}`. Nothing else exists on it.
- `enc, dim, offsets, dims = train_lib.encode_features(splits, meta, temporal)`
  — `enc[split]` is `(N, F) int32` with per-field offsets applied (fields:
  user, video, author, tab, dur_bucket, + optional hour/dow).
- `train_lib.RankFM(dim, k, lr, seed, aux_tasks)` — numpy FM engine:
  `forward(X, H) -> (logits, cache)`, `apply_grads([(cache, dLoss_dlogits)],
  aux_contribs)`, `predict(X, H)`, `state()/load_state()`. `H` is an optional
  `(B, k)` stop-grad extra vector added into the FM interaction sum.
- `train_lib.History(splits, n_users, mode)` — train-positive history pooling
  with leave-one-out on train rows (`pooled`, `batch_vectors`).
  **`n_users` is `meta["field_dims"]["user"]`** — pass that exactly. There is no
  `meta["n_users"]` key; `meta` contains only `field_dims`, whose keys are
  `user`, `video`, `author`, `tab`. Reading any other key off `meta` raises
  KeyError and wastes the iteration.
- `train_lib.evaluate(user_ids, labels, scores)` — the OFFICIAL scorer
  (imported from the starter kit; never reimplement it).
- `train_lib.train_numpy_fm(cfg, enc, splits, meta, log)` /
  `train_lib.train_torch(...)` — full training loops if you only want to wrap.

Torch 2.3 (CPU) and numpy are available. No other ML deps are guaranteed.

## Hard rules for every solution

1. Conform to the CLI contract in `seed_solution.py`'s docstring
   (`--menu-choices`, `--output-dir`, metrics.json + scores_*.npy, non-zero
   exit + stderr trace on failure).
   **Cast metric values with `float(...)` before `json.dump`.** numpy scalars
   (`np.float32`/`np.float64`) raise `TypeError: Object of type float32 is not
   JSON serializable` and waste the iteration. Write
   `json.dump({k: float(v) for k, v in metrics.items()}, fh)`.
   Score arrays are saved with `np.save`, which needs no casting.
2. Model selection / early stopping may use VALID metrics only. Never read
   test labels; never compute test metrics. `scores_test.npy` is written
   blind.
3. Keep total runtime under the harness timeout (default 20 min); the numpy
   FM baseline takes ~1 min.
4. No external data beyond the KuaiRand-Pure files (and locked files are off
   the menu entirely).

## Path B worked examples (illustrative, NOT templates to copy)

These show the *kind* of hypothesis that needs custom code. Do not reproduce
them verbatim — they are examples of the reasoning, not a menu of new options.

**Not Path B** — "stronger L2 might help." The `regularization` axis already
expresses this. That is Path A with `regularization=l2_1e4`. Writing custom
code for it wastes an iteration and tests nothing new.

**Path B example 1 — a training-example formation the menu cannot express.**
Hypothesis: "pairs should be formed only between items shown in the same
session (same hour), because cross-session pairs compare items under different
user intent." No axis controls *how pairs are grouped*; `neg_sampling` only
controls which negative is drawn. Custom code: group train rows by
(user, hour), form BPR pairs within groups only, reuse `RankFM` and
`evaluate` unchanged.

**Path B example 2 — a data representation the menu cannot express.**
Hypothesis: "an item's embedding should be initialised from the mean of the
users who long-viewed it, so rare items start near their audience instead of
at random." No axis touches initialisation. Custom code: build the mapping
from `load_cache()`, seed `RankFM.V` accordingly, train normally.

The test for Path B is a question, not a feeling: *which existing axis would
express this?* If you can name one, it is Path A.
