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
the loss / score_prior / user_history / multitask / model / temporal / training axes, early
stopping on valid primary, and writing `metrics.json`, `scores_valid.npy`,
`scores_test.npy` to `output_dir`. A solution that only picks menu options is
just `seed_solution.py`.

The child-written `metrics.json` is compatibility output, not trusted evidence.
The parent runner recomputes validation metrics from `scores_valid.npy` with the
official evaluator, preserves the reported values separately, and writes a
hash-bearing `verification.json`. Provider/API secrets are removed from the
training subprocess environment. A Python audit-hook guard also blocks ordinary
generated-code access to raw data, `.env`, prior-run folders, subprocesses, and
network sockets. It is defense in depth, not an OS security sandbox; hostile
native code still requires a container or restricted worker account.

Most `score_prior` options are model-agnostic, train-only post-training ranking signals. The
`bayesian_item_author` option estimates smoothed long-view logits for videos and
authors; `recency_bayesian_item_author` exponentially downweights older training
rows. Both shrink rare/unseen entities to the training global mean and blend the
centered prior logit with the model logit. Their statistics read training labels
only; validation labels are used only by evaluation, and test outcomes are absent.
The primitive is available as `train_lib.bayesian_prior_scores(splits, mode)`.

`score_prior=batch_repeat_fatigue` is a downstream, label-free exception. It
uses the complete row-aligned input batch to count repeated `(true user, video)`
exposures, standardizes both model score and repetition signal within each user,
then applies one fixed `-0.10` fatigue penalty. It uses `user_raw` so unseen users
never collapse into the shared train-vocabulary UNK code. This is transductive
batch inference, is disclosed in `batch_context_info.json`, and never reads an
outcome column.

## Path B — custom code (for ideas beyond the menu)

Reusable pieces, all importable from `train_lib`:

- `splits, meta = train_lib.load_cache()` — per-split dict of numpy columns:
  `user, video, author, tab` (int codes; train vocab, UNK = last id;
  `meta["field_dims"][col]` = vocab size incl. UNK), `duration_ms, hourmin,
  date, time_ms`, plus `user_raw` and `video_raw` (original string ids — use
  `user_raw` as the user_ids passed to `evaluate`). Row order == official `data.load()` order,
  so per-row score arrays are submission-aligned by index.
  Train and validation also contain `long_view, is_click, is_like, is_forward,
  play_time_ms`. Test contains no outcome columns; it can only be scored blind.
  Cache schema v3 records SHA-256 digests for its three raw source files; the
  trusted parent rebuilds if that source manifest changes, while guarded child
  runs reuse the already-verified cache without raw-file access.
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
2. Model selection / early stopping may use VALID metrics only. Test outcomes
   are absent from `load_cache()`; never open raw test rows to recover them or
   compute test metrics. `scores_test.npy` is written blind.
3. Keep total runtime under the harness timeout (default 20 min); the numpy
   FM baseline takes ~1 min.
4. No external data beyond the KuaiRand-Pure files (and locked files are off
   the menu entirely).
