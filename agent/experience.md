<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [HELPED] iter 0 -- The incumbent single-model FM+BPR+recency recipe may still gain a small but real amount from forced
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "n_checkpoints": 5, "checkpoint_combine": true} scored 0.6049 as the first scored node (no prior best to compare against).

### [DEAD_END] iter 3 -- Node 1's 0.60527 may be reproducible by staying inside the already-validated FM+BPR+recency family b
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32, "patience": 4, "n_checkpoints": 5, "checkpoint_combine": true} scored 0.6040 vs the then-best 0.6053 (-0.00127 = -1.6 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [NEUTRAL] iter 4 -- Directly replicate node 1's current best single-run result with the exact full incumbent configurati
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "n_checkpoints": 5, "checkpoint_combine": true} scored 0.6049 vs the then-best 0.6053 (-0.00034 = -0.4 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- Probe the standalone GRU4Rec-style sequential model under the exact selected menu configuration to t
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6030 as the first scored node (no prior best to compare against).

### [HELPED] iter 1 -- Reproduce the broader incumbent FM branch directly: fm_numpy with bpr_pairwise, uniform_1 negatives,
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 from 0.6030 (+0.00198 = +2.5 sigma).

### [DEAD_END] iter 9 -- Run the matched in-pipeline FM anchor that reverts the node-1 bundle back toward the official baseli
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6015 vs the then-best 0.6050 (-0.00349 = -4.4 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [HELPED] iter 0 -- A standalone GRU4Rec-style sequential scorer, using the selected conservative BPR setup, may extract
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6033 as the first scored node (no prior best to compare against).

### [HELPED] iter 1 -- Re-run the globally strongest menu branch — FM with BPR, recency-weighted pooled history, hour+dow t
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 from 0.6033 (+0.00167 = +2.1 sigma).

### [DEAD_END] iter 5 -- Single-axis confirmation by removal: test whether recency_weighted_pool is actually contributing in
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6042 vs the then-best 0.6054 (-0.00122 = -1.5 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [NEUTRAL] iter 6 -- Confirm the incumbent FM+BPR+recency branch while slightly de-biasing noisy single-run epoch selecti
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "n_checkpoints": 3, "checkpoint_combine": true} scored 0.6051 vs the then-best 0.6054 (-0.00032 = -0.4 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- A modest auxiliary multitask signal on click/like/forward added to the current best FM+BPR+recency+t
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6046 as the first scored node (no prior best to compare against).

### [CRASHED] iter 1 -- Adding top-checkpoint combination to the current strongest FM+BPR+recency+time+aux_click_like_forwar
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "n_checkpoints": 5, "checkpoint_combine": true} failed with evaluation_failure: RuntimeError: injected failure (harness robustness test) -- Scoring itself failed. Use train_lib's official evaluate on (user_raw, labels, scores); never reimplement the metric.

### [HELPED] iter 1 -- Capability transfer from the accumulated research record: reproduce the verified incumbent single-mo
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 from 0.6015 (+0.00349 = +4.4 sigma).

### [DEAD_END] iter 8 -- Test whether the current best FM+BPR+recency+time recipe is mildly under-capacity specifically under
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32} scored 0.6045 vs the then-best 0.6054 (-0.00087 = -1.1 sigma) -- worse beyond seed noise, not worth repeating as-is.

