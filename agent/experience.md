<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [NEUTRAL] iter 4 -- Confirm the stronger incumbent-style FM branch in a fresh single run: fm_numpy with BPR, recency-wei
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6038 vs the then-best 0.6032 (+0.00054 = +0.7 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- A standalone GRU4Rec-style sequential model with BPR and a conservative longer/lower-LR schedule may
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6031 as the first scored node (no prior best to compare against).

### [HELPED] iter 0 -- The remaining menu-side headroom is most plausibly in pipeline constants rather than a new mechanism
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_1e3", "k": 4, "lr": 0.0005, "epochs": 20, "patience": 5, "l2": 0.001} scored 0.5980 as the first scored node (no prior best to compare against).

### [DEAD_END] iter 1 -- A clean single-axis ablation of the current FM+recency bundle should show whether the apparent value
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_1e3", "k": 4, "lr": 0.0005, "epochs": 20, "patience": 5, "l2": 0.001} scored 0.5873 vs the then-best 0.5980 (-0.01074 = -13.4 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [NEUTRAL] iter 2 -- A single-run confirmation of the selected incumbent-like FM+BPR recipe should test whether our local
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_1e3", "k": 4, "lr": 0.0005, "epochs": 20, "patience": 5, "l2": 0.001, "n_checkpoints": 5, "checkpoint_combine": true} scored 0.5982 vs the then-best 0.5980 (+0.00011 = +0.1 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 6 -- The recent local incumbent is likely being held back by an explicitly strong L2 override rather than
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 4, "lr": 0.0005, "epochs": 20, "patience": 5, "n_checkpoints": 5, "checkpoint_combine": true} raised valid primary to 0.6040 from 0.5982 (+0.00587 = +7.3 sigma).

### [NEUTRAL] iter 7 -- A matched confirmation ablation of node 6 without checkpoint combination should clarify whether its
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 4, "lr": 0.0005, "epochs": 20, "patience": 5, "n_checkpoints": 1, "checkpoint_combine": false} scored 0.6038 vs the then-best 0.6040 (-0.00026 = -0.3 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

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

