<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [NEUTRAL] iter 3 -- A direct confirmation-style ablation from the current best single-run recipe back to fm_numpy, while
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6038 vs the then-best 0.6040 (-0.00025 = -0.3 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [NEUTRAL] iter 6 -- Root-cause fix only: the prior checkpoint-averaging confirmation run failed because `train_lib.Histo
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "dcn_lite", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "n_checkpoints": 5, "checkpoint_combine": true} scored 0.6040 vs the then-best 0.6040 (+0.00000 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [DEAD_END] iter 9 -- Confirm whether the current best FM+BPR+recency recipe is still slightly under-optimised rather than
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.0007, "epochs": 18, "patience": 5} scored 0.6035 vs the then-best 0.6050 (-0.00145 = -1.8 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [HELPED] iter 0 -- A standalone GRU4Rec-style sequential model with BPR should modestly improve within-user ranking by
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6024 as the first scored node (no prior best to compare against).

### [DEAD_END] iter 1 -- Confirm whether the stronger already-observed frontier recipe—classical FM with pointwise_logloss, r
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6015 vs the then-best 0.6024 (-0.00085 = -1.1 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [HELPED] iter 3 -- To confirm whether the small single-seed edge of the current best branch is actually due to the GRU
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6032 from 0.6024 (+0.00083 = +1.0 sigma).

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

