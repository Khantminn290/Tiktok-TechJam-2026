<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [NEUTRAL] iter 4 -- The current best single run may be slightly inflated by combining multiple checkpoints selected usin
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32, "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 1, "checkpoint_combine": false} scored 0.6043 vs the then-best 0.6043 (+0.00000 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- A direct standalone test of the GRU4Rec-style sequential scorer is still warranted because the prior
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6024 as the first scored node (no prior best to compare against).

### [NEUTRAL] iter 1 -- A clean architecture ablation should test whether the single-run incumbent's gain actually comes fro
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6024 vs the then-best 0.6024 (+0.00002 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 2 -- Confirmation-oriented rerun of the strongest promising branch from the broader frontier: pointwise_l
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6042 from 0.6024 (+0.00174 = +2.2 sigma).

### [HELPED] iter 0 -- The incumbent family may be under-optimized rather than under-specified: frontier bests cluster arou
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "epochs": 12, "patience": 4, "n_checkpoints": 3, "checkpoint_combine": true} scored 0.6026 as the first scored node (no prior best to compare against).

### [HELPED] iter 1 -- A clean architecture ablation should replace the incumbent's deepfm_mlp with fm_numpy while holding
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "epochs": 12, "patience": 4, "n_checkpoints": 3, "checkpoint_combine": true} raised valid primary to 0.6034 from 0.6026 (+0.00085 = +1.1 sigma).

### [NEUTRAL] iter 5 -- The incumbent FM+BPR+recency-history recipe may be getting a small, non-generalising lift from combi
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "epochs": 12, "patience": 4, "n_checkpoints": 1, "checkpoint_combine": false} scored 0.6038 vs the then-best 0.6034 (+0.00034 = +0.4 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- The current best history mechanism is recency-weighted pooling, but its decay constant is still at t
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "hist_tau_days": 7.0, "n_checkpoints": 1} scored 0.6037 as the first scored node (no prior best to compare against).

### [NEUTRAL] iter 2 -- A confirmation rerun of the strongest non-FM branch, DCN-lite with otherwise standard strong setting
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "dcn_lite", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "hist_tau_days": 7.0, "n_checkpoints": 1} scored 0.6039 vs the then-best 0.6037 (+0.00025 = +0.3 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [NEUTRAL] iter 3 -- The strongest remaining uncertainty in the current best branch is likely training dynamics rather th
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "dcn_lite", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "hist_tau_days": 7.0, "n_checkpoints": 1, "epochs": 16, "patience": 5} scored 0.6039 vs the then-best 0.6039 (+0.00000 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- DIN attention remains one of the few still-open mechanisms with a frontier best near the top (0.6043
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "k32", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.0005, "epochs": 12, "patience": 4} scored 0.6043 as the first scored node (no prior best to compare against).

### [DEAD_END] iter 1 -- Ablate the incumbent's representation by jointly removing deepfm_mlp and din_attention while keeping
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "k32", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.0005, "epochs": 12, "patience": 4} scored 0.6018 vs the then-best 0.6043 (-0.00259 = -3.2 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [NEUTRAL] iter 3 -- Root cause of the last failure: the custom Path B script depended on train_lib.training_dynamics(),
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.0005, "epochs": 12, "patience": 4} scored 0.6038 vs the then-best 0.6043 (-0.00058 = -0.7 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

