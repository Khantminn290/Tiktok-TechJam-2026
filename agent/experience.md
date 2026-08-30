<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [HELPED] iter 1 -- A clean ablation of the incumbent's user_history=none choice should reveal whether simple pooled pos
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 from 0.6033 (+0.00166 = +2.1 sigma).

### [HELPED] iter 0 -- A standalone GRU4Rec-style sequential scorer is still worth a direct probe because the benchmark is
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.001, "epochs": 10, "patience": 3, "bs": 512} scored 0.6001 as the first scored node (no prior best to compare against).

### [HELPED] iter 1 -- A clean model ablation should test whether the only scored run underperformed because GRU-based sequ
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.001, "epochs": 10, "patience": 3, "bs": 512} raised valid primary to 0.6016 from 0.6001 (+0.00147 = +1.8 sigma).

### [NEUTRAL] iter 2 -- Confirm whether adding the lightweight auxiliary click head to the current FM+BPR incumbent is a rea
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "aux_click", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.001, "epochs": 10, "patience": 3, "bs": 512} scored 0.6022 vs the then-best 0.6016 (+0.00061 = +0.8 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [NEUTRAL] iter 3 -- Adding recency-weighted pooled positive-history to the current FM+BPR+aux_click incumbent should mod
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.001, "epochs": 10, "patience": 3, "bs": 512} scored 0.6021 vs the then-best 0.6022 (-0.00011 = -0.1 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [NEUTRAL] iter 4 -- Confirm the strongest history-based FM branch seen in the broader frontier: keeping the proven FM+BP
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.001, "epochs": 10, "patience": 3, "bs": 512} scored 0.6018 vs the then-best 0.6022 (-0.00041 = -0.5 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- A more fully tuned DIN-style DeepFM run may slightly beat the best FM recipe by using candidate-cond
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32, "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 3, "checkpoint_combine": true} scored 0.6022 as the first scored node (no prior best to compare against).

### [HELPED] iter 1 -- A clean loss ablation of the current incumbent should swap only bpr_pairwise for pointwise_logloss w
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32, "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 3, "checkpoint_combine": true} raised valid primary to 0.6043 from 0.6022 (+0.00213 = +2.7 sigma).

### [DEAD_END] iter 3 -- A clean architecture ablation should switch the incumbent pointwise setup from DeepFM+DIN to plain n
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32, "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 3, "checkpoint_combine": true} scored 0.6015 vs the then-best 0.6043 (-0.00284 = -3.6 sigma) -- worse beyond seed noise, not worth repeating as-is.

### [NEUTRAL] iter 4 -- The current best single run may be slightly inflated by combining multiple checkpoints selected usin
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "k": 32, "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 1, "checkpoint_combine": false} scored 0.6043 vs the then-best 0.6043 (+0.00000 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- A direct standalone test of the GRU4Rec-style sequential scorer is still warranted because the prior
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6024 as the first scored node (no prior best to compare against).

### [NEUTRAL] iter 1 -- A clean architecture ablation should test whether the single-run incumbent's gain actually comes fro
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6024 vs the then-best 0.6024 (+0.00002 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 2 -- Confirmation-oriented rerun of the strongest promising branch from the broader frontier: pointwise_l
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6042 from 0.6024 (+0.00174 = +2.2 sigma).

