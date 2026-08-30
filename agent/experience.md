<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [DEAD_END] iter 1 -- A clean ablation of the incumbent's multitask choice by adding only the lightweight aux_click head o
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "aux_click", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6032, below the then-current best 0.6034 -- not worth repeating as-is.

### [DEAD_END] iter 3 -- A clean loss ablation should reveal whether the incumbent's small lift over the FM baseline is actua
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6021, below the then-current best 0.6034 -- not worth repeating as-is.

### [HELPED] iter 6 -- A clean ablation of the incumbent's training=default choice: keep the current best BPR+FM+mean-poole
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6037 (previous best 0.6034).

### [HELPED] iter 0 -- Exploit the strongest known recipe by adding the still-helpful recency-weighted positive-history fea
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6050 as the first scored node (no prior best to compare against).

### [HELPED] iter 0 -- Probe the only remaining major standalone model family that may capture information missing from set
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6033 as the first scored node (no prior best to compare against).

### [HELPED] iter 1 -- A clean model ablation should swap the incumbent's gru4rec_seq for fm_numpy while holding the rest o
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6042 from 0.6033 (+0.00089 = +1.1 sigma).

### [NEUTRAL] iter 3 -- Confirm whether the strongest remaining lightweight history variant is a real improvement in the set
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6050 vs the then-best 0.6042 (+0.00078 = +1.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- Test the standalone GRU4Rec-style sequential model exactly as selected: prior negative evidence only
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6033 as the first scored node (no prior best to compare against).

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

