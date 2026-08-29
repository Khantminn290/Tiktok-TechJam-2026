<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [HELPED] iter 1 -- The remaining broad, additive signal family that has not been directly tested in the current stronge
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6046 (previous best n/a).

### [HELPED] iter 2 -- A sequential additive signal may still improve within-user ranking even though set-based history poo
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6033 (previous best n/a).

### [NEUTRAL] iter 3 -- Combine Candidate A's strongest broad additive signals — recency_weighted_pool user history plus aux
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6046, no clear change vs the running best.

### [HELPED] iter 4 -- A clean ablation of the current best recipe's multitask auxiliary heads will reveal whether aux_clic
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 (previous best 0.6046).

### [HELPED] iter 5 -- Ablate the multitask auxiliary heads from the current best FM+BPR recipe: if aux_click_like_forward
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 (previous best 0.6046).

### [HELPED] iter 6 -- Ablate the contribution of multitask auxiliary heads by removing aux_click_like_forward while keepin
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6050 (previous best 0.6046).

### [NEUTRAL] iter 7 -- I take the exact backbone from Candidate A (BPR pairwise + fm_numpy + recency_weighted_pool + hour_p
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6050, no clear change vs the running best.

### [DEAD_END] iter 8 -- Confirmation should focus on the only still-promising observed-once branch that changes the represen
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6033, below the then-current best 0.6050 -- not worth repeating as-is.

### [DEAD_END] iter 9 -- Confirmation should target the only still-promising observed-once branch rather than re-running the
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6033, below the then-current best 0.6050 -- not worth repeating as-is.

### [DEAD_END] iter 10 -- Confirmation should focus on the only still-promising observed-once branch: rerun the GRU4Rec sequen
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "random_log_valid_unbiased_check", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6033, below the then-current best 0.6050 -- not worth repeating as-is.

### [HELPED] iter 0 -- Because validation ranking lists are short (mean 5.69 impressions/user) while train users have subst
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6034 (previous best n/a).

### [HELPED] iter 0 -- Adding simple mean-pooled positive user history to the strongest broad-signal FM+BPR core should imp
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6034 (previous best n/a).

### [DEAD_END] iter 1 -- A clean ablation of the incumbent's multitask choice by adding only the lightweight aux_click head o
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "aux_click", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6032, below the then-current best 0.6034 -- not worth repeating as-is.

### [DEAD_END] iter 3 -- A clean loss ablation should reveal whether the incumbent's small lift over the FM baseline is actua
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6021, below the then-current best 0.6034 -- not worth repeating as-is.

### [HELPED] iter 6 -- A clean ablation of the incumbent's training=default choice: keep the current best BPR+FM+mean-poole
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6037 (previous best 0.6034).

