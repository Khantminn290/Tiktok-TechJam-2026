<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [DEAD_END] iter 6 -- Test whether the remaining untried explicit-signal branch inside the strongest FM recipe — adding th
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "random_log_valid_unbiased_check", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6046, below the then-current best 0.6050 -- not worth repeating as-is.

### [DEAD_END] iter 7 -- Try the only still-plincipled broad additive signal family adjacent to the current best FM+BPR recip
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward_watch", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "random_log_valid_unbiased_check", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6048, below the then-current best 0.6050 -- not worth repeating as-is.

### [DEAD_END] iter 8 -- Try the remaining untested broad-signal schedule variant on top of the current best recipe: keep BPR
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "random_log_valid_unbiased_check", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6010, below the then-current best 0.6050 -- not worth repeating as-is.

### [NEUTRAL] iter 9 -- The current best node already matches the strongest measured pattern on this dataset: broad, uniform
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "random_log_valid_unbiased_check", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6050, no clear change vs the running best.

### [HELPED] iter 0 -- The strongest measured recipe is broad-signal FM+BPR with recency-weighted positive history and hour
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} raised valid primary to 0.6019 (previous best n/a).

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

