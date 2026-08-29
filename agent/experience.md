<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [CORRECTION] iter 7 -- best node corrected: 6 -> 7
A 5-seed reseed found node 6's single-seed score (0.6035) was seed-lucky; node 7's true mean (0.6037) is actually higher. node 7 mean 0.6037 over 5 seeds beat node 6's mean (0.6032) under the same reseed pass -- node 6's single-seed pick of 0.6035 did not hold up as the true best. Don't treat a single high score as decisive without checking its variance.

### [CRASHED] iter 8 -- Keep the current strongest menu recipe unchanged, but reduce seed variance and slightly improve rank
menu_choices={"loss": "bpr_pairwise", "user_history": "mean_pool_positives", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "none"}. TIMEOUT: training run exceeded 1200s and was killed.

### [CRASHED] iter 9 -- Keep the current best menu recipe conceptually intact but replace single-seed training with a small
menu_choices={"loss": "bpr_pairwise", "user_history": "mean_pool_positives", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "none"}. TIMEOUT: training run exceeded 1200s and was killed.

### [CRASHED] iter 10 -- Keep the current strongest recipe fixed but make training more robust by averaging scores from sever
menu_choices={"loss": "bpr_pairwise", "user_history": "mean_pool_positives", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "none"}. TIMEOUT: training run exceeded 1200s and was killed.
### [HELPED] iter 0 -- Switching the baseline FM from pointwise logloss to a ranking-aligned hybrid listwise objective shou
menu_choices={"loss": "listwise_softmax_plus_pointwise", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none"} raised valid primary to 0.5986 (previous best n/a).

### [NEUTRAL] iter 1 -- Adding train-period positive-history pooling to the current strongest ranking-aligned FM should impr
menu_choices={"loss": "listwise_softmax_plus_pointwise", "user_history": "mean_pool_positives", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none"} scored 0.5985, no clear change vs the running best.

### [HELPED] iter 2 -- Try pure pairwise BPR on the baseline numpy FM, keeping all other axes at baseline. GAUC and nDCG@5
menu_choices={"loss": "bpr_pairwise", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none"} raised valid primary to 0.6032 (previous best 0.5986).

### [HELPED] iter 3 -- Starting from the current best BPR FM, add recency-weighted positive-history pooling so each candida
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none"} raised valid primary to 0.6034 (previous best 0.6032).

### [HELPED] iter 4 -- Starting from the current best BPR + recency-weighted history FM, add the strongest untried high-pri
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "none", "training": "default", "data_extras": "none"} raised valid primary to 0.6038 (previous best 0.6034).

### [HELPED] iter 5 -- Add hour-of-day and day-of-week interaction fields to the current best BPR + recency-weighted-histor
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"} raised valid primary to 0.6039 (previous best 0.6038).

### [CRASHED] iter 0 -- The strongest measured recipe so far is BPR + recency-weighted history + auxiliary feedback + hour/d
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"}. KeyError: "attribute 'forward' already exists"

### [HELPED] iter 1 -- The prior DeepFM attempt failed because train_lib's torch multitask head stores auxiliary tasks in a
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"} raised valid primary to 0.6034 (previous best n/a).

### [NEUTRAL] iter 2 -- Try the still-untested architecture alternative dcn_lite while keeping the strongest non-crashing ra
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click", "model": "dcn_lite", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"} scored 0.6034, no clear change vs the running best.

### [CRASHED] iter 3 -- Test whether the stronger auxiliary-supervision option that already helped on fm_numpy also transfer
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"}. KeyError: "attribute 'forward' already exists"

### [CRASHED] iter 4 -- Fix the DeepFM + aux_click_like_forward crash without changing the intended recipe: the failure is n
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"}. TypeError: History.batch_vectors() missing 2 required positional arguments: 'users' and 'split_is_train'

### [CRASHED] iter 5 -- Fix the custom DeepFM debug script for the exact same recipe by calling train_lib.History through it
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none"}. TypeError: '>' not supported between instances of 'NoneType' and 'int'

### [HELPED] iter 6 -- Test whether the untried lower_lr_longer schedule can unlock a small gain for the strongest stable a
menu_choices={"loss": "bpr_pairwise", "user_history": "recency_weighted_pool", "multitask": "aux_click", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none"} raised valid primary to 0.6036 (previous best 0.6034).

### [HELPED] iter 7 -- Test whether DIN-style candidate-conditioned history attention can convert the hundreds of train-per
menu_choices={"loss": "bpr_pairwise", "user_history": "din_attention", "multitask": "aux_click", "model": "deepfm_mlp", "temporal": "hour_plus_dow", "training": "lower_lr_longer", "data_extras": "none"} raised valid primary to 0.6036 (previous best 0.6036).

