<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

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

### [HELPED] iter 0 -- Seeded baseline: the validated best config from the prior search phase
menu_choices={"loss":"bpr_pairwise","user_history":"recency_weighted_pool","multitask":"aux_click_like_forward","model":"fm_numpy","temporal":"hour_plus_dow","training":"default"} scores 0.6039 single-seed (5-seed mean 0.60367; 0.60515 when 5-seed rank-averaged, +4.44 sigma over the 0.6016 baseline). THIS IS THE SCORE TO BEAT. Nine separate interventions have already been measured and ruled out against it -- see the menu tested_dead_ends. A NEW untested axis exists: model=gru4rec_seq, a GRU over the user chronologically-ordered history (KuaiRand is published as a SEQUENTIAL recommendation dataset and no model here has ever used order). It scored 0.6030 untuned on one seed. Whether it is worth pursuing is an open question.

### [DEAD_END] iter 1 -- The strongest measured recipe already uses broad, uniform BPR signal plus recency-weighted history,
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_2", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6037, below the then-current best 0.6039 -- not worth repeating as-is.

### [DEAD_END] iter 2 -- The strongest current recipe already uses broad, uniform BPR signal plus recency-weighted history, a
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_2", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6037, below the then-current best 0.6039 -- not worth repeating as-is.

### [DEAD_END] iter 3 -- The freshest high-value single-pass test is to keep the current strongest FM recipe fixed and only i
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_2", "user_history": "recency_weighted_pool", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6037, below the then-current best 0.6039 -- not worth repeating as-is.

