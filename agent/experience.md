<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

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

### [HELPED] iter 0 -- A standalone GRU4Rec-style sequential scorer may outperform set-based history pooling because KuaiRa
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6031 as the first scored node (no prior best to compare against).

### [NEUTRAL] iter 1 -- Confirm whether node 0's single-run gain actually comes from the GRU4Rec-style sequential architectu
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6037 vs the then-best 0.6031 (+0.00069 = +0.9 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [NEUTRAL] iter 5 -- Confirm whether the safer incumbent is the broader, repeatedly strong FM+BPR+recency-history recipe
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6038 vs the then-best 0.6037 (+0.00002 = +0.0 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [HELPED] iter 0 -- DIN attention remains one of the few frontier directions with non-saturated upside, but prior probes
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "din_attention", "multitask": "none", "model": "deepfm_mlp", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 3, "checkpoint_combine": true} scored 0.6042 as the first scored node (no prior best to compare against).

### [NEUTRAL] iter 4 -- A contemporaneous confirmation anchor with the long-standing strong classical recipe (fm_numpy + bpr
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "recency_weighted_pool", "multitask": "none", "model": "fm_numpy", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default", "lr": 0.0005, "epochs": 12, "patience": 4, "n_checkpoints": 1, "checkpoint_combine": false} scored 0.6038 vs the then-best 0.6042 (-0.00040 = -0.5 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

### [CRASHED] iter 0 -- Because validation ranking lists are very short (median 4 impressions/user) while training users con
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "default", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} failed with unknown: Fix exactly this and resubmit. Do not change your hypothesis: the experiment has not been tested yet. -- Cause unclear. Reduce the experiment to the smallest version that still tests the hypothesis and re-run to localise the fault.

### [HELPED] iter 1 -- Root cause of the last failure was purely mechanical: the script imported `traceback`, which is not
menu_choices={"loss": "pointwise_logloss", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6029 as the first scored node (no prior best to compare against).

### [NEUTRAL] iter 2 -- The observed gru4rec_seq gain at 0.60294 may come from the sequential architecture rather than from
menu_choices={"loss": "bpr_pairwise", "neg_sampling": "uniform_1", "user_history": "none", "multitask": "none", "model": "gru4rec_seq", "temporal": "none", "training": "lower_lr_longer", "data_extras": "none", "sample_weighting": "per_row", "regularization": "l2_default"} scored 0.6031 vs the then-best 0.6029 (+0.00012 = +0.1 sigma) -- INSIDE the 0.0008 noise floor, so this says nothing either way. Treat as untested, not as evidence.

