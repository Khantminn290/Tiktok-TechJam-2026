<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [CORRECTION] iter 7 -- best node corrected: 6 -> 7
A 5-seed reseed found node 6's single-seed score (0.6035) was seed-lucky; node 7's true mean (0.6037) is actually higher. node 7 mean 0.6037 over 5 seeds beat node 6's mean (0.6032) under the same reseed pass -- node 6's single-seed pick of 0.6035 did not hold up as the true best. Don't treat a single high score as decisive without checking its variance.

### [CRASHED] iter 8 -- Keep the current strongest menu recipe unchanged, but reduce seed variance and slightly improve rank
menu_choices={"loss": "bpr_pairwise", "user_history": "mean_pool_positives", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "none"}. TIMEOUT: training run exceeded 1200s and was killed.

### [CRASHED] iter 9 -- Keep the current best menu recipe conceptually intact but replace single-seed training with a small
menu_choices={"loss": "bpr_pairwise", "user_history": "mean_pool_positives", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "none"}. TIMEOUT: training run exceeded 1200s and was killed.

### [CRASHED] iter 10 -- Keep the current strongest recipe fixed but make training more robust by averaging scores from sever
menu_choices={"loss": "bpr_pairwise", "user_history": "mean_pool_positives", "multitask": "aux_click_like_forward", "model": "fm_numpy", "temporal": "hour_plus_dow", "training": "two_stage_finetune", "data_extras": "none"}. TIMEOUT: training run exceeded 1200s and was killed.
