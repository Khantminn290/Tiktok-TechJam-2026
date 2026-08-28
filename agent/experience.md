<!-- Curated experience memory: lessons, not events. Auto-compacted to a fixed character budget -- oldest whole entries are dropped first, never truncated mid-entry. Distinct from logs/journal.jsonl (the complete, unpruned run log). Written by the harness only; generated code cannot write here (see agent/executor.py PROTECTED_PATHS). -->

### [CORRECTION] iter 7 -- best node corrected: 6 -> 7
A 5-seed reseed found node 6's single-seed score (0.6035) was seed-lucky; node 7's true mean (0.6037) is actually higher. node 7 mean 0.6037 over 5 seeds beat node 6's mean (0.6032) under the same reseed pass -- node 6's single-seed pick of 0.6035 did not hold up as the true best. Don't treat a single high score as decisive without checking its variance.

