# Autonomy audit — what the student agent actually knew

## Verdict on the first self-test: **Level C — knowledge replay**

I previously reported that the agent "independently reproduced the E5 discovery".
**That claim was wrong and is retracted.** The audit below is the evidence.

### What the agent was shown

`pipeline_lab.render_for_prompt()` was injected into every planning prompt and
contained, verbatim:

- "selection_rule_test(): ... OVERTURNED a rejected idea -- snapshot ensembling
  ... **measured honestly it wins +0.87 sigma, t=5.54, 22/24**"
- "e.g. menu_choices with ... **`\"snapshot_ensemble\": 5` with
  `\"snapshot_force\": true`** averages the top-5 checkpoints instead of taking
  the single best epoch"
- "**Checkpoint averaging is +0.87 sigma for a SINGLE model but redundant once 16
  seeds are ensembled (-0.01 sigma)**, because seed averaging removes the same
  variance."

### What the agent produced

| Agent output | Where it came from |
|---|---|
| proposed checkpoint averaging | stated in the prompt |
| `snapshot_ensemble: 5, snapshot_force: True` | the literal worked example, including `5` |
| predicted "+0.0005 to +0.0008" | 0.87 sigma x 0.0008 = 0.0007 — arithmetic on a number it was given |
| "only relevant for a single model (not multi-seed ensembles)" | the caveat, near-verbatim |

Every element of the "discovery" is traceable to the prompt. This is replay.

### Compounding leak: the tool NAMES

Even with the prose removed, `snapshot_ensemble` and especially `snapshot_force`
("adopt the snapshot without the biased same-set guard") encode the teacher's
conclusion in the identifier. This is the
`validation_safe_checkpoint_averaging()` failure mode: the name carries the
answer.

### Other channels — checked, clean

| Channel | Hits |
|---|---|
| `agent/experience.md` | 0 |
| `logs/feature_registry.jsonl` | 0 |
| frontier render | 0 |
| `agent/prompts.py` | 0 |
| `config/modification_menu.json` | 3 (pre-existing snapshot dead-end entries — legitimate negative knowledge, but they do name the mechanism) |

The leak was concentrated in the one block I wrote.

## What a clean test requires

Remove from the student environment:
1. all specific findings and effect sizes about checkpoint averaging
2. the worked example naming `snapshot_ensemble: 5`
3. leading identifier names that state a conclusion

Keep (this is legitimate Level-B transfer — a *method*, not an *answer*):
- the general capabilities and how to invoke them
- the general methodological principle that a score computed on the data that
  selected it is not evidence
- negative knowledge with scope and confidence, not blanket bans
