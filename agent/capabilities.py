"""The single authoritative statement of what the agent can do, and from where.

This module exists because of a measured failure, not a design preference. In
the clean autonomy evaluation, 5 of 7 Path B nodes crashed. Not one was an ML
error. The agent believed `training_dynamics()` was callable from inside its
generated experiment script, wrote `train_lib.training_dynamics()`, and crashed;
spent the next iteration diagnosing that crash; misused another API; crashed
again. Three of six iterations in one run were spent failing to reach a
capability the prompt had told it it had.

The bug was never in the agent's reasoning. It was that **two different action
spaces were described as one**. Some capabilities run in the orchestrator,
between iterations, and can never be imported by generated code. Others are
ordinary Python that generated code can and should import. Nothing recorded
which was which, so the agent had to guess, and guessed wrong.

So: one registry, two consumers.

    capabilities.py  (this file -- the contract)
        |
        +--> rendered into the prompt, so the agent's mental model comes
        |    from the same source as the enforcement
        |
        +--> read by preflight.py, which rejects a script that calls a
             capability from a context where it does not exist -- BEFORE
             the expensive run starts
        |
        +--> read by runtime/research_tools.py, the real importable surface

The rule that keeps this honest: **never expose an API merely to satisfy the
LLM**. Where a capability genuinely cannot run in generated code, it is marked
orchestration-only and its entry names the mechanism that DOES work there.
`training_dynamics` is the worked example -- inside a generated script the
useful thing is not a second training run, it is capturing the epoch curve of
the training the script is already doing, so the entry says exactly that.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

# ---------------------------------------------------------------- taxonomy ---
# What a capability DOES, which is what determines when it is the right call.
INSPECT = "INSPECT"       # read static structure; no data, no training
MEASURE = "MEASURE"       # measure the data or an existing artifact; no training
MODIFY = "MODIFY"         # change the pipeline that will be trained
TRAIN = "TRAIN"           # spend a training run
EVALUATE = "EVALUATE"     # score predictions
CONFIRM = "CONFIRM"       # decide whether an effect survives scrutiny
ENSEMBLE = "ENSEMBLE"     # combine multiple models' predictions

KINDS = (INSPECT, MEASURE, MODIFY, TRAIN, EVALUATE, CONFIRM, ENSEMBLE)

# ----------------------------------------------------------------- contexts ---
ORCHESTRATOR = "orchestrator"      # the inspect/planning phase, between runs
GENERATED = "generated_code"       # inside the experiment script the agent writes

FREE, CHEAP, ONE_RUN = "free", "cheap", "one_training_run"


@dataclass(frozen=True)
class Capability:
    name: str
    kind: str
    purpose: str                    # what it measures, neutrally stated
    when: str                       # the situation in which it is the right call
    resolves: str                   # which uncertainty it reduces
    inputs: str
    outputs: str
    contexts: tuple                 # where it may be invoked
    module: str | None              # import path for generated code, if any
    cost: str
    mutates_pipeline: bool
    validation: str                 # what must hold for its result to mean anything
    failure_modes: str
    instead: str = ""               # if unavailable here, what to use instead
    aliases: tuple = field(default_factory=tuple)
    # MACHINE-READABLE return shape. Prose in `outputs` is for a human reader;
    # this is what preflight checks a call site against.
    #   {"kind": "dict",  "keys": [...]}                 -> NOT unpackable
    #   {"kind": "tuple", "arity": 2, "names": [...]}     -> unpackable, arity 2
    #   {"kind": "list_of_tuple", "arity": 3, ...}        -> each element unpacks
    # Every Path B crash in the last run was a call site disagreeing with one of
    # these, so they are data rather than documentation.
    returns: dict = field(default_factory=dict)
    # Machine-readable signature: {"required": [...], "optional": [...],
    # "var_keyword": bool}. A call site missing a required argument is a
    # TypeError that costs a full training run to discover -- observed as
    # `selection_rule_test() missing 1 required positional argument: 'rules'`.
    params: dict = field(default_factory=dict)
    example: str = ""               # correct usage, copy-pasteable

    @property
    def return_kind(self) -> str:
        return (self.returns or {}).get("kind", "unknown")

    @property
    def unpackable(self) -> bool:
        """May a call site write `a, b = f(...)`?"""
        return self.return_kind == "tuple"

    @property
    def return_arity(self) -> int | None:
        return (self.returns or {}).get("arity")

    @property
    def orchestrator_tool(self) -> bool:
        return ORCHESTRATOR in self.contexts

    @property
    def importable(self) -> bool:
        """Available inside generated code at all -- not necessarily by import."""
        return GENERATED in self.contexts

    @property
    def invoked_by_import(self) -> bool:
        """Reached with an `import`, as opposed to being set in configuration.

        `pipeline_override` is the case that forces this distinction: it is
        fully available to generated code, but you use it by putting a key in
        menu_choices, not by importing a function. Conflating the two would make
        the contract claim there is a module to import when there is not.
        """
        return self.importable and bool(self.module)

    @property
    def expensive(self) -> bool:
        return self.cost == ONE_RUN


# ----------------------------------------------------------------- registry ---
# Purposes are written NEUTRALLY. They say what a capability measures, never
# what it found -- a description that leaks a research outcome turns the tool
# list into an answer key.
_REGISTRY: dict[str, Capability] = {}


def register(c: Capability) -> Capability:
    _REGISTRY[c.name] = c
    return c


register(Capability(
    name="get_within_user_auc", kind=MEASURE,
    purpose="AUC of a single feature computed WITHIN each user, on a chosen split.",
    when="Before proposing a feature, to see whether it separates positives from "
         "negatives inside a user's own impression list.",
    resolves="Whether a candidate signal carries any within-user ranking "
             "information at all.",
    inputs="feature: str, split: str = 'valid'",
    outputs="{'auc': float, 'n_users': int, 'coverage': float}",
    contexts=(ORCHESTRATOR, GENERATED), module="data_tools", cost=FREE,
    mutates_pipeline=False,
    validation="Both metrics rank WITHIN a user, so a feature constant across a "
               "user's rows contributes nothing regardless of its global AUC.",
    failure_modes="Unknown feature name raises KeyError. A feature with near-zero "
                  "within-user variation returns an AUC near 0.5 that means "
                  "'no variation', not 'no signal'.",
    returns={"kind": "dict", "keys": ["auc", "n_users", "coverage"]},
    example='d = get_within_user_auc("play_time_ms")',
    params={"required": ["feature"], "optional": ["split", "cache_dir"]}))

register(Capability(
    name="get_user_history_stats", kind=MEASURE,
    purpose="Distribution of per-user impression counts on a split.",
    when="When reasoning about anything that pools or weights a user's history.",
    resolves="Whether history length differs between the split used for training "
             "and the split used for ranking.",
    inputs="split: str = 'train'",
    outputs="{'mean': float, 'p50': float, 'p90': float, 'n_users': int}",
    contexts=(ORCHESTRATOR, GENERATED), module="data_tools", cost=FREE,
    mutates_pipeline=False,
    validation="Descriptive only. A difference between splits is a fact about the "
               "data, not evidence that any particular mechanism will help.",
    failure_modes="Unknown split name raises KeyError.",
    returns={"kind": "dict", "keys": ["mean", "p50", "p90", "n_users"]},
    example='d = get_user_history_stats("train")',
    params={"required": [], "optional": ["split", "cache_dir"]}))

register(Capability(
    name="get_label_rate_by_segment", kind=MEASURE,
    purpose="Positive-label rate across bins of a feature.",
    when="When asking whether a segment of the data behaves differently.",
    resolves="Whether label prevalence varies with a covariate.",
    inputs="feature: str, n_bins: int = 10",
    outputs="{'bins': [...], 'rates': [...], 'counts': [...]}",
    contexts=(ORCHESTRATOR, GENERATED), module="data_tools", cost=FREE,
    mutates_pipeline=False,
    validation="A rate difference across segments does not imply a within-user "
               "ranking gain; the metric ranks inside users, not across them.",
    failure_modes="Unknown feature raises KeyError; sparse bins give noisy rates.",
    returns={"kind": "dict", "keys": ["bins", "rates", "counts"]},
    example='d = get_label_rate_by_segment("duration_ms")',
    params={"required": ["feature"], "optional": ["n_bins", "cache_dir"]}))

register(Capability(
    name="get_feature_stats", kind=MEASURE,
    purpose="Summary statistics for one cached feature column.",
    when="To check coverage, range and missingness before using a feature.",
    resolves="Whether a feature is populated and varies at all.",
    inputs="feature: str, split: str = 'train'",
    outputs="{'mean','std','min','max','missing',...}",
    contexts=(ORCHESTRATOR, GENERATED), module="data_tools", cost=FREE,
    mutates_pipeline=False,
    validation="Near-zero std can be a float artefact; compare on a scale-relative "
               "tolerance rather than testing equality.",
    failure_modes="Unknown feature raises KeyError.",
    returns={"kind": "dict", "keys": ["mean", "std", "min", "max", "missing"]},
    example='d = get_feature_stats("duration_ms")',
    params={"required": ["feature"], "optional": ["split", "cache_dir"]}))

register(Capability(
    name="hardcoded_constants", kind=INSPECT,
    purpose="Modelling constants written into the training library, and whether "
            "each one is reachable through a pipeline override.",
    when="When the search keeps landing within noise and the thing that matters "
         "may be a constant no menu axis can express.",
    resolves="Which parts of the pipeline are currently outside the search space.",
    inputs="(none)",
    outputs="list of {'name','default','line','kind','override_key',"
            "'reachable_by_agent'}",
    contexts=(ORCHESTRATOR,), module=None, cost=FREE, mutates_pipeline=False,
    validation="Reports that a constant EXISTS. It says nothing about whether "
               "changing it helps -- that requires a paired measurement.",
    failure_modes="Static source scan; a constant computed at runtime is invisible "
                  "to it.",
    instead="In generated code you do not need this: set the constant directly in "
            "the cfg dict you pass to train_lib, or via menu_choices overrides."))

register(Capability(
    name="training_dynamics", kind=TRAIN,
    purpose="The validation-score curve across epochs with early stopping "
            "disabled, plus where it peaks and how far it moves afterwards.",
    when="When you do not know whether the model is over- or under-fitting, and "
         "the answer would change what you try next.",
    resolves="Over- vs under-fitting; whether the stopping rule or the capacity "
             "is the binding constraint.",
    inputs="seeds: tuple = (0,), max_epochs: int = 40",
    outputs="{'seeds': {seed: {'curve', 'peak_epoch', 'peak_primary', "
            "'decline_sigma'}}, 'verdict': str}",
    contexts=(ORCHESTRATOR,), module=None, cost=ONE_RUN, mutates_pipeline=False,
    validation="The peak epoch is an argmax over many validation evaluations, so "
               "it is fitted to validation. Treat the SHAPE of the curve as the "
               "finding, not the exact peak.",
    failure_modes="Costs a full training run; allowed at most once per agent run.",
    instead="INSIDE GENERATED CODE, do not call this and do not re-train to get "
            "it. Your script is already training. Capture its own curve:\n"
            "      from research_tools import incumbent_cfg\n"
            "      cfg, enc = incumbent_cfg(splits, meta)\n"
            "      cfg['capture_epoch_scores'] = []\n"
            "      train_lib.train_numpy_fm(cfg, enc, splits, meta, print)\n"
            "  ...and that list then holds (epoch, valid_primary, scores) for "
            "every epoch. Same data, no extra training run. Use incumbent_cfg "
            "to build the config -- train_numpy_fm needs 13 required keys and "
            "hand-assembling them costs an iteration per missing key."))

register(Capability(
    name="selection_rule_test", kind=CONFIRM,
    purpose="Whether a rule for CHOOSING among candidates generalises, by "
            "selecting on one half of the users and scoring on the other, in "
            "both directions, over several independent splits.",
    when="Whenever a number was produced by picking the best of several options "
         "on the same data it is now being scored on.",
    resolves="Whether an apparent gain is a real effect or fit to the split that "
             "selected it.",
    inputs="per_epoch_scores (seeds, epochs, rows), users, labels, "
           "rules: {name: fn(per_epoch_primaries, per_epoch_scores) -> scores}",
    outputs="{'reference_rule', 'n_evaluations', 'rules': {name: {'mean_delta', "
            "'sigma', 't', 'wins', 'n', 'generalises'}}}",
    contexts=(ORCHESTRATOR, GENERATED), module="research_tools", cost=CHEAP,
    mutates_pipeline=False,
    validation="Needs per-epoch predictions to already exist. The first rule in "
               "`rules` is the reference every other is measured against.",
    failure_modes="Passing scores for a single epoch makes the comparison "
                  "meaningless; shapes must line up with users/labels.",
    returns={"kind": "dict", "keys": ["reference_rule", "n_evaluations", "rules"]},
    example='out = selection_rule_test(per_epoch, users, labels, rules)',
    params={"required": ["per_epoch_scores", "users", "labels", "rules"], "optional": ["n_splits", "seed"]}))

register(Capability(
    name="capture_selection_rule_test", kind=CONFIRM,
    purpose="Test fixed best-epoch versus top-epoch averaging rules on held-out "
            "users using one training run's captured validation curve.",
    when="After train_numpy_fm captured epoch scores and you need to check whether "
         "a checkpoint-selection rule generalises without manually rebuilding the "
         "selection_rule_test tensor and callable mapping.",
    resolves="Whether a simple epoch-selection rule survives held-out-user splits, "
             "rather than merely fitting the validation rows that selected it.",
    inputs="capture_epoch_scores from cfg, users, labels",
    outputs="the selection_rule_test result for best_epoch, mean_top2 and mean_top3 "
            "when enough captured epochs exist",
    contexts=(ORCHESTRATOR, GENERATED), module="research_tools", cost=CHEAP,
    mutates_pipeline=False,
    validation="Diagnostic only. It does not create per-epoch test predictions and "
               "cannot itself authorise a submission change.",
    failure_modes="Requires non-empty (epoch, valid_primary, scores_valid) tuples "
                  "whose score arrays match users and labels.",
    returns={"kind": "dict", "keys": ["reference_rule", "n_evaluations", "rules"]},
    example=("res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)\n"
             "diag = capture_selection_rule_test(cfg['capture_epoch_scores'], "
             "valid_users, valid_labels)"),
    params={"required": ["capture_epoch_scores", "users", "labels"],
            "optional": ["n_splits", "seed"]}))

register(Capability(
    name="free_recombination", kind=ENSEMBLE,
    purpose="Compare aggregation rules over stored member predictions by "
            "resampling member subsets, with no training at all.",
    when="When choosing how to combine models, or checking whether a combination "
         "rule that won once wins repeatedly.",
    resolves="Whether an aggregation rule's advantage survives resampling.",
    inputs="member_scores (members, rows), users, labels, "
           "rules: {name: fn(subset_scores) -> scores}, n_subsets, subset",
    outputs="{'reference_rule', 'n_subsets', 'rules': {name: {...}}}",
    contexts=(ORCHESTRATOR, GENERATED), module="research_tools", cost=CHEAP,
    mutates_pipeline=False,
    validation="Requires >= 2 stored members. A rule winning on one subset is not "
               "a result; the resampled win-rate is the evidence.",
    failure_modes="Fewer members than `subset` silently shrinks the subset.",
    returns={"kind": "dict", "keys": ["reference_rule", "n_subsets", "rules"]},
    example='out = free_recombination(members, users, labels, rules)',
    params={"required": ["member_scores", "users", "labels", "rules"], "optional": ["n_subsets", "subset", "seed"]}))

register(Capability(
    name="audit_comparison", kind=CONFIRM,
    purpose="Grade a claimed improvement by HOW IT WAS MEASURED -- seeds, "
            "pairing, how many variants were compared, whether the winner was "
            "chosen on the data now scoring it.",
    when="Before believing any improvement, including your own.",
    resolves="What a number is worth as evidence, independent of its size.",
    inputs="delta, n_seeds, paired, n_candidates_compared, "
           "selected_on_eval_data, confirmed_out_of_sample",
    outputs="{'severity','verdict','findings',...}",
    contexts=(ORCHESTRATOR, GENERATED), module="research_tools", cost=FREE,
    mutates_pipeline=False,
    validation="Advisory. It grades the DESIGN of a comparison and cannot know "
               "intent, so it never blocks an experiment.",
    failure_modes="Garbage in: if n_candidates_compared understates how many "
                  "things were really tried, the audit understates the risk.",
    returns={"kind": "dict", "keys": ["delta", "sigma", "severity", "verdict", "findings", "trustworthy"]},
    example='a = audit_comparison(delta, n_seeds=5, paired=True)',
    params={"required": ["delta"], "optional": ["n_seeds", "paired", "n_candidates_compared", "selected_on_eval_data", "seed_sd", "confirmed_out_of_sample"]}))

register(Capability(
    name="selection_pressure", kind=CONFIRM,
    purpose="How large an apparent gain arises purely from picking the best of n "
            "noisy comparisons.",
    when="Before quoting any 'we tried n things and this was best' number.",
    resolves="The bar a real effect must clear to be distinguishable from "
             "selection.",
    inputs="n_candidates_compared: int",
    outputs="{'expected_max_delta','expected_max_sigma','reading'}",
    contexts=(ORCHESTRATOR, GENERATED), module="research_tools", cost=FREE,
    mutates_pipeline=False,
    validation="Assumes independent comparisons at the benchmark noise floor.",
    failure_modes="None; pure arithmetic.",
    returns={"kind": "dict", "keys": ["n", "expected_max_delta", "expected_max_sigma", "reading"]},
    example='sp = selection_pressure(5)',
    params={"required": ["n_candidates_compared"], "optional": ["seed_sd"]}))

register(Capability(
    name="pipeline_override", kind=MODIFY,
    purpose="Set a training-pipeline constant the menu cannot express, by putting "
            "it directly in menu_choices.",
    when="When the axis you want to change is a constant rather than a component.",
    resolves="Whether a pipeline constant, not a component choice, is limiting.",
    inputs="one or more of: k, lr, epochs, patience, l2, bs, hist_tau_days, "
           "aux_weight, n_checkpoints, checkpoint_combine",
    outputs="(applied to the training run; no return value)",
    contexts=(ORCHESTRATOR, GENERATED), module=None, cost=FREE,
    mutates_pipeline=True,
    validation="Range-checked at the menu boundary. Changing several at once "
               "makes the result uninterpretable -- vary one.",
    failure_modes="Out-of-range values are rejected with a MenuError naming the "
                  "bound."))

register(Capability(
    name="ensemble_construction", kind=ENSEMBLE,
    purpose="Train k independent seeds of one configuration, rank-normalise "
            "their predictions and average them, then score the result.",
    when="Once a configuration is CONFIRMED, or has scored repeatedly. Averaging "
         "seeds of a configuration that is not actually good just buys a "
         "precise estimate of a mediocre number.",
    resolves="How much of the submitted score is seed variance rather than the "
             "model.",
    inputs="menu_choices for the configuration, and k (number of seeds)",
    outputs="{'primary','mean_member','sd_member','gain_over_mean_member',...}",
    contexts=(ORCHESTRATOR,), module=None, cost=ONE_RUN, mutates_pipeline=False,
    validation="Measure the gain against the MEAN member, never the best one. "
               "The best of k draws sits above the mean by construction, so "
               "that comparison reports a gain even when ensembling does "
               "nothing. Fix k before looking at any score, and use every seed "
               "trained -- dropping a weak member is selection on validation.",
    failure_modes="Costs k training runs. Fewer than 4 members makes the "
                  "average itself noisy.",
    instead="You do not implement this yourself. The loop schedules it as an "
            "experiment once something is worth ensembling; propose it by "
            "saying so in your hypothesis.",
    returns={"kind": "dict", "keys": ["primary", "mean_member",
                                      "gain_over_mean_member", "k"]},
    example="(scheduled by the loop; see agent/ensemble_experiment.py)"))

register(Capability(
    name="incumbent_cfg", kind=MODIFY,
    purpose="Build a COMPLETE, valid training config for the incumbent, plus "
            "its feature encoding, with any key overridden by keyword.",
    when="Before every direct call to train_numpy_fm. This is how you obtain a "
         "config; do not assemble one by hand.",
    resolves="Nothing on its own -- it is the correct starting point for any "
             "one-axis experiment.",
    inputs="splits, meta, choices=None, **overrides "
           "(e.g. hist_tau_days=7.0, k=32)",
    outputs="(cfg: dict, enc) ready to pass straight to train_numpy_fm",
    contexts=(ORCHESTRATOR, GENERATED), module="research_tools", cost=FREE,
    mutates_pipeline=True,
    validation="Change ONE key per experiment. Overriding several at once makes "
               "the result uninterpretable.",
    failure_modes="Needs the loaded splits and meta. Overriding 'multitask' "
                  "also rebuilds 'aux_tasks' for you.",
    returns={"kind": "tuple", "arity": 2, "names": ["cfg", "enc"]},
    example="cfg, enc = incumbent_cfg(splits, meta, hist_tau_days=7.0)",
    params={"required": ["splits", "meta"], "optional": ["choices"], "var_keyword": True}))

register(Capability(
    name="train_numpy_fm", kind=TRAIN,
    purpose="Train one model under an explicit cfg dict and return its "
            "validation and test score vectors.",
    when="The normal way a generated experiment produces predictions. Get the "
         "cfg from incumbent_cfg -- never hand-build it.",
    resolves="What a specific configuration actually scores.",
    inputs="cfg: dict, enc, splits, meta, log: callable. The cfg must contain "
           "ALL of: dim, k, lr, bs, epochs, patience, seed, loss, history, "
           "multitask, model, training, aux_tasks. Use "
           "`cfg, enc = incumbent_cfg(splits, meta, <overrides>)` to get one. "
           "Optionally set cfg['capture_epoch_scores'] = [] first; after "
           "training that list holds (epoch, valid_primary, scores) per epoch.",
    outputs="A DICT with keys: scores_valid, scores_test, model, hist. "
            "NOT a tuple -- `a, b = train_numpy_fm(...)` raises "
            "'too many values to unpack'.",
    contexts=(GENERATED,), module="train_lib", cost=ONE_RUN,
    mutates_pipeline=True,
    validation="One seed is one draw. A single-seed difference is not evidence.",
    failure_modes="An INCOMPLETE cfg raises KeyError listing every missing key "
                  "at once. An unrecognised extra key is silently ignored, so "
                  "check spelling against the override list. Destructuring the "
                  "return as a tuple raises ValueError.",
    returns={"kind": "dict",
             "keys": ["scores_valid", "scores_test", "model", "hist"]},
    example=(
        "from research_tools import incumbent_cfg\n"
        "cfg, enc = incumbent_cfg(splits, meta)\n"
        "cfg['seed'] = args.seed\n"
        "cfg['capture_epoch_scores'] = []          # optional, see below\n"
        "res = train_lib.train_numpy_fm(cfg, enc, splits, meta, print)\n"
        "valid_scores = res['scores_valid']        # NOT tuple unpacking\n"
        "test_scores  = res['scores_test']\n"
        "# cfg['capture_epoch_scores'] now holds one tuple PER EPOCH:\n"
        "#     (epoch:int, valid_primary:float, scores_valid:np.ndarray)\n"
        "# THREE elements, and the array is the VALID split only. There is no\n"
        "# per-epoch test vector -- test scores exist only in res['scores_test'].\n"
        "for epoch, valid_primary, valid_scores_at_epoch in "
        "cfg['capture_epoch_scores']:\n"
        "    ..."),
    params={"required": ["cfg", "enc", "splits", "meta", "log"],
            "optional": []}))

register(Capability(
    name="capture_epoch_scores", kind=MEASURE,
    purpose="Per-epoch validation scores from the training run your script is "
            "already doing, at no extra cost.",
    when="Whenever you want the epoch curve, or per-epoch predictions to feed "
         "selection_rule_test.",
    resolves="How validation moves across epochs, and whether a stopping or "
             "checkpoint rule generalises.",
    inputs="Set cfg['capture_epoch_scores'] = [] BEFORE calling train_numpy_fm. "
           "The list is filled in place during training.",
    outputs="A LIST with one 3-tuple per epoch: "
            "(epoch:int, valid_primary:float, scores_valid:np.ndarray). "
            "VALID SPLIT ONLY -- there is no per-epoch test vector.",
    contexts=(GENERATED,), module=None, cost=FREE, mutates_pipeline=False,
    validation="The per-epoch arrays are VALID-split predictions. Do not look "
               "for a test vector in them; take test predictions from "
               "train_numpy_fm's returned dict instead.",
    failure_modes="Expecting 2 or 4 elements per entry, or expecting a "
                  "(valid, test) pair inside an entry, raises at unpack time "
                  "AFTER a full training run has been paid for.",
    returns={"kind": "list_of_tuple", "arity": 3,
             "names": ["epoch", "valid_primary", "scores_valid"],
             "split": "valid"},
    example=("cfg['capture_epoch_scores'] = []\n"
             "train_lib.train_numpy_fm(cfg, enc, splits, meta, print)\n"
             "for epoch, valid_primary, scores_valid in "
             "cfg['capture_epoch_scores']:\n"
             "    ...   # exactly three names, valid split only")))

register(Capability(
    name="evaluate", kind=EVALUATE,
    purpose="The official metric: GAUC, nDCG@5 and their mean.",
    when="To score predictions on the valid split.",
    resolves="What a set of predictions is worth under the competition metric.",
    inputs="users: list, labels: array, scores: array",
    outputs="{'GAUC': float, 'nDCG@5': float, 'primary': float}",
    contexts=(ORCHESTRATOR, GENERATED), module="evaluate", cost=FREE,
    mutates_pipeline=False,
    validation="Never call this on the test split. Scoring ground truth; do not "
               "modify.",
    failure_modes="Misaligned array lengths raise; scores must be in the cache's "
                  "row order.",
    returns={"kind": "dict", "keys": ["GAUC", "nDCG@5", "primary"]},
    example='m = evaluate(list(users), labels, scores); print(m["primary"])',
    params={"required": ["users", "labels", "scores"], "optional": []}))


# ------------------------------------------------------------------ lookups ---
def all_capabilities() -> dict:
    return dict(_REGISTRY)


def get(name: str) -> Capability | None:
    return _REGISTRY.get(name)


def importable_names() -> set:
    return {c.name for c in _REGISTRY.values() if c.importable}


def orchestrator_names() -> set:
    return {c.name for c in _REGISTRY.values() if c.orchestrator_tool}


def orchestration_only() -> set:
    """Capabilities that exist ONLY between iterations. Calling one from
    generated code is the exact mistake this registry was built to stop."""
    return {c.name for c in _REGISTRY.values()
            if c.orchestrator_tool and not c.importable}


def modules_for_generated_code() -> set:
    return {c.module for c in _REGISTRY.values() if c.importable and c.module}


def expensive_names() -> set:
    return {c.name for c in _REGISTRY.values() if c.expensive}


# ------------------------------------------------------------------- render ---
def render_for_prompt(context: str = GENERATED) -> str:
    """Compact, machine-readable contract for the LLM.

    One line per capability, fixed field order, so it is cheap in tokens and
    unambiguous to parse. The agent's picture of its action space and the
    preflight check that enforces it are generated from the same registry --
    they cannot drift apart.
    """
    L = ["## CAPABILITY CONTRACT (authoritative -- this is your entire action space)",
         "Fields: name | kind | where | import | cost | purpose",
         "'where': ORCH = between iterations only; CODE = importable in your "
         "generated script; BOTH = either."]
    for name in sorted(_REGISTRY):
        c = _REGISTRY[name]
        where = ("BOTH" if c.orchestrator_tool and c.importable
                 else "ORCH" if c.orchestrator_tool else "CODE")
        imp = f"from {c.module} import {c.name}" if (c.importable and c.module) else "-"
        L.append(f"  {c.name} | {c.kind} | {where} | {imp} | {c.cost} | {c.purpose}")

    only = sorted(orchestration_only())
    if only:
        L += ["", "ORCHESTRATION-ONLY -- these DO NOT EXIST inside your generated "
                  "script. Importing or calling one there is a guaranteed crash:"]
        for n in only:
            c = _REGISTRY[n]
            L.append(f"  {n}: {c.instead or 'request it in the inspect phase instead.'}")

    # Return SHAPES, and worked examples for the two that every recorded Path B
    # crash came from. Prose about a return type is evidently not enough: the
    # agent read "{'scores_valid': ...}" and still destructured it as a tuple.
    L += ["", "RETURN SHAPES — check these before you write the call site."]
    for name in sorted(_REGISTRY):
        c = _REGISTRY[name]
        if not c.returns or not c.importable:
            continue
        r = c.returns
        if r.get("kind") == "dict":
            L.append(f"  {name} -> dict, NOT unpackable. keys: "
                     f"{', '.join(r.get('keys') or [])}")
        elif r.get("kind") == "tuple":
            L.append(f"  {name} -> tuple of {r.get('arity')}: "
                     f"{', '.join(r.get('names') or [])}")
        elif r.get("kind") == "list_of_tuple":
            L.append(f"  {name} -> list of {r.get('arity')}-tuples: "
                     f"({', '.join(r.get('names') or [])}); "
                     f"{r.get('split', '')} split only")
    for name in ("train_numpy_fm", "capture_epoch_scores"):
        c = _REGISTRY.get(name)
        if c and c.example:
            L += ["", f"WORKED EXAMPLE — {name}:"]
            L += [f"    {ln}" for ln in c.example.splitlines()]

    L += ["", "RULES.",
          "  1. In generated code you may import ONLY: "
          + ", ".join(sorted(modules_for_generated_code())) + ".",
          "  2. Every capability above is real and tested. If you need something "
          "not on this list, implement it yourself in plain Python -- do not "
          "guess at an API.",
          "  3. Your script is checked against this contract BEFORE it runs. A "
          "call to something that does not exist in your context, or a call "
          "site that disagrees with a return shape above, is caught in "
          "preflight and costs you no training time -- but it does cost an "
          "attempt, so read the list.",
          "  4. You can also read this contract AT RUNTIME from inside your "
          "script: `from research_tools import contract, describe` then "
          "`contract('train_numpy_fm')['returns']` or `print(describe(...))`. "
          "Ask rather than guess."]
    return "\n".join(L)


def as_json() -> str:
    """The contract as data, for tests and for anything that needs to reason
    about capabilities programmatically."""
    return json.dumps(
        {n: {"kind": c.kind, "purpose": c.purpose, "when": c.when,
             "resolves": c.resolves, "inputs": c.inputs, "outputs": c.outputs,
             "contexts": list(c.contexts), "module": c.module, "cost": c.cost,
             "orchestrator_tool": c.orchestrator_tool,
             "importable_from_generated_python": c.importable,
             "expensive": c.expensive, "mutates_pipeline": c.mutates_pipeline,
             "validation": c.validation, "failure_modes": c.failure_modes,
             "instead": c.instead}
         for n, c in sorted(_REGISTRY.items())}, indent=2)


CONTRACT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "runtime", "capability_contract.json")


def export_contract(path: str = CONTRACT_JSON) -> str:
    """Write the contract where GENERATED CODE can actually read it.

    The experiment subprocess has only `runtime/` and the starter kit on its
    PYTHONPATH, so it cannot import this module. Exporting the registry as data
    into runtime/ makes the contract executable from inside a generated script
    (`from research_tools import contract`) without duplicating the registry --
    there is still exactly one source of truth, and a test asserts the exported
    file matches it.

    Contains no data, no labels and no scores: it is a description of the API
    surface, so handing it to generated code cannot leak the hidden test.
    """
    payload = {
        "schema": "capability_contract/1",
        "capabilities": {
            n: {"kind": c.kind, "purpose": c.purpose, "when": c.when,
                "inputs": c.inputs, "outputs": c.outputs,
                "returns": c.returns, "example": c.example,
                "module": c.module, "cost": c.cost,
                "importable_from_generated_python": c.invoked_by_import,
                "orchestrator_only": c.orchestrator_tool and not c.importable,
                "validation": c.validation, "failure_modes": c.failure_modes}
            for n, c in sorted(_REGISTRY.items())},
        "importable_modules": sorted(modules_for_generated_code()),
        "orchestration_only": sorted(orchestration_only()),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return path


def contract_payload() -> dict:
    """The exported structure, regenerated. One code path, so no drift."""
    with open(export_contract()) as fh:
        return json.load(fh)


def render_full(name: str) -> str:
    """The complete contract entry for one capability, for a human reader."""
    c = _REGISTRY.get(name)
    if not c:
        return f"no such capability: {name}"
    return "\n".join([
        f"NAME                {c.name}",
        f"KIND                {c.kind}",
        f"PURPOSE             {c.purpose}",
        f"WHEN TO USE         {c.when}",
        f"RESOLVES            {c.resolves}",
        f"INPUTS              {c.inputs}",
        f"OUTPUTS             {c.outputs}",
        f"INVOCATION CONTEXT  {', '.join(c.contexts)}",
        f"ORCHESTRATOR TOOL   {'YES' if c.orchestrator_tool else 'NO'}",
        f"IMPORTABLE IN CODE  {'YES' if c.importable else 'NO'}"
        + (f"  ({c.module})" if c.importable and c.module else ""),
        f"EXPENSIVE           {'YES' if c.expensive else 'NO'}  ({c.cost})",
        f"MUTATES PIPELINE    {'YES' if c.mutates_pipeline else 'NO'}",
        f"VALIDATION          {c.validation}",
        f"FAILURE MODES       {c.failure_modes}",
    ] + ([f"INSTEAD             {c.instead}"] if c.instead else []))


# Keep the machine-readable copy in runtime/ in step with this registry. It is
# cheap, and a stale contract handed to generated code is worse than none.
try:
    export_contract()
except OSError:            # read-only checkout; the in-process registry still works
    pass


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--name")
    a = ap.parse_args()
    print(as_json() if a.json else
          render_full(a.name) if a.name else render_for_prompt())
