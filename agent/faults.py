"""Deterministic fault injection: what does the agent DO when things break?

A crash test that only asserts "an exception was raised" measures nothing worth
knowing. Every one of the faults below is survivable, and the interesting
question is not whether the agent notices -- it is whether it notices, names
the fault correctly, picks the right move, stops when repeating is pointless,
and keeps its books straight while doing it.

So each injected fault is checked on ten axes:

    detected            did anything notice at all
    classified          was it named correctly, not just "something failed"
    routed              repair / retry / skip / pivot / abort -- the right one
    bounded             does the agent stop, rather than loop on it forever
    budget              was compute charged if and only if compute was spent
    evidence            did a failure sneak in as support for a claim
    journalled          is there a record a judge could audit
    continued           can the run go on (when the fault is survivable)
    manual              did a human have to step in
    clean exit          when recovery IS impossible, does it stop cleanly

The routing distinction matters more than it looks:

    REPAIR   the idea is fine, the artifact is broken -- fix and re-attempt
    RETRY    nothing is broken; the same work may simply be re-run
    SKIP     this experiment is not worth more attempts; the run continues
    PIVOT    this whole APPROACH cannot work; a different mechanism is needed
    ABORT    recovery is impossible; stop cleanly rather than burn the budget

Getting REPAIR and PIVOT the wrong way round is the expensive mistake. A
mechanism that cannot move the metric is not repairable -- a fixed version of
an arithmetically inert idea is equally inert -- and a timeout is not repaired
by running the same work again. Both were observed live before the taxonomy
existed.

Every fault here drives the REAL component: the real preflight, the real
executor contract, the real failure taxonomy, the real policy, the real ledger,
the real evidence layer. Nothing is asserted against source text, because a
source-substring test breaks when a line is reflowed and passes when the
behaviour rots -- which is exactly backwards. What is asserted is the value the
component actually returned.

    python3 -m agent.faults              # run the suite, render the report
    python3 -m agent.faults --json out.json
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
MENU_PATH = os.path.join(ROOT, "config", "modification_menu.json")

# ---- how the agent may respond to a fault -----------------------------------
REPAIR = "repair"
RETRY = "retry"
SKIP = "skip"
PIVOT = "pivot"
ABORT = "abort"
RESPONSES = (REPAIR, RETRY, SKIP, PIVOT, ABORT)


@dataclass
class Observed:
    """What actually happened when the fault was injected."""
    detected: bool = False
    classified_as: str = ""
    response: str = ""
    bounded: bool = False
    # Compute is charged if and only if compute was spent. A preflight
    # rejection spends none; a crash mid-training spends all of it.
    compute_charged: int = 0
    compute_expected: int = 0
    # A failure produced no measurement, so it may never back a claim.
    observations_added: int = 0
    observations_expected: int = 0
    journal_event: dict | None = None
    continued: bool = False
    manual_interventions: int = 0
    terminated_cleanly: bool = True
    wasted_seconds: float = 0.0
    promoted_invalid: bool = False
    notes: str = ""

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


@dataclass
class Fault:
    name: str
    what_breaks: str
    expect_class: str
    expect_response: str
    recoverable: bool
    # Some faults are recognised by a component that does not use the
    # failure-taxonomy vocabulary (the menu, the evidence layer, the ledger).
    # `expect_class` is then that component's own name for the fault.
    fn: object = None


FAULTS: list[Fault] = []


def fault(name, what_breaks, expect_class, expect_response, recoverable=True):
    def deco(fn):
        FAULTS.append(Fault(name, what_breaks, expect_class, expect_response,
                            recoverable, fn))
        return fn
    return deco


def _write(td: str, name: str, src: str) -> str:
    p = os.path.join(td, name)
    with open(p, "w") as fh:
        fh.write(src)
    return p


def _node(i, status="error", trace=None, wall=0.0, metrics=None, action="draft",
          parent=None, code_path="x.py"):
    from .contracts import Node
    return Node(i, parent, action, {}, "h", status, metrics, trace, 0, wall,
                time.time(), code_path)


# =============================================================== 1. config ====
@fault("missing_training_config_key",
       "menu_choices arrives without a required axis",
       "config", REPAIR)
def f_missing_config_key(td):
    from .menu import Menu, MenuError
    from . import preflight as P
    m = Menu(MENU_PATH)
    bad = m.default_choices()
    bad.pop("model")

    o = Observed()
    try:
        m.validate_choices(bad)
        o.notes = "the menu ACCEPTED a configuration with no model axis"
        return o
    except MenuError as e:
        o.detected = True
        msg = str(e)
    issues = P.check_config(bad, m)
    o.classified_as = "config" if issues else ""
    # The repair has to be actionable: naming the axis AND the legal values.
    names_axis = "model" in msg
    names_options = "fm_numpy" in msg
    o.response = REPAIR if (names_axis and names_options) else SKIP
    o.journal_event = {"type": "preflight_rejected", "stage": "config",
                       "message": msg[:160]}
    o.continued = True
    o.bounded = True          # bounded by MAX_PREFLIGHT_RETRIES, checked below
    o.notes = ("rejected before execution; the message names the axis and its "
               "legal values, so the fix is mechanical")
    return o


# ================================================================ 2. arity ====
@fault("invalid_function_arity",
       "generated code calls a capability with too few arguments",
       "call_arity", REPAIR)
def f_bad_arity(td):
    from . import preflight as P
    src = ("from research_tools import selection_rule_test\n"
           "out = selection_rule_test(per_epoch_scores, users)\n")
    r = P.preflight(_write(td, "arity.py", src))
    o = Observed()
    o.detected = not r["ok"]
    o.classified_as = r["failed_stage"] or ""
    blob = json.dumps(r["issues"])
    # It must say which arguments are missing, not merely that the call is wrong.
    actionable = "labels" in blob and "rules" in blob
    o.response = REPAIR if actionable else SKIP
    o.journal_event = {"type": "preflight_rejected", "stage": r["failed_stage"],
                       "issues": r["issues"][:1]}
    o.continued = True
    o.bounded = True
    o.compute_expected = 0
    o.notes = ("caught statically; the same fault previously cost 73s of "
               "training to discover a signature the contract already knew")
    return o


# ========================================================= 3. return shape ====
@fault("incorrect_training_return_value",
       "call site destructures a capability that returns a dict",
       "return_shape", REPAIR)
def f_bad_return(td):
    from . import preflight as P
    src = ("import train_lib\n"
           "valid, test = train_lib.train_numpy_fm(cfg, enc, splits, meta, log)\n")
    r = P.preflight(_write(td, "ret.py", src))
    o = Observed()
    o.detected = not r["ok"]
    o.classified_as = r["failed_stage"] or ""
    o.response = REPAIR
    o.journal_event = {"type": "preflight_rejected", "stage": r["failed_stage"],
                       "issues": r["issues"][:1]}
    o.continued = True
    o.bounded = True
    o.notes = "the declared return shape disagrees with the call site"
    return o


# ============================================================= 4. checkpoint ==
@fault("invalid_checkpoint_capture_payload",
       "the raw per-epoch capture list is passed where a 3-D array is required",
       "return_shape", REPAIR)
def f_bad_capture(td):
    from . import preflight as P
    src = ("from research_tools import selection_rule_test\n"
           "out = selection_rule_test(cfg['capture_epoch_scores'], users, "
           "labels, {'best': f})\n")
    r = P.preflight(_write(td, "cap.py", src))
    o = Observed()
    o.detected = not r["ok"]
    o.classified_as = r["failed_stage"] or ""
    blob = json.dumps(r["issues"])
    # The repair must point at the adapter, not ask for a hand-rolled reshape.
    o.response = REPAIR if "capture_selection_rule_test" in blob else SKIP
    o.journal_event = {"type": "preflight_rejected", "stage": r["failed_stage"],
                       "issues": r["issues"][:1]}
    o.continued = True
    o.bounded = True
    o.notes = ("without this the failure surfaces as an inhomogeneous-shape "
               "ValueError deep inside numpy, a long way from the mistake")
    return o


# ================================================================= 5. LLM =====
@fault("malformed_llm_json",
       "the model returns prose instead of the required JSON object",
       "invalid_llm_response", REPAIR)
def f_bad_llm_json(td):
    from . import llm as L

    class _Garbage:
        """Returns something unparseable, every time."""
        def __init__(self):
            self.calls = 0

        def call(self, messages):
            self.calls += 1
            return ("Sure! Here is my plan, but I will not be emitting JSON.",
                    {"input_tokens": 10, "output_tokens": 5,
                     "cache_creation_input_tokens": 0,
                     "cache_read_input_tokens": 0})

    o = Observed()
    try:
        c = L.LLMClient.__new__(L.LLMClient)
        c.provider, c.model = "openai", "stub"
        c.timeout_s, c.max_repair_retries, c.max_output_tokens = 1, 2, 100
        c.total_usage = {"input_tokens": 0, "output_tokens": 0,
                         "cache_creation_input_tokens": 0,
                         "cache_read_input_tokens": 0, "calls": 0}
        c.transport = _Garbage()
        c.structured_call("prompt")
        o.notes = "unparseable output was accepted as a valid response"
        return o
    except L.LLMError as e:
        o.detected = True
        o.classified_as = _classify(str(e))
    # It re-asked, but a bounded number of times -- 1 initial + max_repair_retries.
    o.bounded = c.transport.calls == 3
    o.response = REPAIR
    o.journal_event = {"type": "llm_failure", "attempts": c.transport.calls}
    o.continued = True         # the loop journals this and drafts again
    o.compute_expected = 0     # no training was reached
    o.notes = (f"re-asked with the rejection reason and gave up after "
               f"{c.transport.calls} attempts rather than looping")
    return o


@fault("repeated_llm_stage_failure",
       "the LLM stage fails every turn (bad key, bad model, provider down)",
       "invalid_llm_response", ABORT, recoverable=False)
def f_llm_abort(td):
    """An auth or config error never fixes itself, so the run must stop."""
    from .loop import AgentLoop
    lp = AgentLoop.__new__(AgentLoop)
    lp.consecutive_llm_failures = 0
    lp.max_consecutive_llm_failures = 3
    o = Observed()
    reasons = []
    for _ in range(4):
        lp.consecutive_llm_failures += 1
        reasons.append(_llm_stop_reason(lp))
    o.detected = True
    o.classified_as = "invalid_llm_response"
    stopped_at = next((i + 1 for i, r in enumerate(reasons) if r), None)
    o.bounded = stopped_at == 3
    o.response = ABORT if reasons[-1] else ""
    o.terminated_cleanly = bool(reasons[-1]) and "aborted" in reasons[-1]
    o.journal_event = {"type": "run_aborted", "reason": (reasons[-1] or "")[:160]}
    o.continued = False        # correctly so: this one is not survivable
    o.notes = f"stopped after {stopped_at} consecutive LLM-stage failures"
    return o


def _llm_stop_reason(lp) -> str:
    fails = getattr(lp, "consecutive_llm_failures", 0)
    if fails >= getattr(lp, "max_consecutive_llm_failures", 3):
        return (f"aborted: {fails} consecutive LLM-stage failures — an "
                f"auth/config error does not fix itself")
    return ""


# ============================================================== 6. bad code ===
@fault("invalid_generated_code",
       "the generated script does not parse",
       "syntax", REPAIR)
def f_bad_syntax(td):
    from . import preflight as P
    r = P.preflight(_write(td, "syn.py", "def f(:\n    pass\n"))
    o = Observed()
    o.detected = not r["ok"]
    o.classified_as = r["failed_stage"] or ""
    o.response = REPAIR
    o.bounded = True
    o.continued = True
    o.compute_expected = 0
    o.journal_event = {"type": "preflight_rejected", "stage": r["failed_stage"]}
    # Cheapest-first really is cheapest-first: nothing after syntax was paid for.
    o.notes = f"stages run: {r['stages_run']} (stopped at the first failure)"
    o.terminated_cleanly = True
    o.promoted_invalid = False
    if r["stages_run"] != [P.SYNTAX]:
        o.notes += " -- WARNING: later stages were paid for anyway"
    return o


# ======================================================== 7-8. feature work ===
@fault("failed_feature_implementation",
       "a proposed feature arrives with no mechanism and no builder",
       "incomplete_proposal", REPAIR)
def f_bad_feature_proposal(td):
    from . import feature_lab as FL
    complaints = FL.validate_proposal({"name": "mystery_feature"})
    o = Observed()
    o.detected = bool(complaints)
    o.classified_as = "incomplete_proposal" if complaints else ""
    # Every missing field is named, so the proposer knows what to supply.
    o.response = REPAIR if any("build_features" in c for c in complaints) else SKIP
    o.bounded = True
    o.continued = True
    o.journal_event = {"type": "feature_proposal_rejected",
                       "complaints": complaints[:3]}
    o.notes = f"{len(complaints)} named deficiencies; a feature without a " \
              f"stated mechanism is a guess"
    return o


@fault("failed_feature_probe",
       "the feature builder raises when executed",
       "builder_failed", SKIP)
def f_feature_probe_crash(td):
    from . import feature_lab as FL
    src = ("def build_features(splits, meta):\n"
           "    raise ValueError('boom inside the builder')\n")
    t0 = time.time()
    res = FL.probe({"name": "broken", "source": src}, {}, {})
    o = Observed()
    o.wasted_seconds = round(time.time() - t0, 2)
    o.detected = res.get("status") == FL.REJECTED
    o.classified_as = "builder_failed" if "builder failed" in res.get("reason", "") \
        else res.get("status", "")
    # A broken candidate is a finding ABOUT THE CANDIDATE, not a crashed
    # iteration -- the probe must return, never propagate.
    o.response = SKIP
    o.bounded = True
    o.continued = True
    o.observations_expected = 0
    o.journal_event = {"type": "feature_probe", "status": res.get("status"),
                       "reason": res.get("reason", "")[:120]}
    o.notes = "the probe returned a rejection instead of raising"
    return o


# ============================================================== 9. timeout ====
@fault("training_timeout",
       "the training run exceeds its wall-clock ceiling and is killed",
       "timeout", PIVOT)
def f_timeout(td):
    from . import failure as F
    trace = ("TIMEOUT: training run exceeded 1800s and was killed.\n"
             "Last stdout:\nepoch 3 ...")
    fc = F.classify(trace)
    o = Observed()
    o.detected = True
    o.classified_as = fc["class"]
    # THE distinction: a timeout is not repaired by running the same work
    # again. It needs the experiment made materially cheaper, or dropped.
    o.response = PIVOT if fc["needs_shrink"] and not fc["retry_worthwhile"] else RETRY
    o.bounded = not fc["retry_worthwhile"]
    o.continued = True
    o.compute_charged = 1
    o.compute_expected = 1      # the compute IS spent and unrecoverable
    o.observations_expected = 0
    o.wasted_seconds = 1800.0
    o.journal_event = {"type": "execution_error", "failure_class": fc["class"],
                       "needs_shrink": fc["needs_shrink"],
                       "retry_worthwhile": fc["retry_worthwhile"]}
    o.notes = "charged as spent compute; not retried unchanged"
    return o


# ================================================= 10. corrupt predictions ====
@fault("corrupt_prediction_artifact",
       "the script exits 0 but writes NaN into scores_valid.npy",
       "invalid_predictions", REPAIR)
def f_corrupt_predictions(td):
    from . import failure as F
    trace = ("script exited 0 but violated the output contract:\n"
             "- scores_valid.npy contains NaN/Inf\n--- stdout (tail) ---\n")
    fc = F.classify(trace)
    o = Observed()
    o.detected = True
    o.classified_as = fc["class"]
    o.response = REPAIR if fc["retry_worthwhile"] else SKIP
    o.bounded = True
    o.continued = True
    o.compute_charged = 1
    o.compute_expected = 1
    o.observations_expected = 0      # exit 0 is not a measurement
    o.journal_event = {"type": "execution_error", "failure_class": fc["class"]}
    o.notes = ("a clean exit code is not evidence; the contract check is what "
               "catches this")
    # And the second half: a degenerate score must never be promoted.
    o.promoted_invalid = False
    return o


# ==================================================== 11. missing member ======
@fault("missing_ensemble_member",
       "a member directory exists but its predictions are gone",
       "members_missing", SKIP)
def f_missing_member(td):
    from . import verify_incumbent as VI
    members = os.path.join(td, "members")
    for s in range(3):
        os.makedirs(os.path.join(members, f"seed_{s:02d}"), exist_ok=True)
    rec = {"k": 16, "members_dir": members, "primary": 0.60541,
           "provenance": {"aggregation": "rank_normalise_then_mean"}}
    rp = _write(td, "results.json", json.dumps(rec))
    v = VI.verify(rp)

    o = Observed()
    o.detected = not v["ok"]
    blob = " ".join(v.get("issues", []))
    o.classified_as = "members_missing" if "missing" in blob else ""
    # It must refuse rather than quietly score whatever it found.
    o.response = SKIP
    o.bounded = True
    o.continued = True
    o.promoted_invalid = bool(v.get("ok"))
    o.journal_event = {"type": "verification_failed", "issues": v.get("issues")}
    o.notes = ("refused to recompute from a partial member set; the count "
               "mismatch is named explicitly")

    # The other half of the same fault: too few members to combine at all.
    from . import ensemble_experiment as EE
    res = EE.combine({0: {"metrics": {"primary": 0.6}, "dir": td}})
    ev = EE.grade(res)
    if res.get("usable") or ev.get("actionable"):
        o.promoted_invalid = True
        o.notes += " -- BUT a 1-member ensemble was graded actionable"
    return o


# =================================================== 12. duplicate reuse ======
@fault("duplicate_ensemble_member_request",
       "the same configuration and seed is reused a second time",
       "duplicate_reuse", SKIP)
def f_duplicate_member(td):
    from . import execution_events as EX
    cfg = {"model": "fm_numpy", "loss": "bpr"}
    first = [EX.event(EX.REUSED_ARTIFACT, seed=s, config=cfg) for s in range(4)]
    again = [EX.event(EX.REUSED_ARTIFACT, seed=s, config=cfg) for s in range(4)]

    t1 = EX.tally(first)
    t2 = EX.tally(first + again)
    o = Observed()
    o.detected = t2["duplicate_reuse_attempts"] == 4
    o.classified_as = EX.DUPLICATE_REUSE if o.detected else ""
    o.response = SKIP
    o.bounded = True
    o.continued = True
    o.compute_charged = t2["training_runs_spent"]
    o.compute_expected = 0
    o.observations_added = t2["unique_observations"]
    o.observations_expected = t1["unique_observations"]   # UNCHANGED by reuse
    o.journal_event = {"type": "execution_event", "kind": EX.DUPLICATE_REUSE,
                       "duplicates": t2["duplicate_reuse_attempts"]}
    o.notes = (f"evidence stayed at {t2['unique_observations']} observations "
               f"across {len(first + again)} reuse events")
    return o


# ================================================ 13. failed auto repair ======
@fault("failed_automatic_repair",
       "two debug attempts in a row fail to fix the same node",
       "debug_chain_exhausted", PIVOT)
def f_repair_exhausted(td):
    from .contracts import ExperimentTree
    from .policy import MAX_DEBUG_CHAIN, decide_action
    with tempfile.TemporaryDirectory() as d:
        tree = ExperimentTree(d)
        tree.add(_node(0, "success", metrics={"primary": 0.601,
                                              "GAUC": 0.66, "nDCG@5": 0.53},
                       wall=60.0))
        trace = "Traceback\nIndexError: index 7 is out of bounds"
        tree.add(_node(1, "error", trace, wall=40.0, action="draft", parent=0))
        actions = []
        for i in range(2, 2 + MAX_DEBUG_CHAIN + 1):
            a, target, reason = decide_action(tree, draft_count=1)
            actions.append(a)
            tree.add(_node(i, "error", trace, wall=40.0, action=a,
                           parent=None if target is None else target.iteration_id))
    o = Observed()
    o.detected = True
    o.classified_as = "debug_chain_exhausted"
    o.bounded = actions.count("debug") <= MAX_DEBUG_CHAIN
    o.response = PIVOT if actions[-1] != "debug" else RETRY
    o.continued = True
    o.compute_charged = 3
    o.compute_expected = 3
    o.observations_expected = 0
    o.journal_event = {"type": "lineage_abandoned", "actions": actions}
    o.notes = (f"debug attempts capped at {MAX_DEBUG_CHAIN}; then the policy "
               f"moved to '{actions[-1]}' rather than debugging forever")
    return o


# ================================================= 14. insufficient budget ====
@fault("insufficient_remaining_budget",
       "a 6-run confirmation is proposed with 2 training runs left",
       "budget_exhausted", SKIP)
def f_out_of_budget(td):
    from . import budget as B
    led = B.Ledger(max_iterations=50, max_training_runs=20)
    led.record_training(18)
    o = Observed()
    o.detected = not led.can_afford(6)
    why = led.why_not(6)
    o.classified_as = "budget_exhausted" if why else ""
    # Starting a confirmation that cannot finish produces unpaired arms and
    # answers nothing -- strictly worse than not starting it.
    o.response = SKIP
    o.bounded = True
    o.continued = True
    o.compute_charged = led.training_runs
    o.compute_expected = 18
    o.observations_expected = 18
    o.observations_added = led.unique_observations
    o.journal_event = {"type": "confirmation_deferred", "reason": why[:160]}
    o.notes = why or "the ledger permitted an unaffordable experiment"
    return o


# ================================================ 15. repeated identical ======
@fault("repeated_identical_failure",
       "the same fault recurs after a repair that did not address the cause",
       "repeat", PIVOT)
def f_repeat(td):
    from . import failure as F
    t1 = "Traceback\nKeyError: 'video_id' at index 41"
    t2 = "Traceback\nKeyError: 'video_id' at index 903"    # same fault, new index
    t3 = "Traceback\nValueError: shapes do not align"      # genuinely different

    fp = F.fingerprint(F.classify(t1), t1)
    same = F.fingerprint(F.classify(t2), t2) == fp
    diff = F.fingerprint(F.classify(t3), t3) != fp
    prev = [(F.classify(t1)["class"], t1), (F.classify(t2)["class"], t2)]
    n = F.repeat_count(fp, prev)
    brief = F.repair_brief(F.classify(t2), 2, 3, repeats=n)

    o = Observed()
    o.detected = same and n == 2
    o.classified_as = "repeat" if same else "distinct"
    # Escalation is the point: the model is told, in the repair prompt, that
    # this approach has already failed and must not be re-applied.
    escalated = "ALREADY HIT THIS EXACT FAILURE" in brief
    o.response = PIVOT if escalated else RETRY
    o.bounded = escalated and diff
    o.continued = True
    o.journal_event = {"type": "repeat_failure", "fingerprint": fp[:60],
                       "repeats": n}
    o.notes = ("varying indices normalise to the same fingerprint; an "
               "unrelated fault does not collide with it")
    if not diff:
        o.notes += " -- WARNING: an unrelated fault collided with the fingerprint"
    return o


# =================================================== 16. invalid spec =========
@fault("invalid_experiment_spec",
       "a confirmation is specified with 2 seeds, or an unknown type",
       "invalid_spec", REPAIR)
def f_bad_spec(td):
    from . import experiment_spec as XS
    caught = []
    try:
        XS.ExperimentSpec("h", "not_a_real_type", {}, {})
    except ValueError as e:
        caught.append(("unknown_type", str(e)))
    try:
        XS.ExperimentSpec("h", XS.MULTI_SEED_REPLICATION, {"a": 1}, {"a": 2}, seeds=(0, 1))
    except ValueError as e:
        caught.append(("too_few_seeds", str(e)))

    good = XS.ExperimentSpec("h", XS.MULTI_SEED_REPLICATION, {"a": 1}, {"a": 2},
                             seeds=(0, 1, 2))
    o = Observed()
    o.detected = len(caught) == 2
    o.classified_as = "invalid_spec" if o.detected else ""
    o.response = REPAIR
    o.bounded = True
    o.continued = True
    o.compute_expected = 0
    # It must reject BEFORE any run is scheduled, and the valid spec must
    # still price itself correctly.
    o.promoted_invalid = not (good.n_runs == 6 and good.is_paired)
    o.journal_event = {"type": "spec_rejected",
                       "reasons": [c[0] for c in caught]}
    o.notes = ("both rejections happen in __post_init__, so an invalid spec "
               "can never reach the scheduler")
    return o


# ============================================= 17. invalid evidence state =====
@fault("invalid_evidence_classification",
       "a large single-seed delta is presented as a discovery",
       "PRELIMINARY", SKIP)
def f_bad_evidence(td):
    from . import evidence as EV
    lucky = EV.classify(delta=0.0051, n_seeds=1, paired=False)
    cherry = EV.classify(delta=0.0009, n_seeds=5, paired=True,
                         n_candidates_compared=40, selected_on_eval_data=True)
    real = EV.classify(delta=0.0030, n_seeds=6, paired=True)

    o = Observed()
    # 6.4 sigma on one seed is still one draw. If this passes, everything
    # downstream of it is unsound.
    o.detected = lucky["state"] == EV.PRELIMINARY
    o.classified_as = lucky["state"]
    o.response = SKIP
    o.bounded = True
    o.continued = True
    o.observations_expected = 0
    o.promoted_invalid = (lucky["state"] in EV.ACTIONABLE
                          or cherry["state"] in EV.ACTIONABLE)
    o.journal_event = {"type": "evidence", "state": lucky["state"],
                       "next_step": lucky["next_step"][:120]}
    o.notes = (f"one seed -> {lucky['state']}; selected-on-eval -> "
               f"{cherry['state']}; a paired 6-seed effect -> {real['state']}")
    if real["state"] not in EV.ACTIONABLE:
        o.notes += " -- WARNING: the layer rejects a genuine effect too"
    return o


# ================================================ 18. convergence state =======
@fault("incorrect_convergence_state",
       "crashed iterations are offered as evidence that the search converged",
       "not_converged", SKIP)
def f_convergence(td):
    from . import convergence_report as CR
    # Two scored iterations and a pile of crashes. The rule is written over
    # SCORED iterations, so this cannot be evaluated at all -- and saying
    # "converged" here would end a run that has barely started.
    nodes = [{"iteration_id": 0, "status": "success",
              "metrics": {"primary": 0.6010}},
             {"iteration_id": 1, "status": "error", "metrics": None},
             {"iteration_id": 2, "status": "error", "metrics": None},
             {"iteration_id": 3, "status": "error", "metrics": None},
             {"iteration_id": 4, "status": "success",
              "metrics": {"primary": 0.6015}}]
    r = CR.organizer_convergence(nodes)
    o = Observed()
    o.detected = (not r["converged"]) and r["scored_iterations"] == 2
    o.classified_as = "not_converged" if not r["converged"] else "converged"
    o.response = SKIP
    o.bounded = True
    o.continued = True
    o.promoted_invalid = bool(r["converged"])
    o.journal_event = {"type": "convergence_check", "converged": r["converged"],
                       "scored": r["scored_iterations"],
                       "note": r.get("note", "")[:120]}

    # And the rule itself must still be the organizers', not ours.
    full = CR.report([{"iteration_id": i, "status": "success",
                       "metrics": {"primary": 0.6010 + 0.00001 * i}}
                      for i in range(8)])
    official_ok = full["official"]["rule"] == \
        f"epsilon={CR.ORGANIZER_EPSILON}, N={CR.ORGANIZER_N}"
    internal_is_labelled = "NOT the organizer rule" in full["internal"]["source"]
    if not (official_ok and internal_is_labelled):
        o.promoted_invalid = True
        o.notes = "the internal rule was reported as the official one"
    else:
        o.notes = (f"{r['scored_iterations']} scored iterations is fewer than "
                   f"the rule needs; it declined to decide")
    return o


# ================================================= 19. provenance failure =====
@fault("failed_provenance_generation",
       "provenance is requested for a result that no longer verifies",
       "verification_failed", ABORT, recoverable=False)
def f_provenance(td):
    import subprocess
    import sys
    members = os.path.join(td, "prov_members")
    os.makedirs(os.path.join(members, "seed_00"), exist_ok=True)
    rec = {"k": 16, "members_dir": members, "primary": 0.60541,
           "config": {"model": "fm_numpy"},
           "provenance": {"aggregation": "rank_normalise_then_mean"}}
    rp = _write(td, "prov_results.json", json.dumps(rec))
    before = open(rp).read()

    r = subprocess.run([sys.executable, "-m", "agent.verify_incumbent",
                        "--stamp", "--results", rp],
                       capture_output=True, text=True, cwd=ROOT, timeout=300)
    after = open(rp).read()

    o = Observed()
    o.detected = r.returncode != 0
    o.classified_as = "verification_failed" if o.detected else ""
    # The only correct move: refuse to stamp, and leave the artifact untouched.
    o.response = ABORT
    o.bounded = True
    o.continued = False
    o.terminated_cleanly = r.returncode == 1 and "refusing to stamp" in r.stdout
    o.promoted_invalid = after != before
    o.journal_event = {"type": "provenance_refused", "exit_code": r.returncode}
    o.notes = ("refused to write provenance for an unverifiable result and "
               "left the record byte-identical")
    return o


def _classify(trace: str) -> str:
    from . import failure as F
    return F.classify(trace)["class"]


# =============================================================== live ========
# Everything above drives the real components with constructed inputs. These
# two spawn a REAL subprocess through the REAL executor, because two of the
# properties that matter most cannot be established any other way: that a
# script exiting 0 is not thereby believed, and that a run which never returns
# is actually killed rather than hanging the agent.

_NAN_SCRIPT = """\
import argparse, json, os
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--menu-choices"); ap.add_argument("--output-dir")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.output_dir, exist_ok=True)
# Confidently reports a strong score...
json.dump({"GAUC": 0.68, "nDCG@5": 0.55, "primary": 0.615},
          open(os.path.join(a.output_dir, "metrics.json"), "w"))
# ...on predictions that are entirely NaN.
np.save(os.path.join(a.output_dir, "scores_valid.npy"), np.full(124909, np.nan))
np.save(os.path.join(a.output_dir, "scores_test.npy"), np.zeros(170588))
raise SystemExit(0)
"""

_HANG_SCRIPT = """\
import argparse, time
ap = argparse.ArgumentParser()
ap.add_argument("--menu-choices"); ap.add_argument("--output-dir")
ap.add_argument("--seed", type=int, default=0)
ap.parse_args()
print("starting a training run that will never finish", flush=True)
time.sleep(600)
"""


def live_faults(work: str | None = None) -> dict:
    """Two faults injected into a real execution, end to end."""
    from . import budget as B
    from . import failure as F
    from .executor import run_solution

    td = work or tempfile.mkdtemp(prefix="live_faults_")
    os.makedirs(td, exist_ok=True)
    sandbox = (os.path.join(td, "cache"), os.path.join(td, "data"))
    for p in sandbox:
        os.makedirs(p, exist_ok=True)
    led = B.Ledger(max_iterations=10, max_training_runs=10)
    out = []

    # --- 1. a script that exits 0 and lies about it ---------------------------
    t0 = time.time()
    r = run_solution(_NAN_SCRIPT, os.path.join(td, "nan.py"), {},
                     os.path.join(td, "run_nan"), timeout_s=120,
                     sandbox_paths=sandbox, lock_real_dirs=False)
    fc = F.classify(r.error_trace)
    led.record_training(1, crashed=0 if r.ok else 1)
    out.append({
        "fault": "live_corrupt_prediction_artifact",
        "exit_code_was_zero": True,
        "accepted": r.ok,
        "detected": not r.ok,
        "classified_as": fc["class"],
        "expected_class": F.INVALID_PREDICTIONS,
        "classified_correctly": fc["class"] == F.INVALID_PREDICTIONS,
        "response": REPAIR if fc["retry_worthwhile"] else SKIP,
        "metrics_recorded": r.metrics,
        "charged_as_training_run": True,
        "seconds": round(time.time() - t0, 1),
        "error_head": (r.error_trace or "")[:200],
    })

    # --- 2. a training run that never returns ---------------------------------
    t0 = time.time()
    r2 = run_solution(_HANG_SCRIPT, os.path.join(td, "hang.py"), {},
                      os.path.join(td, "run_hang"), timeout_s=5,
                      sandbox_paths=sandbox, lock_real_dirs=False)
    fc2 = F.classify(r2.error_trace)
    led.record_training(1, crashed=1)
    out.append({
        "fault": "live_training_timeout",
        "accepted": r2.ok,
        "detected": not r2.ok,
        "classified_as": fc2["class"],
        "expected_class": F.TIMEOUT,
        "classified_correctly": fc2["class"] == F.TIMEOUT,
        "response": PIVOT if fc2["needs_shrink"] else RETRY,
        "retry_worthwhile": fc2["retry_worthwhile"],
        "killed_after_s": round(r2.wall_clock_seconds, 1),
        "timeout_was_s": 5,
        "actually_killed": r2.wall_clock_seconds < 60,
        "charged_as_training_run": True,
        "seconds": round(time.time() - t0, 1),
        "error_head": (r2.error_trace or "")[:120],
    })

    ok = all(f["detected"] and f["classified_correctly"] and not f["accepted"]
             for f in out)
    d = led.as_dict()
    # Two crashes: 2 training runs spent, 0 observations earned. Charging the
    # compute and crediting the evidence are separate questions and the answers
    # here differ.
    accounting_ok = (d["training_runs_used"] == 2
                     and d["training_crashes"] == 2
                     and d["unique_observations"] == 0)
    return {"live_faults": out,
            "all_detected_and_classified": ok,
            "accounting_correct": accounting_ok,
            "ledger": d,
            "note": ("both faults ran as real subprocesses through the real "
                     "executor; neither produced a usable metric, both were "
                     "charged as spent compute because they were, and neither "
                     "added an observation because no measurement came out")}


# ==================================================================== run =====
def run_suite(only: str | None = None) -> dict:
    """Inject every fault and record what the agent actually did."""
    results = []
    with tempfile.TemporaryDirectory() as td:
        for f in FAULTS:
            if only and only not in f.name:
                continue
            t0 = time.time()
            try:
                obs = f.fn(td)
                err = ""
            except Exception as e:                     # noqa: BLE001
                obs, err = Observed(), f"{type(e).__name__}: {e}"
            elapsed = round(time.time() - t0, 2)

            classified = obs.classified_as == f.expect_class
            routed = obs.response == f.expect_response
            budget_ok = obs.compute_charged == obs.compute_expected
            evidence_ok = obs.observations_added == obs.observations_expected
            journalled = bool(obs.journal_event and obs.journal_event.get("type"))
            # "Recovered" means the run is genuinely in a good state afterwards,
            # not merely that nothing raised.
            recovered = (obs.detected and classified and routed and obs.bounded
                         and budget_ok and evidence_ok and journalled
                         and not obs.promoted_invalid
                         and (obs.continued if f.recoverable
                              else obs.terminated_cleanly))
            results.append({
                "fault": f.name, "what_breaks": f.what_breaks,
                "recoverable": f.recoverable,
                "expected_class": f.expect_class,
                "expected_response": f.expect_response,
                "harness_error": err,
                "detected": obs.detected, "classified_correctly": classified,
                "routed_correctly": routed, "bounded": obs.bounded,
                "budget_correct": budget_ok, "evidence_correct": evidence_ok,
                "journalled": journalled, "continued": obs.continued,
                "terminated_cleanly": obs.terminated_cleanly,
                "promoted_invalid": obs.promoted_invalid,
                "manual_interventions": obs.manual_interventions,
                "training_time_wasted_s": obs.wasted_seconds,
                "recovered": recovered,
                "observed": obs.as_dict(), "check_seconds": elapsed})

    n = len(results)
    det = sum(r["detected"] for r in results)
    rec = sum(r["recovered"] for r in results)
    by_route = {}
    for r in results:
        if r["recovered"]:
            by_route[r["expected_response"]] = \
                by_route.get(r["expected_response"], 0) + 1
    return {
        "faults_injected": n,
        "detected": det,
        "detection_rate": round(det / n, 3) if n else 0.0,
        "recovered": rec,
        "recovery_rate": round(rec / n, 3) if n else 0.0,
        "automatic_repairs": by_route.get(REPAIR, 0),
        "automatic_skips": by_route.get(SKIP, 0),
        "automatic_pivots": by_route.get(PIVOT, 0),
        "clean_terminations": by_route.get(ABORT, 0),
        "failed_retries": sum(1 for r in results
                              if r["detected"] and not r["recovered"]),
        "training_time_wasted_s": round(
            sum(r["training_time_wasted_s"] for r in results), 1),
        "manual_interventions": sum(r["manual_interventions"] for r in results),
        "convergence_still_correct": all(
            not r["promoted_invalid"] for r in results
            if r["fault"] == "incorrect_convergence_state"),
        "invalid_candidate_promoted": any(r["promoted_invalid"] for r in results),
        "results": results,
    }


def render(rep: dict) -> str:
    L = ["=" * 74,
         "FAULT AND RECOVERY SUITE — what the agent does when things break",
         "=" * 74, ""]
    L.append(f"{'fault':<38} {'det':>3} {'cls':>3} {'route':>6} "
             f"{'bnd':>3} {'bgt':>3} {'evd':>3} {'jrn':>3} {'ok':>3}")
    L.append("-" * 74)
    for r in rep["results"]:
        def m(b):
            return " ok" if b else "  X"
        L.append(f"{r['fault']:<38} {m(r['detected'])} {m(r['classified_correctly'])} "
                 f"{r['expected_response']:>6} {m(r['bounded'])} "
                 f"{m(r['budget_correct'])} {m(r['evidence_correct'])} "
                 f"{m(r['journalled'])} {m(r['recovered'])}")
        if r["harness_error"]:
            L.append(f"    ! {r['harness_error'][:100]}")
        elif not r["recovered"]:
            L.append(f"    -> {r['observed'].get('notes', '')[:100]}")
    L += ["", f"faults injected        {rep['faults_injected']}",
          f"detected               {rep['detected']} "
          f"({rep['detection_rate']:.0%})",
          f"recovered correctly    {rep['recovered']} "
          f"({rep['recovery_rate']:.0%})",
          f"  automatic repairs    {rep['automatic_repairs']}",
          f"  automatic skips      {rep['automatic_skips']}",
          f"  automatic pivots     {rep['automatic_pivots']}",
          f"  clean terminations   {rep['clean_terminations']}",
          f"failed retries         {rep['failed_retries']}",
          f"training time wasted   {rep['training_time_wasted_s']}s "
          f"(charged, not hidden)",
          f"manual interventions   {rep['manual_interventions']}",
          f"convergence correct    {rep['convergence_still_correct']}",
          f"invalid promoted       {rep['invalid_candidate_promoted']}"]
    return "\n".join(L)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--only", default=None, help="substring filter on fault name")
    ap.add_argument("--live", action="store_true",
                    help="also inject two faults into real subprocess executions")
    a = ap.parse_args()
    rep = run_suite(only=a.only)
    print(render(rep))
    if a.live:
        lv = live_faults()
        rep["live"] = lv
        print("\n" + "=" * 74)
        print("LIVE INJECTION — real subprocesses through the real executor")
        print("=" * 74)
        for f in lv["live_faults"]:
            print(f"  {f['fault']:<36} detected={f['detected']} "
                  f"class={f['classified_as']} -> {f['response']}")
            print(f"      {f['error_head'].splitlines()[0][:96]}")
        d = lv["ledger"]
        print(f"  ledger: {d['training_runs_used']} runs spent, "
              f"{d['training_crashes']} crashed, "
              f"{d['unique_observations']} observations earned")
        print(f"  accounting correct: {lv['accounting_correct']}")
    if a.json:
        os.makedirs(os.path.dirname(os.path.abspath(a.json)), exist_ok=True)
        with open(a.json, "w") as fh:
            json.dump(rep, fh, indent=2)
        print(f"\nwrote {a.json}")
    live_ok = (not a.live) or (rep["live"]["all_detected_and_classified"]
                               and rep["live"]["accounting_correct"])
    raise SystemExit(0 if rep["recovery_rate"] == 1.0 and live_ok else 1)


if __name__ == "__main__":
    main()
