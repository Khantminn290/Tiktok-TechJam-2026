"""Autonomous feature discovery: probe a candidate before paying for training.

The agent could previously only SELECT features, from a human-authored menu
whose entries already carried their own justification. It could not invent one.
That is the weakest link in the loop this project is supposed to automate --
"engineer features" is one of the two stages that is carried out almost
entirely in code, and it was the one stage the agent did not write code for.

This module supplies the missing half. The agent writes a build_features()
function; this runs it, measures whether it is worth a training run, and
records the verdict so the same idea is not rediscovered.

Why a probe rather than just training: a training run costs ~70s and answers
one question badly (a single seed against a 0.0008 noise floor). The probe
costs seconds and answers the question that actually gates the decision --
does this feature carry ranking signal the incumbent does not already have?
Screening first is what makes autonomous feature search affordable rather than
a random walk with a training run per step.

The measurements, in the order they can kill a candidate:

  1. LEAKAGE      -- AST review of the builder, reusing agent.leakage_check.
                     A feature that reads evaluation labels is refused, not
                     scored.
  2. VALIDITY     -- right length per split, finite fraction, cardinality.
  3. WITHIN-USER  -- does it vary INSIDE one user's impression list? Both
                     metrics rank within a user, so a feature constant across
                     a user's rows is worth exactly 0.5 AUC by construction.
                     Measured here at exactly 0.5000 for user-level scalars.
                     This kills a whole class of plausible-sounding ideas for
                     free.
  4. REDUNDANCY   -- correlation with the features the model already has.
  5. INCREMENTAL  -- residual value over the incumbent's own predictions, via
                     agent.error_analysis.feature_probe. This is the one that
                     decides, because a strong standalone number means nothing
                     if the embeddings already encode it: item long_view rate
                     scores 0.639 alone and HURTS at every blend weight.
  6. COST         -- wall-clock to build.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)

NOISE = 0.0008
REGISTRY = os.path.join(ROOT, "logs", "feature_registry.jsonl")

# lifecycle
PROPOSED, PROBED, PROMISING = "PROPOSED", "PROBED", "PROMISING"
VALIDATED, REJECTED, RETIRED = "VALIDATED", "REJECTED", "RETIRED"

# Ratio of train-side to eval-side within-user variation above which a feature
# is refused. Both earlier failures measured ~6x.
SKEW_LIMIT = 2.5

REQUIRED_FIELDS = ("name", "hypothesis", "mechanism", "incremental_value",
                   "leakage_check", "source_columns", "source")


# Outcome columns. Reading any of these for a split other than "train" turns
# the feature into (a function of) the label being predicted.
LABEL_COLUMNS = ("long_view", "is_click", "is_like", "is_forward",
                 "play_time_ms", "is_follow", "is_comment", "is_hate",
                 "is_profile_enter")


def label_leak_findings(source: str) -> list:
    """Builder-specific leakage: a LABEL read from any split except train.

    The general leakage checker is tuned for solution SCRIPTS, where reading
    validation labels is legitimate (early stopping scores against them). For a
    feature BUILDER it is not: whatever it returns becomes an input to the model
    for that same row. Verified gap -- a builder returning
    splits[s]["long_view"] for every split passes the general checker cleanly.

    The rule enforced here: labels may be read from splits["train"] only.
    """
    import ast
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"does not parse: {e}"]
    # `tr = splits["train"]` then `tr["long_view"]` is the ordinary way to write
    # this and must not be flagged, so track names bound to the train split.
    train_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and _is_train_split(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    train_aliases.add(t.id)
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        key = node.slice
        if not (isinstance(key, ast.Constant) and key.value in LABEL_COLUMNS):
            continue
        base = node.value
        ok = (_is_train_split(base)
              or (isinstance(base, ast.Name) and base.id in train_aliases))
        if not ok:
            where = getattr(base, "id", None) or type(base).__name__
            out.append(
                f"reads label {key.value!r} from something other than "
                f"splits['train'] (base: {where}, line {node.lineno}) -- a "
                f"feature may only use TRAIN-window outcomes")
    return out


def _is_train_split(node) -> bool:
    """True for the expression splits["train"]."""
    import ast as _ast
    return (isinstance(node, _ast.Subscript)
            and isinstance(node.slice, _ast.Constant)
            and node.slice.value == "train")


def validate_proposal(p: dict) -> list:
    """Missing fields, as a list of complaints. A feature without a stated
    mechanism is a guess, and guesses are what this module exists to filter."""
    missing = [f for f in REQUIRED_FIELDS
               if not str((p or {}).get(f, "")).strip()]
    out = [f"missing or empty: {f}" for f in missing]
    if p and "build_features" not in str(p.get("source", "")):
        out.append("source must define build_features(splits, meta)")
    return out


def _within_user_variation(u, v) -> float:
    """Fraction of multi-row users whose value is not constant."""
    u = np.asarray(u); v = np.asarray(v, dtype=np.float64)
    o = np.argsort(u, kind="stable")
    ub, vb = u[o], v[o]
    if not np.isfinite(vb).any():
        return 0.0
    starts = np.searchsorted(ub, np.unique(ub))
    ends = np.r_[starts[1:], len(ub)]
    scale = float(np.nanmax(vb) - np.nanmin(vb))
    tol = max(1e-12, 1e-9 * scale)
    out = []
    for s_, e_ in zip(starts, ends):
        if e_ - s_ <= 1:
            continue
        w = vb[s_:e_]
        if not np.isfinite(w).any():
            out.append(False)
            continue
        out.append(float(np.nanmax(w) - np.nanmin(w)) > tol)
    return float(np.mean(out)) if out else 0.0


def key_diagnostics(splits: dict, meta: dict) -> str:
    """Which (user x KEY) features are even estimable, as prompt evidence.

    Derived from the post-mortem on this project's first two feature failures.
    Both were user-by-author, both measured within-user variation of ~0.165 on
    valid, and the cause was not the idea: `author` varies across 99.1% of a
    user's impressions, but only 3.4% of validation rows have that (user,
    author) pair in train. The other 96.6% fall back to a user-constant value,
    and a user-constant feature is worth exactly 0.5 AUC. A user-by-KEY feature
    is only estimable when the KEY has low enough cardinality to be observed
    per user.
    """
    va, tr = splits["valid"], splits["train"]
    u, nU = va["user"], meta["field_dims"]["user"]
    edges = np.quantile(tr["duration_ms"], np.linspace(0, 1, 11)[1:-1])
    keys = {
        "video": (va["video"], tr["video"]),
        "author": (va["author"], tr["author"]),
        "tab": (va["tab"], tr["tab"]),
        "duration_bucket": (np.searchsorted(edges, va["duration_ms"]),
                            np.searchsorted(edges, tr["duration_ms"])),
        "hour": ((va["hourmin"] // 100), (tr["hourmin"] // 100)),
    }
    L = ["## (user x KEY) FEASIBILITY -- measured on this dataset",
         "A user-by-KEY feature only works when the pair is actually OBSERVED in "
         "train; otherwise it backs off to a user-constant value, which is worth "
         "exactly 0.5 AUC.",
         f"{'key':<18}{'varies within user':>20}{'(user,key) in train':>22}"]
    for k, (vc, tc) in keys.items():
        vc = np.asarray(vc); tc = np.asarray(tc)
        var = _within_user_variation(u, vc.astype(float))
        K = int(max(tc.max(), vc.max())) + 1
        seen = np.zeros(nU * K, dtype=bool)
        seen[tr["user"].astype(np.int64) * K + tc.astype(np.int64)] = True
        cov = float(seen[u.astype(np.int64) * K + vc.astype(np.int64)].mean())
        L.append(f"{k:<18}{var:>19.1%}{cov:>21.1%}")
    L.append("Read this before proposing a user-by-KEY feature: author and video "
             "are effectively UNESTIMABLE per user here (3.4% and 1.6%). Two "
             "candidates have already failed exactly this way.")
    return "\n".join(L)


def probe(proposal: dict, splits: dict, meta: dict,
          incumbent_scores=None) -> dict:
    """Run one candidate through the screen. Never raises: a broken builder is
    a finding about the candidate, not a crash of the iteration."""
    from agent import error_analysis as EA
    from agent.leakage_check import check_source, verdict as leak_verdict

    name = str(proposal.get("name", "unnamed"))
    src = str(proposal.get("source", ""))
    res = {"name": name, "status": PROBED, "checks": {}}

    # 1. leakage -- refused before execution, never scored
    findings = check_source(src)
    lv = leak_verdict(findings)
    label_leaks = label_leak_findings(src)
    res["checks"]["leakage"] = {"block": lv["block"], "fatal": lv["n_fatal"],
                                "warn": lv["n_warn"],
                                "label_leaks": label_leaks}
    if lv["block"]:
        res.update(status=REJECTED,
                   reason="leakage review refused the builder before running it")
        return res
    if label_leaks:
        res.update(status=REJECTED,
                   reason="label leakage: " + label_leaks[0])
        return res

    # 2. validity
    import sys
    sys.path.insert(0, os.path.join(ROOT, "runtime"))
    import train_lib
    t0 = time.time()
    try:
        built = train_lib.build_extra_features(src, splits, meta)
    except Exception as e:
        res.update(status=REJECTED,
                   reason=f"builder failed: {type(e).__name__}: {str(e)[:200]}")
        return res
    res["checks"]["build_seconds"] = round(time.time() - t0, 2)
    res["features_built"] = sorted(built)

    va = splits["valid"]
    u, y = va["user"], va["long_view"]
    per_feature = {}
    best_gain = -1.0
    for fname, cols in built.items():
        v = cols["valid"]
        finite = np.isfinite(v)
        d = {"coverage": round(float(finite.mean()), 4),
             "distinct_values": int(len(np.unique(v[finite]))) if finite.any() else 0}
        if not finite.any():
            d["verdict"] = "all values missing"
            per_feature[fname] = d
            continue
        # 3. within-user variation -- the structural gate.
        # ptp (max-min), not std: np.nanstd over IDENTICAL float64 values
        # returns ~1e-17 rather than 0, which made a strictly user-constant
        # feature look like it varied inside 17.8% of users.
        d["varies_within_user_frac"] = round(
            _within_user_variation(u, np.where(finite, v, np.nan)), 4)

        # 3b. TRAIN/SERVE SKEW -- the defect that actually sank both earlier
        # candidates. A user-by-author feature is fully populated on TRAIN
        # (every pair is observed there by construction) but only 3.4% of
        # validation rows have that pair, so it collapses to a user-constant
        # value exactly where it is scored. Measured: within-user variation
        # 1.000 on train against 0.167 on valid, a 6x skew -- the model learns
        # to lean on information that is not there at evaluation, which is why
        # that feature scored -16.5 sigma rather than merely nothing.
        tr_var = _within_user_variation(splits["train"]["user"], cols["train"])
        d["varies_within_user_TRAIN"] = round(tr_var, 4)
        d["train_serve_skew"] = round(
            tr_var / max(d["varies_within_user_frac"], 1e-9), 2)
        if (d["varies_within_user_frac"] > 0.01
                and d["train_serve_skew"] >= SKEW_LIMIT):
            d["verdict"] = (
                f"TRAIN/SERVE SKEW {d['train_serve_skew']}x -- varies within "
                f"{tr_var:.1%} of users at TRAIN but only "
                f"{d['varies_within_user_frac']:.1%} at EVALUATION. The model "
                f"would learn to rely on information that is absent when it is "
                f"scored. This is what sank the earlier user-by-author "
                f"features at -16.5 sigma.")
            per_feature[fname] = d
            continue
        if d["varies_within_user_frac"] < 0.01:
            d["verdict"] = ("CONSTANT within users -- cannot move GAUC or nDCG@5 "
                            "by construction")
            per_feature[fname] = d
            continue
        # 4. redundancy against what the model already sees
        filled = np.where(finite, v, np.nanmedian(v[finite]))
        d["corr_with_existing"] = {
            k: round(float(np.corrcoef(filled, va[k].astype(float))[0, 1]), 3)
            for k in ("video", "author", "duration_ms", "tab")
            if k in va}
        # 5. incremental value over the incumbent
        if incumbent_scores is not None:
            pr = EA.feature_probe(u, y, incumbent_scores, filled, fname)
            d["standalone_wAUC"] = pr["standalone_wAUC"]
            d["incremental_sigma"] = pr["best_gain_sigma"]
            d["adds_signal"] = pr["adds_signal"]
            d["verdict"] = pr["verdict"]
            best_gain = max(best_gain, pr["best_gain_sigma"])
        else:
            d["standalone_wAUC"] = round(EA.within_user_auc(u, y, filled), 4)
            d["verdict"] = "no incumbent scores available for a residual test"
        per_feature[fname] = d

    res["per_feature"] = per_feature
    res["best_incremental_sigma"] = round(best_gain, 2) if best_gain > -1 else None
    if any(d.get("adds_signal") for d in per_feature.values()):
        res["status"] = PROMISING
        res["reason"] = ("adds residual signal over the incumbent -- worth a "
                         "full training run")
    elif all(d.get("verdict", "").startswith("CONSTANT") for d in per_feature.values()):
        res["status"] = REJECTED
        res["reason"] = ("constant within users, so it cannot move a within-user "
                         "ranking metric")
    elif any(d.get("verdict", "").startswith("TRAIN/SERVE SKEW")
             for d in per_feature.values()):
        res["status"] = REJECTED
        res["reason"] = next(d["verdict"] for d in per_feature.values()
                             if d.get("verdict", "").startswith("TRAIN/SERVE"))
    else:
        res["status"] = PROBED
        res["reason"] = (f"no usable residual signal "
                         f"(best {res['best_incremental_sigma']} sigma, under the "
                         f"{NOISE / 2} bar) -- cheap to run anyway, but the probe "
                         f"does not justify it")
    return res


# ------------------------------------------------------------------ registry
def record(entry: dict, path: str = REGISTRY) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    entry = dict(entry)
    entry.setdefault("timestamp_utc",
                     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(path, "a") as fh:
        fh.write(json.dumps(entry) + "\n")


def load_registry(path: str = REGISTRY) -> list:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def _norm(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def already_tried(name: str, source: str = "", path: str = REGISTRY) -> dict | None:
    """Has this feature been proposed before? Matches on normalised name and on
    the builder body, so a rename does not smuggle a known failure back in."""
    n, s = _norm(name), _norm(source)
    for e in load_registry(path):
        if _norm(e.get("name", "")) == n:
            return e
        if s and e.get("source") and _norm(e["source"]) == s:
            return e
    return None


def render_for_prompt(path: str = REGISTRY, limit: int = 12) -> str:
    """What the agent has already learned about features, fed back in."""
    reg = load_registry(path)
    if not reg:
        return ""
    latest: dict = {}
    for e in reg:
        latest[_norm(e.get("name", ""))] = e
    rows = list(latest.values())[-limit:]
    L = ["## FEATURE REGISTRY (your own past feature research -- do not repeat)"]
    for e in rows:
        L.append(f"- {e.get('name')}: {e.get('status')} "
                 f"({str(e.get('reason', ''))[:110]})")
    return "\n".join(L)


# ---------------------------------------------------------------- prompting
FEATURE_CONTRACT = '''You may propose a NEW FEATURE that does not exist in the menu.

The menu is prior knowledge, not the boundary of what you may try. A feature you
invent from the evidence is worth more than another menu combination -- but only
if it is grounded, so every proposal must survive a probe before it costs a
training run.

Write a builder with EXACTLY this contract:

    import numpy as np
    def build_features(splits, meta):
        """splits: {"train"|"valid"|"test": {column: np.ndarray}}
           meta:   {"field_dims": {"user": int, "video": int, "author": int, "tab": int}}
           return: {feature_name: {split_name: float array, one value per row}}"""
        ...

Columns available per split: user, video, author, tab (integer codes),
duration_ms, hourmin, date, time_ms (numeric), user_raw.
Outcome columns (long_view, is_click, is_like, is_forward, play_time_ms,
is_follow, is_comment, is_hate, is_profile_enter) exist ONLY on splits["train"]
and may be read ONLY from there.

HARD RULES, checked mechanically before your code runs:
  * Read outcome columns from splits["train"] only. Reading them from valid or
    test is target leakage and the proposal is refused, not scored.
  * Return one finite-or-NaN float per row of each split. NaN is fine and gets
    its own bucket; a wrong length is refused.
  * The value for a row must depend only on information that existed BEFORE
    that impression.

WHAT MAKES A FEATURE WORTH PROPOSING -- read this before writing code:
  * It must VARY WITHIN a single user's impression list. Both metrics rank
    within a user, so a feature constant across that user's rows contributes
    exactly nothing: user-level scalars measure 0.5000 AUC, exactly, by
    construction. This has already been measured; do not re-propose it.
  * It must add signal the model does not already have. The model already
    learns an embedding per user, video, author and tab, so item-level rates
    and counts are largely redundant: item long_view rate scores 0.639 alone
    yet HURTS at every blend weight tested. A strong standalone number is not
    evidence.

  * It must vary within a user's list AT EVALUATION, not just during training.
    Measured on the two candidates that already failed: both varied inside
    ~100% of users on TRAIN and only ~16% on VALID, a 6x skew, because
    (user, author) is observed for just 3.4% of validation rows and everything
    else fell back to a user-constant value. The model then learned to lean on
    information that is absent when it is scored, and the full experiment came
    out at -16.5 sigma -- far worse than merely useless. Any candidate with a
    train/serve skew above 2.5x is now refused before it costs a run.

So the useful direction is a feature that varies within a user's list AND is not
recoverable from the item identity alone AND is actually OBSERVABLE per user at
evaluation time. The feasibility table below says which keys satisfy the last
condition -- read it before choosing one.
'''


def build_feature_prompt(state_block: str, error_block: str,
                         registry_block: str, key_block: str = "") -> str:
    """Phase: propose ONE feature from the evidence, or decline."""
    from .prompts import STATIC_CONTEXT
    return "\n\n".join([
        STATIC_CONTEXT,
        "## FEATURE DISCOVERY\n"
        "You are deciding whether there is a NEW feature worth testing, based on "
        "the evidence below. Declining is a legitimate and often correct answer "
        "-- propose nothing rather than something you cannot justify.",
        state_block or "",
        error_block or "",
        registry_block or "",
        key_block or "",
        FEATURE_CONTRACT,
        "Reply with JSON only:\n"
        "{\n"
        '  "propose": true | false,\n'
        '  "decline_reason": "<why nothing is worth proposing, if propose=false>",\n'
        '  "name": "<snake_case feature name>",\n'
        '  "hypothesis": "<what you expect and why, from the evidence above>",\n'
        '  "mechanism": "<what user/item behaviour this represents>",\n'
        '  "incremental_value": "<why the model cannot already recover this '
        'from its existing embeddings>",\n'
        '  "leakage_check": "<why every value predates the impression it '
        'describes>",\n'
        '  "source_columns": "<columns used>",\n'
        '  "cost": "<rough cost to compute>",\n'
        '  "source": "<python source defining build_features(splits, meta)>"\n'
        "}",
    ])


def render_probe_for_prompt(res: dict) -> str:
    """The probe result, written back into the planning prompt."""
    if not res:
        return ""
    L = [f"## FEATURE PROBE: {res.get('name')} -> {res.get('status')}",
         f"reason: {res.get('reason', '')}"]
    for f, d in (res.get("per_feature") or {}).items():
        L.append(f"- {f}: coverage {d.get('coverage')}, "
                 f"varies within user {d.get('varies_within_user_frac')}, "
                 f"standalone wAUC {d.get('standalone_wAUC')}, "
                 f"incremental {d.get('incremental_sigma')} sigma")
        if d.get("verdict"):
            L.append(f"  verdict: {d['verdict']}")
    if res.get("status") == PROMISING:
        L.append("This cleared the probe. Use it: put the builder source in "
                 "menu_choices.feature_source for this experiment.")
    else:
        L.append("This did NOT clear the probe. Do not spend a training run on "
                 "it; it is recorded so it will not be proposed again.")
    return "\n".join(L)


def render_discovery_log(path: str = REGISTRY) -> str:
    """The audit trail of autonomous feature research, for a human reader.

    This is the evidence that the agent did research rather than picked from a
    list: for each candidate, what it observed, what it hypothesised, the code
    it wrote, what the probe measured, and what was decided.
    """
    reg = load_registry(path)
    if not reg:
        return "no feature research recorded yet"
    L = []
    for e in reg:
        p = e.get("probe") or {}
        L += ["=" * 74,
              f"[FEATURE DISCOVERY]  {e.get('name')}",
              "=" * 74,
              f"Hypothesis      : {e.get('hypothesis', '')}",
              f"Mechanism       : {e.get('mechanism', '')}",
              f"Incremental     : {e.get('incremental_value', '')}",
              f"Leakage check   : {e.get('leakage_check', '')}",
              f"Source columns  : {e.get('source_columns', '')}"]
        lk = (p.get("checks") or {}).get("leakage") or {}
        L.append(f"Leakage gate    : "
                 f"{'REFUSED' if lk.get('block') or lk.get('label_leaks') else 'PASS'} "
                 f"(fatal {lk.get('fatal', 0)}, label-leaks "
                 f"{len(lk.get('label_leaks') or [])})")
        for f, d in (p.get("per_feature") or {}).items():
            L.append(f"Probe           : coverage {d.get('coverage')}, "
                     f"varies within user {d.get('varies_within_user_frac')}, "
                     f"standalone wAUC {d.get('standalone_wAUC')}, "
                     f"incremental {d.get('incremental_sigma')} sigma")
            if d.get("verdict"):
                L.append(f"                  {d['verdict']}")
        x = e.get("experiment")
        if x:
            L.append(f"Full experiment : baseline {x.get('control_mean')} -> "
                     f"candidate {x.get('feature_mean')}  "
                     f"delta {x.get('paired_delta'):+.5f} "
                     f"({x.get('sigma')} sigma, wins {x.get('wins')})")
        L += [f"Decision        : {e.get('status')}",
              f"Reason          : {e.get('reason', '')}",
              f"Builder         : {len((e.get('source') or '').splitlines())} lines "
              f"of agent-written code",
              ""]
    n_prop = len(reg)
    n_rej = sum(1 for e in reg if e.get("status") == REJECTED)
    L += ["-" * 74,
          f"{n_prop} candidate(s) researched, {n_rej} rejected outright, "
          f"none re-proposed (deduplicated by name and by builder body)."]
    return "\n".join(L)


if __name__ == "__main__":
    print(render_discovery_log())
