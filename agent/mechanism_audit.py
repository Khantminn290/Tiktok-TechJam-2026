"""Does the code actually implement the mechanism the hypothesis claimed?

A declaration is not evidence. A clean process exit is not evidence. This
project has both kinds of false positive on record:

  * three Path B nodes declared custom research code, and all three were
    reimplementing the incumbent rather than adding a mechanism -- then errored;
  * three more "succeeded" by applying a per-user affine transform to the
    trained model's scores. Two of them returned BYTE-IDENTICAL metrics
    (0.60340 / 0.66981 / 0.53700) and were both recorded as successes.

That second class is the dangerous one, because it looks like a result. Both
competition metrics rank WITHIN a user, so any strictly increasing per-user
transform of the scores -- (s - mean)/std, a per-user shift, a per-user scale,
a per-user rank -- cannot change GAUC or nDCG@5 at all. It is not a weak
mechanism; it is arithmetically incapable of moving the number it is being
scored on.

So this module asks two separate questions:

    1. STRUCTURAL FUTILITY -- can this change the metric even in principle?
    2. MECHANISM PRESENCE  -- is the claimed mechanism actually in the code?

Both are static and cheap, and both run BEFORE an experiment is scored. Neither
can prove a mechanism correct; they exist to catch the specific ways this
project has already been fooled.
"""
from __future__ import annotations

import ast
import re

# What a hypothesis can claim, and the code evidence that would support it.
MECHANISMS = {
    "auxiliary_task": (
        r"aux|auxiliar|multi-?task|multitask",
        r"aux_tasks|aux_forward|aux_targets|is_follow|is_comment|is_hate|"
        r"is_profile_enter|is_click|is_like|is_forward"),
    "new_loss": (r"\bloss\b|objective|bpr|softmax|lambdarank|focal|pairwise",
                 r"grad|sigmoid|log\(|logaddexp|softmax|np\.exp|backward"),
    "new_feature": (r"feature|embedding|encode|represent",
                    r"bincount|add\.at|column_stack|concatenate|np\.stack|"
                    r"searchsorted|digitize"),
    "sequence": (r"sequen|recurrent|gru|attention|history",
                 r"argsort|cumsum|roll|lag|window|attention|softmax"),
    "ensembling": (r"ensembl|average|blend", r"mean\(|average\(|rank"),
    "post_processing": (r"post-?hoc|post-?process|calibrat|normalis|normaliz|"
                        r"shrink|rescal", r"scores\s*[-+*/]|clip|rank"),
}

# Per-user statistics that, applied to scores, give a monotone transform.
_PER_USER_STAT = re.compile(
    r"(group|per[_ -]?user|by[_ -]?user|user_(mean|std|min|max|sum|cnt|count))"
    r"|np\.add\.at|np\.bincount|np\.unique\([^)]*return_inverse", re.I)
_SCORE_ARITH = re.compile(
    r"scores?\s*[-+*/]|[-+*/]\s*scores?|\(\s*scores?\s*-|standardi[sz]|"
    r"z[_ -]?score|affine|normali[sz]e", re.I)


def _calls(tree) -> set:
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute):
                out.add(f.attr)
            elif isinstance(f, ast.Name):
                out.add(f.id)
    return out


# For a Path A experiment the mechanism lives in train_lib and is SELECTED by
# configuration -- the script is a generic pass-through. Auditing only the
# source would flag every menu-driven experiment as "mechanism not evidenced",
# which is a systematic false positive, so the chosen config counts as evidence.
# Pipeline overrides that evidence a mechanism when set literally in a script.
PIPELINE_MECHANISM = {
    "ensembling": ("snapshot_ensemble", "snapshot_force"),
    "new_feature": ("hist_tau_days",),
}

CONFIG_EVIDENCE = {
    "auxiliary_task": ("multitask",),
    "new_loss": ("loss",),
    "new_feature": ("temporal", "user_history", "data_extras"),
    "sequence": ("model", "user_history"),
}


def audit(src: str, hypothesis: str = "", declared_path: str = "A",
          menu_choices: dict | None = None) -> dict:
    """Static audit of one generated script against its stated hypothesis."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"parses": False, "error": str(e)[:200], "structurally_null": False,
                "mechanisms_claimed": [], "mechanisms_evidenced": [],
                "verdict": "does not parse"}

    calls = _calls(tree)
    wraps_run = bool(re.search(r"train_lib\.run\s*\(", src))
    n_def = sum(isinstance(n, ast.FunctionDef) for n in ast.walk(tree))
    n_loop = sum(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree))
    np_ops = len(re.findall(r"np\.\w+", src))

    # --- 1. structural futility -------------------------------------------
    # A per-user statistic combined arithmetically with the scores is a
    # per-user affine map. Both metrics are within-user ranking metrics, so a
    # strictly increasing per-user map leaves them EXACTLY unchanged.
    per_user = bool(_PER_USER_STAT.search(src))
    touches_scores = bool(_SCORE_ARITH.search(src))
    trains_model = ("fit" in calls or "run" in calls or n_loop > 0) and np_ops >= 5
    structurally_null = bool(per_user and touches_scores and wraps_run
                             and not _introduces_learning(src))

    # --- 2. mechanism presence --------------------------------------------
    h = (hypothesis or "").lower()
    claimed, evidenced, missing = [], [], []
    mc = menu_choices or {}
    for name, (claim_pat, code_pat) in MECHANISMS.items():
        if re.search(claim_pat, h):
            claimed.append(name)
            in_code = bool(re.search(code_pat, src, re.I))
            # A Path B script can invoke a mechanism by passing a pipeline
            # override in a dict literal it hands to train_lib.run -- the agent
            # did exactly this for snapshot_ensemble, and auditing the source
            # patterns alone called a working experiment "not evidenced".
            if not in_code and any(
                    re.search(rf'["\']{re.escape(ax)}["\']\s*:', src)
                    for ax in CONFIG_EVIDENCE.get(name, ()) + tuple(
                        PIPELINE_MECHANISM.get(name, ()))):
                in_code = True
            # a non-default choice on a relevant axis selects the mechanism
            in_cfg = any(str(mc.get(ax, "none")).lower() not in ("none", "default", "")
                         for ax in CONFIG_EVIDENCE.get(name, ()))
            if in_code or in_cfg:
                evidenced.append(name)
            else:
                missing.append(name)

    genuine_b = (not wraps_run and (n_def >= 1 or n_loop > 0) and np_ops >= 3)

    # Blocking on a null transform alone would over-reach: a script can train a
    # genuinely different configuration AND separately add a useless transform,
    # and the training is still worth scoring. So the transform is always
    # flagged, but it only BLOCKS when it is the script's sole claimed novelty.
    only_postproc = claimed == ["post_processing"] or (structurally_null and not claimed)
    if structurally_null and only_postproc:
        verdict = ("STRUCTURALLY NULL -- the only novelty is a per-user "
                   "transform of the scores. Both metrics rank WITHIN a user, "
                   "so a monotone per-user map cannot change either one. Do not "
                   "score this.")
    elif structurally_null:
        verdict = ("post-processing is STRUCTURALLY NULL (a per-user monotone "
                   "map cannot move a within-user metric) -- any difference "
                   "measured here comes from the rest of the script, not from "
                   "it")
    elif missing:
        verdict = (f"MECHANISM NOT EVIDENCED -- the hypothesis claims "
                   f"{', '.join(missing)} but the code shows no sign of it.")
    elif declared_path.upper() == "B" and wraps_run and not claimed:
        verdict = ("declares Path B but delegates to train_lib.run() with no "
                   "claimed mechanism -- this is Path A wearing a Path B label")
    else:
        verdict = "mechanism present in the code"

    return {"parses": True, "wraps_train_lib_run": wraps_run,
            "defs": n_def, "loops": n_loop, "np_ops": np_ops,
            "genuine_custom_code": genuine_b,
            "per_user_statistic": per_user, "arithmetic_on_scores": touches_scores,
            "structurally_null": structurally_null,
            "mechanisms_claimed": claimed,
            "mechanisms_evidenced": evidenced,
            "mechanisms_missing": missing,
            "trains_model": trains_model,
            "verdict": verdict,
            "postprocessing_null": structurally_null,
            "blocks_scoring": bool(structurally_null and only_postproc),
            "should_score": not (structurally_null and only_postproc) and not missing}


def _introduces_learning(src: str) -> bool:
    """A script that also TRAINS something new is not merely post-processing."""
    return bool(re.search(r"aux_tasks|aux_forward|\.fit\(|gradient|backward|"
                          r"for\s+epoch|learning_rate|lr\s*=", src))


def render(a: dict) -> str:
    L = ["## MECHANISM AUDIT", f"verdict: {a['verdict']}"]
    if not a["parses"]:
        return "\n".join(L)
    L.append(f"claimed: {a['mechanisms_claimed'] or '-'} | "
             f"evidenced: {a['mechanisms_evidenced'] or '-'} | "
             f"missing: {a['mechanisms_missing'] or '-'}")
    L.append(f"wraps train_lib.run: {a['wraps_train_lib_run']} | "
             f"defs {a['defs']} loops {a['loops']} np {a['np_ops']}")
    if a["structurally_null"]:
        L.append("!! This experiment cannot move GAUC or nDCG@5 by construction. "
                 "Propose something that reorders items WITHIN a single user's "
                 "list.")
    return "\n".join(L)
