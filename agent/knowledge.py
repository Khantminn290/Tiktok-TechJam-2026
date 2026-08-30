"""Research memory: claims that carry their own scope, confidence and doubt.

The existing `tested_dead_ends` list is a good record and a blunt instrument. It
stores what was tried and what happened, which is genuinely valuable -- it is
the reason the agent does not re-run known failures. But every entry reads as a
global, permanent verdict, and two things go wrong with that.

**A scoped result becomes a universal ban.** One entry in this project's own
menu says snapshot ensembling was rejected because validation "peaks at epoch 4
then declines monotonically". That premise is a property of ONE training
configuration's epoch curve, not of the method. Stored as a flat dead end, it
closes the method everywhere, including in configurations whose curve looks
nothing like that. The entry itself had to be amended by hand with a scope note
and a "re-measure before relying on this" warning, which is exactly the
structure that should have been there from the start.

**Nothing can weaken a conclusion except deleting it.** A record that cannot
absorb counterevidence forces a choice between keeping something known to be
wrong and destroying the evidence that it was ever believed. Both are bad.

So a claim here is not a sentence. It is a sentence plus:

    SCOPE                       the conditions under which it was measured
    EVIDENCE                    what was actually observed
    CONFIDENCE                  how much it should be trusted
    COUNTEREVIDENCE             observations that push against it
    STATUS                      including CONTESTED and SUPERSEDED
    WHAT WOULD CHANGE THIS      stated in advance, so the claim is falsifiable

That last field is the important one. A belief whose owner cannot say what
would change it is not a research finding.
"""
from __future__ import annotations

import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
STORE = os.path.join(ROOT, "logs", "research_memory.jsonl")

# Status of a claim, distinct from the EVIDENCE STATE of a single measurement.
OPEN = "OPEN"                 # believed, within scope
CONTESTED = "CONTESTED"       # counterevidence exists; do not rely on it
SUPERSEDED = "SUPERSEDED"     # replaced by a later, better-measured claim
RETRACTED = "RETRACTED"       # withdrawn; kept visible on purpose

HIGH, MEDIUM, LOW = "high", "medium", "low"
_ORDER = {HIGH: 3, MEDIUM: 2, LOW: 1}
_DOWN = {HIGH: MEDIUM, MEDIUM: LOW, LOW: LOW}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def make_claim(claim: str, evidence: str, scope: str, confidence: str = MEDIUM,
               what_would_change_this: str = "",
               evidence_state: str | None = None, scope_tags: dict | None = None,
               node: int | None = None, source: str = "agent") -> dict:
    """Build a claim record. `scope` and `what_would_change_this` are required
    in spirit: a claim without them is an opinion."""
    return {"id": None, "claim": claim.strip(), "evidence": evidence.strip(),
            "scope": scope.strip(), "scope_tags": scope_tags or {},
            "confidence": confidence if confidence in _ORDER else MEDIUM,
            "evidence_state": evidence_state,
            "what_would_change_this": what_would_change_this.strip(),
            "counterevidence": [], "status": OPEN, "node": node,
            "source": source, "recorded_utc": _now(), "updated_utc": _now()}


def load(path: str = STORE) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as fh:
        for ln in fh:
            if ln.strip():
                try:
                    out.append(json.loads(ln))
                except json.JSONDecodeError:
                    pass
    return out


def _rewrite(claims: list, path: str = STORE) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for c in claims:
            fh.write(json.dumps(c) + "\n")
    os.replace(tmp, path)


def record(claim_rec: dict, path: str = STORE) -> dict:
    claims = load(path)
    claim_rec = dict(claim_rec)
    claim_rec["id"] = (max([c.get("id") or 0 for c in claims]) + 1) if claims else 1
    claims.append(claim_rec)
    _rewrite(claims, path)
    return claim_rec


def add_counterevidence(claim_id: int, observation: str, scope: str = "",
                        node: int | None = None, path: str = STORE) -> dict | None:
    """Push against a claim without deleting it.

    Confidence drops one level and the claim becomes CONTESTED. It stays
    readable, with both the original evidence and what argued against it --
    which is the record a later reader actually needs.
    """
    claims = load(path)
    for c in claims:
        if c.get("id") != claim_id:
            continue
        c["counterevidence"].append({"observation": observation.strip(),
                                     "scope": scope.strip(), "node": node,
                                     "recorded_utc": _now()})
        c["confidence"] = _DOWN[c.get("confidence", MEDIUM)]
        if c["status"] == OPEN:
            c["status"] = CONTESTED
        c["updated_utc"] = _now()
        _rewrite(claims, path)
        return c
    return None


def supersede(claim_id: int, by_claim_id: int, path: str = STORE) -> dict | None:
    claims = load(path)
    for c in claims:
        if c.get("id") == claim_id:
            c["status"] = SUPERSEDED
            c["superseded_by"] = by_claim_id
            c["updated_utc"] = _now()
            _rewrite(claims, path)
            return c
    return None


def in_scope(claim: dict, context: dict | None) -> bool:
    """Does this claim apply to the situation currently being considered?

    A claim measured under one set of conditions says nothing about a different
    set. When a context tag disagrees with the claim's scope tag, the claim is
    OUT OF SCOPE -- not evidence for, and not evidence against.
    """
    if not context:
        return True
    for k, v in (claim.get("scope_tags") or {}).items():
        if k in context and context[k] != v:
            return False
    return True


def applicable(context: dict | None = None, path: str = STORE) -> list:
    """Claims worth acting on here: in scope, and not contested or withdrawn."""
    return [c for c in load(path)
            if c.get("status") == OPEN and in_scope(c, context)]


def render_for_prompt(context: dict | None = None, limit: int = 12,
                      path: str = STORE) -> str:
    claims = load(path)
    if not claims:
        return ""
    L = ["## RESEARCH MEMORY — what we believe, and how strongly",
         "Each claim carries the SCOPE it was measured in. A claim outside its "
         "scope is not evidence either way; treat it as an open question, not a "
         "closed one."]
    shown = sorted(claims, key=lambda c: (-_ORDER.get(c.get("confidence"), 0),
                                          -(c.get("id") or 0)))[:limit]
    for c in shown:
        mark = {OPEN: "", CONTESTED: "  [CONTESTED]", SUPERSEDED: "  [SUPERSEDED]",
                RETRACTED: "  [RETRACTED]"}.get(c.get("status"), "")
        scope_note = "" if in_scope(c, context) else "  (OUT OF SCOPE HERE)"
        L.append(f"\n- [{c.get('id')}] {c['claim']}{mark}{scope_note}")
        L.append(f"    scope:      {c.get('scope') or '(unstated)'}")
        L.append(f"    evidence:   {c.get('evidence')}"
                 + (f"  [{c['evidence_state']}]" if c.get("evidence_state") else ""))
        L.append(f"    confidence: {c.get('confidence')}")
        if c.get("counterevidence"):
            for ce in c["counterevidence"][:2]:
                L.append(f"    AGAINST:    {ce['observation']}"
                         + (f"  (scope: {ce['scope']})" if ce.get("scope") else ""))
        if c.get("what_would_change_this"):
            L.append(f"    would change my mind: {c['what_would_change_this']}")
    L.append("\nIf you find evidence against one of these, say so and cite the "
             "claim id. Contradicting a recorded claim with a measurement is "
             "progress, not a mistake.")
    return "\n".join(L)


def render(path: str = STORE) -> str:
    claims = load(path)
    if not claims:
        return "research memory is empty"
    by_status: dict = {}
    for c in claims:
        by_status.setdefault(c.get("status", OPEN), []).append(c)
    L = ["=" * 74, "RESEARCH MEMORY", "=" * 74,
         f"{len(claims)} claims: " + ", ".join(
             f"{k} {len(v)}" for k, v in sorted(by_status.items()))]
    for c in claims:
        L.append(f"\n[{c['id']}] ({c['status']}, {c['confidence']}) {c['claim']}")
        L.append(f"     scope: {c['scope']}")
        L.append(f"     evidence: {c['evidence']}")
        for ce in c.get("counterevidence", []):
            L.append(f"     AGAINST: {ce['observation']}")
        if c.get("what_would_change_this"):
            L.append(f"     falsifiable by: {c['what_would_change_this']}")
    return "\n".join(L)


if __name__ == "__main__":
    print(render())
