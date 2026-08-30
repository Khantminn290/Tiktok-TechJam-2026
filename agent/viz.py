"""Render the agent's experiment tree as a page you can actually read.

The journal is the authoritative record, but it is JSONL: fine for tooling,
useless for seeing at a glance what the agent was *thinking* and where the
search branched. This turns it into one self-contained HTML file — no CDN, no
build step, no network — showing for every node:

    the question it could not answer, the hypothesis it formed, the experiment
    it ran, what it scored, and how much that result was allowed to count for.

Branching comes from `parent_id`, so the shape on screen is the shape the search
actually took: drafts at the root, `improve` nodes hanging off whatever they
extended, `debug` chains off the node that failed, and paired confirmations
attached to the node they were testing.

Evidence state is recomputed here from `agent.evidence` rather than read from
the journal, so the page cannot show a node as more established than the rules
would currently allow.

Usage:
    python3 -m agent.viz                       # writes viz/tree.html
    python3 -m agent.viz --open                # ...and opens it
    python3 -m agent.viz --serve               # serve on localhost:8000
    python3 -m agent.viz --journal <path.jsonl>
"""
from __future__ import annotations

import argparse
import html
import json
import os
import webbrowser

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
BASELINE = 0.6016
NOISE = 0.0008


def _load(path: str) -> list:
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


def _event(n: dict, kind: str) -> dict:
    for e in (n.get("events") or []):
        if e.get("type") == kind:
            return e
    return {}


def _esc(s, limit: int = 0) -> str:
    s = "" if s is None else str(s)
    if limit and len(s) > limit:
        s = s[:limit] + "…"
    return html.escape(s)


def build_tree(nodes: list) -> list:
    """Roots, each with `children`, ordered by iteration."""
    by_id = {n["iteration_id"]: dict(n, children=[]) for n in nodes}
    roots = []
    for n in sorted(by_id.values(), key=lambda x: x["iteration_id"]):
        pid = n.get("parent_id")
        if pid is not None and pid in by_id and pid != n["iteration_id"]:
            by_id[pid]["children"].append(n)
        else:
            roots.append(n)
    return roots


def _evidence_for(n: dict) -> dict | None:
    m = n.get("metrics") or {}
    if not m:
        return None
    from . import evidence as EV
    paired = _event(n, "paired_result")
    if paired:
        res = paired.get("result") or {}
        ev = dict(paired.get("evidence") or {})
        ev.setdefault("state", "UNTESTED")
        ev["n_seeds"] = res.get("n")
        ev["paired"] = True
        return ev
    return EV.classify(delta=m["primary"] - BASELINE, n_seeds=1)


def _node_html(n: dict, depth: int = 0) -> str:
    m = n.get("metrics") or {}
    q = _event(n, "inquiry")
    alloc = _event(n, "allocation")
    paired = _event(n, "paired_result")
    ev = _evidence_for(n)

    status = n.get("status")
    trace = n.get("error_trace") or ""
    if "PREFLIGHT REJECTED" in trace:
        cls, badge = "preflight", "PREFLIGHT — no compute spent"
    elif status == "success":
        cls, badge = "ok", "completed"
    else:
        cls, badge = "fail", "crashed"

    path = (n.get("implementation_path") or "A").upper()
    action = n.get("action") or "?"
    score = ""
    if m:
        d = m["primary"] - BASELINE
        score = (f'<span class="score">{m["primary"]:.5f}</span>'
                 f'<span class="delta">{d:+.5f} ({d / NOISE:+.2f}σ)</span>')

    parts = [f'<div class="node {cls}" style="margin-left:{depth * 26}px">']
    parts.append(
        f'<div class="hdr"><span class="id">#{n["iteration_id"]}</span>'
        f'<span class="tag act-{action}">{_esc(action)}</span>'
        f'<span class="tag path">Path {path}</span>'
        f'<span class="tag st">{badge}</span>{score}</div>')

    if q.get("question"):
        parts.append(f'<div class="q"><b>Question</b> {_esc(q["question"], 300)}</div>')
    if q.get("observation"):
        parts.append(f'<div class="obs"><b>Observed</b> {_esc(q["observation"], 300)}</div>')
    if q.get("hypotheses"):
        hyps = q["hypotheses"]
        if isinstance(hyps, str):
            hyps = [hyps]
        items = "".join(f"<li>{_esc(h, 220)}</li>" for h in list(hyps)[:4])
        parts.append(f'<div class="hyp"><b>Competing hypotheses</b><ul>{items}</ul></div>')
    if n.get("hypothesis"):
        parts.append(f'<div class="plan"><b>Experiment</b> {_esc(n["hypothesis"], 320)}</div>')

    mc = n.get("menu_choices") or {}
    if mc:
        keys = [f"{k}={v}" for k, v in mc.items()
                if k not in ("feature_source",) and v not in (None, "none")]
        if mc.get("feature_source"):
            keys.append("feature_source=<agent-written builder>")
        parts.append(f'<div class="cfg">{_esc(", ".join(keys), 300)}</div>')

    if paired:
        res = paired.get("result") or {}
        if res.get("usable"):
            parts.append(
                f'<div class="paired"><b>Paired confirmation</b> '
                f'control {res["control_mean"]:.5f} → treatment '
                f'{res["treatment_mean"]:.5f} · Δ {res["delta"]:+.5f} '
                f'({res["sigma"]:+.2f}σ) · t={res["t"]} · '
                f'{res["wins"]}/{res["n"]} seeds · '
                f'promote={"YES" if paired.get("promote") else "NO"}</div>')

    if ev:
        note = _esc(ev.get("why", ""), 200)
        parts.append(
            f'<div class="ev ev-{ev["state"].lower()}"><b>{ev["state"]}</b>'
            f'{" · " + str(ev["n_seeds"]) + " seed(s)" if ev.get("n_seeds") else ""}'
            f'<span class="why">{note}</span></div>')

    if alloc.get("choice"):
        parts.append(f'<div class="alloc">allocator chose '
                     f'<b>{_esc(alloc["choice"])}</b></div>')

    if cls == "fail" or cls == "preflight":
        first = next((l for l in trace.splitlines()
                      if l.strip() and not l.startswith(("File ", "  File", "Traceback"))),
                     "")
        parts.append(f'<div class="err">{_esc(first, 220)}</div>')

    parts.append("</div>")
    for c in n["children"]:
        parts.append(_node_html(c, depth + 1))
    return "".join(parts)


CSS = """
:root{--bg:#0f1116;--fg:#e6e6e6;--mut:#9aa0aa;--line:#232733;--ok:#2ea043;
--fail:#d1434b;--pre:#b58900;--card:#161923;--accent:#4f8cc9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:22px 26px;border-bottom:1px solid var(--line)}
h1{margin:0 0 6px;font-size:19px}
.sub{color:var(--mut);font-size:13px}
.summary{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:8px 12px;min-width:120px}
.kpi .v{font-size:17px;font-weight:600}
.kpi .k{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.5px}
main{padding:20px 26px 60px}
.node{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--mut);
border-radius:8px;padding:11px 14px;margin:8px 0}
.node.ok{border-left-color:var(--ok)}
.node.fail{border-left-color:var(--fail)}
.node.preflight{border-left-color:var(--pre)}
.hdr{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.id{font-weight:700;color:var(--accent)}
.tag{font-size:11px;padding:2px 7px;border-radius:99px;background:#222736;color:var(--mut)}
.act-confirm{background:#1d3a5c;color:#9ec9f5}
.act-improve{background:#1d4030;color:#8fd7a8}
.act-debug{background:#4a2330;color:#f0a0aa}
.score{margin-left:auto;font-weight:700}
.delta{color:var(--mut);margin-left:8px;font-size:12px}
.q,.obs,.hyp,.plan,.cfg,.paired,.ev,.alloc,.err{margin-top:5px;font-size:13px}
.q b,.obs b,.hyp b,.plan b,.paired b{color:var(--accent);margin-right:5px}
.obs,.plan{color:#c8cdd6}
.hyp ul{margin:4px 0 0 18px;padding:0;color:#c8cdd6}
.cfg{color:var(--mut);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px}
.paired{background:#12202e;border-radius:6px;padding:7px 9px}
.ev{font-size:12px;padding:5px 9px;border-radius:6px;background:#1b1f2b}
.ev .why{color:var(--mut);margin-left:8px}
.ev-confirmed{background:#123020;color:#8fd7a8}
.ev-rejected,.ev-unconfirmed{background:#2e1c20;color:#f0a0aa}
.ev-preliminary{background:#2e2a17;color:#e3cf87}
.alloc{color:var(--mut);font-size:12px}
.err{color:#f0a0aa;font-family:ui-monospace,Menlo,monospace;font-size:11.5px}
.legend{color:var(--mut);font-size:12px;margin-top:10px}
"""


def render(nodes: list, title: str = "Agent experiment tree") -> str:
    roots = build_tree(nodes)
    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    best = max((n["metrics"]["primary"] for n in scored), default=None)
    confirms = [n for n in nodes if (n.get("action") or "") == "confirm"]
    promoted = sum(1 for n in nodes
                   if (_event(n, "paired_result") or {}).get("promote"))
    pre = sum(1 for n in nodes
              if "PREFLIGHT REJECTED" in (n.get("error_trace") or ""))
    crashed = sum(1 for n in nodes if n.get("status") == "error") - pre

    def kpi(k, v):
        return f'<div class="kpi"><div class="v">{v}</div><div class="k">{k}</div></div>'

    kpis = "".join([
        kpi("nodes", len(nodes)),
        kpi("scored", len(scored)),
        kpi("crashed", crashed),
        kpi("preflight (free)", pre),
        kpi("confirmations", len(confirms)),
        kpi("promoted", promoted),
        kpi("best primary", f"{best:.5f}" if best else "—"),
        kpi("Δ vs baseline", f"{best - BASELINE:+.5f}" if best else "—"),
    ])
    body = "".join(_node_html(r) for r in roots) or "<p>No nodes in this journal.</p>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>{CSS}</style></head><body>
<header><h1>{html.escape(title)}</h1>
<div class="sub">Generated from the journal. Indentation is the real search
tree — a child node branched from its parent. Evidence states are recomputed
from the current rules, so nothing shows as more established than it is.</div>
<div class="summary">{kpis}</div>
<div class="legend">green = completed · red = crashed · amber = rejected by
preflight (no compute spent) · baseline {BASELINE}, σ = {NOISE}</div>
</header><main>{body}</main></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(ROOT, "logs", "journal.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "viz", "tree.html"))
    ap.add_argument("--title", default="Agent experiment tree")
    ap.add_argument("--open", action="store_true", help="open in a browser")
    ap.add_argument("--serve", type=int, nargs="?", const=8000, default=None,
                    metavar="PORT", help="serve the viz/ directory")
    a = ap.parse_args()

    nodes = _load(a.journal)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        fh.write(render(nodes, a.title))
    print(f"wrote {a.out}  ({len(nodes)} nodes from {a.journal})")

    if a.open:
        webbrowser.open("file://" + os.path.abspath(a.out))
    if a.serve:
        import functools
        import http.server
        import socketserver
        d = os.path.dirname(os.path.abspath(a.out))
        h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=d)
        url = f"http://localhost:{a.serve}/{os.path.basename(a.out)}"
        print(f"serving {d} at {url}   (ctrl-c to stop)")
        with socketserver.TCPServer(("", a.serve), h) as srv:
            try:
                srv.serve_forever()
            except KeyboardInterrupt:
                print("\nstopped")


if __name__ == "__main__":
    main()
