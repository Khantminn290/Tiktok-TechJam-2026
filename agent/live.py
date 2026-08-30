"""Watch the agent work, live. Built to be screen-recorded.

`agent/viz.py` renders a finished run for inspection. This is the other job:
showing the loop *while it happens*, because the thing worth demonstrating is
not a final number — it is an agent forming a hypothesis, choosing what to
measure, hitting an error, recovering from it, and refusing to promote a result
it cannot justify.

It serves two endpoints and lets the browser do the redrawing:

    /            the shell — layout, styles, poll loop
    /state.json  the current journal, summarised

Polling rather than pushing, because the journal is a file the agent appends to
and a poll cannot wedge the run. New nodes animate in and the newest one is
highlighted, so a recording shows *when* each decision landed rather than a
wall of text that was always there.

What it surfaces, chosen for what a viewer needs to believe the agent is
autonomous:

  * a live status line — running or idle, iteration, elapsed, spend, and
    training runs used against the cap
  * the DECISION for each node: the allocator's chosen family, the research
    category, and the question the agent could not answer
  * ERRORS as first-class events, with what happened next — a preflight
    rejection that cost nothing, a debug chain that recovered, or an abandoned
    branch
  * evidence tiers, so a viewer sees a good-looking single-seed result being
    held at PRELIMINARY rather than adopted

Usage:
    python3 -m agent.live                    # http://localhost:8000
    python3 -m agent.live --port 8080
    python3 -m agent.live --journal <path>
"""
from __future__ import annotations

import argparse
import json
import os
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
LOGS = os.path.join(ROOT, "logs")
BASELINE = 0.6016
NOISE = 0.0008


def _load_jsonl(path: str) -> list:
    out = []
    if not os.path.exists(path):
        return out
    try:
        with open(path) as fh:
            for ln in fh:
                if ln.strip():
                    try:
                        out.append(json.loads(ln))
                    except json.JSONDecodeError:
                        pass
    except OSError:
        # The agent chmods protected paths while a subprocess runs. A transient
        # read failure must not kill the view; the next poll picks it up.
        return []
    return out


def _event(n: dict, kind: str) -> dict:
    for e in (n.get("events") or []):
        if e.get("type") == kind:
            return e
    return {}


def _agent_running() -> bool:
    """Is a run actually in flight? Cheap check, no psutil dependency."""
    try:
        import subprocess
        r = subprocess.run(["pgrep", "-f", "run_agent.py"],
                           capture_output=True, text=True, timeout=5)
        return bool(r.stdout.strip())
    except Exception:                              # noqa: BLE001
        return False


def _first_error_line(trace: str) -> str:
    for line in (trace or "").splitlines():
        s = line.strip()
        if s and not s.startswith(("File ", "Traceback", "--- ", "exit code")):
            return s[:160]
    return ""


def summarise(n: dict, nodes: list) -> dict:
    """One node, reduced to what a viewer needs to see."""
    from . import evidence as EV

    m = n.get("metrics") or {}
    q = _event(n, "inquiry")
    alloc = _event(n, "allocation")
    cat = _event(n, "research_category")
    paired = _event(n, "paired_result")
    trace = n.get("error_trace") or ""

    if "PREFLIGHT REJECTED" in trace:
        state = "preflight"
    elif n.get("status") == "success":
        state = "ok"
    else:
        state = "fail"

    ev = None
    if paired:
        res = paired.get("result") or {}
        ev = dict(paired.get("evidence") or {})
        ev["n_seeds"] = res.get("n")
    elif m:
        ev = EV.classify(delta=m["primary"] - BASELINE, n_seeds=1)

    # What happened AFTER a failure is the interesting half: did the agent get
    # itself out of it?
    outcome = ""
    if state == "fail":
        nxt = [x for x in nodes if x.get("parent_id") == n["iteration_id"]
               and (x.get("action") or "") == "debug"]
        if nxt:
            recovered = any(x.get("status") == "success" for x in nxt)
            outcome = ("recovered by the debug chain" if recovered
                       else f"debug chain tried {len(nxt)}x, then abandoned")
        else:
            outcome = "branch abandoned"
    elif state == "preflight":
        outcome = "rejected before training — no compute spent"

    hyps = q.get("hypotheses")
    if isinstance(hyps, str):
        hyps = [hyps]

    return {
        "id": n.get("iteration_id"),
        "parent": n.get("parent_id"),
        "action": n.get("action") or "?",
        "path": (n.get("implementation_path") or "A").upper(),
        "state": state,
        "outcome": outcome,
        "primary": round(m["primary"], 5) if m else None,
        "delta": round(m["primary"] - BASELINE, 5) if m else None,
        "sigma": round((m["primary"] - BASELINE) / NOISE, 2) if m else None,
        "category": cat.get("category"),
        "allocator": alloc.get("choice"),
        "question": (q.get("question") or "")[:260],
        "observation": (q.get("observation") or "")[:260],
        "hypotheses": [str(h)[:180] for h in (hyps or [])][:3],
        "plan": (n.get("hypothesis") or "")[:300],
        "error": _first_error_line(trace),
        "seconds": round(n.get("wall_clock_seconds") or 0, 1),
        "evidence": ({"state": ev.get("state"), "why": (ev.get("why") or "")[:180],
                      "n_seeds": ev.get("n_seeds")} if ev else None),
        "paired": ({"control": (paired.get("result") or {}).get("control_mean"),
                    "treatment": (paired.get("result") or {}).get("treatment_mean"),
                    "delta": (paired.get("result") or {}).get("delta"),
                    "sigma": (paired.get("result") or {}).get("sigma"),
                    "t": (paired.get("result") or {}).get("t"),
                    "n": (paired.get("result") or {}).get("n"),
                    "promote": paired.get("promote")} if paired else None),
    }


def state(journal: str) -> dict:
    nodes = _load_jsonl(journal)
    summary_path = os.path.join(LOGS, "final_summary.json")
    fs = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as fh:
                fs = json.load(fh)
        except (OSError, json.JSONDecodeError):
            fs = {}

    scored = [n for n in nodes if n.get("status") == "success" and n.get("metrics")]
    best = max((n["metrics"]["primary"] for n in scored), default=None)
    pre = sum(1 for n in nodes
              if "PREFLIGHT REJECTED" in (n.get("error_trace") or ""))
    crashed = sum(1 for n in nodes if n.get("status") == "error") - pre
    confirms = [n for n in nodes if (n.get("action") or "") == "confirm"]
    promoted = sum(1 for n in nodes if _event(n, "paired_result").get("promote"))
    led = fs.get("budget_ledger") or {}
    spend = fs.get("spend") or {}

    return {
        "running": _agent_running(),
        "generated": time.strftime("%H:%M:%S"),
        "nodes": [summarise(n, nodes) for n in
                  sorted(nodes, key=lambda x: x.get("iteration_id", 0))],
        "kpi": {
            "nodes": len(nodes),
            "scored": len(scored),
            "crashed": crashed,
            "preflight": pre,
            "confirmations": len(confirms),
            "promoted": promoted,
            "best": round(best, 5) if best else None,
            "delta": round(best - BASELINE, 5) if best else None,
            "sigma": round((best - BASELINE) / NOISE, 2) if best else None,
            "training_runs": led.get("training_runs_used"),
            "training_cap": led.get("max_training_runs"),
            "spend": spend.get("total_usd"),
            "spend_cap": spend.get("ceiling_usd"),
            "incumbent": 0.60541,
        },
    }


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Autonomous ML Research Agent — live</title><style>
:root{--bg:#0d0f14;--fg:#eceff4;--mut:#8b94a3;--line:#1f2430;--card:#151924;
--ok:#3fb950;--fail:#f85149;--pre:#d29922;--acc:#58a6ff;--warn:#e3cf87}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;background:#0d0f14ee;backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);padding:16px 26px;z-index:9}
.top{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
h1{margin:0;font-size:18px;letter-spacing:.2px}
.live{display:flex;align-items:center;gap:7px;font-size:13px;font-weight:600}
.dot{width:9px;height:9px;border-radius:50%;background:var(--mut)}
.dot.on{background:var(--ok);box-shadow:0 0 0 0 rgba(63,185,80,.7);
animation:pulse 1.8s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 9px rgba(63,185,80,0)}
100%{box-shadow:0 0 0 0 rgba(63,185,80,0)}}
.kpis{display:flex;gap:9px;flex-wrap:wrap;margin-top:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;
padding:7px 12px;min-width:104px}
.kpi .v{font-size:18px;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .k{color:var(--mut);font-size:10.5px;text-transform:uppercase;letter-spacing:.6px}
.kpi.good .v{color:var(--ok)} .kpi.bad .v{color:var(--fail)}
.kpi.warn .v{color:var(--pre)}
main{padding:18px 26px 80px;max-width:1180px}
.node{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--mut);
border-radius:9px;padding:13px 16px;margin:9px 0;animation:in .45s ease}
@keyframes in{from{opacity:0;transform:translateY(-7px)}to{opacity:1}}
.node.ok{border-left-color:var(--ok)}
.node.fail{border-left-color:var(--fail)}
.node.preflight{border-left-color:var(--pre)}
.node.newest{box-shadow:0 0 0 2px var(--acc)}
.hdr{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.id{font-weight:800;color:var(--acc);font-size:15px}
.tag{font-size:11px;padding:2px 8px;border-radius:99px;background:#232a3a;color:#aab4c4}
.tag.confirm{background:#16324f;color:#9ecbff}
.tag.improve{background:#123524;color:#7ee2a8}
.tag.debug{background:#4a1f28;color:#ff9fa8}
.tag.B{background:#3a2a4d;color:#d5b3ff}
.score{margin-left:auto;font-weight:800;font-variant-numeric:tabular-nums;font-size:16px}
.sig{color:var(--mut);font-size:12px;margin-left:7px;font-weight:400}
.row{margin-top:7px;font-size:13.5px}
.row b{color:var(--acc);margin-right:6px}
.mut{color:#b9c1cd}
ul{margin:5px 0 0 19px;padding:0;color:#b9c1cd}
.err{margin-top:7px;background:#2a161a;border-radius:6px;padding:7px 10px;
color:#ff9fa8;font-family:ui-monospace,Menlo,monospace;font-size:12px}
.outcome{margin-top:6px;font-size:12.5px;color:var(--warn)}
.ev{margin-top:7px;display:inline-block;font-size:12px;padding:4px 10px;
border-radius:6px;background:#1b2130;font-weight:600}
.ev.CONFIRMED{background:#10331f;color:#7ee2a8}
.ev.PRELIMINARY{background:#332c14;color:#e8d18a}
.ev.REJECTED,.ev.UNCONFIRMED{background:#33191d;color:#ff9fa8}
.ev .why{font-weight:400;color:var(--mut);margin-left:8px}
.paired{margin-top:7px;background:#101d2b;border-radius:7px;padding:8px 11px;font-size:13px}
.idle{color:var(--mut);padding:40px 0;text-align:center}
</style></head><body>
<header><div class="top">
<h1>Autonomous ML Research Agent</h1>
<span class="live"><span class="dot" id="dot"></span><span id="run">—</span></span>
<span style="color:var(--mut);font-size:12px;margin-left:auto" id="ts"></span>
</div><div class="kpis" id="kpis"></div></header>
<main id="tree"><div class="idle">waiting for the journal…</div></main>
<script>
let seen = new Set();
const f = (v,d=5)=> v==null ? "—" : Number(v).toFixed(d);
function kpi(k,v,cls){return `<div class="kpi ${cls||''}"><div class="v">${v}</div>
<div class="k">${k}</div></div>`;}
function render(s){
  document.getElementById('dot').className = 'dot' + (s.running?' on':'');
  document.getElementById('run').textContent = s.running ? 'RUNNING' : 'idle';
  document.getElementById('ts').textContent = 'updated ' + s.generated;
  const k = s.kpi;
  document.getElementById('kpis').innerHTML =
    kpi('nodes', k.nodes) +
    kpi('completed', k.scored, 'good') +
    kpi('crashed', k.crashed, k.crashed?'bad':'') +
    kpi('preflight (free)', k.preflight, k.preflight?'warn':'') +
    kpi('confirmations', k.confirmations) +
    kpi('promoted', k.promoted) +
    kpi('best primary', f(k.best)) +
    kpi('Δ vs baseline', k.delta==null?'—':(k.delta>0?'+':'')+f(k.delta)+' ('+k.sigma+'σ)') +
    kpi('training runs', (k.training_runs??'—')+' / '+(k.training_cap??'—')) +
    kpi('spend', k.spend==null?'—':'$'+Number(k.spend).toFixed(2));
  const depth = {}; const html = [];
  const newest = s.nodes.length ? s.nodes[s.nodes.length-1].id : null;
  for (const n of s.nodes){
    depth[n.id] = (n.parent!=null && depth[n.parent]!=null) ? depth[n.parent]+1 : 0;
    const isNew = !seen.has(n.id); seen.add(n.id);
    let h = `<div class="node ${n.state}${n.id===newest?' newest':''}"
      style="margin-left:${depth[n.id]*28}px">`;
    h += `<div class="hdr"><span class="id">#${n.id}</span>
      <span class="tag ${n.action}">${n.action}</span>
      <span class="tag ${n.path}">Path ${n.path}</span>`;
    if (n.category) h += `<span class="tag">${n.category}</span>`;
    if (n.primary!=null) h += `<span class="score">${f(n.primary)}
      <span class="sig">${n.delta>0?'+':''}${f(n.delta)} · ${n.sigma}σ</span></span>`;
    h += `</div>`;
    if (n.allocator) h += `<div class="row mut">allocator chose <b
      style="color:var(--acc)">${n.allocator}</b></div>`;
    if (n.observation) h += `<div class="row"><b>Observed</b><span class="mut">${n.observation}</span></div>`;
    if (n.question) h += `<div class="row"><b>Question</b>${n.question}</div>`;
    if (n.hypotheses && n.hypotheses.length)
      h += `<div class="row"><b>Competing hypotheses</b><ul>`+
           n.hypotheses.map(x=>`<li>${x}</li>`).join('')+`</ul></div>`;
    if (n.plan) h += `<div class="row"><b>Experiment</b><span class="mut">${n.plan}</span></div>`;
    if (n.paired) h += `<div class="paired"><b>Paired confirmation</b> control
      ${f(n.paired.control)} → treatment ${f(n.paired.treatment)} · Δ
      ${n.paired.delta>0?'+':''}${f(n.paired.delta)} (${n.paired.sigma}σ) ·
      t=${n.paired.t} · ${n.paired.n} seeds ·
      promote=<b>${n.paired.promote?'YES':'NO'}</b></div>`;
    if (n.error) h += `<div class="err">${n.error}</div>`;
    if (n.outcome) h += `<div class="outcome">→ ${n.outcome}</div>`;
    if (n.evidence) h += `<div class="ev ${n.evidence.state}">${n.evidence.state}`+
      (n.evidence.n_seeds?` · ${n.evidence.n_seeds} seed(s)`:``)+
      `<span class="why">${n.evidence.why}</span></div>`;
    h += `</div>`; html.push(h);
  }
  document.getElementById('tree').innerHTML =
    html.join('') || '<div class="idle">no nodes yet — the run has not scored anything</div>';
}
async function poll(){
  try{ const r = await fetch('state.json?t='+Date.now());
       render(await r.json()); }catch(e){}
  setTimeout(poll, 3000);
}
poll();
</script></body></html>"""


def serve(journal: str, port: int = 8000) -> None:
    import http.server

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                          # noqa: N802
            if self.path.startswith("/state.json"):
                body = json.dumps(state(journal), default=str).encode()
                ctype = "application/json"
            else:
                body = PAGE.encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):                 # quiet: this gets recorded
            pass

    with http.server.ThreadingHTTPServer(("", port), H) as srv:
        print(f"live view  →  http://localhost:{port}")
        print(f"journal    →  {journal}")
        print("polls every 3s; leave it open while the agent runs. ctrl-c to stop.")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default=os.path.join(LOGS, "journal.jsonl"))
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--once", action="store_true",
                    help="print one state snapshot as JSON and exit")
    a = ap.parse_args()
    if a.once:
        print(json.dumps(state(a.journal), indent=2, default=str))
        return
    serve(a.journal, a.port)


if __name__ == "__main__":
    main()
