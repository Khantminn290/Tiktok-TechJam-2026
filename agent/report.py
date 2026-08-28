"""Run-report renderer: turns logs/journal.jsonl into a human-readable summary
(for the Devpost writeup and results table).

Usage: python3 -m agent.report [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.contracts import error_headline  # noqa: E402

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS = os.path.join(_ROOT, "logs")
BASELINE_VALID = 0.6016
BASELINE_SEED_STD = 0.0008   # official FM baseline's own 5-seed std


def load_journal():
    path = os.path.join(LOGS, "journal.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render() -> str:
    nodes = load_journal()
    if not nodes:
        return "journal is empty — no run recorded yet"
    lines = []
    succ = [n for n in nodes if n["status"] == "success" and n.get("metrics")]
    best = max(succ, key=lambda n: n["metrics"]["primary"], default=None)
    tok = sum(n.get("tokens_used", 0) for n in nodes)
    tb = {}
    for n in nodes:
        for k, v in (n.get("token_breakdown") or {}).items():
            tb[k] = tb.get(k, 0) + v
    train_s = sum(n.get("wall_clock_seconds", 0.0) for n in nodes)
    errors = [n for n in nodes if n["status"] == "error"]
    recovered = sum(1 for n in nodes if n["action"] == "debug"
                    and n["status"] == "success")

    ipath = os.path.join(LOGS, "interventions.jsonl")
    interventions = []
    if os.path.exists(ipath):
        with open(ipath) as fh:
            interventions = [json.loads(line) for line in fh if line.strip()]

    lines.append("=" * 78)
    lines.append("AUTONOMOUS ML RESEARCH AGENT — RUN REPORT (KuaiRand-Pure)")
    lines.append("=" * 78)
    lines.append(f"iterations: {len(nodes)}  "
                 f"(success {len(succ)}, error {len(errors)}, "
                 f"debug-recoveries {recovered})")
    if best:
        m = best["metrics"]
        lines.append(f"best: node {best['iteration_id']} — valid primary "
                     f"{m['primary']:.4f} (GAUC {m['GAUC']:.4f}, "
                     f"nDCG@5 {m['nDCG@5']:.4f})")
        d = m["primary"] - BASELINE_VALID
        sig = d / BASELINE_SEED_STD
        verdict = ("within seed noise" if abs(sig) < 2
                   else "beyond seed noise" if abs(sig) < 3
                   else "clearly beyond seed noise")
        lines.append(f"delta over official baseline (valid primary "
                     f"{BASELINE_VALID}): {d:+.4f}  "
                     f"= {sig:+.1f}x the baseline's own seed std "
                     f"({BASELINE_SEED_STD}) - {verdict}")
        lines.append(f"best menu choices: {json.dumps(best['menu_choices'])}")
    lines.append(f"total LLM tokens (input+output, all types summed): {tok:,d}")
    if tb:
        fresh = tb.get("input_tokens", 0) + tb.get("cache_creation_input_tokens", 0)
        lines.append(f"  of which: fresh input {fresh:,d} "
                     f"(uncached {tb.get('input_tokens', 0):,d} + cache-writes "
                     f"{tb.get('cache_creation_input_tokens', 0):,d}), "
                     f"cache reads {tb.get('cache_read_input_tokens', 0):,d}, "
                     f"output {tb.get('output_tokens', 0):,d}")
    lines.append(f"total training wall-clock: {train_s/60:.1f} min")
    lines.append(f"manual interventions: {len(interventions)}")
    for iv in interventions:
        lines.append(f"  - {iv.get('time_iso', '?')}: {iv['reason']}")

    fs = os.path.join(LOGS, "final_summary.json")
    if os.path.exists(fs):
        with open(fs) as fh:
            s = json.load(fh)
        lines.append(f"stop reason: {s.get('stop_reason')}")
        lines.append(f"total agent wall-clock: "
                     f"{s.get('total_agent_wall_clock_s', 0)/60:.1f} min")
        lines.append(f"GPU-hours (measured): {s.get('gpu_hours', 0.0)}"
                     f"  devices used: {', '.join(s.get('devices_used', ['cpu']))}")
        sp = s.get("spend")
        if sp:
            lines.append(f"LLM spend: ${sp.get('total_usd', 0):.4f} of "
                         f"${sp.get('ceiling_usd', 0):.2f} ceiling  "
                         f"({sp.get('provider')}:{sp.get('model')}, "
                         f"{sp.get('rate_card', '')})")
            lines.append(f"  mean ${sp.get('mean_usd_per_iteration', 0):.4f}/iteration, "
                         f"max ${sp.get('max_usd_per_iteration', 0):.4f}")

    lines.append("-" * 78)
    lines.append("per-iteration history:")
    for n in nodes:
        if n["status"] == "success":
            sc = f"primary {n['metrics']['primary']:.4f}"
        else:
            sc = f"ERROR ({error_headline(n.get('error_trace'), 70)})"
        parent = "" if n.get("parent_id") is None else f"<-{n['parent_id']}"
        hyp = (n.get("hypothesis") or "(llm-stage failure)").replace("\n", " ")
        lines.append(f"  [{n['iteration_id']:2d}] {n['action']:7s}{parent:5s} {sc}")
        lines.append(f"       {hyp[:140]}")
    return "\n".join(lines)


def render_tree_html() -> str:
    """AIDE-style search-tree view: one card per node, nested under its parent."""
    nodes = load_journal()
    by_parent = {}
    for n in nodes:
        by_parent.setdefault(n.get("parent_id"), []).append(n)
    best = max((n for n in nodes if n["status"] == "success" and n.get("metrics")),
               key=lambda n: n["metrics"]["primary"], default=None)
    best_id = best["iteration_id"] if best else None

    def card(n, depth):
        if n["status"] == "success":
            m = n["metrics"]
            score = (f"primary <b>{m['primary']:.4f}</b> "
                     f"(GAUC {m['GAUC']:.4f}, nDCG@5 {m['nDCG@5']:.4f})")
            cls = "best" if n["iteration_id"] == best_id else "ok"
        else:
            score = f"ERROR — {error_headline(n.get('error_trace'), 120)}"
            cls = "err"
        esc = (lambda s: (s or "").replace("&", "&amp;").replace("<", "&lt;")
               .replace(">", "&gt;"))
        html = [f'<div class="node {cls}" style="margin-left:{depth * 28}px">',
                f'<div class="hdr">#{n["iteration_id"]} '
                f'<span class="act">{n["action"]}</span> {score}</div>',
                f'<div class="ch">{esc(json.dumps(n["menu_choices"]))}</div>',
                f'<div class="hyp">{esc(n.get("hypothesis"))[:600]}</div>']
        if n.get("decide_reason"):
            html.append(f'<div class="why">policy: {esc(n["decide_reason"])[:300]}</div>')
        html.append("</div>")
        for c in by_parent.get(n["iteration_id"], []):
            html.append(card(c, depth + 1))
        return "\n".join(html)

    body = "\n".join(card(n, 0) for n in by_parent.get(None, []))
    return f"""<!doctype html><meta charset="utf-8">
<title>Agent search tree — KuaiRand-Pure</title>
<style>
 body{{font:14px/1.45 -apple-system,Segoe UI,sans-serif;background:#0f1115;color:#e6e6e6;padding:24px}}
 h1{{font-size:18px}}
 .node{{border-left:3px solid #444;background:#171a21;padding:8px 12px;margin:6px 0;border-radius:6px}}
 .node.ok{{border-color:#3b82f6}} .node.best{{border-color:#22c55e;background:#16241a}}
 .node.err{{border-color:#ef4444;background:#241618}}
 .hdr{{font-weight:600}} .act{{color:#9ca3af;font-weight:400}}
 .ch{{color:#93c5fd;font-family:ui-monospace,monospace;font-size:11px;margin:4px 0}}
 .hyp{{color:#d1d5db;font-size:12px}} .why{{color:#9ca3af;font-size:11px;margin-top:4px;font-style:italic}}
</style>
<h1>Autonomous agent search tree — KuaiRand-Pure ({len(nodes)} iterations,
best {'' if best is None else f"node {best_id} @ {best['metrics']['primary']:.4f}"})</h1>
{body}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None,
                    help="also write a machine-readable summary to this path")
    ap.add_argument("--html", nargs="?", const=os.path.join(LOGS, "tree.html"),
                    default=None, help="write the search-tree visualization")
    a = ap.parse_args()
    text = render()
    print(text)
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(load_journal(), fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {a.json}")
    if a.html:
        with open(a.html, "w") as fh:
            fh.write(render_tree_html())
        print(f"wrote {a.html}")
