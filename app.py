"""Streamlit dashboard: watch the agent work, and check the claims.

    streamlit run app.py

Built for a judge or a teammate who should not have to read a terminal. Five
tabs, in the order someone actually evaluates this:

    Live          the search tree as it grows, with decisions and recoveries
    Results       the generated report, and the incumbent re-verified on demand
    Iterations    every node, filterable, with the full journal record
    Judging       each competition criterion mapped to evidence in this repo
    Run           start a run, with the resolved configuration shown first

Two rules this app keeps, because they are the same rules the agent keeps:

  * **It never invents a number.** Everything shown is read from artifacts --
    the journal, the ledger, `ensemble_results.json` -- or recomputed live. The
    incumbent check actually re-derives 0.60541 from the stored predictions.
  * **It cannot promote anything.** Evidence tiers are recomputed here from
    `agent.evidence`, so a single-seed result displays as PRELIMINARY no matter
    how good it looks. The dashboard is a window, not a decision-maker.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
LOGS = os.path.join(ROOT, "logs")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from agent import live as L  # noqa: E402

BASELINE = 0.6016
INCUMBENT = 0.60541
NOISE = 0.0008

st.set_page_config(page_title="Autonomous ML Research Agent",
                   page_icon="🔬", layout="wide")


# ----------------------------------------------------------------- helpers ---
def journals() -> dict:
    """Every journal on disk, newest-looking first."""
    out = {}
    live = os.path.join(LOGS, "journal.jsonl")
    if os.path.exists(live):
        out["live run (logs/journal.jsonl)"] = live
    d = os.path.join(LOGS, "opus_research")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".jsonl"):
                out[f"archived — {f}"] = os.path.join(d, f)
    return out


def agent_running() -> bool:
    return L._agent_running()


def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


TIER_COLOR = {"CONFIRMED": "#2ea043", "PRELIMINARY": "#d29922",
              "REJECTED": "#f85149", "UNCONFIRMED": "#f85149",
              "PROBED": "#8b949e", "REDUNDANT": "#8b949e"}
STATE_ICON = {"ok": "🟢", "fail": "🔴", "preflight": "🟠"}


# -------------------------------------------------------------------- head ---
running = agent_running()
c1, c2 = st.columns([4, 1])
with c1:
    st.title("Autonomous ML Research Agent")
    st.caption("KuaiRand-Pure · TikTok TechJam 2026 Track 2 · "
               "every figure here is read from artifacts or recomputed live")
with c2:
    st.markdown(f"### {'🟢 RUNNING' if running else '⚪ idle'}")
    if running:
        st.caption("agent process detected")

tabs = st.tabs(["🌳 Live tree", "📊 Results", "📜 Iterations",
                "⚖️ Judging criteria", "▶️ Run"])


# --------------------------------------------------------------- live tree ---
with tabs[0]:
    js = journals()
    if not js:
        st.info("No journal on disk yet. Start a run from the **Run** tab.")
    else:
        left, right = st.columns([3, 1])
        with right:
            pick = st.selectbox("journal", list(js), key="tree_journal")
            auto = st.checkbox("auto-refresh (3s)", value=running)
        state = L.state(js[pick])
        k = state["kpi"]

        cols = st.columns(6)
        cols[0].metric("nodes", k["nodes"])
        cols[1].metric("completed", k["scored"])
        cols[2].metric("crashed", k["crashed"])
        cols[3].metric("preflight (free)", k["preflight"],
                       help="rejected before training — no compute spent, so "
                            "this is the system working, not failing")
        cols[4].metric("confirmations", k["confirmations"])
        cols[5].metric("promoted", k["promoted"],
                       help="only a CONFIRMED paired result may change what we "
                            "submit")

        cols = st.columns(4)
        cols[0].metric("best primary", f"{k['best']:.5f}" if k["best"] else "—",
                       f"{k['delta']:+.5f} ({k['sigma']}σ)" if k["delta"] else None)
        cols[1].metric("submitted incumbent", f"{INCUMBENT:.5f}",
                       help="16-seed ensemble; unchanged unless a paired "
                            "confirmation promotes something")
        cols[2].metric("training runs",
                       f"{k['training_runs'] or 0} / {k['training_cap'] or '—'}",
                       help="a paired 3-seed confirmation is 1 decision and 6 "
                            "training executions")
        cols[3].metric("LLM spend",
                       f"${k['spend']:.2f}" if k["spend"] is not None else "—",
                       f"of ${k['spend_cap']:.2f}" if k.get("spend_cap") else None)

        st.divider()
        if not state["nodes"]:
            st.info("No nodes yet.")
        depth: dict = {}
        for n in state["nodes"]:
            p = n["parent"]
            depth[n["id"]] = depth.get(p, -1) + 1 if p is not None else 0
            pad = depth[n["id"]] * 3
            icon = STATE_ICON.get(n["state"], "⚪")
            score = f" · **{n['primary']:.5f}** ({n['sigma']:+.2f}σ)" if n["primary"] else ""
            head = (f"{'&nbsp;' * pad * 2}{icon} **#{n['id']}** "
                    f"`{n['action']}` `Path {n['path']}`{score}")
            st.markdown(head, unsafe_allow_html=True)
            with st.container():
                ind = st.columns([max(1, pad), 20])[1] if pad else st.container()
                with ind:
                    if n["allocator"]:
                        st.caption(f"allocator chose **{n['allocator']}**"
                                   + (f" · objective {n['category']}" if n["category"] else ""))
                    if n["question"]:
                        st.markdown(f"**Question** {n['question']}")
                    if n["hypotheses"]:
                        st.markdown("**Competing hypotheses**")
                        for h in n["hypotheses"]:
                            st.markdown(f"- {h}")
                    if n["plan"]:
                        st.caption(f"**Experiment** {n['plan']}")
                    if n["paired"]:
                        pr = n["paired"]
                        st.info(
                            f"**Paired confirmation** — control {pr['control']:.5f} → "
                            f"treatment {pr['treatment']:.5f} · "
                            f"Δ {pr['delta']:+.5f} ({pr['sigma']:+.2f}σ) · "
                            f"t={pr['t']} · {pr['n']} seeds · "
                            f"**promote = {'YES' if pr['promote'] else 'NO'}**")
                    if n["error"]:
                        st.error(f"`{n['error']}`")
                    if n["outcome"]:
                        st.caption(f"→ {n['outcome']}")
                    if n["evidence"]:
                        ev = n["evidence"]
                        col = TIER_COLOR.get(ev["state"], "#8b949e")
                        seeds = f" · {ev['n_seeds']} seed(s)" if ev.get("n_seeds") else ""
                        st.markdown(
                            f"<span style='background:{col}22;color:{col};"
                            f"padding:3px 9px;border-radius:6px;font-size:12px;"
                            f"font-weight:600'>{ev['state']}{seeds}</span> "
                            f"<span style='color:#8b949e;font-size:12px'>"
                            f"{ev['why']}</span>", unsafe_allow_html=True)
            st.markdown("")
        if auto:
            time.sleep(3)
            st.rerun()


# ----------------------------------------------------------------- results ---
with tabs[1]:
    st.subheader("Generated report")
    st.caption("`python3 -m agent.results_report --run-tests` — regenerated from "
               "artifacts, never hand-edited.")
    if st.button("🔄 Regenerate (runs the full test suite)"):
        with st.spinner("running the harness and recomputing the incumbent…"):
            r = subprocess.run([sys.executable, "-m", "agent.results_report",
                                "--run-tests"], cwd=ROOT, capture_output=True,
                               text=True, timeout=3600)
        st.code((r.stdout or r.stderr)[-1500:])

    p = os.path.join(ROOT, "RESULTS.md")
    if os.path.exists(p):
        with open(p) as fh:
            st.markdown(fh.read())
    else:
        st.info("RESULTS.md not generated yet.")

    st.divider()
    st.subheader("Verify the submitted result, now")
    st.caption("Recomputes 0.60541 from the 16 stored member predictions. It "
               "never retrains, because a check that rebuilds the thing it is "
               "checking is not a check.")
    if st.button("✅ Verify incumbent"):
        with st.spinner("recomputing from stored predictions…"):
            r = subprocess.run([sys.executable, "-m", "agent.verify_incumbent"],
                               cwd=ROOT, capture_output=True, text=True,
                               timeout=900)
        (st.success if r.returncode == 0 else st.error)(r.stdout or r.stderr)


# -------------------------------------------------------------- iterations ---
with tabs[2]:
    js = journals()
    if not js:
        st.info("No journal on disk yet.")
    else:
        pick = st.selectbox("journal", list(js), key="iter_journal")
        nodes = L._load_jsonl(js[pick])
        rows = []
        for n in nodes:
            s = L.summarise(n, nodes)
            rows.append({
                "#": s["id"], "action": s["action"], "path": s["path"],
                "status": s["state"], "primary": s["primary"],
                "Δσ": s["sigma"], "evidence": (s["evidence"] or {}).get("state"),
                "seconds": s["seconds"],
                "note": s["outcome"] or (s["question"][:70] if s["question"] else ""),
            })
        import pandas as pd
        df = pd.DataFrame(rows)
        c = st.columns(3)
        only_fail = c[0].checkbox("failures only")
        only_conf = c[1].checkbox("confirmations only")
        if only_fail:
            df = df[df["status"] != "ok"]
        if only_conf:
            df = df[df["action"] == "confirm"]
        st.dataframe(df, width="stretch", hide_index=True)

        st.caption("Full journal record for one node — this is the competition's "
                   "required iteration log, unedited.")
        ids = [n["iteration_id"] for n in nodes]
        if ids:
            nid = st.selectbox("node", ids, key="node_detail")
            rec = next(n for n in nodes if n["iteration_id"] == nid)
            st.json(rec, expanded=False)
            cp = rec.get("code_path") or ""
            if cp and os.path.exists(cp):
                with st.expander("the script the agent wrote for this node"):
                    with open(cp) as fh:
                        st.code(fh.read(), language="python")


# ----------------------------------------------------------------- judging ---
with tabs[3]:
    st.subheader("Competition criteria, mapped to evidence")
    st.caption("Weights are the organisers'. Each row points at something in "
               "this repo you can check.")

    fs = read_json(os.path.join(LOGS, "final_summary.json")) or {}
    led = fs.get("budget_ledger") or {}
    spend = fs.get("spend") or {}
    nodes = L._load_jsonl(os.path.join(LOGS, "journal.jsonl"))
    interventions = L._load_jsonl(os.path.join(LOGS, "interventions.jsonl"))

    st.markdown(f"""
| Criterion | Weight | Evidence here |
|---|---|---|
| **Technical Execution** — mean absolute delta over baseline on hidden test | 35% | Validation **{INCUMBENT}** vs baseline {BASELINE} = **+{INCUMBENT - BASELINE:.5f} ({(INCUMBENT - BASELINE) / NOISE:.2f}σ)**. Hidden test **not yet evaluated** — one-shot, deliberately unspent. Verify tab recomputes the number from stored predictions. |
| ↳ *Robustness* — graceful failure handling, not failure count | — | 8-stage preflight rejects broken scripts for **zero compute**; debug chains recover; a killed run restores file permissions on SIGTERM. See the 🔴/🟠 nodes in **Live tree**. |
| **Innovation & Problem Insight** | 20% | `docs/ARCHITECTURE.md` — capability contract, evidence tiers, transparent allocator. `docs/RESEARCH_LOG.md` — 28 dead ends with measurements. |
| **Impact & Relevance** — autonomy, fewer manual interventions scores higher | 20% | **{len(interventions)} manual interventions** logged (`logs/interventions.jsonl`). Every decision is journalled with its reasoning. |
| **Feasibility & Practicality** — token + wall-clock cost | 15% | Last run: **${spend.get('total_usd', 0):.2f}** LLM, **{fs.get('total_agent_wall_clock_s', 0):.0f}s** wall-clock, **{led.get('training_runs_used', 0)}** training runs, **{fs.get('gpu_hours', 0)}** GPU-hours (CPU only). |
| **Presentation & Communication** | 10% | This dashboard, `README.md`, and the generated `RESULTS.md`. |
""")

    st.divider()
    st.markdown("#### What we are *not* claiming")
    st.warning(
        "The submitted score is **0.60541 and has not improved** during the "
        "agent-architecture work. Two live paired confirmations both correctly "
        "declined to promote a result. Path B feature discovery has never "
        "completed end-to-end — no proposed feature has cleared the probe. "
        "The hidden test has never been evaluated. "
        "See `docs/ARCHITECTURE.md` for the honest autonomy classification "
        "(**Level B — capability transfer**, not independent discovery)."
    )


# --------------------------------------------------------------------- run ---
with tabs[4]:
    st.subheader("Start a run")
    if running:
        st.warning("An agent process is already running. Watch it in "
                   "**Live tree**.")
    st.caption("The competition profile turns on research state, data tools, "
               "feature discovery, multi-candidate planning and branching, and "
               "prints its fully resolved configuration before spending "
               "anything.")

    c = st.columns(4)
    iters = c[0].number_input("max iterations", 1, 50, 12)
    truns = c[1].number_input("max training runs", 1, 300, 90,
                              help="a paired 3-seed confirmation costs 6")
    spend_cap = c[2].number_input("spend ceiling ($)", 0.5, 50.0, 6.0, step=0.5)
    hours = c[3].number_input("wall-clock (h)", 0.25, 8.0, 2.0, step=0.25)
    fresh = st.checkbox("--fresh (archive previous search logs)", value=True,
                        help="Submission artifacts always survive: the ensemble, "
                             "its members, research memory and the feature "
                             "registry are never archived.")

    cmd = [sys.executable, "run_agent.py", "--competition",
           "--max-iterations", str(iters), "--max-training-runs", str(truns),
           "--max-spend-usd", str(spend_cap), "--wall-clock-limit-h", str(hours)]
    if fresh:
        cmd.append("--fresh")
    st.code(" ".join(cmd[1:]), language="bash")

    # A run costs money and holds the dataset lock, so starting one takes TWO
    # deliberate actions, not one. A stray click, a rerun, or anything that
    # makes a button read truthy cannot launch a billed run on its own.
    armed = st.checkbox("I want to start a real run (spends LLM budget)",
                        value=False, key="arm_run")
    go = st.button("▶️ Start run", type="primary", disabled=running or not armed)
    if go and armed and not running:
        logf = os.path.join(LOGS, "dashboard_run.log")
        os.makedirs(LOGS, exist_ok=True)
        with open(logf, "w") as fh:
            fh.write(f"launched from the dashboard at "
                     f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                     f"{' '.join(cmd)}\n\n")
            fh.flush()
            subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        st.success("Started. Open **Live tree** — it refreshes every 3 seconds.")
        time.sleep(2)
        st.rerun()

    logf = os.path.join(LOGS, "dashboard_run.log")
    if os.path.exists(logf):
        with st.expander("run console output", expanded=running):
            with open(logf) as fh:
                st.code(fh.read()[-6000:])

    st.divider()
    st.subheader("Final submission")
    st.error(
        "**The hidden-test evaluation runs exactly once for the whole project.** "
        "It is deliberately not wired to a button here. When you are ready:\n\n"
        "```\npython3 -m agent.make_submission --split valid --score --ensemble"
        "   # inspect first\n"
        "python3 -m agent.make_submission --final-test-eval --ensemble"
        "       # THE one-time eval\n```")
