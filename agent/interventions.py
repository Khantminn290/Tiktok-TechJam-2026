"""Manual-intervention log (append-only, separate from the agent's own journal).

Any human action during a run — editing config, restarting, unlocking a menu
option, killing a stuck process — must be recorded here. The count feeds the
Impact & Relevance (autonomy) score and is a required deliverable.

Usage: python3 -m agent.interventions "reason for the intervention"
       python3 -m agent.interventions --list
"""
from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(_ROOT, "logs", "interventions.jsonl")


def log_intervention(reason: str) -> dict:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    rec = {"timestamp": time.time(),
           "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
           "reason": reason}
    with open(PATH, "a") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def list_interventions() -> list[dict]:
    if not os.path.exists(PATH):
        return []
    with open(PATH) as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        rows = list_interventions()
        print(f"{len(rows)} manual intervention(s) logged")
        for r in rows:
            print(f"  {r['time_iso']}  {r['reason']}")
    elif len(sys.argv) >= 2:
        rec = log_intervention(" ".join(sys.argv[1:]))
        print(f"logged intervention: {rec['reason']}")
    else:
        print(__doc__)
