"""Migrate a legacy journal + solutions/ + runs/ into self-contained node bundles.

Usage:
  python -m agent.migrate_logs --source logs/archive_YYYYMMDD_HHMMSS --destination logs

The source is never modified. Each destination node becomes:
  logs/nodes/node_NNN/{solution.py,record.json,metrics.json,resource.json,scores_*.npy}
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def migrate(source: str, destination: str, activate: bool = False) -> dict:
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)
    journal = os.path.join(source, "journal.jsonl")
    if not os.path.isfile(journal):
        raise FileNotFoundError(f"journal not found: {journal}")

    with open(journal, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]
    nodes_dir = os.path.join(destination, "nodes")
    project_root = os.path.dirname(destination)
    if os.path.basename(destination) == "smoke":
        project_root = os.path.dirname(os.path.dirname(destination))
    os.makedirs(nodes_dir, exist_ok=True)
    rewritten = []

    for record in records:
        node_id = int(record["iteration_id"])
        node_dir = os.path.join(nodes_dir, f"node_{node_id:03d}")
        os.makedirs(node_dir, exist_ok=True)

        legacy_solution = os.path.join(source, "solutions", f"node_{node_id:03d}.py")
        solution = os.path.join(node_dir, "solution.py")
        if os.path.isfile(legacy_solution):
            shutil.copy2(legacy_solution, solution)
            record["code_path"] = os.path.relpath(solution, project_root)
        elif record.get("code_path") and os.path.isfile(record["code_path"]):
            shutil.copy2(record["code_path"], solution)
            record["code_path"] = os.path.relpath(solution, project_root)
        else:
            record["code_path"] = ""

        legacy_run = os.path.join(source, "runs", f"node_{node_id:03d}")
        if os.path.isdir(legacy_run):
            for name in os.listdir(legacy_run):
                src = os.path.join(legacy_run, name)
                dst = os.path.join(node_dir, name)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

        with open(os.path.join(node_dir, "record.json"), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        rewritten.append(record)

    if activate:
        os.makedirs(destination, exist_ok=True)
        with open(os.path.join(destination, "journal.jsonl"), "w", encoding="utf-8") as fh:
            for record in rewritten:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        for name in ("best_solution.py", "best_metrics.json", "final_summary.json",
                     "tree.html", "interventions.jsonl"):
            src = os.path.join(source, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(destination, name))
        best_path = os.path.join(destination, "best_metrics.json")
        if os.path.isfile(best_path):
            with open(best_path, encoding="utf-8") as fh:
                best = json.load(fh)
            best_id = int(best["iteration_id"])
            best["code_path"] = os.path.relpath(os.path.join(
                nodes_dir, f"node_{best_id:03d}", "solution.py"), project_root)
            with open(best_path, "w", encoding="utf-8") as fh:
                json.dump(best, fh, indent=2, ensure_ascii=False)

    return {"source": source, "destination": destination,
            "nodes": len(rewritten), "activated": activate}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--destination", required=True)
    ap.add_argument("--activate", action="store_true")
    args = ap.parse_args()
    print(json.dumps(migrate(args.source, args.destination, args.activate), indent=2))


if __name__ == "__main__":
    main()
