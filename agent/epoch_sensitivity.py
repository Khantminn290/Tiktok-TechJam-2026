"""Decide whether epoch-selection research is justified by stored captures.

This is deliberately diagnostic-only. The repository stores per-epoch
validation scores, but not per-epoch test predictions. Consequently those
captures can tell us whether stopping is seed-sensitive; they cannot safely
select and ship a new stopping rule after inspecting official validation.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "logs", "opus_research",
                             "epoch_primaries.json")
DEFAULT_OUT = os.path.join(ROOT, "results", "epoch_sensitivity.json")
NOISE_FLOOR = 0.0008


def analyse(curves: dict, noise_floor: float = NOISE_FLOOR) -> dict:
    peaks = []
    plausible_spreads = []
    for seed_text, points in sorted(curves.items(), key=lambda x: int(x[0])):
        clean = [(int(epoch), float(score)) for epoch, score in points]
        if not clean:
            continue
        peak_epoch, peak_score = max(clean, key=lambda x: x[1])
        window = [score for epoch, score in clean
                  if abs(epoch - peak_epoch) <= 2]
        spread = max(window) - min(window) if window else 0.0
        peaks.append({"seed": int(seed_text), "peak_epoch": peak_epoch,
                      "peak_primary": peak_score,
                      "local_window_spread": spread,
                      "local_window_sigma": spread / noise_floor})
        plausible_spreads.append(spread)

    epochs = [p["peak_epoch"] for p in peaks]
    scores = [p["peak_primary"] for p in peaks]
    mean_local = statistics.mean(plausible_spreads) if plausible_spreads else 0.0
    material = mean_local >= noise_floor
    return {
        "schema": "epoch_sensitivity/1",
        "source": os.path.relpath(DEFAULT_INPUT, ROOT),
        "split": "validation only",
        "seeds": len(peaks),
        "noise_floor": noise_floor,
        "peaks": peaks,
        "peak_epoch_range": ([min(epochs), max(epochs)] if epochs else None),
        "peak_primary_std": (statistics.pstdev(scores) if len(scores) > 1 else 0.0),
        "mean_local_window_spread": mean_local,
        "mean_local_window_sigma": mean_local / noise_floor,
        "stopping_is_materially_seed_sensitive": material,
        "decision": "do_not_promote_rolling_origin",
        "reason": (
            "The captures show stopping sensitivity, but contain validation "
            "predictions only. Choosing a rule from them now would tune on the "
            "official validation split, and no matching per-epoch test artifact "
            "exists. Keep the verified fixed ensemble unchanged. A future "
            "rolling-origin experiment must be preregistered and selected "
            "entirely inside train dates before official validation is read."
        ),
    }


def write(input_path: str = DEFAULT_INPUT, out: str = DEFAULT_OUT) -> dict:
    with open(input_path) as fh:
        curves = json.load(fh)
    result = analyse(curves)
    result["source"] = os.path.relpath(input_path, ROOT)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(result, fh, indent=2)
    os.replace(tmp, out)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    result = write(a.input, a.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
