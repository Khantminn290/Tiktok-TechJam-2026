"""Spend accounting for the LLM budget ceiling.

Cost is computed from REAL token counts returned by the provider's API response —
never estimated from prompt length. Rates live in config/model_rates.json (data,
not logic) so they can be corrected without touching code.

Fail-safe by design: a model missing from the rate table is priced with the
deliberately-high `_unknown_model_fallback`, so an unpriced model over-estimates
spend and trips the ceiling early rather than quietly overspending.
"""
from __future__ import annotations

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RATES_PATH = os.path.join(_ROOT, "config", "model_rates.json")


class RateTable:
    def __init__(self, path: str = RATES_PATH):
        with open(path) as fh:
            self.raw = json.load(fh)
        self.fallback = self.raw["_unknown_model_fallback"]

    def lookup(self, provider: str, model: str) -> tuple[dict, bool]:
        """Returns (rate, is_known). Unknown models get the high fallback rate."""
        table = self.raw.get(provider, {})
        if model in table:
            return table[model], True
        # tolerate dated snapshots like gpt-5.4-2026-03-05 / claude-haiku-4-5-2025…
        for name, rate in table.items():
            if model.startswith(name + "-"):
                return rate, True
        return self.fallback, False

    def cached_multiplier(self, provider: str) -> float:
        return float(self.raw.get("cached_input_multiplier", {}).get(provider, 1.0))

    def cost_usd(self, provider: str, model: str, usage: dict) -> float:
        """usage keys: input_tokens, output_tokens, cache_read_input_tokens.

        `input_tokens` is treated as the uncached portion; cached reads are billed
        at the provider's cached multiplier.
        """
        rate, _ = self.lookup(provider, model)
        mult = self.cached_multiplier(provider)
        fresh_in = max(0, int(usage.get("input_tokens", 0)))
        cached_in = max(0, int(usage.get("cache_read_input_tokens", 0)))
        # Anthropic bills cache *writes* at a premium; approximate at the standard
        # input rate, which is the conservative direction for a budget guard.
        write_in = max(0, int(usage.get("cache_creation_input_tokens", 0)))
        out = max(0, int(usage.get("output_tokens", 0)))
        return (
            (fresh_in + write_in) / 1e6 * rate["input"]
            + cached_in / 1e6 * rate["input"] * mult
            + out / 1e6 * rate["output"]
        )

    def describe(self, provider: str, model: str) -> str:
        rate, known = self.lookup(provider, model)
        tag = "" if known else "  [UNPRICED — using fail-safe fallback rate]"
        return (f"${rate['input']:.2f}/1M in, ${rate['output']:.2f}/1M out{tag}")


class SpendTracker:
    """Running spend + the 'would the next call exceed the ceiling?' decision."""

    def __init__(self, provider: str, model: str, ceiling_usd: float,
                 rates: RateTable | None = None):
        self.provider = provider
        self.model = model
        self.ceiling_usd = float(ceiling_usd)
        self.rates = rates or RateTable()
        self.total_usd = 0.0
        self.per_call: list[float] = []

    def record(self, usage: dict) -> float:
        cost = self.rates.cost_usd(self.provider, self.model, usage)
        self.total_usd += cost
        self.per_call.append(cost)
        return cost

    # Cold-start estimate, used only before the first iteration has been billed.
    # Calibrated against measured runs (~3.2k input / ~0.5k output per iteration)
    # with roughly 3x headroom. The previous 30k/8k guess was ~10x reality, which
    # made any ceiling under about $0.06 refuse to start the run at all.
    COLD_START_INPUT_TOKENS = 10_000
    COLD_START_OUTPUT_TOKENS = 2_000

    def estimated_next_call_usd(self) -> float:
        """Worst observed iteration so far — conservative, so the guard stops
        before an expensive call rather than after it."""
        if self.per_call:
            return max(self.per_call)
        rate, _ = self.rates.lookup(self.provider, self.model)
        return ((self.COLD_START_INPUT_TOKENS / 1e6) * rate["input"]
                + (self.COLD_START_OUTPUT_TOKENS / 1e6) * rate["output"])

    def would_exceed(self) -> tuple[bool, str]:
        """Stop before the next call would breach the ceiling.

        The first iteration is always allowed: the guard is measurement-driven, and
        with nothing measured yet a pre-run guess is the only input available. One
        iteration cannot be a runaway, and refusing to start looks like a broken
        agent. Everything after that is gated on real observed cost, so the worst
        case is a single-iteration overshoot on a very small ceiling.
        """
        if not self.per_call:
            return False, ""
        nxt = self.estimated_next_call_usd()
        if self.total_usd + nxt > self.ceiling_usd:
            return True, (
                f"spend ceiling reached: ${self.total_usd:.4f} spent over "
                f"{len(self.per_call)} iteration(s), next estimated at ${nxt:.4f}, "
                f"ceiling ${self.ceiling_usd:.2f} (raise it with --max-spend-usd)")
        return False, ""

    def summary(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model,
            "total_usd": round(self.total_usd, 6),
            "ceiling_usd": self.ceiling_usd,
            "iterations_billed": len(self.per_call),
            "mean_usd_per_iteration": round(
                sum(self.per_call) / len(self.per_call), 6) if self.per_call else 0.0,
            "max_usd_per_iteration": round(max(self.per_call), 6) if self.per_call else 0.0,
            "rate_card": self.rates.describe(self.provider, self.model),
        }
