"""LLM client for the research agent — provider-agnostic.

Transport is pluggable (OpenAI by default, Anthropic as a second option); the
response contract and the bounded-repair-on-violation behavior are identical for
every provider, because the agent loop downstream depends on them:

    structured_call(prompt, validate_choices)
        -> ({hypothesis, menu_choices, code, expected_effect}, usage, events)

Resolution order for provider/model: environment variable (from .env) first, then
config/llm_config.json. API keys are read ONLY from the environment — never from
the config file, which is committed.

Token usage is read from the provider's own API response for both backends, so the
Feasibility & Practicality numbers are measured rather than estimated.
"""
from __future__ import annotations

import json
import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
LLM_CONFIG_PATH = os.path.join(_ROOT, "config", "llm_config.json")
ENV_PATH = os.path.join(_ROOT, ".env")

PROVIDERS = ("openai", "anthropic")

# The response contract every downstream consumer depends on.
# Keys required of EVERY response, regardless of implementation path.
RESPONSE_SCHEMA = {
    "hypothesis": str,       # what to try and why (judged text — be specific)
    "code": str,             # FULL solution script (not a diff)
    "expected_effect": str,  # predicted effect on valid GAUC/nDCG@5
    "rationale": dict,       # {idea, why_expected_to_help, grounded_in} — problem insight
    "implementation_path": str,   # "A" (menu/template) or "B" (custom mechanism)
    "research_category": str,     # exploration|exploitation|ablation|confirmation|integration
}

# Path A additionally requires menu_choices; Path B requires code_summary instead.
# menu_choices was previously mandatory for EVERY response, which forced the
# model to commit to a menu selection before it could even consider custom
# code -- a measured cause of Path B being chosen ~0 times in 54 nodes.
PATH_A_EXTRA = {"menu_choices": dict}
PATH_B_EXTRA = {"code_summary": str}
RESEARCH_CATEGORIES = ("exploration", "exploitation", "ablation",
                       "confirmation", "integration")

# grounded_in must name something concrete; reject generic non-answers for free
# (no retry cost for the model, no verification of TRUTH — just laziness).
_GENERIC_GROUNDING_PHRASES = (
    "general ml intuition", "general intuition", "common sense", "common ml practice",
    "it seemed reasonable", "standard practice", "general knowledge", "just a guess",
)

SYSTEM_PROMPT = (
    "You are the research brain of an autonomous ML research agent. "
    "Respond with exactly ONE JSON object and nothing else — no prose before or "
    "after it, no markdown fences. Always required: hypothesis (string), code "
    "(string containing the COMPLETE runnable python script), expected_effect "
    "(string), rationale (object with idea, why_expected_to_help, grounded_in), "
    "implementation_path ('A' or 'B'), research_category (one of exploration, "
    "exploitation, ablation, confirmation, integration). "
    "If implementation_path is 'A' you must also give menu_choices (object). "
    "If it is 'B' you must instead give code_summary (string) describing the "
    "mechanism you implemented and why existing primitives cannot express it. "
    "grounded_in must name something SPECIFIC — a measurement, a recorded "
    "result, a menu option description, or a named paper/method — never a "
    "generic appeal to 'ML intuition'."
)


class LLMError(RuntimeError):
    """Any LLM-stage failure. AgentLoop catches this and journals an error node."""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
def load_dotenv(path: str = ENV_PATH) -> None:
    """Load .env into os.environ without overwriting anything already set."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_llm_config(path: str = LLM_CONFIG_PATH) -> dict:
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def resolve_provider(cfg: dict | None = None) -> str:
    cfg = load_llm_config() if cfg is None else cfg
    provider = (os.environ.get("PROVIDER")
                or cfg.get("provider") or "openai").strip().lower()
    if provider not in PROVIDERS:
        raise LLMError(f"unknown PROVIDER '{provider}' — choose one of {PROVIDERS}")
    return provider


def key_env_var(provider: str) -> str:
    return "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"


def resolve_model(provider: str, cfg: dict | None = None, test: bool = False) -> str:
    """Env wins over config. MODEL_NAME is the provider-neutral override."""
    cfg = load_llm_config() if cfg is None else cfg
    section = cfg.get(provider, {})
    if test:
        return (os.environ.get("TEST_MODEL")
                or section.get("test_model")
                or section.get("model") or "")
    if provider == "openai":
        model = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL")
    else:
        model = os.environ.get("MODEL_NAME") or os.environ.get("ANTHROPIC_MODEL")
    return model or section.get("model") or ""


PLACEHOLDER_MARKERS = ("...", "your-key", "yourkey", "changeme", "xxx", "<", "replace")


def looks_like_placeholder(key: str) -> bool:
    """True for an un-edited .env.example value.

    Without this, a teammate who copies the template and forgets to paste the key
    sails past preflight and only finds out after burning the whole iteration cap
    on 401s.
    """
    k = (key or "").strip()
    if len(k) < 20:
        return True
    low = k.lower()
    return any(m in low for m in PLACEHOLDER_MARKERS)


def preflight(test: bool = False, verify_key: bool = False) -> dict:
    """Validate provider/model/key BEFORE any expensive work. Raises LLMError.

    verify_key=True additionally makes one free metadata call (models.list) to
    prove the credential actually authenticates — no tokens, no spend.

    Returns a dict describing the resolved configuration.
    """
    load_dotenv()
    cfg = load_llm_config()
    provider = resolve_provider(cfg)
    model = resolve_model(provider, cfg, test=test)
    var = key_env_var(provider)
    key = os.environ.get(var)

    if not key:
        raise LLMError(
            f"no API key for provider '{provider}': environment variable {var} is "
            f"not set.\nCopy .env.example to .env and fill it in "
            f"(cp .env.example .env), or switch provider with PROVIDER=<name>.")
    if looks_like_placeholder(key):
        raise LLMError(
            f"{var} still looks like the .env.example placeholder "
            f"({key[:12]!r}...).\nPaste the real key into .env — ask Khant for the "
            f"shared team key. Nothing has been spent.")
    if not model:
        raise LLMError(
            f"no model configured for provider '{provider}'. Set "
            f"{'OPENAI_MODEL' if provider == 'openai' else 'MODEL_NAME'} in .env "
            f"or add one to config/llm_config.json.")
    try:
        if provider == "openai":
            import openai  # noqa: F401
        else:
            import anthropic  # noqa: F401
    except ModuleNotFoundError as e:
        pkg = "openai" if provider == "openai" else "anthropic"
        raise LLMError(f"provider '{provider}' needs the `{pkg}` package: "
                       f"python3 -m pip install {pkg}") from e

    verified = False
    if verify_key:
        # free metadata call: proves the credential authenticates, spends nothing
        try:
            if provider == "openai":
                from openai import OpenAI
                OpenAI(api_key=key,
                       base_url=os.environ.get("OPENAI_BASE_URL") or None,
                       timeout=30.0).models.list()
            else:
                import anthropic
                anthropic.Anthropic(api_key=key, timeout=30.0).models.list()
            verified = True
        except Exception as e:
            raise LLMError(
                f"{var} was rejected by {provider}: {type(e).__name__}: "
                f"{str(e)[:200]}\nCheck the key in .env. Nothing has been spent."
            ) from e

    return {"provider": provider, "model": model, "key_var": var,
            "key_present": True, "key_verified": verified, "test_mode": test}


# --------------------------------------------------------------------------
# transports — each returns (reply_text, usage_dict) in harness-normalized keys
# --------------------------------------------------------------------------
class _OpenAITransport:
    def __init__(self, model: str, timeout_s: int, max_output_tokens: int):
        from openai import OpenAI
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],
                             base_url=os.environ.get("OPENAI_BASE_URL") or None,
                             timeout=float(timeout_s))

    def call(self, messages: list) -> tuple[str, dict]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=self.max_output_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise LLMError(f"OpenAI call failed: {type(e).__name__}: {e}") from e
        u = getattr(resp, "usage", None)
        cached = 0
        details = getattr(u, "prompt_tokens_details", None)
        if details is not None:
            cached = int(getattr(details, "cached_tokens", 0) or 0)
        total_in = int(getattr(u, "prompt_tokens", 0) or 0)
        usage = {"input_tokens": max(0, total_in - cached),
                 "output_tokens": int(getattr(u, "completion_tokens", 0) or 0),
                 "cache_creation_input_tokens": 0,
                 "cache_read_input_tokens": cached}
        return (resp.choices[0].message.content or ""), usage


class _AnthropicTransport:
    def __init__(self, model: str, timeout_s: int, max_output_tokens: int):
        import anthropic
        self.anthropic = anthropic
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"],
                                          timeout=float(timeout_s))

    def call(self, messages: list) -> tuple[str, dict]:
        system = SYSTEM_PROMPT
        turns = [m for m in messages if m["role"] != "system"]
        try:
            with self.client.messages.stream(
                    model=self.model, max_tokens=self.max_output_tokens,
                    system=system, messages=turns) as stream:
                resp = stream.get_final_message()
        except Exception as e:
            raise LLMError(f"Anthropic call failed: {type(e).__name__}: {e}") from e
        u = getattr(resp, "usage", None)
        usage = {k: int(getattr(u, k, 0) or 0) for k in
                 ("input_tokens", "output_tokens",
                  "cache_creation_input_tokens", "cache_read_input_tokens")}
        text = "".join(b.text for b in resp.content
                       if getattr(b, "type", None) == "text")
        return text, usage


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------
class LLMClient:
    def __init__(self, model: str | None = None, timeout_s: int = 1200,
                 max_repair_retries: int = 2, provider: str | None = None,
                 test: bool = False):
        load_dotenv()
        cfg = load_llm_config()
        self.provider = (provider or resolve_provider(cfg)).lower()
        self.model = model or resolve_model(self.provider, cfg, test=test)
        self.timeout_s = timeout_s
        self.max_repair_retries = max_repair_retries
        self.max_output_tokens = int(cfg.get("max_output_tokens", 32000))

        var = key_env_var(self.provider)
        if not os.environ.get(var):
            raise LLMError(f"{var} is not set — see .env.example")

        if self.provider == "openai":
            self.transport = _OpenAITransport(self.model, timeout_s,
                                              self.max_output_tokens)
        else:
            self.transport = _AnthropicTransport(self.model, timeout_s,
                                                 self.max_output_tokens)

        self.total_usage = {"input_tokens": 0, "output_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0, "calls": 0}

    # ---------- raw call ----------
    def _call_raw(self, messages: list) -> tuple[str, dict]:
        text, usage = self.transport.call(messages)
        for k, v in usage.items():
            self.total_usage[k] += v
        self.total_usage["calls"] += 1
        return text, usage

    # ---------- response parsing (identical across providers) ----------
    @staticmethod
    def _extract_json(text: str) -> dict:
        """Pull the JSON object out of the reply (tolerates code fences/preamble)."""
        m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        candidate = m.group(1) if m else text
        start = candidate.find("{")
        if start < 0:
            raise LLMError("no JSON object found in LLM reply")
        depth, in_str, esc = 0, False, False
        for i, ch in enumerate(candidate[start:], start):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(candidate[start:i + 1])
        raise LLMError("unbalanced JSON object in LLM reply")

    @staticmethod
    def _schema_problems(obj: dict) -> list[str]:
        probs = []
        required = dict(RESPONSE_SCHEMA)
        path = str(obj.get("implementation_path", "")).strip().upper()
        # Path-conditional requirements. Critically, Path B does NOT require
        # menu_choices -- the model must be able to choose custom code without
        # first committing to a menu selection.
        if path == "B":
            required.update(PATH_B_EXTRA)
        else:
            required.update(PATH_A_EXTRA)
        for key, typ in required.items():
            if key not in obj:
                probs.append(f"missing required key '{key}'")
            elif not isinstance(obj[key], typ):
                probs.append(f"'{key}' must be {typ.__name__}, "
                             f"got {type(obj[key]).__name__}")
        if path not in ("A", "B"):
            probs.append("implementation_path must be exactly 'A' or 'B'")
        cat = str(obj.get("research_category", "")).strip().lower()
        if cat not in RESEARCH_CATEGORIES:
            probs.append(f"research_category must be one of {list(RESEARCH_CATEGORIES)}")
        if path == "B":
            cs = str(obj.get("code_summary", "")).strip()
            if len(cs) < 40:
                probs.append("Path B requires a code_summary (>=40 chars) naming the "
                             "mechanism implemented and why existing primitives "
                             "cannot express it")
        if isinstance(obj.get("code"), str) and len(obj["code"].strip()) < 50:
            probs.append("'code' must be a complete runnable script, not a stub")
        if isinstance(obj.get("rationale"), dict):
            r = obj["rationale"]
            for sub in ("idea", "why_expected_to_help", "grounded_in"):
                v = r.get(sub)
                if not isinstance(v, str) or len(v.strip()) < 15:
                    probs.append(f"rationale.{sub} must be a specific, non-trivial "
                                 f"string (>=15 chars)")
            grounded = str(r.get("grounded_in", "")).strip().lower()
            if grounded and any(p in grounded for p in _GENERIC_GROUNDING_PHRASES):
                probs.append("rationale.grounded_in is a generic non-answer -- name a "
                             "SPECIFIC menu axis/option description, a "
                             "baseline_scores.json number, or a named paper/method")
        return probs

    # ---------- structured call (contract unchanged) ----------
    def structured_call(self, prompt: str, validate_choices=None) -> tuple[dict, dict, list]:
        """Returns (response_obj, usage_totals_for_this_node, events).

        validate_choices: optional callable(dict) -> normalized dict, raising with a
        readable message (the menu's validity check runs INSIDE the retry loop, so
        an invalid combination is bounced back to the LLM before anything executes).
        """
        events = []
        node_usage = {"input_tokens": 0, "output_tokens": 0,
                      "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
        base = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}]
        messages = list(base)
        last_err = None

        for attempt in range(1 + self.max_repair_retries):
            text, rec = self._call_raw(messages)
            for k in node_usage:
                node_usage[k] += rec[k]
            try:
                obj = self._extract_json(text)
                probs = self._schema_problems(obj)
                if probs:
                    raise LLMError("response schema violations: " + "; ".join(probs))
                # Menu validation applies ONLY to Path A. For Path B the menu
                # is not the contract -- running this unconditionally is the
                # downstream coercion that made custom code effectively
                # unreachable even after the schema allowed it.
                if validate_choices is not None and \
                        str(obj.get("implementation_path", "A")).upper() != "B":
                    obj["menu_choices"] = validate_choices(obj["menu_choices"])
                obj.setdefault("menu_choices", {})
                return obj, node_usage, events
            except (LLMError, ValueError, json.JSONDecodeError) as e:
                last_err = str(e)
                events.append({"type": "llm_response_rejected",
                               "attempt": attempt, "reason": last_err[:500]})
                messages = base + [
                    {"role": "user", "content":
                        "YOUR PREVIOUS RESPONSE WAS REJECTED. Reason:\n" + last_err
                        + "\nRespond again with ONE valid JSON object exactly matching "
                          "the required schema {hypothesis, menu_choices, code, "
                          "expected_effect}."}]
        raise LLMError(f"LLM failed schema/menu validation after "
                       f"{1 + self.max_repair_retries} attempts: {last_err}")

    def json_call(self, prompt: str) -> tuple[dict, dict]:
        """A single JSON call WITHOUT the solution-script schema.

        structured_call enforces RESPONSE_SCHEMA (hypothesis/menu_choices/code/
        expected_effect/rationale). Any other JSON shape -- e.g. the data-
        inspection phase's {"requests": [...]} -- fails that schema, exhausts
        the repair retries and raises. That is exactly what happened on the
        first run with --data-tools: the inspect phase silently failed every
        round and the agent never saw a single measurement.
        """
        text, usage = self._call_raw(
            [{"role": "system", "content":
              "Respond with exactly ONE JSON object and nothing else."},
             {"role": "user", "content": prompt}])
        try:
            return self._extract_json(text), usage
        except LLMError:
            return {}, usage

    def tokens_for_report(self) -> dict:
        u = self.total_usage
        return {**u, "provider": self.provider, "model": self.model,
                "input_plus_output": u["input_tokens"]
                + u["cache_creation_input_tokens"]
                + u["cache_read_input_tokens"] + u["output_tokens"]}
