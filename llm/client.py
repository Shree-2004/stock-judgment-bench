"""Generate one brief, traced end to end.

Langfuse v4 is OpenTelemetry-based: `start_as_current_observation` context
managers rather than v2's `trace()`/`generation()` objects. Most tutorials
online still show the v2 API and will not work here.

**Tracing is optional by construction.** If Langfuse is unconfigured or down,
briefs still generate and still get scored — the benchmark's result must not
depend on an observability vendor being reachable. What tracing adds is the
ability to ask *why* a score moved, which is worth a lot and is worth nothing
if its absence can silently halve your dataset.

One trace per brief, carrying:
  - input  : the exact snapshot figures + filings the model saw
  - output : the raw text and the parsed structure
  - scores : l1_grounding, l1_clean, and parse success
  - tags   : prompt version, model, ticker, cutoff — the dimensions step 6
             compares across.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import requests

from config import settings
from evals.number_check import check_numbers
from llm.brief import Brief, parse_brief
from llm.prompts import DEFAULT_VERSION, build_prompt

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")

# WHAT THE LIMITS ACTUALLY ARE — read from the error bodies, not inferred
# from response headers. Three failed pacing attempts came from trusting
# `x-ratelimit-*` headers that were never the binding constraint.
#
#   Groq free tier:   "tokens per day (TPD): Limit 200000" — a DAILY cap.
#                     At ~1,540 tokens/brief that is 130 briefs/day, i.e. 26
#                     days for this benchmark. Not viable.
#   Gemini free tier: "GenerateRequestsPerMinutePerProjectPerModel-FreeTier",
#                     quotaValue 5, retryDelay 15s — a per-MINUTE request cap
#                     with no daily limit surfaced. 5/min ≈ 11h for 3,400
#                     briefs, which is viable.
#
# So pacing is request-rate based for Gemini, and the 429 body's retryDelay is
# authoritative over any local arithmetic.
# Measured per model, by exhausting each and reading the error body:
#   gemini-2.5-flash       5 req/min, and a brutal 20 req/DAY. Unusable.
#   gemini-3-flash-preview 5 req/min.
#   gemini-3.1-flash-lite 15 req/min, and 71 consecutive calls found no daily
#                          cap at all. 55 real briefs: 1,027 tokens each,
#                          55/55 parsed. This is the one that works.
# Paced at 5s (12/min) to stay under the 15/min ceiling with margin.
# Historical note on the old 5 req/min pacing: Measured: the call
# itself takes ~12.9s, so the interval and the latency largely overlap and the
# real cost per brief is ~15s, not 15+13. Sitting exactly on 12s produced 429s
# whose 15s retryDelay cost far more than the margin saves, so pace at 15s and
# stop bouncing off the limit.
GEMINI_MIN_INTERVAL_S = 5.0

# Groq's per-minute view (kept for reference; the daily cap binds first):
# every model on this account reports 1000 req/min but only 8000 tok/min.
#
# Measured over 260 real briefs: mean 1,544 tokens (761 prompt + 783
# completion), p90 1,710, max 1,969. 8000/1544 = 5.18 briefs/min, i.e. 11.6s
# apart — so the earlier fixed 12s floor sat exactly ON the limit with no
# margin, and every brief above the mean tipped the bucket. That produced a
# precise 50/50 success/failure alternation and a 368-hour ETA.
#
# A fixed interval cannot work when cost varies. This tracks the tokens
# actually spent in a trailing 60s window and spends only up to SAFETY of the
# budget, which self-tunes to whatever the briefs really cost.
TOKENS_PER_MINUTE = 8000
SAFETY = 0.80
ESTIMATED_TOKENS = 1800          # p90 plus headroom, from the measurement above
WINDOW_S = 60.0

_rate_lock = threading.Lock()
_token_events: deque = deque()   # (monotonic_time, tokens_spent)
_remaining_tokens: float | None = None
_reset_after_s: float = 0.0
_last_header_time: float = 0.0


def _parse_reset(value: str | None) -> float:
    """Groq writes resets as '1m26.4s' / '4.62s' / '820ms'."""
    if not value:
        return 0.0
    text = str(value).strip()
    total, number = 0.0, ""
    try:
        i = 0
        while i < len(text):
            ch = text[i]
            if ch.isdigit() or ch == ".":
                number += ch
                i += 1
                continue
            if text[i:i + 2] == "ms":
                total += float(number or 0) / 1000.0
                number, i = "", i + 2
                continue
            unit = {"m": 60.0, "s": 1.0, "h": 3600.0}.get(ch)
            if unit is not None:
                total += float(number or 0) * unit
            number, i = "", i + 1
        if number:
            total += float(number)
    except ValueError:
        return 0.0
    return total


def _note_rate_headers(headers) -> None:
    global _remaining_tokens, _reset_after_s, _last_header_time
    with _rate_lock:
        try:
            raw = headers.get("x-ratelimit-remaining-tokens")
            if raw is not None:
                _remaining_tokens = float(raw)
        except (TypeError, ValueError):
            pass
        _reset_after_s = _parse_reset(headers.get("x-ratelimit-reset-tokens"))
        _last_header_time = time.monotonic()


_last_request_time: float = 0.0


def _wait_min_interval(min_interval_s: float) -> None:
    """Uniform spacing for request-rate limits (Gemini's binding constraint)."""
    global _last_request_time
    with _rate_lock:
        gap = time.monotonic() - _last_request_time
        wait = min_interval_s - gap if _last_request_time else 0.0
    if wait > 0:
        time.sleep(wait)
    with _rate_lock:
        _last_request_time = time.monotonic()


def record_usage(total_tokens: int) -> None:
    """Book real spend against the rolling window."""
    if not total_tokens:
        return
    with _rate_lock:
        _token_events.append((time.monotonic(), int(total_tokens)))


def _spent_in_window(now: float) -> float:
    while _token_events and now - _token_events[0][0] > WINDOW_S:
        _token_events.popleft()
    return sum(t for _, t in _token_events)


def _wait_for_token_budget(estimated_tokens: int = ESTIMATED_TOKENS) -> None:
    """Block until the trailing-minute budget can afford this call."""
    budget = TOKENS_PER_MINUTE * SAFETY
    while True:
        with _rate_lock:
            now = time.monotonic()
            spent = _spent_in_window(now)
            if spent + estimated_tokens <= budget:
                return
            oldest_at = _token_events[0][0] if _token_events else now
        # Sleep until the oldest spend ages out of the window.
        time.sleep(max(0.5, WINDOW_S - (now - oldest_at) + 0.5))


def _wait_for_header_backoff() -> None:
    """Second layer: honour Groq's own accounting if it says we are short."""
    with _rate_lock:
        remaining = _remaining_tokens
        reset_in = _reset_after_s
        seen_at = _last_header_time
    if remaining is None or remaining >= ESTIMATED_TOKENS:
        return
    elapsed = time.monotonic() - seen_at
    sleep_for = max(0.0, reset_in - elapsed) + 1.0
    if sleep_for > 0:
        time.sleep(min(sleep_for, 70.0))


class DailyQuotaExhausted(RuntimeError):
    """The provider's per-DAY quota is spent; nothing will work until reset.

    Distinct from a transient rate limit on purpose. Retrying through this
    burns hours and, worse, marks thousands of perfectly good cells as
    "failed" — which is a data-integrity problem, not just wasted time. The
    runner stops cleanly instead and resumes tomorrow.

    Measured free-tier daily caps:
      gemini-2.5-flash        20 / day
      gemini-3.1-flash-lite  500 / day   <- current model, 7.4 days for 3,690
      groq gpt-oss-120b      200,000 tokens/day (~176 briefs at effort=low)
    """


class LLMUnavailableError(RuntimeError):
    """The model was asked and would not answer.

    Distinct from "answered with something unparseable": a failed call is not
    evidence about the model's grounding, and scoring it as a bad brief would
    corrupt the dataset.
    """


@dataclass
class BriefResult:
    ticker: str
    cutoff: str
    prompt_version: str
    model: str
    brief: Brief
    l1: Any
    raw_text: str = ""
    usage: dict = field(default_factory=dict)
    latency_s: float = 0.0
    trace_id: str | None = None

    def as_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "cutoff": self.cutoff,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "brief": self.brief.as_dict(),
            "l1": self.l1.as_dict() if self.l1 else None,
            "usage": self.usage,
            "latency_s": round(self.latency_s, 2),
            "trace_id": self.trace_id,
        }


# ---------------------------------------------------------------------------
# langfuse
# ---------------------------------------------------------------------------

_langfuse = None
_langfuse_tried = False


def get_langfuse():
    """The client, or None. Never raises — tracing is best-effort."""
    global _langfuse, _langfuse_tried
    if _langfuse_tried:
        return _langfuse
    _langfuse_tried = True
    if not settings.langfuse_configured:
        return None
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        if not client.auth_check():
            return None
        _langfuse = client
    except Exception:
        _langfuse = None
    return _langfuse


def flush_traces() -> None:
    client = get_langfuse()
    if client:
        try:
            client.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# the model call
# ---------------------------------------------------------------------------

def call_groq(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float | None = None,
    max_attempts: int = 6,
    timeout: int = 90,
) -> tuple[str, dict]:
    """One chat completion. Returns (text, usage). Raises on failure."""
    key = settings.groq_api_key
    if not key:
        raise LLMUnavailableError("GROQ_API_KEY is not set.")

    payload = {
        "model": model or settings.llm_model,
        "temperature": settings.llm_temperature if temperature is None else temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    # gpt-oss are reasoning models and spend most of their output budget on
    # hidden reasoning tokens. Measured on a real brief: default 790 completion
    # tokens vs 359 at effort=low, total 1,564 -> 1,133, with the structured
    # contract still parsing. Under a tokens-per-DAY cap that 28% is 28% off
    # the calendar, so it is taken unless explicitly overridden.
    if settings.reasoning_effort and "gpt-oss" in (model or settings.llm_model):
        payload["reasoning_effort"] = settings.reasoning_effort

    last_error = "unknown"
    for attempt in range(max_attempts):
        _wait_for_token_budget()
        _wait_for_header_backoff()
        try:
            resp = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = f"network: {exc}"
            time.sleep(2 ** attempt)
            continue

        _note_rate_headers(resp.headers)

        if resp.status_code == 429:
            if "tokens per day" in resp.text.lower() or "TPD" in resp.text:
                raise DailyQuotaExhausted(f"groq TPD exhausted: {resp.text[:140]}")
            # Groq publishes a retry hint; honour it rather than guessing.
            try:
                wait = float(resp.headers.get("retry-after", 2 ** (attempt + 1)))
            except (TypeError, ValueError):
                wait = _parse_reset(resp.headers.get("x-ratelimit-reset-tokens")) or 2 ** (attempt + 1)
            record_usage(ESTIMATED_TOKENS)
            time.sleep(min(max(wait, 1.0), 70))
            last_error = "rate limited"
            continue

        if resp.status_code != 200:
            last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
            time.sleep(2 ** attempt)
            continue

        data = resp.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            last_error = f"unexpected response shape: {str(data)[:160]}"
            continue
        return text, data.get("usage", {}) or {}

    raise LLMUnavailableError(last_error)


def call_gemini(
    system: str,
    user: str,
    model: str | None = None,
    temperature: float | None = None,
    max_attempts: int = 6,
    timeout: int = 120,
) -> tuple[str, dict]:
    """One Gemini generateContent call. Returns (text, usage)."""
    key = settings.google_api_key
    if not key:
        raise LLMUnavailableError("GOOGLE_API_KEY is not set.")
    model = model or settings.llm_model

    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {
            "temperature": settings.llm_temperature if temperature is None else temperature,
            "maxOutputTokens": 2048,
        },
    }

    last_error = "unknown"
    for attempt in range(max_attempts):
        _wait_min_interval(GEMINI_MIN_INTERVAL_S)
        try:
            resp = requests.post(
                GEMINI_URL.format(model=model),
                params={"key": key},
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = f"network: {exc}"
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 429:
            if _is_daily_quota(resp):
                raise DailyQuotaExhausted(_quota_detail(resp))
            # The body carries an authoritative retryDelay; use it rather than
            # guessing. Guessing is what cost three runs.
            delay = _gemini_retry_delay(resp) or (5 * (attempt + 1))
            last_error = "rate limited"
            time.sleep(min(delay + 1.0, 70.0))
            continue

        if resp.status_code != 200:
            last_error = f"HTTP {resp.status_code}: {resp.text[:160]}"
            time.sleep(2 ** attempt)
            continue

        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            last_error = f"unexpected response shape: {str(data)[:200]}"
            continue

        u = data.get("usageMetadata", {}) or {}
        usage = {
            "prompt_tokens": u.get("promptTokenCount", 0),
            "completion_tokens": u.get("candidatesTokenCount", 0),
            "total_tokens": u.get("totalTokenCount", 0),
            # The build that actually answered. A model name can be repointed
            # by the provider; this is how the live ledger would notice.
            "model_version": data.get("modelVersion"),
        }
        return text, usage

    raise LLMUnavailableError(last_error)


def _is_daily_quota(resp) -> bool:
    """Does this 429 name a per-DAY quota? Read the body, not the headers."""
    try:
        for d in (resp.json().get("error", {}).get("details") or []):
            for v in (d.get("violations") or []):
                if "PerDay" in str(v.get("quotaId", "")):
                    return True
    except Exception:
        pass
    return False


def _quota_detail(resp) -> str:
    try:
        for d in (resp.json().get("error", {}).get("details") or []):
            for v in (d.get("violations") or []):
                if "PerDay" in str(v.get("quotaId", "")):
                    return f"{v.get('quotaId')} = {v.get('quotaValue')} requests/day"
    except Exception:
        pass
    return "daily quota exhausted"


def _gemini_retry_delay(resp) -> float | None:
    try:
        for d in (resp.json().get("error", {}).get("details") or []):
            if d.get("@type", "").endswith("RetryInfo"):
                raw = str(d.get("retryDelay", "")).rstrip("s")
                return float(raw)
    except Exception:
        pass
    return None


def call_llm(system: str, user: str, model: str | None = None) -> tuple[str, dict]:
    """Dispatch on the configured provider."""
    provider = settings.llm_provider
    if provider == "gemini":
        return call_gemini(system, user, model=model)
    if provider == "groq":
        return call_groq(system, user, model=model)
    raise LLMUnavailableError(f"Unknown LLM_PROVIDER {provider!r}")


# ---------------------------------------------------------------------------
# one brief, traced and scored
# ---------------------------------------------------------------------------

def generate_brief(
    snapshot,
    disclosures,
    prompt_version: str = DEFAULT_VERSION,
    model: str | None = None,
    trace: bool = True,
) -> BriefResult:
    """Build the prompt, call the model, parse, score L1, trace the lot."""
    prompt, user_message = build_prompt(snapshot, disclosures, prompt_version)
    model = model or settings.llm_model
    truth = snapshot.figures()

    client = get_langfuse() if trace else None
    span_cm = (
        client.start_as_current_observation(
            name="brief",
            as_type="span",
            input={
                "ticker": snapshot.ticker,
                "as_of": snapshot.as_of,
                "figures": truth,
                "disclosures": disclosures.as_prompt_block(),
            },
            metadata={
                "prompt_version": prompt.version,
                "model": model,
                "ticker": snapshot.ticker,
                "cutoff": snapshot.as_of,
                "disclosure_count": len(getattr(disclosures, "items", []) or []),
            },
        )
        if client
        else nullcontext()
    )

    started = time.monotonic()
    with span_cm as span:
        gen_cm = (
            client.start_as_current_observation(
                name="llm",
                as_type="generation",
                input=[{"role": "system", "content": prompt.system},
                       {"role": "user", "content": user_message}],
                model=model,
                model_parameters={"temperature": settings.llm_temperature},
            )
            if client
            else nullcontext()
        )

        with gen_cm as generation:
            text, usage = call_llm(prompt.system, user_message, model=model)
            if generation is not None and hasattr(generation, "update"):
                generation.update(
                    output=text,
                    usage_details={
                        "input": usage.get("prompt_tokens", 0),
                        "output": usage.get("completion_tokens", 0),
                        "total": usage.get("total_tokens", 0),
                    },
                )

        latency = time.monotonic() - started
        parsed = parse_brief(text)
        l1 = check_numbers(parsed.cited_figures, truth, settings.number_match_tolerance)

        trace_id = None
        if client:
            try:
                trace_id = client.get_current_trace_id()
                if span is not None and hasattr(span, "update"):
                    span.update(output=parsed.as_dict())
                # None means "cited nothing checkable" — record it as a
                # separate signal rather than coercing it to 0.0.
                if l1.score is not None:
                    client.create_score(name="l1_grounding", value=l1.score,
                                        data_type="NUMERIC", trace_id=trace_id)
                client.create_score(name="l1_clean", value=int(l1.clean),
                                    data_type="BOOLEAN", trace_id=trace_id)
                client.create_score(name="parsed", value=int(parsed.parsed),
                                    data_type="BOOLEAN", trace_id=trace_id)
                client.create_score(name="figures_cited", value=l1.cited,
                                    data_type="NUMERIC", trace_id=trace_id)
            except Exception:
                pass   # tracing must never break generation

    return BriefResult(
        ticker=snapshot.ticker,
        cutoff=snapshot.as_of,
        prompt_version=prompt.version,
        model=model,
        brief=parsed,
        l1=l1,
        raw_text=text,
        usage=usage,
        latency_s=latency,
        trace_id=trace_id,
    )
