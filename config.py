"""Central config: env vars, model choice, thresholds.

Nothing here fails at import time if a key is missing — steps 1 and 2 run with
no keys at all, and later steps ask for what they need when they need it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else value


@dataclass(frozen=True)
class Settings:
    # --- disclosures (step 2) ---
    # BSE needs no key. GDELT was rejected on rate limiting; Finnhub does not
    # cover Indian equities on the free tier. See the README.
    disclosure_limit: int = int(_env("DISCLOSURE_LIMIT", "5") or 5)

    # --- llm (step 3) ---
    # Verified against this account on 2026-09-13: llama-3.3-70b-versatile is
    # NOT available here. The general-purpose models on offer are
    # openai/gpt-oss-120b, openai/gpt-oss-20b, qwen/qwen3.8-27b, qwen/qwen3.6-27b.
    llm_provider: str = _env("LLM_PROVIDER", "groq") or "groq"
    llm_model: str = _env("LLM_MODEL", "openai/gpt-oss-120b") or "openai/gpt-oss-120b"
    llm_temperature: float = float(_env("LLM_TEMPERATURE", "0") or 0)
    # "low" | "medium" | "high" | "" to disable. gpt-oss only.
    reasoning_effort: str = _env("REASONING_EFFORT", "low") or ""
    groq_api_key: str | None = _env("GROQ_API_KEY")
    google_api_key: str | None = _env("GOOGLE_API_KEY") or _env("GEMINI_API_KEY")
    openai_api_key: str | None = _env("OPENAI_API_KEY")
    anthropic_api_key: str | None = _env("ANTHROPIC_API_KEY")

    # --- langfuse (step 3) ---
    # SDK v4 is OpenTelemetry-based and reads LANGFUSE_* from the environment
    # itself; these are kept explicit so failures are legible rather than silent.
    # EU host verified 200 for these keys; the US host returns 401.
    langfuse_public_key: str | None = _env("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = _env("LANGFUSE_SECRET_KEY")
    langfuse_host: str = (
        _env("LANGFUSE_HOST") or _env("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"
    )

    # --- evals (step 4+) ---
    # A cited number counts as matching a source figure within this relative
    # tolerance — models legitimately round ("Rs 20.4T" for 20,438,050,139,042).
    number_match_tolerance: float = float(_env("NUMBER_MATCH_TOLERANCE", "0.01") or 0.01)

    @property
    def langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def llm_key(self) -> str | None:
        return {
            "groq": self.groq_api_key,
            "gemini": self.google_api_key,
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
        }.get(self.llm_provider)


settings = Settings()
