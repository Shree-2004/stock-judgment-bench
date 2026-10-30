"""The Gemini call records which build answered, offline.

The live ledger's model-continuity rule (README, analysis plan item 6) depends
on `modelVersion` surviving into the stored usage.
"""
from __future__ import annotations

from types import SimpleNamespace

from llm import client


class _Resp:
    status_code = 200

    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


def _patch(monkeypatch, body):
    monkeypatch.setattr(client, "settings", SimpleNamespace(google_api_key="k", llm_model="m", llm_temperature=0.0))
    monkeypatch.setattr(client, "_wait_min_interval", lambda s: None)
    monkeypatch.setattr(client.requests, "post", lambda *a, **kw: _Resp(body))


def test_usage_carries_the_model_version(monkeypatch):
    _patch(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "VERDICT: NEUTRAL"}]}}],
                         "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "totalTokenCount": 13},
                         "modelVersion": "gemini-3.1-flash-lite-001"})
    text, usage = client.call_gemini("sys", "user")
    assert text == "VERDICT: NEUTRAL"
    assert usage == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
                     "model_version": "gemini-3.1-flash-lite-001"}


def test_missing_model_version_is_recorded_as_unknown_not_guessed(monkeypatch):
    _patch(monkeypatch, {"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
    assert client.call_gemini("sys", "user")[1]["model_version"] is None
