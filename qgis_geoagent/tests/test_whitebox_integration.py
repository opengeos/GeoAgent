"""Tests for OpenGeoAgent Whitebox integration wiring."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any


def test_dependencies_include_whitebox() -> None:
    """Verify the plugin dependency installer checks Whitebox."""
    from open_geoagent.deps_manager import REQUIRED_PACKAGES

    assert ("whitebox", "whitebox>=2.3.6") in REQUIRED_PACKAGES


def test_chat_worker_uses_whitebox_factory(monkeypatch) -> None:
    """ChatWorker.run dispatches the chat through ``geoagent.for_whitebox``.

    Asserts behavior rather than source text: the factory is monkeypatched
    and ``ChatWorker.run`` is invoked synchronously so a refactor that
    silently swaps the factory will fail this test.
    """
    import geoagent
    from open_geoagent.dialogs.chat_dock import ChatWorker, SAMPLE_PROMPTS

    captured: dict[str, Any] = {}

    class _StubResponse:
        success = True
        answer_text = "ok"
        error_message = ""
        executed_tools: list = []
        tool_calls: list = []
        cancelled_tools: list = []
        execution_time = 0.0

    class _StubAgent:
        def chat(self, prompt: str) -> _StubResponse:
            captured["prompt"] = prompt
            return _StubResponse()

    def _stub_factory(iface, **kwargs):
        captured["iface"] = iface
        captured["kwargs"] = kwargs
        return _StubAgent()

    monkeypatch.setattr(geoagent, "for_whitebox", _stub_factory)

    # Avoid pulling QgsProject (the qgis stub returns None for instance()).
    monkeypatch.setitem(
        sys.modules,
        "qgis.core",
        types.SimpleNamespace(QgsProject=types.SimpleNamespace(instance=lambda: None)),
    )

    sentinel_iface = object()
    worker = ChatWorker(
        iface=sentinel_iface,
        prompt="hello",
        provider="anthropic",
        model_id="claude-x",
        fast=False,
        max_tokens=1024,
        auto_approve_tools=True,
    )

    emitted: dict[str, Any] = {}
    worker.finished.connect(lambda payload: emitted.setdefault("payload", payload))

    worker.run()

    assert captured.get("iface") is sentinel_iface
    assert captured["prompt"] == "hello"
    assert captured["kwargs"]["fast"] is False
    assert "confirm" in captured["kwargs"]
    assert captured["kwargs"]["confirm"](types.SimpleNamespace(args={})) is True
    assert emitted["payload"]["success"] is True
    assert any("WhiteboxTools" in prompt for prompt in SAMPLE_PROMPTS)


def test_chat_worker_streams_when_enabled(monkeypatch) -> None:
    """ChatWorker.run streams deltas through chunk_received when enabled."""
    import geoagent
    from open_geoagent.dialogs.chat_dock import ChatWorker

    captured: dict[str, Any] = {}

    class _StubAgent:
        async def stream_chat(self, prompt: str):
            captured["prompt"] = prompt
            yield {"data": "hel"}
            await asyncio.sleep(0)
            yield {"data": "lo"}

    def _stub_factory(iface, **kwargs):
        captured["iface"] = iface
        captured["kwargs"] = kwargs
        return _StubAgent()

    monkeypatch.setattr(geoagent, "for_whitebox", _stub_factory)
    monkeypatch.setitem(
        sys.modules,
        "qgis.core",
        types.SimpleNamespace(QgsProject=types.SimpleNamespace(instance=lambda: None)),
    )

    sentinel_iface = object()
    worker = ChatWorker(
        iface=sentinel_iface,
        prompt="stream please",
        provider="openai-codex",
        model_id="gpt-5.5",
        fast=True,
        max_tokens=1024,
        auto_approve_tools=True,
        stream=True,
    )

    chunks: list[str] = []
    emitted: dict[str, Any] = {}
    worker.chunk_received.connect(chunks.append)
    worker.finished.connect(lambda payload: emitted.setdefault("payload", payload))

    worker.run()

    assert captured.get("iface") is sentinel_iface
    assert captured["prompt"] == "stream please"
    assert captured["kwargs"]["fast"] is True
    assert chunks == ["hel", "lo"]
    assert emitted["payload"]["success"] is True
    assert emitted["payload"]["answer"] == "hello"
    assert emitted["payload"]["streamed"] is True
