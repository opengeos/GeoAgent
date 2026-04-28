"""Tests for OpenGeoAgent Whitebox integration wiring."""

from __future__ import annotations

import inspect


def test_dependencies_include_whitebox() -> None:
    """Verify the plugin dependency installer checks Whitebox."""
    from open_geoagent.deps_manager import REQUIRED_PACKAGES

    assert ("whitebox", "whitebox>=2.3.6") in REQUIRED_PACKAGES


def test_chat_worker_uses_whitebox_factory() -> None:
    """Verify chat requests are routed through the Whitebox factory."""
    from open_geoagent.dialogs.chat_dock import ChatWorker, SAMPLE_PROMPTS

    run_source = inspect.getsource(ChatWorker.run)
    assert "for_whitebox" in run_source
    assert any("WhiteboxTools" in prompt for prompt in SAMPLE_PROMPTS)
