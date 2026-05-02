"""Tests for GeoAgent Solara UI helpers."""

from __future__ import annotations

import importlib
import sys
import types

from geoagent.core.safety import ConfirmRequest
from geoagent.ui import app


def test_default_model_for_provider() -> None:
    """Verify provider default model ids."""
    assert app.default_model_for_provider("openai-codex") == "gpt-5.5"
    assert app.default_model_for_provider("anthropic") == "claude-sonnet-4-6"
    assert app.default_model_for_provider("unknown") == ""


def test_confirmation_callback_denies_by_default() -> None:
    """Verify confirmation-required tools are denied unless auto-approve is on."""
    request = ConfirmRequest(tool_name="remove_layer", args={"name": "A"})
    assert app.confirmation_callback(False)(request) is False
    assert app.confirmation_callback(True)(request) is True
    assert app.confirmation_preview(False) is False
    assert app.confirmation_preview(True) is True


def test_create_ui_map_binding_prefers_anymap(monkeypatch) -> None:
    """Verify the UI prefers anymap when it is importable."""
    calls: list[tuple[object, dict]] = []

    class FakeMap:
        pass

    def fake_for_anymap(map_obj, **kwargs):
        calls.append((map_obj, kwargs))
        return "agent"

    fake_anymap = types.SimpleNamespace(Map=FakeMap)
    monkeypatch.setitem(sys.modules, "anymap", fake_anymap)
    monkeypatch.delitem(sys.modules, "leafmap", raising=False)
    monkeypatch.setattr("geoagent.for_anymap", fake_for_anymap)

    binding = app.create_ui_map_binding()
    assert binding.map_library == "anymap"
    assert isinstance(binding.map_obj, FakeMap)

    agent = app.create_bound_agent(
        binding,
        provider="openai",
        model_id="gpt-test",
        fast=True,
        auto_approve=True,
    )
    assert agent == "agent"
    assert calls
    assert calls[0][1]["config"].provider == "openai"
    assert calls[0][1]["config"].model == "gpt-test"
    assert calls[0][1]["fast"] is True
    assert calls[0][1]["confirm"](ConfirmRequest(tool_name="x", args={})) is True


def test_create_ui_map_binding_falls_back_to_leafmap(monkeypatch) -> None:
    """Verify leafmap is used when anymap is unavailable."""

    class FakeMap:
        pass

    def fake_for_leafmap(map_obj, **kwargs):
        return "agent"

    fake_leafmap = types.SimpleNamespace(Map=FakeMap)
    monkeypatch.setitem(sys.modules, "anymap", None)
    monkeypatch.setitem(sys.modules, "leafmap", fake_leafmap)
    monkeypatch.setattr("geoagent.for_leafmap", fake_for_leafmap)

    binding = app.create_ui_map_binding()
    assert binding.map_library == "leafmap"
    assert isinstance(binding.map_obj, FakeMap)


def test_create_ui_map_binding_missing_dependencies(monkeypatch) -> None:
    """Verify missing map packages produce an actionable error."""
    monkeypatch.setitem(sys.modules, "anymap", None)
    monkeypatch.setitem(sys.modules, "leafmap", None)

    try:
        app.create_ui_map_binding()
    except RuntimeError as exc:
        message = str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected missing map packages to fail")

    assert "GeoAgent[anymap,ui]" in message
    assert "GeoAgent[leafmap,ui]" in message


def test_ui_package_imports_without_provider_credentials() -> None:
    """Verify the UI package import does not initialize model providers."""
    module = importlib.import_module("geoagent.ui")
    assert hasattr(module, "launch_ui")


def test_solara_pages_import_with_stubbed_solara(monkeypatch) -> None:
    """Verify Solara page modules import without provider credentials."""
    fake_solara = types.ModuleType("solara")
    fake_solara.component = lambda fn: fn
    monkeypatch.setitem(sys.modules, "solara", fake_solara)
    for name in (
        "geoagent.ui.workspace",
        "geoagent.ui.pages.00_home",
        "geoagent.ui.pages.01_chat",
    ):
        sys.modules.pop(name, None)

    home = importlib.import_module("geoagent.ui.pages.00_home")
    chat = importlib.import_module("geoagent.ui.pages.01_chat")

    assert callable(home.Page)
    assert callable(chat.Page)
