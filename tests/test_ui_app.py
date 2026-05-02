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


def test_default_provider_uses_environment(monkeypatch) -> None:
    """Verify default_provider() reflects the available env credential."""
    for var in (
        "OPENAI_API_KEY",
        "OPENAI_CODEX_ACCESS_TOKEN",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "LITELLM_API_KEY",
        "LITELLM_MODEL",
        "LITELLM_BASE_URL",
        "OLLAMA_HOST",
        "USE_OLLAMA",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert app.default_provider() == "openai"

    monkeypatch.delenv("OPENAI_API_KEY")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test")
    assert app.default_provider() == "anthropic"


def test_compact_tool_call_truncates_structurally() -> None:
    """Verify oversized arg payloads are compacted before stringification."""
    huge_value = "x" * 100_000
    call = {
        "name": "run_pyqgis_script",
        "args": {"code": huge_value, "extra": list(range(50))},
        "result": "ok",
    }
    out = app.compact_tool_call(call, max_chars=1200)
    assert "run_pyqgis_script" in out
    assert "result=ok" in out
    assert "...[truncated]" in out
    assert "+40 more" in out
    assert len(out) <= 1200


def test_build_prompt_with_context_includes_history() -> None:
    """Verify recent user/assistant turns are folded into the next prompt."""
    history = [
        {"role": "user", "text": "Add a basemap of Tokyo."},
        {"role": "assistant", "text": "Added OpenStreetMap centred on Tokyo."},
    ]
    prompt = app.build_prompt_with_context(history, "Now zoom to Shibuya.")
    assert "Add a basemap of Tokyo." in prompt
    assert "Added OpenStreetMap centred on Tokyo." in prompt
    assert "Now zoom to Shibuya." in prompt
    assert "User:" in prompt and "Assistant:" in prompt


def test_build_prompt_with_context_returns_prompt_when_empty() -> None:
    """Verify the helper is a no-op when there is no usable history."""
    assert app.build_prompt_with_context([], "Hello") == "Hello"


def test_format_response_message_success_and_error() -> None:
    """Verify GeoAgentResponse-like inputs produce the expected message dicts."""

    class FakeResp:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    success = FakeResp(
        success=True,
        answer_text="Done.",
        executed_tools=["add_basemap"],
        cancelled_tools=[],
        error_message=None,
        tool_calls=[],
    )
    msg = app.format_response_message(success)
    assert msg["status"] == "ok"
    assert "Done." in msg["text"]
    assert "Executed tools: add_basemap" in msg["text"]

    failure = FakeResp(
        success=False,
        answer_text="",
        executed_tools=[],
        cancelled_tools=["remove_layer"],
        error_message="Confirmation denied",
        tool_calls=[],
    )
    err_msg = app.format_response_message(failure)
    assert err_msg["status"] == "error"
    assert "Cancelled tools: remove_layer" in err_msg["text"]
    assert "Confirmation denied" in err_msg["text"]


def test_dispatch_prompt_success_flow(monkeypatch) -> None:
    """Verify a successful send appends the assistant reply and tool calls."""

    class FakeResp:
        success = True
        answer_text = "Centred on Knoxville."
        executed_tools = ["set_center"]
        cancelled_tools = []
        error_message = None
        tool_calls = [{"name": "set_center", "args": {"lat": 35.96, "lon": -83.92}}]

    class FakeAgent:
        def __init__(self):
            self.calls: list[str] = []

        def chat(self, prompt: str):
            self.calls.append(prompt)
            return FakeResp()

    fake_agent = FakeAgent()

    def factory(binding, **kwargs):
        factory.kwargs = kwargs
        factory.binding = binding
        return fake_agent

    binding = app.UiMapBinding(map_obj=object(), map_library="anymap", factory=factory)
    history = [
        {"role": "user", "text": "Earlier turn"},
        {"role": "assistant", "text": "Earlier reply"},
    ]

    new_history, tool_calls = app.dispatch_prompt(
        "Centre on Knoxville",
        history=history,
        binding=binding,
        provider="openai",
        model_id="gpt-test",
        fast=False,
        auto_approve=True,
        create_agent=factory,
    )

    assert new_history[: len(history)] == history
    assert new_history[-2] == {"role": "user", "text": "Centre on Knoxville"}
    assistant_msg = new_history[-1]
    assert assistant_msg["status"] == "ok"
    assert "Centred on Knoxville." in assistant_msg["text"]
    assert "Executed tools: set_center" in assistant_msg["text"]
    assert tool_calls == [{"name": "set_center", "args": {"lat": 35.96, "lon": -83.92}}]
    assert factory.kwargs["provider"] == "openai"
    assert factory.kwargs["auto_approve"] is True
    sent_prompt = fake_agent.calls[0]
    assert "Earlier turn" in sent_prompt
    assert "Earlier reply" in sent_prompt
    assert "Centre on Knoxville" in sent_prompt


def test_dispatch_prompt_failure_records_error() -> None:
    """Verify a missing binding produces an error message in the transcript."""
    new_history, tool_calls = app.dispatch_prompt(
        "Hello",
        history=[],
        binding=None,
        binding_error="anymap is not installed",
        provider="openai",
        model_id="gpt-test",
        fast=False,
        auto_approve=False,
    )
    assert tool_calls == []
    assert new_history[0] == {"role": "user", "text": "Hello"}
    assert new_history[1]["status"] == "error"
    assert "anymap is not installed" in new_history[1]["text"]


def test_dispatch_prompt_handles_agent_exception() -> None:
    """Verify chat exceptions are surfaced as error messages, not raised."""

    class BoomAgent:
        def chat(self, prompt: str):
            raise RuntimeError("network down")

    def factory(binding, **kwargs):
        return BoomAgent()

    binding = app.UiMapBinding(map_obj=object(), map_library="anymap", factory=factory)

    new_history, tool_calls = app.dispatch_prompt(
        "Hello",
        history=[],
        binding=binding,
        provider="openai",
        model_id="gpt-test",
        fast=False,
        auto_approve=False,
        create_agent=factory,
    )
    assert tool_calls == []
    assert new_history[-1]["status"] == "error"
    assert "network down" in new_history[-1]["text"]
