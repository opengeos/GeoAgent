"""Tests for settings diagnostics and installer selection helpers."""

from __future__ import annotations

import types

from open_geoagent.dialogs.settings_dock import SETTINGS_PREFIX, collect_diagnostics


class _FakeSettings:
    """Small QSettings stand-in for diagnostics tests."""

    def __init__(self, values):
        self.values = dict(values)

    def value(self, key, default="", type=str):  # noqa: A002
        value = self.values.get(key, default)
        if type is bool:
            return bool(value)
        if type is int:
            return int(value)
        return value


def test_collect_diagnostics_redacts_credentials(monkeypatch, tmp_path) -> None:
    """Diagnostics expose credential presence only, never secret values."""
    from open_geoagent import deps_manager, uv_manager

    monkeypatch.setattr(deps_manager, "check_dependencies", lambda: [])
    monkeypatch.setattr(deps_manager, "venv_exists", lambda: True)
    monkeypatch.setattr(deps_manager, "get_venv_dir", lambda: "/tmp/venv")
    monkeypatch.setattr(
        deps_manager, "get_venv_site_packages", lambda: "/tmp/venv/site-packages"
    )
    monkeypatch.setattr(uv_manager, "get_uv_path", lambda: "/tmp/uv")
    monkeypatch.setattr(uv_manager, "verify_uv", lambda: (True, "uv ok"))

    settings = _FakeSettings(
        {
            f"{SETTINGS_PREFIX}provider": "openai",
            f"{SETTINGS_PREFIX}model": "gpt-test",
            f"{SETTINGS_PREFIX}openai_api_key": "sk-secret",
        }
    )
    (tmp_path / "metadata.txt").write_text("version=1.2.3\n", encoding="utf-8")

    diagnostics = collect_diagnostics(settings, str(tmp_path))
    text = str(diagnostics)

    assert diagnostics["credential_presence"]["openai_api_key"]["saved"] is True
    assert diagnostics["model"]["provider"] == "openai"
    assert "sk-secret" not in text


def test_uv_usable_requires_successful_verification(monkeypatch) -> None:
    """A stale uv file should not be treated as usable."""
    from open_geoagent import deps_manager

    monkeypatch.setattr(
        deps_manager,
        "uv_manager",
        types.SimpleNamespace(uv_exists=lambda: True, verify_uv=lambda: (False, "bad")),
        raising=False,
    )

    # Patch the relative import target through sys.modules by monkeypatching the
    # imported module functions directly.
    import open_geoagent.uv_manager as uv_manager

    monkeypatch.setattr(uv_manager, "uv_exists", lambda: True)
    monkeypatch.setattr(uv_manager, "verify_uv", lambda: (False, "bad"))

    assert deps_manager._uv_usable() is False
