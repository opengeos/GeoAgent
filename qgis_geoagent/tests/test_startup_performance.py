"""Startup-path tests for the OpenGeoAgent QGIS plugin."""

from __future__ import annotations

import importlib
import sys
import types


def test_plugin_package_import_does_not_import_dependency_manager() -> None:
    """Importing the plugin package should not run dependency checks."""
    for module_name in list(sys.modules):
        if module_name == "open_geoagent" or module_name.startswith("open_geoagent."):
            sys.modules.pop(module_name, None)

    importlib.import_module("open_geoagent")

    assert "open_geoagent.deps_manager" not in sys.modules


def test_all_dependencies_met_uses_lightweight_spec_checks(monkeypatch) -> None:
    """The chat-open dependency gate must not import provider packages."""
    from open_geoagent import deps_manager

    monkeypatch.setattr(
        deps_manager,
        "REQUIRED_PACKAGES",
        [("geoagent", "GeoAgent[providers]>=1.2.0"), ("openai", "openai>=1.0")],
    )
    monkeypatch.setattr(deps_manager, "ensure_venv_packages_available", lambda: True)

    checked: list[str] = []

    def fake_find_spec(import_name: str):
        checked.append(import_name)
        return types.SimpleNamespace(name=import_name)

    def fail_import_module(import_name: str):
        raise AssertionError(f"unexpected import of {import_name}")

    monkeypatch.setattr(deps_manager.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(deps_manager.importlib, "import_module", fail_import_module)

    assert deps_manager.all_dependencies_met() is True
    assert checked == ["geoagent", "openai"]
