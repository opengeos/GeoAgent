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
        "CORE_RUNTIME_PACKAGES",
        [("geoagent", "GeoAgent[providers]>=1.3.0"), ("strands", "strands-agents")],
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
    assert checked == ["geoagent", "strands"]


def test_required_dependencies_include_core_runtime_packages() -> None:
    """Dependency gate should catch partial GeoAgent installs."""
    from open_geoagent.deps_manager import REQUIRED_PACKAGES

    assert ("geoagent", "GeoAgent[providers]>=1.3.0") in REQUIRED_PACKAGES
    assert ("strands", "strands-agents>=1.37") in REQUIRED_PACKAGES
    assert ("pydantic", "pydantic>=2.0") in REQUIRED_PACKAGES


def test_dependency_groups_include_optional_workflow_packages() -> None:
    """Optional workflow stacks should be grouped instead of globally required."""
    from open_geoagent.deps_manager import DEPENDENCY_GROUPS, REQUIRED_PACKAGES

    assert ("whitebox", "whitebox>=2.3.6") not in REQUIRED_PACKAGES
    assert ("whitebox", "whitebox>=2.3.6") in DEPENDENCY_GROUPS["WhiteboxTools"]
    assert ("pystac_client", "pystac-client>=0.8") in DEPENDENCY_GROUPS["STAC"]
    assert ("gee_data_catalogs", "gee-data-catalogs") not in DEPENDENCY_GROUPS[
        "GEE Data Catalogs"
    ]
    assert ("ee", "earthengine-api>=1.0") in DEPENDENCY_GROUPS["GEE Data Catalogs"]


def test_python_runtime_error_mentions_required_version(monkeypatch) -> None:
    """The installer should fail clearly on unsupported QGIS Python versions."""
    from open_geoagent import deps_manager

    monkeypatch.setattr(deps_manager, "MIN_PYTHON_VERSION", (99, 0))

    assert deps_manager.python_runtime_supported() is False
    assert "Python 99.0 or newer" in deps_manager.python_runtime_error()
