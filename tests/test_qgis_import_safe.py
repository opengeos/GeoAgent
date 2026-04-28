"""Verify ``geoagent.tools.qgis`` is import-safe when QGIS is missing.

The module-level imports of ``geoagent.tools.qgis`` must NOT pull in the
``qgis`` package; only tool *bodies* may do that lazily. This test confirms
that on a fresh import, no ``qgis`` module is loaded into ``sys.modules``.
"""

from __future__ import annotations

import importlib
import sys

import pytest


def _reimport(name: str):
    """Force a fresh import of ``name``."""
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def test_module_import_does_not_pull_qgis() -> None:
    """Verify that module import does not pull qgis."""
    if "qgis" in sys.modules:
        pytest.skip("qgis is already imported in this environment.")
    module = _reimport("geoagent.tools.qgis")
    assert hasattr(module, "qgis_tools")
    assert "qgis" not in sys.modules
    assert "qgis.core" not in sys.modules


def test_qgis_tools_with_none_iface_returns_empty_list() -> None:
    """Verify that qgis tools with none iface returns empty list."""
    from geoagent.tools.qgis import qgis_tools

    assert qgis_tools(None) == []
    assert qgis_tools(None, None) == []
