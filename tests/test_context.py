"""Tests for the GeoAgentContext dataclass."""

from __future__ import annotations

from pathlib import Path

from geoagent.core.context import GeoAgentContext


def test_default_context_has_workdir_set_to_cwd() -> None:
    ctx = GeoAgentContext()
    assert ctx.map_obj is None
    assert ctx.qgis_iface is None
    assert ctx.qgis_project is None
    assert isinstance(ctx.workdir, Path)
    assert ctx.workdir.exists()
    assert ctx.user_preferences == {}


def test_with_overrides_returns_new_instance() -> None:
    ctx = GeoAgentContext()
    ctx2 = ctx.with_overrides(current_layer="NDVI")
    assert ctx is not ctx2
    assert ctx2.current_layer == "NDVI"
    assert ctx.current_layer is None


def test_user_preferences_dict_is_independent_per_instance() -> None:
    a = GeoAgentContext()
    b = GeoAgentContext()
    a.user_preferences["theme"] = "dark"
    assert b.user_preferences == {}
