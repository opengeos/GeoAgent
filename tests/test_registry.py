"""Tests for the tool registry."""

from __future__ import annotations

import pytest

from geoagent.core import registry
from geoagent.core.decorators import geo_tool


@pytest.fixture(autouse=True)
def _clean_registry():
    registry.clear()
    registry._PKG_CACHE.clear()
    yield
    registry.clear()
    registry._PKG_CACHE.clear()


def _make_tool(name: str, **meta_kwargs):
    @geo_tool(category=meta_kwargs.pop("category", "data"), **meta_kwargs)
    def fn() -> str:
        """Tool docstring."""
        return name

    fn.name = name  # rename for clean comparisons
    return fn


def test_register_and_all_tools() -> None:
    tool = _make_tool("alpha")
    registry.register(tool)
    assert registry.all_tools() == [tool]


def test_register_many() -> None:
    a = _make_tool("a")
    b = _make_tool("b")
    registry.register_many([a, b])
    assert {t.name for t in registry.all_tools()} == {"a", "b"}


def test_unregister() -> None:
    tool = _make_tool("alpha")
    registry.register(tool)
    registry.unregister("alpha")
    assert registry.all_tools() == []


def test_get_tools_filters_by_category() -> None:
    map_tool = _make_tool("map_tool", category="map")
    data_tool = _make_tool("data_tool", category="data")
    registry.register_many([map_tool, data_tool])

    out = registry.get_tools(categories={"map"})
    assert {t.name for t in out} == {"map_tool"}


def test_get_tools_filters_by_package_availability() -> None:
    available = _make_tool("ok", requires_packages=("os",))
    unavailable = _make_tool(
        "bad", requires_packages=("definitely_not_a_real_pkg_xyz",)
    )
    registry.register_many([available, unavailable])

    out = registry.get_tools(available_only=True)
    assert {t.name for t in out} == {"ok"}


def test_get_tools_disable_availability_filter() -> None:
    available = _make_tool("ok", requires_packages=("os",))
    unavailable = _make_tool("bad", requires_packages=("nope_pkg_xyz",))
    registry.register_many([available, unavailable])

    out = registry.get_tools(available_only=False)
    assert {t.name for t in out} == {"ok", "bad"}
