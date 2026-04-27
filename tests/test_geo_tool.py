"""Tests for the @geo_tool decorator."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from geoagent.core.decorators import (
    GEO_META_KEY,
    geo_tool,
    get_category,
    get_geo_meta,
    get_required_packages,
    needs_confirmation,
)


def test_geo_tool_produces_basetool() -> None:
    @geo_tool(category="data")
    def my_tool(x: int) -> int:
        """Square a number."""
        return x * x

    assert isinstance(my_tool, BaseTool)
    assert my_tool.name == "my_tool"
    assert my_tool.description.strip().startswith("Square a number.")


def test_geo_tool_metadata_stamped() -> None:
    @geo_tool(
        category="map",
        requires_confirmation=True,
        requires_packages=("leafmap",),
        context_keys=("map_obj",),
    )
    def remove_layer(name: str) -> str:
        """Remove a layer."""
        return f"Removed {name}"

    meta = get_geo_meta(remove_layer)
    assert meta["category"] == "map"
    assert meta["requires_confirmation"] is True
    assert meta["requires_packages"] == ["leafmap"]
    assert meta["context_keys"] == ["map_obj"]


def test_needs_confirmation_helper() -> None:
    @geo_tool(category="data")
    def safe_tool() -> str:
        """A safe tool."""
        return "ok"

    @geo_tool(category="io", requires_confirmation=True)
    def dangerous_tool() -> str:
        """A dangerous tool."""
        return "danger"

    assert needs_confirmation(safe_tool) is False
    assert needs_confirmation(dangerous_tool) is True


def test_get_category_and_packages() -> None:
    @geo_tool(category="qgis", requires_packages=("qgis", "PyQt5"))
    def my_qgis_tool() -> str:
        """A QGIS tool."""
        return "qgis"

    assert get_category(my_qgis_tool) == "qgis"
    assert get_required_packages(my_qgis_tool) == ["qgis", "PyQt5"]


def test_geo_tool_with_explicit_name_and_description() -> None:
    @geo_tool(
        category="data",
        name="renamed_tool",
        description="Override description.",
    )
    def original_name() -> str:
        """Original docstring."""
        return "x"

    assert original_name.name == "renamed_tool"
    assert original_name.description == "Override description."


def test_metadata_uses_namespaced_key() -> None:
    @geo_tool(category="data")
    def my_tool() -> str:
        """Doc."""
        return "x"

    assert GEO_META_KEY in my_tool.metadata
    assert "category" in my_tool.metadata[GEO_META_KEY]


def test_get_geo_meta_returns_empty_for_plain_tools() -> None:
    from langchain_core.tools import tool

    @tool
    def plain(x: int) -> int:
        """Plain LangChain tool."""
        return x

    assert get_geo_meta(plain) == {}
    assert needs_confirmation(plain) is False
