"""End-to-end test: mapping subagent searches STAC and adds a COG layer.

Reproduces the ``docs/examples/live_mapping.ipynb`` "Add a Sentinel-2 RGB
COG layer over Knoxville TN for July 2024" flow with a fake LLM and a
:class:`MockLeafmap`. ``search_stac``'s network call is monkeypatched to
return a canned item, so the test runs offline.

Pins three contracts at once:

1. The mapping subagent now has access to ``stac_tools()`` (the regression
   the bug report flagged).
2. The mock map records the COG layer with the resolved asset URL — i.e.
   the layer actually gets added rather than the agent silently giving up.
3. ``executed_tools`` surfaces the subagent's inner tool calls
   (``search_stac`` and ``add_cog_layer``), not just the parent ``task``
   dispatcher.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent import GeoAgent, GeoAgentContext
from geoagent.testing import MockLeafmap
from tests._fakes import make_fake

_KNOXVILLE_BBOX = [-84.05, 35.85, -83.80, 36.05]
_DATETIME_RANGE = "2024-07-01/2024-07-31"
_FAKE_VISUAL_URL = "https://example.invalid/sentinel2/visual.tif"
_LAYER_NAME = "Sentinel-2 RGB Knoxville 2024-07-15"


def _fake_stac_items() -> list[dict[str, object]]:
    """Canned single-item STAC response with a ``visual`` COG asset.

    Returns:
        A list with one item dict shaped like the real ``search_stac``
        output: id, datetime, bbox, collection, cloud_cover, assets.
    """
    return [
        {
            "id": "S2A_TILE_2024-07-15",
            "datetime": "2024-07-15T16:30:00Z",
            "bbox": _KNOXVILLE_BBOX,
            "collection": "sentinel-2-l2a",
            "cloud_cover": 5.0,
            "assets": {"visual": _FAKE_VISUAL_URL},
        }
    ]


def test_mapping_subagent_searches_stac_then_adds_cog(monkeypatch) -> None:
    """The mapping subagent resolves a natural-language query into a layer.

    Drives the agent through:
        coordinator -> task(mapping) -> search_stac -> add_cog_layer

    and asserts the MockLeafmap recorded the COG and that
    ``executed_tools`` includes the inner subagent calls.
    """
    from geoagent.core.tools import stac as stac_mod

    monkeypatch.setattr(
        stac_mod.search_stac,
        "func",
        lambda *args, **kwargs: _fake_stac_items(),
    )

    fake_map = MockLeafmap(center=[-83.92, 35.96], zoom=10)
    agent = GeoAgent(
        llm=make_fake(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc-task-1",
                            "name": "task",
                            "args": {
                                "description": (
                                    "Add a Sentinel-2 RGB COG layer over "
                                    "Knoxville TN for July 2024."
                                ),
                                "subagent_type": "mapping",
                            },
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc-search-1",
                            "name": "search_stac",
                            "args": {
                                "query": "Sentinel-2 RGB Knoxville July 2024",
                                "catalog": "microsoft-pc",
                                "bbox": _KNOXVILLE_BBOX,
                                "datetime_range": _DATETIME_RANGE,
                                "collections": ["sentinel-2-l2a"],
                                "max_items": 5,
                                "max_cloud_cover": 20,
                            },
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "tc-cog-1",
                            "name": "add_cog_layer",
                            "args": {
                                "url": _FAKE_VISUAL_URL,
                                "name": _LAYER_NAME,
                                "colormap": "viridis",
                            },
                        }
                    ],
                ),
                AIMessage(content=f"Added {_LAYER_NAME!r} as a COG layer."),
                AIMessage(content="Done."),
            ]
        ),
        context=GeoAgentContext(map_obj=fake_map),
    )

    resp = agent.chat("Add a Sentinel-2 RGB COG layer over Knoxville TN for July 2024.")

    assert resp.success, resp.error_message

    cog_layers = [layer for layer in fake_map.layers if layer.get("type") == "cog"]
    assert len(cog_layers) == 1, fake_map.layers
    assert cog_layers[0]["url"] == _FAKE_VISUAL_URL
    assert cog_layers[0]["name"] == _LAYER_NAME

    assert "task" in resp.executed_tools
    assert "search_stac" in resp.executed_tools
    assert "add_cog_layer" in resp.executed_tools


def test_mapping_subagent_includes_stac_tools() -> None:
    """The mapping subagent's tool list must include ``search_stac``.

    Pins the structural fix at the spec level so a future refactor that
    drops ``stac_tools()`` from the mapping subagent breaks the build
    immediately, even before any LLM run.
    """
    from geoagent.agents.mapping import mapping_subagent

    spec = mapping_subagent(MockLeafmap())
    assert spec is not None
    tool_names = {getattr(t, "name", None) for t in spec["tools"]}
    assert "search_stac" in tool_names
    assert "add_cog_layer" in tool_names
