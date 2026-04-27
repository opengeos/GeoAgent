"""End-to-end test: mapping subagent searches STAC and adds a layer.

Reproduces the ``docs/examples/live_mapping.ipynb`` "Add a Sentinel-2 RGB
layer over Knoxville TN for July 2024" flow with a fake LLM and a
:class:`MockLeafmap`. ``search_stac``'s network call is monkeypatched to
return a canned item, so the test runs offline.

Pins four contracts:

1. The mapping subagent has access to ``stac_tools()`` (the regression
   the bug report flagged).
2. For Planetary Computer items, the canonical render path is
   ``add_stac_layer(collection, item, assets, titiler_endpoint="pc")``
   so PC's hosted TiTiler signs SAS-protected asset URLs internally.
3. The mock map records the layer with the right collection / item /
   asset and the ``titiler_endpoint`` kwarg gets passed through.
4. ``executed_tools`` surfaces the subagent's inner tool calls
   (``search_stac`` and ``add_stac_layer``), not just the parent
   ``task`` dispatcher.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from geoagent import GeoAgent, GeoAgentContext
from geoagent.testing import MockLeafmap
from tests._fakes import make_fake

_KNOXVILLE_BBOX = [-84.05, 35.85, -83.80, 36.05]
_DATETIME_RANGE = "2024-07-01/2024-07-31"
_ITEM_ID = "S2A_TILE_2024-07-15"
_LAYER_NAME = "Sentinel-2 RGB Knoxville 2024-07-15"
_FAKE_VISUAL_HREF = "https://example.invalid/sentinel2/visual.tif"


def _fake_stac_items() -> list[dict[str, object]]:
    """Canned single-item STAC response with a ``visual`` asset.

    Returns:
        A list with one item dict shaped like the real ``search_stac``
        output: id, datetime, bbox, collection, cloud_cover, assets.
    """
    return [
        {
            "id": _ITEM_ID,
            "datetime": "2024-07-15T16:30:00Z",
            "bbox": _KNOXVILLE_BBOX,
            "collection": "sentinel-2-l2a",
            "cloud_cover": 5.0,
            "assets": {"visual": _FAKE_VISUAL_HREF},
        }
    ]


def test_mapping_subagent_pc_path_uses_stac_layer_with_pc_titiler(monkeypatch) -> None:
    """For Planetary Computer items, render via add_stac_layer + pc TiTiler.

    Drives the agent through:
        coordinator -> task(mapping) -> search_stac -> add_stac_layer

    where ``add_stac_layer`` is called with ``titiler_endpoint="pc"`` so
    Microsoft's hosted TiTiler signs SAS-protected hrefs internally.
    Asserts the MockLeafmap recorded the STAC layer with the right
    fields and that the inner subagent tools surface in
    ``executed_tools``.
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
                                    "Add a Sentinel-2 RGB layer over "
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
                            "id": "tc-stac-1",
                            "name": "add_stac_layer",
                            "args": {
                                "collection": "sentinel-2-l2a",
                                "item": _ITEM_ID,
                                "assets": ["visual"],
                                "name": _LAYER_NAME,
                                "titiler_endpoint": "pc",
                            },
                        }
                    ],
                ),
                AIMessage(content=f"Added {_LAYER_NAME!r}."),
                AIMessage(content="Done."),
            ]
        ),
        context=GeoAgentContext(map_obj=fake_map),
    )

    resp = agent.chat("Add a Sentinel-2 RGB layer over Knoxville TN for July 2024.")

    assert resp.success, resp.error_message

    stac_layers = [layer for layer in fake_map.layers if layer.get("type") == "stac"]
    assert len(stac_layers) == 1, fake_map.layers
    layer = stac_layers[0]
    assert layer["collection"] == "sentinel-2-l2a"
    assert layer["item"] == _ITEM_ID
    assert layer["assets"] == ["visual"]
    assert layer["name"] == _LAYER_NAME
    # The crux of this test: titiler_endpoint must reach leafmap so PC's
    # hosted TiTiler handles SAS signing — the public TiTiler default
    # would fail with KeyError 'tiles' on an unsigned PC href.
    assert layer["titiler_endpoint"] == "pc"

    assert "task" in resp.executed_tools
    assert "search_stac" in resp.executed_tools
    assert "add_stac_layer" in resp.executed_tools


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
    assert "add_stac_layer" in tool_names
    assert "add_cog_layer" in tool_names


def test_add_cog_layer_refuses_planetary_computer_url() -> None:
    """A Planetary Computer blob URL must be rejected with a clear redirect.

    PC's public TiTiler cannot tile raw ``*.blob.core.windows.net``
    hrefs (KeyError: 'tiles'). The wrapper detects PC URLs and returns
    a redirect message pointing at the canonical
    ``add_stac_layer(..., titiler_endpoint="pc")`` path. The redirect
    is returned as a plain string (a ``ToolMessage`` body), so the LLM
    can read the guidance and retry rather than the chat crashing.
    """
    from geoagent.tools.leafmap import leafmap_tools

    fake_map = MockLeafmap()
    tools = leafmap_tools(fake_map)
    add_cog = next(t for t in tools if t.name == "add_cog_layer")

    pc_url = (
        "https://sentinel2l2a01.blob.core.windows.net/sentinel2-l2/"
        "17/S/KV/2024/07/29/T17SKV_20240729T160829_TCI_10m.tif"
    )
    result = add_cog.invoke({"url": pc_url, "name": "S2 RGB"})

    assert "add_stac_layer" in result
    assert "titiler_endpoint" in result
    assert fake_map.layers == [], "the PC URL must not have been added"


def test_add_cog_layer_returns_error_string_on_leafmap_failure() -> None:
    """When leafmap raises, the tool returns a string so the LLM can recover.

    A previous bug propagated leafmap's ``KeyError`` up through deepagents
    and crashed the whole chat (``executed_tools=[]``). The wrapper now
    catches the exception and returns a descriptive string, which
    becomes the ``ToolMessage`` content the LLM sees on its next turn.
    """
    from geoagent.tools.leafmap import leafmap_tools

    class _BoomMap(MockLeafmap):
        def add_cog_layer(self, *args, **kwargs):  # type: ignore[override]
            raise RuntimeError("simulated TiTiler failure")

    boom_map = _BoomMap()
    tools = leafmap_tools(boom_map)
    add_cog = next(t for t in tools if t.name == "add_cog_layer")

    result = add_cog.invoke({"url": "https://example.com/public.tif", "name": "X"})

    assert isinstance(result, str)
    assert "add_cog_layer failed" in result
    assert "RuntimeError" in result
    assert "simulated TiTiler failure" in result
