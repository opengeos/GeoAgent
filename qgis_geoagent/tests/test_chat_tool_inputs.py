"""Tests for QGIS chat transcript tool-input formatting."""

from open_geoagent.dialogs.chat_dock import _conversation_markdown, _format_tool_calls


def test_conversation_markdown_includes_full_history() -> None:
    """Verify copying Markdown includes every chat turn."""
    text = _conversation_markdown(
        [
            {"sender": "You", "body": "hello", "markdown": False},
            {"sender": "OpenGeoAgent", "body": "**Hi**", "markdown": True},
            {"sender": "You", "body": "run flow accumulation", "markdown": False},
            {
                "sender": "OpenGeoAgent",
                "body": "Done\n\nTool inputs:\n- **`run_whitebox_tool`**",
                "markdown": True,
            },
        ]
    )

    assert text.startswith("## You\n\nhello")
    assert "## OpenGeoAgent\n\n**Hi**" in text
    assert "## You\n\nrun flow accumulation" in text
    assert "Tool inputs:" in text
    assert text.count("## OpenGeoAgent") == 2


def test_format_tool_calls_labels_routed_whitebox_tools() -> None:
    """Verify brokered Whitebox calls show the actual routed tool name."""
    text = _format_tool_calls(
        [
            {
                "name": "run_whitebox_tool",
                "args": {
                    "tool_name": "d8_pointer",
                    "parameters": {
                        "dem": "/tmp/filled_dem.tif",
                        "output": "/tmp/d8_pointer.tif",
                    },
                    "output_layer_name": "D8 pointer",
                    "working_dir": "/tmp",
                },
            },
            {
                "name": "get_whitebox_tool_info",
                "args": {"tool_name": "raster_streams_to_vector"},
            },
            {
                "name": "list_project_layers",
                "args": {},
            },
        ]
    )

    assert "- **`run_whitebox_tool -> d8_pointer`**:" in text
    assert "`tool_name=d8_pointer`" not in text
    assert "`dem=/tmp/filled_dem.tif`" in text
    assert "`output=/tmp/d8_pointer.tif`" in text
    assert "- **`get_whitebox_tool_info -> raster_streams_to_vector`**" in text
    assert "- **`list_project_layers`**" in text


def test_format_tool_calls_collapses_repeated_noop_defaults() -> None:
    """Verify repeated model retries are grouped after dropping no-op defaults."""
    text = _format_tool_calls(
        [
            {
                "name": "run_whitebox_fill_sinks",
                "args": {
                    "add_outputs_to_qgis": True,
                    "fill_pits": False,
                    "fix_flats": False,
                    "flat_increment": 0,
                    "layer_name": "",
                    "max_depth": 0,
                    "max_length": 0,
                    "method": "fill_depressions_wang_and_liu",
                    "output_layer_name": "DEM filled",
                    "output_path": "",
                    "verbose": False,
                },
            },
            {
                "name": "run_whitebox_fill_sinks",
                "args": {
                    "method": "fill_depressions_wang_and_liu",
                    "output_layer_name": "DEM filled",
                },
            },
        ]
    )

    assert "repeated 2 times" in text
    assert text.count("run_whitebox_fill_sinks") == 1
    assert "`method=fill_depressions_wang_and_liu`" in text
    assert "max_depth" not in text
    assert "fill_pits" not in text
