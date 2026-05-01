"""Tests for QGIS chat transcript tool-input formatting."""

from qgis.PyQt.QtCore import QPoint, QRect, QSize
from qgis.PyQt.QtGui import QColor, QImage

from open_geoagent.dialogs.chat_dock import (
    SETTINGS_PREFIX,
    _build_chat_content,
    _console_ready_pyqgis_script,
    _conversation_markdown,
    _format_tool_calls,
    _grab_screen_rect,
    _grab_widget_global_rect,
    _image_to_png_bytes,
    _image_model_from_settings,
    _images_from_output_text,
    _latest_executable_snippet,
    _latest_pyqgis_script,
    _markdown_to_basic_html,
    _message_images_html,
    _message_images_markdown,
    _prepare_output_images,
    _normalized_crop_rect,
    _parse_markdown_transcript,
    _permission_allows_tool,
    _project_history_key,
)


class _FakeSettings:
    """Small QSettings stand-in for chat helper tests."""

    def __init__(self, values=None):
        self.values = dict(values or {})

    def value(self, key, default="", type=str):  # noqa: A002
        return self.values.get(key, default)


def _qimage_format(name):
    """Return QImage format across PyQt enum API variants."""
    container = getattr(QImage, "Format", QImage)
    return getattr(container, name)


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


def test_conversation_markdown_includes_output_images() -> None:
    """Verify copied Markdown includes generated image references."""
    text = _conversation_markdown(
        [
            {
                "sender": "OpenGeoAgent",
                "body": "Generated map",
                "markdown": True,
                "images": [{"path": "/tmp/open geoagent/map.png", "alt": "Map"}],
            }
        ]
    )

    assert "Generated map" in text
    assert "![Map](</tmp/open geoagent/map.png>)" in text


def test_markdown_renderer_supports_image_references() -> None:
    """Verify Markdown image output is rendered as inline HTML."""
    html = _markdown_to_basic_html("![Map](https://example.com/map.png)")

    assert "<img " in html
    assert "https://example.com/map.png" in html


def test_message_images_html_uses_clickable_thumbnail() -> None:
    """Verify generated image chat output renders as a clickable thumbnail."""
    html = _message_images_html(
        [
            {
                "path": "/tmp/geoagent_images/cat.png",
                "format": "png",
                "_href": "opengeoagent-image:0",
            }
        ]
    )

    assert "href='opengeoagent-image:0'" in html
    assert "width='180'" in html
    assert "/tmp/geoagent_images/cat.png" in html


def test_prepare_output_images_writes_model_image_bytes() -> None:
    """Verify model image bytes become transcript-safe local artifacts."""
    prepared = _prepare_output_images(
        [
            {
                "format": "png",
                "mime_type": "image/png",
                "bytes": b"\x89PNG\r\n\x1a\nfake",
            }
        ]
    )

    assert len(prepared) == 1
    assert prepared[0]["path"].endswith(".png")
    assert "bytes" not in prepared[0]
    assert _message_images_markdown(prepared).startswith("![OpenGeoAgent image 1](")


def test_images_from_output_text_extracts_generated_path() -> None:
    """Verify generated-image output paths become renderable image metadata."""
    images = _images_from_output_text(
        "Output path: /tmp/geoagent_images/cat-20260430-163529-1.png\n"
        "Elapsed: 18.95s"
    )

    assert images == [
        {
            "path": "/tmp/geoagent_images/cat-20260430-163529-1.png",
            "format": "png",
            "mime_type": "image/png",
        }
    ]


def test_image_to_png_bytes_serializes_qimage() -> None:
    """Verify pasted Qt images can be converted to model-ready PNG bytes."""
    image = QImage(12, 10, _qimage_format("Format_RGB32"))
    image.fill(QColor("red"))

    data, width, height = _image_to_png_bytes(image)

    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert width == 12
    assert height == 10


def test_build_chat_content_adds_image_blocks() -> None:
    """Verify image attachments are sent as Strands content blocks."""
    payload = _build_chat_content(
        "Describe this map screenshot.",
        [
            {
                "bytes": b"png-bytes",
                "format": "png",
                "width": 12,
                "height": 10,
            }
        ],
    )

    assert payload == [
        {"text": "Describe this map screenshot."},
        {"image": {"format": "png", "source": {"bytes": b"png-bytes"}}},
    ]


def test_image_model_from_settings_prefers_saved_then_env(monkeypatch) -> None:
    """Verify image generation model selection has a stable default path."""
    monkeypatch.setenv("GEOAGENT_IMAGE_MODEL", "gpt-image-1")
    saved_settings = _FakeSettings({f"{SETTINGS_PREFIX}image_model": "gpt-image-2"})

    assert _image_model_from_settings(saved_settings) == "gpt-image-2"
    assert _image_model_from_settings(_FakeSettings()) == "gpt-image-1"

    monkeypatch.delenv("GEOAGENT_IMAGE_MODEL")
    assert _image_model_from_settings(_FakeSettings()) == "gpt-image-2"


def test_latest_pyqgis_script_extracts_last_run_script() -> None:
    """Verify the copy-script button gets the full latest PyQGIS snippet."""
    script = _latest_pyqgis_script(
        [
            {"name": "get_active_layer", "args": {}},
            {
                "name": "run_pyqgis_script",
                "args": {"code": "print('first')"},
            },
            {
                "name": "run_pyqgis_script",
                "args": {"code": "active_layer.triggerRepaint()"},
            },
        ]
    )

    assert script == "active_layer.triggerRepaint()"


def test_latest_executable_snippet_supports_gee_and_results() -> None:
    """Verify Copy Script can find non-PyQGIS executable snippets."""
    snippet = _latest_executable_snippet(
        [
            {
                "name": "run_pyqgis_script",
                "args": {"code": "iface.mapCanvas().refresh()"},
            },
            {
                "name": "load_gee_dataset",
                "args": {},
                "result": {
                    "earth_engine_python_snippet": "image = ee.Image('NASA/NASADEM')"
                },
            },
            {
                "name": "run_gee_python_snippet",
                "args": {"code": "add_layer(ee.Image('x'), {}, 'x')"},
            },
        ]
    )

    assert snippet == {
        "kind": "Earth Engine",
        "tool_name": "run_gee_python_snippet",
        "code": "add_layer(ee.Image('x'), {}, 'x')",
    }


def test_latest_executable_snippet_finds_gee_loader_result() -> None:
    """Verify GEE load tools enable Copy Script from returned snippets."""
    snippet = _latest_executable_snippet(
        [
            {
                "name": "load_gee_dataset",
                "args": {"asset_id": "NASA/NASADEM_HGT/001"},
                "result": {
                    "success": True,
                    "earth_engine_python_snippet": (
                        "asset_id = 'NASA/NASADEM_HGT/001'\n"
                        "image = ee.Image(asset_id)"
                    ),
                },
            }
        ]
    )

    assert snippet == {
        "kind": "Earth Engine",
        "tool_name": "load_gee_dataset",
        "code": "asset_id = 'NASA/NASADEM_HGT/001'\nimage = ee.Image(asset_id)",
    }


def test_latest_executable_snippet_builds_gee_loader_script_from_args() -> None:
    """Verify Copy Script enables when Strands records only GEE tool inputs."""
    snippet = _latest_executable_snippet(
        [
            {
                "name": "load_gee_dataset",
                "args": {
                    "asset_id": "NASA/NASADEM_HGT/001",
                    "asset_type": "Image",
                    "bands": "elevation",
                    "layer_name": "Global NASADEM elevation",
                    "min_value": 0,
                    "max_value": 3000,
                    "palette": "0000ff,00ffff,ffff00,ff0000",
                },
            }
        ]
    )

    assert snippet["kind"] == "Earth Engine"
    assert snippet["tool_name"] == "load_gee_dataset"
    assert "asset_id = 'NASA/NASADEM_HGT/001'" in snippet["code"]
    assert "image = ee.Image(asset_id)" in snippet["code"]
    assert "'bands': 'elevation'" in snippet["code"]
    assert "Global NASADEM elevation" in snippet["code"]


def test_console_ready_pyqgis_script_defines_geoagent_context_names() -> None:
    """Verify copied scripts can run in the QGIS Python Console."""
    script = _console_ready_pyqgis_script("layer = active_layer\ncanvas.refresh()")

    assert "from qgis.utils import iface" in script
    assert "QgsProject.instance()" in script
    assert "active_layer = iface.activeLayer()" in script
    assert script.endswith("layer = active_layer\ncanvas.refresh()\n")


def test_normalized_crop_rect_clamps_reversed_selection() -> None:
    """Verify regional screenshot rectangles normalize and stay in bounds."""
    rect = _normalized_crop_rect(QRect(90, 80, -50, -40), QRect(0, 0, 70, 60))

    assert rect == QRect(40, 40, 30, 20)


def test_grab_screen_rect_passes_screen_local_rect() -> None:
    """Verify screen-region capture converts global coords to screen-local coords."""

    class _Screen:
        def __init__(self) -> None:
            self.calls = []

        def geometry(self):
            return QRect(10, 20, 100, 80)

        def grabWindow(self, win_id, x, y, width, height):
            self.calls.append((win_id, x, y, width, height))
            return "pixmap"

    screen = _Screen()
    result = _grab_screen_rect(screen, QRect(90, 80, 80, 60))

    assert result == "pixmap"
    assert screen.calls == [(0, 80, 60, 20, 20)]


def test_grab_widget_global_rect_maps_global_selection_to_widget() -> None:
    """Verify QGIS-window region capture crops from widget coordinates."""

    class _Widget:
        def __init__(self) -> None:
            self.calls = []
            self.offset = QPoint(20, 30)

        def mapToGlobal(self, point):
            return QPoint(self.offset.x() + point.x(), self.offset.y() + point.y())

        def mapFromGlobal(self, point):
            return QPoint(point.x() - self.offset.x(), point.y() - self.offset.y())

        def size(self):
            return QSize(100, 80)

        def grab(self, rect):
            self.calls.append(rect)
            return "pixmap"

    widget = _Widget()
    result = _grab_widget_global_rect(widget, QRect(90, 80, 80, 60))

    assert result == "pixmap"
    assert widget.calls == [QRect(70, 50, 30, 30)]


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


def test_parse_markdown_transcript_round_trips_exported_history() -> None:
    """Verify imported Markdown becomes project history messages."""
    messages = _parse_markdown_transcript(
        "## You\n\nhello\n\n## OpenGeoAgent\n\n**ok**\n"
    )

    assert messages == [
        {"sender": "You", "body": "hello", "markdown": False},
        {"sender": "OpenGeoAgent", "body": "**ok**", "markdown": True},
    ]


def test_unsaved_project_has_no_history_key(monkeypatch) -> None:
    """Untitled QGIS projects should not load shared chat history."""
    import sys
    import types

    monkeypatch.setitem(
        sys.modules,
        "qgis.core",
        types.SimpleNamespace(
            QgsProject=types.SimpleNamespace(
                instance=lambda: types.SimpleNamespace(fileName=lambda: "")
            )
        ),
    )

    assert _project_history_key(None) == ""


def test_agent_mode_load_guard_does_not_seed_prompt() -> None:
    """Restoring a saved workflow mode should not prefill a new chat prompt."""
    from open_geoagent.dialogs.chat_dock import ChatDockWidget

    class _PromptInput:
        def __init__(self):
            self.text = ""

        def toPlainText(self):
            return self.text

        def setPlainText(self, text):
            self.text = text

    dock = type("FakeDock", (), {})()
    dock._loading_settings = True
    dock.prompt_input = _PromptInput()

    ChatDockWidget._on_agent_mode_changed(dock, "GEE Data Catalogs")

    assert dock.prompt_input.toPlainText() == ""


def test_transcribed_prompt_moves_cursor_to_end() -> None:
    """Transcribed text should leave the prompt cursor at the end."""
    from open_geoagent.dialogs.chat_dock import ChatDockWidget

    class _PromptInput:
        def __init__(self):
            self.text = ""
            self.moved = False
            self.focused = False

        def toPlainText(self):
            return self.text

        def setPlainText(self, text):
            self.text = text

        def moveCursor(self, _cursor):
            self.moved = True

        def setFocus(self):
            self.focused = True

    dock = type("FakeDock", (), {})()
    dock.prompt_input = _PromptInput()
    dock._move_prompt_cursor_to_end = lambda: ChatDockWidget._move_prompt_cursor_to_end(
        dock
    )

    ChatDockWidget._insert_transcribed_prompt(dock, "hello world")

    assert dock.prompt_input.toPlainText() == "hello world"
    assert dock.prompt_input.moved is True
    assert dock.prompt_input.focused is True


def test_permission_profiles_filter_sensitive_tools() -> None:
    """Verify inspect-only mode hides mutating/long-running tools."""

    class _Meta:
        category = "qgis"
        requires_confirmation = True
        destructive = False
        long_running = False

    assert not _permission_allows_tool("Inspect only", "run_pyqgis_script", _Meta())
    assert _permission_allows_tool("Execute Scripts", "run_pyqgis_script", _Meta())
    assert _permission_allows_tool("Execute PyQGIS", "run_pyqgis_script", _Meta())
    assert _permission_allows_tool("Trusted auto-approve", "run_pyqgis_script", _Meta())


def test_execute_scripts_profile_allows_gee_snippet_tool() -> None:
    """Verify script execution profile covers GEE snippets, not only PyQGIS."""

    class _Meta:
        category = "gee_data_catalogs"
        requires_confirmation = True
        destructive = False
        long_running = True

    assert not _permission_allows_tool(
        "Inspect only", "run_gee_python_snippet", _Meta()
    )
    assert _permission_allows_tool("Execute Scripts", "run_gee_python_snippet", _Meta())
