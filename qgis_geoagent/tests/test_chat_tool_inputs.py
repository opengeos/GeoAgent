"""Tests for QGIS chat transcript tool-input formatting."""

from qgis.PyQt.QtCore import QPoint, QRect, QSize
from qgis.PyQt.QtGui import QColor, QImage

from open_geoagent.dialogs.chat_dock import (
    _build_chat_content,
    _conversation_markdown,
    _format_tool_calls,
    _grab_screen_rect,
    _grab_widget_global_rect,
    _image_to_png_bytes,
    _normalized_crop_rect,
)


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
