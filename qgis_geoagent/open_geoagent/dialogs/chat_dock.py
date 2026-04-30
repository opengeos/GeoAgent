"""Dockable OpenGeoAgent chat interface."""

import asyncio
import hashlib
import os
import html
import json
import re
import time
import traceback

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import (
    QByteArray,
    QBuffer,
    QEvent,
    QIODevice,
    QObject,
    QPoint,
    QRect,
    QSize,
    Qt,
    QSettings,
    QThread,
    QTimer,
    pyqtSignal,
)
from qgis.PyQt.QtGui import QCursor, QGuiApplication, QPixmap, QTextCursor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QRubberBand,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

SETTINGS_PREFIX = "OpenGeoAgent/"
DEFAULT_PROVIDER = "openai-codex"
DEFAULT_MODELS = {
    "bedrock": "us.anthropic.claude-sonnet-4-6",
    "openai": "gpt-5.5",
    "openai-codex": "gpt-5.5",
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-pro-preview",
    "ollama": "qwen3.5:4b",
    "litellm": "openai/gpt-5.5",
}
PROVIDERS = [
    "anthropic",
    "bedrock",
    "gemini",
    "litellm",
    "ollama",
    "openai",
    "openai-codex",
]
MAX_CONTEXT_MESSAGES = 12
MAX_CONTEXT_CHARS = 12000
MAX_IMAGE_ATTACHMENTS = 4
MAX_IMAGE_EDGE = 1568
IMAGE_THUMBNAIL_SIZE = 72
SAMPLE_PROMPTS = [
    "Summarize the current QGIS project layers, CRS, extents, and feature counts.",
    "Zoom to the active layer and describe what it contains.",
    (
        "List visible layers and identify any layers with no features or "
        "invalid data sources."
    ),
    "Add an OpenStreetMap basemap and zoom to the project extent.",
    (
        "Inspect the active vector layer fields and suggest useful styling or "
        "labeling options."
    ),
    (
        "Select features in the active layer where population is greater than "
        "100000, then zoom to the selected features."
    ),
    (
        "Run a buffer around the active layer by 1000 meters and add the "
        "output to the project."
    ),
    "Find a WhiteboxTools command for calculating slope from the active DEM layer.",
    "Run a WhiteboxTools hillshade analysis on a local DEM and add the result to QGIS.",
    "Search WhiteboxTools for watershed or flow accumulation tools.",
    "Create a concise map QA checklist for this project before I export it.",
]
AGENT_MODES = [
    "General QGIS",
    "WhiteboxTools",
    "NASA Earthdata",
    "NASA OPERA",
    "GEE Data Catalogs",
    "STAC",
]
DEFAULT_AGENT_MODE = "General QGIS"
PERMISSION_PROFILES = [
    "Inspect only",
    "Edit layers",
    "Run processing",
    "Execute PyQGIS",
    "Trusted auto-approve",
]
DEFAULT_PERMISSION_PROFILE = "Inspect only"
WORKFLOW_PROMPTS = {
    "NASA Earthdata": [
        (
            "Search NASA Earthdata for datasets about surface water in the "
            "current map extent."
        ),
        (
            "Search NASA Earthdata granules for the selected dataset, display "
            "footprints, and summarize available raster links."
        ),
    ],
    "NASA OPERA": [
        "List available NASA OPERA datasets and recommend one for water mapping.",
        (
            "Search OPERA DSWx data for the current map extent and display "
            "matching footprints."
        ),
    ],
    "GEE Data Catalogs": [
        (
            "Search the Earth Engine data catalog for Sentinel-2 surface "
            "reflectance datasets."
        ),
        (
            "Find a dataset for land cover mapping and explain how it can be "
            "loaded into QGIS."
        ),
    ],
    "STAC": [
        (
            "Guide me through searching a STAC catalog for imagery over the "
            "current map extent."
        ),
        (
            "Check STAC-related dependencies and outline the steps to add a "
            "STAC raster layer in QGIS."
        ),
    ],
}


def _all_sample_prompts():
    """Return general sample prompts plus guided workflow prompts."""
    prompts = list(SAMPLE_PROMPTS)
    for mode_prompts in WORKFLOW_PROMPTS.values():
        prompts.extend(mode_prompts)
    return prompts


def _permission_allows_tool(permission_profile, tool_name, meta=None):
    """Return whether a QGIS-related tool may be exposed for a profile."""
    profile = permission_profile or DEFAULT_PERMISSION_PROFILE
    name = str(tool_name or "")
    category = str(getattr(meta, "category", "") or "")
    requires_confirmation = bool(getattr(meta, "requires_confirmation", False))
    destructive = bool(getattr(meta, "destructive", False))
    long_running = bool(getattr(meta, "long_running", False))

    if profile == "Trusted auto-approve":
        return True
    if profile == "Execute PyQGIS":
        return True
    if name == "run_pyqgis_script":
        return False
    if profile == "Run processing":
        return True
    if category in {"whitebox", "nasa_earthdata", "nasa_opera", "gee_data_catalogs"}:
        return profile in {"Run processing", "Execute PyQGIS", "Trusted auto-approve"}
    if profile == "Edit layers":
        return not destructive and name != "run_processing_algorithm"
    return not (requires_confirmation or destructive or long_running)


def _filter_tools_for_permission(agent, permission_profile):
    """Filter an agent's tool surface when the core factory lacks profile support."""
    try:
        registry = getattr(agent, "tool_registry", None)
        strands_agent = getattr(agent, "strands_agent", None)
        tools = list(
            getattr(strands_agent, "tools", None) or getattr(agent, "_tool_list", [])
        )
        if not tools:
            return agent
        filtered = []
        for tool in tools:
            name = (
                getattr(tool, "tool_name", "")
                or getattr(tool, "__name__", "")
                or getattr(tool, "name", "")
            )
            meta = getattr(tool, "_geoagent_meta", None)
            if meta is None and registry is not None and name:
                try:
                    meta = registry.get(name)
                except Exception:
                    meta = None
            if _permission_allows_tool(permission_profile, name, meta):
                filtered.append(tool)
        if len(filtered) != len(tools) and hasattr(agent, "_tool_list"):
            agent._tool_list = filtered
            if hasattr(agent, "_rebuild_strands_agent"):
                agent._rebuild_strands_agent()
    except Exception:
        pass
    return agent


def _project_history_key(iface):
    """Return a stable QSettings key suffix for the current QGIS project."""
    path = ""
    try:
        from qgis.core import QgsProject

        path = QgsProject.instance().fileName() or ""
    except Exception:
        path = ""
    if not path:
        try:
            project = getattr(iface, "project", lambda: None)()
            path = project.fileName() if project is not None else ""
        except Exception:
            path = ""
    raw = path or "unsaved-project"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{SETTINGS_PREFIX}history/{digest}"


def _parse_markdown_transcript(text):
    """Parse exported Markdown transcript blocks back into chat messages."""
    messages = []
    sender = None
    body_lines = []
    for line in str(text or "").splitlines():
        if line.startswith("## "):
            if sender is not None:
                body = "\n".join(body_lines).strip()
                if body:
                    messages.append(
                        {"sender": sender, "body": body, "markdown": sender != "You"}
                    )
            sender = line[3:].strip() or "OpenGeoAgent"
            body_lines = []
        else:
            body_lines.append(line)
    if sender is not None:
        body = "\n".join(body_lines).strip()
        if body:
            messages.append(
                {"sender": sender, "body": body, "markdown": sender != "You"}
            )
    return messages


def _job_status_text(job):
    """Return compact text for one recorded job."""
    tools = job.get("tools") or ""
    error = job.get("error") or ""
    parts = [job.get("status", "")]
    if tools:
        parts.append(f"tools: {tools}")
    if error:
        parts.append(f"error: {error[:120]}")
    return "; ".join(part for part in parts if part)


def _default_model_for_provider(provider):
    """Return the UI default model id for a provider."""
    return DEFAULT_MODELS.get(provider, "")


def _setting(settings, key, default="", value_type=str):
    """Read a plugin setting value."""
    return settings.value(f"{SETTINGS_PREFIX}{key}", default, type=value_type)


def _apply_environment_from_settings(settings):
    """Apply provider credentials from QSettings to the current QGIS process."""
    # Mapping of QSettings key names to environment variable names. The strings
    # are identifiers, not credentials; the actual values are read from QSettings
    # at runtime. ``# pragma: allowlist secret`` silences detect-secrets keyword
    # heuristics on lines that contain "api_key" / "API_KEY".
    env_map = {
        "openai_api_key": "OPENAI_API_KEY",  # pragma: allowlist secret
        "anthropic_api_key": "ANTHROPIC_API_KEY",  # pragma: allowlist secret
        "gemini_api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "aws_region": "AWS_REGION",
        "ollama_host": "OLLAMA_HOST",
        "litellm_api_key": "LITELLM_API_KEY",  # pragma: allowlist secret
        "litellm_base_url": "LITELLM_BASE_URL",
    }
    for key, env_names in env_map.items():
        value = _setting(settings, key, "").strip()
        if value:
            if isinstance(env_names, str):
                env_names = (env_names,)
            for env_name in env_names:
                os.environ[env_name] = value


def _qt_value(enum_name, member_name):
    """Return a Qt enum member across PyQt enum API variants."""
    container = getattr(Qt, enum_name, Qt)
    return getattr(container, member_name)


def _enum_value(cls, enum_name, member_name):
    """Return an enum member from either scoped or legacy Qt APIs."""
    container = getattr(cls, enum_name, cls)
    return getattr(container, member_name)


def _exec_dialog(dialog):
    """Execute a dialog across PyQt API variants."""
    exec_fn = getattr(dialog, "exec", None) or getattr(dialog, "exec_", None)
    return exec_fn()


def _exec_menu(menu, pos):
    """Execute a menu across PyQt API variants."""
    exec_fn = getattr(menu, "exec", None) or getattr(menu, "exec_", None)
    return exec_fn(pos)


def _plain_text_to_html(text):
    """Convert plain text to basic HTML."""
    return html.escape(text).replace("\n", "<br>")


def _inline_markdown_to_html(text):
    """Convert inline Markdown spans to HTML."""
    text = html.escape(text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    return text


def _markdown_to_basic_html(markdown):
    """Small fallback renderer for common Markdown when Qt lacks setMarkdown."""
    lines = markdown.splitlines()
    html_lines = []
    in_ul = False
    in_ol = False
    in_code = False
    code_lines = []

    def close_lists():
        """Close any open HTML list elements."""
        nonlocal in_ul, in_ol
        if in_ul:
            html_lines.append("</ul>")
            in_ul = False
        if in_ol:
            html_lines.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                html_lines.append(
                    f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>"
                )
                code_lines = []
                in_code = False
            else:
                close_lists()
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            close_lists()
            html_lines.append("")
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", stripped)
        if heading:
            close_lists()
            level = len(heading.group(1))
            html_lines.append(
                f"<h{level}>{_inline_markdown_to_html(heading.group(2))}</h{level}>"
            )
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet:
            if in_ol:
                html_lines.append("</ol>")
                in_ol = False
            if not in_ul:
                html_lines.append("<ul>")
                in_ul = True
            html_lines.append(f"<li>{_inline_markdown_to_html(bullet.group(1))}</li>")
            continue

        numbered = re.match(r"^\d+\.\s+(.+)$", stripped)
        if numbered:
            if in_ul:
                html_lines.append("</ul>")
                in_ul = False
            if not in_ol:
                html_lines.append("<ol>")
                in_ol = True
            html_lines.append(f"<li>{_inline_markdown_to_html(numbered.group(1))}</li>")
            continue

        close_lists()
        html_lines.append(f"<p>{_inline_markdown_to_html(stripped)}</p>")

    if in_code:
        html_lines.append(
            f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>"
        )
    close_lists()
    return "\n".join(html_lines)


def _format_tool_value(value, max_length=220):
    """Format a tool argument value for compact display."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, sort_keys=True)
        except TypeError:
            text = str(value)
    text = text.replace("\n", " ").strip()
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


def _tool_call_label_and_args(name, args):
    """Return a display label plus args for a recorded tool call."""
    display_args = dict(args)
    routed_tool = str(display_args.pop("tool_name", "") or "").strip()
    if name == "run_whitebox_tool" and routed_tool:
        parameters = display_args.pop("parameters", None)
        if isinstance(parameters, dict):
            display_args.update(parameters)
        return f"{name} -> {routed_tool}", display_args
    if name == "get_whitebox_tool_info" and routed_tool:
        return f"{name} -> {routed_tool}", display_args
    return name, display_args


def _clean_tool_display_args(args):
    """Drop no-op optional values that make repeated calls look different."""
    cleaned = {}
    default_values = {
        "add_outputs_to_qgis": True,
        "category": "",
        "fill_pits": False,
        "fix_flats": False,
        "flat_increment": 0,
        "layer_name": "",
        "max_depth": 0,
        "max_length": 0,
        "max_procs": None,
        "output_path": "",
        "verbose": False,
    }
    for key, value in args.items():
        if key in default_values and value == default_values[key]:
            continue
        if value is None or value == "":
            continue
        cleaned[key] = value
    return cleaned


def _format_tool_calls(tool_calls):
    """Return Markdown lines describing tool input parameters."""
    if not tool_calls:
        return ""
    lines = ["Tool inputs:"]
    grouped = []
    index_by_signature = {}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = str(call.get("name") or "").strip()
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if not name:
            continue
        label, args = _tool_call_label_and_args(name, args)
        args = _clean_tool_display_args(args)
        try:
            args_key = json.dumps(args, sort_keys=True, default=str)
        except TypeError:
            args_key = str(sorted(args.items()))
        signature = (label, args_key)
        if signature in index_by_signature:
            grouped[index_by_signature[signature]][2] += 1
        else:
            index_by_signature[signature] = len(grouped)
            grouped.append([label, args, 1])

    for label, args, count in grouped:
        suffix = f" (repeated {count} times)" if count > 1 else ""
        if args:
            params = ", ".join(
                f"`{key}={_format_tool_value(value)}`"
                for key, value in sorted(args.items())
            )
            lines.append(f"- **`{label}`**{suffix}: {params}")
        else:
            lines.append(f"- **`{label}`**{suffix}")
    return "\n".join(lines) if len(lines) > 1 else ""


def _latest_pyqgis_script(tool_calls):
    """Return the last PyQGIS script found in recorded tool calls."""
    for call in reversed(tool_calls or []):
        if not isinstance(call, dict):
            continue
        if str(call.get("name") or "").strip() != "run_pyqgis_script":
            continue
        args = call.get("args")
        if not isinstance(args, dict):
            continue
        code = str(args.get("code") or "").strip()
        if code:
            return code
    return ""


def _console_ready_pyqgis_script(code):
    """Return PyQGIS code that can run in the QGIS Python Console."""
    code = str(code or "").strip()
    if not code:
        return ""
    preamble = """# OpenGeoAgent PyQGIS script.
# This preamble makes scripts copied from GeoAgent runnable in the QGIS Python Console.
try:
    iface
except NameError:
    from qgis.utils import iface

try:
    project
except NameError:
    from qgis.core import QgsProject
    project = QgsProject.instance()

try:
    canvas
except NameError:
    canvas = iface.mapCanvas()

try:
    active_layer
except NameError:
    active_layer = iface.activeLayer()
"""
    return f"{preamble}\n{code}\n"


def _conversation_markdown(messages):
    """Return the full chat transcript as Markdown."""
    blocks = []
    for msg in messages:
        sender = str(msg.get("sender") or "").strip()
        body = str(msg.get("display_body") or msg.get("body") or "").strip()
        if not sender or not body:
            continue
        blocks.append(f"## {sender}\n\n{body}")
    return "\n\n".join(blocks)


def _scaled_image_for_attachment(image):
    """Return an image copy small enough for model upload."""
    if image.width() <= MAX_IMAGE_EDGE and image.height() <= MAX_IMAGE_EDGE:
        return image
    keep_aspect = _qt_value("AspectRatioMode", "KeepAspectRatio")
    smooth = _qt_value("TransformationMode", "SmoothTransformation")
    return image.scaled(MAX_IMAGE_EDGE, MAX_IMAGE_EDGE, keep_aspect, smooth)


def _image_to_png_bytes(image):
    """Serialize a Qt image-like object to PNG bytes."""
    if hasattr(image, "toImage"):
        image = image.toImage()
    if image is None or not hasattr(image, "isNull") or image.isNull():
        raise ValueError("Clipboard image is empty or unsupported.")

    image = _scaled_image_for_attachment(image)
    data = QByteArray()
    buffer = QBuffer(data)
    write_only = _enum_value(QIODevice, "OpenModeFlag", "WriteOnly")
    if not buffer.open(write_only):
        raise ValueError("Could not prepare clipboard image for upload.")
    try:
        if not image.save(buffer, "PNG"):
            raise ValueError("Could not encode clipboard image as PNG.")
    finally:
        buffer.close()
    return bytes(data.data()), image.width(), image.height()


def _attachment_to_content_block(attachment):
    """Convert one image attachment into a Strands content block."""
    return {
        "image": {
            "format": attachment["format"],
            "source": {"bytes": attachment["bytes"]},
        }
    }


def _build_chat_content(prompt, attachments):
    """Build a Strands-compatible text plus image payload."""
    if not attachments:
        return prompt
    content = [{"text": prompt}]
    content.extend(_attachment_to_content_block(item) for item in attachments)
    return content


def _normalized_crop_rect(rect, bounds):
    """Normalize and clamp a crop rectangle to pixmap bounds."""
    if rect is None or bounds is None:
        return QRect()
    rect = rect.normalized()
    if hasattr(rect, "intersected"):
        rect = rect.intersected(bounds)
    if rect.isNull() or rect.width() <= 1 or rect.height() <= 1:
        return QRect()
    return rect


def _crop_pixmap(pixmap, rect):
    """Return a normalized rectangular crop from a pixmap."""
    if pixmap is None or pixmap.isNull() or rect is None:
        return QPixmap()
    rect = _normalized_crop_rect(rect, pixmap.rect())
    if rect.isNull():
        return QPixmap()
    return pixmap.copy(rect)


def _screen_for_widget(widget):
    """Return the screen containing a widget, falling back to cursor/primary."""
    screen = None
    if widget is not None:
        try:
            window_handle = widget.windowHandle()
        except Exception:
            window_handle = None
        if window_handle is not None:
            screen = window_handle.screen()
        if screen is None and hasattr(widget, "screen"):
            try:
                screen = widget.screen()
            except Exception:
                screen = None
    if screen is None and hasattr(QGuiApplication, "screenAt"):
        screen = QGuiApplication.screenAt(QCursor.pos())
    return screen or QGuiApplication.primaryScreen()


def _grab_screen_rect(screen, rect):
    """Grab a global desktop rectangle from a screen."""
    if screen is None or rect is None:
        return QPixmap()
    screen_geometry = screen.geometry()
    rect = _normalized_crop_rect(rect, screen_geometry)
    if rect.isNull():
        return QPixmap()
    local = rect.translated(-screen_geometry.topLeft())
    return screen.grabWindow(0, local.x(), local.y(), local.width(), local.height())


def _global_widget_rect(widget):
    """Return a widget's geometry in global screen coordinates."""
    if widget is None:
        return QRect()
    return QRect(widget.mapToGlobal(QPoint(0, 0)), widget.size())


def _grab_widget_global_rect(widget, global_rect):
    """Grab a global rectangle from a QWidget by mapping it into widget space."""
    if widget is None or global_rect is None:
        return QPixmap()
    widget_rect = _global_widget_rect(widget)
    rect = _normalized_crop_rect(global_rect, widget_rect)
    if rect.isNull():
        return QPixmap()
    local_rect = QRect(widget.mapFromGlobal(rect.topLeft()), rect.size())
    return widget.grab(local_rect)


class PromptTextEdit(QPlainTextEdit):
    """Prompt editor with chat-friendly keyboard shortcuts."""

    send_requested = pyqtSignal()
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()
    image_pasted = pyqtSignal(object)

    def insertFromMimeData(self, source):
        """Handle pasted clipboard images before falling back to text paste."""
        if source.hasImage():
            self.image_pasted.emit(source.imageData())
            if source.hasText():
                super().insertFromMimeData(source)
            return
        super().insertFromMimeData(source)

    def keyPressEvent(self, event):
        """Handle send and prompt-history keyboard shortcuts."""
        key = event.key()
        modifiers = event.modifiers()
        control = _qt_value("KeyboardModifier", "ControlModifier")
        key_return = _qt_value("Key", "Key_Return")
        key_enter = _qt_value("Key", "Key_Enter")
        key_up = _qt_value("Key", "Key_Up")
        key_down = _qt_value("Key", "Key_Down")

        if modifiers & control and key in (key_return, key_enter):
            self.send_requested.emit()
            event.accept()
            return
        if key == key_up and not modifiers:
            self.previous_requested.emit()
            event.accept()
            return
        if key == key_down and not modifiers:
            self.next_requested.emit()
            event.accept()
            return

        super().keyPressEvent(event)


class AttachmentThumbnail(QLabel):
    """Clickable thumbnail for a pending image attachment."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        """Open preview on left-click."""
        if event.button() == _qt_value("MouseButton", "LeftButton"):
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class CanvasRegionCapture(QObject):
    """Event filter that lets the user drag a screenshot rectangle on a canvas."""

    finished = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, widget, parent=None):
        super().__init__(parent)
        self.widget = widget
        self.origin = QPoint()
        self.rubber_band = None
        self.active = False
        self._prior_mouse_tracking = None

    def start(self):
        """Begin capturing mouse events from the canvas widget."""
        self.widget.installEventFilter(self)
        try:
            self._prior_mouse_tracking = bool(self.widget.hasMouseTracking())
        except Exception:
            self._prior_mouse_tracking = None
        self.widget.setMouseTracking(True)
        self.widget.setCursor(_qt_value("CursorShape", "CrossCursor"))
        self.widget.setFocus()

    def stop(self):
        """Stop capture and remove temporary UI."""
        self.widget.removeEventFilter(self)
        self.widget.unsetCursor()
        if self._prior_mouse_tracking is not None:
            self.widget.setMouseTracking(self._prior_mouse_tracking)
            self._prior_mouse_tracking = None
        if self.rubber_band is not None:
            self.rubber_band.hide()
            self.rubber_band.deleteLater()
            self.rubber_band = None
        self.active = False

    def eventFilter(self, watched, event):
        """Capture a press-drag-release rectangle or cancel on Escape."""
        if watched is not self.widget:
            return False
        event_type = event.type()
        mouse_press = _enum_value(QEvent, "Type", "MouseButtonPress")
        mouse_move = _enum_value(QEvent, "Type", "MouseMove")
        mouse_release = _enum_value(QEvent, "Type", "MouseButtonRelease")
        key_press = _enum_value(QEvent, "Type", "KeyPress")
        left_button = _qt_value("MouseButton", "LeftButton")
        escape_key = _qt_value("Key", "Key_Escape")

        if event_type == key_press and event.key() == escape_key:
            self.stop()
            self.cancelled.emit()
            return True

        if event_type == mouse_press and event.button() == left_button:
            self.origin = event.pos()
            band_shape = _enum_value(QRubberBand, "Shape", "Rectangle")
            self.rubber_band = QRubberBand(band_shape, self.widget)
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            self.active = True
            return True

        if event_type == mouse_move and self.active and self.rubber_band is not None:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
            return True

        if (
            event_type == mouse_release
            and self.active
            and event.button() == left_button
        ):
            rect = QRect(self.origin, event.pos()).normalized()
            self.stop()
            self.finished.emit(rect)
            return True

        return False


class ScreenRegionCapture(QWidget):
    """Fullscreen overlay for selecting a screenshot region outside the canvas."""

    finished = pyqtSignal(object)
    cancelled = pyqtSignal()

    def __init__(self, screen):
        super().__init__(None)
        self.screen_obj = screen or QGuiApplication.primaryScreen()
        self.origin = QPoint()
        self.rubber_band = None
        self.active = False
        frameless = _qt_value("WindowType", "FramelessWindowHint")
        on_top = _qt_value("WindowType", "WindowStaysOnTopHint")
        tool = _qt_value("WindowType", "Tool")
        self.setWindowFlags(frameless | on_top | tool)
        self.setCursor(_qt_value("CursorShape", "CrossCursor"))
        self.setStyleSheet("background: rgba(0, 0, 0, 35);")
        translucent = _qt_value("WidgetAttribute", "WA_TranslucentBackground")
        self.setAttribute(translucent, True)
        self.setGeometry(self.screen_obj.geometry())

    def start(self):
        """Show the overlay on the target screen."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus()

    def keyPressEvent(self, event):
        """Cancel region capture on Escape."""
        if event.key() == _qt_value("Key", "Key_Escape"):
            self.cancelled.emit()
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Start the screen region selection."""
        if event.button() != _qt_value("MouseButton", "LeftButton"):
            return
        self.origin = event.pos()
        band_shape = _enum_value(QRubberBand, "Shape", "Rectangle")
        self.rubber_band = QRubberBand(band_shape, self)
        self.rubber_band.setGeometry(QRect(self.origin, QSize()))
        self.rubber_band.show()
        self.active = True

    def mouseMoveEvent(self, event):
        """Update the selected screen region."""
        if self.active and self.rubber_band is not None:
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        """Finish screen region selection and emit the global selection rectangle."""
        if not self.active or event.button() != _qt_value("MouseButton", "LeftButton"):
            return
        local_rect = QRect(self.origin, event.pos()).normalized()
        global_rect = QRect(self.mapToGlobal(local_rect.topLeft()), local_rect.size())
        self.active = False
        if self.rubber_band is not None:
            self.rubber_band.hide()
        self.hide()
        QTimer.singleShot(100, lambda: self._finish_capture(global_rect))

    def _finish_capture(self, global_rect):
        """Emit the selected region after the overlay has been hidden."""
        self.finished.emit(global_rect)
        self.close()
        self.deleteLater()


class ChatWorker(QThread):
    """Run GeoAgent chat without blocking the QGIS UI."""

    finished = pyqtSignal(dict)
    chunk_received = pyqtSignal(str)

    def __init__(
        self,
        iface,
        prompt,
        provider,
        model_id,
        fast,
        max_tokens,
        auto_approve_tools=False,
        stream=False,
        agent_mode=DEFAULT_AGENT_MODE,
        permission_profile=DEFAULT_PERMISSION_PROFILE,
        parent=None,
    ):
        super().__init__(parent)
        self.iface = iface
        self.prompt = prompt
        self.provider = provider
        self.model_id = model_id or None
        self.fast = fast
        self.max_tokens = max_tokens
        self.auto_approve_tools = bool(auto_approve_tools)
        self.stream = bool(stream)
        self.agent_mode = agent_mode or DEFAULT_AGENT_MODE
        self.permission_profile = permission_profile or DEFAULT_PERMISSION_PROFILE

    def run(self):
        """Create a GeoAgent QGIS agent and execute one chat turn."""
        try:
            from geoagent import GeoAgentConfig
            import geoagent

            try:
                from qgis.core import QgsProject

                project = QgsProject.instance()
            except Exception:
                project = None

            config = GeoAgentConfig(
                provider=self.provider,
                model=self.model_id,
                max_tokens=self.max_tokens,
            )
            factory_name = {
                "General QGIS": "for_qgis",
                "WhiteboxTools": "for_whitebox",
                "NASA Earthdata": "for_nasa_earthdata",
                "NASA OPERA": "for_nasa_opera",
                "GEE Data Catalogs": "for_gee_data_catalogs",
                "STAC": "for_qgis",
            }.get(self.agent_mode, "for_qgis")
            factory = getattr(geoagent, factory_name)
            kwargs = {
                "project": project,
                "config": config,
                "fast": self.fast,
                "confirm": self._confirm_tool,
            }
            if self.permission_profile:
                kwargs["permission_profile"] = self.permission_profile
            try:
                agent = factory(self.iface, **kwargs)
            except TypeError as exc:
                if "permission_profile" not in str(exc):
                    raise
                kwargs.pop("permission_profile", None)
                agent = factory(self.iface, **kwargs)
                agent = _filter_tools_for_permission(agent, self.permission_profile)
            agent = _filter_tools_for_permission(agent, self.permission_profile)
            if self.agent_mode == "STAC":
                self.prompt = (
                    "You are in STAC guidance mode. The current GeoAgent STAC "
                    "tool surface is limited, so provide dependency checks, "
                    "catalog-search steps, and QGIS loading guidance without "
                    "pretending to run missing STAC loader tools.\n\n"
                    f"{self.prompt}"
                )
            if self.stream:
                self._run_streaming_chat(agent)
                return

            response = agent.chat(self.prompt)

            if self.isInterruptionRequested():
                self.finished.emit(
                    {
                        "success": False,
                        "answer": "",
                        "error": "",
                        "tools": ", ".join(response.executed_tools or []),
                        "tool_calls": response.tool_calls or [],
                        "cancelled": ", ".join(response.cancelled_tools or []),
                        "elapsed": f"{response.execution_time:.2f}s",
                        "cancelled_by_user": True,
                    }
                )
                return

            self.finished.emit(
                {
                    "success": bool(response.success),
                    "answer": response.answer_text or "",
                    "error": response.error_message or "",
                    "tools": ", ".join(response.executed_tools or []),
                    "tool_calls": response.tool_calls or [],
                    "cancelled": ", ".join(response.cancelled_tools or []),
                    "elapsed": f"{response.execution_time:.2f}s",
                    "cancelled_by_user": False,
                }
            )
        except Exception as exc:
            self.finished.emit(
                {
                    "success": False,
                    "answer": "",
                    "error": f"{exc}\n\n{traceback.format_exc()}",
                    "tools": "",
                    "cancelled": "",
                    "elapsed": "",
                    "cancelled_by_user": self.isInterruptionRequested(),
                }
            )

    def _run_streaming_chat(self, agent):
        """Stream one chat turn and emit text chunks as they arrive."""
        started_at = time.time()
        chunks = []
        final_result = None

        async def _collect_stream():
            nonlocal final_result
            async for event in agent.stream_chat(self.prompt):
                if self.isInterruptionRequested():
                    break
                if not isinstance(event, dict):
                    continue
                if "data" in event:
                    chunk = str(event["data"])
                    chunks.append(chunk)
                    self.chunk_received.emit(chunk)
                if "result" in event:
                    final_result = event["result"]

        asyncio.run(_collect_stream())

        tool_metrics = getattr(
            getattr(final_result, "metrics", None), "tool_metrics", {}
        )
        executed_tools = (
            list(tool_metrics.keys()) if isinstance(tool_metrics, dict) else []
        )
        stop_reason = str(getattr(final_result, "stop_reason", "end_turn"))
        success = stop_reason not in ("cancelled", "guardrail_intervened")
        if self.isInterruptionRequested():
            success = False

        self.finished.emit(
            {
                "success": success,
                "answer": "".join(chunks),
                "error": "" if success else f"stop_reason={stop_reason}",
                "tools": ", ".join(executed_tools),
                "tool_calls": list(getattr(agent, "_tool_calls", []) or []),
                "cancelled": ", ".join(getattr(agent, "_cancelled", []) or []),
                "elapsed": f"{time.time() - started_at:.2f}s",
                "cancelled_by_user": self.isInterruptionRequested(),
                "streamed": True,
            }
        )

    def _confirm_tool(self, request):
        """Ask the QGIS user before running confirmation-required tools."""
        if self.isInterruptionRequested():
            return False
        if self.auto_approve_tools:
            return True

        from geoagent.tools._qt_marshal import run_on_qt_gui_thread

        def _ask():
            parent = None
            try:
                parent = self.iface.mainWindow()
            except Exception as exc:
                QgsMessageLog.logMessage(
                    "Could not resolve QGIS main window for confirmation dialog: "
                    f"{exc}",
                    "OpenGeoAgent",
                    Qgis.MessageLevel.Warning,
                )

            arg_lines = []
            for key, value in request.args.items():
                text = str(value)
                if len(text) > 160:
                    text = text[:157] + "..."
                arg_lines.append(f"{key}: {text}")

            details = "\n".join(arg_lines) if arg_lines else "No arguments"
            message = (
                f"GeoAgent wants to run:\n\n{request.tool_name}\n\n"
                f"{details}\n\nAllow this action?"
            )
            reply = QMessageBox.question(
                parent,
                "Confirm GeoAgent Tool",
                message,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes

        return bool(run_on_qt_gui_thread(_ask))


class ChatDockWidget(QDockWidget):
    """Dock widget that sends user prompts to a GeoAgent QGIS agent."""

    def __init__(self, iface, parent=None):
        super().__init__("OpenGeoAgent Chat", parent)
        self.iface = iface
        self.settings = QSettings()
        self._worker = None
        self._region_capture = None
        self._screen_region_capture = None
        self._prompt_history = []
        self._history_index = None
        self._messages = []
        self._last_pyqgis_script = ""
        self._image_attachments = []
        self._streaming_message_index = None
        self._streaming_answer = ""
        self._status_started_at = None
        self._status_base_text = "Running"
        self._status_frame = 0
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._update_running_status)
        self._stream_render_timer = QTimer(self)
        self._stream_render_timer.setSingleShot(True)
        self._stream_render_timer.setInterval(75)
        self._stream_render_timer.timeout.connect(self._flush_streaming_render)
        self._history_key = _project_history_key(iface)
        self._jobs = []
        self._active_job_index = None

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(220)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the chat dock widgets and signal connections."""
        main_widget = QWidget()
        self.setWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(8)

        self.model_group = QGroupBox("Model")
        self.model_group.setCheckable(True)
        self.model_group.setChecked(True)
        self.model_group.toggled.connect(self._on_model_section_toggled)
        model_group_layout = QVBoxLayout(self.model_group)
        model_group_layout.setContentsMargins(8, 8, 8, 8)

        self.model_controls = QWidget()
        model_layout = QFormLayout(self.model_controls)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_group_layout.addWidget(self.model_controls)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(PROVIDERS)
        self.provider_combo.setMinimumContentsLength(8)
        self.provider_combo.setSizeAdjustPolicy(
            _enum_value(
                QComboBox,
                "SizeAdjustPolicy",
                "AdjustToMinimumContentsLengthWithIcon",
            )
        )
        self.provider_combo.setSizePolicy(
            _enum_value(QSizePolicy, "Policy", "Ignored"),
            _enum_value(QSizePolicy, "Policy", "Fixed"),
        )
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        model_layout.addRow("Provider:", self.provider_combo)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Use provider default")
        self.model_input.setSizePolicy(
            _enum_value(QSizePolicy, "Policy", "Ignored"),
            _enum_value(QSizePolicy, "Policy", "Fixed"),
        )
        model_layout.addRow("Model:", self.model_input)

        self.agent_mode_combo = QComboBox()
        self.agent_mode_combo.addItems(AGENT_MODES)
        self.agent_mode_combo.setSizePolicy(
            _enum_value(QSizePolicy, "Policy", "Ignored"),
            _enum_value(QSizePolicy, "Policy", "Fixed"),
        )
        self.agent_mode_combo.currentTextChanged.connect(self._on_agent_mode_changed)
        model_layout.addRow("Agent mode:", self.agent_mode_combo)

        self.permission_combo = QComboBox()
        self.permission_combo.addItems(PERMISSION_PROFILES)
        self.permission_combo.setSizePolicy(
            _enum_value(QSizePolicy, "Policy", "Ignored"),
            _enum_value(QSizePolicy, "Policy", "Fixed"),
        )
        model_layout.addRow("Permissions:", self.permission_combo)

        self.fast_check = QCheckBox("Fast mode")
        self.stream_check = QCheckBox("Stream output")
        self.auto_approve_tools_check = QCheckBox("Auto approve running tools")
        self.auto_approve_tools_check.setToolTip(
            "Run confirmation-required tools without prompting for this session."
        )
        self.stream_check.setToolTip(
            "Show model text as it arrives instead of waiting for the full response."
        )
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.fast_check)
        mode_layout.addWidget(self.stream_check)
        mode_layout.addWidget(self.auto_approve_tools_check)
        mode_layout.addStretch(1)
        model_layout.addRow("", mode_layout)

        layout.addWidget(self.model_group)

        sample_layout = QHBoxLayout()
        self.sample_combo = QComboBox()
        self.sample_combo.addItem("Sample prompts...")
        self.sample_combo.addItems(_all_sample_prompts())
        self.sample_combo.setMinimumContentsLength(14)
        self.sample_combo.setSizeAdjustPolicy(
            _enum_value(
                QComboBox,
                "SizeAdjustPolicy",
                "AdjustToMinimumContentsLengthWithIcon",
            )
        )
        self.sample_combo.setSizePolicy(
            _enum_value(QSizePolicy, "Policy", "Ignored"),
            _enum_value(QSizePolicy, "Policy", "Fixed"),
        )
        self.sample_combo.currentTextChanged.connect(self._select_sample_prompt)
        sample_layout.addWidget(self.sample_combo, 1)
        layout.addLayout(sample_layout)

        self.jobs_group = QGroupBox("Jobs")
        self.jobs_group.setCheckable(True)
        self.jobs_group.setChecked(True)
        self.jobs_group.toggled.connect(self._on_jobs_section_toggled)
        jobs_layout = QVBoxLayout(self.jobs_group)

        self.jobs_controls = QWidget()
        jobs_controls_layout = QVBoxLayout(self.jobs_controls)
        jobs_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.jobs_table = QTableWidget(0, 4)
        self.jobs_table.setHorizontalHeaderLabels(
            ["Status", "Mode", "Prompt", "Details"]
        )
        self.jobs_table.setMaximumHeight(120)
        self.jobs_table.setSelectionBehavior(
            _enum_value(QAbstractItemView, "SelectionBehavior", "SelectRows")
        )
        jobs_controls_layout.addWidget(self.jobs_table)
        jobs_btn_layout = QHBoxLayout()
        self.rerun_job_btn = QPushButton("Rerun Job")
        self.rerun_job_btn.clicked.connect(self._rerun_selected_job)
        jobs_btn_layout.addWidget(self.rerun_job_btn)
        self.cancel_job_btn = QPushButton("Cancel Active")
        self.cancel_job_btn.clicked.connect(self._cancel_running_task)
        self.cancel_job_btn.setEnabled(False)
        jobs_btn_layout.addWidget(self.cancel_job_btn)
        jobs_btn_layout.addStretch(1)
        jobs_controls_layout.addLayout(jobs_btn_layout)
        jobs_layout.addWidget(self.jobs_controls)
        layout.addWidget(self.jobs_group)

        self.transcript = QTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setAcceptRichText(True)
        self.transcript.setPlaceholderText("Conversation will appear here.")
        layout.addWidget(self.transcript, 1)

        self.prompt_input = PromptTextEdit()
        self.prompt_input.setPlaceholderText(
            "Ask about the current QGIS project or request a map action."
        )
        self.prompt_input.setMaximumHeight(90)
        self.prompt_input.send_requested.connect(self._send_prompt)
        self.prompt_input.previous_requested.connect(self._previous_prompt)
        self.prompt_input.next_requested.connect(self._next_prompt)
        self.prompt_input.image_pasted.connect(self._add_clipboard_image)

        self.attachment_bar = QWidget()
        self.attachment_layout = QHBoxLayout(self.attachment_bar)
        self.attachment_layout.setContentsMargins(0, 0, 0, 0)
        self.attachment_layout.setSpacing(6)
        self.attachment_bar.setVisible(False)
        layout.addWidget(self.attachment_bar)
        layout.addWidget(self.prompt_input)

        primary_button_layout = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_prompt)
        primary_button_layout.addWidget(self.send_btn)

        self.screenshot_btn = QPushButton("Screenshot")
        screenshot_menu = QMenu(self.screenshot_btn)
        screenshot_menu.addAction("Capture Map Canvas", self._capture_map_canvas)
        screenshot_menu.addAction("Select Region", self._start_region_capture)
        screenshot_menu.addSeparator()
        screenshot_menu.addAction("Capture QGIS Window", self._capture_qgis_window)
        screenshot_menu.addAction(
            "Select Screen Region", self._start_screen_region_capture
        )
        self.screenshot_btn.setMenu(screenshot_menu)
        primary_button_layout.addWidget(self.screenshot_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setToolTip(
            "Stops GeoAgent at the next checkpoint. The in-flight model "
            "request and any tool already running cannot be aborted "
            "mid-call; cancellation takes effect between steps."
        )
        self.cancel_btn.clicked.connect(self._cancel_running_task)
        primary_button_layout.addWidget(self.cancel_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_transcript)
        primary_button_layout.addWidget(self.clear_btn)

        self.export_md_btn = QPushButton("Export")
        self.export_md_btn.clicked.connect(self._export_transcript_markdown)
        primary_button_layout.addWidget(self.export_md_btn)

        self.import_md_btn = QPushButton("Import")
        self.import_md_btn.clicked.connect(self._import_transcript_markdown)
        primary_button_layout.addWidget(self.import_md_btn)

        self.copy_md_btn = QPushButton("Copy Markdown")
        self.copy_md_btn.setEnabled(False)
        self.copy_md_btn.clicked.connect(self._copy_transcript_markdown)
        primary_button_layout.addWidget(self.copy_md_btn)

        self.copy_script_btn = QPushButton("Copy Script")
        self.copy_script_btn.setEnabled(False)
        self.copy_script_btn.setToolTip(
            "Copy the most recent PyQGIS script executed by GeoAgent."
        )
        self.copy_script_btn.clicked.connect(self._copy_last_pyqgis_script)
        primary_button_layout.addWidget(self.copy_script_btn)
        layout.addLayout(primary_button_layout)

        self.status_label = QLabel("Ready. Ctrl+Enter sends. Up/Down cycles prompts.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _load_settings(self):
        """Load persisted model settings into the dock controls."""
        provider = _setting(self.settings, "provider", DEFAULT_PROVIDER)
        index = self.provider_combo.findText(provider)
        if index < 0:
            index = self.provider_combo.findText(DEFAULT_PROVIDER)
        self.provider_combo.setCurrentIndex(index if index >= 0 else 0)

        model = _setting(self.settings, "model", "")
        if not model:
            model = _default_model_for_provider(self.provider_combo.currentText())
        self.model_input.setText(model)

        self.fast_check.setChecked(_setting(self.settings, "fast_mode", False, bool))
        self.stream_check.setChecked(_setting(self.settings, "stream_chat", True, bool))
        self.auto_approve_tools_check.setChecked(
            _setting(self.settings, "auto_approve_tools", False, bool)
        )
        expanded = _setting(self.settings, "model_section_expanded", True, bool)
        self.model_group.setChecked(expanded)
        self.model_controls.setVisible(expanded)
        jobs_expanded = _setting(self.settings, "jobs_section_expanded", True, bool)
        self.jobs_group.setChecked(jobs_expanded)
        self.jobs_controls.setVisible(jobs_expanded)
        mode = _setting(self.settings, "agent_mode", DEFAULT_AGENT_MODE)
        index = self.agent_mode_combo.findText(mode)
        self.agent_mode_combo.setCurrentIndex(index if index >= 0 else 0)
        profile = _setting(
            self.settings,
            "permission_profile",
            DEFAULT_PERMISSION_PROFILE,
        )
        index = self.permission_combo.findText(profile)
        self.permission_combo.setCurrentIndex(index if index >= 0 else 0)
        self._load_project_history()

    def _save_model_settings(self):
        """Persist the selected provider, model, and fast-mode setting."""
        provider = self.provider_combo.currentText()
        model = self.model_input.text().strip() or _default_model_for_provider(provider)
        if model and not self.model_input.text().strip():
            self.model_input.setText(model)
        self.settings.setValue(f"{SETTINGS_PREFIX}provider", provider)
        self.settings.setValue(f"{SETTINGS_PREFIX}model", model)
        self.settings.setValue(
            f"{SETTINGS_PREFIX}fast_mode", self.fast_check.isChecked()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}stream_chat", self.stream_check.isChecked()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}auto_approve_tools",
            self.auto_approve_tools_check.isChecked(),
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}agent_mode", self.agent_mode_combo.currentText()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}permission_profile", self.permission_combo.currentText()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}model_section_expanded", self.model_group.isChecked()
        )

    def _on_provider_changed(self, provider):
        """Update the model field when the provider changes."""
        self.model_input.setText(_default_model_for_provider(provider))

    def _on_model_section_toggled(self, expanded):
        """Show or hide model controls to keep the dock compact."""
        if hasattr(self, "model_controls"):
            self.model_controls.setVisible(bool(expanded))
        self.settings.setValue(
            f"{SETTINGS_PREFIX}model_section_expanded", bool(expanded)
        )

    def _on_jobs_section_toggled(self, expanded):
        """Show or hide job controls to keep the dock compact."""
        if hasattr(self, "jobs_controls"):
            self.jobs_controls.setVisible(bool(expanded))
        self.settings.setValue(
            f"{SETTINGS_PREFIX}jobs_section_expanded", bool(expanded)
        )

    def _on_agent_mode_changed(self, mode):
        """Load a mode-specific workflow prompt when useful."""
        if not hasattr(self, "prompt_input"):
            return
        prompts = WORKFLOW_PROMPTS.get(mode, [])
        if prompts and not self.prompt_input.toPlainText().strip():
            self.prompt_input.setPlainText(prompts[0])

    def _add_clipboard_image(self, image):
        """Attach an image pasted into the prompt editor."""
        if len(self._image_attachments) >= MAX_IMAGE_ATTACHMENTS:
            QMessageBox.information(
                self,
                "OpenGeoAgent",
                f"Attach at most {MAX_IMAGE_ATTACHMENTS} images per message.",
            )
            return
        try:
            image_bytes, width, height = _image_to_png_bytes(image)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                f"Could not attach clipboard image:\n\n{exc}",
            )
            return
        self._image_attachments.append(
            {
                "bytes": image_bytes,
                "format": "png",
                "width": width,
                "height": height,
            }
        )
        self._render_image_attachments()
        self.status_label.setText(
            f"Attached {len(self._image_attachments)} image"
            f"{'s' if len(self._image_attachments) != 1 else ''}."
        )
        self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _render_image_attachments(self):
        """Refresh the attachment thumbnail strip."""
        while self.attachment_layout.count():
            item = self.attachment_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not self._image_attachments:
            self.attachment_bar.setVisible(False)
            return

        for index, attachment in enumerate(self._image_attachments):
            chip = QWidget()
            chip_layout = QVBoxLayout(chip)
            chip_layout.setContentsMargins(0, 0, 0, 0)
            chip_layout.setSpacing(2)

            thumb = AttachmentThumbnail()
            thumb.setFixedSize(IMAGE_THUMBNAIL_SIZE, IMAGE_THUMBNAIL_SIZE)
            thumb.setStyleSheet("border: 1px solid #BDBDBD; background: #FAFAFA;")
            thumb.setAlignment(_qt_value("AlignmentFlag", "AlignCenter"))
            thumb.setToolTip("Click to preview image")
            thumb.setCursor(_qt_value("CursorShape", "PointingHandCursor"))
            pixmap = QPixmap()
            pixmap.loadFromData(attachment["bytes"], "PNG")
            if not pixmap.isNull():
                keep_aspect = _qt_value("AspectRatioMode", "KeepAspectRatio")
                smooth = _qt_value("TransformationMode", "SmoothTransformation")
                thumb.setPixmap(
                    pixmap.scaled(
                        QSize(IMAGE_THUMBNAIL_SIZE, IMAGE_THUMBNAIL_SIZE),
                        keep_aspect,
                        smooth,
                    )
                )
            thumb.clicked.connect(
                lambda checked=False, i=index: self._preview_image_attachment(i)
            )
            chip_layout.addWidget(thumb)

            remove_btn = QPushButton("Remove")
            remove_btn.setFixedWidth(IMAGE_THUMBNAIL_SIZE)
            remove_btn.clicked.connect(
                lambda checked=False, i=index: self._remove_image_attachment(i)
            )
            chip_layout.addWidget(remove_btn)
            self.attachment_layout.addWidget(chip)

        self.attachment_layout.addStretch(1)
        self.attachment_bar.setVisible(True)

    def _preview_image_attachment(self, index):
        """Show a larger preview of a pending image attachment."""
        if index < 0 or index >= len(self._image_attachments):
            return
        attachment = self._image_attachments[index]
        image_bytes = attachment["bytes"]
        pixmap = QPixmap()
        pixmap.loadFromData(image_bytes, "PNG")
        if pixmap.isNull():
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                "Could not preview this image attachment.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(
            f"Image Preview ({attachment['width']} x {attachment['height']})"
        )
        dialog.resize(900, 700)

        layout = QVBoxLayout(dialog)
        scroll = QScrollArea(dialog)
        scroll.setWidgetResizable(True)
        image_label = QLabel()
        image_label.setAlignment(_qt_value("AlignmentFlag", "AlignCenter"))
        image_label.setContextMenuPolicy(
            _qt_value("ContextMenuPolicy", "CustomContextMenu")
        )

        available = _screen_for_widget(self).availableGeometry()
        max_width = max(320, int(available.width() * 0.8))
        max_height = max(240, int(available.height() * 0.8))
        if pixmap.width() > max_width or pixmap.height() > max_height:
            keep_aspect = _qt_value("AspectRatioMode", "KeepAspectRatio")
            smooth = _qt_value("TransformationMode", "SmoothTransformation")
            pixmap = pixmap.scaled(max_width, max_height, keep_aspect, smooth)
        image_label.setPixmap(pixmap)
        scroll.setWidget(image_label)
        layout.addWidget(scroll)

        def save_image():
            """Save the original PNG attachment bytes."""
            default_name = f"opengeoagent-image-{time.strftime('%Y%m%d-%H%M%S')}.png"
            default_path = os.path.join(os.path.expanduser("~"), default_name)
            path, _selected_filter = QFileDialog.getSaveFileName(
                dialog,
                "Save Image",
                default_path,
                "PNG Images (*.png);;All Files (*)",
            )
            if not path:
                return
            if not os.path.splitext(path)[1]:
                path = f"{path}.png"
            try:
                with open(path, "wb") as f:
                    f.write(image_bytes)
            except Exception as exc:
                QMessageBox.warning(
                    dialog,
                    "OpenGeoAgent",
                    f"Could not save image:\n\n{exc}",
                )
                return
            self.status_label.setText(f"Saved image to {path}")
            self.status_label.setStyleSheet("color: green; font-size: 10px;")

        def show_context_menu(pos):
            """Show preview image actions."""
            menu = QMenu(image_label)
            menu.addAction("Save Image As...", save_image)
            _exec_menu(menu, image_label.mapToGlobal(pos))

        image_label.customContextMenuRequested.connect(show_context_menu)

        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(save_image)
        button_layout.addWidget(save_btn)
        button_layout.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_layout.addWidget(close_btn)
        layout.addLayout(button_layout)
        _exec_dialog(dialog)

    def _remove_image_attachment(self, index):
        """Remove one pending image attachment."""
        if 0 <= index < len(self._image_attachments):
            self._image_attachments.pop(index)
            self._render_image_attachments()

    def _clear_image_attachments(self):
        """Clear all pending image attachments."""
        self._image_attachments = []
        self._render_image_attachments()

    def _map_canvas_widget(self):
        """Return the best QWidget to capture for the current QGIS map canvas."""
        try:
            canvas = self.iface.mapCanvas()
        except Exception:
            canvas = None
        if canvas is None:
            return None
        try:
            viewport = canvas.viewport()
        except Exception:
            viewport = None
        return viewport or canvas

    def _attach_screenshot_pixmap(self, pixmap, label):
        """Attach a captured screenshot pixmap to the pending chat message."""
        if pixmap is None or pixmap.isNull():
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                "Could not capture a screenshot.",
            )
            return
        before = len(self._image_attachments)
        self._add_clipboard_image(pixmap.toImage())
        if len(self._image_attachments) > before:
            self.status_label.setText(f"Attached {label} screenshot.")
            self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _capture_map_canvas(self):
        """Capture the full QGIS map canvas and attach it to the prompt."""
        widget = self._map_canvas_widget()
        if widget is None:
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                "No QGIS map canvas is available to capture.",
            )
            return
        self._attach_screenshot_pixmap(widget.grab(), "map canvas")

    def _capture_qgis_window(self):
        """Capture the QGIS window containing this dock."""
        window = self.window()
        screen = _screen_for_widget(window)
        if screen is None or window is None:
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                "No QGIS window is available to capture.",
            )
            return
        pixmap = window.grab()
        if pixmap.isNull():
            try:
                win_id = int(window.winId())
            except Exception:
                win_id = 0
            pixmap = screen.grabWindow(win_id)
        self._attach_screenshot_pixmap(pixmap, "QGIS window")

    def _start_region_capture(self):
        """Let the user drag a rectangular screenshot region on the map canvas."""
        widget = self._map_canvas_widget()
        if widget is None:
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                "No QGIS map canvas is available to capture.",
            )
            return
        if self._region_capture is not None:
            self._region_capture.stop()
        capture = CanvasRegionCapture(widget, self)
        capture.finished.connect(
            lambda rect, w=widget: self._finish_region_capture(w, rect)
        )
        capture.cancelled.connect(self._cancel_region_capture)
        self._region_capture = capture
        capture.start()
        self.status_label.setText(
            "Drag over the map canvas to attach a screenshot region. "
            "Press Esc to cancel."
        )
        self.status_label.setStyleSheet("color: #1976D2; font-size: 10px;")

    def _finish_region_capture(self, widget, rect):
        """Crop the selected map canvas region and attach it to the prompt."""
        self._region_capture = None
        pixmap = _crop_pixmap(widget.grab(), rect)
        if pixmap.isNull():
            QMessageBox.information(
                self,
                "OpenGeoAgent",
                "Select a larger screenshot region.",
            )
            self.status_label.setText("Screenshot region was too small.")
            self.status_label.setStyleSheet("color: gray; font-size: 10px;")
            return
        self._attach_screenshot_pixmap(pixmap, "map region")

    def _cancel_region_capture(self):
        """Handle cancelled regional screenshot capture."""
        self._region_capture = None
        self.status_label.setText("Screenshot capture cancelled.")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")

    def _start_screen_region_capture(self):
        """Let the user drag a screenshot region anywhere on the current screen."""
        screen = _screen_for_widget(self.window())
        if screen is None:
            QMessageBox.warning(
                self,
                "OpenGeoAgent",
                "No screen is available to capture.",
            )
            return
        if self._screen_region_capture is not None:
            self._screen_region_capture.close()
            self._screen_region_capture = None
        capture = ScreenRegionCapture(screen)
        capture.finished.connect(self._finish_screen_region_capture)
        capture.cancelled.connect(self._cancel_screen_region_capture)
        self._screen_region_capture = capture
        capture.start()
        self.status_label.setText(
            "Drag anywhere on the screen to attach a screenshot region. "
            "Press Esc to cancel."
        )
        self.status_label.setStyleSheet("color: #1976D2; font-size: 10px;")

    def _finish_screen_region_capture(self, global_rect):
        """Attach a selected desktop screenshot region to the prompt."""
        self._screen_region_capture = None
        window = self.window()
        pixmap = _grab_widget_global_rect(window, global_rect)
        if pixmap.isNull():
            screen = _screen_for_widget(window)
            pixmap = _grab_screen_rect(screen, global_rect)
        self._attach_screenshot_pixmap(pixmap, "screen region")

    def _cancel_screen_region_capture(self):
        """Handle cancelled desktop screenshot region capture."""
        if self._screen_region_capture is not None:
            self._screen_region_capture.close()
        self._screen_region_capture = None
        self.status_label.setText("Screen screenshot capture cancelled.")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")

    def _send_prompt(self):
        """Start a chat request for the current prompt."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt and not self._image_attachments:
            return
        if self._worker is not None:
            QMessageBox.information(
                self,
                "OpenGeoAgent",
                "A request is already running. Wait for it to finish first.",
            )
            return

        provider = self.provider_combo.currentText()
        self._save_model_settings()
        _apply_environment_from_settings(self.settings)
        if provider == "openai-codex":
            try:
                from ..oauth import ensure_openai_oauth_environment

                ensure_openai_oauth_environment(
                    self.settings,
                    codex=True,
                )
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "OpenGeoAgent",
                    f"OpenAI OAuth is not ready:\n\n{exc}",
                )
                return

        model_id = self.model_input.text().strip() or _default_model_for_provider(
            provider
        )
        if model_id and not self.model_input.text().strip():
            self.model_input.setText(model_id)
        fast = self.fast_check.isChecked()
        stream = self.stream_check.isChecked()
        auto_approve_tools = self.auto_approve_tools_check.isChecked()
        agent_mode = self.agent_mode_combo.currentText()
        permission_profile = self.permission_combo.currentText()
        if permission_profile == "Trusted auto-approve":
            auto_approve_tools = True
        max_tokens = self.settings.value(f"{SETTINGS_PREFIX}max_tokens", 4096, type=int)
        if not prompt:
            prompt = "Describe the attached image."
        self._record_prompt(prompt)
        prompt_with_context = self._build_prompt_with_context(prompt)
        attachments = [dict(item) for item in self._image_attachments]
        chat_payload = _build_chat_content(prompt_with_context, attachments)

        display_body = None
        if attachments:
            plural = "s" if len(attachments) != 1 else ""
            display_body = (
                f"{prompt}\n\n"
                f"[Attached image{plural} sent with this message: {len(attachments)}. "
                "The image content is not retained in later text-only context.]"
            )
        self._append_message("You", prompt, markdown=False, display_body=display_body)
        self._streaming_message_index = None
        self._streaming_answer = ""
        self.prompt_input.clear()
        self._clear_image_attachments()
        self.status_label.setStyleSheet("color: #1976D2; font-size: 10px;")
        self._start_running_status(
            "Streaming GeoAgent" if stream else "Running GeoAgent"
        )
        self.send_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.cancel_job_btn.setEnabled(True)
        self._active_job_index = self._add_job(
            {
                "prompt": prompt,
                "provider": provider,
                "model": model_id,
                "mode": agent_mode,
                "permission_profile": permission_profile,
                "status": "Running",
                "started_at": time.time(),
                "tools": "",
                "error": "",
            }
        )

        self._worker = ChatWorker(
            self.iface,
            chat_payload,
            provider,
            model_id,
            fast,
            max_tokens,
            auto_approve_tools,
            stream,
            agent_mode,
            permission_profile,
            self,
        )
        self._worker.chunk_received.connect(self._on_worker_chunk)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _build_prompt_with_context(self, prompt):
        """Include recent chat transcript so follow-up turns have context."""
        if not self._messages:
            return prompt

        history_lines = []
        for msg in self._messages[-MAX_CONTEXT_MESSAGES:]:
            body = msg.get("body", "").strip()
            if not body:
                continue
            role = "User" if msg.get("sender") == "You" else "Assistant"
            history_lines.append(f"{role}: {body}")

        if not history_lines:
            return prompt

        history = "\n\n".join(history_lines)
        if len(history) > MAX_CONTEXT_CHARS:
            history = history[-MAX_CONTEXT_CHARS:]
            history = f"[Earlier history truncated]\n{history}"

        return (
            "Use the recent conversation history for context. The current user "
            "request is the authoritative request to answer now.\n\n"
            f"Recent conversation:\n{history}\n\n"
            f"Current user request:\n{prompt}"
        )

    def _select_sample_prompt(self, prompt):
        """Copy the selected sample prompt into the editor."""
        if prompt and prompt != "Sample prompts...":
            self.prompt_input.setPlainText(prompt)
            self.prompt_input.setFocus()

    def _record_prompt(self, prompt):
        """Store a submitted prompt in history."""
        if not self._prompt_history or self._prompt_history[-1] != prompt:
            self._prompt_history.append(prompt)
        self._history_index = None

    def _previous_prompt(self):
        """Load the previous prompt from history."""
        if not self._prompt_history:
            return
        if self._history_index is None:
            self._history_index = len(self._prompt_history) - 1
        else:
            self._history_index = (self._history_index - 1) % len(self._prompt_history)
        self._set_prompt_from_history()

    def _next_prompt(self):
        """Load the next prompt from history."""
        if not self._prompt_history:
            return
        if self._history_index is None:
            self._history_index = 0
        else:
            self._history_index = (self._history_index + 1) % len(self._prompt_history)
        self._set_prompt_from_history()

    def _set_prompt_from_history(self):
        """Set prompt from history."""
        self.prompt_input.setPlainText(self._prompt_history[self._history_index])
        self.prompt_input.setFocus()

    def _on_worker_finished(self, result):
        """Render the completed chat worker result."""
        self._stop_running_status()
        if self._stream_render_timer.isActive():
            self._stream_render_timer.stop()
        if result.get("cancelled_by_user"):
            self._append_message("OpenGeoAgent", "Cancelled by user.", markdown=False)
            self.status_label.setText("Cancelled")
            self.status_label.setStyleSheet("color: gray; font-size: 10px;")
            self._finish_active_job("Cancelled", result)
        elif result.get("success"):
            answer = result.get("answer") or "(No text response.)"
            details = []
            tool_calls = result.get("tool_calls") or []
            pyqgis_script = _latest_pyqgis_script(tool_calls)
            if pyqgis_script:
                self._last_pyqgis_script = pyqgis_script
                self.copy_script_btn.setEnabled(True)
            tool_inputs = _format_tool_calls(tool_calls)
            if tool_inputs:
                details.append(tool_inputs)
            if result.get("tools"):
                details.append(f"Tools: {result['tools']}")
            if result.get("elapsed"):
                details.append(f"Elapsed: {result['elapsed']}")
            if details:
                answer = f"{answer}\n\n" + "\n".join(details)
            if result.get("streamed") and self._streaming_message_index is not None:
                self._update_message(
                    self._streaming_message_index,
                    answer,
                    markdown=True,
                )
            else:
                self._append_message("OpenGeoAgent", answer, markdown=True)
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: gray; font-size: 10px;")
            self._finish_active_job("Succeeded", result)
        else:
            error = result.get("error") or "Unknown error"
            cancelled = result.get("cancelled")
            if cancelled:
                error = f"{error}\nCancelled tools: {cancelled}"
            self._append_message("OpenGeoAgent", f"Error:\n{error}", markdown=False)
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            self._finish_active_job("Failed", result)

        self.send_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_job_btn.setEnabled(False)
        self._worker = None
        self._streaming_message_index = None
        self._streaming_answer = ""
        self._active_job_index = None

    def _on_worker_chunk(self, chunk):
        """Buffer a streamed model text chunk and schedule a debounced render."""
        if not chunk:
            return
        if self._streaming_message_index is None:
            self._streaming_message_index = len(self._messages)
            self._messages.append(
                {"sender": "OpenGeoAgent", "body": "", "markdown": True}
            )
            self.copy_md_btn.setEnabled(True)
        self._streaming_answer += chunk
        if not self._stream_render_timer.isActive():
            self._stream_render_timer.start()

    def _flush_streaming_render(self):
        """Render the latest buffered streaming answer."""
        if self._streaming_message_index is None:
            return
        self._update_message(
            self._streaming_message_index,
            self._streaming_answer,
            markdown=True,
        )

    def _cancel_running_task(self):
        """Request cancellation of the in-flight chat worker."""
        worker = self._worker
        if worker is None:
            return
        worker.requestInterruption()
        self.cancel_btn.setEnabled(False)
        self.cancel_job_btn.setEnabled(False)
        self.status_label.setText(
            "Cancellation requested. Waiting for the current model "
            "call or tool to finish before stopping."
        )
        self.status_label.setStyleSheet("color: #EF6C00; font-size: 10px;")
        self._start_running_status("Cancelling GeoAgent")

    def _start_running_status(self, base_text):
        """Start or update the animated status text."""
        self._status_base_text = base_text
        if self._status_started_at is None:
            self._status_started_at = time.monotonic()
            self._status_frame = 0
        if not self._status_timer.isActive():
            self._status_timer.start()
        self._update_running_status()

    def _stop_running_status(self):
        """Stop the animated status text."""
        if self._status_timer.isActive():
            self._status_timer.stop()
        self._status_started_at = None
        self._status_frame = 0

    def _update_running_status(self):
        """Refresh the animated status text."""
        if self._status_started_at is None:
            return
        elapsed = int(time.monotonic() - self._status_started_at)
        spinner = ("-", "\\", "|", "/")[self._status_frame % 4]
        self._status_frame += 1
        dots = "." * (self._status_frame % 4)
        if elapsed >= 30:
            suffix = "large QGIS operations can take a while"
        elif elapsed >= 10:
            suffix = "running tools and waiting for the model"
        else:
            suffix = "working"
        self.status_label.setText(
            f"{spinner} {self._status_base_text}{dots} {elapsed}s - {suffix}"
        )

    def _append_message(self, sender, message, markdown=False, display_body=None):
        """Append a chat message and refresh the transcript.

        ``body`` holds the canonical prompt or response text used for context
        building. ``display_body`` optionally overrides what is shown in the
        UI and copied transcripts so that ephemeral metadata (such as image
        attachment markers) does not bleed into later conversation history.
        """
        body = message.strip()
        entry = {"sender": sender, "body": body, "markdown": markdown}
        if display_body is not None:
            entry["display_body"] = display_body.strip()
        self._messages.append(entry)
        self.copy_md_btn.setEnabled(bool(self._messages))
        self._render_transcript()
        self._save_project_history()

    def _update_message(self, index, message, markdown=False):
        """Update an existing chat message and refresh the transcript."""
        if index < 0 or index >= len(self._messages):
            return
        self._messages[index]["body"] = message
        self._messages[index]["markdown"] = markdown
        self.copy_md_btn.setEnabled(bool(self._messages))
        self._render_transcript()
        self._save_project_history()

    def _render_transcript(self):
        """Render the stored chat messages as HTML."""
        blocks = []
        for msg in self._messages:
            sender = html.escape(msg["sender"])
            display = msg.get("display_body") or msg["body"]
            if msg["markdown"]:
                body = _markdown_to_basic_html(display)
            else:
                body = f"<p>{_plain_text_to_html(display)}</p>"
            blocks.append(
                "<div style='margin-bottom: 12px;'>"
                f"<p style='font-weight: 600; margin-bottom: 4px;'>{sender}</p>"
                f"{body}"
                "</div>"
            )
        blocks.append("<div style='height: 1em;'>&nbsp;</div>")
        self.transcript.setHtml("\n".join(blocks))
        end_cursor = getattr(getattr(QTextCursor, "MoveOperation", QTextCursor), "End")
        self.transcript.moveCursor(end_cursor)

    def _copy_transcript_markdown(self):
        """Copy the full chat transcript to the clipboard as Markdown."""
        transcript = _conversation_markdown(self._messages)
        if not transcript:
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(transcript)
            self.status_label.setText("Copied chat history as Markdown.")
            self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _copy_last_pyqgis_script(self):
        """Copy the most recent executed PyQGIS script to the clipboard."""
        script = _console_ready_pyqgis_script(self._last_pyqgis_script)
        if not script:
            self.status_label.setText("No PyQGIS script is available to copy.")
            self.status_label.setStyleSheet("color: gray; font-size: 10px;")
            return
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(script)
            self.status_label.setText("Copied PyQGIS script.")
            self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _export_transcript_markdown(self):
        """Save the chat transcript as Markdown."""
        transcript = _conversation_markdown(self._messages)
        if not transcript:
            return
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Chat Transcript",
            "open_geoagent_chat.md",
            "Markdown (*.md);;Text (*.txt)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(transcript)
        self.status_label.setText("Exported chat transcript.")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _import_transcript_markdown(self):
        """Import a Markdown transcript into the current project history."""
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Import Chat Transcript",
            "",
            "Markdown (*.md);;Text (*.txt);;All files (*)",
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            messages = _parse_markdown_transcript(f.read())
        if not messages:
            QMessageBox.information(self, "OpenGeoAgent", "No chat messages found.")
            return
        self._messages = messages
        self.copy_md_btn.setEnabled(True)
        self._render_transcript()
        self._save_project_history()
        self.status_label.setText("Imported chat transcript.")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _clear_transcript(self):
        """Clear all rendered chat messages."""
        self._messages = []
        self._last_pyqgis_script = ""
        self.copy_script_btn.setEnabled(False)
        self.copy_md_btn.setEnabled(False)
        self.transcript.clear()
        self._save_project_history()

    def _load_project_history(self):
        """Load persisted chat messages for the current QGIS project."""
        raw = self.settings.value(self._history_key, "", type=str)
        if not raw:
            return
        try:
            messages = json.loads(raw)
        except (TypeError, ValueError):
            return
        if not isinstance(messages, list):
            return
        self._messages = [
            item
            for item in messages
            if isinstance(item, dict) and item.get("sender") and item.get("body")
        ]
        self.copy_md_btn.setEnabled(bool(self._messages))
        self._render_transcript()

    def _save_project_history(self):
        """Persist chat messages for the current QGIS project."""
        try:
            payload = json.dumps(self._messages[-80:])
            self.settings.setValue(self._history_key, payload)
        except Exception:
            pass

    def _add_job(self, job):
        """Record a submitted chat job and refresh the jobs table."""
        self._jobs.append(dict(job))
        self._render_jobs()
        return len(self._jobs) - 1

    def _finish_active_job(self, status, result):
        """Attach final status details to the active job."""
        if self._active_job_index is None:
            return
        if self._active_job_index < 0 or self._active_job_index >= len(self._jobs):
            return
        job = self._jobs[self._active_job_index]
        job["status"] = status
        job["elapsed"] = result.get("elapsed", "")
        job["tools"] = result.get("tools", "")
        job["tool_calls"] = result.get("tool_calls", [])
        job["error"] = result.get("error", "")
        self._render_jobs()

    def _render_jobs(self):
        """Refresh the compact jobs table."""
        self.jobs_table.setRowCount(len(self._jobs))
        for row, job in enumerate(self._jobs):
            values = [
                job.get("status", ""),
                job.get("mode", ""),
                job.get("prompt", ""),
                _job_status_text(job),
            ]
            for col, value in enumerate(values):
                self.jobs_table.setItem(row, col, QTableWidgetItem(str(value)))
        if self._jobs:
            self.jobs_table.scrollToBottom()
            self.jobs_table.selectRow(len(self._jobs) - 1)

    def _selected_job_index(self):
        """Return the selected job row or None."""
        ranges = self.jobs_table.selectedRanges()
        if not ranges:
            return None
        row = ranges[0].topRow()
        if row < 0 or row >= len(self._jobs):
            return None
        return row

    def _rerun_selected_job(self):
        """Copy a completed job prompt/settings back into the controls and send it."""
        index = self._selected_job_index()
        if index is None:
            return
        job = self._jobs[index]
        self.prompt_input.setPlainText(job.get("prompt", ""))
        for combo, key in (
            (self.provider_combo, "provider"),
            (self.agent_mode_combo, "mode"),
            (self.permission_combo, "permission_profile"),
        ):
            value = job.get(key, "")
            found = combo.findText(value)
            if found >= 0:
                combo.setCurrentIndex(found)
        model = job.get("model", "")
        if model:
            self.model_input.setText(model)
        self._send_prompt()

    def _shutdown_running_state(self):
        """Stop the animated status timer when the dock is dismissed."""
        try:
            self._stop_running_status()
            if self._region_capture is not None:
                self._region_capture.stop()
                self._region_capture = None
            if self._screen_region_capture is not None:
                self._screen_region_capture.close()
                self._screen_region_capture = None
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Failed to stop running status timer during dock shutdown: {exc}",
                "OpenGeoAgent",
                Qgis.MessageLevel.Warning,
            )

    def hideEvent(self, event):
        """Stop the animated status timer when the dock is hidden."""
        self._shutdown_running_state()
        super().hideEvent(event)

    def closeEvent(self, event):
        """Stop the animated status timer when the dock is closed."""
        self._shutdown_running_state()
        super().closeEvent(event)
