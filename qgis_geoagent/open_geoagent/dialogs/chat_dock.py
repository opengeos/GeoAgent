"""Dockable OpenGeoAgent chat interface."""

import os
import html
import json
import re
import time
import traceback

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt.QtCore import Qt, QSettings, QThread, QTimer, pyqtSignal
from qgis.PyQt.QtGui import QGuiApplication, QTextCursor
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSizePolicy,
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
    "bedrock",
    "openai",
    "openai-codex",
    "anthropic",
    "gemini",
    "ollama",
    "litellm",
]
MAX_CONTEXT_MESSAGES = 12
MAX_CONTEXT_CHARS = 12000
SAMPLE_PROMPTS = [
    "Summarize the current QGIS project layers, CRS, extents, and feature counts.",
    "Zoom to the active layer and describe what it contains.",
    "List visible layers and identify any layers with no features or invalid data sources.",
    "Add an OpenStreetMap basemap and zoom to the project extent.",
    "Inspect the active vector layer fields and suggest useful styling or labeling options.",
    "Select features in the active layer where population is greater than 100000, then zoom to the selected features.",
    "Run a buffer around the active layer by 1000 meters and add the output to the project.",
    "Find a WhiteboxTools command for calculating slope from the active DEM layer.",
    "Run a WhiteboxTools hillshade analysis on a local DEM and add the result to QGIS.",
    "Search WhiteboxTools for watershed or flow accumulation tools.",
    "Create a concise map QA checklist for this project before I export it.",
]


def _default_model_for_provider(provider):
    """Return the UI default model id for a provider."""
    return DEFAULT_MODELS.get(provider, "")


def _setting(settings, key, default="", value_type=str):
    """Read a plugin setting value."""
    return settings.value(f"{SETTINGS_PREFIX}{key}", default, type=value_type)


def _apply_environment_from_settings(settings):
    """Apply provider credentials from QSettings to the current QGIS process."""
    env_map = {
        "openai_api_key": "OPENAI_API_KEY",
        "anthropic_api_key": "ANTHROPIC_API_KEY",
        "gemini_api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "aws_region": "AWS_REGION",
        "ollama_host": "OLLAMA_HOST",
        "litellm_api_key": "LITELLM_API_KEY",
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


def _conversation_markdown(messages):
    """Return the full chat transcript as Markdown."""
    blocks = []
    for msg in messages:
        sender = str(msg.get("sender") or "").strip()
        body = str(msg.get("body") or "").strip()
        if not sender or not body:
            continue
        blocks.append(f"## {sender}\n\n{body}")
    return "\n\n".join(blocks)


class PromptTextEdit(QPlainTextEdit):
    """Prompt editor with chat-friendly keyboard shortcuts."""

    send_requested = pyqtSignal()
    previous_requested = pyqtSignal()
    next_requested = pyqtSignal()

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


class ChatWorker(QThread):
    """Run GeoAgent chat without blocking the QGIS UI."""

    finished = pyqtSignal(dict)

    def __init__(
        self,
        iface,
        prompt,
        provider,
        model_id,
        fast,
        max_tokens,
        auto_approve_tools=False,
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

    def run(self):
        """Create a GeoAgent QGIS agent and execute one chat turn."""
        try:
            from geoagent import GeoAgentConfig
            from geoagent import for_whitebox

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
            agent = for_whitebox(
                self.iface,
                project=project,
                config=config,
                fast=self.fast,
                confirm=self._confirm_tool,
            )
            response = agent.chat(self.prompt)

            self.finished.emit(
                {
                    "success": bool(response.success),
                    "answer": response.answer_text or "",
                    "error": response.error_message or "",
                    "tools": ", ".join(response.executed_tools or []),
                    "tool_calls": response.tool_calls or [],
                    "cancelled": ", ".join(response.cancelled_tools or []),
                    "elapsed": f"{response.execution_time:.2f}s",
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
                }
            )

    def _confirm_tool(self, request):
        """Ask the QGIS user before running confirmation-required tools."""
        if self.auto_approve_tools:
            return True

        from geoagent.tools._qt_marshal import run_on_qt_gui_thread

        def _ask():
            parent = None
            try:
                parent = self.iface.mainWindow()
            except Exception as exc:
                QgsMessageLog.logMessage(
                    f"Could not resolve QGIS main window for confirmation dialog: {exc}",
                    "OpenGeoAgent",
                    Qgis.Warning,
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
        self._prompt_history = []
        self._history_index = None
        self._messages = []
        self._status_started_at = None
        self._status_base_text = "Running"
        self._status_frame = 0
        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._update_running_status)

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(280)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the chat dock widgets and signal connections."""
        main_widget = QWidget()
        self.setWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(8)

        model_group = QGroupBox("Model")
        model_layout = QFormLayout(model_group)

        self.provider_combo = QComboBox()
        self.provider_combo.addItems(PROVIDERS)
        self.provider_combo.setMinimumContentsLength(10)
        self.provider_combo.setSizeAdjustPolicy(
            _enum_value(
                QComboBox,
                "SizeAdjustPolicy",
                "AdjustToMinimumContentsLengthWithIcon",
            )
        )
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        model_layout.addRow("Provider:", self.provider_combo)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Use provider default")
        model_layout.addRow("Model:", self.model_input)

        self.fast_check = QCheckBox("Fast mode")
        self.auto_approve_tools_check = QCheckBox("Auto approve running tools")
        self.auto_approve_tools_check.setToolTip(
            "Run confirmation-required tools without prompting for this session."
        )
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(self.fast_check)
        mode_layout.addWidget(self.auto_approve_tools_check)
        mode_layout.addStretch(1)
        model_layout.addRow("", mode_layout)

        layout.addWidget(model_group)

        sample_layout = QHBoxLayout()
        self.sample_combo = QComboBox()
        self.sample_combo.addItem("Sample prompts...")
        self.sample_combo.addItems(SAMPLE_PROMPTS)
        self.sample_combo.setMinimumContentsLength(22)
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
        layout.addWidget(self.prompt_input)

        primary_button_layout = QHBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self._send_prompt)
        primary_button_layout.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_transcript)
        primary_button_layout.addWidget(self.clear_btn)

        self.copy_md_btn = QPushButton("Copy Markdown")
        self.copy_md_btn.setEnabled(False)
        self.copy_md_btn.clicked.connect(self._copy_transcript_markdown)
        primary_button_layout.addWidget(self.copy_md_btn)
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
        self.auto_approve_tools_check.setChecked(
            _setting(self.settings, "auto_approve_tools", False, bool)
        )

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
            f"{SETTINGS_PREFIX}auto_approve_tools",
            self.auto_approve_tools_check.isChecked(),
        )

    def _on_provider_changed(self, provider):
        """Update the model field when the provider changes."""
        self.model_input.setText(_default_model_for_provider(provider))

    def _send_prompt(self):
        """Start a chat request for the current prompt."""
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
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

        self._record_prompt(prompt)
        model_id = self.model_input.text().strip() or _default_model_for_provider(
            provider
        )
        if model_id and not self.model_input.text().strip():
            self.model_input.setText(model_id)
        fast = self.fast_check.isChecked()
        auto_approve_tools = self.auto_approve_tools_check.isChecked()
        max_tokens = self.settings.value(f"{SETTINGS_PREFIX}max_tokens", 4096, type=int)
        prompt_with_context = self._build_prompt_with_context(prompt)

        self._append_message("You", prompt, markdown=False)
        self.prompt_input.clear()
        self.status_label.setStyleSheet("color: #1976D2; font-size: 10px;")
        self._start_running_status("Running GeoAgent")
        self.send_btn.setEnabled(False)

        self._worker = ChatWorker(
            self.iface,
            prompt_with_context,
            provider,
            model_id,
            fast,
            max_tokens,
            auto_approve_tools,
            self,
        )
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
        if result.get("success"):
            answer = result.get("answer") or "(No text response.)"
            details = []
            tool_inputs = _format_tool_calls(result.get("tool_calls") or [])
            if tool_inputs:
                details.append(tool_inputs)
            if result.get("tools"):
                details.append(f"Tools: {result['tools']}")
            if result.get("elapsed"):
                details.append(f"Elapsed: {result['elapsed']}")
            if details:
                answer = f"{answer}\n\n" + "\n".join(details)
            self._append_message("OpenGeoAgent", answer, markdown=True)
            self.status_label.setText("Ready")
            self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        else:
            error = result.get("error") or "Unknown error"
            cancelled = result.get("cancelled")
            if cancelled:
                error = f"{error}\nCancelled tools: {cancelled}"
            self._append_message("OpenGeoAgent", f"Error:\n{error}", markdown=False)
            self.status_label.setText("Error")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")

        self.send_btn.setEnabled(True)
        self._worker = None

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

    def _append_message(self, sender, message, markdown=False):
        """Append a chat message and refresh the transcript."""
        body = message.strip()
        self._messages.append({"sender": sender, "body": body, "markdown": markdown})
        self.copy_md_btn.setEnabled(bool(self._messages))
        self._render_transcript()

    def _render_transcript(self):
        """Render the stored chat messages as HTML."""
        blocks = []
        for msg in self._messages:
            sender = html.escape(msg["sender"])
            if msg["markdown"]:
                body = _markdown_to_basic_html(msg["body"])
            else:
                body = f"<p>{_plain_text_to_html(msg['body'])}</p>"
            blocks.append(
                "<div style='margin-bottom: 12px;'>"
                f"<p style='font-weight: 600; margin-bottom: 4px;'>{sender}</p>"
                f"{body}"
                "</div>"
            )
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

    def _clear_transcript(self):
        """Clear all rendered chat messages."""
        self._messages = []
        self.copy_md_btn.setEnabled(False)
        self.transcript.clear()

    def _shutdown_running_state(self):
        """Stop the animated status timer when the dock is dismissed."""
        try:
            self._stop_running_status()
        except Exception as exc:
            QgsMessageLog.logMessage(
                f"Failed to stop running status timer during dock shutdown: {exc}",
                "OpenGeoAgent",
                Qgis.Warning,
            )

    def hideEvent(self, event):
        """Stop the animated status timer when the dock is hidden."""
        self._shutdown_running_state()
        super().hideEvent(event)

    def closeEvent(self, event):
        """Stop the animated status timer when the dock is closed."""
        self._shutdown_running_state()
        super().closeEvent(event)
