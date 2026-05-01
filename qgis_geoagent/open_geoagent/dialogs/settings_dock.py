"""Settings and dependency management for OpenGeoAgent."""

import json
import os
import platform
import sys
import time

from qgis.PyQt.QtCore import Qt, QSettings, QThread, QTimer, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices, QFont, QGuiApplication, QKeySequence
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QKeySequenceEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .chat_dock import (
    DEFAULT_MODELS,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_VOICE_SHORTCUT,
    IMAGE_MODELS,
    PROVIDERS,
    SETTINGS_PREFIX,
    TRANSCRIPTION_MODELS,
    VOICE_SHORTCUT_SETTING,
)
from ..oauth import (
    CODEX_DEFAULT_CONFIG,
    OAUTH_CONFIG_KEYS,
    OPENAI_CODEX_AUTH_EXTRA_PARAMS,
    OPENAI_CODEX_CALLBACK_PATH,
    OPENAI_CODEX_CALLBACK_PORT,
    clear_token_payload,
    store_token_payload,
)

ENV_FALLBACKS = {
    "openai_api_key": ("OPENAI_API_KEY",),
    "openai_org_id": ("OPENAI_ORG_ID",),
    "openai_project_id": ("OPENAI_PROJECT_ID",),
    "anthropic_api_key": ("ANTHROPIC_API_KEY",),
    "gemini_api_key": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "aws_region": ("AWS_REGION", "AWS_DEFAULT_REGION"),
    "ollama_host": ("OLLAMA_HOST",),
    "litellm_api_key": ("LITELLM_API_KEY",),
    "litellm_base_url": ("LITELLM_BASE_URL",),
}


def _apply_environment_from_settings(settings):
    """Apply saved provider credentials to the current process."""
    for key, env_names in ENV_FALLBACKS.items():
        value = settings.value(f"{SETTINGS_PREFIX}{key}", "", type=str).strip()
        if not value:
            value = _env_fallback(*env_names)
        if value:
            for env_name in env_names:
                os.environ[env_name] = value


def _plugin_version(plugin_dir):
    """Read the plugin metadata version."""
    try:
        with open(os.path.join(plugin_dir, "metadata.txt"), "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("version="):
                    return line.split("=", 1)[1].strip()
    except OSError:
        pass
    return "Unknown"


def _geoagent_version():
    """Return the installed GeoAgent package version without importing it."""
    try:
        from importlib import metadata

        return metadata.version("GeoAgent")
    except Exception:
        return "Unknown"


def _model_requires_default_temperature(provider, model_id):
    """Return True when the selected model rejects non-default temperature."""
    normalized = str(model_id or "").lower()
    prefixes = (
        "gpt-5",
        "openai/gpt-5",
        "o1",
        "openai/o1",
        "o3",
        "openai/o3",
        "o4",
        "openai/o4",
    )
    return provider in {"openai", "litellm"} and normalized.startswith(prefixes)


def collect_diagnostics(
    settings,
    plugin_dir,
    latest_install_status="",
    latest_test_status="",
):
    """Return redacted OpenGeoAgent diagnostics as a JSON-friendly dict."""
    from ..deps_manager import (
        check_dependencies,
        dependency_group_names,
        get_venv_dir,
        get_venv_site_packages,
        venv_exists,
    )
    from ..uv_manager import get_uv_path, verify_uv

    try:
        uv_ok, uv_message = verify_uv()
    except Exception as exc:
        uv_ok, uv_message = False, str(exc)
    try:
        from qgis.core import Qgis

        qgis_version = getattr(Qgis, "QGIS_VERSION", "Unknown")
    except Exception:
        qgis_version = "Unknown"

    def _check_dependencies(group_name=None):
        try:
            if group_name is None:
                return check_dependencies()
            return check_dependencies(group_name)
        except TypeError:
            return check_dependencies()

    credential_presence = {}
    for key, env_names in ENV_FALLBACKS.items():
        saved = settings.value(f"{SETTINGS_PREFIX}{key}", "", type=str).strip()
        env_value = _env_fallback(*env_names)
        credential_presence[key] = {
            "saved": bool(saved),
            "environment": bool(env_value),
        }

    return {
        "plugin_version": _plugin_version(plugin_dir),
        "geoagent_version": _geoagent_version(),
        "qgis_version": qgis_version,
        "python": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "venv": {
            "exists": venv_exists(),
            "path": get_venv_dir(),
            "site_packages": get_venv_site_packages(),
        },
        "uv": {
            "path": get_uv_path(),
            "verified": bool(uv_ok),
            "message": uv_message,
        },
        "dependencies": _check_dependencies(),
        "dependency_groups": {
            group: _check_dependencies(group) for group in dependency_group_names()
        },
        "model": {
            "provider": settings.value(
                f"{SETTINGS_PREFIX}provider", DEFAULT_PROVIDER, type=str
            ),
            "model": settings.value(f"{SETTINGS_PREFIX}model", "", type=str),
            "transcription_model": settings.value(
                f"{SETTINGS_PREFIX}transcription_model",
                os.environ.get(
                    "OPENAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL
                ),
                type=str,
            ),
            "image_model": settings.value(
                f"{SETTINGS_PREFIX}image_model",
                os.environ.get("GEOAGENT_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
                type=str,
            ),
            "voice_shortcut": settings.value(
                f"{SETTINGS_PREFIX}{VOICE_SHORTCUT_SETTING}",
                DEFAULT_VOICE_SHORTCUT,
                type=str,
            )
            or DEFAULT_VOICE_SHORTCUT,
            "max_tokens": settings.value(
                f"{SETTINGS_PREFIX}max_tokens", 4096, type=int
            ),
        },
        "credential_presence": credential_presence,
        "latest_install_status": latest_install_status,
        "latest_provider_test_status": latest_test_status,
    }


class ProviderTestWorker(QThread):
    """Run a tiny provider smoke test outside the QGIS UI thread."""

    finished = pyqtSignal(dict)

    def __init__(self, provider, model_id, max_tokens, settings, parent=None):
        super().__init__(parent)
        self.provider = provider
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.settings = settings

    def run(self):
        """Create a minimal no-tool Strands agent and send a short prompt."""
        try:
            _apply_environment_from_settings(self.settings)
            if self.provider == "openai-codex":
                from ..oauth import ensure_openai_oauth_environment

                ensure_openai_oauth_environment(self.settings, codex=True)
            from geoagent import GeoAgentConfig
            from geoagent.core.model import resolve_model
            from strands import Agent

            token_floor = 4096 if self.provider == "ollama" else 1024
            cfg = GeoAgentConfig(
                provider=self.provider,
                model=self.model_id or None,
                temperature=(
                    1
                    if _model_requires_default_temperature(self.provider, self.model_id)
                    else 0
                ),
                max_tokens=max(int(self.max_tokens or token_floor), token_floor),
            )
            model = resolve_model(cfg)
            agent = Agent(
                model=model,
                tools=[],
                system_prompt="You are a provider connectivity test. Reply briefly.",
                callback_handler=None,
            )
            prompt = "Reply with exactly: ok"
            if self.provider == "ollama":
                prompt = "/no_think\nReply with exactly: ok"
            agent(prompt)
            self.finished.emit({"success": True, "message": "Provider test succeeded."})
        except Exception as exc:
            self.finished.emit({"success": False, "message": str(exc)})


def _enum_value(cls, enum_name, member_name):
    """Return an enum member from either scoped or legacy Qt APIs."""
    container = getattr(cls, enum_name, cls)
    return getattr(container, member_name)


def _key_sequence_text(sequence):
    """Return a portable string for a QKeySequence."""
    sequence_format = getattr(QKeySequence, "SequenceFormat", QKeySequence)
    portable = getattr(sequence_format, "PortableText", None)
    if portable is not None:
        return sequence.toString(portable)
    return sequence.toString()


def _single_chord_sequence(sequence):
    """Return a QKeySequence trimmed to its first chord.

    QKeySequenceEdit can record multi-step shortcuts (eg. ``Ctrl+K, Ctrl+C``)
    but the voice toggle matcher only compares a single key press, so any
    trailing chords would be unreachable.
    """
    if sequence.isEmpty() or sequence.count() <= 1:
        return sequence
    return QKeySequence(sequence[0])


def _env_fallback(*env_names):
    """Return the first non-empty environment value from ``env_names``."""
    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            return value
    return ""


class OAuthLoginWorker(QThread):
    """Run an OpenAI OAuth login flow without blocking QGIS."""

    auth_url = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = dict(config)

    def run(self):
        """Open a loopback OAuth flow and exchange the callback code."""
        try:
            from ..oauth import complete_loopback_flow, start_loopback_flow

            is_codex = bool(self.config.get("codex"))
            flow = start_loopback_flow(
                self.config["authorization_url"],
                client_id=self.config["client_id"],
                scope=self.config.get("scope", ""),
                redirect_host="localhost" if is_codex else "127.0.0.1",
                port=OPENAI_CODEX_CALLBACK_PORT if is_codex else 0,
                callback_path=OPENAI_CODEX_CALLBACK_PATH if is_codex else "/callback",
                extra_params=OPENAI_CODEX_AUTH_EXTRA_PARAMS if is_codex else None,
                fallback_port=not is_codex,
            )
            self.auth_url.emit(flow.authorization_url)
            token = complete_loopback_flow(
                flow,
                token_url=self.config["token_url"],
                client_id=self.config["client_id"],
            )
            self.finished.emit({"success": True, "token": token, "error": ""})
        except Exception as exc:
            self.finished.emit({"success": False, "token": {}, "error": str(exc)})


class OAuthRefreshWorker(QThread):
    """Refresh OpenAI OAuth tokens without blocking QGIS."""

    finished = pyqtSignal(dict)

    def __init__(self, config, refresh_token, parent=None):
        super().__init__(parent)
        self.config = dict(config)
        self.refresh_token = refresh_token

    def run(self):
        """Refresh the OAuth token."""
        try:
            from ..oauth import refresh_oauth_token

            token = refresh_oauth_token(
                self.config["token_url"],
                client_id=self.config["client_id"],
                refresh_token=self.refresh_token,
                scope=self.config.get("scope", ""),
            )
            self.finished.emit({"success": True, "token": token, "error": ""})
        except Exception as exc:
            self.finished.emit({"success": False, "token": {}, "error": str(exc)})


class SettingsDockWidget(QDockWidget):
    """Dock widget for configuring OpenGeoAgent."""

    def __init__(self, iface, parent=None):
        super().__init__("OpenGeoAgent Settings", parent)
        self.iface = iface
        self.settings = QSettings()
        self._deps_worker = None
        self._oauth_worker = None
        self._provider_test_worker = None
        self._latest_install_status = ""
        self._latest_provider_test_status = ""

        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(280)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the settings dock widgets and tabs."""
        main_widget = QWidget()
        self.setWidget(main_widget)

        layout = QVBoxLayout(main_widget)
        layout.setSpacing(10)

        header_label = QLabel("OpenGeoAgent Settings")
        header_font = QFont()
        header_font.setPointSize(12)
        header_font.setBold(True)
        header_label.setFont(header_font)
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header_label)

        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        self.tab_widget.addTab(self._create_dependencies_tab(), "Dependencies")
        self.tab_widget.addTab(self._create_model_tab(), "Model")

        button_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Settings")
        self.save_btn.clicked.connect(self._save_settings)
        button_layout.addWidget(self.save_btn)

        self.test_provider_btn = QPushButton("Test Provider")
        self.test_provider_btn.clicked.connect(self._test_provider)
        button_layout.addWidget(self.test_provider_btn)

        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)

        diagnostics_layout = QHBoxLayout()
        self.copy_diagnostics_btn = QPushButton("Copy Diagnostics")
        self.copy_diagnostics_btn.clicked.connect(self._copy_diagnostics)
        diagnostics_layout.addWidget(self.copy_diagnostics_btn)

        self.save_diagnostics_btn = QPushButton("Save Diagnostics")
        self.save_diagnostics_btn.clicked.connect(self._save_diagnostics)
        diagnostics_layout.addWidget(self.save_diagnostics_btn)
        layout.addLayout(diagnostics_layout)

        self.status_label = QLabel("Settings loaded")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _create_dependencies_tab(self):
        """Create the dependency status and installer tab."""
        from ..deps_manager import dependency_group_names

        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Install GeoAgent and provider clients in an isolated environment. "
            "This does not modify the QGIS Python installation."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 10px; padding: 5px;")
        layout.addWidget(info_label)

        group_layout = QFormLayout()
        self.dependency_group_combo = QComboBox()
        self.dependency_group_combo.addItems(dependency_group_names())
        self.dependency_group_combo.currentTextChanged.connect(
            self._refresh_dependency_status
        )
        group_layout.addRow("Dependency set:", self.dependency_group_combo)
        layout.addLayout(group_layout)

        deps_group = QGroupBox("Package Status")
        self.deps_layout = QVBoxLayout(deps_group)
        self.dep_status_labels = {}
        layout.addWidget(deps_group)

        self.deps_overall_label = QLabel("Checking dependencies...")
        self.deps_overall_label.setWordWrap(True)
        self.deps_overall_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.deps_overall_label)

        self.deps_progress_bar = QProgressBar()
        self.deps_progress_bar.setRange(0, 100)
        self.deps_progress_bar.setVisible(False)
        layout.addWidget(self.deps_progress_bar)

        self.deps_progress_label = QLabel("")
        self.deps_progress_label.setWordWrap(True)
        self.deps_progress_label.setStyleSheet("font-size: 10px;")
        self.deps_progress_label.setVisible(False)
        layout.addWidget(self.deps_progress_label)

        self.install_deps_btn = QPushButton("Install Dependencies")
        self.install_deps_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                font-weight: bold;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover { background-color: #1565C0; }
            QPushButton:disabled { background-color: #BDBDBD; }
        """)
        self.install_deps_btn.clicked.connect(self._install_dependencies)
        layout.addWidget(self.install_deps_btn)

        self.refresh_deps_btn = QPushButton("Refresh Status")
        self.refresh_deps_btn.clicked.connect(self._refresh_dependency_status)
        layout.addWidget(self.refresh_deps_btn)

        note_label = QLabel(
            "Packages are installed under ~/.open_geoagent/. Restart QGIS if "
            "new packages are not detected immediately."
        )
        note_label.setWordWrap(True)
        note_label.setStyleSheet("font-size: 9px; font-style: italic;")
        layout.addWidget(note_label)
        layout.addStretch()

        QTimer.singleShot(100, self._refresh_dependency_status)
        return widget

    def _create_model_tab(self):
        """Create the model provider and credential tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        model_group = QGroupBox("Provider")
        form = QFormLayout(model_group)

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
        form.addRow("Provider:", self.provider_combo)

        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Provider default")
        form.addRow("Model:", self.model_input)

        self.fast_check = QCheckBox("Use fast GeoAgent prompt")
        form.addRow("", self.fast_check)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(256, 32768)
        self.max_tokens_spin.setValue(4096)
        self.max_tokens_spin.setSingleStep(256)
        form.addRow("Max tokens:", self.max_tokens_spin)

        layout.addWidget(model_group)

        credentials_group = QGroupBox("Credentials and Hosts")
        credentials_form = QFormLayout(credentials_group)

        password_mode = getattr(getattr(QLineEdit, "EchoMode", QLineEdit), "Password")

        self.openai_key_input = QLineEdit()
        self.openai_key_input.setEchoMode(password_mode)
        credentials_form.addRow("OpenAI API key:", self.openai_key_input)

        self.openai_org_input = QLineEdit()
        self.openai_org_input.setEchoMode(password_mode)
        self.openai_org_input.setPlaceholderText("Optional OpenAI organization ID")
        credentials_form.addRow("OpenAI org ID:", self.openai_org_input)

        self.openai_project_input = QLineEdit()
        self.openai_project_input.setEchoMode(password_mode)
        self.openai_project_input.setPlaceholderText("Optional OpenAI project ID")
        credentials_form.addRow("OpenAI project ID:", self.openai_project_input)

        self.anthropic_key_input = QLineEdit()
        self.anthropic_key_input.setEchoMode(password_mode)
        credentials_form.addRow("Anthropic API key:", self.anthropic_key_input)

        self.gemini_key_input = QLineEdit()
        self.gemini_key_input.setEchoMode(password_mode)
        credentials_form.addRow("Gemini API key:", self.gemini_key_input)

        self.aws_region_input = QLineEdit()
        self.aws_region_input.setPlaceholderText("e.g. us-east-1")
        credentials_form.addRow("AWS region:", self.aws_region_input)

        self.ollama_host_input = QLineEdit()
        self.ollama_host_input.setPlaceholderText("http://127.0.0.1:11434")
        credentials_form.addRow("Ollama host:", self.ollama_host_input)

        self.litellm_key_input = QLineEdit()
        self.litellm_key_input.setEchoMode(password_mode)
        credentials_form.addRow("LiteLLM API key:", self.litellm_key_input)

        self.litellm_base_url_input = QLineEdit()
        self.litellm_base_url_input.setPlaceholderText("https://proxy.example.com")
        credentials_form.addRow("LiteLLM base URL:", self.litellm_base_url_input)

        layout.addWidget(credentials_group)

        image_group = QGroupBox("Image Generation")
        image_form = QFormLayout(image_group)

        self.image_model_combo = QComboBox()
        self.image_model_combo.addItems(IMAGE_MODELS)
        self.image_model_combo.setEditable(True)
        self.image_model_combo.setMinimumContentsLength(18)
        self.image_model_combo.setSizeAdjustPolicy(
            _enum_value(
                QComboBox,
                "SizeAdjustPolicy",
                "AdjustToMinimumContentsLengthWithIcon",
            )
        )
        image_form.addRow("Image model:", self.image_model_combo)

        image_note = QLabel(
            "Direct image generation uses the OpenAI Images API. "
            "`gpt-image-2` is the default; choose `gpt-image-1` when you want "
            "the lower-cost fallback model."
        )
        image_note.setWordWrap(True)
        image_note.setStyleSheet("font-size: 10px; color: gray;")
        image_form.addRow(image_note)
        layout.addWidget(image_group)

        voice_group = QGroupBox("Voice Transcription")
        voice_form = QFormLayout(voice_group)

        self.transcription_model_combo = QComboBox()
        self.transcription_model_combo.addItems(TRANSCRIPTION_MODELS)
        self.transcription_model_combo.setEditable(True)
        self.transcription_model_combo.setMinimumContentsLength(24)
        self.transcription_model_combo.setSizeAdjustPolicy(
            _enum_value(
                QComboBox,
                "SizeAdjustPolicy",
                "AdjustToMinimumContentsLengthWithIcon",
            )
        )
        voice_form.addRow("Transcription model:", self.transcription_model_combo)

        self.voice_shortcut_edit = QKeySequenceEdit()
        voice_form.addRow("Mic shortcut:", self.voice_shortcut_edit)

        voice_note = QLabel(
            "Voice input uses the OpenAI transcription API with the OpenAI API "
            "key above and may incur API costs. ChatGPT/Codex OAuth is not used "
            "for transcription. The shortcut starts and stops recording when the "
            "chat dock has focus."
        )
        voice_note.setWordWrap(True)
        voice_note.setStyleSheet("font-size: 10px; color: gray;")
        voice_form.addRow(voice_note)

        layout.addWidget(voice_group)

        oauth_group = QGroupBox("ChatGPT Login")
        oauth_form = QFormLayout(oauth_group)

        oauth_note = QLabel(
            "Login opens ChatGPT in your browser using the Codex OAuth flow."
        )
        oauth_note.setWordWrap(True)
        oauth_note.setStyleSheet("font-size: 10px; color: gray;")
        oauth_form.addRow(oauth_note)

        oauth_button_layout = QHBoxLayout()
        self.openai_oauth_login_btn = QPushButton("Login with ChatGPT")
        self.openai_oauth_login_btn.clicked.connect(self._login_openai_oauth)
        oauth_button_layout.addWidget(self.openai_oauth_login_btn)

        self.openai_oauth_refresh_btn = QPushButton("Refresh")
        self.openai_oauth_refresh_btn.clicked.connect(self._refresh_openai_oauth)
        oauth_button_layout.addWidget(self.openai_oauth_refresh_btn)

        self.openai_oauth_logout_btn = QPushButton("Logout")
        self.openai_oauth_logout_btn.clicked.connect(self._logout_openai_oauth)
        oauth_button_layout.addWidget(self.openai_oauth_logout_btn)
        oauth_form.addRow("", oauth_button_layout)

        self.openai_oauth_status_label = QLabel("Not logged in")
        self.openai_oauth_status_label.setWordWrap(True)
        self.openai_oauth_status_label.setStyleSheet("font-size: 10px; color: gray;")
        oauth_form.addRow("Status:", self.openai_oauth_status_label)

        layout.addWidget(oauth_group)

        note = QLabel(
            "Credential values are saved in QGIS settings and applied to the "
            "current QGIS process when a chat request runs. ChatGPT login tokens "
            "are stored in QGIS Auth Manager."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(note)
        layout.addStretch()
        return widget

    def _refresh_dependency_status(self, *_args):
        """Refresh dependency labels from the dependency checker."""
        from ..deps_manager import check_dependencies, packages_for_group

        group_name = self.dependency_group_combo.currentText()
        packages = packages_for_group(group_name)
        if set(self.dep_status_labels) != {name for name, _ in packages}:
            self._rebuild_dependency_rows(packages)

        deps = check_dependencies(group_name)
        all_ok = True

        for dep in deps:
            label = self.dep_status_labels.get(dep["name"])
            if label is None:
                continue
            if dep["installed"]:
                version_str = dep["version"] or "installed"
                label.setText(f"Installed ({version_str})")
                label.setStyleSheet("color: green; font-weight: bold;")
            else:
                label.setText("Not installed")
                label.setStyleSheet("color: red;")
                all_ok = False

        if all_ok:
            self.deps_overall_label.setText(
                f"All {group_name} dependencies are installed."
            )
            self.deps_overall_label.setStyleSheet(
                "color: green; font-weight: bold; padding: 5px;"
            )
            self.install_deps_btn.setVisible(False)
        else:
            missing_count = sum(1 for d in deps if not d["installed"])
            self.deps_overall_label.setText(
                f"{missing_count} {group_name} package(s) missing. "
                "Click Install Dependencies."
            )
            self.deps_overall_label.setStyleSheet(
                "color: #E65100; font-weight: bold; padding: 5px;"
            )
            self.install_deps_btn.setVisible(True)
            self.install_deps_btn.setEnabled(True)

    def _rebuild_dependency_rows(self, packages):
        """Rebuild package status rows for the selected dependency group."""
        while self.deps_layout.count():
            item = self.deps_layout.takeAt(0)
            child_layout = item.layout()
            widget = item.widget()
            if child_layout is not None:
                while child_layout.count():
                    child = child_layout.takeAt(0)
                    child_widget = child.widget()
                    if child_widget is not None:
                        child_widget.deleteLater()
            if widget is not None:
                widget.deleteLater()

        self.dep_status_labels = {}
        for import_name, pip_name in packages:
            row_layout = QHBoxLayout()
            name_label = QLabel(f"  {pip_name}")
            name_label.setWordWrap(True)
            name_label.setMinimumWidth(120)
            name_label.setMaximumWidth(190)
            status_label = QLabel("Checking...")
            status_label.setStyleSheet("color: gray;")
            row_layout.addWidget(name_label)
            row_layout.addWidget(status_label)
            row_layout.addStretch()
            self.deps_layout.addLayout(row_layout)
            self.dep_status_labels[import_name] = status_label

    def _install_dependencies(self):
        """Start background dependency installation."""
        from ..deps_manager import DepsInstallWorker

        group_name = self.dependency_group_combo.currentText()
        self.install_deps_btn.setEnabled(False)
        self.install_deps_btn.setText("Installing...")
        self.refresh_deps_btn.setEnabled(False)

        self.deps_progress_bar.setVisible(True)
        self.deps_progress_bar.setValue(0)
        self.deps_progress_label.setVisible(True)
        self.deps_progress_label.setText("Starting installation...")

        self._deps_worker = DepsInstallWorker(group_name)
        self._deps_worker.progress.connect(self._on_deps_install_progress)
        self._deps_worker.finished.connect(self._on_deps_install_finished)
        self._deps_worker.start()

    def _on_deps_install_progress(self, percent, message):
        """Handle deps install progress."""
        self.deps_progress_bar.setValue(percent)
        self.deps_progress_label.setText(message)

    def _on_deps_install_finished(self, success, message):
        """Handle deps install finished."""
        self.deps_progress_bar.setVisible(False)
        self.deps_progress_label.setVisible(False)
        self.install_deps_btn.setText("Install Dependencies")
        self.refresh_deps_btn.setEnabled(True)

        if success:
            self._latest_install_status = message
            self.deps_overall_label.setText(message)
            self.deps_overall_label.setStyleSheet(
                "color: green; font-weight: bold; padding: 5px;"
            )
            self.iface.messageBar().pushSuccess(
                "OpenGeoAgent", "Dependencies installed successfully."
            )
            self._refresh_dependency_status()
            QMessageBox.information(
                self,
                "Dependencies Installed",
                "Dependencies have been installed successfully.\n\n"
                "Restart QGIS if the plugin cannot import them immediately.",
            )
        else:
            self._latest_install_status = message
            self.deps_overall_label.setText("Installation failed.")
            self.deps_overall_label.setStyleSheet(
                "color: red; font-weight: bold; padding: 5px;"
            )
            self.install_deps_btn.setEnabled(True)
            QMessageBox.critical(
                self,
                "Installation Failed",
                f"Failed to install dependencies:\n\n{message}\n\n"
                'Manual fallback: pip install "GeoAgent[providers]>=1.4.0"',
            )

        self._deps_worker = None

    def show_dependencies_tab(self):
        """Switch the settings dock to the dependencies tab."""
        self.tab_widget.setCurrentIndex(0)

    def _oauth_config(self):
        """Return the built-in ChatGPT/Codex login settings."""
        config = dict(CODEX_DEFAULT_CONFIG)
        config["codex"] = True
        return config

    def _set_oauth_buttons_enabled(self, enabled):
        """Enable or disable OAuth action buttons."""
        self.openai_oauth_login_btn.setEnabled(enabled)
        self.openai_oauth_refresh_btn.setEnabled(enabled)
        self.openai_oauth_logout_btn.setEnabled(enabled)

    def _login_openai_oauth(self):
        """Start OpenAI OAuth login."""
        if self._oauth_worker is not None:
            return
        try:
            config = self._oauth_config()
        except Exception as exc:
            QMessageBox.warning(self, "ChatGPT Login", str(exc))
            return

        self.openai_oauth_status_label.setText("Waiting for browser login...")
        self.openai_oauth_status_label.setStyleSheet("font-size: 10px; color: #1976D2;")
        self._set_oauth_buttons_enabled(False)
        self._oauth_worker = OAuthLoginWorker(config, self)
        self._oauth_worker.auth_url.connect(self._open_oauth_browser)
        self._oauth_worker.finished.connect(self._on_oauth_worker_finished)
        self._oauth_worker.start()

    def _refresh_openai_oauth(self):
        """Refresh the stored OpenAI OAuth token."""
        if self._oauth_worker is not None:
            return
        try:
            config = self._oauth_config()
            from ..oauth import load_token_payload

            payload = load_token_payload(self.settings)
            refresh_token = str(payload.get("refresh_token", "")).strip()
            if not refresh_token:
                raise ValueError("No refresh token is stored. Login again.")
        except Exception as exc:
            QMessageBox.warning(self, "ChatGPT Login", str(exc))
            return

        self.openai_oauth_status_label.setText("Refreshing token...")
        self.openai_oauth_status_label.setStyleSheet("font-size: 10px; color: #1976D2;")
        self._set_oauth_buttons_enabled(False)
        self._oauth_worker = OAuthRefreshWorker(config, refresh_token, self)
        self._oauth_worker.finished.connect(self._on_oauth_worker_finished)
        self._oauth_worker.start()

    def _logout_openai_oauth(self):
        """Clear stored OpenAI OAuth tokens."""
        try:
            clear_token_payload(self.settings)
        except Exception as exc:
            QMessageBox.warning(self, "ChatGPT Login", str(exc))
            return
        self._refresh_oauth_status()
        self.iface.messageBar().pushSuccess("OpenGeoAgent", "ChatGPT logged out.")

    def _open_oauth_browser(self, url):
        """Open the OAuth authorization URL in the user's browser."""
        QDesktopServices.openUrl(QUrl(url))

    def _on_oauth_worker_finished(self, result):
        """Persist OAuth tokens from the login or refresh worker."""
        self._set_oauth_buttons_enabled(True)
        self._oauth_worker = None
        if not result.get("success"):
            self.openai_oauth_status_label.setText("Login failed")
            self.openai_oauth_status_label.setStyleSheet("font-size: 10px; color: red;")
            QMessageBox.critical(self, "ChatGPT Login", result.get("error", "Failed"))
            return
        try:
            store_token_payload(self.settings, result["token"])
            index = self.provider_combo.findText("openai-codex")
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)
                self.settings.setValue(f"{SETTINGS_PREFIX}provider", "openai-codex")
                model = self.model_input.text().strip() or DEFAULT_MODELS.get(
                    "openai-codex", ""
                )
                if model:
                    self.model_input.setText(model)
                    self.settings.setValue(f"{SETTINGS_PREFIX}model", model)
        except Exception as exc:
            self.openai_oauth_status_label.setText("Token storage failed")
            self.openai_oauth_status_label.setStyleSheet("font-size: 10px; color: red;")
            QMessageBox.critical(self, "ChatGPT Login", str(exc))
            return
        self._refresh_oauth_status()
        self.iface.messageBar().pushSuccess("OpenGeoAgent", "ChatGPT connected.")

    def _refresh_oauth_status(self):
        """Update the OAuth login status label."""
        authcfg = self.settings.value(f"{SETTINGS_PREFIX}openai_oauth_authcfg", "")
        expires_at = self.settings.value(
            f"{SETTINGS_PREFIX}openai_oauth_expires_at", "", type=str
        )
        if not str(authcfg).strip():
            self.openai_oauth_status_label.setText("Not logged in")
            self.openai_oauth_status_label.setStyleSheet(
                "font-size: 10px; color: gray;"
            )
            return
        if expires_at:
            try:
                expiry = time.strftime(
                    "%Y-%m-%d %H:%M:%S",
                    time.localtime(float(expires_at)),
                )
                text = f"Logged in. Access token expires at {expiry}."
            except (TypeError, ValueError):
                text = "Logged in. Access token expiry is unknown."
        else:
            text = "Logged in. Access token expiry is unknown."
        self.openai_oauth_status_label.setText(text)
        self.openai_oauth_status_label.setStyleSheet("font-size: 10px; color: green;")

    def _on_provider_changed(self, provider):
        """Update the model field when the provider changes."""
        self.model_input.setText(DEFAULT_MODELS.get(provider, ""))

    def _load_settings(self):
        """Load persisted settings into the form fields."""
        provider = self.settings.value(
            f"{SETTINGS_PREFIX}provider", DEFAULT_PROVIDER, type=str
        )
        index = self.provider_combo.findText(provider)
        if index < 0:
            index = self.provider_combo.findText(DEFAULT_PROVIDER)
        self.provider_combo.setCurrentIndex(index if index >= 0 else 0)

        model = self.settings.value(f"{SETTINGS_PREFIX}model", "", type=str)
        self.model_input.setText(model or DEFAULT_MODELS.get(provider, ""))
        self.fast_check.setChecked(
            self.settings.value(f"{SETTINGS_PREFIX}fast_mode", False, type=bool)
        )
        self.max_tokens_spin.setValue(
            self.settings.value(f"{SETTINGS_PREFIX}max_tokens", 4096, type=int)
        )
        transcription_model = (
            self.settings.value(
                f"{SETTINGS_PREFIX}transcription_model",
                os.environ.get(
                    "OPENAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL
                ),
                type=str,
            ).strip()
            or DEFAULT_TRANSCRIPTION_MODEL
        )
        if self.transcription_model_combo.findText(transcription_model) < 0:
            self.transcription_model_combo.addItem(transcription_model)
        self.transcription_model_combo.setCurrentText(transcription_model)
        image_model = (
            self.settings.value(
                f"{SETTINGS_PREFIX}image_model",
                os.environ.get("GEOAGENT_IMAGE_MODEL", DEFAULT_IMAGE_MODEL),
                type=str,
            ).strip()
            or DEFAULT_IMAGE_MODEL
        )
        if self.image_model_combo.findText(image_model) < 0:
            self.image_model_combo.addItem(image_model)
        self.image_model_combo.setCurrentText(image_model)
        voice_shortcut = (
            self.settings.value(
                f"{SETTINGS_PREFIX}{VOICE_SHORTCUT_SETTING}",
                DEFAULT_VOICE_SHORTCUT,
                type=str,
            ).strip()
            or DEFAULT_VOICE_SHORTCUT
        )
        self.voice_shortcut_edit.setKeySequence(QKeySequence(voice_shortcut))

        self._credential_inputs = (
            ("openai_api_key", self.openai_key_input),
            ("openai_org_id", self.openai_org_input),
            ("openai_project_id", self.openai_project_input),
            ("anthropic_api_key", self.anthropic_key_input),
            ("gemini_api_key", self.gemini_key_input),
            ("aws_region", self.aws_region_input),
            ("ollama_host", self.ollama_host_input),
            ("litellm_api_key", self.litellm_key_input),
            ("litellm_base_url", self.litellm_base_url_input),
        )
        self._env_sourced_credentials = {}
        for key, widget in self._credential_inputs:
            value, from_env = self._credential_value(key)
            widget.setText(value)
            if from_env:
                self._env_sourced_credentials[key] = value

        self._refresh_oauth_status()

    def _credential_value(self, key):
        """Return ``(value, from_env)`` for a credential field.

        Looks up ``key`` in QSettings first; if no saved value exists, falls
        back to the configured environment variables. ``from_env`` is True
        only when the returned value originated from the environment.
        """
        saved = self.settings.value(f"{SETTINGS_PREFIX}{key}", "", type=str)
        if str(saved).strip():
            return str(saved), False
        fallback = _env_fallback(*ENV_FALLBACKS.get(key, ()))
        return fallback, bool(fallback)

    def _save_settings(self):
        """Persist settings from the form fields."""
        self.settings.setValue(
            f"{SETTINGS_PREFIX}provider", self.provider_combo.currentText()
        )
        self.settings.setValue(f"{SETTINGS_PREFIX}model", self.model_input.text())
        self.settings.setValue(
            f"{SETTINGS_PREFIX}fast_mode", self.fast_check.isChecked()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}max_tokens", self.max_tokens_spin.value()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}transcription_model",
            self.transcription_model_combo.currentText().strip()
            or DEFAULT_TRANSCRIPTION_MODEL,
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}image_model",
            self.image_model_combo.currentText().strip() or DEFAULT_IMAGE_MODEL,
        )
        sequence = _single_chord_sequence(self.voice_shortcut_edit.keySequence())
        if sequence != self.voice_shortcut_edit.keySequence():
            self.voice_shortcut_edit.setKeySequence(sequence)
        voice_shortcut = _key_sequence_text(sequence)
        self.settings.setValue(
            f"{SETTINGS_PREFIX}{VOICE_SHORTCUT_SETTING}",
            voice_shortcut.strip() or DEFAULT_VOICE_SHORTCUT,
        )
        for key, widget in getattr(self, "_credential_inputs", ()):
            current = widget.text()
            env_value = self._env_sourced_credentials.get(key)
            if env_value is not None and current == env_value:
                # Field was pre-filled from an environment variable and the
                # user did not change it; skip persisting the env-sourced
                # secret to QSettings.
                continue
            self.settings.setValue(f"{SETTINGS_PREFIX}{key}", current)
        self.status_label.setText("Settings saved")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")
        self.iface.messageBar().pushSuccess("OpenGeoAgent", "Settings saved.")

    def _test_provider(self):
        """Run a tiny provider smoke test in a background worker."""
        if self._provider_test_worker is not None:
            return
        self._save_settings()
        provider = self.provider_combo.currentText()
        model_id = self.model_input.text().strip() or DEFAULT_MODELS.get(provider, "")
        self.test_provider_btn.setEnabled(False)
        self.status_label.setText("Testing provider...")
        self.status_label.setStyleSheet("color: #1976D2; font-size: 10px;")
        self._provider_test_worker = ProviderTestWorker(
            provider,
            model_id,
            self.max_tokens_spin.value(),
            self.settings,
            self,
        )
        self._provider_test_worker.finished.connect(self._on_provider_test_finished)
        self._provider_test_worker.start()

    def _on_provider_test_finished(self, result):
        """Display provider smoke-test result."""
        self.test_provider_btn.setEnabled(True)
        self._provider_test_worker = None
        message = result.get("message", "")
        self._latest_provider_test_status = message
        if result.get("success"):
            self.status_label.setText(message)
            self.status_label.setStyleSheet("color: green; font-size: 10px;")
            self.iface.messageBar().pushSuccess("OpenGeoAgent", message)
        else:
            self.status_label.setText("Provider test failed")
            self.status_label.setStyleSheet("color: red; font-size: 10px;")
            QMessageBox.critical(self, "Provider Test Failed", message)

    def _diagnostics_text(self):
        """Return redacted diagnostics as pretty JSON."""
        payload = collect_diagnostics(
            self.settings,
            os.path.dirname(os.path.dirname(__file__)),
            self._latest_install_status,
            self._latest_provider_test_status,
        )
        return json.dumps(payload, indent=2, sort_keys=True)

    def _copy_diagnostics(self):
        """Copy redacted diagnostics to the clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._diagnostics_text())
        self.status_label.setText("Copied diagnostics.")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _save_diagnostics(self):
        """Save redacted diagnostics to a JSON file."""
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save OpenGeoAgent Diagnostics",
            "open_geoagent_diagnostics.json",
            "JSON (*.json);;Text (*.txt)",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self._diagnostics_text())
        self.status_label.setText("Saved diagnostics.")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")

    def _reset_defaults(self):
        """Reset saved model settings after user confirmation."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Reset OpenGeoAgent settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            clear_token_payload(self.settings)
        except Exception as exc:
            QMessageBox.warning(self, "ChatGPT Login", str(exc))
            return

        for key in [
            "provider",
            "model",
            "fast_mode",
            "max_tokens",
            "transcription_model",
            "image_model",
            VOICE_SHORTCUT_SETTING,
            "voice_shortcut",
            "openai_api_key",
            "openai_org_id",
            "openai_project_id",
            "anthropic_api_key",
            "gemini_api_key",
            "aws_region",
            "ollama_host",
            "litellm_api_key",
            "litellm_base_url",
            *OAUTH_CONFIG_KEYS,
            "openai_oauth_authcfg",
            "openai_oauth_expires_at",
            "openai_oauth_token_type",
        ]:
            self.settings.remove(f"{SETTINGS_PREFIX}{key}")
        self._load_settings()
        self.status_label.setText("Defaults restored")
        self.status_label.setStyleSheet("color: #E65100; font-size: 10px;")
