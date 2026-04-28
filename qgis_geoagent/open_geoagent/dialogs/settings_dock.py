"""Settings and dependency management for OpenGeoAgent."""

from qgis.PyQt.QtCore import Qt, QSettings, QTimer
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
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from qgis.PyQt.QtGui import QFont

from .chat_dock import DEFAULT_MODELS, PROVIDERS, SETTINGS_PREFIX


def _enum_value(cls, enum_name, member_name):
    """Return an enum member from either scoped or legacy Qt APIs."""
    container = getattr(cls, enum_name, cls)
    return getattr(container, member_name)


class SettingsDockWidget(QDockWidget):
    """Dock widget for configuring OpenGeoAgent."""

    def __init__(self, iface, parent=None):
        super().__init__("OpenGeoAgent Settings", parent)
        self.iface = iface
        self.settings = QSettings()
        self._deps_worker = None

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

        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.clicked.connect(self._reset_defaults)
        button_layout.addWidget(self.reset_btn)
        layout.addLayout(button_layout)

        self.status_label = QLabel("Settings loaded")
        self.status_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.status_label)

    def _create_dependencies_tab(self):
        """Create the dependency status and installer tab."""
        from ..deps_manager import REQUIRED_PACKAGES

        widget = QWidget()
        layout = QVBoxLayout(widget)

        info_label = QLabel(
            "Install GeoAgent and provider clients in an isolated environment. "
            "This does not modify the QGIS Python installation."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 10px; padding: 5px;")
        layout.addWidget(info_label)

        deps_group = QGroupBox("Package Status")
        deps_layout = QVBoxLayout(deps_group)

        self.dep_status_labels = {}
        for import_name, pip_name in REQUIRED_PACKAGES:
            row_layout = QHBoxLayout()
            name_label = QLabel(f"  {pip_name}")
            name_label.setWordWrap(True)
            name_label.setMinimumWidth(120)
            name_label.setMaximumWidth(170)
            status_label = QLabel("Checking...")
            status_label.setStyleSheet("color: gray;")
            row_layout.addWidget(name_label)
            row_layout.addWidget(status_label)
            row_layout.addStretch()
            deps_layout.addLayout(row_layout)
            self.dep_status_labels[import_name] = status_label

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

        note = QLabel(
            "Credential values are saved in QGIS settings and applied to the "
            "current QGIS process when a chat request runs."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: gray;")
        layout.addWidget(note)
        layout.addStretch()
        return widget

    def _refresh_dependency_status(self):
        """Refresh dependency labels from the dependency checker."""
        from ..deps_manager import check_dependencies

        deps = check_dependencies()
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
            self.deps_overall_label.setText("All dependencies are installed.")
            self.deps_overall_label.setStyleSheet(
                "color: green; font-weight: bold; padding: 5px;"
            )
            self.install_deps_btn.setVisible(False)
        else:
            missing_count = sum(1 for d in deps if not d["installed"])
            self.deps_overall_label.setText(
                f"{missing_count} package(s) missing. Click Install Dependencies."
            )
            self.deps_overall_label.setStyleSheet(
                "color: #E65100; font-weight: bold; padding: 5px;"
            )
            self.install_deps_btn.setVisible(True)
            self.install_deps_btn.setEnabled(True)

    def _install_dependencies(self):
        """Start background dependency installation."""
        from ..deps_manager import DepsInstallWorker

        self.install_deps_btn.setEnabled(False)
        self.install_deps_btn.setText("Installing...")
        self.refresh_deps_btn.setEnabled(False)

        self.deps_progress_bar.setVisible(True)
        self.deps_progress_bar.setValue(0)
        self.deps_progress_label.setVisible(True)
        self.deps_progress_label.setText("Starting installation...")

        self._deps_worker = DepsInstallWorker()
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
            self.deps_overall_label.setText("Installation failed.")
            self.deps_overall_label.setStyleSheet(
                "color: red; font-weight: bold; padding: 5px;"
            )
            self.install_deps_btn.setEnabled(True)
            QMessageBox.critical(
                self,
                "Installation Failed",
                f"Failed to install dependencies:\n\n{message}\n\n"
                'Manual fallback: pip install "GeoAgent[providers]>=1.0.0"',
            )

        self._deps_worker = None

    def show_dependencies_tab(self):
        """Switch the settings dock to the dependencies tab."""
        self.tab_widget.setCurrentIndex(0)

    def _on_provider_changed(self, provider):
        """Update the model field when the provider changes."""
        self.model_input.setText(DEFAULT_MODELS.get(provider, ""))

    def _load_settings(self):
        """Load persisted settings into the form fields."""
        provider = self.settings.value(f"{SETTINGS_PREFIX}provider", "openai", type=str)
        index = self.provider_combo.findText(provider)
        self.provider_combo.setCurrentIndex(index if index >= 0 else 1)

        model = self.settings.value(f"{SETTINGS_PREFIX}model", "", type=str)
        self.model_input.setText(model or DEFAULT_MODELS.get(provider, ""))
        self.fast_check.setChecked(
            self.settings.value(f"{SETTINGS_PREFIX}fast_mode", False, type=bool)
        )
        self.max_tokens_spin.setValue(
            self.settings.value(f"{SETTINGS_PREFIX}max_tokens", 4096, type=int)
        )
        self.openai_key_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}openai_api_key", "", type=str)
        )
        self.anthropic_key_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}anthropic_api_key", "", type=str)
        )
        self.gemini_key_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}gemini_api_key", "", type=str)
        )
        self.aws_region_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}aws_region", "", type=str)
        )
        self.ollama_host_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}ollama_host", "", type=str)
        )
        self.litellm_key_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}litellm_api_key", "", type=str)
        )
        self.litellm_base_url_input.setText(
            self.settings.value(f"{SETTINGS_PREFIX}litellm_base_url", "", type=str)
        )

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
            f"{SETTINGS_PREFIX}openai_api_key", self.openai_key_input.text()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}anthropic_api_key", self.anthropic_key_input.text()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}gemini_api_key", self.gemini_key_input.text()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}aws_region", self.aws_region_input.text()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}ollama_host", self.ollama_host_input.text()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}litellm_api_key", self.litellm_key_input.text()
        )
        self.settings.setValue(
            f"{SETTINGS_PREFIX}litellm_base_url",
            self.litellm_base_url_input.text(),
        )
        self.status_label.setText("Settings saved")
        self.status_label.setStyleSheet("color: green; font-size: 10px;")
        self.iface.messageBar().pushSuccess("OpenGeoAgent", "Settings saved.")

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

        for key in [
            "provider",
            "model",
            "fast_mode",
            "max_tokens",
            "openai_api_key",
            "anthropic_api_key",
            "gemini_api_key",
            "aws_region",
            "ollama_host",
            "litellm_api_key",
            "litellm_base_url",
        ]:
            self.settings.remove(f"{SETTINGS_PREFIX}{key}")
        self._load_settings()
        self.status_label.setText("Defaults restored")
        self.status_label.setStyleSheet("color: #E65100; font-size: 10px;")
