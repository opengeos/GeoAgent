"""OpenGeoAgent QGIS plugin main class."""

import os
import re

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QToolBar, QMessageBox


class OpenGeoAgent:
    """QGIS plugin that exposes GeoAgent through a dockable chat interface."""

    def __init__(self, iface):
        """Initialize the plugin with the QGIS interface."""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = None
        self.toolbar = None
        self._chat_dock = None
        self._settings_dock = None

    def add_action(
        self,
        icon_path,
        text,
        callback,
        enabled_flag=True,
        add_to_menu=True,
        add_to_toolbar=True,
        status_tip=None,
        checkable=False,
        parent=None,
    ):
        """Create a QAction and add it to the plugin menu and toolbar."""
        action = QAction(QIcon(icon_path), text, parent)
        action.triggered.connect(callback)
        action.setEnabled(enabled_flag)
        action.setCheckable(checkable)

        if status_tip is not None:
            action.setStatusTip(status_tip)

        if add_to_toolbar:
            self.toolbar.addAction(action)
        if add_to_menu:
            self.menu.addAction(action)

        self.actions.append(action)
        return action

    def initGui(self):
        """Create OpenGeoAgent menu entries and toolbar buttons."""
        self.menu = QMenu("&OpenGeoAgent")
        self.iface.mainWindow().menuBar().addMenu(self.menu)

        self.toolbar = QToolBar("OpenGeoAgent Toolbar")
        self.toolbar.setObjectName("OpenGeoAgentToolbar")
        self.iface.addToolBar(self.toolbar)

        icon_base = os.path.join(self.plugin_dir, "icons")
        main_icon = os.path.join(icon_base, "icon.svg")
        if not os.path.exists(main_icon):
            main_icon = ":/images/themes/default/mIconChat.svg"

        settings_icon = os.path.join(icon_base, "settings.svg")
        if not os.path.exists(settings_icon):
            settings_icon = ":/images/themes/default/mActionOptions.svg"

        about_icon = os.path.join(icon_base, "about.svg")
        if not os.path.exists(about_icon):
            about_icon = ":/images/themes/default/mActionHelpContents.svg"

        self.chat_action = self.add_action(
            main_icon,
            "Open Chat",
            self.toggle_chat_dock,
            status_tip="Open the OpenGeoAgent chat panel",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        self.settings_action = self.add_action(
            settings_icon,
            "Settings",
            self.toggle_settings_dock,
            status_tip="Configure OpenGeoAgent providers, models, and dependencies",
            checkable=True,
            parent=self.iface.mainWindow(),
        )

        self.menu.addSeparator()

        self.add_action(
            ":/images/themes/default/mActionRefresh.svg",
            "Check for Updates...",
            self.show_update_checker,
            add_to_toolbar=False,
            status_tip="Check for OpenGeoAgent plugin updates from GitHub",
            parent=self.iface.mainWindow(),
        )

        self.add_action(
            about_icon,
            "About OpenGeoAgent",
            self.show_about,
            add_to_toolbar=False,
            status_tip="About OpenGeoAgent",
            parent=self.iface.mainWindow(),
        )

    def unload(self):
        """Remove plugin UI elements from QGIS."""
        if self._chat_dock:
            self.iface.removeDockWidget(self._chat_dock)
            self._chat_dock.deleteLater()
            self._chat_dock = None

        if self._settings_dock:
            self.iface.removeDockWidget(self._settings_dock)
            self._settings_dock.deleteLater()
            self._settings_dock = None

        if self.toolbar:
            for action in self.actions:
                self.toolbar.removeAction(action)
            self.iface.mainWindow().removeToolBar(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None

        if self.menu:
            self.menu.deleteLater()
            self.menu = None

        self.actions = []

    def toggle_chat_dock(self):
        """Toggle the chat dock widget."""
        if self._dependencies_missing():
            self._show_settings_dock(dependencies_tab=True)
            self.chat_action.setChecked(False)
            try:
                self.iface.messageBar().pushWarning(
                    "OpenGeoAgent",
                    "Install missing dependencies before opening the chat panel.",
                )
            except Exception:
                pass
            return

        if self._chat_dock is None:
            try:
                from .dialogs.chat_dock import ChatDockWidget

                self._chat_dock = ChatDockWidget(self.iface, self.iface.mainWindow())
                self._chat_dock.setObjectName("OpenGeoAgentChatDock")
                self._chat_dock.visibilityChanged.connect(
                    self._on_chat_visibility_changed
                )
                self.iface.addDockWidget(
                    Qt.DockWidgetArea.RightDockWidgetArea, self._chat_dock
                )
                self._chat_dock.show()
                self._chat_dock.raise_()
                return
            except Exception as exc:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "OpenGeoAgent",
                    f"Failed to create chat panel:\n{exc}",
                )
                self.chat_action.setChecked(False)
                return

        if self._chat_dock.isVisible():
            self._chat_dock.hide()
        else:
            self._chat_dock.show()
            self._chat_dock.raise_()

    def _on_chat_visibility_changed(self, visible):
        """Handle chat visibility changed."""
        self.chat_action.setChecked(visible)

    def _dependencies_missing(self):
        """Return True when required Python packages are not importable."""
        try:
            from .deps_manager import all_dependencies_met

            return not all_dependencies_met()
        except Exception:
            return True

    def _ensure_settings_dock(self):
        """Create the settings dock if needed."""
        if self._settings_dock is not None:
            return True

        try:
            from .dialogs.settings_dock import SettingsDockWidget

            self._settings_dock = SettingsDockWidget(
                self.iface, self.iface.mainWindow()
            )
            self._settings_dock.setObjectName("OpenGeoAgentSettingsDock")
            self._settings_dock.visibilityChanged.connect(
                self._on_settings_visibility_changed
            )
            self.iface.addDockWidget(
                Qt.DockWidgetArea.RightDockWidgetArea, self._settings_dock
            )
            return True
        except Exception as exc:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "OpenGeoAgent",
                f"Failed to create settings panel:\n{exc}",
            )
            self.settings_action.setChecked(False)
            return False

    def _show_settings_dock(self, dependencies_tab=False):
        """Ensure the settings dock is visible and optionally select dependencies."""
        if not self._ensure_settings_dock():
            return
        self._settings_dock.show()
        self._settings_dock.raise_()
        self.settings_action.setChecked(True)
        if dependencies_tab and hasattr(self._settings_dock, "show_dependencies_tab"):
            self._settings_dock.show_dependencies_tab()

    def toggle_settings_dock(self):
        """Toggle the settings dock widget."""
        if self._settings_dock is None:
            self._show_settings_dock()
            return

        if self._settings_dock.isVisible():
            self._settings_dock.hide()
        else:
            self._settings_dock.show()
            self._settings_dock.raise_()

    def _on_settings_visibility_changed(self, visible):
        """Handle settings visibility changed."""
        self.settings_action.setChecked(visible)

    def show_about(self):
        """Display the about dialog."""
        version = "Unknown"
        try:
            metadata_path = os.path.join(self.plugin_dir, "metadata.txt")
            with open(metadata_path, "r", encoding="utf-8") as f:
                version_match = re.search(r"^version=(.+)$", f.read(), re.MULTILINE)
                if version_match:
                    version = version_match.group(1).strip()
        except Exception as exc:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "OpenGeoAgent",
                f"Could not read version from metadata.txt:\n{exc}",
            )

        about_text = f"""
<h2>OpenGeoAgent</h2>
<p>Version: {version}</p>
<p>Author: Qiusheng Wu</p>

<p>OpenGeoAgent brings the GeoAgent QGIS tool surface into a dockable QGIS
chat interface. It can inspect project layers, navigate the map canvas, add
data, run selected QGIS actions, and answer questions using configurable
LLM providers.</p>

<h3>Features</h3>
<ul>
<li>Dockable chatbot panel for QGIS projects</li>
<li>Provider and model selection for Bedrock, OpenAI, Anthropic, Google Gemini, and Ollama</li>
<li>Sample prompts, prompt history, and Ctrl+Enter prompt sending</li>
<li>One-click dependency installer in an isolated virtual environment</li>
<li>GitHub update checker and packaging scripts</li>
</ul>

<h3>Links</h3>
<ul>
<li><a href="https://github.com/opengeos/GeoAgent">GitHub Repository</a></li>
<li><a href="https://github.com/opengeos/GeoAgent/issues">Report Issues</a></li>
</ul>

<p>Licensed under the MIT License.</p>
"""
        QMessageBox.about(self.iface.mainWindow(), "About OpenGeoAgent", about_text)

    def show_update_checker(self):
        """Display the update checker dialog."""
        try:
            from .dialogs.update_checker import UpdateCheckerDialog
        except ImportError as exc:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "OpenGeoAgent",
                f"Failed to import update checker dialog:\n{exc}",
            )
            return

        try:
            dialog = UpdateCheckerDialog(self.plugin_dir, self.iface.mainWindow())
            dialog.exec()
        except Exception as exc:
            QMessageBox.critical(
                self.iface.mainWindow(),
                "OpenGeoAgent",
                f"Failed to open update checker:\n{exc}",
            )
