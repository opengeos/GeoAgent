"""Layout regression tests for the OpenGeoAgent settings dock."""

from open_geoagent.dialogs.settings_dock import SettingsDockWidget
from qgis.PyQt.QtWidgets import QApplication, QScrollArea


def test_settings_tabs_scroll_while_action_footer_stays_fixed(monkeypatch) -> None:
    """A short dock must scroll form content without hiding its actions."""
    from open_geoagent import deps_manager

    monkeypatch.setattr(deps_manager, "check_dependencies", lambda *_args: [])
    app = QApplication.instance() or QApplication([])
    dock = SettingsDockWidget(None)
    dock.resize(360, 620)
    app.processEvents()

    assert isinstance(dock.dependencies_scroll, QScrollArea)
    assert isinstance(dock.model_scroll, QScrollArea)
    assert dock.dependencies_scroll.widget() is not None
    assert dock.model_scroll.widget() is not None
    assert not dock.model_scroll.isAncestorOf(dock.save_btn)
    assert not dock.model_scroll.isAncestorOf(dock.test_provider_btn)
    assert dock.minimumSizeHint().height() <= 620

    dock.close()


def test_floating_settings_dock_fits_available_screen(monkeypatch) -> None:
    """Restored oversized floating geometry must not leave actions off-screen."""
    from open_geoagent import deps_manager

    monkeypatch.setattr(deps_manager, "check_dependencies", lambda *_args: [])
    app = QApplication.instance() or QApplication([])
    dock = SettingsDockWidget(None)
    dock.setFloating(True)
    available = dock.screen().availableGeometry().adjusted(8, 8, -8, -8)
    dock.resize(available.width(), available.height())
    dock.show()
    app.processEvents()

    assert dock.height() <= available.height()
    assert available.contains(dock.frameGeometry())

    dock.close()
