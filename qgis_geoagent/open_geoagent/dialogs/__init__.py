"""OpenGeoAgent dock widgets and dialogs.

The dialog classes are imported lazily so opening one panel does not import
all plugin dialogs and their helper modules.
"""

__all__ = ["ChatDockWidget", "SettingsDockWidget", "UpdateCheckerDialog"]


def __getattr__(name):
    """Import dialog classes on first attribute access."""
    if name == "ChatDockWidget":
        from .chat_dock import ChatDockWidget

        return ChatDockWidget
    if name == "SettingsDockWidget":
        from .settings_dock import SettingsDockWidget

        return SettingsDockWidget
    if name == "UpdateCheckerDialog":
        from .update_checker import UpdateCheckerDialog

        return UpdateCheckerDialog
    raise AttributeError(name)
