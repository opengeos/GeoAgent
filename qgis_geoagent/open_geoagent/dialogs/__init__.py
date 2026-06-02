"""OpenGeoAgent dock widgets and dialogs.

The dialog classes are imported lazily so opening one panel does not import
all plugin dialogs and their helper modules.
"""

# ---------------------------------------------------------------------------
# Global SSL bypass — runs at plugin load time, before any provider client
# is constructed.  Covers both sync (httpx.Client) and async
# (httpx.AsyncClient) paths used by the OpenAI SDK.
# Only active when OPENAI_SSL_VERIFY=0 is set in the environment.
# ---------------------------------------------------------------------------
import os as _os
import ssl as _ssl

if _os.environ.get("OPENAI_SSL_VERIFY", "1").strip() == "0":
    _ssl._create_default_https_context = _ssl._create_unverified_context  # noqa: SLF001

    import httpx as _httpx

    _OriginalClient = _httpx.Client
    _OriginalAsyncClient = _httpx.AsyncClient

    class _UnverifiedClient(_OriginalClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            super().__init__(*args, **kwargs)

    class _UnverifiedAsyncClient(_OriginalAsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            super().__init__(*args, **kwargs)

    _httpx.Client = _UnverifiedClient
    _httpx.AsyncClient = _UnverifiedAsyncClient

# ---------------------------------------------------------------------------

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
