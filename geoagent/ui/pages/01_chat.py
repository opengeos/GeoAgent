"""Solara chat page (minimal) — use the Python API for full control.

GeoAgent 1.0 uses Strands Agents. Install ``GeoAgent[ui,leafmap,openai]`` (or
other provider extra), set API keys, then use :class:`geoagent.GeoAgent` in
this page or in a notebook.
"""

from __future__ import annotations

import solara
import solara.alias as rv

from geoagent import __version__


@solara.component
def Page():
    """Render the Solara page."""
    with solara.AppBarTitle("GeoAgent chat"):
        pass
    with solara.Column(align="center"):
        rv.Markdown(f"# GeoAgent {__version__}")
        rv.Markdown(
            "Configure a model provider via environment variables, then "
            "use `from geoagent import for_leafmap` and `agent.chat(...)` "
            "from the Python console or a notebook."
        )


__routes__ = ["/", Page]
