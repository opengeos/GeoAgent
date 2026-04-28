"""Utilities for 00 home.."""

import solara


@solara.component
def Page():
    """Render the Solara page."""
    with solara.Column(align="center"):
        markdown = """
        ## 🌍 GeoAgent

        An AI agent for geospatial data analysis and visualization.

        **Features:**
        - Natural language interface for geospatial data workflows
        - Multi-LLM support (OpenAI, Anthropic, Google Gemini, Ollama)
        - Interactive MapLibre maps with layer accumulation
        - Dynamic catalog discovery across 134+ Planetary Computer collections
        - Code transparency showing generated Python at each step

        Click **Chat** above to start querying geospatial data.
        """
        solara.Markdown(markdown)
