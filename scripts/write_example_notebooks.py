#!/usr/bin/env python3
"""Generate docs/examples notebooks."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import nbformat
    from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook
except ModuleNotFoundError:

    def new_code_cell(source: str) -> dict:
        """Return a minimal Jupyter code cell."""
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }

    def new_markdown_cell(source: str) -> dict:
        """Return a minimal Jupyter markdown cell."""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        }

    def new_notebook(*, metadata: dict, cells: list[dict]) -> dict:
        """Return a minimal Jupyter notebook document."""
        return {
            "cells": cells,
            "metadata": metadata,
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    class _NbFormatFallback:
        """Small writer compatible with ``nbformat.write`` for this script."""

        @staticmethod
        def write(notebook: dict, path: str) -> None:
            Path(path).write_text(
                json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    nbformat = _NbFormatFallback()

DEFAULT_MODEL = "claude-sonnet-4-6"
CODEX_MODEL = "gpt-5.5"


def main() -> None:
    """Run the script entry point."""
    meta = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
    }

    intro = new_notebook(
        metadata=meta,
        cells=[
            new_markdown_cell(
                "# GeoAgent 1.0 — Anthropic (Claude)\n\n"
                "- Set **`ANTHROPIC_API_KEY`** in your environment.\n"
                "- Optionally set **`ANTHROPIC_MODEL`** "
                f"(default: `{DEFAULT_MODEL}` — Claude Sonnet 4.6).\n"
                '- Install: `pip install "GeoAgent[anthropic]"`.'
            ),
            new_code_cell('%pip install -q "GeoAgent[anthropic]"'),
            new_code_cell(
                "import os\n"
                "\n"
                'if not os.environ.get("ANTHROPIC_API_KEY"):\n'
                '    raise RuntimeError("Set ANTHROPIC_API_KEY before running this '
                'cell.")\n'
                "\n"
                f'MODEL = os.environ.get("ANTHROPIC_MODEL", "{DEFAULT_MODEL}")\n'
                'print("Using model:", MODEL)'
            ),
            new_code_cell(
                "from geoagent import GeoAgent\n"
                "from geoagent.core.config import GeoAgentConfig\n"
                "\n"
                "agent = GeoAgent(\n"
                "    config=GeoAgentConfig(\n"
                '        provider="anthropic",\n'
                "        model=MODEL,\n"
                "        temperature=0.0,\n"
                "        max_tokens=2048,\n"
                "    ),\n"
                ")\n"
                "\n"
                'resp = agent.chat("In one sentence, what is STAC in Earth '
                'observation?")\n'
                'print("success:", resp.success)\n'
                "print(resp.answer_text)\n"
                'print("executed_tools:", resp.executed_tools)'
            ),
        ],
    )

    live = new_notebook(
        metadata=meta,
        cells=[
            new_markdown_cell(
                "# GeoAgent + leafmap (MapLibre) + Claude\n\n"
                "- **`ANTHROPIC_API_KEY`** required.\n"
                '- Install: `pip install "GeoAgent[anthropic,leafmap]"`.\n'
                "- MapLibre camera state lives on **`m.view_state`** (not ipyleaflet). "
                "The `get_map_state` tool reads that structure.\n"
                "- Optional **`ANTHROPIC_MODEL`** "
                f"(default `{DEFAULT_MODEL}`)."
            ),
            new_code_cell('%pip install -q "GeoAgent[anthropic,leafmap]" leafmap'),
            new_code_cell(
                "import os\n"
                "\n"
                'if not os.environ.get("ANTHROPIC_API_KEY"):\n'
                '    raise RuntimeError("Set ANTHROPIC_API_KEY")\n'
                "\n"
                f'MODEL = os.environ.get("ANTHROPIC_MODEL", "{DEFAULT_MODEL}")'
            ),
            new_code_cell(
                "import leafmap.maplibregl as leafmap\n"
                "from geoagent import for_leafmap\n"
                "from geoagent.core.config import GeoAgentConfig\n"
                "from geoagent.tools.leafmap import leafmap_tools\n"
                "\n"
                'm = leafmap.Map(center=[-83.92, 35.96], zoom=9, height="520px")\n'
                "m"
            ),
            new_markdown_cell("## Map widget"),
            new_code_cell(
                "\n"
                'vs = getattr(m, "view_state", None)\n'
                "if isinstance(vs, dict):\n"
                '    print("view_state keys:", list(vs.keys()))\n'
                "else:\n"
                '    print("view_state:", type(vs).__name__ if vs is not None '
                "else None)\n"
                "\n"
                "cfg = GeoAgentConfig(\n"
                '    provider="anthropic",\n'
                "    model=MODEL,\n"
                "    temperature=0.0,\n"
                "    max_tokens=4096,\n"
                ")\n"
                "agent = for_leafmap(m, config=cfg, fast=True)\n"
                "\n"
                "get_state = next(t for t in leafmap_tools(m) if t.tool_name == "
                '"get_map_state")\n'
                "st = get_state()\n"
                'print("get_map_state zoom / center:", '
                'st.get("zoom"), st.get("center"))\n'
                "\n"
                "resp = agent.chat(\n"
                '    "Using the map state, what is the zoom level? One sentence."\n'
                ")\n"
                "print(resp.answer_text)\n"
                'print("tools:", resp.executed_tools)'
            ),
            new_code_cell(
                'resp = agent.chat("Fly to San Francisco")\n'
                "print(resp.answer_text)\n"
                'print("tools:", resp.executed_tools)'
            ),
        ],
    )

    qgis_nb = new_notebook(
        metadata=meta,
        cells=[
            new_markdown_cell(
                "# GeoAgent + QGIS tools + Claude (mock iface)\n\n"
                "This notebook runs **without QGIS** using `geoagent.testing` mocks so "
                "you can try the agent in Jupyter.\n\n"
                "**In QGIS** (Python console or plugin), use the real interface:\n\n"
                "```python\n"
                "from qgis.utils import iface\n"
                "from geoagent import for_qgis\n"
                "from geoagent.core.config import GeoAgentConfig\n"
                "\n"
                'agent = for_qgis(iface, config=GeoAgentConfig(provider="anthropic", '
                "model=MODEL))\n"
                'agent.chat("List project layers.")\n'
                "```\n\n"
                'Requires `pip install "GeoAgent[anthropic]"` and '
                "`ANTHROPIC_API_KEY`."
            ),
            new_code_cell('%pip install -q "GeoAgent[anthropic]"'),
            new_code_cell(
                "import os\n"
                "\n"
                'if not os.environ.get("ANTHROPIC_API_KEY"):\n'
                '    raise RuntimeError("Set ANTHROPIC_API_KEY")\n'
                "\n"
                f'MODEL = os.environ.get("ANTHROPIC_MODEL", "{DEFAULT_MODEL}")'
            ),
            new_code_cell(
                "from geoagent import for_qgis\n"
                "from geoagent.testing import MockQGISIface, MockQGISLayer, "
                "MockQGISProject\n"
                "\n"
                "project = MockQGISProject()\n"
                'project.addMapLayer(MockQGISLayer("StudyArea", "/tmp/a.shp"))\n'
                "iface = MockQGISIface(project=project)\n"
                "\n"
                "agent = for_qgis(\n"
                "    iface,\n"
                "    project,\n"
                '    provider="anthropic",\n'
                "    model_id=MODEL,\n"
                "    fast=True,\n"
                ")\n"
                "\n"
                'resp = agent.chat("List the layer names in this project in a short '
                'bullet list.")\n'
                "print(resp.answer_text)\n"
                'print("tools:", resp.executed_tools)'
            ),
            new_code_cell(
                "from geoagent import for_qgis\n"
                "import os\n"
                "\n"
                'if not os.environ.get("ANTHROPIC_API_KEY"):\n'
                '    raise RuntimeError("Set ANTHROPIC_API_KEY")\n'
                "\n"
                f'MODEL = os.environ.get("ANTHROPIC_MODEL", "{DEFAULT_MODEL}")\n'
                "\n"
                "agent = for_qgis(\n"
                "    iface,\n"
                '    provider="anthropic",\n'
                "    model_id=MODEL,\n"
                "    fast=True,\n"
                ")\n"
                "\n"
                'resp = agent.chat("List the layer names in this project in a short '
                'bullet list.")\n'
                "print(resp.answer_text)\n"
                'print("tools:", resp.executed_tools)'
            ),
        ],
    )

    codex_nb = new_notebook(
        metadata=meta,
        cells=[
            new_markdown_cell(
                "# GeoAgent + ChatGPT/Codex OAuth\n\n"
                "This notebook uses the same ChatGPT/Codex OAuth flow as the QGIS "
                "plugin, but stores the login in GeoAgent's user config directory "
                "for regular Python and Jupyter sessions.\n\n"
                '- Install: `pip install "GeoAgent[openai]"`.\n'
                "- Run the login cell once. Later notebooks can reuse the stored "
                "login automatically."
            ),
            new_code_cell('%pip install -q "GeoAgent[openai]"'),
            new_code_cell(
                "from geoagent import ensure_openai_codex_environment\n"
                "from geoagent import login_openai_codex\n"
                "\n"
                "try:\n"
                "    ensure_openai_codex_environment()\n"
                '    print("Using stored ChatGPT/Codex login.")\n'
                "except RuntimeError:\n"
                "    login_openai_codex()\n"
                '    print("Logged in with ChatGPT/Codex OAuth.")'
            ),
            new_code_cell(
                "import os\n"
                "\n"
                f'MODEL = os.environ.get("OPENAI_CODEX_MODEL", "{CODEX_MODEL}")\n'
                'print("Using model:", MODEL)'
            ),
            new_code_cell(
                "from geoagent import GeoAgent, GeoAgentConfig\n"
                "\n"
                "agent = GeoAgent(\n"
                "    config=GeoAgentConfig(\n"
                '        provider="openai-codex",\n'
                "        model=MODEL,\n"
                "        temperature=0.0,\n"
                "        max_tokens=2048,\n"
                "    ),\n"
                ")\n"
                "\n"
                'resp = agent.chat("In one sentence, what is GeoAgent useful for?")\n'
                'print("success:", resp.success)\n'
                "print(resp.answer_text)\n"
                'print("executed_tools:", resp.executed_tools)'
            ),
            new_markdown_cell("## Leafmap Example"),
            new_code_cell('%pip install -q "GeoAgent[leafmap]" leafmap'),
            new_code_cell(
                "import leafmap.maplibregl as leafmap\n"
                "from geoagent import for_leafmap\n"
                "\n"
                'm = leafmap.Map(center=[-83.92, 35.96], zoom=9, height="520px")\n'
                "m"
            ),
            new_code_cell(
                "map_agent = for_leafmap(\n"
                "    m,\n"
                '    provider="openai-codex",\n'
                "    model_id=MODEL,\n"
                "    fast=True,\n"
                ")\n"
                "\n"
                "resp = map_agent.chat(\n"
                '    "Using the current map state, describe the map center and zoom."\n'
                ")\n"
                "print(resp.answer_text)\n"
                'print("tools:", resp.executed_tools)'
            ),
        ],
    )

    nbformat.write(intro, "docs/examples/intro.ipynb")
    nbformat.write(live, "docs/examples/live_mapping.ipynb")
    nbformat.write(qgis_nb, "docs/examples/qgis_agent.ipynb")
    nbformat.write(codex_nb, "docs/examples/openai_codex.ipynb")
    print(
        "Wrote docs/examples/intro.ipynb, live_mapping.ipynb, "
        "qgis_agent.ipynb, openai_codex.ipynb"
    )


if __name__ == "__main__":
    main()
