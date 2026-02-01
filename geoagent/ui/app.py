"""Solara chat UI for GeoAgent.

Features:
- Persistent map widget with layers accumulating across queries
- Native ipywidget rendering (no HTML export, full interactivity)
- Chat interface with status updates
- Sidebar with provider/model selection
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import solara
from leafmap.maplibregl import Map as MapLibreMap

from geoagent.core.agent import GeoAgent
from geoagent.core.llm import get_llm, PROVIDERS
from geoagent.core.models import GeoAgentResponse


# ---------------------------------------------------------------------------
# Reactive state
# ---------------------------------------------------------------------------

messages: solara.Reactive[List[Dict[str, str]]] = solara.reactive([])
provider: solara.Reactive[str] = solara.reactive("openai")
model: solara.Reactive[str] = solara.reactive("")
processing: solara.Reactive[bool] = solara.reactive(False)
status_text: solara.Reactive[str] = solara.reactive("")
last_code: solara.Reactive[str] = solara.reactive("")

# Agent and map — stored outside reactive to persist across renders
_agent_store: Dict[str, Any] = {"agent": None, "key": None}
_map_store: Dict[str, Any] = {"map": None}

PROVIDER_LIST = list(PROVIDERS.keys())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_default_model(prov: str) -> str:
    return PROVIDERS.get(prov, {}).get("default_model", "gpt-4.1")


def _get_or_create_agent(prov: str, mdl: str) -> GeoAgent:
    key = f"{prov}:{mdl}"
    if _agent_store["key"] != key or _agent_store["agent"] is None:
        llm = get_llm(provider=prov, model=mdl)
        _agent_store["agent"] = GeoAgent(llm=llm, provider=prov, model=mdl)
        _agent_store["key"] = key
    return _agent_store["agent"]


def _get_or_create_map() -> MapLibreMap:
    if _map_store["map"] is None:
        _map_store["map"] = MapLibreMap(
            center=[0, 20],
            zoom=2,
            height="100%",
            style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        )
    return _map_store["map"]


def _add_result_to_map(m: MapLibreMap, result: GeoAgentResponse):
    """Transfer layers from result map to persistent map."""
    if result.map is None:
        return
    src_map = result.map
    if not hasattr(src_map, "calls"):
        return

    for call in src_map.calls:
        method_name = call[0]
        args = call[1] if len(call) > 1 else ()
        # Replay source/layer/paint/layout/bounds calls on persistent map
        if method_name in (
            "addSource",
            "addLayer",
            "setPaintProperty",
            "setLayoutProperty",
            "fitBounds",
        ):
            try:
                method = getattr(m, method_name, None)
                if method and callable(method):
                    if isinstance(args, tuple):
                        method(*args)
                    else:
                        method(args)
            except Exception:
                pass


def _run_query(query: str):
    """Run a GeoAgent query in a background thread."""
    processing.value = True
    status_text.value = "🔍 Parsing query…"
    messages.value = [*messages.value, {"role": "user", "content": query}]

    try:
        prov = provider.value
        mdl = model.value or _get_default_model(prov)
        status_text.value = "⚙️ Initializing agent…"
        agent = _get_or_create_agent(prov, mdl)

        status_text.value = "📡 Searching data & analyzing…"
        result = agent.chat(query)

        # Build summary
        if result.success:
            items = result.data.total_items if result.data else 0
            parts = []
            if result.plan:
                intent = getattr(result.plan, "intent", "")
                dataset = getattr(result.plan, "dataset", "")
                loc = getattr(result.plan, "location", None)
                if intent:
                    parts.append(intent)
                if dataset:
                    parts.append(dataset)
                if isinstance(loc, dict) and loc.get("name"):
                    parts.append(loc["name"])
            t = f"{result.execution_time:.1f}s" if result.execution_time else ""
            meta = f"{items} items" + (f" • {t}" if t else "")
            summary = " • ".join(parts) if parts else "Done"
            text = f"✅ {summary} ({meta})"
        else:
            text = f"❌ {result.error_message or 'An error occurred.'}"

        messages.value = [*messages.value, {"role": "assistant", "content": text}]

        # Add layers to persistent map
        status_text.value = "🗺️ Updating map…"
        m = _get_or_create_map()
        _add_result_to_map(m, result)

        # Store code
        code = result.code or ""
        if (
            not code
            and result.analysis
            and getattr(result.analysis, "code_generated", None)
        ):
            code = result.analysis.code_generated
        last_code.value = code

        status_text.value = ""

    except Exception as e:
        messages.value = [
            *messages.value,
            {"role": "assistant", "content": f"❌ Error: {e}"},
        ]
        status_text.value = ""
    finally:
        processing.value = False


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@solara.component
def ChatMessage(msg: Dict[str, str]):
    role = msg["role"]
    icon = "mdi-account" if role == "user" else "mdi-earth"
    color = "primary" if role == "user" else "success"
    with solara.Row(style={"margin": "4px 0"}):
        solara.v.Icon(
            children=[icon], color=color, style_="margin-right: 8px; margin-top: 4px;"
        )
        solara.Markdown(msg["content"])


@solara.component
def ChatPanel():
    query, set_query = solara.use_state("")

    with solara.Column(style={"height": "100%"}):
        solara.Markdown("### 💬 Chat")

        # Messages
        with solara.Column(
            style={
                "flex": "1",
                "overflow-y": "auto",
                "max-height": "calc(100vh - 250px)",
                "padding": "8px",
            }
        ):
            for msg in messages.value:
                ChatMessage(msg)

            # Status indicator
            if status_text.value:
                with solara.Row(style={"margin": "8px 0"}):
                    solara.v.ProgressCircular(indeterminate=True, size=20, width=2)
                    solara.Text(status_text.value, style={"margin-left": "8px"})

        # Input
        def on_send():
            q = query.strip()
            if q and not processing.value:
                set_query("")
                threading.Thread(target=_run_query, args=(q,), daemon=True).start()

        solara.InputText(
            label="Ask about geospatial data…",
            value=query,
            on_value=set_query,
            disabled=processing.value,
            style={"width": "100%"},
        )
        solara.Button(
            "Send",
            on_click=on_send,
            disabled=processing.value or not query.strip(),
            color="primary",
            style={"margin-top": "4px"},
        )


@solara.component
def MapPanel():
    m = _get_or_create_map()
    with solara.Column(style={"height": "100%"}):
        solara.Markdown("### 🗺️ Map")
        solara.display(m)

        if last_code.value:
            show_code, set_show_code = solara.use_state(False)
            solara.Button(
                "Show Code" if not show_code else "Hide Code",
                on_click=lambda: set_show_code(not show_code),
                text=True,
                icon_name="mdi-code-tags",
            )
            if show_code:
                solara.Preformatted(last_code.value)


@solara.component
def Sidebar():
    def on_provider_change(value):
        provider.value = value
        model.value = _get_default_model(value)

    solara.Markdown("### Settings")
    solara.Select(
        label="Provider",
        value=provider.value,
        values=PROVIDER_LIST,
        on_value=on_provider_change,
    )
    solara.InputText(
        label="Model",
        value=model.value or _get_default_model(provider.value),
        on_value=model.set,
    )

    def on_new_chat():
        messages.value = []
        last_code.value = ""
        status_text.value = ""

    solara.Button(
        "New Chat",
        on_click=on_new_chat,
        outlined=True,
        style={"margin-top": "12px"},
    )


@solara.component
def Page():
    solara.Title("GeoAgent")

    with solara.Sidebar():
        Sidebar()

    with solara.Columns([1, 1]):
        ChatPanel()
        MapPanel()
