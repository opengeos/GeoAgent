"""GeoAgent Chat UI — Solara sidebar + persistent map.

The UI map is passed directly to the agent as target_map so that all
leafmap operations (add_stac_layer, set_center, etc.) happen on the
*same* widget the browser is displaying.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import solara
import leafmap.maplibregl as leafmap

from geoagent.core.agent import GeoAgent
from geoagent.core.llm import get_llm, PROVIDERS

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

messages: solara.Reactive[List[Dict[str, str]]] = solara.reactive([])
provider: solara.Reactive[str] = solara.reactive("ollama")
model: solara.Reactive[str] = solara.reactive("")
processing: solara.Reactive[bool] = solara.reactive(False)
status_text: solara.Reactive[str] = solara.reactive("")
last_code: solara.Reactive[str] = solara.reactive("")

_agent_store: Dict[str, Any] = {"agent": None, "key": None}
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


def _make_map_element():
    """Create the default map element (called once, memoized)."""
    return leafmap.Map.element(
        center=[0, 20],
        zoom=2,
        height="750px",
        style="dark-matter",
        use_message_queue=True,
        add_floating_sidebar=False,
    )


def _run_query(query: str, target_map) -> str:
    """Run a GeoAgent query synchronously.

    Passes the UI map directly as target_map so the VizAgent adds layers
    to the *displayed* widget.

    Returns:
        Response text to display
    """
    try:
        prov = provider.value
        mdl = model.value or _get_default_model(prov)
        agent = _get_or_create_agent(prov, mdl)

        logger.info(f"Running query: {query}")
        result = agent.chat(query, target_map=target_map)
        logger.info(f"Query result: success={result.success}, items={result.data.total_items if result.data else 0}")

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

            # Store code
            code = result.code or ""
            if (
                not code
                and result.analysis
                and getattr(result.analysis, "code_generated", None)
            ):
                code = result.analysis.code_generated
            last_code.value = code
        else:
            text = f"❌ {result.error_message or 'An error occurred.'}"

        return text

    except Exception as e:
        import traceback
        traceback.print_exc()
        logger.error(f"Query failed: {e}")
        return f"❌ Error: {e}"


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@solara.component
def Page():
    # Persist map element across re-renders
    map_el = solara.use_memo(_make_map_element, dependencies=[])
    map_widget_ref = solara.use_ref(None)

    query, set_query = solara.use_state("")
    show_code, set_show_code = solara.use_state(False)

    def _on_mount():
        try:
            widget = solara.get_widget(map_el)
            map_widget_ref.current = widget
            if hasattr(widget, "use_message_queue"):
                widget.use_message_queue(True)
            if hasattr(widget, "create_container") and getattr(
                widget, "container", None
            ) is None:
                widget.create_container()
            if hasattr(widget, "add_call"):
                widget.add_call("resize")
        except Exception as e:
            logger.debug(f"Map resize skipped: {e}")

    solara.use_effect(lambda: (_on_mount(), None), dependencies=[map_el])

    def on_send():
        q = query.strip()
        if q and not processing.value:
            # Add user message
            messages.value = [*messages.value, {"role": "user", "content": q}]
            set_query("")

            # Set processing state
            processing.value = True
            status_text.value = "🔍 Searching..."

            target_map = map_widget_ref.current
            if target_map is None:
                messages.value = [
                    *messages.value,
                    {
                        "role": "assistant",
                        "content": "⚠️ Map is still loading. Please try again in a moment.",
                    },
                ]
                processing.value = False
                status_text.value = ""
                return

            # Run query synchronously (important for map updates)
            response = _run_query(q, target_map)

            # Add response
            messages.value = [*messages.value, {"role": "assistant", "content": response}]

            # Reset state
            processing.value = False
            status_text.value = ""

    with solara.Sidebar():
        solara.Markdown("### 💬 GeoAgent Chat")

        with solara.Card(margin=0, elevation=0):
            solara.Select(
                label="Provider",
                value=provider.value,
                values=PROVIDER_LIST,
                on_value=lambda v: (
                    setattr(provider, "value", v),
                    setattr(model, "value", _get_default_model(v)),
                ),
            )
            solara.InputText(
                label="Model",
                value=model.value or _get_default_model(provider.value),
                on_value=model.set,
            )

        # Chat messages container
        with solara.Card(margin=0, elevation=0):
            if messages.value:
                for msg in messages.value:
                    icon = "🧑" if msg["role"] == "user" else "🌍"
                    solara.Markdown(f"{icon} {msg['content']}")
            else:
                solara.Text("No messages yet. Ask about geospatial data!")

            # Status
            if status_text.value:
                solara.Info(status_text.value)

        # Input area
        with solara.Card(margin=0, elevation=0):
            solara.InputText(
                label="Ask about geospatial data…",
                value=query,
                on_value=set_query,
                disabled=processing.value,
                continuous_update=True,
            )

            with solara.Row():
                solara.Button(
                    "SEND",
                    on_click=on_send,
                    disabled=processing.value or not query.strip(),
                    color="primary",
                )

                # Code toggle
                if last_code.value:
                    solara.Button(
                        "SHOW CODE" if not show_code else "HIDE CODE",
                        on_click=lambda: set_show_code(not show_code),
                        text=True,
                    )

                # New Chat
                def on_new_chat():
                    messages.value = []
                    last_code.value = ""
                    status_text.value = ""

                solara.Button("NEW CHAT", on_click=on_new_chat, outlined=True)

            if show_code and last_code.value:
                solara.Preformatted(last_code.value)

    # Map as sole main content
    solara.Column(
        children=[map_el],
        style={
            "width": "100%",
            "height": "100%",
            "min_height": "750px",
            "isolation": "isolate",
        },
    )
