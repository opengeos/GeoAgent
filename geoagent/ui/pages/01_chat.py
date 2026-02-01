"""GeoAgent Chat UI — Solara sidebar + persistent map.

The UI map is passed directly to the agent as target_map so that all
leafmap operations (add_stac_layer, set_center, etc.) happen on the
*same* widget the browser is displaying.  When _rendered=True the
MapWidget.add_call() path sends calls straight to the JS frontend via
self.send(), so layers appear immediately — even from a background thread.

Layout:
    Sidebar uses simple top-to-bottom flow with a height-bounded
    scrollable messages area so the input stays visible.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import inspect

import solara
from solara.alias import rv as v
from solara.components.input import use_change
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


def _make_map():
    """Create the default map (called once, memoized)."""
    m = leafmap.Map(
        center=[0, 20],
        zoom=2,
        height="750px",
        style="dark-matter",
        use_message_queue=True,
        add_floating_sidebar=False,
        add_sidebar=True,
        sidebar_visible=False,
    )
    return m


def _run_query(query: str, target_map, status_callback=None) -> str:
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
        result = _chat_with_status(
            agent, query, target_map=target_map, status_callback=status_callback
        )
        logger.info(
            f"Query result: success={result.success}, "
            f"items={result.data.total_items if result.data else 0}"
        )

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
def ChatMessage(role: str, content: str):
    """Single chat bubble."""
    is_user = role == "user"
    icon = "🧑" if is_user else "🌍"
    bg = "#e3f2fd" if is_user else "#f5f5f5"
    solara.Markdown(
        f"{icon} {content}",
        style={
            "background": bg,
            "border-radius": "8px",
            "padding": "8px 12px",
            "margin": "4px 0",
        },
    )


@solara.component
def ChatInput(
    value: str,
    on_value,
    on_enter,
    disabled: bool = False,
):
    """Text input with continuous updates and Enter-to-send."""
    reactive_value = solara.use_reactive(value, on_value)

    def set_value_cast(val: Optional[str]):
        reactive_value.value = "" if val is None else str(val)

    def on_v_model(val):
        set_value_cast(val)

    text_field = v.TextField(
        v_model=reactive_value.value,
        on_v_model=on_v_model,
        label="Ask about geospatial data…",
        disabled=disabled,
        type="text",
        dense=True,
        hide_details=True,
    )

    def handle_enter(val):
        if on_enter:
            on_enter(val)

    use_change(text_field, handle_enter, enabled=not disabled, update_events=["keyup.enter"])
    return text_field


@solara.component
def Page():
    # Persist map across re-renders
    m = solara.use_memo(_make_map, dependencies=[])

    query, set_query = solara.use_state("")
    show_code, set_show_code = solara.use_state(False)

    def _on_mount():
        try:
            if hasattr(m, "use_message_queue"):
                m.use_message_queue(True)
            if hasattr(m, "create_container") and getattr(m, "container", None) is None:
                m.create_container()
            if hasattr(m, "add_call"):
                m.add_call("resize")
        except Exception as e:
            logger.debug(f"Map resize skipped: {e}")

    def _mount_effect():
        _on_mount()
        return None

    solara.use_effect(_mount_effect, dependencies=[])

    def do_send(text=None):
        """Send a query. *text* comes from Enter key; falls back to query state."""
        q = (text if isinstance(text, str) else query).strip()
        if not q or processing.value:
            return

        # Add user message
        messages.value = [*messages.value, {"role": "user", "content": q}]
        set_query("")

        # Set processing state
        processing.value = True

        stage_labels = {
            "planning": "🧭 Planning",
            "fetch_data": "🔎 Searching data",
            "analysis": "📊 Analyzing",
            "visualize": "🗺️ Visualizing",
        }
        snippet = q if len(q) <= 60 else f"{q[:57]}…"

        def on_stage(stage: str):
            label = stage_labels.get(stage, stage)
            status_text.value = f"{label} • {snippet}"

        status_text.value = f"{stage_labels['planning']} • {snippet}"

        # Run query synchronously (important for map updates)
        response = _run_query(q, m, status_callback=on_stage)

        # Add response
        messages.value = [
            *messages.value,
            {"role": "assistant", "content": response},
        ]

        # Reset state
        processing.value = False
        status_text.value = ""

    # ── Sidebar layout ──
    with solara.Sidebar():
        with solara.Column(
            style={
                "height": "100%",
                "min-height": "0",
                "display": "flex",
                "flex-direction": "column",
                "box-sizing": "border-box",
                "padding-bottom": "8px",
            }
        ):
            # Title
            solara.Markdown("### 💬 GeoAgent Chat")

            # Provider / Model
            with solara.Row(style={"gap": "8px"}):
                solara.Select(
                    label="Provider",
                    value=provider.value,
                    values=PROVIDER_LIST,
                    on_value=lambda val: (
                        setattr(provider, "value", val),
                        setattr(model, "value", _get_default_model(val)),
                    ),
                    style={"flex": "1"},
                )
                solara.InputText(
                    label="Model",
                    value=model.value or _get_default_model(provider.value),
                    on_value=model.set,
                    style={"flex": "1"},
                )

            solara.HTML(tag="hr", style={"margin": "4px 0"})

            # Scrollable chat messages (fills remaining space)
            with solara.Column(
                style={
                    "flex": "1 1 auto",
                    "min-height": "0",
                    "overflow-y": "auto",
                    "padding": "4px 0",
                }
            ):
                if messages.value:
                    for msg in messages.value:
                        ChatMessage(role=msg["role"], content=msg["content"])
                else:
                    solara.Text(
                        "No messages yet. Ask about geospatial data!",
                        style={"color": "#888", "font-style": "italic"},
                    )

                # Status indicator
                if status_text.value:
                    solara.Info(status_text.value)

            solara.HTML(tag="hr", style={"margin": "4px 0"})

            with solara.Column(
                style={
                    "position": "sticky",
                    "bottom": "0",
                    "background": "var(--v-background-base, #fff)",
                    "padding-top": "6px",
                }
            ):
                with solara.Column(style={"gap": "6px"}):
                    # Input field
                    ChatInput(
                        value=query,
                        on_value=set_query,
                        on_enter=do_send,
                        disabled=processing.value,
                    )

                    # Action buttons
                    with solara.Row(style={"gap": "4px", "flex-wrap": "wrap"}):
                        solara.Button(
                            "Send",
                            on_click=lambda: do_send(query),
                            disabled=processing.value or not query.strip(),
                            color="primary",
                            icon_name="mdi-send",
                        )

                        def on_new_chat():
                            messages.value = []
                            last_code.value = ""
                            status_text.value = ""

                        solara.Button(
                            "New Chat",
                            on_click=on_new_chat,
                            outlined=True,
                            icon_name="mdi-plus",
                        )

                        if last_code.value:
                            solara.Button(
                                "Code" if not show_code else "Hide Code",
                                on_click=lambda: set_show_code(not show_code),
                                text=True,
                                icon_name="mdi-code-tags",
                            )

                if show_code and last_code.value:
                    solara.Preformatted(
                        last_code.value,
                        style={
                            "max-height": "200px",
                            "overflow-y": "auto",
                            "font-size": "12px",
                        },
                    )

    # ── Main content: map ──
    with solara.Column(
        style={
            "width": "100%",
            "height": "750px",
            "min_height": "750px",
            "isolation": "isolate",
        }
    ):
        solara.display(m.container if getattr(m, "container", None) is not None else m)
def _chat_with_status(agent: GeoAgent, query: str, target_map, status_callback=None):
    if status_callback is None:
        return agent.chat(query, target_map=target_map)
    try:
        sig = inspect.signature(agent.chat)
        params = sig.parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in params.values()
        )
        if "status_callback" in params or accepts_kwargs:
            return agent.chat(
                query, target_map=target_map, status_callback=status_callback
            )
    except (TypeError, ValueError):
        pass
    return agent.chat(query, target_map=target_map)
