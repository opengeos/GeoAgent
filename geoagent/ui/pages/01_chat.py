"""GeoAgent Chat UI — Solara sidebar + persistent map."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional

import solara
import leafmap.maplibregl as leafmap

from geoagent.core.agent import GeoAgent
from geoagent.core.llm import get_llm, PROVIDERS


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

messages: solara.Reactive[List[Dict[str, str]]] = solara.reactive([])
provider: solara.Reactive[str] = solara.reactive("ollama")
model: solara.Reactive[str] = solara.reactive("")
processing: solara.Reactive[bool] = solara.reactive(False)
status_text: solara.Reactive[str] = solara.reactive("")
last_code: solara.Reactive[str] = solara.reactive("")

# Store new map calls from query results to replay on the displayed map
new_map_calls: solara.Reactive[Optional[list]] = solara.reactive(None)

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
    return leafmap.Map(
        center=[0, 20],
        zoom=2,
        height="750px",
        style="dark-matter",
    )


def _run_query(query: str):
    """Run a GeoAgent query in a background thread.

    Does NOT use target_map. Instead, stores the result map's calls
    in a reactive so they can be replayed on the UI map in the render context.
    """
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

            # Store new calls from the result map (skip initial 4 control calls)
            if result.map and hasattr(result.map, "calls"):
                new_map_calls.value = list(result.map.calls[4:])
        else:
            text = f"❌ {result.error_message or 'An error occurred.'}"

        messages.value = [*messages.value, {"role": "assistant", "content": text}]

        # Store code
        code = result.code or ""
        if (
            not code
            and result.analysis
            and getattr(result.analysis, "code_generated", None)
        ):
            code = result.analysis.code_generated
        last_code.value = code

    except Exception as e:
        import traceback

        traceback.print_exc()
        messages.value = [
            *messages.value,
            {"role": "assistant", "content": f"❌ Error: {e}"},
        ]
    finally:
        processing.value = False
        status_text.value = ""


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@solara.component
def Page():
    # Persist map across re-renders
    m = solara.use_memo(_make_map, dependencies=[])

    query, set_query = solara.use_state("")
    show_code, set_show_code = solara.use_state(False)

    # Replay new map calls onto the displayed map (runs in render context)
    calls = new_map_calls.value
    if calls is not None:
        for call in calls:
            method_name = call[0]
            args = call[1] if len(call) > 1 else ()
            m.calls = m.calls + [call]
        new_map_calls.value = None

    with solara.Sidebar():
        solara.Markdown("### 💬 GeoAgent Chat")

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

        solara.Markdown("---")

        # Chat messages
        for msg in messages.value:
            icon = "🧑" if msg["role"] == "user" else "🌍"
            solara.Markdown(f"{icon} {msg['content']}")

        # Status
        if status_text.value:
            solara.Text(status_text.value)

        solara.Markdown("---")

        # Input
        solara.InputText(
            label="Ask about geospatial data…",
            value=query,
            on_value=set_query,
            disabled=processing.value,
        )

        def on_send():
            q = query.strip()
            if q and not processing.value:
                set_query("")
                threading.Thread(target=_run_query, args=(q,), daemon=True).start()

        solara.Button(
            "Send",
            on_click=on_send,
            disabled=processing.value or not query.strip(),
            color="primary",
        )

        # Code toggle
        if last_code.value:
            solara.Button(
                "Show Code" if not show_code else "Hide Code",
                on_click=lambda: set_show_code(not show_code),
                text=True,
            )
            if show_code:
                solara.Preformatted(last_code.value)

        # New Chat
        def on_new_chat():
            messages.value = []
            last_code.value = ""
            status_text.value = ""

        solara.Button("New Chat", on_click=on_new_chat, outlined=True)

    # Map as sole main content
    m.element()
