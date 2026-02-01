"""GeoAgent Chat UI — map with chat sidebar."""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import ipywidgets as widgets
import solara
import leafmap.maplibregl as leafmap

from geoagent.core.agent import GeoAgent
from geoagent.core.llm import get_llm, PROVIDERS
from geoagent.core.models import GeoAgentResponse


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

messages: solara.Reactive[List[Dict[str, str]]] = solara.reactive([])
provider: solara.Reactive[str] = solara.reactive("openai")
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


def _run_query(query: str, m, output: widgets.Output, status_label: widgets.Label):
    """Run a GeoAgent query in a background thread."""
    processing.value = True
    status_label.value = "🔍 Parsing query…"
    messages.value = [*messages.value, {"role": "user", "content": query}]

    try:
        prov = provider.value
        mdl = model.value or _get_default_model(prov)
        status_label.value = "⚙️ Initializing agent…"
        agent = _get_or_create_agent(prov, mdl)

        status_label.value = "📡 Searching data & analyzing…"
        result = agent.chat(query, target_map=m)

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

        # Show result in output
        with output:
            print(f"🌍 {text}")

        # Store code
        code = result.code or ""
        if (
            not code
            and result.analysis
            and getattr(result.analysis, "code_generated", None)
        ):
            code = result.analysis.code_generated
        last_code.value = code
        status_label.value = ""

    except Exception as e:
        import traceback

        err_msg = f"❌ Error: {e}"
        messages.value = [
            *messages.value,
            {"role": "assistant", "content": err_msg},
        ]
        status_label.value = ""
        with output:
            print(f"🌍 {err_msg}")
            traceback.print_exc()
    finally:
        processing.value = False
        status_label.value = ""


def create_map():
    """Create the map with chat sidebar."""
    m = leafmap.Map(
        center=[0, 20],
        zoom=2,
        height="750px",
        style="dark-matter",
    )
    # Use container sidebar (not floating) so widgets appear in to_solara()
    m.add_floating_sidebar_flag = False
    m.create_container(sidebar_visible=True)

    # --- Chat widgets (ipywidgets, not Solara) ---
    output = widgets.Output(
        layout=widgets.Layout(max_height="400px", overflow_y="auto")
    )

    query_input = widgets.Text(
        placeholder="Ask about geospatial data…",
        layout=widgets.Layout(width="100%"),
    )

    provider_dropdown = widgets.Dropdown(
        options=PROVIDER_LIST,
        value=provider.value,
        description="Provider:",
        layout=widgets.Layout(width="100%"),
    )

    model_input = widgets.Text(
        value=model.value or _get_default_model(provider.value),
        description="Model:",
        layout=widgets.Layout(width="100%"),
    )

    status_label = widgets.Label(value="")

    def on_provider_change(change):
        provider.value = change["new"]
        model_input.value = _get_default_model(change["new"])

    provider_dropdown.observe(on_provider_change, names=["value"])

    def on_model_change(change):
        model.value = change["new"]

    model_input.observe(on_model_change, names=["value"])

    def do_submit(_=None):
        q = query_input.value.strip()
        if not q or processing.value:
            return
        # Show user message immediately
        with output:
            print(f"🧑 {q}")
        query_input.value = ""
        status_label.value = "⏳ Processing…"
        threading.Thread(
            target=_run_query,
            args=(q, m, output, status_label),
            daemon=True,
        ).start()

    query_input.on_submit(do_submit)

    send_button = widgets.Button(
        description="Send",
        button_style="primary",
        layout=widgets.Layout(width="100%"),
    )
    send_button.on_click(do_submit)

    chat_box = widgets.VBox(
        [
            widgets.HTML("<h3>💬 GeoAgent Chat</h3>"),
            provider_dropdown,
            model_input,
            widgets.HTML("<hr>"),
            output,
            widgets.HTML("<hr>"),
            query_input,
            send_button,
            status_label,
        ],
        layout=widgets.Layout(padding="8px"),
    )

    m.add_to_sidebar(chat_box, label="Chat", widget_icon="mdi-chat")
    return m


@solara.component
def Page():
    m = create_map()
    return m.to_solara()
