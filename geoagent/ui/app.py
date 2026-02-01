"""Streamlit chat UI for GeoAgent.

Features:
- Chat interface with progress status indicators
- Sidebar with provider/model selection (model updates with provider)
- Single map panel for the latest result (not per-message)
- Expandable generated code section
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import streamlit as st
from streamlit.components.v1 import html as st_html

from geoagent.core.agent import GeoAgent
from geoagent.core.llm import get_llm, PROVIDERS
from geoagent.core.models import GeoAgentResponse


PAGE_TITLE = "GeoAgent"
PAGE_ICON = "🌍"
MAP_HEIGHT_PX = 600


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def _ensure_state():
    defaults = {
        "messages": [],
        "agent": None,
        "_agent_key": None,
        "_prev_provider": None,
        "last_map_html": None,
        "last_code": None,
        "last_summary": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _make_agent(provider: str, model: str) -> GeoAgent:
    llm = get_llm(provider=provider, model=model)
    return GeoAgent(llm=llm, provider=provider, model=model)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _sidebar():
    st.sidebar.markdown("### Settings")

    provider_list = list(PROVIDERS.keys())
    provider = st.sidebar.selectbox("Provider", options=provider_list, key="provider")

    # Reset model when provider changes
    if st.session_state._prev_provider != provider:
        st.session_state._prev_provider = provider
        st.session_state["_model_input"] = PROVIDERS[provider]["default_model"]
    elif "_model_input" not in st.session_state:
        st.session_state["_model_input"] = PROVIDERS.get(provider, {}).get("default_model", "")

    model = st.sidebar.text_input("Model", key="_model_input")

    if st.sidebar.button("New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_map_html = None
        st.session_state.last_code = None
        st.session_state.last_summary = None
        st.rerun()

    # Recreate agent when provider/model change
    agent_key = f"{provider}:{model}"
    if st.session_state._agent_key != agent_key or st.session_state.agent is None:
        try:
            with st.sidebar:
                with st.spinner("Initializing agent…"):
                    st.session_state.agent = _make_agent(provider, model)
                    st.session_state._agent_key = agent_key
        except Exception as e:
            st.sidebar.error(f"Failed to initialize agent: {e}")

    return provider, model


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _render_chat_history():
    """Render chat messages — text only, no maps in history."""
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "user":
                st.write(msg["content"])
            else:
                text = msg.get("content", "")
                if text:
                    st.write(text)


def _build_summary(result: GeoAgentResponse) -> str:
    """Build a one-line text summary of a result."""
    if not result.success:
        return f"❌ {result.error_message or 'An error occurred.'}"

    parts = []
    if result.plan:
        intent = getattr(result.plan, "intent", "")
        if intent:
            parts.append(intent)
        dataset = getattr(result.plan, "dataset", "")
        if dataset:
            parts.append(dataset)
        loc = getattr(result.plan, "location", None)
        if isinstance(loc, dict) and loc.get("name"):
            parts.append(loc["name"])

    items = result.data.total_items if result.data else 0
    t = f"{result.execution_time:.1f}s" if result.execution_time else ""
    meta = f"{items} items" + (f" • {t}" if t else "")

    summary = " • ".join(parts) if parts else "Done"
    return f"✅ {summary} ({meta})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    _ensure_state()
    provider, model = _sidebar()

    # --- Layout: chat (left) | map (right) ---
    chat_col, map_col = st.columns([1, 1])

    with chat_col:
        st.markdown(f"### 💬 Chat")
        _render_chat_history()

        prompt = st.chat_input("Ask about geospatial data…")
        if prompt:
            # Append and display user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            # Process with progress
            with st.chat_message("assistant"):
                agent: GeoAgent = st.session_state.agent
                if agent is None:
                    st.error("Agent is not initialized. Check provider settings.")
                    st.session_state.messages.append(
                        {"role": "assistant", "content": "❌ Agent not initialized."}
                    )
                else:
                    with st.status("Processing query…", expanded=True) as status:
                        st.write("🔍 Parsing query and planning…")
                        try:
                            result = agent.chat(prompt)
                        except Exception as e:
                            result = None
                            error_text = f"❌ Error: {e}"
                            st.error(error_text)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": error_text}
                            )
                            status.update(label="Failed", state="error")

                        if result is not None:
                            # Update progress
                            if result.success:
                                st.write("📊 Data retrieved. Rendering map…")
                                status.update(
                                    label=f"Done in {result.execution_time:.1f}s",
                                    state="complete",
                                    expanded=False,
                                )
                            else:
                                status.update(
                                    label="Completed with errors", state="error"
                                )

                            summary = _build_summary(result)
                            st.write(summary)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": summary}
                            )

                            # Cache latest map/code for right panel
                            try:
                                if result.map is not None and hasattr(
                                    result.map, "to_html"
                                ):
                                    st.session_state.last_map_html = (
                                        result.map.to_html()
                                    )
                                else:
                                    st.session_state.last_map_html = None
                            except Exception:
                                st.session_state.last_map_html = None

                            code = result.code or (
                                result.analysis.code_generated
                                if result.analysis
                                and getattr(result.analysis, "code_generated", None)
                                else ""
                            )
                            st.session_state.last_code = code or None
                            st.session_state.last_summary = summary

    # --- Right panel: latest map + code ---
    with map_col:
        st.markdown("### 🗺️ Map")
        if st.session_state.last_map_html:
            st_html(st.session_state.last_map_html, height=MAP_HEIGHT_PX)
        else:
            st.info("Run a query to see results here.")

        if st.session_state.last_code:
            with st.expander("Generated Code", expanded=False):
                st.code(st.session_state.last_code, language="python")


if __name__ == "__main__":
    main()
