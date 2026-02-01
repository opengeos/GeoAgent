"""Streamlit chat UI for GeoAgent.

Features:
- Clean chat interface using st.chat_input / st.chat_message
- Sidebar with provider selector, model input, and New Chat button
- Assistant responses show summary text, interactive map, generated code
- Data summary line with items found and execution time
- Uses st.session_state to persist chat history and agent instance
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st
from streamlit.components.v1 import html as st_html

from geoagent.core.agent import GeoAgent
from geoagent.core.llm import get_llm, PROVIDERS
from geoagent.core.models import GeoAgentResponse


PAGE_TITLE = "GeoAgent"
PAGE_ICON = "🌍"
MAP_HEIGHT_PX = 500


def _default_model(provider: str) -> str:
    provider = (provider or "openai").lower()
    if provider in PROVIDERS:
        return PROVIDERS[provider]["default_model"]
    return "gpt-4.1"


def _ensure_state():
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, Any]] = []
    if "provider" not in st.session_state:
        st.session_state.provider = "openai"
    if "model" not in st.session_state:
        st.session_state.model = _default_model(st.session_state.provider)
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "_agent_key" not in st.session_state:
        st.session_state._agent_key = None


def _make_agent(provider: str, model: str) -> GeoAgent:
    # Create specific LLM for explicit provider/model, then wrap in GeoAgent
    llm = get_llm(provider=provider, model=model)
    return GeoAgent(llm=llm, provider=provider, model=model)


def _maybe_recreate_agent(provider: str, model: str):
    key = f"{provider}:{model}"
    if st.session_state._agent_key != key or st.session_state.agent is None:
        st.session_state.agent = _make_agent(provider, model)
        st.session_state._agent_key = key


def _sidebar():
    st.sidebar.markdown("### Settings")

    provider = st.sidebar.selectbox(
        "Provider",
        options=list(PROVIDERS.keys()),
        index=list(PROVIDERS.keys()).index(st.session_state.provider)
        if st.session_state.provider in PROVIDERS
        else 0,
        key="provider",
    )

    # Suggest default model when provider changes, but allow custom text
    suggested = _default_model(provider)
    model = st.sidebar.text_input(
        "Model",
        value=st.session_state.model or suggested,
        placeholder=suggested,
        key="model",
    )

    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("New Chat", use_container_width=True):
            st.session_state.messages = []

    with col2:
        st.write("")

    # Recreate agent when provider/model change
    try:
        _maybe_recreate_agent(provider, model)
    except Exception as e:
        st.sidebar.error(f"Failed to initialize agent: {e}")


def _render_assistant_block(result: GeoAgentResponse):
    # Summary text
    if result.success:
        items = (result.data.total_items if result.data else 0) or 0
        intent = getattr(result.plan, "intent", "").strip() if result.plan else ""
        dataset = getattr(result.plan, "dataset", "") if result.plan else ""
        loc = ""
        if result.plan and getattr(result.plan, "location", None):
            loc_field = result.plan.location
            name = loc_field.get("name") if isinstance(loc_field, dict) else None
            loc = f" • {name}" if name else ""
        st.write(
            f"Intent: {intent or 'analysis'} • Dataset: {dataset or 'auto'}{loc}"
        )
    else:
        st.error(result.error_message or "An error occurred.")

    # Map (if available)
    try:
        if result.map is not None and hasattr(result.map, "to_html"):
            map_html = result.map.to_html()
            st_html(map_html, height=MAP_HEIGHT_PX)
    except Exception as e:
        st.warning(f"Map rendering failed: {e}")

    # Data summary line
    try:
        items = (result.data.total_items if result.data else 0) or 0
    except Exception:
        items = 0
    exec_time = result.execution_time or 0.0
    st.caption(f"Items: {items} • Time: {exec_time:.2f}s")

    # Generated code
    code_text = result.code or (
        result.analysis.code_generated if result.analysis and getattr(result.analysis, "code_generated", None) else ""
    )
    if code_text:
        with st.expander("Generated Code", expanded=False):
            st.code(code_text, language="python")


def _render_history():
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            if role == "user":
                st.write(msg.get("content", ""))
            else:
                # Assistant message contains result and maybe text
                text = msg.get("content")
                if text:
                    st.write(text)
                result: Optional[GeoAgentResponse] = msg.get("result")
                if isinstance(result, GeoAgentResponse):
                    _render_assistant_block(result)


def main():
    st.set_page_config(layout="wide", page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    st.title(PAGE_TITLE)

    _ensure_state()
    _sidebar()

    # Render prior messages
    _render_history()

    # Chat input
    prompt = st.chat_input("Ask GeoAgent about geospatial data…")
    if prompt:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            try:
                agent: GeoAgent = st.session_state.agent
                if agent is None:
                    raise RuntimeError("Agent is not initialized.")

                result = agent.chat(prompt)

                # Append assistant message to history with the result
                st.session_state.messages.append(
                    {"role": "assistant", "content": "", "result": result}
                )

                # Render now
                _render_assistant_block(result)

            except Exception as e:
                error_text = f"Error: {e}"
                st.error(error_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_text}
                )


if __name__ == "__main__":
    main()

