"""Streamlit entry point for the unified skill-based experience study copilot."""

from __future__ import annotations

from pathlib import Path
import webbrowser

from core.copilot_agent import CopilotEvent, UnifiedCopilot

try:
    import streamlit as st
except ImportError:  # pragma: no cover - depends on environment
    st = None  # type: ignore[assignment]


EMPTY_STATE_SUGGESTIONS = (
    (
        "info",
        "**Profile Data**\n\n`Profile data/input/example.csv and tell me the columns.`",
    ),
    (
        "success",
        "**Run a Sweep**\n\n`Run a 1-way sweep on Gender after profiling my dataset.`",
    ),
    (
        "warning",
        "**Visualize**\n\n`Generate the combined report for the latest sweep.`",
    ),
)


def _require_streamlit() -> None:
    if st is None:  # pragma: no cover - depends on environment
        raise RuntimeError("Streamlit is required to run the web app.")


def _render_empty_state() -> None:
    _require_streamlit()
    st.markdown("### Experience Study Copilot")
    st.markdown(
        "This session uses a unified skill package with session-scoped deterministic "
        "artifacts for profiling, feature engineering, dimensional sweeps, and visualization."
    )
    columns = st.columns(3)
    for column, (variant, message) in zip(columns, EMPTY_STATE_SUGGESTIONS):
        with column:
            getattr(st, variant)(message)
    st.markdown("---")


def _render_visualization_card(visualization_path: str | None, widget_key_prefix: str) -> None:
    _require_streamlit()
    if not visualization_path:
        return
    html_path = Path(visualization_path)
    if not html_path.exists():
        return

    resolved_path = html_path.resolve()
    with st.container(border=True):
        st.markdown("##### Generated Visualization Artifact")
        st.caption(f"`{resolved_path.name}`")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button(
                "Open in Browser",
                key=f"{widget_key_prefix}-open-visualization",
            ):
                webbrowser.open(resolved_path.as_uri())
        with col2:
            st.caption("Open the standalone report or preview it inline below.")
        with st.expander("Preview Visualization Inline", expanded=False):
            st.components.v1.html(
                resolved_path.read_text(encoding="utf-8"),
                height=600,
                scrolling=True,
            )


def _render_sidebar() -> bool:
    _require_streamlit()
    with st.sidebar:
        st.title("Copilot Controls")
        st.caption(f"Session: `{st.session_state['session_id']}`")
        if st.button("Clear Conversation", type="primary", use_container_width=True):
            st.session_state["copilot"].reset_session()
            st.session_state["session_id"] = st.session_state["copilot"].state.session_id
            st.session_state["history"] = []
            st.rerun()
            return True
    return False


def _consume_copilot_events(
    events: list[CopilotEvent],
    *,
    status_panel,
    response_placeholder,
) -> tuple[str, str | None]:
    _require_streamlit()
    rendered_response = ""
    latest_visualization_path: str | None = None
    latest_status = "Processing request..."

    for event in events:
        if event.type == "status":
            latest_status = event.message
            status_panel.write(event.message)
            status_panel.update(label=event.message, state="running", expanded=True)
        elif event.type == "tool_start":
            status_panel.write(event.message)
        elif event.type == "tool_result":
            result = event.data.get("result", {})
            if not result.get("ok", False):
                status_panel.write(event.message)
        elif event.type == "artifact_update":
            latest_visualization_path = event.data.get("latest_visualization_path") or latest_visualization_path
        elif event.type == "text_delta":
            rendered_response += event.message
            response_placeholder.markdown(rendered_response)
        elif event.type == "final":
            latest_status = "Completed request."
            payload = event.data.get("artifact_state", {})
            latest_visualization_path = (
                payload.get("latest_visualization_path") or latest_visualization_path
            )
            if not rendered_response:
                rendered_response = event.message
                response_placeholder.markdown(rendered_response)

    status_panel.update(label=latest_status, state="complete", expanded=False)
    return rendered_response, latest_visualization_path


def render_app() -> None:
    _require_streamlit()
    st.set_page_config(
        page_title="Experience Study Copilot",
        page_icon="📊",
        layout="wide",
    )

    if "session_id" not in st.session_state:
        st.session_state["session_id"] = UnifiedCopilot.new_session_id()
    if "copilot" not in st.session_state:
        st.session_state["copilot"] = UnifiedCopilot(
            session_id=st.session_state["session_id"]
        )
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if _render_sidebar():
        return

    st.title("Digital Process: Experience Study")
    st.caption(
        "Unified skill-based copilot with session-scoped deterministic artifacts and "
        "dynamic tool gating."
    )
    _render_empty_state()

    for index, item in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.markdown(item["prompt"])
        with st.chat_message("assistant"):
            st.markdown(item["response"])
            _render_visualization_card(
                item.get("visualization_path"),
                widget_key_prefix=f"history-{index}",
            )

    prompt = st.chat_input(
        "Ask the copilot to profile data, engineer features, run a sweep, or generate the combined report."
    )
    if not prompt or not prompt.strip():
        return

    cleaned_prompt = prompt.strip()
    with st.chat_message("user"):
        st.markdown(cleaned_prompt)
    with st.chat_message("assistant"):
        status_panel = st.status("Starting copilot...", expanded=True)
        response_placeholder = st.empty()
        events = list(st.session_state["copilot"].process_message(cleaned_prompt))
        response, visualization_path = _consume_copilot_events(
            events,
            status_panel=status_panel,
            response_placeholder=response_placeholder,
        )
        _render_visualization_card(visualization_path, widget_key_prefix="current")

    st.session_state["history"].append(
        {
            "prompt": cleaned_prompt,
            "response": response,
            "visualization_path": visualization_path,
        }
    )


def main() -> None:
    render_app()


if __name__ == "__main__":
    main()
