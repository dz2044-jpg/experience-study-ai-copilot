"""Minimal Streamlit entry point for the Experience Study AI Copilot."""

import re
from pathlib import Path
from typing import Optional

from agents.orchestrator import StudyOrchestrator

try:
    import streamlit as st
except ImportError:  # pragma: no cover - depends on environment
    st = None  # type: ignore[assignment]


VISUALIZATION_PATH_RE = re.compile(r"(?P<path>/\S+\.html)")


def _extract_visualization_path(response: str) -> Optional[str]:
    """Extract an absolute HTML artifact path from an assistant response."""
    match = VISUALIZATION_PATH_RE.search(response)
    if not match:
        return None
    return match.group("path")


def _render_assistant_response(response: str, visualization_path: Optional[str] = None) -> None:
    """Render assistant text and, when available, embed the generated HTML chart."""
    if st is None:
        raise RuntimeError("Streamlit is required to run the web app. Install project dependencies first.")

    st.markdown(response)

    resolved_path = visualization_path or _extract_visualization_path(response)
    if not resolved_path:
        return

    html_path = Path(resolved_path)
    if not html_path.exists():
        return

    with st.expander("View visualization"):
        st.components.v1.html(html_path.read_text(encoding="utf-8"), height=850, scrolling=True)


def render_app() -> None:
    """Render a thin Streamlit shell around the orchestrator."""
    if st is None:
        raise RuntimeError("Streamlit is required to run the web app. Install project dependencies first.")

    st.set_page_config(page_title="Experience Study AI Copilot", layout="wide")
    st.title("Experience Study AI Copilot")
    st.caption("Route actuarial data prep, A/E analysis, and visualization requests through the orchestrator.")

    if "orchestrator" not in st.session_state:
        st.session_state["orchestrator"] = StudyOrchestrator()
    if "history" not in st.session_state:
        st.session_state["history"] = []

    for item in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(item["prompt"])
        with st.chat_message("assistant"):
            _render_assistant_response(item["response"], item.get("visualization_path"))

    prompt = st.chat_input("Ask the copilot to profile data, run a sweep, or generate a chart.")
    if prompt and prompt.strip():
        cleaned_prompt = prompt.strip()
        with st.chat_message("user"):
            st.markdown(cleaned_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state["orchestrator"].process_query(cleaned_prompt)
            visualization_path = _extract_visualization_path(response)
            _render_assistant_response(response, visualization_path)
        st.session_state["history"].append(
            {
                "prompt": cleaned_prompt,
                "response": response,
                "visualization_path": visualization_path,
            }
        )


def main() -> None:
    """Application entry point."""
    render_app()


if __name__ == "__main__":
    main()
