"""Minimal Streamlit entry point for the Experience Study AI Copilot."""

from agents.orchestrator import StudyOrchestrator

try:
    import streamlit as st
except ImportError:  # pragma: no cover - depends on environment
    st = None  # type: ignore[assignment]


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
            st.markdown(item["response"])

    prompt = st.chat_input("Ask the copilot to profile data, run a sweep, or generate a chart.")
    if prompt and prompt.strip():
        cleaned_prompt = prompt.strip()
        with st.chat_message("user"):
            st.markdown(cleaned_prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state["orchestrator"].process_query(cleaned_prompt)
            st.markdown(response)
        st.session_state["history"].append({"prompt": cleaned_prompt, "response": response})


def main() -> None:
    """Application entry point."""
    render_app()


if __name__ == "__main__":
    main()
