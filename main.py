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

    with st.form("copilot_prompt", clear_on_submit=True):
        prompt = st.text_area(
            "Request",
            placeholder="Example: Profile the inforce dataset, then continue to the next step.",
            height=140,
        )
        submitted = st.form_submit_button("Run")

    if submitted and prompt.strip():
        response = st.session_state["orchestrator"].process_query(prompt.strip())
        st.session_state["history"].append({"prompt": prompt.strip(), "response": response})

    for item in reversed(st.session_state["history"]):
        st.markdown(f"**You**: {item['prompt']}")
        st.markdown(f"**Copilot**: {item['response']}")


def main() -> None:
    """Application entry point."""
    render_app()


if __name__ == "__main__":
    main()
