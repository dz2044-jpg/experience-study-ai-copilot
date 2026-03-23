"""Minimal Streamlit entry point for the Experience Study AI Copilot."""

from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue
import re
import time
import webbrowser
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

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


def _stream_response_chunks(response: str, words_per_chunk: int = 1) -> Iterator[str]:
    """Yield markdown-safe chunks for UI-only streaming."""
    in_fenced_block = False

    for line in response.splitlines(keepends=True):
        stripped = line.strip()
        is_fence = stripped.startswith("```")
        is_table = stripped.startswith("|")
        is_code_like = line.startswith("    ") or line.startswith("\t")

        if is_fence or in_fenced_block or is_table or is_code_like or stripped == "":
            yield line
            if is_fence:
                in_fenced_block = not in_fenced_block
            continue

        words = re.findall(r"\S+\s*", line)
        if not words:
            yield line
            continue

        for index in range(0, len(words), words_per_chunk):
            yield "".join(words[index : index + words_per_chunk])


def _stream_pause_seconds(chunk: str) -> float:
    """Return a natural-feeling pause for a streamed chunk."""
    stripped = chunk.strip()
    if not stripped:
        if "\n" in chunk:
            return 0.18
        return 0.03
    if stripped.endswith((".", "!", "?")):
        return 0.12
    if stripped.endswith((",", ":", ";")):
        return 0.07
    if chunk.endswith("\n"):
        return 0.1
    return 0.035


def _stream_response_with_pacing(
    response: str,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Yield streamed response chunks with natural pauses between them."""
    for chunk in _stream_response_chunks(response):
        yield chunk
        sleep_fn(_stream_pause_seconds(chunk))


def _stream_markdown_via_placeholder(
    target: Any,
    response: str,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> str:
    """Fallback renderer for containers that lack write_stream support."""
    rendered_text = ""
    for chunk in _stream_response_with_pacing(response, sleep_fn=sleep_fn):
        rendered_text += chunk
        target.markdown(rendered_text)
    return rendered_text


def _render_streaming_assistant_text(response: str, target: Optional[Any] = None) -> str:
    """Render the active assistant turn with streaming when supported."""
    if st is None:
        raise RuntimeError("Streamlit is required to run the web app. Install project dependencies first.")

    render_target = target or st

    if not hasattr(render_target, "write_stream"):
        if hasattr(render_target, "markdown"):
            return _stream_markdown_via_placeholder(render_target, response)
        render_target.markdown(response)
        return response

    streamed_response = render_target.write_stream(_stream_response_with_pacing(response))
    if isinstance(streamed_response, str):
        return streamed_response
    if streamed_response is None:
        if hasattr(render_target, "markdown"):
            return _stream_markdown_via_placeholder(render_target, response)
        return response
    return "".join(str(chunk) for chunk in streamed_response)


def _wait_for_response_with_status(orchestrator: StudyOrchestrator, prompt: str, status_panel: Any) -> str:
    """Render orchestrator and agent progress updates inside a live status panel."""
    status_events: Queue[str] = Queue()

    def record_status(message: str) -> None:
        if message:
            status_events.put(message)

    if hasattr(orchestrator, "set_status_callback"):
        orchestrator.set_status_callback(record_status)

    status_panel.update(label="Working through the request...", state="running", expanded=True)
    latest_status = "Working through the request..."

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(orchestrator.process_query, prompt)

            while not future.done():
                while True:
                    try:
                        latest_status = status_events.get_nowait()
                    except Empty:
                        break
                    status_panel.write(latest_status)
                    status_panel.update(label=latest_status, state="running", expanded=True)
                time.sleep(0.05)

            response = future.result()
    finally:
        if hasattr(orchestrator, "set_status_callback"):
            orchestrator.set_status_callback(None)

    while True:
        try:
            latest_status = status_events.get_nowait()
        except Empty:
            break
        status_panel.write(latest_status)

    status_panel.update(label=latest_status, state="complete", expanded=False)
    return response


def _render_assistant_response(
    response: str,
    visualization_path: Optional[str] = None,
    widget_key_prefix: str = "assistant",
    render_text: bool = True,
) -> None:
    """Render assistant text and, when available, embed the generated HTML chart."""
    if st is None:
        raise RuntimeError("Streamlit is required to run the web app. Install project dependencies first.")

    if render_text:
        st.markdown(response)

    resolved_path = visualization_path or _extract_visualization_path(response)
    if not resolved_path:
        return

    html_path = Path(resolved_path)
    if not html_path.exists():
        return

    resolved_html_path = html_path.resolve()
    if st.button("Open in browser", key=f"{widget_key_prefix}-open-visualization-{resolved_html_path}"):
        webbrowser.open(resolved_html_path.as_uri())
    st.caption(f"Saved HTML artifact: {resolved_html_path}")

    with st.expander("View visualization"):
        st.components.v1.html(resolved_html_path.read_text(encoding="utf-8"), height=1400, scrolling=True)


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

    for idx, item in enumerate(st.session_state["history"]):
        with st.chat_message("user"):
            st.markdown(item["prompt"])
        with st.chat_message("assistant"):
            _render_assistant_response(
                item["response"],
                item.get("visualization_path"),
                widget_key_prefix=f"history-{idx}",
            )

    prompt = st.chat_input(
        "Ask the copilot to profile data, run a sweep (e.g., 'Sweep Gender where Issue_Age > 50'), or generate a chart."
    )
    if prompt and prompt.strip():
        cleaned_prompt = prompt.strip()
        with st.chat_message("user"):
            st.markdown(cleaned_prompt)
        with st.chat_message("assistant"):
            status_panel = st.status("Starting orchestrator...", expanded=True)
            response_placeholder = st.empty()
            response = _wait_for_response_with_status(
                st.session_state["orchestrator"],
                cleaned_prompt,
                status_panel,
            )
            rendered_response = _render_streaming_assistant_text(response, target=response_placeholder)
            visualization_path = _extract_visualization_path(rendered_response)
            _render_assistant_response(
                rendered_response,
                visualization_path,
                widget_key_prefix="current",
                render_text=False,
            )
        st.session_state["history"].append(
            {
                "prompt": cleaned_prompt,
                "response": rendered_response,
                "visualization_path": visualization_path,
            }
        )


def main() -> None:
    """Application entry point."""
    render_app()


if __name__ == "__main__":
    main()
