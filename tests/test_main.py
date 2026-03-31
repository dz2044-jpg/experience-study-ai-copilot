import importlib
import sys
import time
from contextlib import contextmanager


class _DummyComponentsV1:
    def __init__(self, calls):
        self.calls = calls

    def html(self, html, height=None, scrolling=False):
        self.calls.append(("html", {"html": html, "height": height, "scrolling": scrolling}))


class _DummyContext:
    def __init__(self, calls, kind, payload=None):
        self.calls = calls
        self.kind = kind
        self.payload = payload

    def __enter__(self):
        self.calls.append((self.kind, self.payload))
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyPlaceholder:
    def __init__(self, calls):
        self.calls = calls

    def markdown(self, text):
        self.calls.append(("placeholder_markdown", text))

    def write_stream(self, stream):
        chunks = list(stream)
        self.calls.append(("write_stream", chunks))
        return "".join(chunks)


class _DummyStatus:
    def __init__(self, calls, label, expanded):
        self.calls = calls
        self.calls.append(("status", {"label": label, "expanded": expanded}))

    def update(self, label=None, state=None, expanded=None):
        self.calls.append(("status_update", {"label": label, "state": state, "expanded": expanded}))

    def write(self, text):
        self.calls.append(("status_write", text))


class _DummyStreamlit:
    def __init__(self):
        self.session_state = {}
        self.calls = []
        self.chat_input_value = None
        self.button_values = {}
        self.components = type("DummyComponents", (), {"v1": _DummyComponentsV1(self.calls)})()

    @property
    def sidebar(self):
        return _DummyContext(self.calls, "sidebar")

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, text):
        self.calls.append(("title", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def markdown(self, text):
        self.calls.append(("markdown", text))

    def info(self, text):
        self.calls.append(("info", text))

    def success(self, text):
        self.calls.append(("success", text))

    def warning(self, text):
        self.calls.append(("warning", text))

    def chat_input(self, prompt):
        self.calls.append(("chat_input", prompt))
        return self.chat_input_value

    def button(self, label, key=None, **kwargs):
        self.calls.append(("button", {"label": label, "key": key, **kwargs}))
        if key in self.button_values:
            return self.button_values[key]
        if label in self.button_values:
            return self.button_values[label]
        return False

    def empty(self):
        self.calls.append(("empty", None))
        return _DummyPlaceholder(self.calls)

    def status(self, label, expanded=False):
        return _DummyStatus(self.calls, label, expanded)

    @contextmanager
    def chat_message(self, role, avatar=None):
        self.calls.append(("chat_message", {"role": role, "avatar": avatar}))
        yield

    def expander(self, label, expanded=False):
        return _DummyContext(self.calls, "expander", {"label": label, "expanded": expanded})

    def container(self, border=False):
        return _DummyContext(self.calls, "container", {"border": border})

    def columns(self, spec):
        self.calls.append(("columns", spec))
        count = spec if isinstance(spec, int) else len(spec)
        return [_DummyContext(self.calls, "column", index) for index in range(count)]

    def write_stream(self, stream):
        chunks = list(stream)
        self.calls.append(("write_stream", chunks))
        return "".join(chunks)

    def rerun(self):
        self.calls.append(("rerun", None))


def _load_main(monkeypatch, dummy_st):
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)
    main = importlib.import_module("main")
    main = importlib.reload(main)
    monkeypatch.setattr(main.time, "sleep", lambda _: None)
    return main


def test_render_app_constructs_streamlit_shell(monkeypatch):
    dummy_st = _DummyStreamlit()
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert "orchestrator" in dummy_st.session_state
    assert any(call[0] == "title" and call[1] == "Experience Study AI Copilot" for call in dummy_st.calls)
    assert any(call[0] == "sidebar" for call in dummy_st.calls)
    assert any(
        call[0] == "set_page_config"
        and call[1] == {
            "page_title": "Experience Study AI Copilot",
            "layout": "wide",
            "page_icon": "📊",
        }
        for call in dummy_st.calls
    )
    assert (
        "Ask the copilot to profile data, run a sweep (e.g., 'Sweep Gender where Issue_Age > 50'), or generate a chart."
        in [call[1] for call in dummy_st.calls if call[0] == "chat_input"]
    )


def test_render_app_renders_empty_state_when_history_is_empty(monkeypatch):
    dummy_st = _DummyStreamlit()
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    markdown_calls = [call[1] for call in dummy_st.calls if call[0] == "markdown"]
    assert "### 👋 Welcome to the Actuarial AI Copilot" in markdown_calls
    assert any(call[0] == "columns" and call[1] == 3 for call in dummy_st.calls)
    assert any(call[0] == "info" and "Profile Data" in call[1] for call in dummy_st.calls)
    assert any(call[0] == "success" and "Run an A/E Sweep" in call[1] for call in dummy_st.calls)
    assert any(call[0] == "warning" and "Visualize" in call[1] for call in dummy_st.calls)


def test_render_app_keeps_welcome_guidance_visible_when_history_exists(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.session_state["history"] = [{"prompt": "existing", "response": "answer"}]
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert any(call[0] == "info" and "Profile Data" in call[1] for call in dummy_st.calls)
    assert any(call[0] == "success" and "Run an A/E Sweep" in call[1] for call in dummy_st.calls)
    assert any(call[0] == "warning" and "Visualize" in call[1] for call in dummy_st.calls)


def test_render_app_renders_history_in_chronological_order(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.session_state["history"] = [
        {"prompt": "first prompt", "response": "first response"},
        {"prompt": "second prompt", "response": "second response"},
    ]
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    markdown_calls = [call[1] for call in dummy_st.calls if call[0] == "markdown"]
    relevant_calls = [text for text in markdown_calls if text in {"first prompt", "first response", "second prompt", "second response"}]
    assert relevant_calls == ["first prompt", "first response", "second prompt", "second response"]
    assert not any(call[0] == "write_stream" for call in dummy_st.calls)


def test_render_app_renders_history_visualization_from_saved_metadata(monkeypatch, tmp_path):
    dummy_st = _DummyStreamlit()
    html_path = tmp_path / "history_chart.html"
    html_path.write_text("<div>history chart</div>", encoding="utf-8")
    dummy_st.session_state["history"] = [
        {
            "prompt": "show me a visualization",
            "response": f"Visualization report generated: {html_path}",
            "visualization_path": str(html_path),
        }
    ]
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert ("container", {"border": True}) in dummy_st.calls
    assert ("expander", {"label": "Preview Visualization Inline", "expanded": False}) in dummy_st.calls
    assert any(call[0] == "button" and call[1]["label"] == "🌐 Open in Browser" for call in dummy_st.calls)
    assert any(call[0] == "caption" and "`history_chart.html`" in call[1] for call in dummy_st.calls)
    html_calls = [call for call in dummy_st.calls if call[0] == "html"]
    assert len(html_calls) == 1
    assert html_calls[0][1]["html"] == "<div>history chart</div>"
    assert html_calls[0][1]["height"] == 600
    assert html_calls[0][1]["scrolling"] is True


def test_render_app_uses_unique_button_keys_for_repeated_visualization_paths(monkeypatch, tmp_path):
    dummy_st = _DummyStreamlit()
    html_path = tmp_path / "shared_chart.html"
    html_path.write_text("<div>shared chart</div>", encoding="utf-8")
    dummy_st.session_state["history"] = [
        {
            "prompt": "first visualization",
            "response": f"Visualization report generated: {html_path}",
            "visualization_path": str(html_path),
        },
        {
            "prompt": "second visualization",
            "response": f"Visualization report generated: {html_path}",
            "visualization_path": str(html_path),
        },
    ]
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    button_calls = [call[1] for call in dummy_st.calls if call[0] == "button" and call[1]["label"] == "🌐 Open in Browser"]
    assert len(button_calls) == 2
    assert button_calls[0]["key"] != button_calls[1]["key"]


def test_render_app_open_button_launches_browser(monkeypatch, tmp_path):
    dummy_st = _DummyStreamlit()
    dummy_st.button_values["🌐 Open in Browser"] = True
    html_path = tmp_path / "chart.html"
    html_path.write_text("<div>chart</div>", encoding="utf-8")
    main = _load_main(monkeypatch, dummy_st)

    opened = {}

    def fake_open(target: str) -> None:
        opened["target"] = target

    monkeypatch.setattr(main.webbrowser, "open", fake_open)
    main._render_assistant_response(
        f"Visualization report generated: {html_path}",
        visualization_path=str(html_path),
        widget_key_prefix="manual-test",
    )

    assert opened["target"] == html_path.resolve().as_uri()


def test_render_app_processes_chat_input_and_appends_history(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.chat_input_value = "run a 1-way sweep"
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert dummy_st.session_state["history"] == [
        {
            "prompt": "run a 1-way sweep",
            "response": "processed: run a 1-way sweep",
            "visualization_path": None,
        }
    ]
    assert any(call[0] == "markdown" and call[1] == "run a 1-way sweep" for call in dummy_st.calls)
    write_stream_calls = [call[1] for call in dummy_st.calls if call[0] == "write_stream"]
    assert len(write_stream_calls) == 1
    assert "".join(write_stream_calls[0]) == "processed: run a 1-way sweep"
    assert any(call[0] == "empty" for call in dummy_st.calls)
    assert any(call[0] == "status" for call in dummy_st.calls)
    chat_messages = [call[1] for call in dummy_st.calls if call[0] == "chat_message"]
    assert chat_messages == [
        {"role": "user", "avatar": "👤"},
        {"role": "assistant", "avatar": "📊"},
    ]


def test_render_app_processes_visualization_response_and_stores_metadata(monkeypatch, tmp_path):
    dummy_st = _DummyStreamlit()
    dummy_st.chat_input_value = "generate a visualization"
    html_path = tmp_path / "chart.html"
    html_path.write_text("<div>chart</div>", encoding="utf-8")
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"Visualization report generated: {html_path}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert dummy_st.session_state["history"] == [
        {
            "prompt": "generate a visualization",
            "response": f"Visualization report generated: {html_path}",
            "visualization_path": str(html_path),
        }
    ]
    assert ("expander", {"label": "Preview Visualization Inline", "expanded": False}) in dummy_st.calls
    assert any(call[0] == "button" and call[1]["label"] == "🌐 Open in Browser" for call in dummy_st.calls)
    html_calls = [call for call in dummy_st.calls if call[0] == "html"]
    assert len(html_calls) == 1
    assert html_calls[0][1]["height"] == 600
    write_stream_calls = [call[1] for call in dummy_st.calls if call[0] == "write_stream"]
    assert len(write_stream_calls) == 1
    assert "".join(write_stream_calls[0]) == f"Visualization report generated: {html_path}"


def test_render_app_ignores_missing_visualization_path(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.chat_input_value = "run a 1-way sweep"
    main = _load_main(monkeypatch, dummy_st)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return "processed: run a 1-way sweep"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert not any(call[0] == "html" for call in dummy_st.calls)
    assert not any(
        call[0] == "button" and call[1]["label"] == "🌐 Open in Browser" for call in dummy_st.calls
    )
    write_stream_calls = [call[1] for call in dummy_st.calls if call[0] == "write_stream"]
    assert len(write_stream_calls) == 1
    assert "".join(write_stream_calls[0]) == "processed: run a 1-way sweep"


def test_render_app_clear_conversation_resets_session_and_reruns(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.button_values["🗑️ Clear Conversation"] = True
    existing_orchestrator = object()
    dummy_st.session_state["history"] = [{"prompt": "existing", "response": "response"}]
    dummy_st.session_state["orchestrator"] = existing_orchestrator
    main = _load_main(monkeypatch, dummy_st)

    created = []

    class DummyOrchestrator:
        def __init__(self):
            created.append(self)

        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert dummy_st.session_state["history"] == []
    assert dummy_st.session_state["orchestrator"] is created[0]
    assert dummy_st.session_state["orchestrator"] is not existing_orchestrator
    assert ("rerun", None) in dummy_st.calls
    assert not any(call[0] == "chat_input" for call in dummy_st.calls)


def test_stream_response_chunks_keeps_structured_markdown_intact(monkeypatch):
    dummy_st = _DummyStreamlit()
    main = _load_main(monkeypatch, dummy_st)

    response = (
        "This is a streamed response with a little pacing.\n\n"
        "| Rank | Cohort |\n"
        "| --- | --- |\n"
        "| 1 | Smoker=Yes |\n"
        "```python\n"
        "print('hello')\n"
        "```\n"
    )

    chunks = list(main._stream_response_chunks(response))

    assert "".join(chunks) == response
    assert chunks[:8] == [
        "This ",
        "is ",
        "a ",
        "streamed ",
        "response ",
        "with ",
        "a ",
        "little ",
    ]
    assert "pacing.\n" in chunks
    assert "\n" in chunks
    assert "| Rank | Cohort |\n" in chunks
    assert "```python\n" in chunks
    assert "print('hello')\n" in chunks


def test_stream_response_with_pacing_uses_natural_pauses(monkeypatch):
    dummy_st = _DummyStreamlit()
    main = _load_main(monkeypatch, dummy_st)

    pauses = []
    response = "Hello, world.\n\nNext line.\n"

    chunks = list(main._stream_response_with_pacing(response, sleep_fn=pauses.append))

    assert "".join(chunks) == response
    assert pauses == [0.07, 0.12, 0.18, 0.035, 0.12]


def test_wait_for_response_with_status_updates_live_status(monkeypatch):
    dummy_st = _DummyStreamlit()
    main = _load_main(monkeypatch, dummy_st)

    original_sleep = time.sleep

    class SlowOrchestrator:
        def __init__(self):
            self.status_callback = None

        def set_status_callback(self, callback):
            self.status_callback = callback

        def process_query(self, prompt):
            if self.status_callback:
                self.status_callback("Orchestrator: classifying intent with local heuristics.")
            original_sleep(0.05)
            if self.status_callback:
                self.status_callback("Orchestrator: routing request to Lead Actuary.")
            return f"processed: {prompt}"

    status_panel = _DummyStatus(dummy_st.calls, "Starting orchestrator...", True)
    response = main._wait_for_response_with_status(SlowOrchestrator(), "stream this", status_panel)

    assert response == "processed: stream this"
    status_writes = [call[1] for call in dummy_st.calls if call[0] == "status_write"]
    assert status_writes == [
        "Orchestrator: classifying intent with local heuristics.",
        "Orchestrator: routing request to Lead Actuary.",
    ]
    status_updates = [call[1] for call in dummy_st.calls if call[0] == "status_update"]
    assert status_updates[0]["state"] == "running"
    assert status_updates[-1]["state"] == "complete"
