import importlib
import sys
from contextlib import contextmanager


class _DummyComponentsV1:
    def __init__(self, calls):
        self.calls = calls

    def html(self, html, height=None, scrolling=False):
        self.calls.append(("html", {"html": html, "height": height, "scrolling": scrolling}))


class _DummyStreamlit:
    def __init__(self):
        self.session_state = {}
        self.calls = []
        self.chat_input_value = None
        self.button_value = False
        self.components = type("DummyComponents", (), {"v1": _DummyComponentsV1(self.calls)})()

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, text):
        self.calls.append(("title", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def chat_input(self, prompt):
        self.calls.append(("chat_input", prompt))
        return self.chat_input_value

    def button(self, label, key=None):
        self.calls.append(("button", {"label": label, "key": key}))
        return self.button_value

    @contextmanager
    def chat_message(self, role):
        self.calls.append(("chat_message", role))
        yield

    @contextmanager
    def spinner(self, text):
        self.calls.append(("spinner", text))
        yield

    @contextmanager
    def expander(self, label):
        self.calls.append(("expander", label))
        yield

    def markdown(self, text):
        self.calls.append(("markdown", text))


def test_render_app_constructs_streamlit_shell(monkeypatch):
    dummy_st = _DummyStreamlit()
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert "orchestrator" in dummy_st.session_state
    assert any(call[0] == "title" for call in dummy_st.calls)


def test_render_app_renders_history_in_chronological_order(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.session_state["history"] = [
        {"prompt": "first prompt", "response": "first response"},
        {"prompt": "second prompt", "response": "second response"},
    ]
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    markdown_calls = [call[1] for call in dummy_st.calls if call[0] == "markdown"]
    assert markdown_calls == [
        "first prompt",
        "first response",
        "second prompt",
        "second response",
    ]


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
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert ("expander", "View visualization") in dummy_st.calls
    assert any(call[0] == "button" and call[1]["label"] == "Open in browser" for call in dummy_st.calls)
    assert any(call[0] == "caption" and "Saved HTML artifact:" in call[1] for call in dummy_st.calls)
    html_calls = [call for call in dummy_st.calls if call[0] == "html"]
    assert len(html_calls) == 1
    assert html_calls[0][1]["html"] == "<div>history chart</div>"
    assert html_calls[0][1]["height"] == 1400
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
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return f"processed: {prompt}"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    button_calls = [call[1] for call in dummy_st.calls if call[0] == "button"]
    assert len(button_calls) == 2
    assert button_calls[0]["key"] != button_calls[1]["key"]


def test_render_app_open_button_launches_browser(monkeypatch, tmp_path):
    dummy_st = _DummyStreamlit()
    dummy_st.button_value = True
    html_path = tmp_path / "chart.html"
    html_path.write_text("<div>chart</div>", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

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
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

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
    markdown_calls = [call[1] for call in dummy_st.calls if call[0] == "markdown"]
    assert markdown_calls == [
        "run a 1-way sweep",
        "processed: run a 1-way sweep",
    ]


def test_render_app_processes_visualization_response_and_stores_metadata(monkeypatch, tmp_path):
    dummy_st = _DummyStreamlit()
    dummy_st.chat_input_value = "generate a visualization"
    html_path = tmp_path / "chart.html"
    html_path.write_text("<div>chart</div>", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

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
    assert ("expander", "View visualization") in dummy_st.calls
    assert any(call[0] == "button" and call[1]["label"] == "Open in browser" for call in dummy_st.calls)
    html_calls = [call for call in dummy_st.calls if call[0] == "html"]
    assert len(html_calls) == 1


def test_render_app_ignores_missing_visualization_path(monkeypatch):
    dummy_st = _DummyStreamlit()
    dummy_st.chat_input_value = "run a 1-way sweep"
    monkeypatch.setitem(sys.modules, "streamlit", dummy_st)

    main = importlib.import_module("main")
    main = importlib.reload(main)

    class DummyOrchestrator:
        def process_query(self, prompt):
            return "processed: run a 1-way sweep"

    monkeypatch.setattr(main, "StudyOrchestrator", DummyOrchestrator)
    main.render_app()

    assert not any(call[0] == "html" for call in dummy_st.calls)
    assert not any(call[0] == "button" for call in dummy_st.calls)
