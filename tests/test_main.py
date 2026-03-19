import importlib
import sys
from types import SimpleNamespace


class _DummyForm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyStreamlit:
    def __init__(self):
        self.session_state = {}
        self.calls = []

    def set_page_config(self, **kwargs):
        self.calls.append(("set_page_config", kwargs))

    def title(self, text):
        self.calls.append(("title", text))

    def caption(self, text):
        self.calls.append(("caption", text))

    def form(self, _name, clear_on_submit=False):
        self.calls.append(("form", clear_on_submit))
        return _DummyForm()

    def text_area(self, *_args, **_kwargs):
        return ""

    def form_submit_button(self, _label):
        return False

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
        "**You**: first prompt",
        "**Copilot**: first response",
        "**You**: second prompt",
        "**Copilot**: second response",
    ]
