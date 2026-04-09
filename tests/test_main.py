from __future__ import annotations

from core.copilot_agent import CopilotEvent
import main


class _FakeStatusPanel:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.updates: list[dict[str, object]] = []

    def write(self, message: str) -> None:
        self.writes.append(message)

    def update(self, **kwargs) -> None:
        self.updates.append(kwargs)


class _FakeResponsePlaceholder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def markdown(self, message: str) -> None:
        self.messages.append(message)


def test_consume_copilot_events_keeps_fallback_status_only(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    final_message = "Columns in `/tmp/analysis_inforce.parquet` (2):\n- `Policy_Number`: `string`"
    success_tool_message = "Inspected the schema for `/tmp/analysis_inforce.parquet`."

    response, visualization_path = main._consume_copilot_events(
        [
            CopilotEvent("status", message="Copilot received a new request."),
            CopilotEvent(
                "status",
                message="OpenAI is unavailable. Using deterministic local planning.",
            ),
            CopilotEvent("tool_start", message="Executing `inspect_dataset_schema`."),
            CopilotEvent(
                "tool_result",
                message=success_tool_message,
                data={"result": {"ok": True, "kind": "schema"}},
            ),
            CopilotEvent("text_delta", message=final_message),
            CopilotEvent("final", message=final_message, data={"artifact_state": {}}),
        ],
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert "OpenAI is unavailable. Using deterministic local planning." in status_panel.writes
    assert success_tool_message not in status_panel.writes
    assert response == final_message
    assert visualization_path is None
