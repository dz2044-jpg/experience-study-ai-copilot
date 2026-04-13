import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.copilot_agent import UnifiedCopilot
from tests.conftest import final_message


class _FakeToolCall:
    def __init__(self, *, tool_call_id: str, name: str, arguments: dict[str, object]) -> None:
        self.id = tool_call_id
        self.function = SimpleNamespace(name=name, arguments=json.dumps(arguments))

    def model_dump(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


class _FakeClient:
    def __init__(self, messages: list[SimpleNamespace]) -> None:
        self._messages = list(messages)
        self.calls: list[dict[str, object]] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = self._messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_tool_gating_hides_visualization_before_sweep(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    intent = copilot._summarize_intent("Generate the combined report.")

    assert "generate_combined_report" not in copilot._enabled_tool_names(intent)


def test_analysis_requires_prepared_dataset(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(copilot.process_message("Run a 1-way sweep on Gender."))

    assert "Profile a dataset first" in final_message(events)


def test_extract_top_n_caps_large_requests(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    assert copilot._extract_top_n("Run a sweep and show the top 500 cohorts.") == 20


def test_profile_and_columns_request_profiles_then_inspects_prepared_dataset(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(copilot.process_message(f"Profile {sample_csv_path} and tell me the columns."))

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    message = final_message(events)

    assert tool_starts == [
        "Executing `profile_dataset`.",
        "Executing `inspect_dataset_schema`.",
    ]
    assert "Profiled the source dataset and saved" in message
    assert "Columns in" in message
    assert message.index("`Policy_Number`") < message.index("`Duration`") < message.index("`MAC`")
    assert copilot.state.prepared_dataset_ready is True
    assert copilot.state.prepared_dataset_path is not None
    assert copilot.state.prepared_dataset_path.exists()


def test_schema_request_for_bare_filename_uses_session_artifact_without_reprofiling(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message("What are columns in analysis_inforce.parquet?"))

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    message = final_message(events)

    assert tool_starts == ["Executing `inspect_dataset_schema`."]
    assert copilot.state.prepared_dataset_path is not None
    assert f"Columns in `{copilot.state.prepared_dataset_path.resolve()}`" in message
    assert "`Policy_Number`" in message
    assert "`Risk_Class`" in message


def test_pure_schema_request_does_not_mutate_prepared_artifact_state(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    prepared_path = copilot.state.prepared_dataset_path
    events = list(copilot.process_message("Show me the schema for the current dataset."))

    assert prepared_path is not None
    assert copilot.state.prepared_dataset_path == prepared_path
    assert copilot.state.prepared_dataset_ready is True
    assert not any(event.type == "artifact_update" for event in events)
    assert "Columns in" in final_message(events)


def test_analysis_only_request_runs_after_profile(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message("Run a 1-way sweep on Gender."))
    message = final_message(events)

    assert "Completed a 1-way dimensional sweep" in message
    assert "Summary of Sweep Results" in message
    assert "| Cohort Dimension | Actual Deaths (MAC) | Expected (MEC) | A/E Ratio (Count) | A/E Ratio (Amount) |" in message
    assert "| Gender=M | 2.00 | 0.79 | 2.53 | 0.85 |" in message
    assert "| Gender=F | 1.00 | 0.63 | 1.59 | 0.39 |" in message
    assert "Credibility interval detail is available in the chat explorer and generated report." in message
    assert copilot.state.latest_sweep_ready is True
    assert copilot.state.latest_sweep_path is not None
    assert copilot.state.latest_sweep_path.exists()


def test_visualize_only_request_runs_after_sweep(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))
    list(copilot.process_message("Run a 1-way sweep on Gender."))

    events = list(copilot.process_message("Generate the combined report."))

    assert "visualization report" in final_message(events).lower()
    assert copilot.state.latest_visualization_ready is True
    assert copilot.state.latest_visualization_path is not None
    assert copilot.state.latest_visualization_path.exists()


def test_full_pipeline_runs_in_order_with_session_local_artifacts(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(
        copilot.process_message(
            f"Profile {sample_csv_path}, group Issue_Age into 3 equal-width bands, "
            "run a 1-way sweep on Gender, and generate the combined report."
        )
    )

    message = final_message(events).lower()
    assert "prepared dataset" in message
    assert "dimensional sweep" in message
    assert "visualization report" in message
    assert copilot.state.prepared_dataset_ready is True
    assert copilot.state.latest_sweep_ready is True
    assert copilot.state.latest_visualization_ready is True


def test_filtered_analysis_request_respects_filter_clause(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    list(copilot.process_message("Run a 1-way sweep on Gender where Smoker = Yes."))

    sweep_path = copilot.state.latest_sweep_path
    assert sweep_path is not None
    result_df = pd.read_csv(sweep_path)
    assert not result_df.empty
    assert "Dimensions" in result_df.columns


def test_failed_tool_call_surfaces_exact_error_without_retrying(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    list(copilot.process_message(f"Profile {sample_csv_path}"))

    events = list(copilot.process_message("Group Unknown Column into 3 equal-width bands."))

    tool_starts = [event.message for event in events if event.type == "tool_start"]
    message = final_message(events)

    assert tool_starts == ["Executing `create_categorical_bands`."]
    assert copilot.state.prepared_dataset_path is not None
    assert (
        f"Column `Unknown_Column` was not found in `{copilot.state.prepared_dataset_path}`."
        in message
    )


def test_final_llm_response_strips_thinking_from_user_output_and_history(tmp_path: Path):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    copilot.client = _FakeClient(
        [
            SimpleNamespace(
                content="<thinking>Check state and prerequisites.</thinking>\nSchema inspection is ready.",
                tool_calls=[],
            )
        ]
    )

    events = list(copilot.process_message("Show me the schema for /tmp/example.csv."))

    assert final_message(events) == "Schema inspection is ready."
    assert "<thinking>" not in final_message(events)
    assert [event.message for event in events if event.type == "text_delta"] == [
        "Schema ",
        "inspection ",
        "is ",
        "ready.",
    ]
    assert copilot.history[-1]["content"] == "Schema inspection is ready."


def test_tool_call_keeps_internal_thinking_for_model_but_not_user_output(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")
    fake_client = _FakeClient(
        [
            SimpleNamespace(
                content="<thinking>Check state, then inspect exact schema path.</thinking>",
                tool_calls=[
                    _FakeToolCall(
                        tool_call_id="call_1",
                        name="inspect_dataset_schema",
                        arguments={"data_path": str(sample_csv_path)},
                    )
                ],
            ),
            SimpleNamespace(
                content="<thinking>Summarize the tool output only.</thinking>\nSchema inspection completed.",
                tool_calls=[],
            ),
        ]
    )
    copilot.client = fake_client

    events = list(copilot.process_message(f"Show me the schema for {sample_csv_path}."))

    assert [event.message for event in events if event.type == "tool_start"] == [
        "Executing `inspect_dataset_schema`."
    ]
    assert final_message(events) == "Schema inspection completed."
    assert "<thinking>" not in final_message(events)
    assert copilot.history[-1]["content"] == "Schema inspection completed."
    second_call_messages = fake_client.calls[1]["messages"]
    assistant_messages = [
        message
        for message in second_call_messages
        if message["role"] == "assistant"
    ]
    assert any("<thinking>" in message["content"] for message in assistant_messages)
