from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from core.artifact_manifest import file_sha256, upsert_artifact_entry
from core.copilot_agent import CopilotEvent
import main
from skills.experience_study_skill.ai_models import (
    AIActionResponse,
    AICohortRow,
    AISweepPacket,
    AIValidationIssue,
    AIValidationResult,
)


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


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


class _FakeStreamlitApp:
    def __init__(self, *, prompt: str | None, copilot: object) -> None:
        self.prompt = prompt
        self.session_state: dict[str, object] = {
            "session_id": "session-a",
            "copilot": copilot,
            "history": [],
            "ai_interpretation_response": None,
        }

    def set_page_config(self, **kwargs) -> None:
        return None

    def title(self, message: str) -> None:
        return None

    def caption(self, message: str) -> None:
        return None

    def markdown(self, message: str) -> None:
        return None

    def chat_input(self, message: str) -> str | None:
        return self.prompt

    def chat_message(self, role: str) -> _NullContext:
        return _NullContext()

    def status(self, message: str, *, expanded: bool) -> _FakeStatusPanel:
        return _FakeStatusPanel()

    def empty(self) -> _FakeResponsePlaceholder:
        return _FakeResponsePlaceholder()


class _FakeRenderCopilot:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.state = SimpleNamespace(panel_ready=False)

    def process_message(self, user_input: str) -> list[CopilotEvent]:
        self.calls.append(f"process:{user_input}")
        self.state.panel_ready = True
        return [
            CopilotEvent(
                "final",
                message="done",
                data={"artifact_state": {}},
            )
        ]


def _write_ai_sweep(path: Path) -> Path:
    pd.DataFrame(
        [
            {
                "Dimensions": "Gender=M",
                "Sum_MAC": 2,
                "Sum_MOC": 10.0,
                "Sum_MEC": 1.0,
                "Sum_MAF": 100000.0,
                "Sum_MEF": 80000.0,
                "AE_Ratio_Count": 2.0,
                "AE_Ratio_Amount": 1.25,
                "AE_Count_CI_Lower": 0.25,
                "AE_Count_CI_Upper": 3.5,
                "AE_Amount_CI_Lower": 0.30,
                "AE_Amount_CI_Upper": 3.75,
            },
            {
                "Dimensions": "Region=West",
                "Sum_MAC": 4,
                "Sum_MOC": 7.0,
                "Sum_MEC": 1.4,
                "Sum_MAF": 120000.0,
                "Sum_MEF": 40000.0,
                "AE_Ratio_Count": 0.8,
                "AE_Ratio_Amount": 3.0,
                "AE_Count_CI_Lower": 0.15,
                "AE_Count_CI_Upper": 2.0,
                "AE_Amount_CI_Lower": 0.55,
                "AE_Amount_CI_Upper": 4.5,
            },
        ]
    ).to_csv(path, index=False)
    return path


def _write_sweep_manifest(manifest_path: Path, sweep_path: Path) -> Path:
    upsert_artifact_entry(
        manifest_path,
        artifact_type="sweep_summary",
        path=sweep_path,
        generating_tool="run_dimensional_sweep",
        parameters={"depth": 1},
        source_artifacts=[],
    )
    return manifest_path


def _refresh_noop() -> None:
    return None


def _example_packet(
    *,
    state_fingerprint: str = "state-a",
    packet_fingerprint: str = "packet-a",
) -> AISweepPacket:
    return AISweepPacket(
        source_artifact_path="sweep_summary.csv",
        source_content_hash="hash-a",
        state_fingerprint=state_fingerprint,
        packet_fingerprint=packet_fingerprint,
        rows=[
            AICohortRow(
                evidence_ref="row_0001",
                Dimensions="Gender=M",
                Dimension_Columns=["Gender"],
                Sum_MAC=2.0,
                Sum_MOC=10.0,
                Sum_MEC=1.0,
                Sum_MAF=100000.0,
                Sum_MEF=80000.0,
                AE_Ratio_Count=2.0,
                AE_Ratio_Amount=1.25,
            ),
            AICohortRow(
                evidence_ref="row_0002",
                Dimensions="[masked cohort label]",
                Dimension_Columns=["Gender"],
                Sum_MAC=0.0,
                Sum_MOC=8.0,
                Sum_MEC=1.2,
                Sum_MAF=0.0,
                Sum_MEF=85000.0,
                AE_Ratio_Count=0.0,
                AE_Ratio_Amount=0.0,
                low_credibility=True,
                masking_reason="low_volume",
                caution_flags=["low_volume"],
            ),
        ],
    )


def test_render_app_renders_ai_panel_after_current_chat_processing(monkeypatch) -> None:
    calls: list[str] = []
    copilot = _FakeRenderCopilot(calls)
    fake_st = _FakeStreamlitApp(prompt=" Run sweep now ", copilot=copilot)

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main, "_render_sidebar", lambda: False)
    monkeypatch.setattr(main, "_render_empty_state", lambda: calls.append("empty_state"))
    monkeypatch.setattr(main, "_render_sweep_explorer", lambda sweep_results: None)
    monkeypatch.setattr(
        main,
        "_render_visualization_card",
        lambda visualization_path, widget_key_prefix: None,
    )

    def fake_render_ai_panel(panel_copilot) -> None:
        calls.append(f"ai_panel:{panel_copilot.state.panel_ready}")

    monkeypatch.setattr(main, "_render_ai_interpretation_panel", fake_render_ai_panel)

    main.render_app()

    assert calls == ["empty_state", "process:Run sweep now", "ai_panel:True"]
    assert fake_st.session_state["history"] == [
        {
            "prompt": "Run sweep now",
            "response": "done",
            "visualization_path": None,
            "sweep_results": None,
        }
    ]


def test_render_app_renders_ai_panel_without_prompt(monkeypatch) -> None:
    calls: list[str] = []
    copilot = _FakeRenderCopilot(calls)
    fake_st = _FakeStreamlitApp(prompt=None, copilot=copilot)

    monkeypatch.setattr(main, "st", fake_st)
    monkeypatch.setattr(main, "_render_sidebar", lambda: False)
    monkeypatch.setattr(main, "_render_empty_state", lambda: calls.append("empty_state"))

    def fake_render_ai_panel(panel_copilot) -> None:
        calls.append(f"ai_panel:{panel_copilot.state.panel_ready}")

    monkeypatch.setattr(main, "_render_ai_interpretation_panel", fake_render_ai_panel)

    main.render_app()

    assert calls == ["empty_state", "ai_panel:False"]
    assert fake_st.session_state["history"] == []


def test_consume_copilot_events_keeps_fallback_status_only(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    final_message = "Columns in `/tmp/analysis_inforce.parquet` (2):\n- `Policy_Number`: `string`"
    success_tool_message = "Inspected the schema for `/tmp/analysis_inforce.parquet`."

    response, visualization_path, sweep_results = main._consume_copilot_events(
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
    assert sweep_results is None


def test_consume_copilot_events_captures_analysis_rows(monkeypatch) -> None:
    monkeypatch.setattr(main, "st", object())
    status_panel = _FakeStatusPanel()
    response_placeholder = _FakeResponsePlaceholder()
    rows = [
        {
            "Dimensions": "Gender=M",
            "Sum_MAC": 2,
            "Sum_MOC": 3.7,
            "Sum_MEC": 0.789,
            "Sum_MAF": 270000.0,
            "Sum_MEF": 317000.0,
            "AE_Ratio_Count": 2.5316,
            "AE_Ratio_Amount": 0.8517,
            "AE_Count_CI_Lower": 0.8,
            "AE_Count_CI_Upper": 4.2,
            "AE_Amount_CI_Lower": 0.2,
            "AE_Amount_CI_Upper": 1.5,
        }
    ]

    _, _, sweep_results = main._consume_copilot_events(
        [
            CopilotEvent("status", message="Copilot received a new request."),
            CopilotEvent(
                "tool_result",
                message="Completed a 1-way dimensional sweep.",
                data={"result": {"ok": True, "kind": "analysis", "data": {"results": rows}}},
            ),
            CopilotEvent("final", message="done", data={"artifact_state": {}}),
        ],
        status_panel=status_panel,
        response_placeholder=response_placeholder,
    )

    assert sweep_results == rows


def test_build_sweep_display_frame_renames_and_rounds() -> None:
    display_df = main._build_sweep_display_frame(
        [
            {
                "Dimensions": "Risk_Class=Standard",
                "Sum_MAC": 14,
                "Sum_MOC": 10.135,
                "Sum_MEC": 1.384,
                "Sum_MAF": 120000.123,
                "Sum_MEF": 95000.567,
                "AE_Ratio_Count": 10.115,
                "AE_Ratio_Amount": 1.789,
                "AE_Count_CI_Lower": 0.904,
                "AE_Count_CI_Upper": 2.226,
                "AE_Amount_CI_Lower": 0.377,
                "AE_Amount_CI_Upper": 2.444,
            }
        ]
    )

    assert list(display_df.columns) == [
        "Cohort Dimension",
        "Actual Deaths (MAC)",
        "Exposure (MOC)",
        "Expected (MEC)",
        "Actual Amount (MAF)",
        "Expected Amount (MEF)",
        "A/E Ratio (Count)",
        "A/E Ratio (Amount)",
        "Count CI Lower",
        "Count CI Upper",
        "Amount CI Lower",
        "Amount CI Upper",
    ]
    row = display_df.iloc[0].to_dict()
    assert row["Cohort Dimension"] == "Risk_Class=Standard"
    assert row["Actual Deaths (MAC)"] == 14.0
    assert row["Expected (MEC)"] == 1.38
    assert row["A/E Ratio (Count)"] == 10.12
    assert row["Amount CI Upper"] == 2.44
    assert isinstance(display_df, pd.DataFrame)


def test_ai_panel_readiness_reports_missing_artifacts_and_manifest_hash(tmp_path: Path) -> None:
    empty_state = SimpleNamespace(
        latest_sweep_path=None,
        artifact_manifest_path=None,
        latest_state_fingerprint=None,
        refresh=_refresh_noop,
    )

    empty_readiness = main._get_ai_panel_readiness(empty_state)

    assert empty_readiness.checks == {
        "latest_sweep": False,
        "artifact_manifest": False,
        "state_fingerprint": False,
        "sweep_manifest_hash": False,
    }
    assert empty_readiness.ready is False

    sweep_path = _write_ai_sweep(tmp_path / "sweep_summary.csv")
    missing_manifest_state = SimpleNamespace(
        latest_sweep_path=sweep_path,
        artifact_manifest_path=None,
        latest_state_fingerprint="state-a",
        refresh=_refresh_noop,
    )

    missing_manifest_readiness = main._get_ai_panel_readiness(missing_manifest_state)

    assert missing_manifest_readiness.checks["latest_sweep"] is True
    assert missing_manifest_readiness.checks["artifact_manifest"] is False
    assert missing_manifest_readiness.checks["state_fingerprint"] is True
    assert missing_manifest_readiness.checks["sweep_manifest_hash"] is False

    manifest_path = _write_sweep_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        sweep_path,
    )
    ready_state = SimpleNamespace(
        latest_sweep_path=sweep_path,
        artifact_manifest_path=manifest_path,
        latest_state_fingerprint="state-a",
        refresh=_refresh_noop,
    )

    ready = main._get_ai_panel_readiness(ready_state)

    assert ready.ready is True
    assert ready.sweep_content_hash == file_sha256(sweep_path)
    assert ready.sweep_hash_matches_file is True
    assert main._manifest_content_hash_for_path(manifest_path, sweep_path) == file_sha256(
        sweep_path
    )


def test_select_top_cohort_ignores_invalid_values_and_preserves_tie_order() -> None:
    rows = [
        SimpleNamespace(evidence_ref="missing", AE_Ratio_Amount=None),
        SimpleNamespace(evidence_ref="bad", AE_Ratio_Amount="not numeric"),
        SimpleNamespace(evidence_ref="lower", AE_Ratio_Amount="1.9"),
        SimpleNamespace(evidence_ref="first-high", AE_Ratio_Amount=2.5),
        SimpleNamespace(evidence_ref="second-high", AE_Ratio_Amount=2.5),
    ]

    assert main._select_top_cohort_evidence_ref(rows) == "first-high"
    assert main._select_top_cohort_evidence_ref(rows[:2]) is None


def test_format_ai_cohort_label_is_readable_but_evidence_ref_stable() -> None:
    row = SimpleNamespace(
        evidence_ref="row_0007",
        Dimensions="Gender=M | Risk_Class=Preferred",
        AE_Ratio_Amount=1.421,
        Sum_MAC=18,
    )

    label = main._format_ai_cohort_label(row)

    assert "Gender=M | Risk_Class=Preferred" in label
    assert "A/E Amount=1.42" in label
    assert "MAC=18.00" in label
    assert "row_0007" in label


def test_ai_response_sections_render_metadata_and_stale_status() -> None:
    packet = _example_packet()
    response = AIActionResponse(
        action_name="explain_cohort",
        source_mode="fallback",
        response_text="AI summary text should stay separate from deterministic facts.",
        evidence_refs=["row_0001"],
        caution_flags=["low_volume"],
        next_review_steps=["Review the evidence ref."],
        state_fingerprint="state-a",
        packet_fingerprint="packet-a",
        validation=AIValidationResult(
            warnings=[
                AIValidationIssue(
                    code="review_caution",
                    message="Review credibility before report use.",
                )
            ],
        ),
    )
    readiness = main._AIArtifactReadiness(
        checks={
            "latest_sweep": True,
            "artifact_manifest": True,
            "state_fingerprint": True,
            "sweep_manifest_hash": True,
        },
        state_fingerprint="state-a",
        sweep_content_hash="hash-a",
        actual_sweep_content_hash="hash-a",
    )
    record = main._build_ai_response_record(
        action_name="explain_cohort",
        selected_evidence_ref="row_0001",
        response=response,
        packet=packet,
        readiness=readiness,
    )

    sections = main._build_ai_response_sections(record, packet, readiness)

    assert sections["summary_text"] == response.response_text
    assert sections["source_mode"] == "fallback"
    assert sections["evidence_refs"] == ["row_0001"]
    assert sections["key_findings"] == [
        "row_0001: Gender=M; MAC=2.00, MEC=1.00, A/E count=2.00, A/E amount=1.25."
    ]
    assert sections["caution_flags"] == ["low_volume", "review_caution"]
    assert sections["validation_issues"] == [
        "review_caution: Review credibility before report use."
    ]
    assert sections["next_review_steps"] == ["Review the evidence ref."]
    assert sections["freshness_status"] == "Fresh"
    assert sections["state_fingerprint"] == "state-a"
    assert sections["packet_fingerprint"] == "packet-a"
    assert sections["sweep_content_hash"] == "hash-a"
    assert sections["sweep_hash_status"] == "Manifest hash matches latest sweep file."

    stale_readiness = main._AIArtifactReadiness(
        checks=readiness.checks,
        state_fingerprint="state-b",
        sweep_content_hash="hash-b",
        actual_sweep_content_hash="hash-b",
    )
    stale_packet = _example_packet(
        state_fingerprint="state-b",
        packet_fingerprint="packet-b",
    )

    stale_sections = main._build_ai_response_sections(record, stale_packet, stale_readiness)

    assert stale_sections["freshness_status"] == "Stale"
    assert stale_sections["freshness_mismatches"] == [
        "state fingerprint",
        "packet fingerprint",
        "sweep content hash",
    ]


def test_ai_interpretation_action_falls_back_without_configured_client(tmp_path: Path) -> None:
    sweep_path = _write_ai_sweep(tmp_path / "sweep_summary.csv")
    manifest_path = _write_sweep_manifest(
        tmp_path / "audit" / "artifact_manifest.json",
        sweep_path,
    )
    readiness = main._get_ai_panel_readiness(
        SimpleNamespace(
            latest_sweep_path=sweep_path,
            artifact_manifest_path=manifest_path,
            latest_state_fingerprint="state-a",
            refresh=_refresh_noop,
        )
    )
    packet = main._build_ai_packet_for_panel(readiness)

    record = main._run_ai_interpretation_action(
        copilot=SimpleNamespace(client=None, model="test-model"),
        action_name="summarize_sweep",
        packet=packet,
        readiness=readiness,
    )

    assert record["response"].source_mode == "fallback"
    assert record["action_name"] == "summarize_sweep"
    assert record["response_sweep_content_hash"] == file_sha256(sweep_path)


def test_ai_panel_path_uses_packet_builder_and_orchestrator(monkeypatch) -> None:
    packet = _example_packet()
    response = AIActionResponse(
        action_name="explain_cohort",
        source_mode="llm",
        response_text="Cohort explanation.",
        evidence_refs=["row_0001"],
        state_fingerprint="state-a",
        packet_fingerprint="packet-a",
    )
    calls: dict[str, object] = {}

    def fake_build_latest_sweep_packet(**kwargs):
        calls["build_packet"] = kwargs
        return packet

    def fake_run_ai_action(**kwargs):
        calls["run_ai_action"] = kwargs
        return response

    monkeypatch.setattr(main, "build_latest_sweep_packet", fake_build_latest_sweep_packet)
    monkeypatch.setattr(main, "run_ai_action", fake_run_ai_action)
    readiness = main._AIArtifactReadiness(
        checks={
            "latest_sweep": True,
            "artifact_manifest": True,
            "state_fingerprint": True,
            "sweep_manifest_hash": True,
        },
        sweep_path=Path("sweep_summary.csv"),
        artifact_manifest_path=Path("artifact_manifest.json"),
        state_fingerprint="state-a",
        sweep_content_hash="hash-a",
    )

    built_packet = main._build_ai_packet_for_panel(readiness)
    record = main._run_ai_interpretation_action(
        copilot=SimpleNamespace(client=object(), model="test-model"),
        action_name="explain_cohort",
        packet=built_packet,
        readiness=readiness,
        selected_evidence_ref="row_0001",
    )

    assert calls["build_packet"] == {
        "sweep_path": readiness.sweep_path,
        "artifact_manifest_path": readiness.artifact_manifest_path,
    }
    run_call = calls["run_ai_action"]
    assert run_call["action_name"] == "explain_cohort"
    assert run_call["packet"] is packet
    assert run_call["client"] is not None
    assert run_call["model"] == "test-model"
    assert run_call["action_context"] == {"evidence_ref": "row_0001"}
    assert record["response"] is response
