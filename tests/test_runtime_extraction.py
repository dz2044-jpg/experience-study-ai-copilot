from pathlib import Path

import inspect

from core.copilot_agent import (
    CopilotEvent,
    IntentSummary,
    SessionArtifactState,
    UnifiedCopilot,
)
from core.fallback_planner import FallbackPlanner
from core.prerequisite_guard import guard_missing_prerequisites
from core.response_formatter import ResponseFormatter
import core.fallback_planner as fallback_planner
import core.prerequisite_guard as prerequisite_guard
import core.response_formatter as response_formatter
import core.session_state as session_state


def test_copilot_import_compatibility_remains_valid():
    assert CopilotEvent is not None
    assert UnifiedCopilot is not None
    assert SessionArtifactState is not None
    assert IntentSummary is not None


def test_extracted_modules_do_not_import_copilot_agent():
    for module in (
        fallback_planner,
        prerequisite_guard,
        response_formatter,
        session_state,
    ):
        assert "core.copilot_agent" not in inspect.getsource(module)


def test_prerequisite_guard_guidance_remains_unchanged(tmp_path: Path):
    state = SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)

    schema_intent = IntentSummary(
        explicit_data_path=None,
        wants_profile=False,
        wants_schema=True,
        wants_validate=False,
        wants_band=False,
        wants_regroup=False,
        wants_analysis=False,
        wants_visualize=False,
        wants_full_pipeline=False,
    )
    visualize_intent = IntentSummary(
        explicit_data_path=None,
        wants_profile=False,
        wants_schema=False,
        wants_validate=False,
        wants_band=False,
        wants_regroup=False,
        wants_analysis=False,
        wants_visualize=True,
        wants_full_pipeline=False,
    )

    assert (
        guard_missing_prerequisites(schema_intent, state)
        == "No dataset is available. Profile a dataset first or provide a data_path."
    )
    assert (
        guard_missing_prerequisites(visualize_intent, state)
        == "No sweep artifact exists for this session. Run a dimensional sweep first."
    )


def test_response_formatter_preserves_expected_text_shapes(tmp_path: Path):
    state = SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)
    formatter = ResponseFormatter(state)

    profile_result = {
        "kind": "profile",
        "message": "Profile completed.",
        "data": {
            "total_rows": 1234,
            "columns": ["Policy_Number", "Duration"],
            "unique_policy_count": 99,
        },
        "artifacts": {
            "raw_input_path": "raw.csv",
            "prepared_dataset_path": "analysis_inforce.parquet",
        },
    }
    schema_result = {
        "kind": "schema",
        "message": "Schema inspected.",
        "data": {
            "source_path": "analysis_inforce.parquet",
            "columns": ["Policy_Number"],
            "data_types": {"Policy_Number": "object"},
            "column_count": 1,
        },
        "artifacts": {},
    }

    assert formatter.format_profile_result(profile_result) == (
        "Created the prepared dataset `analysis_inforce.parquet` from `raw.csv`.\n"
        "Profile summary: 1,234 rows, 2 columns, 99 unique policies."
    )
    assert formatter.format_schema_result(schema_result) == (
        "Columns in `analysis_inforce.parquet` (1):\n"
        "- `Policy_Number`: `object`"
    )
    assert (
        formatter.format_compact_result(schema_result)
        == "Inspected the schema for `analysis_inforce.parquet` (1 columns)."
    )
    assert (
        ResponseFormatter.sanitize_user_facing_text(
            "<thinking>Internal scratch.</thinking>\nFinal answer."
        )
        == "Final answer."
    )


def test_fallback_planner_preserves_top_n_cap_and_full_plan_order(tmp_path: Path):
    state = SessionArtifactState(session_id="session-a", output_base_dir=tmp_path)
    planner = FallbackPlanner(state)
    user_input = (
        "Profile data/input/synthetic_inforce.csv, inspect the schema columns, "
        "validate the data, group Issue_Age into 3 equal-width bands, "
        'regroup on Risk_Class {"Preferred":"Preferred"}, '
        "run a 1-way sweep on Gender showing top 500 cohorts, "
        "and generate the combined report."
    )
    intent = IntentSummary(
        explicit_data_path="data/input/synthetic_inforce.csv",
        wants_profile=True,
        wants_schema=True,
        wants_validate=True,
        wants_band=True,
        wants_regroup=True,
        wants_analysis=True,
        wants_visualize=True,
        wants_full_pipeline=True,
    )

    plan, guidance = planner._build_fallback_plan(user_input, intent)

    assert guidance is None
    assert planner._extract_top_n(user_input) == 20
    assert [tool_name for tool_name, _ in plan] == [
        "profile_dataset",
        "inspect_dataset_schema",
        "run_actuarial_data_checks",
        "create_categorical_bands",
        "regroup_categorical_features",
        "run_dimensional_sweep",
        "generate_combined_report",
    ]


def test_deterministic_fallback_event_order_remains_stable(
    tmp_path: Path,
    sample_csv_path: Path,
):
    copilot = UnifiedCopilot(session_id="session-a", output_base_dir=tmp_path / "sessions")

    events = list(copilot.process_message(f"Profile {sample_csv_path}"))
    event_types = [event.type for event in events]

    assert event_types[0:3] == ["status", "status", "tool_start"]
    assert "tool_result" in event_types
    assert "artifact_update" in event_types
    assert event_types[-1] == "final"
    assert event_types.index("tool_start") < event_types.index("tool_result")
    assert event_types.index("tool_result") < event_types.index("artifact_update")
    assert event_types.index("artifact_update") < event_types.index("final")
