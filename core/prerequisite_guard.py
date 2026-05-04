"""Prerequisite guidance and tool gating for the unified copilot runtime."""

from __future__ import annotations

from dataclasses import dataclass

from core.session_state import SessionArtifactState


@dataclass(slots=True)
class IntentSummary:
    """High-level intent classification used for gating and fallback routing."""

    explicit_data_path: str | None
    wants_profile: bool
    wants_schema: bool
    wants_validate: bool
    wants_band: bool
    wants_regroup: bool
    wants_analysis: bool
    wants_visualize: bool
    wants_full_pipeline: bool

    @property
    def is_general(self) -> bool:
        return not any(
            (
                self.wants_profile,
                self.wants_schema,
                self.wants_validate,
                self.wants_band,
                self.wants_regroup,
                self.wants_analysis,
                self.wants_visualize,
                self.wants_full_pipeline,
            )
        )


def guard_missing_prerequisites(
    intent: IntentSummary,
    state: SessionArtifactState,
) -> str | None:
    state.refresh()
    has_prepared = state.prepared_dataset_ready or state.prepared_dataset_path is not None
    has_sweep = state.latest_sweep_ready or state.latest_sweep_path is not None
    has_dataset = (
        intent.explicit_data_path
        or state.prepared_dataset_ready
        or state.prepared_dataset_path is not None
        or state.raw_input_path is not None
    )
    if intent.wants_schema and not has_dataset and not (
        intent.wants_profile or intent.wants_full_pipeline
    ):
        return "No dataset is available. Profile a dataset first or provide a data_path."
    if intent.wants_visualize and not (
        has_sweep or intent.wants_analysis or intent.wants_full_pipeline
    ):
        return "No sweep artifact exists for this session. Run a dimensional sweep first."
    if intent.wants_analysis and not (
        has_prepared
        or intent.wants_profile
        or intent.wants_band
        or intent.wants_regroup
        or intent.wants_full_pipeline
    ):
        return "No prepared dataset exists for this session. Profile a dataset first."
    if (intent.wants_profile or intent.wants_full_pipeline) and not (
        intent.explicit_data_path or state.raw_input_path
    ):
        return "Provide a dataset path to start the experience study workflow."
    if (intent.wants_band or intent.wants_regroup) and not (
        state.prepared_dataset_ready or state.raw_input_path or intent.explicit_data_path
    ):
        return "No dataset is available for feature engineering. Profile a dataset first or provide a data_path."
    return None


def enabled_tool_names(
    intent: IntentSummary,
    state: SessionArtifactState,
) -> set[str]:
    state.refresh()
    has_prepared = state.prepared_dataset_ready or state.prepared_dataset_path is not None
    has_sweep = state.latest_sweep_ready or state.latest_sweep_path is not None
    enabled: set[str] = set()

    if intent.wants_profile or intent.wants_full_pipeline:
        if intent.explicit_data_path or state.raw_input_path:
            enabled.add("profile_dataset")
    if intent.wants_schema:
        if (
            intent.explicit_data_path
            or state.prepared_dataset_ready
            or state.prepared_dataset_path
            or state.raw_input_path
        ):
            enabled.add("inspect_dataset_schema")
    if intent.wants_validate:
        if intent.explicit_data_path or state.raw_input_path or state.prepared_dataset_ready:
            enabled.add("run_actuarial_data_checks")
    if intent.wants_band:
        if intent.explicit_data_path or state.raw_input_path or state.prepared_dataset_ready:
            enabled.add("create_categorical_bands")
    if intent.wants_regroup:
        if intent.explicit_data_path or state.raw_input_path or state.prepared_dataset_ready:
            enabled.add("regroup_categorical_features")
    if has_prepared and (intent.wants_analysis or intent.wants_full_pipeline):
        enabled.add("run_dimensional_sweep")
    if has_sweep and intent.wants_visualize:
        enabled.add("generate_combined_report")
    return enabled
