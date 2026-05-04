"""Compatibility layer for deterministic experience study tools."""

from __future__ import annotations

from typing import Any, Callable

from skills.experience_study_skill.ae_math import (
    _compute_mortality_rate_ci,
    compute_ae_ci,
    compute_ae_ci_amount,
)
from skills.experience_study_skill.feature_engineering import (
    _resolve_feature_source,
    _save_prepared_dataset,
    create_categorical_bands,
    regroup_categorical_features,
)
from skills.experience_study_skill.io import (
    ACTUARIAL_NUMERICS,
    EXCLUDED_DIMENSIONS,
    RAW_MISSING_TOKENS,
    SEMANTIC_NUMERIC_NON_DIMENSIONS,
    SUPPORTED_TABULAR_SUFFIXES,
    VALID_SORT_COLUMNS,
    ToolExecutionContext,
    _choose_dataset_path,
    _classify_feature_type,
    _ensure_output_dir,
    _error_result,
    _MAX_SWEEP_TOP_N,
    _read_tabular_input,
    _resolve_schema_source_path,
    _resolve_sheet_name,
    _tool_result,
    get_tabular_column_types,
    get_tabular_columns,
    inspect_dataset_schema,
    list_excel_sheets,
    load_tabular_input,
    load_tabular_input_as_strings,
    profile_dataset,
)
from skills.experience_study_skill.sweeps import (
    _apply_filters,
    _is_categorical_dimension,
    _selected_dimensions,
    run_dimensional_sweep,
)
from skills.experience_study_skill.validation import (
    _find_raw_non_numeric_values,
    run_actuarial_data_checks,
)
from skills.experience_study_skill.visualization import (
    _build_report_html,
    _build_scatter_figure,
    _build_table_figure,
    _build_treemap_figure,
    _figure_fragment,
    _metric_columns,
    _metric_label,
    _ratio_label,
    _required_columns,
    _split_dimensions,
    _validate_metric,
    generate_combined_report,
)


def get_tool_handlers() -> dict[str, Callable[[dict[str, Any], ToolExecutionContext], dict[str, Any]]]:
    """Return the registry of deterministic tool handlers."""
    return {
        "profile_dataset": lambda args, context: profile_dataset(context=context, **args),
        "inspect_dataset_schema": lambda args, context: inspect_dataset_schema(
            context=context, **args
        ),
        "run_actuarial_data_checks": lambda args, context: run_actuarial_data_checks(
            context=context, **args
        ),
        "create_categorical_bands": lambda args, context: create_categorical_bands(
            context=context, **args
        ),
        "regroup_categorical_features": lambda args, context: regroup_categorical_features(
            context=context, **args
        ),
        "run_dimensional_sweep": lambda args, context: run_dimensional_sweep(
            context=context, **args
        ),
        "generate_combined_report": lambda args, context: generate_combined_report(
            context=context, **args
        ),
    }
