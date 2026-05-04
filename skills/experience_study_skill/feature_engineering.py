"""Feature engineering tools for the experience study skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from skills.experience_study_skill.io import (
    ToolExecutionContext,
    _ensure_output_dir,
    _error_result,
    _tool_result,
    load_tabular_input,
)


def _resolve_feature_source(
    explicit_data_path: str | None,
    context: ToolExecutionContext,
    *,
    require_existing_prepared: bool = False,
) -> Path | None:
    if explicit_data_path:
        path = Path(explicit_data_path)
        return path if path.exists() else None
    if context.prepared_dataset_path and context.prepared_dataset_path.exists():
        return context.prepared_dataset_path
    if require_existing_prepared:
        return None
    if context.raw_input_path and context.raw_input_path.exists():
        return context.raw_input_path
    return None


def _save_prepared_dataset(df: pd.DataFrame, context: ToolExecutionContext) -> Path:
    _ensure_output_dir(context)
    prepared_path = context.canonical_prepared_path()
    df.to_parquet(prepared_path, engine="pyarrow", index=False)
    return prepared_path


def create_categorical_bands(
    *,
    context: ToolExecutionContext,
    source_column: str,
    strategy: str,
    bins: int | None = None,
    custom_bins: list[float] | None = None,
    data_path: str | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    source_path = _resolve_feature_source(data_path, context)
    if source_path is None:
        return _error_result(
            "missing_prerequisite",
            "No prepared dataset is available. Profile a dataset first or provide a data_path.",
        )

    context.emit_status("Applying deterministic categorical banding.")
    df = load_tabular_input(str(source_path), sheet_name=sheet_name if data_path else None)
    if source_column not in df.columns:
        return _error_result(
            "validation_error",
            f"Column `{source_column}` was not found in `{source_path}`.",
            data={"available_columns": list(df.columns)},
        )
    if not pd.api.types.is_numeric_dtype(df[source_column]):
        return _error_result(
            "validation_error",
            f"Column `{source_column}` must be numeric for banding.",
        )

    resolved_bins = bins
    if strategy == "quantiles":
        resolved_bins = bins or 4
        banded = pd.qcut(df[source_column], q=resolved_bins, labels=False, duplicates="drop")
    elif strategy == "equal_width":
        resolved_bins = bins or 5
        banded = pd.cut(df[source_column], bins=resolved_bins, labels=False, duplicates="drop")
    elif strategy == "custom":
        if not custom_bins:
            return _error_result(
                "validation_error",
                "custom strategy requires custom_bins.",
            )
        banded = pd.cut(
            df[source_column],
            bins=custom_bins,
            include_lowest=True,
            duplicates="drop",
        )
    else:
        return _error_result(
            "validation_error",
            f"Unknown banding strategy `{strategy}`.",
        )

    new_column = f"{source_column}_band"
    engineered_df = df.copy()
    engineered_df[new_column] = banded.astype(str)
    prepared_path = _save_prepared_dataset(engineered_df, context)
    return _tool_result(
        True,
        "feature_engineering",
        f"Created `{new_column}` and saved the prepared dataset to `{prepared_path}`.",
        artifacts={"prepared_dataset_path": str(prepared_path)},
        data={
            "source_column": source_column,
            "new_column": new_column,
            "strategy": strategy,
            "bins": resolved_bins,
        },
    )


def regroup_categorical_features(
    *,
    context: ToolExecutionContext,
    source_column: str,
    mapping_dict: dict[str, Any],
    data_path: str | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    source_path = _resolve_feature_source(data_path, context)
    if source_path is None:
        return _error_result(
            "missing_prerequisite",
            "No prepared dataset is available. Profile a dataset first or provide a data_path.",
        )

    context.emit_status("Regrouping categorical values.")
    df = load_tabular_input(str(source_path), sheet_name=sheet_name if data_path else None)
    if source_column not in df.columns:
        return _error_result(
            "validation_error",
            f"Column `{source_column}` was not found in `{source_path}`.",
            data={"available_columns": list(df.columns)},
        )

    new_column = f"{source_column}_regrouped"
    engineered_df = df.copy()
    engineered_df[new_column] = engineered_df[source_column].astype(str).replace(mapping_dict)
    prepared_path = _save_prepared_dataset(engineered_df, context)
    return _tool_result(
        True,
        "feature_engineering",
        f"Created `{new_column}` and saved the prepared dataset to `{prepared_path}`.",
        artifacts={"prepared_dataset_path": str(prepared_path)},
        data={
            "source_column": source_column,
            "new_column": new_column,
            "mapping_applied": mapping_dict,
        },
    )
