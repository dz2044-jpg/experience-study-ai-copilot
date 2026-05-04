"""Dimensional sweep tools for the experience study skill."""

from __future__ import annotations

from itertools import combinations
import re
from typing import Any, Callable, Sequence

import pandas as pd

from skills.experience_study_skill.ae_math import compute_ae_ci, compute_ae_ci_amount
from skills.experience_study_skill.io import (
    ACTUARIAL_NUMERICS,
    EXCLUDED_DIMENSIONS,
    SEMANTIC_NUMERIC_NON_DIMENSIONS,
    VALID_SORT_COLUMNS,
    ToolExecutionContext,
    _MAX_SWEEP_TOP_N,
    _choose_dataset_path,
    _ensure_output_dir,
    _error_result,
    _tool_result,
    load_tabular_input,
)


def _is_categorical_dimension(series: pd.Series, column: str) -> bool:
    if column in EXCLUDED_DIMENSIONS or column in SEMANTIC_NUMERIC_NON_DIMENSIONS:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= 20
    return True


def _apply_filters(df: pd.DataFrame, filters: Sequence[dict[str, Any]]) -> pd.DataFrame:
    filtered = df
    operator_map: dict[str, Callable[[pd.Series, Any], pd.Series]] = {
        "=": lambda series, value: series == value,
        "!=": lambda series, value: series != value,
        ">": lambda series, value: series > value,
        ">=": lambda series, value: series >= value,
        "<": lambda series, value: series < value,
        "<=": lambda series, value: series <= value,
    }
    for filter_spec in filters:
        operator = filter_spec["operator"]
        if operator not in operator_map:
            raise ValueError(f"Unsupported operator: {operator}")
        column = filter_spec["column"]
        if column not in filtered.columns:
            raise KeyError(column)
        filtered = filtered[operator_map[operator](filtered[column], filter_spec["value"])]
    return filtered


def _selected_dimensions(df: pd.DataFrame, selected_columns: list[str] | None) -> list[str]:
    if selected_columns:
        missing = [column for column in selected_columns if column not in df.columns]
        if missing:
            raise KeyError(missing[0])
        invalid = [
            column
            for column in selected_columns
            if column in EXCLUDED_DIMENSIONS or column in SEMANTIC_NUMERIC_NON_DIMENSIONS
        ]
        if invalid:
            raise ValueError(f"Column `{invalid[0]}` is not eligible as a sweep dimension.")
        return list(dict.fromkeys(selected_columns))

    return [
        column for column in df.columns if _is_categorical_dimension(df[column], column)
    ]


def run_dimensional_sweep(
    *,
    context: ToolExecutionContext,
    depth: int = 1,
    filters: list[dict[str, Any]] | None = None,
    selected_columns: list[str] | None = None,
    min_mac: int = 0,
    top_n: int = 20,
    sort_by: str = "AE_Ratio_Amount",
    data_path: str | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    top_n = max(1, min(top_n, _MAX_SWEEP_TOP_N))

    if sort_by not in VALID_SORT_COLUMNS:
        return _error_result(
            "validation_error",
            f"sort_by must be one of {sorted(VALID_SORT_COLUMNS)}.",
        )

    source_path = _choose_dataset_path(data_path, context, require_prepared=True)
    if source_path is None:
        return _error_result(
            "missing_prerequisite",
            "No prepared dataset exists for this session. Profile a dataset first.",
        )

    context.emit_status(f"Running a deterministic {depth}-way dimensional sweep.")
    df = load_tabular_input(str(source_path))
    missing_core = [column for column in ACTUARIAL_NUMERICS if column not in df.columns]
    if missing_core:
        return _error_result(
            "validation_error",
            f"Prepared dataset is missing required columns: {missing_core}.",
        )

    try:
        filtered_df = _apply_filters(df, filters or [])
        dimensions = _selected_dimensions(filtered_df, selected_columns)
    except KeyError as exc:
        missing_column = str(exc).strip("'")
        return _error_result(
            "validation_error",
            f"Column `{missing_column}` was not found in the prepared dataset.",
            data={"available_columns": list(df.columns)},
        )
    except ValueError as exc:
        return _error_result("validation_error", str(exc))

    if len(dimensions) < depth:
        return _error_result(
            "validation_error",
            f"Need at least {depth} eligible dimensions, found {len(dimensions)}.",
            data={"available_columns": dimensions},
        )

    all_results: list[dict[str, Any]] = []
    for dimension_group in combinations(dimensions, depth):
        grouped = (
            filtered_df.groupby(list(dimension_group), dropna=False)
            .agg(
                Sum_MAC=("MAC", "sum"),
                Sum_MOC=("MOC", "sum"),
                Sum_MEC=("MEC", "sum"),
                Sum_MAF=("MAF", "sum"),
                Sum_MEF=("MEF", "sum"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["Sum_MAC"] >= min_mac]
        if grouped.empty:
            continue

        grouped["AE_Ratio_Count"] = grouped["Sum_MAC"] / grouped["Sum_MEC"]
        grouped["AE_Ratio_Amount"] = grouped["Sum_MAF"] / grouped["Sum_MEF"]

        count_cis = grouped.apply(
            lambda row: compute_ae_ci(
                mac=row["Sum_MAC"],
                moc=row["Sum_MOC"],
                mec=row["Sum_MEC"],
                confidence_level=confidence_level,
            ),
            axis=1,
        )
        amount_cis = grouped.apply(
            lambda row: compute_ae_ci_amount(
                mac=row["Sum_MAC"],
                moc=row["Sum_MOC"],
                mec=row["Sum_MEC"],
                actual_amount=row["Sum_MAF"],
                expected_amount=row["Sum_MEF"],
                confidence_level=confidence_level,
            ),
            axis=1,
        )

        grouped["AE_Count_CI_Lower"] = [ci[0] for ci in count_cis]
        grouped["AE_Count_CI_Upper"] = [ci[1] for ci in count_cis]
        grouped["AE_Amount_CI_Lower"] = [ci[0] for ci in amount_cis]
        grouped["AE_Amount_CI_Upper"] = [ci[1] for ci in amount_cis]
        grouped["Dimensions"] = grouped.apply(
            lambda row: " | ".join(f"{column}={row[column]}" for column in dimension_group),
            axis=1,
        )

        for _, row in grouped.iterrows():
            all_results.append(
                {
                    "Dimensions": row["Dimensions"],
                    "Sum_MAC": int(row["Sum_MAC"]),
                    "Sum_MOC": float(row["Sum_MOC"]),
                    "Sum_MEC": float(row["Sum_MEC"]),
                    "Sum_MAF": float(row["Sum_MAF"]),
                    "Sum_MEF": float(row["Sum_MEF"]),
                    "AE_Ratio_Count": float(row["AE_Ratio_Count"]),
                    "AE_Ratio_Amount": float(row["AE_Ratio_Amount"]),
                    "AE_Count_CI_Lower": row["AE_Count_CI_Lower"],
                    "AE_Count_CI_Upper": row["AE_Count_CI_Upper"],
                    "AE_Amount_CI_Lower": row["AE_Amount_CI_Lower"],
                    "AE_Amount_CI_Upper": row["AE_Amount_CI_Upper"],
                }
            )

    if not all_results:
        return _tool_result(
            True,
            "analysis",
            "No cohorts met the requested visibility threshold.",
            data={"results": []},
        )

    result_df = pd.DataFrame(all_results).sort_values(sort_by, ascending=False)
    top_results = result_df.head(top_n)
    _ensure_output_dir(context)

    slug_columns = selected_columns or dimensions
    slug = "_".join(
        re.sub(r"[^A-Za-z0-9]+", "_", column).strip("_").lower() for column in slug_columns
    )
    slug = slug[:120].rstrip("_") or "all_dimensions"
    dynamic_path = context.output_dir / f"sweep_summary_{depth}_{slug}.csv"
    latest_path = context.canonical_sweep_path()
    latest_depth_path = context.sweep_depth_path(depth)
    result_df.to_csv(dynamic_path, index=False)
    result_df.to_csv(latest_path, index=False)
    result_df.to_csv(latest_depth_path, index=False)

    return _tool_result(
        True,
        "analysis",
        f"Completed a {depth}-way dimensional sweep and saved the latest summary to `{latest_path}`.",
        artifacts={
            "prepared_dataset_path": str(source_path),
            "sweep_output_path": str(dynamic_path),
            "sweep_summary_path": str(latest_path),
            "sweep_depth": depth,
            "sweep_depth_path": str(latest_depth_path),
        },
        data={"results": top_results.to_dict(orient="records"), "depth": depth},
    )
