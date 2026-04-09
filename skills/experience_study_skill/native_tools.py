"""Deterministic tools for the experience study skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Callable, Sequence
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from scipy import stats


ACTUARIAL_NUMERICS = ["MAC", "MEC", "MAF", "MEF", "MOC"]
SUPPORTED_TABULAR_SUFFIXES = {".csv", ".parquet", ".xlsx"}
EXCLUDED_DIMENSIONS = {
    "Policy_Number",
    "MAC",
    "MOC",
    "MEC",
    "MAF",
    "MEF",
    "COLA",
}
SEMANTIC_NUMERIC_NON_DIMENSIONS = {"Face_Amount", "Issue_Age", "Age", "Duration"}
VALID_SORT_COLUMNS = {
    "AE_Ratio_Count",
    "AE_Ratio_Amount",
    "Sum_MAC",
    "Sum_MOC",
    "Sum_MEC",
    "Sum_MAF",
    "Sum_MEF",
}
RAW_MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a"}
_NEUTRAL_COLOR = "#1f4e79"
_FIGURE_CONFIG = {"displaylogo": False, "responsive": True}
_SCATTER_X_MAX = 3.0
_TREEMAP_COLOR_MAX = 2.0
_MAX_SWEEP_TOP_N = 20


@dataclass(slots=True)
class ToolExecutionContext:
    """Server-side execution context passed into deterministic tools."""

    session_id: str
    output_dir: Path
    raw_input_path: Path | None = None
    prepared_dataset_path: Path | None = None
    latest_sweep_path: Path | None = None
    latest_sweep_paths_by_depth: dict[int, Path] = field(default_factory=dict)
    latest_visualization_path: Path | None = None
    status_events: list[str] = field(default_factory=list)

    def emit_status(self, message: str) -> None:
        if message:
            self.status_events.append(message)

    def canonical_prepared_path(self) -> Path:
        return self.output_dir / "analysis_inforce.parquet"

    def canonical_sweep_path(self) -> Path:
        return self.output_dir / "sweep_summary.csv"

    def sweep_depth_path(self, depth: int) -> Path:
        return self.output_dir / f"sweep_summary_latest_{depth}.csv"

    def next_visualization_path(self) -> Path:
        return self.output_dir / f"combined_ae_report_{uuid4().hex[:10]}.html"


def _tool_result(
    ok: bool,
    kind: str,
    message: str,
    *,
    artifacts: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "kind": kind,
        "message": message,
        "artifacts": artifacts or {},
        "data": data or {},
    }


def _error_result(kind: str, message: str, *, data: dict[str, Any] | None = None) -> dict[str, Any]:
    return _tool_result(False, kind, message, data=data)


def _ensure_output_dir(context: ToolExecutionContext) -> None:
    context.output_dir.mkdir(parents=True, exist_ok=True)


def list_excel_sheets(path: str) -> list[str]:
    workbook = pd.ExcelFile(path, engine="openpyxl")
    return workbook.sheet_names


def _resolve_sheet_name(path: Path, sheet_name: str | None) -> str | None:
    if path.suffix.lower() != ".xlsx":
        return None
    if sheet_name:
        return sheet_name
    sheets = list_excel_sheets(str(path))
    return sheets[0] if sheets else None


def _read_tabular_input(
    path: str,
    sheet_name: str | None = None,
    *,
    raw_strings: bool = False,
) -> pd.DataFrame:
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {suffix or '<none>'}. Supported formats: {sorted(SUPPORTED_TABULAR_SUFFIXES)}"
        )

    if suffix == ".csv":
        read_kwargs = {"dtype": str, "keep_default_na": False} if raw_strings else {}
        return pd.read_csv(input_path, **read_kwargs)

    if suffix == ".parquet":
        df = pd.read_parquet(input_path)
        if raw_strings:
            return df.fillna("").astype(str)
        return df

    resolved_sheet_name = _resolve_sheet_name(input_path, sheet_name)
    read_kwargs: dict[str, Any] = {
        "sheet_name": resolved_sheet_name,
        "engine": "openpyxl",
    }
    if raw_strings:
        read_kwargs.update({"dtype": str, "keep_default_na": False})
    return pd.read_excel(input_path, **read_kwargs)


def load_tabular_input(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    df = _read_tabular_input(path, sheet_name=sheet_name, raw_strings=False)
    for column in ACTUARIAL_NUMERICS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    if "Policy_Number" in df.columns and not pd.api.types.is_string_dtype(df["Policy_Number"]):
        df["Policy_Number"] = df["Policy_Number"].astype(str)
    return df


def load_tabular_input_as_strings(path: str, sheet_name: str | None = None) -> pd.DataFrame:
    return _read_tabular_input(path, sheet_name=sheet_name, raw_strings=True)


def get_tabular_columns(path: str, sheet_name: str | None = None) -> list[str]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(input_path, nrows=0).columns.tolist()
    if suffix == ".parquet":
        return list(pq.read_schema(input_path).names)
    resolved_sheet_name = _resolve_sheet_name(input_path, sheet_name)
    return pd.read_excel(
        input_path,
        sheet_name=resolved_sheet_name,
        engine="openpyxl",
        nrows=0,
    ).columns.tolist()


def get_tabular_column_types(path: str, sheet_name: str | None = None) -> dict[str, str]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        sample = pd.read_csv(input_path, nrows=1000)
        return {column: str(dtype) for column, dtype in sample.dtypes.items()}
    if suffix == ".parquet":
        schema = pq.read_schema(input_path)
        return {field.name: str(field.type) for field in schema}
    resolved_sheet_name = _resolve_sheet_name(input_path, sheet_name)
    sample = pd.read_excel(
        input_path,
        sheet_name=resolved_sheet_name,
        engine="openpyxl",
        nrows=1000,
    )
    return {column: str(dtype) for column, dtype in sample.dtypes.items()}


def _classify_feature_type(df: pd.DataFrame, column: str) -> str:
    if column in ACTUARIAL_NUMERICS or column in {"Face_Amount", "Issue_Age", "Age"}:
        return "numerical"
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return "numerical" if series.nunique(dropna=True) > 20 else "categorical"
    return "categorical"


def _choose_dataset_path(
    explicit_path: str | None,
    context: ToolExecutionContext,
    *,
    require_prepared: bool = False,
) -> Path | None:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.exists() else None
    if require_prepared:
        if context.prepared_dataset_path and context.prepared_dataset_path.exists():
            return context.prepared_dataset_path
        return None
    if context.prepared_dataset_path and context.prepared_dataset_path.exists():
        return context.prepared_dataset_path
    if context.raw_input_path and context.raw_input_path.exists():
        return context.raw_input_path
    return None


def _resolve_schema_source_path(
    explicit_path: str | None,
    context: ToolExecutionContext,
) -> Path | None:
    if explicit_path:
        requested_path = Path(explicit_path)
        candidates = [requested_path]
        if not requested_path.is_absolute():
            candidates.append(context.output_dir / requested_path)
            candidates.append(Path.cwd() / requested_path)
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return None
    if context.prepared_dataset_path and context.prepared_dataset_path.exists():
        return context.prepared_dataset_path.resolve()
    if context.raw_input_path and context.raw_input_path.exists():
        return context.raw_input_path.resolve()
    return None


def _find_raw_non_numeric_values(data_path: str, sheet_name: str | None = None) -> list[str]:
    raw_df = load_tabular_input_as_strings(data_path, sheet_name=sheet_name)
    issues: list[str] = []
    for column in ACTUARIAL_NUMERICS:
        if column not in raw_df.columns:
            continue
        raw_values = raw_df[column].fillna("").astype(str).str.strip()
        missing_mask = raw_values.str.lower().isin(RAW_MISSING_TOKENS)
        parsed_values = pd.to_numeric(raw_values.where(~missing_mask, pd.NA), errors="coerce")
        invalid_count = int((~missing_mask & parsed_values.isna()).sum())
        if invalid_count:
            issues.append(
                f"{column} contains {invalid_count} non-numeric raw value(s) that cannot be parsed."
            )
    return issues


def profile_dataset(
    *,
    data_path: str,
    context: ToolExecutionContext,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    source_path = Path(data_path)
    if not source_path.exists():
        return _error_result("validation_error", f"File not found: {data_path}")

    context.emit_status("Loading and profiling the source dataset.")
    df = load_tabular_input(str(source_path), sheet_name=sheet_name)
    _ensure_output_dir(context)
    prepared_path = context.canonical_prepared_path()
    df.to_parquet(prepared_path, engine="pyarrow", index=False)

    total_rows = len(df)
    memory_bytes = int(df.memory_usage(deep=True).sum())
    unique_policies = int(df["Policy_Number"].nunique()) if "Policy_Number" in df.columns else 0
    data = {
        "total_rows": total_rows,
        "columns": list(df.columns),
        "data_types": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "feature_classification": {
            column: _classify_feature_type(df, column) for column in df.columns
        },
        "memory_usage_bytes": memory_bytes,
        "memory_usage_human": f"{memory_bytes / 1024:.2f} KB",
        "unique_policy_count": unique_policies,
        "null_counts": {column: int(df[column].isna().sum()) for column in df.columns},
    }
    message = (
        f"Profiled `{source_path}` and created the session-local prepared dataset at "
        f"`{prepared_path}`."
    )
    return _tool_result(
        True,
        "profile",
        message,
        artifacts={
            "raw_input_path": str(source_path),
            "prepared_dataset_path": str(prepared_path),
        },
        data=data,
    )


def inspect_dataset_schema(
    *,
    context: ToolExecutionContext,
    data_path: str | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    source_path = _resolve_schema_source_path(data_path, context)
    if source_path is None:
        if data_path:
            return _error_result("validation_error", f"File not found: {data_path}")
        return _error_result(
            "missing_prerequisite",
            "No dataset is available. Profile a dataset first or provide a data_path.",
        )

    context.emit_status("Inspecting the dataset schema.")
    columns = get_tabular_columns(str(source_path), sheet_name=sheet_name)
    column_types = get_tabular_column_types(str(source_path), sheet_name=sheet_name)
    ordered_types = {column: column_types.get(column, "unknown") for column in columns}
    return _tool_result(
        True,
        "schema",
        f"Inspected the schema for `{source_path}`.",
        data={
            "source_path": str(source_path),
            "columns": columns,
            "column_count": len(columns),
            "data_types": ordered_types,
        },
    )


def run_actuarial_data_checks(
    *,
    context: ToolExecutionContext,
    data_path: str | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    source_path = _choose_dataset_path(data_path, context)
    if source_path is None:
        return _error_result(
            "missing_prerequisite",
            "No dataset is available. Profile a raw dataset first or provide a data_path.",
        )

    context.emit_status("Running deterministic actuarial validation checks.")
    issues = _find_raw_non_numeric_values(str(source_path), sheet_name=sheet_name)
    df = load_tabular_input(str(source_path), sheet_name=sheet_name)

    if "Policy_Number" in df.columns and not pd.api.types.is_string_dtype(df["Policy_Number"]):
        issues.append("Policy_Number must be a string type.")

    numeric_columns = ["MAC", "MEC", "MOC"]
    for column in ["MAF", "MEF"]:
        if column in df.columns:
            numeric_columns.append(column)
    for column in numeric_columns:
        if column in df.columns and not pd.api.types.is_numeric_dtype(df[column]):
            issues.append(f"{column} must be numeric.")

    int_columns = ["Face_Amount", "Duration"]
    age_column = "Issue_Age" if "Issue_Age" in df.columns else "Age" if "Age" in df.columns else None
    if age_column:
        int_columns.append(age_column)
    for column in int_columns:
        if column not in df.columns:
            continue
        if pd.api.types.is_integer_dtype(df[column]):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            if not (df[column].dropna() == df[column].dropna().astype(int)).all():
                issues.append(f"{column} must contain integer values.")
        else:
            issues.append(f"{column} must be an integer field.")

    if "MAC" in df.columns:
        invalid_mac = df[~df["MAC"].isin([0, 1])].dropna(subset=["MAC"])
        if not invalid_mac.empty:
            issues.append(f"MAC must be 0 or 1. Found {len(invalid_mac)} invalid rows.")

    if "MEC" in df.columns:
        invalid_mec = df[~((df["MEC"] > 0) & (df["MEC"] < 1))].dropna(subset=["MEC"])
        if not invalid_mec.empty:
            issues.append(
                f"MEC must be strictly between 0 and 1. Found {len(invalid_mec)} invalid rows."
            )

    if "Face_Amount" in df.columns:
        invalid_face = df[df["Face_Amount"] <= 0]
        if not invalid_face.empty:
            issues.append(
                f"Face_Amount must be greater than 0. Found {len(invalid_face)} invalid rows."
            )

    if age_column:
        invalid_age = df[df[age_column] <= 0]
        if not invalid_age.empty:
            issues.append(f"{age_column} must be greater than 0. Found {len(invalid_age)} invalid rows.")

    if "MOC" in df.columns:
        invalid_moc = df[~((df["MOC"] > 0) & (df["MOC"] <= 1.0))].dropna(subset=["MOC"])
        if not invalid_moc.empty:
            issues.append(
                f"MOC must be strictly greater than 0 and less than or equal to 1.0. Found {len(invalid_moc)} invalid rows."
            )

    if "Policy_Number" in df.columns and "Duration" in df.columns:
        duplicates = df[df.duplicated(subset=["Policy_Number", "Duration"], keep=False)]
        if not duplicates.empty:
            issues.append(
                f"Duplicate Policy_Number + Duration combinations found: {duplicates.groupby(['Policy_Number', 'Duration']).ngroups} unique pairs."
            )

    if "MAC" in df.columns and "COLA" in df.columns:
        mac0_cola_not_null = df[(df["MAC"] == 0) & (df["COLA"].notna()) & (df["COLA"] != "")]
        if not mac0_cola_not_null.empty:
            issues.append(
                f"COLA must be null when MAC=0. Found {len(mac0_cola_not_null)} invalid rows."
            )
        mac1_cola_null = df[(df["MAC"] == 1) & (df["COLA"].isna() | (df["COLA"] == ""))]
        if not mac1_cola_null.empty:
            issues.append(
                f"COLA must not be null when MAC=1. Found {len(mac1_cola_null)} invalid rows."
            )

    if "MAC" in df.columns and "MOC" in df.columns:
        mac1_rows = df[df["MAC"] == 1]
        moc_not_one = mac1_rows[(mac1_rows["MOC"] - 1.0).abs() > 1e-9]
        if not moc_not_one.empty:
            issues.append(
                f"MOC must be exactly 1.0 when MAC=1. Found {len(moc_not_one)} invalid rows."
            )

    if "Policy_Number" in df.columns and "Duration" in df.columns and "MAC" in df.columns:
        death_rows = df[df["MAC"] == 1][["Policy_Number", "Duration"]]
        violations = 0
        for _, row in death_rows.iterrows():
            higher_duration = df[
                (df["Policy_Number"] == row["Policy_Number"])
                & (df["Duration"] > row["Duration"])
            ]
            violations += len(higher_duration)
        if violations:
            issues.append(
                f"Death exposure logic violated by {violations} rows after the death duration."
            )

    status = "PASS" if not issues else "FAIL"
    return _tool_result(
        True,
        "validation",
        f"Actuarial validation completed for `{source_path}` with status `{status}`.",
        data={
            "status": status,
            "issues": issues,
            "feature_classification": {
                column: _classify_feature_type(df, column) for column in df.columns
            },
        },
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


def _compute_mortality_rate_ci(
    mac: float,
    moc: float,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    if pd.isna(mac) or pd.isna(moc) or moc <= 0 or mac < 0 or mac > moc:
        return (None, None)
    alpha_beta = mac + 0.5
    beta_beta = moc - mac + 0.5
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    lower_rate = stats.beta.ppf(lower_quantile, alpha_beta, beta_beta)
    upper_rate = stats.beta.ppf(upper_quantile, alpha_beta, beta_beta)
    return (lower_rate, upper_rate)


def compute_ae_ci(
    mac: float,
    moc: float,
    mec: float,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    rate_lower, rate_upper = _compute_mortality_rate_ci(mac, moc, confidence_level)
    if rate_lower is None or rate_upper is None or mec <= 0:
        return (None, None)
    credible_deaths_lower = rate_lower * moc
    credible_deaths_upper = rate_upper * moc
    return (credible_deaths_lower / mec, credible_deaths_upper / mec)


def compute_ae_ci_amount(
    mac: float,
    moc: float,
    mec: float,
    actual_amount: float,
    expected_amount: float,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    if (
        pd.isna(mac)
        or pd.isna(moc)
        or pd.isna(mec)
        or pd.isna(actual_amount)
        or pd.isna(expected_amount)
        or mec <= 0
        or expected_amount <= 0
    ):
        return (None, None)

    average_claim = actual_amount / mac if mac > 0 else expected_amount / mec
    rate_lower, rate_upper = _compute_mortality_rate_ci(mac, moc, confidence_level)
    if rate_lower is None or rate_upper is None:
        return (None, None)

    credible_amount_lower = rate_lower * moc * average_claim
    credible_amount_upper = rate_upper * moc * average_claim
    return (credible_amount_lower / expected_amount, credible_amount_upper / expected_amount)


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


def _validate_metric(metric: str) -> None:
    if metric not in {"count", "amount"}:
        raise ValueError("metric must be 'count' or 'amount'")


def _metric_columns(metric: str) -> dict[str, str]:
    _validate_metric(metric)
    if metric == "count":
        return {
            "ratio": "AE_Ratio_Count",
            "actual": "Sum_MAC",
            "expected": "Sum_MEC",
            "ci_low": "AE_Count_CI_Lower",
            "ci_high": "AE_Count_CI_Upper",
        }
    return {
        "ratio": "AE_Ratio_Amount",
        "actual": "Sum_MAF",
        "expected": "Sum_MEF",
        "ci_low": "AE_Amount_CI_Lower",
        "ci_high": "AE_Amount_CI_Upper",
    }


def _metric_label(metric: str) -> str:
    return "Count" if metric == "count" else "Amount"


def _ratio_label(metric: str) -> str:
    return f"A/E Ratio ({_metric_label(metric)})"


def _required_columns(df: pd.DataFrame, columns: Sequence[str], data_path: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")


def _build_scatter_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    required = [
        "Dimensions",
        "Sum_MOC",
        metric_columns["actual"],
        metric_columns["expected"],
        metric_columns["ratio"],
        metric_columns["ci_low"],
        metric_columns["ci_high"],
    ]
    _required_columns(df, required, data_path)
    prepared = df.copy().sort_values(metric_columns["ratio"], ascending=False)
    prepared["display_ratio"] = prepared[metric_columns["ratio"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_low"] = prepared[metric_columns["ci_low"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_high"] = prepared[metric_columns["ci_high"]].clip(lower=0, upper=_SCATTER_X_MAX)

    fig = go.Figure(
        go.Scatter(
            x=prepared["display_ratio"],
            y=prepared["Dimensions"],
            mode="markers",
            marker={
                "color": _NEUTRAL_COLOR,
                "size": 14,
                "line": {"color": "#ffffff", "width": 1.2},
                "opacity": 0.92,
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (prepared["display_ci_high"] - prepared["display_ratio"]).clip(lower=0),
                "arrayminus": (prepared["display_ratio"] - prepared["display_ci_low"]).clip(lower=0),
                "visible": True,
                "color": _NEUTRAL_COLOR,
                "thickness": 1.4,
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{_ratio_label(metric)}: "
                "%{x:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#c05252", line_width=2)
    fig.update_xaxes(title_text=_ratio_label(metric), range=[0, _SCATTER_X_MAX])
    fig.update_yaxes(title_text="Cohort", autorange="reversed")
    fig.update_layout(
        title=f"Forest Plot ({_metric_label(metric)})",
        height=max(460, min(320 + max(len(prepared), 1) * 54, 920)),
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#ffffff",
        font={"color": "#1f2933", "family": "Avenir Next, Segoe UI, Arial, sans-serif"},
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
        showlegend=False,
    )
    return fig


def _build_table_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    required = [
        "Dimensions",
        "Sum_MOC",
        metric_columns["actual"],
        metric_columns["expected"],
        metric_columns["ratio"],
        metric_columns["ci_low"],
        metric_columns["ci_high"],
    ]
    _required_columns(df, required, data_path)
    prepared = df.copy().sort_values(metric_columns["ratio"], ascending=False)
    fig = go.Figure(
        go.Table(
            header={
                "values": [
                    "Cohort",
                    "Exposure (MOC)",
                    "Actual",
                    "Expected",
                    _ratio_label(metric),
                    "95% CI",
                ],
                "fill_color": "#e7eef5",
                "align": "left",
                "font": {"color": "#102a43", "size": 12},
            },
            cells={
                "values": [
                    prepared["Dimensions"],
                    prepared["Sum_MOC"].map(lambda value: f"{float(value):,.2f}"),
                    prepared[metric_columns["actual"]].map(lambda value: f"{float(value):,.2f}"),
                    prepared[metric_columns["expected"]].map(lambda value: f"{float(value):,.2f}"),
                    prepared[metric_columns["ratio"]].map(lambda value: f"{float(value):.2f}"),
                    prepared[metric_columns["ci_low"]].map(lambda value: f"{float(value):.1%}")
                    + " - "
                    + prepared[metric_columns["ci_high"]].map(lambda value: f"{float(value):.1%}"),
                ],
                "align": "left",
                "fill_color": "#ffffff",
                "height": 30,
            },
        )
    )
    fig.update_layout(
        title=f"Filtered Cohort Detail ({_metric_label(metric)})",
        height=max(260, min(140 + max(len(prepared), 1) * 32, 1800)),
        paper_bgcolor="#fbfaf7",
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
    )
    return fig


def _split_dimensions(label: str) -> list[str]:
    return [part.strip() for part in str(label).split("|") if part.strip()]


def _build_treemap_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    value_column = "Sum_MOC" if metric == "count" else "Sum_MAF"
    required = ["Dimensions", metric_columns["ratio"], value_column]
    _required_columns(df, required, data_path)

    parents = []
    labels = []
    values = []
    colors = []
    for _, row in df.iterrows():
        parts = _split_dimensions(row["Dimensions"])
        labels.append(parts[-1] if parts else row["Dimensions"])
        parents.append(" / ".join(parts[:-1]) if len(parts) > 1 else "")
        values.append(float(row[value_column]))
        colors.append(float(row[metric_columns["ratio"]]))

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker={
                "colors": colors,
                "colorscale": "RdYlGn_r",
                "cmid": 1.0,
                "cmin": 0.0,
                "cmax": _TREEMAP_COLOR_MAX,
                "line": {"width": 1, "color": "#ffffff"},
            },
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"{_ratio_label(metric)}: "
                "%{color:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Risk Treemap ({_metric_label(metric)})",
        height=900,
        paper_bgcolor="#fbfaf7",
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
    )
    return fig


def _figure_fragment(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_FIGURE_CONFIG)


def _build_report_html(
    *,
    title: str,
    metric: str,
    scatter_fragment: str,
    table_fragment: str,
    treemap_fragment: str,
) -> str:
    plotly_js = get_plotlyjs()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 32px 24px 48px;
      background:
        radial-gradient(circle at top left, rgba(31, 78, 121, 0.08), transparent 28%),
        linear-gradient(180deg, #faf7f2 0%, #f6f3ed 100%);
      color: #1f2933;
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
    }}
    .report-shell {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    .report-section {{
      margin-top: 26px;
      padding: 22px 22px 18px;
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 24px;
      background: #ffffff;
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.08);
    }}
  </style>
  <script type="text/javascript">{plotly_js}</script>
</head>
<body>
  <main class="report-shell">
    <section class="report-section">{scatter_fragment}</section>
    <section class="report-section">{table_fragment}</section>
    <section class="report-section">{treemap_fragment}</section>
  </main>
</body>
</html>
"""


def generate_combined_report(
    *,
    context: ToolExecutionContext,
    metric: str = "amount",
    data_path: str | None = None,
) -> dict[str, Any]:
    source_path = Path(data_path) if data_path else context.latest_sweep_path
    if source_path is None or not source_path.exists():
        return _error_result(
            "missing_prerequisite",
            "Missing prerequisite. Run dimensional sweep first.",
        )

    context.emit_status("Generating the combined visualization report.")
    df = pd.read_csv(source_path)
    scatter_fragment = _figure_fragment(_build_scatter_figure(df, metric, str(source_path)))
    table_fragment = _figure_fragment(_build_table_figure(df, metric, str(source_path)))
    treemap_fragment = _figure_fragment(_build_treemap_figure(df, metric, str(source_path)))
    report_html = _build_report_html(
        title=f"Combined A/E Visualization Report ({_metric_label(metric)})",
        metric=metric,
        scatter_fragment=scatter_fragment,
        table_fragment=table_fragment,
        treemap_fragment=treemap_fragment,
    )

    _ensure_output_dir(context)
    visualization_path = context.next_visualization_path()
    visualization_path.write_text(report_html, encoding="utf-8")
    return _tool_result(
        True,
        "visualization",
        f"Generated the combined visualization report at `{visualization_path}`.",
        artifacts={
            "sweep_summary_path": str(source_path),
            "visualization_path": str(visualization_path),
        },
        data={"metric": metric},
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
