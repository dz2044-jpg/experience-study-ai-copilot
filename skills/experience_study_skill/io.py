"""I/O helpers and shared deterministic context for the experience study skill."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq


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
