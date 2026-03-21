"""Shared tabular input utilities for raw dataset ingestion."""

from pathlib import Path
from typing import Optional

import pandas as pd


ACTUARIAL_NUMERICS = ["MAC", "MEC", "MAF", "MEF", "MOC"]
SUPPORTED_TABULAR_SUFFIXES = {".csv", ".parquet", ".xlsx"}


def list_excel_sheets(path: str) -> list[str]:
    """Return sheet names for an XLSX workbook."""
    workbook = pd.ExcelFile(path, engine="openpyxl")
    return workbook.sheet_names


def _resolve_sheet_name(path: Path, sheet_name: Optional[str]) -> Optional[str]:
    """Default XLSX reads to the first worksheet when none is specified."""
    if path.suffix.lower() != ".xlsx":
        return None
    if sheet_name:
        return sheet_name
    sheets = list_excel_sheets(str(path))
    return sheets[0] if sheets else None


def _read_tabular_input(
    path: str,
    sheet_name: Optional[str] = None,
    *,
    raw_strings: bool = False,
) -> pd.DataFrame:
    """Read a supported tabular input format."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()

    if suffix not in SUPPORTED_TABULAR_SUFFIXES:
        raise ValueError(
            f"Unsupported file type: {suffix or '<none>'}. "
            f"Supported formats: {sorted(SUPPORTED_TABULAR_SUFFIXES)}"
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
    read_kwargs = {"sheet_name": resolved_sheet_name, "engine": "openpyxl"}
    if raw_strings:
        read_kwargs.update({"dtype": str, "keep_default_na": False})
    return pd.read_excel(input_path, **read_kwargs)


def load_tabular_input(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load a raw tabular dataset and normalize core actuarial column types."""
    df = _read_tabular_input(path, sheet_name=sheet_name, raw_strings=False)

    for col in ACTUARIAL_NUMERICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "Policy_Number" in df.columns and not pd.api.types.is_string_dtype(df["Policy_Number"]):
        df["Policy_Number"] = df["Policy_Number"].astype(str)

    return df


def load_tabular_input_as_strings(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load a raw tabular dataset as strings for source-value validation checks."""
    return _read_tabular_input(path, sheet_name=sheet_name, raw_strings=True)
