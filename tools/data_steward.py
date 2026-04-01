"""
Data Steward Agent backend: Deterministic functions for data validation and feature engineering.

CRITICAL: Never overwrite original user-uploaded data. All outputs save to
data/output/analysis_inforce.parquet.
"""

from dataclasses import dataclass
import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from tools.data_io import (
    CANONICAL_ANALYSIS_OUTPUT_PATH,
    load_tabular_input,
    load_tabular_input_as_strings,
    resolve_prepared_analysis_path,
)

# Records the exact time this script is loaded into memory for the chat session.
_SESSION_START_TIME = time.time()

ANALYSIS_OUTPUT_PATH = CANONICAL_ANALYSIS_OUTPUT_PATH
ACTUARIAL_NUMERICS = ["MAC", "MEC", "MAF", "MEF", "MOC"]
SEMANTIC_NUMERICAL_FEATURES = {"Face_Amount", "Issue_Age", "Age"}
RAW_MISSING_TOKENS = {"", "na", "nan", "null", "none", "n/a"}


@dataclass(frozen=True)
class CategoricalBandSpec:
    """Deterministic specification for a single categorical banding operation."""

    source_column: str
    strategy: str
    bins: Optional[int] = None
    custom_bins: Optional[list[float]] = None


def _load_inforce(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load supported raw input formats and normalize core inforce types."""
    return load_tabular_input(path, sheet_name=sheet_name)


def _load_feature_engineering_frame(
    source_path: str,
    output_path: str,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Session-aware load for feature engineering:
    - If output exists and was modified during this session, append to it.
    - Otherwise, start fresh from source input.
    """
    if os.path.exists(output_path) and os.path.getmtime(output_path) >= _SESSION_START_TIME:
        return _load_inforce(output_path)
    resolved_source_path = resolve_prepared_analysis_path(source_path)
    return _load_inforce(str(resolved_source_path), sheet_name=sheet_name)


def _find_raw_non_numeric_values(data_path: str, sheet_name: Optional[str] = None) -> list[str]:
    """Identify raw source values in actuarial numeric columns that cannot be parsed as numbers."""
    raw_df = load_tabular_input_as_strings(data_path, sheet_name=sheet_name)
    issues: list[str] = []

    for col in ACTUARIAL_NUMERICS:
        if col not in raw_df.columns:
            continue

        raw_values = raw_df[col].fillna("").astype(str).str.strip()
        missing_mask = raw_values.str.lower().isin(RAW_MISSING_TOKENS)
        parsed_values = pd.to_numeric(raw_values.where(~missing_mask, pd.NA), errors="coerce")
        invalid_count = int((~missing_mask & parsed_values.isna()).sum())

        if invalid_count > 0:
            issues.append(
                f"{col} contains {invalid_count} non-numeric raw value(s) that cannot be parsed."
            )

    return issues


def _classify_feature_type(df: pd.DataFrame, column: str) -> str:
    """
    Classify a column for Steward reporting.

    Core actuarial measure fields remain numerical even when they have low cardinality
    because their semantic meaning is quantitative rather than dimensional.
    """
    if column in ACTUARIAL_NUMERICS or column in SEMANTIC_NUMERICAL_FEATURES:
        return "numerical"

    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        return "numerical" if series.nunique() > 20 else "categorical"
    return "categorical"


def profile_dataset(
    data_path: str = "data/input/synthetic_inforce.csv",
    sheet_name: Optional[str] = None,
) -> str:
    """
    Profile the dataset and return descriptive statistics as a JSON string.

    Returns total rows, columns, data types, memory usage, unique policy count,
    and null counts per column.
    """
    path = Path(data_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {data_path}"}, indent=2)

    df = _load_inforce(data_path, sheet_name=sheet_name)

    null_counts = {col: int(df[col].isna().sum()) for col in df.columns}

    # Use deep memory usage for accurate byte count (includes object overhead)
    memory_bytes = int(df.memory_usage(deep=True).sum())

    unique_policies = 0
    if "Policy_Number" in df.columns:
        unique_policies = int(df["Policy_Number"].nunique())

    data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

    result = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "data_types": data_types,
        "memory_usage_bytes": memory_bytes,
        "memory_usage_human": f"{memory_bytes / 1024:.2f} KB",
        "unique_policy_count": unique_policies,
        "null_counts": null_counts,
    }

    return json.dumps(result, indent=2)


def run_actuarial_data_checks(
    data_path: str = "data/input/synthetic_inforce.csv",
    sheet_name: Optional[str] = None,
) -> str:
    """
    Validate the dataset against actuarial rules and return PASS/FAIL with specific issues.
    """
    path = Path(data_path)
    if not path.exists():
        return json.dumps(
            {"status": "FAIL", "issues": [f"File not found: {data_path}"]},
            indent=2,
        )

    issues = _find_raw_non_numeric_values(data_path, sheet_name=sheet_name)
    df = _load_inforce(data_path, sheet_name=sheet_name)

    # --- Type checks ---
    if "Policy_Number" in df.columns:
        if not pd.api.types.is_string_dtype(df["Policy_Number"]):
            issues.append(
                "Policy_Number must be a string type (e.g., read with dtype=str to preserve leading zeros)."
            )

    numeric_cols = ["MAC", "MEC", "MOC"]
    for col in ["MAF", "MEF"]:
        if col in df.columns:
            numeric_cols.append(col)

    for col in numeric_cols:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col} must be numeric.")

    int_cols = ["Face_Amount", "Duration"]
    age_col = "Issue_Age" if "Issue_Age" in df.columns else "Age"
    if age_col in df.columns:
        int_cols.append(age_col)

    for col in int_cols:
        if col in df.columns and not pd.api.types.is_integer_dtype(df[col]):
            # Check if float with no decimals (e.g., 31.0)
            if pd.api.types.is_numeric_dtype(df[col]):
                if not (df[col].dropna() == df[col].dropna().astype(int)).all():
                    issues.append(f"{col} must be integer (found non-integer values).")
            else:
                issues.append(f"{col} must be integer.")

    # --- Feature classification ---
    feature_classification = {}
    for col in df.columns:
        feature_classification[col] = _classify_feature_type(df, col)

    # --- Value limits ---
    if "MAC" in df.columns:
        invalid_mac = df[~df["MAC"].isin([0, 1])].dropna(subset=["MAC"])
        if len(invalid_mac) > 0:
            issues.append(f"MAC must be strictly 0 or 1. Found {len(invalid_mac)} invalid rows.")

    if "MEC" in df.columns:
        mec_valid = (df["MEC"] > 0) & (df["MEC"] < 1)
        invalid_mec = df[~mec_valid].dropna(subset=["MEC"])
        if len(invalid_mec) > 0:
            issues.append(
                f"MEC must be strictly between 0 and 1 (exclusive). Found {len(invalid_mec)} invalid rows."
            )

    if "Face_Amount" in df.columns:
        invalid_face = df[df["Face_Amount"] <= 0]
        if len(invalid_face) > 0:
            issues.append(f"Face_Amount must be > 0. Found {len(invalid_face)} invalid rows.")

    if age_col in df.columns:
        invalid_age = df[df[age_col] <= 0]
        if len(invalid_age) > 0:
            issues.append(f"{age_col} must be > 0 (no negative ages). Found {len(invalid_age)} invalid rows.")

    if "MOC" in df.columns:
        moc_valid = (df["MOC"] > 0) & (df["MOC"] <= 1.0)
        invalid_moc = df[~moc_valid].dropna(subset=["MOC"])
        if len(invalid_moc) > 0:
            issues.append(
                f"MOC must be strictly greater than 0 and less than or equal to 1.0. Found {len(invalid_moc)} invalid rows."
            )

    # --- Logic checks ---
    if "Policy_Number" in df.columns and "Duration" in df.columns:
        dupes = df[df.duplicated(subset=["Policy_Number", "Duration"], keep=False)]
        if len(dupes) > 0:
            n_dupe_pairs = dupes.groupby(["Policy_Number", "Duration"]).ngroups
            issues.append(
                f"Duplicate Policy_Number + Duration combinations: {n_dupe_pairs} unique pairs affected."
            )

    if "MAC" in df.columns and "COLA" in df.columns:
        mac0_cola_not_null = df[(df["MAC"] == 0) & (df["COLA"].notna()) & (df["COLA"] != "")]
        if len(mac0_cola_not_null) > 0:
            issues.append(
                f"COLA logic violation: When MAC=0 (survival), COLA must be null. Found {len(mac0_cola_not_null)} rows."
            )
        mac1_cola_null = df[(df["MAC"] == 1) & (df["COLA"].isna() | (df["COLA"] == ""))]
        if len(mac1_cola_null) > 0:
            issues.append(
                f"COLA logic violation: When MAC=1 (death), COLA must not be null. Found {len(mac1_cola_null)} rows."
            )

    if "MAC" in df.columns and "MOC" in df.columns:
        mac1_rows = df[df["MAC"] == 1]
        moc_not_one = mac1_rows[(mac1_rows["MOC"] - 1.0).abs() > 1e-9]
        if len(moc_not_one) > 0:
            issues.append(
                f"MOC logic violation: When MAC=1 (claim), MOC must be exactly 1.0. Found {len(moc_not_one)} rows."
            )

    # Death exposure: if MAC==1 at duration D, no rows with same policy and duration > D
    if "Policy_Number" in df.columns and "Duration" in df.columns and "MAC" in df.columns:
        death_rows = df[df["MAC"] == 1][["Policy_Number", "Duration"]]
        violations = 0
        for _, row in death_rows.iterrows():
            policy, death_dur = row["Policy_Number"], row["Duration"]
            higher_dur = df[(df["Policy_Number"] == policy) & (df["Duration"] > death_dur)]
            violations += len(higher_dur)
        if violations > 0:
            issues.append(
                f"Death exposure logic: {violations} rows have Duration > death Duration for same Policy_Number."
            )

    status = "PASS" if len(issues) == 0 else "FAIL"
    result = {
        "status": status,
        "issues": issues,
        "feature_classification": feature_classification,
    }

    return json.dumps(result, indent=2)


def create_categorical_bands(
    source_column: str,
    strategy: str,
    bins: Optional[int] = None,
    custom_bins: Optional[list] = None,
    source_path: str = "data/input/synthetic_inforce.csv",
    output_path: str = CANONICAL_ANALYSIS_OUTPUT_PATH,
    sheet_name: Optional[str] = None,
) -> str:
    """
    Create banded categorical column from a numeric source column.

    Strategies: quantiles (pd.qcut), equal_width (pd.cut), custom (pd.cut with custom_bins).
    Saves to output_path; never overwrites source. Returns JSON success message.
    """
    batch_result = create_categorical_bands_batch(
        band_specs=[
            CategoricalBandSpec(
                source_column=source_column,
                strategy=strategy,
                bins=bins,
                custom_bins=custom_bins,
            )
        ],
        source_path=source_path,
        output_path=output_path,
        sheet_name=sheet_name,
    )
    payload = json.loads(batch_result)
    if payload.get("error"):
        return batch_result

    operation = payload["operations"][0]
    result = {
        "success": True,
        "new_column": operation["new_column"],
        "output_path": payload["output_path"],
        "strategy": operation["strategy"],
        "bins": operation.get("bins"),
    }
    return json.dumps(result, indent=2)


def _resolved_band_bins(spec: CategoricalBandSpec) -> Optional[int]:
    """Resolve implicit default bin counts for supported strategies."""
    if spec.strategy == "quantiles":
        return spec.bins if spec.bins is not None else 4
    if spec.strategy == "equal_width":
        return spec.bins if spec.bins is not None else 5
    return spec.bins


def _validate_categorical_band_specs(
    df: pd.DataFrame,
    band_specs: Sequence[CategoricalBandSpec],
) -> Optional[str]:
    """Validate all requested band specs before mutating or writing output."""
    if not band_specs:
        return "No categorical band specifications were provided."

    seen_new_columns: set[str] = set()
    for spec in band_specs:
        if spec.source_column not in df.columns:
            return f"Column '{spec.source_column}' not found in dataset."

        if not pd.api.types.is_numeric_dtype(df[spec.source_column]):
            return f"Column '{spec.source_column}' must be numeric for banding."

        if spec.strategy not in {"quantiles", "equal_width", "custom"}:
            return f"Unknown strategy: {spec.strategy}. Use quantiles, equal_width, or custom."

        resolved_bins = _resolved_band_bins(spec)
        if spec.strategy in {"quantiles", "equal_width"} and (resolved_bins is None or resolved_bins < 1):
            return f"Banding for '{spec.source_column}' requires a positive integer bin count."

        if spec.strategy == "custom" and not spec.custom_bins:
            return "custom strategy requires custom_bins list."

        new_column = f"{spec.source_column}_band"
        if new_column in seen_new_columns:
            return (
                f"Duplicate banding request for '{spec.source_column}' detected in the same prompt. "
                "Each source column can be banded at most once per deterministic batch."
            )
        seen_new_columns.add(new_column)

    return None


def _build_categorical_band_series(
    df: pd.DataFrame,
    spec: CategoricalBandSpec,
) -> tuple[Optional[pd.Series], Optional[str]]:
    """Build a single categorical band series without mutating the source frame."""
    try:
        if spec.strategy == "quantiles":
            resolved_bins = _resolved_band_bins(spec)
            quantile_bands = pd.qcut(df[spec.source_column], q=resolved_bins, labels=False, duplicates="drop")
            realized_bins = int(quantile_bands.dropna().nunique())
            if realized_bins < 3:
                return (
                    None,
                    (
                        f"Quantile banding for '{spec.source_column}' produced only {realized_bins} realized bin(s); "
                        "need at least 3 to keep the engineered feature."
                    ),
                )
            return (quantile_bands.astype(str), None)

        if spec.strategy == "equal_width":
            resolved_bins = _resolved_band_bins(spec)
            return (
                pd.cut(df[spec.source_column], bins=resolved_bins, labels=False, duplicates="drop").astype(str),
                None,
            )

        if spec.strategy == "custom":
            banded = pd.cut(
                df[spec.source_column],
                bins=spec.custom_bins,
                include_lowest=True,
                duplicates="drop",
            )
            return (banded.astype(str), None)
    except Exception as exc:
        return (None, f"Banding failed: {str(exc)}")

    return (None, f"Unknown strategy: {spec.strategy}. Use quantiles, equal_width, or custom.")


def create_categorical_bands_batch(
    band_specs: Sequence[CategoricalBandSpec],
    source_path: str = "data/input/synthetic_inforce.csv",
    output_path: str = CANONICAL_ANALYSIS_OUTPUT_PATH,
    sheet_name: Optional[str] = None,
) -> str:
    """Apply one or more banding operations with a single load/validate/write cycle."""
    src = resolve_prepared_analysis_path(source_path)
    if not src.exists():
        return json.dumps({"error": f"Source file not found: {source_path}"}, indent=2)

    output_path = ANALYSIS_OUTPUT_PATH
    df = _load_feature_engineering_frame(
        source_path=source_path,
        output_path=output_path,
        sheet_name=sheet_name,
    )

    validation_error = _validate_categorical_band_specs(df, band_specs)
    if validation_error:
        return json.dumps({"error": validation_error}, indent=2)

    engineered_df = df.copy()
    operations: list[dict[str, object]] = []

    for spec in band_specs:
        banded_series, build_error = _build_categorical_band_series(engineered_df, spec)
        if build_error:
            return json.dumps({"error": build_error}, indent=2)

        new_column = f"{spec.source_column}_band"
        engineered_df[new_column] = banded_series
        operations.append(
            {
                "source_column": spec.source_column,
                "new_column": new_column,
                "strategy": spec.strategy,
                "bins": _resolved_band_bins(spec),
            }
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    engineered_df.to_parquet(output_path, engine="pyarrow", index=False)

    return json.dumps(
        {
            "success": True,
            "output_path": output_path,
            "operations": operations,
        },
        indent=2,
    )


def regroup_categorical_features(
    source_column: str,
    mapping_dict: dict[str, str],
    source_path: str = CANONICAL_ANALYSIS_OUTPUT_PATH,
    output_path: str = CANONICAL_ANALYSIS_OUTPUT_PATH,
    sheet_name: Optional[str] = None,
) -> str:
    """
    Create regrouped categorical column using a mapping dictionary.
    Unmapped values are left as-is. Saves to output_path. Returns JSON success message.
    """
    src = resolve_prepared_analysis_path(source_path)
    if not src.exists() and not Path(ANALYSIS_OUTPUT_PATH).exists():
        return json.dumps({"error": f"Source file not found: {source_path}"}, indent=2)

    # Session-aware load to preserve previously appended columns in this chat.
    output_path = ANALYSIS_OUTPUT_PATH
    df = _load_feature_engineering_frame(
        source_path=source_path,
        output_path=output_path,
        sheet_name=sheet_name,
    )

    if source_column not in df.columns:
        return json.dumps(
            {"error": f"Column '{source_column}' not found in dataset."},
            indent=2,
        )

    new_col = f"{source_column}_regrouped"
    df[new_col] = df[source_column].astype(str).replace(mapping_dict)

    # Save and replace output file with appended feature columns.
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", index=False)

    result = {
        "success": True,
        "new_column": new_col,
        "output_path": output_path,
        "mapping_applied": mapping_dict,
    }
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    data_path = "data/input/synthetic_inforce.csv"

    print("=== 1. profile_dataset ===")
    print(profile_dataset(data_path=data_path))

    print("\n=== 2. run_actuarial_data_checks ===")
    print(run_actuarial_data_checks(data_path=data_path))

    print("\n=== 3. create_categorical_bands (Issue_Age, custom bins) ===")
    print(
        create_categorical_bands(
            source_column="Issue_Age",
            strategy="custom",
            custom_bins=[0, 25, 45, 65, 100],
            source_path=data_path,
            output_path=CANONICAL_ANALYSIS_OUTPUT_PATH,
        )
    )

    print("\n=== 4. regroup_categorical_features (Risk_Class) ===")
    print(
        regroup_categorical_features(
            source_column="Risk_Class",
            mapping_dict={
                "Standard": "Standard",
                "Standard Plus": "Standard",
                "Preferred": "Preferred",
                "Preferred Plus": "Preferred",
            },
            source_path=CANONICAL_ANALYSIS_OUTPUT_PATH,
            output_path=CANONICAL_ANALYSIS_OUTPUT_PATH,
        )
    )
