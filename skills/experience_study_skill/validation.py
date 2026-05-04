"""Actuarial validation checks for the experience study skill."""

from __future__ import annotations

from typing import Any

import pandas as pd

from skills.experience_study_skill.io import (
    ACTUARIAL_NUMERICS,
    RAW_MISSING_TOKENS,
    ToolExecutionContext,
    _choose_dataset_path,
    _classify_feature_type,
    _error_result,
    _tool_result,
    load_tabular_input,
    load_tabular_input_as_strings,
)


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
