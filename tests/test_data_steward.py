import json
from pathlib import Path
import time

import pandas as pd

from tools import data_steward


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def test_hard_schema_casts_actuarial_columns_to_float64(tmp_path):
    source = tmp_path / "inforce.csv"
    source.write_text(FIXTURE_PATH.read_text())

    df = data_steward._load_inforce(str(source))
    for col in ["MAC", "MEC", "MAF", "MEF", "MOC"]:
        assert col in df.columns
        assert str(df[col].dtype) == "float64"


def test_session_aware_feature_engineering_appends_columns(tmp_path, monkeypatch):
    source = tmp_path / "inforce.csv"
    source.write_text(FIXTURE_PATH.read_text())

    output = tmp_path / "analysis_inforce.parquet"
    monkeypatch.setattr(data_steward, "ANALYSIS_OUTPUT_PATH", str(output))
    # Ensure writes in this test are considered "current session".
    monkeypatch.setattr(data_steward, "_SESSION_START_TIME", time.time() - 60)

    data_steward.create_categorical_bands(
        source_column="Face_Amount",
        strategy="equal_width",
        bins=4,
        source_path=str(source),
    )
    data_steward.create_categorical_bands(
        source_column="Issue_Age",
        strategy="equal_width",
        bins=4,
        source_path=str(source),
    )

    # Add a regrouping call to ensure other feature-engineering tools also append.
    data_steward.regroup_categorical_features(
        source_column="Risk_Class",
        mapping_dict={"Standard": "Std", "Preferred": "Pref"},
        source_path=str(output),
    )

    assert output.exists()
    df = pd.read_parquet(output)
    assert "Face_Amount_band" in df.columns
    assert "Issue_Age_band" in df.columns
    assert "Risk_Class_regrouped" in df.columns


def test_run_actuarial_data_checks_flags_non_numeric_raw_actuarial_values(tmp_path):
    source = tmp_path / "bad_inforce.csv"
    source.write_text(
        "Policy_Number,Duration,MAC,MEC,MOC,Face_Amount,Issue_Age,MEF,MAF,COLA\n"
        "P1,1,foo,0.5,1.0,100000,35,1000,0,\n"
    )

    result = data_steward.run_actuarial_data_checks(str(source))

    assert '"status": "FAIL"' in result
    assert "MAC contains 1 non-numeric raw value" in result


def test_run_actuarial_data_checks_keeps_actuarial_fields_numerical(tmp_path):
    source = tmp_path / "inforce.csv"
    source.write_text(FIXTURE_PATH.read_text())

    result = json.loads(data_steward.run_actuarial_data_checks(str(source)))
    feature_classification = result["feature_classification"]

    for col in ["MAC", "MEC", "MOC", "MAF", "MEF"]:
        assert feature_classification[col] == "numerical"


def test_profile_dataset_supports_parquet_input(tmp_path):
    source_df = pd.read_csv(FIXTURE_PATH)
    source = tmp_path / "inforce.parquet"
    source_df.to_parquet(source, index=False)

    result = json.loads(data_steward.profile_dataset(str(source)))

    assert result["total_rows"] == len(source_df)
    assert result["data_types"]["MAC"] == "float64"


def test_create_categorical_bands_supports_xlsx_sheet_input(tmp_path, monkeypatch):
    source_df = pd.read_csv(FIXTURE_PATH)
    workbook_path = tmp_path / "inforce.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        pd.DataFrame(
            {
                "Policy_Number": ["IGNORE"],
                "MAC": [0],
                "MEC": [0.1],
                "MOC": [1],
                "MAF": [0],
                "MEF": [10],
                "Face_Amount": [1000],
                "Issue_Age": [30],
                "Risk_Class": ["Standard"],
            }
        ).to_excel(writer, sheet_name="IgnoreMe", index=False)
        source_df.to_excel(writer, sheet_name="Inforce", index=False)

    output = tmp_path / "analysis_inforce.parquet"
    monkeypatch.setattr(data_steward, "ANALYSIS_OUTPUT_PATH", str(output))
    monkeypatch.setattr(data_steward, "_SESSION_START_TIME", time.time() - 60)

    result = json.loads(
        data_steward.create_categorical_bands(
            source_column="Face_Amount",
            strategy="equal_width",
            bins=4,
            source_path=str(workbook_path),
            sheet_name="Inforce",
        )
    )

    assert result["success"] is True
    engineered = pd.read_parquet(output)
    assert "Face_Amount_band" in engineered.columns
    assert "IGNORE" not in engineered["Policy_Number"].astype(str).tolist()


def test_create_categorical_bands_reads_legacy_prepared_csv_when_parquet_is_missing(tmp_path, monkeypatch):
    legacy_csv = tmp_path / "analysis_inforce.csv"
    legacy_csv.write_text(FIXTURE_PATH.read_text())
    parquet_output = tmp_path / "analysis_inforce.parquet"

    monkeypatch.setattr(data_steward, "ANALYSIS_OUTPUT_PATH", str(parquet_output))
    monkeypatch.setattr(data_steward, "_SESSION_START_TIME", time.time() - 60)

    result = json.loads(
        data_steward.create_categorical_bands(
            source_column="Face_Amount",
            strategy="equal_width",
            bins=4,
            source_path=str(parquet_output),
        )
    )

    assert result["success"] is True
    engineered = pd.read_parquet(parquet_output)
    assert "Face_Amount_band" in engineered.columns


def test_quantile_banding_returns_controlled_error_when_realized_bins_collapse_below_three(tmp_path, monkeypatch):
    source_df = pd.concat([pd.read_csv(FIXTURE_PATH)] * 3, ignore_index=True)
    source_df["Face_Amount"] = 100000
    source_df.loc[source_df.index[-2:], "Face_Amount"] = 250000

    source = tmp_path / "skewed_inforce.csv"
    source_df.to_csv(source, index=False)

    output = tmp_path / "analysis_inforce.parquet"
    monkeypatch.setattr(data_steward, "ANALYSIS_OUTPUT_PATH", str(output))
    monkeypatch.setattr(data_steward, "_SESSION_START_TIME", time.time() - 60)

    result = json.loads(
        data_steward.create_categorical_bands(
            source_column="Face_Amount",
            strategy="quantiles",
            bins=5,
            source_path=str(source),
        )
    )

    assert "produced only" in result["error"]
    assert not output.exists()
