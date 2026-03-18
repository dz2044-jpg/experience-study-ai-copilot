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

    output = tmp_path / "analysis_inforce.csv"
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
    df = pd.read_csv(output)
    assert "Face_Amount_band" in df.columns
    assert "Issue_Age_band" in df.columns
    assert "Risk_Class_regrouped" in df.columns
