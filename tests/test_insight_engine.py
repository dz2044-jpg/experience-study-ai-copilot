import json
from pathlib import Path

import pandas as pd

from tools.insight_engine import compute_ae_ci_amount, run_dimensional_sweep


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def _pair_keys(dimensions: str):
    return frozenset(part.split("=")[0] for part in dimensions.split(" | "))


def _write_analysis_parquet(path: Path) -> None:
    pd.read_csv(FIXTURE_PATH).to_parquet(path, index=False)


def _write_wide_analysis_parquet(
    path: Path,
    dimension_names: list[str],
    *,
    signal_dimension: str | None = None,
) -> None:
    rows = []
    for index in range(20):
        high_signal = index < 10
        row = {
            "Policy_Number": f"P{index:03d}",
            "MAC": 1.0 if high_signal else 0.0,
            "MOC": 1.0,
            "MEC": 1.0,
            "MAF": 200.0 if high_signal else 50.0,
            "MEF": 100.0,
            "COLA": "",
            "Duration": 1,
        }
        for dimension_name in dimension_names:
            if dimension_name == signal_dimension:
                row[dimension_name] = "High" if high_signal else "Low"
            else:
                row[dimension_name] = "A"
        rows.append(row)

    pd.DataFrame(rows).to_parquet(path, index=False)


def test_pairwise_combinatorial_sweep_generates_all_requested_pairs(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=2,
            selected_columns=["Smoker", "Risk_Class", "Issue_Age_band"],
            min_mac=0,
            top_n=100,
            data_path=str(analysis_path),
        )
    )

    assert "results" in result
    assert "output_path" in result
    assert "latest_output_path" in result
    assert "latest_depth_output_path" in result
    assert result["depth"] == 2

    dynamic_file = tmp_path / result["output_path"]
    latest_file = tmp_path / result["latest_output_path"]
    latest_depth_file = tmp_path / result["latest_depth_output_path"]
    assert dynamic_file.exists()
    assert latest_file.exists()
    assert latest_depth_file.exists()

    dims = [_pair_keys(row["Dimensions"]) for row in result["results"]]
    observed_pairs = set(dims)
    expected_pairs = {
        frozenset({"Smoker", "Risk_Class"}),
        frozenset({"Smoker", "Issue_Age_band"}),
        frozenset({"Risk_Class", "Issue_Age_band"}),
    }
    assert expected_pairs.issubset(observed_pairs)

    df = pd.read_csv(dynamic_file)
    assert "AE_Count_CI_Lower" in df.columns
    assert "AE_Count_CI_Upper" in df.columns
    assert "AE_Amount_CI_Lower" in df.columns
    assert "AE_Amount_CI_Upper" in df.columns


def test_sweep_persists_full_ranked_csvs_while_json_respects_top_n(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Gender", "Smoker", "Risk_Class"],
            min_mac=0,
            top_n=2,
            sort_by="AE_Ratio_Count",
            data_path=str(analysis_path),
        )
    )

    assert len(result["results"]) == 2

    dynamic_file = tmp_path / result["output_path"]
    latest_file = tmp_path / result["latest_output_path"]
    latest_depth_file = tmp_path / result["latest_depth_output_path"]
    dynamic_df = pd.read_csv(dynamic_file)
    latest_df = pd.read_csv(latest_file)
    latest_depth_df = pd.read_csv(latest_depth_file)

    assert len(dynamic_df) > 2
    assert len(latest_df) > 2
    assert len(latest_depth_df) > 2
    assert dynamic_df["AE_Ratio_Count"].tolist() == sorted(dynamic_df["AE_Ratio_Count"].tolist(), reverse=True)
    assert latest_df["AE_Ratio_Count"].tolist() == sorted(latest_df["AE_Ratio_Count"].tolist(), reverse=True)
    assert latest_depth_df["AE_Ratio_Count"].tolist() == sorted(
        latest_depth_df["AE_Ratio_Count"].tolist(),
        reverse=True,
    )


def test_sweep_writes_depth_specific_latest_aliases(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    depth_to_expected = {
        1: "data/output/sweep_summary_latest_1.csv",
        2: "data/output/sweep_summary_latest_2.csv",
        3: "data/output/sweep_summary_latest_3.csv",
    }

    for depth, expected_path in depth_to_expected.items():
        result = json.loads(
            run_dimensional_sweep(
                depth=depth,
                selected_columns=["Gender", "Smoker", "Risk_Class"],
                min_mac=0,
                top_n=5,
                data_path=str(analysis_path),
            )
        )
        assert result["latest_depth_output_path"] == expected_path
        assert (tmp_path / expected_path).exists()


def test_auto_screening_keeps_high_signal_dimension_beyond_first_twelve(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    dimension_names = [f"Dim{index:02d}" for index in range(1, 15)]
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_wide_analysis_parquet(analysis_path, dimension_names, signal_dimension="Dim13")

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=2,
            top_n=200,
            data_path=str(analysis_path),
        )
    )

    assert "results" in result
    assert any("Dim13=" in row["Dimensions"] for row in result["results"])


def test_selected_columns_bypass_auto_screening(monkeypatch, tmp_path):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    from tools import insight_engine

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Auto-screening should not run for explicit selected_columns.")

    monkeypatch.setattr(insight_engine, "_rank_auto_screened_dimensions", fail_if_called)

    result = json.loads(
        run_dimensional_sweep(
            depth=2,
            selected_columns=["Gender", "Smoker"],
            top_n=10,
            data_path=str(analysis_path),
        )
    )

    assert "results" in result
    assert result["depth"] == 2


def test_explicit_selected_columns_return_controlled_error_when_combination_count_is_too_large(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    dimension_names = [f"Dim{index:02d}" for index in range(1, 16)]
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_wide_analysis_parquet(analysis_path, dimension_names)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=2,
            selected_columns=dimension_names,
            top_n=10,
            data_path=str(analysis_path),
        )
    )

    assert result["error"] == "Requested explicit sweep dimensions would generate too many combinations."
    assert result["requested_combination_count"] == 105
    assert result["max_supported_combinations"] == 100


def test_sweep_includes_zero_mac_cohorts_by_default(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=2,
            selected_columns=["Gender", "Smoker"],
            top_n=10,
            data_path=str(analysis_path),
        )
    )

    dimensions = {row["Dimensions"] for row in result["results"]}
    assert "Gender=F | Smoker=Yes" in dimensions or "Smoker=Yes | Gender=F" in dimensions


def test_sweep_supports_numeric_filters(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Gender"],
            filters=[{"column": "Duration", "operator": "<", "value": 3}],
            top_n=10,
            data_path=str(analysis_path),
        )
    )

    rows = {row["Dimensions"]: row for row in result["results"]}
    assert rows["Gender=F"]["Sum_MAC"] == 2
    assert rows["Gender=M"]["Sum_MAC"] == 2


def test_duration_is_eligible_as_an_explicit_sweep_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Duration"],
            top_n=20,
            data_path=str(analysis_path),
        )
    )

    assert "error" not in result
    assert any(row["Dimensions"].startswith("Duration=") for row in result["results"])


def test_raw_face_amount_is_not_an_eligible_sweep_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Face_Amount"],
            top_n=20,
            data_path=str(analysis_path),
        )
    )

    assert result["error"] == "Column 'Face_Amount' is not eligible as a sweep dimension."
    assert "Duration" in result["available_columns"]
    assert "Face_Amount" not in result["available_columns"]
    assert "Issue_Age" not in result["available_columns"]


def test_raw_issue_age_is_not_an_eligible_sweep_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Issue_Age"],
            top_n=20,
            data_path=str(analysis_path),
        )
    )

    assert result["error"] == "Column 'Issue_Age' is not eligible as a sweep dimension."
    assert "Duration" in result["available_columns"]
    assert "Face_Amount" not in result["available_columns"]
    assert "Issue_Age" not in result["available_columns"]


def test_auto_sweep_excludes_raw_face_amount_and_issue_age_but_includes_duration(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            top_n=200,
            data_path=str(analysis_path),
        )
    )

    dimensions = [row["Dimensions"] for row in result["results"]]
    assert any(dimension.startswith("Duration=") for dimension in dimensions)
    assert not any(dimension.startswith("Face_Amount=") for dimension in dimensions)
    assert not any(dimension.startswith("Issue_Age=") for dimension in dimensions)


def test_sweep_supports_string_filters(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Gender"],
            filters=[{"column": "Smoker", "operator": "=", "value": "Yes"}],
            top_n=10,
            data_path=str(analysis_path),
        )
    )

    rows = {row["Dimensions"]: row for row in result["results"]}
    assert rows["Gender=F"]["Sum_MAC"] == 0
    assert rows["Gender=M"]["Sum_MAC"] == 0


def test_sweep_falls_back_to_legacy_csv_when_parquet_is_missing(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = output_dir / "analysis_inforce.csv"
    legacy_path.write_text(FIXTURE_PATH.read_text())

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Gender"],
            top_n=10,
        )
    )

    assert "results" in result
    assert any(row["Dimensions"].startswith("Gender=") for row in result["results"])


def test_sweep_returns_controlled_error_for_invalid_filter_column(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Gender"],
            filters=[{"column": "State", "operator": "=", "value": "California"}],
            data_path=str(analysis_path),
        )
    )

    assert result["error"] == "Column 'State' not found."
    assert "available_columns" in result
    assert "Gender" in result["available_columns"]


def test_sweep_returns_controlled_error_for_invalid_operator(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)

    result = json.loads(
        run_dimensional_sweep(
            depth=1,
            selected_columns=["Gender"],
            filters=[{"column": "Duration", "operator": "LIKE", "value": "5"}],
            data_path=str(analysis_path),
        )
    )

    assert "Unsupported operator 'LIKE'" in result["error"]


def test_compute_ae_ci_amount_returns_none_for_zero_denominators():
    assert compute_ae_ci_amount(0, 1, 0, 0, 0) == (None, None)
