import json
from pathlib import Path

import pandas as pd

from tools.insight_engine import compute_ae_ci_amount, run_dimensional_sweep


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def _pair_keys(dimensions: str):
    return frozenset(part.split("=")[0] for part in dimensions.split(" | "))


def _write_analysis_parquet(path: Path) -> None:
    pd.read_csv(FIXTURE_PATH).to_parquet(path, index=False)


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
