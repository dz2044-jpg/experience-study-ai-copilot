import json
from pathlib import Path

import pandas as pd

from tools.insight_engine import compute_ae_ci_amount, run_dimensional_sweep


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def _pair_keys(dimensions: str):
    return frozenset(part.split("=")[0] for part in dimensions.split(" | "))


def test_pairwise_combinatorial_sweep_generates_all_requested_pairs(tmp_path, monkeypatch):
    # Emulate project data/output structure in a temp working directory.
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = output_dir / "analysis_inforce.csv"
    analysis_path.write_text(FIXTURE_PATH.read_text())

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

    dynamic_file = tmp_path / result["output_path"]
    latest_file = tmp_path / result["latest_output_path"]
    assert dynamic_file.exists()
    assert latest_file.exists()

    dims = [_pair_keys(r["Dimensions"]) for r in result["results"]]
    observed_pairs = set(dims)
    expected_pairs = {
        frozenset({"Smoker", "Risk_Class"}),
        frozenset({"Smoker", "Issue_Age_band"}),
        frozenset({"Risk_Class", "Issue_Age_band"}),
    }
    assert expected_pairs.issubset(observed_pairs)

    # Validate sweep summary CSV contains flattened CI columns for downstream visualization.
    df = pd.read_csv(dynamic_file)
    assert "AE_Count_CI_Lower" in df.columns
    assert "AE_Count_CI_Upper" in df.columns
    assert "AE_Amount_CI_Lower" in df.columns
    assert "AE_Amount_CI_Upper" in df.columns


def test_compute_ae_ci_amount_returns_none_for_zero_denominators():
    assert compute_ae_ci_amount(0, 1, 0, 0, 0) == (None, None)
