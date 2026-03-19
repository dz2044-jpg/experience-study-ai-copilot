import json
from pathlib import Path

import pandas as pd

from agents.agent_steward import DataStewardAgent
from agents.schemas import DimensionalSweepSchema
from tools import data_steward


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def test_dimensional_sweep_schema_exposes_selected_columns():
    schema = DimensionalSweepSchema.model_json_schema()
    assert "selected_columns" in schema["properties"]


def test_steward_agent_can_execute_regroup_tool(tmp_path, monkeypatch):
    source = tmp_path / "analysis_inforce.csv"
    source.write_text(FIXTURE_PATH.read_text())

    output = tmp_path / "analysis_output.csv"
    monkeypatch.setattr(data_steward, "ANALYSIS_OUTPUT_PATH", str(output))
    agent = DataStewardAgent()

    result = json.loads(
        agent._execute_tool(
            "regroup_categorical_features",
            {
                "source_column": "Risk_Class",
                "mapping_dict": {"Standard": "Std", "Preferred": "Pref"},
                "source_path": str(source),
                "output_path": str(output),
            },
        )
    )

    saved_path = Path(result["output_path"])
    assert result["success"] is True
    assert saved_path.exists()

    df = pd.read_csv(saved_path)
    assert "Risk_Class_regrouped" in df.columns
