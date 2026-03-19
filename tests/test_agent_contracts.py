import json
from pathlib import Path

import pandas as pd

import agents.agent_actuary as agent_actuary_module
import agents.agent_analyst as agent_analyst_module
from agents.agent_actuary import ActuaryAgent
from agents.agent_analyst import AnalystAgent
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


def test_actuary_agent_runs_explicit_sweep_without_mapping_confirmation(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.csv"
    analysis_path.write_text(FIXTURE_PATH.read_text())

    captured = {}

    def fake_run_dimensional_sweep(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "results": [
                    {
                        "Dimensions": "Gender=F",
                        "AE_Ratio_Count": 1.2,
                        "AE_Ratio_Amount": 1.4,
                        "AE_Count_CI": [0.8, 1.8],
                        "AE_Amount_CI": [0.9, 2.0],
                    }
                ],
                "output_path": "data/output/sweep_summary_1_gender_smoker_issue_age_band.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
            }
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", fake_run_dimensional_sweep)

    agent = ActuaryAgent()
    response = agent.run(
        "Run a 1-way dimensional sweep on Gender, Smoker, Issue_Age_Band to calculate the A/E ratio by Count and Amount."
    )

    assert "confirm which columns" not in response.lower()
    assert captured["depth"] == 1
    assert captured["selected_columns"] == ["Gender", "Smoker", "Issue_Age_band"]
    assert agent.latest_output_path == "data/output/sweep_summary_1_gender_smoker_issue_age_band.csv"


def test_actuary_agent_returns_missing_column_guidance_for_unprepared_feature(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.csv"
    analysis_path.write_text(FIXTURE_PATH.read_text())

    monkeypatch.chdir(tmp_path)
    agent = ActuaryAgent()

    response = agent.run(
        "Run a 1-way dimensional sweep on Gender, Smoker, Face_Amount_Band to calculate the A/E ratio by Count and Amount."
    )

    assert "requested column(s) not found" in response
    assert "Run Data Steward first" in response


def test_analyst_agent_filters_treemap_to_requested_pair(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Smoker=Yes | Risk_Class=Preferred Plus,71.45,2100000,13.07\n"
        "Smoker=Yes | Issue_Age_band=1,69.10,2100000,9.43\n"
        "Risk_Class=Standard Plus | Issue_Age_band=3,744.03,5600000,2.94\n"
    )

    captured = {}

    def fake_generate_treemap_report(data_path: str, metric: str = "amount") -> str:
        captured["data_path"] = data_path
        captured["metric"] = metric
        captured["dimensions"] = pd.read_csv(data_path)["Dimensions"].tolist()
        return f"Treemap report generated and opened: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_treemap_report", fake_generate_treemap_report)

    agent = AnalystAgent()
    response = agent.run(
        "Generate a treemap of the 2-way sweep on Risk Class and Smoker.",
        data_path=str(sweep_path),
    )

    assert response.startswith("Treemap report generated and opened:")
    assert captured["metric"] == "amount"
    assert captured["dimensions"] == ["Smoker=Yes | Risk_Class=Preferred Plus"]


def test_analyst_agent_rejects_unavailable_treemap_pair(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Smoker=Yes | Risk_Class=Preferred Plus,71.45,2100000,13.07\n"
        "Smoker=Yes | Issue_Age_band=1,69.10,2100000,9.43\n"
    )

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    response = agent.run(
        "Generate a treemap of the 2-way sweep on Gender and Smoker.",
        data_path=str(sweep_path),
    )

    assert "does not contain the requested pair" in response
    assert "gender + smoker" in response.lower()
