import json
from pathlib import Path

import pandas as pd

import agents.agent_actuary as agent_actuary_module
import agents.agent_analyst as agent_analyst_module
import agents.agent_steward as agent_steward_module
from agents.agent_actuary import ActuaryAgent
from agents.agent_analyst import AnalystAgent
from agents.agent_steward import DataStewardAgent
from agents.schemas import DimensionalSweepSchema, VisualizationSchema
from tools import data_steward


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "dummy_inforce.csv"


def _write_analysis_parquet(path: Path) -> None:
    pd.read_csv(FIXTURE_PATH).to_parquet(path, index=False)


def test_dimensional_sweep_schema_exposes_selected_columns():
    schema = DimensionalSweepSchema.model_json_schema()
    assert "selected_columns" in schema["properties"]


def test_dimensional_sweep_schema_exposes_structured_filters():
    schema = DimensionalSweepSchema.model_json_schema()
    filters_schema = schema["properties"]["filters"]
    assert filters_schema["items"]["$ref"].endswith("FilterClauseSchema")


def test_visualization_schema_omits_chart_type_and_exposes_core_fields():
    schema = VisualizationSchema.model_json_schema()

    assert set(schema["properties"]) == {"metric", "data_path"}
    assert "chart_type" not in schema["properties"]


def test_analyst_agent_exposes_single_combined_report_tool():
    agent = AnalystAgent()

    assert [tool["function"]["name"] for tool in agent._tools_spec()] == ["generate_combined_report"]


def test_steward_agent_can_execute_regroup_tool(tmp_path, monkeypatch):
    source = tmp_path / "analysis_inforce.parquet"
    _write_analysis_parquet(source)

    output = tmp_path / "analysis_output.parquet"
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

    df = pd.read_parquet(saved_path)
    assert "Risk_Class_regrouped" in df.columns


def test_steward_agent_schema_request_resolves_bare_analysis_filename_and_lists_engineered_columns(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    engineered = pd.read_parquet(analysis_path)
    engineered["Issue_Age_band"] = engineered["Issue_Age"].apply(lambda value: "older" if value >= 45 else "younger")
    engineered["Face_Amount_band"] = engineered["Face_Amount"].apply(lambda value: "large" if value >= 500000 else "small")
    engineered.to_parquet(analysis_path, index=False)

    monkeypatch.chdir(tmp_path)

    agent = DataStewardAgent()
    response = agent.run("what are the columns in analysis_inforce.parquet")

    assert "data/output/analysis_inforce.parquet" in response
    assert "Issue_Age_band" in response
    assert "Face_Amount_band" in response


def test_steward_agent_profile_request_for_prepared_dataset_lists_engineered_columns(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    engineered = pd.read_parquet(analysis_path)
    engineered["Issue_Age_band"] = engineered["Issue_Age"].apply(lambda value: "older" if value >= 45 else "younger")
    engineered["Face_Amount_band"] = engineered["Face_Amount"].apply(lambda value: "large" if value >= 500000 else "small")
    engineered.to_parquet(analysis_path, index=False)

    monkeypatch.chdir(tmp_path)

    agent = DataStewardAgent()
    response = agent.run("profile data/output/analysis_inforce.parquet")

    assert "Columns, data types, and null counts for `data/output/analysis_inforce.parquet`:" in response
    assert "Issue_Age_band" in response
    assert "Face_Amount_band" in response


def test_steward_agent_raw_schema_request_defaults_to_synthetic_input(monkeypatch):
    captured = {}

    def fake_profile_dataset(data_path: str = "", sheet_name=None) -> str:
        captured["data_path"] = data_path
        return json.dumps(
            {
                "columns": ["MAC", "MEC", "MAF", "MEF"],
                "data_types": {"MAC": "float64", "MEC": "float64", "MAF": "float64", "MEF": "float64"},
                "null_counts": {"MAC": 0, "MEC": 0, "MAF": 0, "MEF": 0},
            }
        )

    monkeypatch.setattr(agent_steward_module, "profile_dataset", fake_profile_dataset)

    agent = DataStewardAgent()
    response = agent.run("Profile the raw synthetic inforce data and list the data types for MAC, MEC, MAF, and MEF.")

    assert captured["data_path"] == "data/input/synthetic_inforce.csv"
    assert "Requested columns for `data/input/synthetic_inforce.csv`:" in response
    assert "- `MAC`: `float64`; nulls: `0`" in response
    assert "- `MEF`: `float64`; nulls: `0`" in response
    assert "Gender" not in response


def test_actuary_agent_runs_explicit_sweep_without_mapping_confirmation(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

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
                "depth": 1,
                "output_path": "data/output/sweep_summary_1_gender_smoker_issue_age_band.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
                "latest_depth_output_path": "data/output/sweep_summary_latest_1.csv",
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
    assert captured["min_mac"] == 0
    assert captured["selected_columns"] == ["Gender", "Smoker", "Issue_Age_band"]
    assert agent.latest_output_path == "data/output/sweep_summary_1_gender_smoker_issue_age_band.csv"
    assert agent.latest_output_alias_path == "data/output/sweep_summary.csv"
    assert agent.latest_depth_output_path == "data/output/sweep_summary_latest_1.csv"
    assert agent.latest_output_paths_by_depth == {1: "data/output/sweep_summary_latest_1.csv"}


def test_actuary_agent_ignores_trailing_calculation_phrase_when_extracting_columns(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

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
                "depth": 1,
                "output_path": "data/output/sweep_summary_1_gender_smoker_risk_class.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
                "latest_depth_output_path": "data/output/sweep_summary_latest_1.csv",
            }
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", fake_run_dimensional_sweep)

    agent = ActuaryAgent()
    response = agent.run(
        "Run a 1-way dimensional sweep on Gender, Smoker and Risk Class and calculate the A/E ratio by Count and Amount."
    )

    assert "requested column(s) not found" not in response
    assert captured["selected_columns"] == ["Gender", "Smoker", "Risk_Class"]


def test_actuary_agent_includes_ranked_table_and_top_ranked_summary(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    captured = {}

    def fake_run_dimensional_sweep(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "results": [
                    {
                        "Dimensions": "Smoker=Yes | Gender=F",
                        "Sum_MAC": 9,
                        "AE_Ratio_Count": 2.4,
                        "AE_Ratio_Amount": 1.8,
                        "AE_Count_CI": [1.1, 3.6],
                        "AE_Amount_CI": [0.9, 2.7],
                    },
                    {
                        "Dimensions": "Smoker=No | Gender=F",
                        "Sum_MAC": 7,
                        "AE_Ratio_Count": 2.1,
                        "AE_Ratio_Amount": 1.6,
                        "AE_Count_CI": [1.0, 3.1],
                        "AE_Amount_CI": [0.8, 2.3],
                    },
                    {
                        "Dimensions": "Smoker=Yes | Gender=M",
                        "Sum_MAC": 5,
                        "AE_Ratio_Count": 1.9,
                        "AE_Ratio_Amount": 1.5,
                        "AE_Count_CI": [0.9, 2.8],
                        "AE_Amount_CI": [0.7, 2.2],
                    },
                ],
                "depth": 1,
                "output_path": "data/output/sweep_summary_1_gender_smoker.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
                "latest_depth_output_path": "data/output/sweep_summary_latest_1.csv",
            }
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", fake_run_dimensional_sweep)

    agent = ActuaryAgent()
    response = agent.run(
        "Run a 1-way dimensional sweep on Gender and Smoker, rank cohorts by AE_Ratio_Count, and show the top 3."
    )

    assert captured["sort_by"] == "AE_Ratio_Count"
    assert captured["top_n"] == 3
    assert "| Rank | Cohort | AE_Ratio_Amount | AE_Ratio_Count | Sum_MAC |" in response
    assert "| 1 | Smoker=Yes \\| Gender=F | 1.80 | 2.40 | 9 |" in response
    assert "| 3 | Smoker=Yes \\| Gender=M | 1.50 | 1.90 | 5 |" in response
    assert "- Top-ranked cohort by AE_Ratio_Count: `Smoker=Yes | Gender=F`" in response
    assert "- Financial risk (AE_Ratio_Amount): 1.80" in response
    assert response.index("| Rank | Cohort | AE_Ratio_Amount | AE_Ratio_Count | Sum_MAC |") < response.index(
        "- Top-ranked cohort by AE_Ratio_Count: `Smoker=Yes | Gender=F`"
    )


def test_actuary_agent_supports_sum_mac_ranking(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    captured = {}

    def fake_run_dimensional_sweep(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "results": [
                    {
                        "Dimensions": "Gender=F",
                        "Sum_MAC": 11,
                        "AE_Ratio_Count": 1.4,
                        "AE_Ratio_Amount": 1.2,
                        "AE_Count_CI": [0.9, 2.0],
                        "AE_Amount_CI": [0.8, 1.9],
                    }
                ],
                "depth": 1,
                "output_path": "data/output/sweep_summary_1_gender.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
                "latest_depth_output_path": "data/output/sweep_summary_latest_1.csv",
            }
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", fake_run_dimensional_sweep)

    agent = ActuaryAgent()
    response = agent.run("Run a 1-way dimensional sweep on Gender and rank cohorts by Sum_MAC.")

    assert captured["sort_by"] == "Sum_MAC"
    assert "- Top-ranked cohort by Sum_MAC: `Gender=F`" in response


def test_actuary_agent_recognizes_two_dimensional_wording(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    captured = {}

    def fake_run_dimensional_sweep(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "results": [
                    {
                        "Dimensions": "Risk_Class=Standard | Gender=F",
                        "Sum_MAC": 6,
                        "AE_Ratio_Count": 1.7,
                        "AE_Ratio_Amount": 1.5,
                        "AE_Count_CI": [0.9, 2.5],
                        "AE_Amount_CI": [0.8, 2.2],
                    }
                ],
                "depth": 2,
                "output_path": "data/output/sweep_summary_2_risk_class_gender_smoker.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
                "latest_depth_output_path": "data/output/sweep_summary_latest_2.csv",
            }
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", fake_run_dimensional_sweep)

    agent = ActuaryAgent()
    response = agent.run("run a 2-dimensional sweep on between Risk class, gender, smoker")

    assert captured["depth"] == 2
    assert captured["selected_columns"] == ["Risk_Class", "Gender", "Smoker"]
    assert response.startswith("2-way dimensional sweep complete on the prepared analysis dataset")


def test_actuary_agent_tracks_depth_specific_aliases_across_runs(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    payloads = iter(
        [
            json.dumps(
                {
                    "results": [{"Dimensions": "Gender=F", "AE_Ratio_Count": 1.2, "AE_Ratio_Amount": 1.1, "AE_Count_CI": [0.8, 1.7], "AE_Amount_CI": [0.7, 1.6]}],
                    "depth": 1,
                    "output_path": "data/output/sweep_summary_1_gender.csv",
                    "latest_output_path": "data/output/sweep_summary.csv",
                    "latest_depth_output_path": "data/output/sweep_summary_latest_1.csv",
                }
            ),
            json.dumps(
                {
                    "results": [{"Dimensions": "Gender=F | Smoker=Yes", "AE_Ratio_Count": 1.5, "AE_Ratio_Amount": 1.4, "AE_Count_CI": [0.9, 2.0], "AE_Amount_CI": [0.8, 1.9]}],
                    "depth": 2,
                    "output_path": "data/output/sweep_summary_2_gender_smoker.csv",
                    "latest_output_path": "data/output/sweep_summary.csv",
                    "latest_depth_output_path": "data/output/sweep_summary_latest_2.csv",
                }
            ),
        ]
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", lambda **kwargs: next(payloads))

    agent = ActuaryAgent()
    agent.run("Run a 1-way dimensional sweep on Gender.")
    agent.run("Run a 2-way dimensional sweep on Gender and Smoker.")

    assert agent.latest_output_paths_by_depth == {
        1: "data/output/sweep_summary_latest_1.csv",
        2: "data/output/sweep_summary_latest_2.csv",
    }


def test_actuary_agent_returns_missing_column_guidance_for_unprepared_feature(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)
    agent = ActuaryAgent()

    response = agent.run(
        "Run a 1-way dimensional sweep on Gender, Smoker, Face_Amount_Band to calculate the A/E ratio by Count and Amount."
    )

    assert "requested column(s) not found" in response
    assert "Run Data Steward first" in response


def test_actuary_agent_extracts_structured_filters_deterministically(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    captured = {}

    def fake_run_dimensional_sweep(**kwargs):
        captured.update(kwargs)
        return json.dumps(
            {
                "results": [
                    {
                        "Dimensions": "Gender=F",
                        "Sum_MAC": 2,
                        "AE_Ratio_Count": 1.1,
                        "AE_Ratio_Amount": 1.0,
                        "AE_Count_CI": [0.7, 1.6],
                        "AE_Amount_CI": [0.6, 1.5],
                    }
                ],
                "depth": 1,
                "output_path": "data/output/sweep_summary_1_gender.csv",
                "latest_output_path": "data/output/sweep_summary.csv",
                "latest_depth_output_path": "data/output/sweep_summary_latest_1.csv",
            }
        )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_actuary_module, "run_dimensional_sweep", fake_run_dimensional_sweep)

    agent = ActuaryAgent()
    response = agent.run("Run a 1-way dimensional sweep on Gender where Duration < 5 and Smoker = Yes.")

    assert "with filters Duration < 5, Smoker = Yes" in response
    assert captured["filters"] == [
        {"column": "Duration", "operator": "<", "value": 5},
        {"column": "Smoker", "operator": "=", "value": "Yes"},
    ]


def test_actuary_agent_surfaces_graceful_invalid_filter_column_error(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        agent_actuary_module,
        "run_dimensional_sweep",
        lambda **kwargs: json.dumps(
            {
                "error": "Column 'State' not found.",
                "available_columns": ["Duration", "Gender", "Issue_Age", "Smoker"],
                "suggested_columns": ["Issue_Age", "Duration"],
            }
        ),
    )

    agent = ActuaryAgent()
    response = agent.run("Run a 1-way dimensional sweep on Gender where State = California.")

    assert "Column 'State' not found." in response
    assert "Try one of these instead: `Issue_Age`, `Duration`." in response


def test_actuary_agent_uses_llm_route_when_filter_phrase_is_ambiguous(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.parquet"
    _write_analysis_parquet(analysis_path)

    monkeypatch.chdir(tmp_path)
    agent = ActuaryAgent()
    agent.client = object()

    assert agent._deterministic_sweep_route(
        "Show me the sweep for Smoker, but only for people from California."
    ) is None


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

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        captured["data_path"] = data_path
        captured["metric"] = metric
        captured["dimensions"] = pd.read_csv(data_path)["Dimensions"].tolist()
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run(
        "Generate a treemap of the 2-way sweep on Risk Class and Smoker.",
        data_path=str(sweep_path),
    )

    assert response.startswith("Visualization report generated:")
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


def test_analyst_agent_uses_combined_report_for_one_way_visualize_request(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=F,150,135,150,0.9,0.7,1.1\n"
    )

    calls = []

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        calls.append((data_path, metric))
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Visualize this sweep for me.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert calls == [(str(sweep_path), "amount")]


def test_analyst_agent_uses_latest_sweep_prompt_without_treating_it_as_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary_latest_1.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=F,150,135,150,0.9,0.7,1.1\n"
    )

    calls = []

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        calls.append((data_path, metric))
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Generate a visualization for the latest sweep.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert calls == [(str(sweep_path), "amount")]


def test_analyst_agent_uses_combined_report_for_multiway_visualize_request(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M | Smoker=Yes,100,120,100,1.2,0.9,1.5\n"
        "Gender=F | Smoker=No,150,135,150,0.9,0.7,1.1\n"
    )

    calls = []

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        calls.append((data_path, metric))
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Visualize this sweep for me.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert calls == [(str(sweep_path), "amount")]


def test_analyst_agent_honors_explicit_scatter_request_on_multiway_data(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M | Smoker=Yes,100,120,100,1.2,0.9,1.5\n"
        "Gender=F | Smoker=No,150,135,150,0.9,0.7,1.1\n"
    )

    calls = []

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        calls.append((data_path, metric))
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Create a scatter report for this sweep.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert calls == [(str(sweep_path), "amount")]


def test_analyst_agent_uses_combined_report_for_mixed_depth_visualize_request(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=M | Smoker=Yes,80,96,80,1.2,0.9,1.5\n"
    )

    calls = []

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        calls.append((data_path, metric))
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Visualize this sweep for me.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert calls == [(str(sweep_path), "amount")]


def test_analyst_agent_filters_univariate_to_requested_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Risk_Class=Preferred,80,110,90,1.22,0.9,1.5\n"
        "Risk_Class=Standard,140,150,130,1.15,0.9,1.4\n"
    )

    captured = {}

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        captured["data_path"] = data_path
        captured["metric"] = metric
        captured["dimensions"] = pd.read_csv(data_path)["Dimensions"].tolist()
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Create a univariate report for Risk Class only.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert captured["metric"] == "amount"
    assert captured["dimensions"] == ["Risk_Class=Preferred", "Risk_Class=Standard"]


def test_analyst_agent_rejects_missing_univariate_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=F,150,135,150,0.9,0.7,1.1\n"
    )

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    response = agent.run("Create a univariate report for Risk Class only.", data_path=str(sweep_path))

    assert "does not contain one-way rows for `risk_class`" in response
    assert "available one-way dimensions" in response.lower()


def test_analyst_agent_filters_treemap_to_requested_single_dimension(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Gender=M,100,120,1.2\n"
        "Risk_Class=Preferred,80,110,1.22\n"
        "Risk_Class=Standard,140,150,1.15\n"
        "Gender=M | Smoker=Yes,60,75,1.25\n"
    )

    captured = {}

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        captured["data_path"] = data_path
        captured["metric"] = metric
        captured["dimensions"] = pd.read_csv(data_path)["Dimensions"].tolist()
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Generate a treemap for Risk Class only.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert captured["metric"] == "amount"
    assert captured["dimensions"] == ["Risk_Class=Preferred", "Risk_Class=Standard"]


def test_analyst_agent_filters_treemap_to_requested_single_dimension_without_only(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Gender=M,100,120,1.2\n"
        "Risk_Class=Preferred,80,110,1.22\n"
        "Risk_Class=Standard,140,150,1.15\n"
        "Gender=M | Smoker=Yes,60,75,1.25\n"
    )

    captured = {}

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        captured["data_path"] = data_path
        captured["metric"] = metric
        captured["dimensions"] = pd.read_csv(data_path)["Dimensions"].tolist()
        return f"Visualization report generated: {data_path}"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(agent_analyst_module, "generate_combined_report", fake_generate_combined_report)

    agent = AnalystAgent()
    response = agent.run("Generate a treemap report for Risk Class.", data_path=str(sweep_path))

    assert response.startswith("Visualization report generated:")
    assert captured["metric"] == "amount"
    assert captured["dimensions"] == ["Risk_Class=Preferred", "Risk_Class=Standard"]


def test_analyst_agent_rejects_missing_single_dimension_treemap(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Gender=M,100,120,1.2\n"
        "Gender=F,150,135,0.9\n"
    )

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    response = agent.run("Generate a treemap for Risk Class only.", data_path=str(sweep_path))

    assert "does not contain one-way rows for `risk_class`" in response
    assert "available one-way dimensions" in response.lower()


def test_analyst_agent_resolves_single_feature_request_to_latest_one_way_alias(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    one_way_path = output_dir / "sweep_summary_latest_1.csv"
    one_way_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F,1.1\n")
    generic_path = output_dir / "sweep_summary.csv"
    generic_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    resolved_path, error = agent.resolve_visualization_data_path(
        "Generate visualization for Gender only.",
        generic_latest_path=str(generic_path),
        depth_alias_paths={1: str(one_way_path), 2: str(output_dir / "sweep_summary_latest_2.csv")},
    )

    assert error is None
    assert resolved_path == str(one_way_path)


def test_analyst_agent_resolves_pair_request_to_latest_two_way_alias(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    two_way_path = output_dir / "sweep_summary_latest_2.csv"
    two_way_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")
    generic_path = output_dir / "sweep_summary.csv"
    generic_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F,1.1\n")

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    resolved_path, error = agent.resolve_visualization_data_path(
        "Generate visualization for Risk Class and Smoker.",
        generic_latest_path=str(generic_path),
        depth_alias_paths={2: str(two_way_path)},
    )

    assert error is None
    assert resolved_path == str(two_way_path)


def test_analyst_agent_requires_matching_depth_alias_when_requested(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    generic_path = output_dir / "sweep_summary.csv"
    generic_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    resolved_path, error = agent.resolve_visualization_data_path(
        "Generate visualization for Gender only.",
        generic_latest_path=str(generic_path),
        depth_alias_paths={2: str(output_dir / "sweep_summary_latest_2.csv")},
    )

    assert resolved_path is None
    assert error is not None
    assert "fresh 1-way sweep artifact" in error


def test_analyst_agent_prefers_explicit_two_way_wording_over_three_listed_columns(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    two_way_path = output_dir / "sweep_summary_latest_2.csv"
    two_way_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")
    generic_path = output_dir / "sweep_summary.csv"
    generic_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")

    monkeypatch.chdir(tmp_path)
    agent = AnalystAgent()

    resolved_path, error = agent.resolve_visualization_data_path(
        "Generate visualization for the 2-way sweep between Risk Class, Gender, and Smoker.",
        generic_latest_path=str(generic_path),
        depth_alias_paths={2: str(two_way_path)},
    )

    assert error is None
    assert resolved_path == str(two_way_path)
