import os
from pathlib import Path

from agents.orchestrator import StudyOrchestrator
from tests.test_agent_contracts import FIXTURE_PATH


def test_continue_after_data_prep_routes_to_pending_analysis(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []

    monkeypatch.setattr(
        orchestrator.data_steward,
        "run",
        lambda message: calls.append(("steward", message)) or "steward-ok",
    )
    monkeypatch.setattr(
        orchestrator.actuary,
        "run",
        lambda message: calls.append(("actuary", message)) or "actuary-ok",
    )

    first = orchestrator.process_query("profile the inforce dataset")
    second = orchestrator.process_query("continue")

    assert first == "steward-ok"
    assert second == "actuary-ok"
    assert calls[0] == ("steward", "profile the inforce dataset")
    assert calls[1][0] == "actuary"
    assert "1-way dimensional sweep" in calls[1][1]


def test_continue_after_analysis_routes_to_pending_visualization(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []
    artifact_path = Path("data/output/test_sweep_summary.csv")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("Dimensions,AE_Ratio_Amount\nGender=M,1.5\n")

    def fake_actuary_run(message):
        calls.append(("actuary", message))
        orchestrator.actuary.latest_output_path = str(artifact_path)
        return "actuary-ok"

    def fake_analyst_run(message, data_path=None):
        calls.append(("analyst", message, data_path))
        return "analyst-ok"

    monkeypatch.setattr(orchestrator.actuary, "run", fake_actuary_run)
    monkeypatch.setattr(orchestrator.analyst_agent, "run", fake_analyst_run)

    first = orchestrator.process_query("run a 1-way sweep")
    second = orchestrator.process_query("proceed")

    assert first == "actuary-ok"
    assert second == "analyst-ok"
    assert calls[0] == ("actuary", "run a 1-way sweep")
    assert calls[1][0] == "analyst"
    assert "Create a scatter report using the amount metric from this sweep summary CSV" in calls[1][1]
    assert calls[1][2] == str(artifact_path)


def test_engineered_band_analysis_request_routes_to_actuary(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []

    monkeypatch.setattr(
        orchestrator.actuary,
        "run",
        lambda message: calls.append(("actuary", message)) or "actuary-ok",
    )
    monkeypatch.setattr(
        orchestrator.data_steward,
        "run",
        lambda message: calls.append(("steward", message)) or "steward-ok",
    )

    result = orchestrator.process_query(
        "Run 2-way dimensional sweeps for all pairs between Smoker, Risk_Class, and Issue_Age_band, "
        "then rank the results by AE_Ratio_Amount."
    )

    assert result == "actuary-ok"
    assert calls == [
        (
            "actuary",
            "Run 2-way dimensional sweeps for all pairs between Smoker, Risk_Class, and Issue_Age_band, "
            "then rank the results by AE_Ratio_Amount.",
        )
    ]


def test_visualization_requires_fresh_analysis_artifact(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()

    result = orchestrator.process_query("Generate a treemap of the 2-way sweep we just ran")

    assert "fresh sweep artifact" in result


def test_missing_treemap_pair_requests_confirmation_then_runs_official_flow(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.csv"
    analysis_path.write_text(FIXTURE_PATH.read_text())
    sweep_path = output_dir / "sweep_summary_2_smoker_risk_class_issue_age_band.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Smoker=Yes | Risk_Class=Preferred,10,1000,1.2\n"
        "Smoker=No | Issue_Age_band=25_45,12,900,0.9\n"
    )

    official_path = output_dir / "sweep_summary_2_gender_smoker.csv"
    official_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Gender=M | Smoker=Yes,15,1200,1.8\n"
    )

    orchestrator = StudyOrchestrator()
    orchestrator.latest_analysis_output_path = str(sweep_path)
    calls = []

    def fake_actuary_run(message):
        calls.append(("actuary", message))
        orchestrator.actuary.latest_output_path = str(official_path)
        return "actuary-official"

    def fake_analyst_run(message, data_path=None):
        calls.append(("analyst", message, data_path))
        return f"analyst-treemap:{data_path}"

    monkeypatch.setattr(orchestrator.actuary, "run", fake_actuary_run)
    monkeypatch.setattr(orchestrator.analyst_agent, "run", fake_analyst_run)

    first = orchestrator.process_query("Generate a treemap of the 2-way sweep on Gender and Smoker.")
    second = orchestrator.process_query("continue")

    assert "reply `continue`" in first.lower()
    assert "official 2-way sweep" in first.lower()
    assert calls[0][0] == "actuary"
    assert "Run a 2-way dimensional sweep on Gender and Smoker" in calls[0][1]
    assert "return all cohorts" in calls[0][1]
    assert calls[1] == ("analyst", "Generate a treemap of the 2-way sweep on Gender and Smoker.", str(official_path))
    assert "actuary-official" in second
    assert f"analyst-treemap:{official_path}" in second
    assert orchestrator.latest_analysis_output_path == str(official_path)


def test_missing_treemap_pair_with_unprepared_columns_returns_steward_guidance(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = output_dir / "analysis_inforce.csv"
    analysis_path.write_text(FIXTURE_PATH.read_text())
    sweep_path = output_dir / "sweep_summary_2_smoker_risk_class.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,AE_Ratio_Amount\n"
        "Smoker=Yes | Risk_Class=Preferred,10,1000,1.2\n"
    )

    orchestrator = StudyOrchestrator()
    orchestrator.latest_analysis_output_path = str(sweep_path)

    result = orchestrator.process_query(
        "Generate a treemap of the 2-way sweep on Face_Amount_Band and Smoker."
    )

    assert "Run Data Steward first" in result
