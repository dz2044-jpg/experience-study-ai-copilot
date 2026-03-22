import os
from pathlib import Path

from agents.orchestrator import StudyOrchestrator


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
        orchestrator.actuary.latest_output_alias_path = str(artifact_path)
        orchestrator.actuary.latest_depth_output_path = str(artifact_path)
        orchestrator.actuary.latest_sweep_depth = 1
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


def test_combined_analysis_and_visualization_prompt_auto_chains(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "sweep_summary_latest_2.csv"
    artifact_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")

    def fake_actuary_run(message):
        calls.append(("actuary", message))
        orchestrator.actuary.latest_output_alias_path = str(output_dir / "sweep_summary.csv")
        orchestrator.actuary.latest_depth_output_path = str(artifact_path)
        orchestrator.actuary.latest_sweep_depth = 2
        return "actuary-ok"

    def fake_analyst_run(message, data_path=None):
        calls.append(("analyst", message, data_path))
        return "Visualization report generated: /tmp/report.html"

    monkeypatch.setattr(orchestrator.actuary, "run", fake_actuary_run)
    monkeypatch.setattr(orchestrator.analyst_agent, "run", fake_analyst_run)

    result = orchestrator.process_query(
        "Run a 2-way dimensional sweep on Gender and Smoker, then generate a visualization."
    )

    assert result == "actuary-ok\n\nVisualization\nVisualization report generated: /tmp/report.html"
    assert calls == [
        ("actuary", "Run a 2-way dimensional sweep on Gender and Smoker, then generate a visualization."),
        ("analyst", "Generate a visualization for the latest sweep.", str(artifact_path)),
    ]
    assert orchestrator.pending_visualization_prompt is None


def test_combined_analysis_and_visualization_prompt_stops_when_no_fresh_artifact(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    calls = []

    def fake_actuary_run(message):
        calls.append(("actuary", message))
        orchestrator.actuary.latest_output_alias_path = None
        orchestrator.actuary.latest_depth_output_path = None
        orchestrator.actuary.latest_sweep_depth = None
        return "actuary-only"

    monkeypatch.setattr(orchestrator.actuary, "run", fake_actuary_run)
    monkeypatch.setattr(
        orchestrator.analyst_agent,
        "run",
        lambda message, data_path=None: calls.append(("analyst", message, data_path)) or "analyst-ok",
    )

    result = orchestrator.process_query(
        "Run a 1-way dimensional sweep on Gender, then generate a visualization."
    )

    assert result == "actuary-only"
    assert calls == [("actuary", "Run a 1-way dimensional sweep on Gender, then generate a visualization.")]


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


def test_data_prep_analysis_visualization_prompt_still_stops_at_steward(monkeypatch):
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
    monkeypatch.setattr(
        orchestrator.analyst_agent,
        "run",
        lambda message, data_path=None: calls.append(("analyst", message, data_path)) or "analyst-ok",
    )

    result = orchestrator.process_query(
        "Check the data, run a 1-way sweep on Gender, then generate a visualization."
    )

    assert result == "steward-ok"
    assert calls == [
        ("steward", "Check the data, run a 1-way sweep on Gender, then generate a visualization.")
    ]


def test_visualization_requires_fresh_analysis_artifact(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()

    result = orchestrator.process_query("Generate a treemap of the 2-way sweep we just ran")

    assert "fresh 2-way sweep artifact" in result


def test_orchestrator_uses_one_way_alias_for_single_feature_visualization(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    generic_path = output_dir / "sweep_summary.csv"
    generic_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")
    one_way_path = output_dir / "sweep_summary_latest_1.csv"
    one_way_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F,1.1\n")
    two_way_path = output_dir / "sweep_summary_latest_2.csv"
    two_way_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")

    orchestrator.latest_analysis_output_path = str(generic_path)
    orchestrator.latest_analysis_output_paths_by_depth = {
        1: str(one_way_path),
        2: str(two_way_path),
    }

    calls = []

    def fake_analyst_run(message, data_path=None):
        calls.append((message, data_path))
        return "analyst-ok"

    monkeypatch.setattr(orchestrator.analyst_agent, "run", fake_analyst_run)

    result = orchestrator.process_query("Generate visualization for Gender only.")

    assert result == "analyst-ok"
    assert calls == [("Generate visualization for Gender only.", str(one_way_path))]


def test_orchestrator_returns_controlled_error_when_requested_depth_alias_is_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    orchestrator = StudyOrchestrator()
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    generic_path = output_dir / "sweep_summary.csv"
    generic_path.write_text("Dimensions,AE_Ratio_Amount\nGender=F | Smoker=Yes,1.3\n")
    orchestrator.latest_analysis_output_path = str(generic_path)
    orchestrator.latest_analysis_output_paths_by_depth = {}

    result = orchestrator.process_query("Generate visualization for Gender only.")

    assert "fresh 1-way sweep artifact" in result
