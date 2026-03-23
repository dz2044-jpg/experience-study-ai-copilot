import inspect
from pathlib import Path

from tools import visualization


def test_visualization_module_does_not_open_browser():
    module_source = inspect.getsource(visualization)

    assert "webbrowser.open" not in module_source


def test_legacy_visualization_wrappers_delegate_to_combined_report(monkeypatch):
    calls = []

    def fake_generate_combined_report(data_path: str, metric: str = "amount") -> str:
        calls.append((data_path, metric))
        return f"Visualization report generated: {data_path}"

    monkeypatch.setattr(visualization, "generate_combined_report", fake_generate_combined_report)

    assert visualization.generate_univariate_report(data_path="first.csv", metric="amount") == "Visualization report generated: first.csv"
    assert visualization.generate_treemap_report(data_path="second.csv", metric="count") == "Visualization report generated: second.csv"
    assert calls == [("first.csv", "amount"), ("second.csv", "count")]


def test_combined_report_generates_with_multiway_labels(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M | Smoker=Yes,120,160,100,1.6,1.2,2.0\n"
        "Gender=F,150,135,150,0.9,0.7,1.1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")
    report_path = Path(response.removeprefix("Visualization report generated: ").strip())

    assert report_path.parent == output_dir.resolve()
    assert report_path.name.startswith("combined_ae_report_")
    assert report_path.suffix == ".html"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Combined A/E Visualization Report (Amount)" in html
    assert "Forest Plot (Amount)" in html
    assert "Filtered Cohort Detail (Amount)" in html
    assert "Risk Treemap (Amount)" in html
    assert "Gender=M | Smoker=Yes" in html
    assert "A\\u002fE Ratio (Amount)" in html
    assert "95% CI" in html
    assert '"range":[0,3.0]' in html
    assert "Displayed x-axis is capped at 3.0 for readability." in html
    assert "Color intensity is capped at A/E 2.0 for readability." in html


def test_combined_report_generates_with_flat_one_way_treemap(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=F,150,135,150,0.9,0.7,1.1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")
    report_path = Path(response.removeprefix("Visualization report generated: ").strip())

    assert report_path.parent == output_dir.resolve()
    assert report_path.name.startswith("combined_ae_report_")
    assert report_path.suffix == ".html"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Combined A/E Visualization Report (Amount)" in html
    assert "A\\u002fE Ratio (Amount)" in html
    assert '"cmin":0.0' in html
    assert '"cmid":1.0' in html
    assert '"cmax":2.0' in html
    assert "Gender=M" in html
    assert "Gender=F" in html
    assert "Overall" not in html
    assert '"parents":["",""]' in html


def test_combined_report_renders_hierarchy_for_two_way_input(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M | Smoker=Yes,100,120,100,1.2,0.9,1.5\n"
        "Gender=M | Smoker=No,90,80,100,0.8,0.6,1.0\n"
        "Gender=F | Smoker=No,150,135,150,0.9,0.7,1.1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")
    report_path = Path(response.removeprefix("Visualization report generated: ").strip())

    assert report_path.parent == output_dir.resolve()
    assert report_path.name.startswith("combined_ae_report_")
    assert report_path.suffix == ".html"
    html = report_path.read_text(encoding="utf-8")
    assert "node::Gender=M" in html
    assert "leaf::Gender=M | Smoker=Yes" in html
    assert "Parent grouping" in html
    assert "Overall" not in html


def test_combined_report_falls_back_to_flat_for_mixed_depth_input(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=M | Smoker=Yes,80,96,80,1.2,0.9,1.5\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")
    report_path = Path(response.removeprefix("Visualization report generated: ").strip())
    html = report_path.read_text(encoding="utf-8")

    assert "node::Gender=M" not in html
    assert "leaf::Gender=M | Smoker=Yes" in html
    assert "Overall" not in html


def test_report_generation_uses_unique_html_paths(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=F,150,135,150,0.9,0.7,1.1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    first_response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")
    second_response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")

    first_path = Path(first_response.removeprefix("Visualization report generated: ").strip())
    second_path = Path(second_response.removeprefix("Visualization report generated: ").strip())

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()


def test_count_metric_is_explicitly_labeled_in_visualization_html(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAC,Sum_MEC,AE_Ratio_Count,AE_Count_CI_Lower,AE_Count_CI_Upper,sample_size\n"
        "Gender=M,100,12,10,1.2,0.9,1.5,100\n"
        "Gender=F,150,9,10,0.9,0.7,1.1,150\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    response = visualization.generate_combined_report(data_path=str(sweep_path), metric="count")
    report_path = Path(response.removeprefix("Visualization report generated: ").strip())
    html = report_path.read_text(encoding="utf-8")

    assert "Combined A/E Visualization Report (Count)" in html
    assert "Forest Plot (Count)" in html
    assert "Filtered Cohort Detail (Count)" in html
    assert "Risk Treemap (Count)" in html
    assert "A\\u002fE Ratio (Count)" in html


def test_scatter_caps_display_at_three_but_preserves_true_hover_values(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Risk_Class=Preferred,100,350,100,3.5,2.8,4.4\n"
        "Risk_Class=Standard,120,110,100,1.1,0.9,1.4\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    response = visualization.generate_combined_report(data_path=str(sweep_path), metric="amount")
    report_path = Path(response.removeprefix("Visualization report generated: ").strip())
    html = report_path.read_text(encoding="utf-8")

    assert '"range":[0,3.0]' in html
    assert '1.1' in html
    assert "3.50" in html
    assert "2.8" in html or "280.0%" in html
    assert "4.4" in html or "440.0%" in html
