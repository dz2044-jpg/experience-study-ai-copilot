import inspect

from tools import visualization


def test_visualization_module_does_not_open_browser():
    module_source = inspect.getsource(visualization)

    assert "webbrowser.open" not in module_source


def test_univariate_wrapper_generates_combined_report_with_multiway_labels(tmp_path, monkeypatch):
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

    response = visualization.generate_univariate_report(data_path=str(sweep_path), metric="amount")
    report_path = output_dir / "temp_univariate_report.html"

    assert response == f"Visualization report generated: {report_path.resolve()}"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Combined A/E Visualization Report" in html
    assert "Forest Plot" in html
    assert "Filtered Cohort Detail" in html
    assert "Risk Treemap" in html
    assert "Gender=M | Smoker=Yes" in html
    assert "A\\u002fE Ratio" in html
    assert "95% CI" in html


def test_treemap_wrapper_generates_combined_report_with_flat_one_way_treemap(tmp_path, monkeypatch):
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

    response = visualization.generate_treemap_report(data_path=str(sweep_path), metric="amount")
    report_path = output_dir / "temp_treemap_report.html"

    assert response == f"Visualization report generated: {report_path.resolve()}"
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Combined A/E Visualization Report" in html
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

    response = visualization.generate_treemap_report(data_path=str(sweep_path), metric="amount")
    report_path = output_dir / "temp_treemap_report.html"

    assert response == f"Visualization report generated: {report_path.resolve()}"
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

    visualization.generate_treemap_report(data_path=str(sweep_path), metric="amount")
    report_path = output_dir / "temp_treemap_report.html"
    html = report_path.read_text(encoding="utf-8")

    assert "node::Gender=M" not in html
    assert "leaf::Gender=M | Smoker=Yes" in html
    assert "Overall" not in html
