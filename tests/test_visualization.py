import inspect

from tools import visualization


def test_visualization_module_does_not_open_browser():
    module_source = inspect.getsource(visualization)

    assert "webbrowser.open" not in module_source


def test_visualization_reports_generate_html_artifacts(tmp_path, monkeypatch):
    output_dir = tmp_path / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = output_dir / "sweep_summary.csv"
    sweep_path.write_text(
        "Dimensions,Sum_MOC,Sum_MAF,Sum_MEF,AE_Ratio_Amount,AE_Amount_CI_Lower,AE_Amount_CI_Upper\n"
        "Smoker=No | Gender=M,100,120,100,1.2,0.9,1.5\n"
        "Gender=M,150,135,150,0.9,0.7,1.1\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    univariate_response = visualization.generate_univariate_report(data_path=str(sweep_path), metric="amount")
    treemap_response = visualization.generate_treemap_report(data_path=str(sweep_path), metric="amount")

    univariate_path = output_dir / "temp_univariate_report.html"
    treemap_path = output_dir / "temp_treemap_report.html"

    assert univariate_response == f"Univariate report generated: {univariate_path.resolve()}"
    assert treemap_response == f"Treemap report generated: {treemap_path.resolve()}"
    assert univariate_path.exists()
    assert treemap_path.exists()
    treemap_html = treemap_path.read_text(encoding="utf-8")
    assert "Smoker=No\\u003cbr\\u003eGender=M" in treemap_html
    assert "A/E" in treemap_html
    assert "middle center" in treemap_html
    assert '"visible":false' in treemap_html
    assert '"mode":"hide"' in treemap_html
    assert "Overall" not in treemap_html
