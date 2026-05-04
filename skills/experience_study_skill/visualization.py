"""Visualization report tools for the experience study skill."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs

from skills.experience_study_skill.io import (
    ToolExecutionContext,
    _ensure_output_dir,
    _error_result,
    _tool_result,
)


_NEUTRAL_COLOR = "#1f4e79"
_FIGURE_CONFIG = {"displaylogo": False, "responsive": True}
_SCATTER_X_MAX = 3.0
_TREEMAP_COLOR_MAX = 2.0


def _validate_metric(metric: str) -> None:
    if metric not in {"count", "amount"}:
        raise ValueError("metric must be 'count' or 'amount'")


def _metric_columns(metric: str) -> dict[str, str]:
    _validate_metric(metric)
    if metric == "count":
        return {
            "ratio": "AE_Ratio_Count",
            "actual": "Sum_MAC",
            "expected": "Sum_MEC",
            "ci_low": "AE_Count_CI_Lower",
            "ci_high": "AE_Count_CI_Upper",
        }
    return {
        "ratio": "AE_Ratio_Amount",
        "actual": "Sum_MAF",
        "expected": "Sum_MEF",
        "ci_low": "AE_Amount_CI_Lower",
        "ci_high": "AE_Amount_CI_Upper",
    }


def _metric_label(metric: str) -> str:
    return "Count" if metric == "count" else "Amount"


def _ratio_label(metric: str) -> str:
    return f"A/E Ratio ({_metric_label(metric)})"


def _required_columns(df: pd.DataFrame, columns: Sequence[str], data_path: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")


def _build_scatter_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    required = [
        "Dimensions",
        "Sum_MOC",
        metric_columns["actual"],
        metric_columns["expected"],
        metric_columns["ratio"],
        metric_columns["ci_low"],
        metric_columns["ci_high"],
    ]
    _required_columns(df, required, data_path)
    prepared = df.copy().sort_values(metric_columns["ratio"], ascending=False)
    prepared["display_ratio"] = prepared[metric_columns["ratio"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_low"] = prepared[metric_columns["ci_low"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_high"] = prepared[metric_columns["ci_high"]].clip(lower=0, upper=_SCATTER_X_MAX)

    fig = go.Figure(
        go.Scatter(
            x=prepared["display_ratio"],
            y=prepared["Dimensions"],
            mode="markers",
            marker={
                "color": _NEUTRAL_COLOR,
                "size": 14,
                "line": {"color": "#ffffff", "width": 1.2},
                "opacity": 0.92,
            },
            error_x={
                "type": "data",
                "symmetric": False,
                "array": (prepared["display_ci_high"] - prepared["display_ratio"]).clip(lower=0),
                "arrayminus": (prepared["display_ratio"] - prepared["display_ci_low"]).clip(lower=0),
                "visible": True,
                "color": _NEUTRAL_COLOR,
                "thickness": 1.4,
            },
            hovertemplate=(
                "<b>%{y}</b><br>"
                f"{_ratio_label(metric)}: "
                "%{x:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=1.0, line_dash="dash", line_color="#c05252", line_width=2)
    fig.update_xaxes(title_text=_ratio_label(metric), range=[0, _SCATTER_X_MAX])
    fig.update_yaxes(title_text="Cohort", autorange="reversed")
    fig.update_layout(
        title=f"Forest Plot ({_metric_label(metric)})",
        height=max(460, min(320 + max(len(prepared), 1) * 54, 920)),
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#ffffff",
        font={"color": "#1f2933", "family": "Avenir Next, Segoe UI, Arial, sans-serif"},
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
        showlegend=False,
    )
    return fig


def _build_table_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    required = [
        "Dimensions",
        "Sum_MOC",
        metric_columns["actual"],
        metric_columns["expected"],
        metric_columns["ratio"],
        metric_columns["ci_low"],
        metric_columns["ci_high"],
    ]
    _required_columns(df, required, data_path)
    prepared = df.copy().sort_values(metric_columns["ratio"], ascending=False)
    fig = go.Figure(
        go.Table(
            header={
                "values": [
                    "Cohort",
                    "Exposure (MOC)",
                    "Actual",
                    "Expected",
                    _ratio_label(metric),
                    "95% CI",
                ],
                "fill_color": "#e7eef5",
                "align": "left",
                "font": {"color": "#102a43", "size": 12},
            },
            cells={
                "values": [
                    prepared["Dimensions"],
                    prepared["Sum_MOC"].map(lambda value: f"{float(value):,.2f}"),
                    prepared[metric_columns["actual"]].map(lambda value: f"{float(value):,.2f}"),
                    prepared[metric_columns["expected"]].map(lambda value: f"{float(value):,.2f}"),
                    prepared[metric_columns["ratio"]].map(lambda value: f"{float(value):.2f}"),
                    prepared[metric_columns["ci_low"]].map(lambda value: f"{float(value):.1%}")
                    + " - "
                    + prepared[metric_columns["ci_high"]].map(lambda value: f"{float(value):.1%}"),
                ],
                "align": "left",
                "fill_color": "#ffffff",
                "height": 30,
            },
        )
    )
    fig.update_layout(
        title=f"Filtered Cohort Detail ({_metric_label(metric)})",
        height=max(260, min(140 + max(len(prepared), 1) * 32, 1800)),
        paper_bgcolor="#fbfaf7",
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
    )
    return fig


def _split_dimensions(label: str) -> list[str]:
    return [part.strip() for part in str(label).split("|") if part.strip()]


def _build_treemap_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    metric_columns = _metric_columns(metric)
    value_column = "Sum_MOC" if metric == "count" else "Sum_MAF"
    required = ["Dimensions", metric_columns["ratio"], value_column]
    _required_columns(df, required, data_path)

    parents = []
    labels = []
    values = []
    colors = []
    for _, row in df.iterrows():
        parts = _split_dimensions(row["Dimensions"])
        labels.append(parts[-1] if parts else row["Dimensions"])
        parents.append(" / ".join(parts[:-1]) if len(parts) > 1 else "")
        values.append(float(row[value_column]))
        colors.append(float(row[metric_columns["ratio"]]))

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker={
                "colors": colors,
                "colorscale": "RdYlGn_r",
                "cmid": 1.0,
                "cmin": 0.0,
                "cmax": _TREEMAP_COLOR_MAX,
                "line": {"width": 1, "color": "#ffffff"},
            },
            hovertemplate=(
                "<b>%{label}</b><br>"
                f"{_ratio_label(metric)}: "
                "%{color:.2f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Risk Treemap ({_metric_label(metric)})",
        height=900,
        paper_bgcolor="#fbfaf7",
        margin={"l": 40, "r": 30, "t": 70, "b": 30},
    )
    return fig


def _figure_fragment(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_FIGURE_CONFIG)


def _build_report_html(
    *,
    title: str,
    metric: str,
    scatter_fragment: str,
    table_fragment: str,
    treemap_fragment: str,
) -> str:
    plotly_js = get_plotlyjs()
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
  <style>
    body {{
      margin: 0;
      padding: 32px 24px 48px;
      background:
        radial-gradient(circle at top left, rgba(31, 78, 121, 0.08), transparent 28%),
        linear-gradient(180deg, #faf7f2 0%, #f6f3ed 100%);
      color: #1f2933;
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
    }}
    .report-shell {{
      max-width: 1280px;
      margin: 0 auto;
    }}
    .report-section {{
      margin-top: 26px;
      padding: 22px 22px 18px;
      border: 1px solid rgba(31, 41, 51, 0.12);
      border-radius: 24px;
      background: #ffffff;
      box-shadow: 0 18px 50px rgba(31, 41, 51, 0.08);
    }}
  </style>
  <script type="text/javascript">{plotly_js}</script>
</head>
<body>
  <main class="report-shell">
    <section class="report-section">{scatter_fragment}</section>
    <section class="report-section">{table_fragment}</section>
    <section class="report-section">{treemap_fragment}</section>
  </main>
</body>
</html>
"""


def generate_combined_report(
    *,
    context: ToolExecutionContext,
    metric: str = "amount",
    data_path: str | None = None,
) -> dict[str, Any]:
    source_path = Path(data_path) if data_path else context.latest_sweep_path
    if source_path is None or not source_path.exists():
        return _error_result(
            "missing_prerequisite",
            "Missing prerequisite. Run dimensional sweep first.",
        )

    context.emit_status("Generating the combined visualization report.")
    df = pd.read_csv(source_path)
    scatter_fragment = _figure_fragment(_build_scatter_figure(df, metric, str(source_path)))
    table_fragment = _figure_fragment(_build_table_figure(df, metric, str(source_path)))
    treemap_fragment = _figure_fragment(_build_treemap_figure(df, metric, str(source_path)))
    report_html = _build_report_html(
        title=f"Combined A/E Visualization Report ({_metric_label(metric)})",
        metric=metric,
        scatter_fragment=scatter_fragment,
        table_fragment=table_fragment,
        treemap_fragment=treemap_fragment,
    )

    _ensure_output_dir(context)
    visualization_path = context.next_visualization_path()
    visualization_path.write_text(report_html, encoding="utf-8")
    return _tool_result(
        True,
        "visualization",
        f"Generated the combined visualization report at `{visualization_path}`.",
        artifacts={
            "sweep_summary_path": str(source_path),
            "visualization_path": str(visualization_path),
        },
        data={"metric": metric},
    )
