"""Standalone plotting engine for combined A/E visualization reports using Plotly."""

import os
from html import escape
from typing import Iterable, Optional
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs


_NEUTRAL_COLOR = "#1f4e79"
_ACCENT_COLORS = [
    "#1f4e79",
    "#d97a2b",
    "#3d7d52",
    "#8b3f70",
    "#6a55a1",
    "#c24848",
]
_MAX_GROUP_COLORS = 6
_FIGURE_CONFIG = {"displaylogo": False, "responsive": True}
_SCATTER_X_MAX = 3.0
_TREEMAP_COLOR_MAX = 2.0


def _validate_metric(metric: str) -> None:
    if metric not in {"amount", "count"}:
        raise ValueError("metric must be 'amount' or 'count'")


def _metric_columns(metric: str) -> dict[str, str]:
    """Resolve the sweep-summary columns needed for the selected metric."""
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
    """Return a display-safe metric label for titles and axes."""
    _validate_metric(metric)
    return "Count" if metric == "count" else "Amount"


def _ratio_label(metric: str) -> str:
    """Return the explicit A/E label shown in visuals."""
    return f"A/E Ratio ({_metric_label(metric)})"


def _treemap_value_spec(df: pd.DataFrame, metric: str) -> tuple[str, str]:
    """Choose the most appropriate value column for treemap box size."""
    _validate_metric(metric)
    candidates = (
        [("sample_size", "Sample Size"), ("Sum_MOC", "Exposure (MOC)")]
        if metric == "count"
        else [("Total_Face_Amount", "Face Amount"), ("Sum_MAF", "Actual Claims")]
    )
    for column, label in candidates:
        if column in df.columns:
            return column, label
    if metric == "count":
        raise ValueError("Treemap count metric requires 'sample_size' or 'Sum_MOC' in the data.")
    raise ValueError("Treemap amount metric requires 'Total_Face_Amount' or 'Sum_MAF' in the data.")


def _required_columns(df: pd.DataFrame, columns: Iterable[str], data_path: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")


def _split_dimensions(label: str) -> list[str]:
    """Split a combined cohort string such as 'Gender=M | Smoker=Yes' into parts."""
    return [part.strip() for part in str(label).split("|") if part.strip()]


def _dimension_depths(labels: Iterable[str]) -> list[int]:
    return [max(1, len(_split_dimensions(label))) for label in labels]


def _format_percent(value: float) -> str:
    return f"{value * 100:0.1f}%"


def _format_ratio(value: float) -> str:
    return f"{value:0.2f}"


def _format_number(value: float) -> str:
    return f"{value:,.2f}"


def _marker_sizes(series: pd.Series, min_size: float = 11, max_size: float = 22) -> list[float]:
    values = series.astype(float)
    if values.empty:
        return []
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value <= min_value:
        midpoint = (min_size + max_size) / 2
        return [midpoint] * len(values)
    scale = (values - min_value) / (max_value - min_value)
    return (min_size + scale * (max_size - min_size)).round(2).tolist()


def _common_layout(fig: go.Figure, title: str, height: int) -> None:
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left"),
        height=height,
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#ffffff",
        font=dict(color="#1f2933", family="Avenir Next, Segoe UI, Arial, sans-serif"),
        margin=dict(l=40, r=30, t=70, b=30),
        hoverlabel=dict(bgcolor="#fffefb", font_size=13, font_family="Avenir Next, Segoe UI, Arial, sans-serif"),
    )


def _scatter_height(cohort_count: int) -> int:
    return max(460, min(320 + max(cohort_count, 1) * 54, 920))


def _table_height(row_count: int) -> int:
    return max(260, min(140 + max(row_count, 1) * 32, 1800))


def _first_dimension_groups(labels: pd.Series) -> Optional[dict[str, str]]:
    """Return a label->group mapping when first-dimension coloring will stay readable."""
    first_parts = labels.astype(str).map(lambda label: _split_dimensions(label)[0] if _split_dimensions(label) else label)
    unique_parts = list(dict.fromkeys(first_parts.tolist()))
    if len(unique_parts) <= 1 or len(unique_parts) > _MAX_GROUP_COLORS:
        return None
    return dict(zip(labels.astype(str), first_parts))


def _customdata_rows(
    df: pd.DataFrame,
    exposure_col: str,
    actual_col: str,
    expected_col: str,
    ratio_col: str,
    ci_low_col: str,
    ci_high_col: str,
) -> list[list[object]]:
    return [
        [
            row["Dimensions"],
            _format_number(row[exposure_col]),
            _format_number(row[actual_col]),
            _format_number(row[expected_col]),
            _format_ratio(row[ratio_col]),
            _format_percent(row[ci_low_col]),
            _format_percent(row[ci_high_col]),
        ]
        for _, row in df.iterrows()
    ]


def _prepare_scatter_source(df: pd.DataFrame, metric: str, data_path: str) -> tuple[pd.DataFrame, dict[str, str]]:
    metric_cols = _metric_columns(metric)
    cohort_col = "Dimensions"
    exposure_col = "Sum_MOC"
    required_cols = [
        cohort_col,
        exposure_col,
        metric_cols["actual"],
        metric_cols["expected"],
        metric_cols["ratio"],
        metric_cols["ci_low"],
        metric_cols["ci_high"],
    ]
    _required_columns(df, required_cols, data_path)

    prepared = df.copy()
    prepared[cohort_col] = prepared[cohort_col].astype(str)
    prepared = prepared.sort_values(metric_cols["ratio"], ascending=False).reset_index(drop=True)
    prepared["display_ratio"] = prepared[metric_cols["ratio"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_low"] = prepared[metric_cols["ci_low"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["display_ci_high"] = prepared[metric_cols["ci_high"]].clip(lower=0, upper=_SCATTER_X_MAX)
    prepared["marker_size"] = _marker_sizes(prepared[exposure_col])
    return prepared, metric_cols


def _build_scatter_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    prepared, metric_cols = _prepare_scatter_source(df, metric, data_path)
    cohort_col = "Dimensions"
    exposure_col = "Sum_MOC"
    ratio_label = _ratio_label(metric)

    display_ae = prepared["display_ratio"].astype(float)
    display_ci_low = prepared["display_ci_low"].astype(float)
    display_ci_high = prepared["display_ci_high"].astype(float)
    err_plus = (display_ci_high - display_ae).clip(lower=0)
    err_minus = (display_ae - display_ci_low).clip(lower=0)

    fig = go.Figure()
    group_map = _first_dimension_groups(prepared[cohort_col])
    if group_map:
        palette = {
            group: _ACCENT_COLORS[index % len(_ACCENT_COLORS)]
            for index, group in enumerate(dict.fromkeys(group_map.values()))
        }
        for group, color in palette.items():
            mask = prepared[cohort_col].map(group_map).eq(group)
            subset = prepared[mask]
            subset_display_ae = subset["display_ratio"].astype(float)
            subset_display_ci_low = subset["display_ci_low"].astype(float)
            subset_display_ci_high = subset["display_ci_high"].astype(float)
            fig.add_trace(
                go.Scatter(
                    x=subset_display_ae,
                    y=subset[cohort_col],
                    mode="markers",
                    name=group,
                    marker=dict(
                        color=color,
                        size=subset["marker_size"],
                        line=dict(color="#ffffff", width=1.2),
                        opacity=0.92,
                    ),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(subset_display_ci_high - subset_display_ae).clip(lower=0),
                        arrayminus=(subset_display_ae - subset_display_ci_low).clip(lower=0),
                        visible=True,
                        color=color,
                        thickness=1.4,
                    ),
                    customdata=_customdata_rows(
                        subset,
                        exposure_col,
                        metric_cols["actual"],
                        metric_cols["expected"],
                        metric_cols["ratio"],
                        metric_cols["ci_low"],
                        metric_cols["ci_high"],
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Exposure (MOC): %{customdata[1]}<br>"
                        "Actual: %{customdata[2]}<br>"
                        "Expected: %{customdata[3]}<br>"
                        f"{ratio_label}: "
                        "%{customdata[4]}<br>"
                        "95% CI: %{customdata[5]} - %{customdata[6]}<extra></extra>"
                    ),
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=display_ae,
                y=prepared[cohort_col],
                mode="markers",
                name=ratio_label,
                marker=dict(
                    color=_NEUTRAL_COLOR,
                    size=prepared["marker_size"],
                    line=dict(color="#ffffff", width=1.2),
                    opacity=0.92,
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=err_plus,
                    arrayminus=err_minus,
                    visible=True,
                    color=_NEUTRAL_COLOR,
                    thickness=1.4,
                ),
                customdata=_customdata_rows(
                    prepared,
                    exposure_col,
                    metric_cols["actual"],
                    metric_cols["expected"],
                    metric_cols["ratio"],
                    metric_cols["ci_low"],
                    metric_cols["ci_high"],
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Exposure (MOC): %{customdata[1]}<br>"
                    "Actual: %{customdata[2]}<br>"
                    "Expected: %{customdata[3]}<br>"
                    f"{ratio_label}: "
                    "%{customdata[4]}<br>"
                    "95% CI: %{customdata[5]} - %{customdata[6]}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=1.0, line_dash="dash", line_color="#c05252", line_width=2)
    fig.update_xaxes(title_text=ratio_label, zeroline=False, gridcolor="#d9e2ec", range=[0, _SCATTER_X_MAX])
    fig.update_yaxes(title_text="Cohort", automargin=True, autorange="reversed")
    _common_layout(fig, f"Forest Plot ({_metric_label(metric)})", height=_scatter_height(len(prepared)))
    fig.update_layout(showlegend=bool(group_map), legend=dict(orientation="h", y=1.08, x=0.02))
    return fig


def _build_table_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    prepared, metric_cols = _prepare_scatter_source(df, metric, data_path)
    ratio_label = _ratio_label(metric)
    fig = go.Figure(
        go.Table(
            header=dict(
                values=[
                    "Cohort",
                    "Exposure (MOC)",
                    "Actual",
                    "Expected",
                    ratio_label,
                    "95% CI",
                ],
                fill_color="#e7eef5",
                align="left",
                font=dict(color="#102a43", size=12),
            ),
            cells=dict(
                values=[
                    prepared["Dimensions"],
                    prepared["Sum_MOC"].map(_format_number),
                    prepared[metric_cols["actual"]].map(_format_number),
                    prepared[metric_cols["expected"]].map(_format_number),
                    prepared[metric_cols["ratio"]].map(_format_ratio),
                    prepared[metric_cols["ci_low"]].map(_format_percent)
                    + " - "
                    + prepared[metric_cols["ci_high"]].map(_format_percent),
                ],
                align="left",
                fill_color="#ffffff",
                height=30,
            ),
        )
    )
    _common_layout(fig, f"Filtered Cohort Detail ({_metric_label(metric)})", height=_table_height(len(prepared)))
    return fig


def _aggregate_treemap_nodes(
    df: pd.DataFrame,
    value_col: str,
    ratio_col: str,
    actual_col: Optional[str],
    expected_col: Optional[str],
) -> pd.DataFrame:
    """Build a true hierarchy for multi-dimensional sweeps without a synthetic root node."""
    leaf_records = []
    parent_totals: dict[str, dict[str, float | str]] = {}

    for _, row in df.iterrows():
        parts = _split_dimensions(row["Dimensions"])
        if not parts:
            continue

        leaf_id = "leaf::" + " | ".join(parts)
        leaf_parent = "" if len(parts) == 1 else "node::" + " | ".join(parts[:-1])
        leaf_records.append(
            {
                "id": leaf_id,
                "parent": leaf_parent,
                "label": parts[-1],
                "full_label": row["Dimensions"],
                "value": float(row[value_col]),
                "ratio": float(row[ratio_col]),
                "actual": float(row[actual_col]) if actual_col and actual_col in row.index else None,
                "expected": float(row[expected_col]) if expected_col and expected_col in row.index else None,
                "is_leaf": True,
            }
        )

        for depth in range(1, len(parts)):
            prefix = " | ".join(parts[:depth])
            node_id = "node::" + prefix
            parent_id = "" if depth == 1 else "node::" + " | ".join(parts[: depth - 1])
            bucket = parent_totals.setdefault(
                node_id,
                {
                    "id": node_id,
                    "parent": parent_id,
                    "label": parts[depth - 1],
                    "full_label": prefix,
                    "value": 0.0,
                    "ratio_numerator": 0.0,
                    "actual": 0.0,
                    "expected": 0.0,
                    "is_leaf": False,
                },
            )
            bucket["value"] = float(bucket["value"]) + float(row[value_col])
            bucket["ratio_numerator"] = float(bucket["ratio_numerator"]) + float(row[ratio_col]) * float(row[value_col])
            if actual_col and actual_col in row.index:
                bucket["actual"] = float(bucket["actual"]) + float(row[actual_col])
            if expected_col and expected_col in row.index:
                bucket["expected"] = float(bucket["expected"]) + float(row[expected_col])

    parent_records = []
    for bucket in parent_totals.values():
        value = float(bucket["value"])
        weighted_ratio = float(bucket["ratio_numerator"]) / value if value else 0.0
        parent_records.append(
            {
                "id": bucket["id"],
                "parent": bucket["parent"],
                "label": bucket["label"],
                "full_label": bucket["full_label"],
                "value": value,
                "ratio": weighted_ratio,
                "actual": bucket["actual"],
                "expected": bucket["expected"],
                "is_leaf": False,
            }
        )

    return pd.DataFrame(parent_records + leaf_records)


def _flat_treemap_nodes(
    df: pd.DataFrame,
    value_col: str,
    ratio_col: str,
    actual_col: Optional[str],
    expected_col: Optional[str],
) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "id": "leaf::" + str(row["Dimensions"]),
                "parent": "",
                "label": str(row["Dimensions"]),
                "full_label": str(row["Dimensions"]),
                "value": float(row[value_col]),
                "ratio": float(row[ratio_col]),
                "actual": float(row[actual_col]) if actual_col and actual_col in row.index else None,
                "expected": float(row[expected_col]) if expected_col and expected_col in row.index else None,
                "is_leaf": True,
            }
        )
    return pd.DataFrame(records)


def _build_treemap_figure(df: pd.DataFrame, metric: str, data_path: str) -> go.Figure:
    _validate_metric(metric)
    if "Dimensions" not in df.columns:
        raise ValueError(f"Missing required column 'Dimensions' in {data_path}")

    metric_cols = _metric_columns(metric)
    ratio_label = _ratio_label(metric)
    value_col, value_label = _treemap_value_spec(df, metric)
    _required_columns(df, [metric_cols["ratio"], value_col], data_path)

    labels = df["Dimensions"].astype(str)
    depths = _dimension_depths(labels)
    max_depth = max(depths) if depths else 1
    has_mixed_depth = len(set(depths)) > 1

    actual_col = metric_cols["actual"] if metric_cols["actual"] in df.columns else None
    expected_col = metric_cols["expected"] if metric_cols["expected"] in df.columns else None

    if max_depth == 1 or has_mixed_depth:
        nodes = _flat_treemap_nodes(df, value_col, metric_cols["ratio"], actual_col, expected_col)
    else:
        nodes = _aggregate_treemap_nodes(df, value_col, metric_cols["ratio"], actual_col, expected_col)

    customdata = [
        [
            row["full_label"],
            _format_number(float(row["value"])),
            _format_ratio(float(row["ratio"])),
            _format_number(float(row["actual"])) if row["actual"] is not None else "n/a",
            _format_number(float(row["expected"])) if row["expected"] is not None else "n/a",
            "Leaf cohort" if row["is_leaf"] else "Parent grouping",
        ]
        for _, row in nodes.iterrows()
    ]

    fig = go.Figure(
        go.Treemap(
            ids=nodes["id"],
            labels=nodes["label"],
            parents=nodes["parent"],
            values=nodes["value"],
            branchvalues="total",
            marker=dict(
                colors=nodes["ratio"],
                colorscale="RdYlGn_r",
                cmid=1.0,
                cmin=0.0,
                cmax=_TREEMAP_COLOR_MAX,
                line=dict(width=1, color="#ffffff"),
            ),
            texttemplate="%{label}<br>A/E: %{customdata[2]}",
            textposition="middle center",
            pathbar=dict(visible=False),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{value_label}: %{customdata[1]}<br>"
                "Actual: %{customdata[3]}<br>"
                "Expected: %{customdata[4]}<br>"
                f"{ratio_label}: "
                "%{customdata[2]}<br>"
                "%{customdata[5]}<extra></extra>"
            ),
            customdata=customdata,
        )
    )
    _common_layout(fig, f"Risk Treemap ({_metric_label(metric)})", height=900)
    fig.update_layout(uniformtext=dict(minsize=10, mode="hide"))
    return fig


def _figure_fragment(fig: go.Figure) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_FIGURE_CONFIG)


def _build_report_html(
    title: str,
    metric: str,
    scatter_fragment: str,
    table_fragment: str,
    treemap_fragment: str,
) -> str:
    plotly_js = get_plotlyjs()
    ratio_label = _ratio_label(metric)
    metric_label = _metric_label(metric)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f3ed;
      --panel: #ffffff;
      --panel-soft: #fbfaf7;
      --ink: #1f2933;
      --muted: #52606d;
      --line: rgba(31, 41, 51, 0.12);
      --accent: #1f4e79;
      --shadow: 0 18px 50px rgba(31, 41, 51, 0.08);
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      padding: 32px 24px 48px;
      background:
        radial-gradient(circle at top left, rgba(31, 78, 121, 0.08), transparent 28%),
        linear-gradient(180deg, #faf7f2 0%, var(--bg) 100%);
      color: var(--ink);
      font-family: "Avenir Next", "Segoe UI", Arial, sans-serif;
    }}

    .report-shell {{
      max-width: 1280px;
      margin: 0 auto;
    }}

    .report-header {{
      margin-bottom: 18px;
      padding: 6px 2px 14px;
    }}

    .eyebrow {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(31, 78, 121, 0.1);
      color: var(--accent);
      font-size: 0.8rem;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1.05;
    }}

    .report-subtitle {{
      margin: 0;
      color: var(--muted);
      font-size: 1rem;
      line-height: 1.55;
      max-width: 760px;
    }}

    .report-section {{
      margin-top: 26px;
      padding: 22px 22px 18px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }}

    .report-section h2 {{
      margin: 0 0 8px;
      font-size: 1.35rem;
      line-height: 1.15;
    }}

    .report-section p {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.5;
    }}

    .figure-shell {{
      border-radius: 18px;
      overflow: hidden;
      background: var(--panel-soft);
    }}
  </style>
  <script type="text/javascript">{plotly_js}</script>
</head>
<body>
  <main class="report-shell">
    <header class="report-header">
      <div class="eyebrow">Experience Study</div>
      <h1>{escape(title)}</h1>
      <p class="report-subtitle">Forest plot, full cohort detail table, and treemap combined into one offline report for browser-first review. Primary metric shown: {escape(ratio_label)}.</p>
    </header>

    <section class="report-section" id="forest-plot">
      <h2>Forest Plot ({escape(metric_label)})</h2>
      <p>Ranked cohort {escape(ratio_label)} view with confidence intervals and exposure-aware marker sizing. Displayed x-axis is capped at {_SCATTER_X_MAX:.1f} for readability.</p>
      <div class="figure-shell">{scatter_fragment}</div>
    </section>

    <section class="report-section" id="cohort-detail">
      <h2>Filtered Cohort Detail ({escape(metric_label)})</h2>
      <p>Full filtered sweep rows shown directly under the forest plot for exact cohort inspection using {escape(ratio_label)}.</p>
      <div class="figure-shell">{table_fragment}</div>
    </section>

    <section class="report-section" id="risk-treemap">
      <h2>Risk Treemap ({escape(metric_label)})</h2>
      <p>Structural risk view using the same filtered cohort slice as the scatter plot and detail table, colored by {escape(ratio_label)}. Color intensity is capped at A/E {_TREEMAP_COLOR_MAX:.1f} for readability.</p>
      <div class="figure-shell">{treemap_fragment}</div>
    </section>
  </main>
</body>
</html>
"""


def _unique_report_output_path(base_filename: str) -> str:
    """Generate a unique HTML artifact path so prior chat visuals are not overwritten."""
    directory, filename = os.path.split(base_filename)
    stem, ext = os.path.splitext(filename)
    unique_suffix = uuid4().hex[:10]
    return os.path.join(directory, f"{stem}_{unique_suffix}{ext}")


def _write_combined_report(data_path: str, metric: str, output_filename: str) -> str:
    _validate_metric(metric)
    df = pd.read_csv(data_path)

    scatter_fragment = _figure_fragment(_build_scatter_figure(df, metric, data_path))
    table_fragment = _figure_fragment(_build_table_figure(df, metric, data_path))
    treemap_fragment = _figure_fragment(_build_treemap_figure(df, metric, data_path))

    metric_label = _metric_label(metric)
    title = f"Combined A/E Visualization Report ({metric_label})"
    html = _build_report_html(title, metric, scatter_fragment, table_fragment, treemap_fragment)

    out_path = os.path.abspath(_unique_report_output_path(output_filename))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(html)
    return f"Visualization report generated: {out_path}"


def generate_univariate_report(data_path: str = "data/output/sweep_summary.csv", metric: str = "amount") -> str:
    """Generate the unified combined visualization report."""
    return _write_combined_report(
        data_path=data_path,
        metric=metric,
        output_filename="data/output/temp_univariate_report.html",
    )


def generate_treemap_report(data_path: str = "data/output/sweep_summary.csv", metric: str = "amount") -> str:
    """Generate the unified combined visualization report."""
    return _write_combined_report(
        data_path=data_path,
        metric=metric,
        output_filename="data/output/temp_treemap_report.html",
    )
