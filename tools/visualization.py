"""Standalone plotting engine for A/E univariate reports using Plotly."""

import os
import webbrowser
from typing import List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _get_ci_columns(metric: str) -> tuple[str, str]:
    """Return (lower_col, upper_col) names for the given metric."""
    if metric == "count":
        return "AE_Count_CI_Lower", "AE_Count_CI_Upper"
    return "AE_Amount_CI_Lower", "AE_Amount_CI_Upper"


def generate_univariate_report(data_path: str, metric: str = "amount") -> str:
    """
    Generate an interactive HTML univariate A/E report and open it in the browser.

    Parameters
    ----------
    data_path : str
        Path to the aggregated A/E summary CSV (e.g., data/sweep_summary.csv).
    metric : str
        Either 'amount' or 'count'. Determines which A/E fields and CIs to use.
    """
    if metric not in {"amount", "count"}:
        raise ValueError("metric must be 'amount' or 'count'")

    df = pd.read_csv(data_path)

    # Core columns
    cohort_col = "Dimensions"
    exposure_col = "Sum_MOC"

    if metric == "count":
        ratio_col = "AE_Ratio_Count"
        actual_col = "Sum_MAC"
        expected_col = "Sum_MEC"
    else:
        ratio_col = "AE_Ratio_Amount"
        actual_col = "Sum_MAF"
        expected_col = "Sum_MEF"

    ci_low_col, ci_high_col = _get_ci_columns(metric)

    required_cols: List[str] = [
        cohort_col,
        exposure_col,
        actual_col,
        expected_col,
        ratio_col,
        ci_low_col,
        ci_high_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    # Compute asymmetric error bars for 95% CI
    ae = df[ratio_col].astype(float)
    ci_low = df[ci_low_col].astype(float)
    ci_high = df[ci_high_col].astype(float)

    err_plus = (ci_high - ae).clip(lower=0)
    err_minus = (ae - ci_low).clip(lower=0)

    # Build figure with scatter (row 1) and table (row 2)
    fig = make_subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.1,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    # Row 1: scatter with error bars
    fig.add_trace(
        go.Scatter(
            x=df[cohort_col],
            y=ae,
            mode="markers",
            marker=dict(color="darkblue", size=10),
            error_y=dict(
                type="data",
                symmetric=False,
                array=err_plus,
                arrayminus=err_minus,
                visible=True,
            ),
            name="A/E Ratio",
        ),
        row=1,
        col=1,
    )

    # Baseline at A/E = 1.0
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", row=1, col=1)

    fig.update_yaxes(title_text="A/E Ratio", row=1, col=1)
    fig.update_xaxes(title_text="Cohort", row=1, col=1)

    # Row 2: data table
    ae_pct = (ae * 100).map(lambda v: f"{v:0.1f}%")
    ci_pct = (ci_low * 100).map(lambda v: f"{v:0.1f}%") + " – " + (ci_high * 100).map(
        lambda v: f"{v:0.1f}%"
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "Cohort",
                    "Exposure (MOC)",
                    "Actual Claims",
                    "Expected Claims",
                    "A/E Ratio",
                    "95% CI",
                ],
                fill_color="#f0f0f0",
                align="left",
            ),
            cells=dict(
                values=[
                    df[cohort_col].astype(str),
                    df[exposure_col],
                    df[actual_col],
                    df[expected_col],
                    ae_pct,
                    ci_pct,
                ],
                align="left",
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Univariate A/E Report" if metric == "amount" else "Univariate A/E Report (Count)",
    )

    # Output HTML and open in browser
    out_path = os.path.abspath("data/temp_univariate_report.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    webbrowser.open("file://" + out_path)

    return f"Univariate report generated and opened: {out_path}"


def generate_treemap_report(data_path: str, metric: str = "amount") -> str:
    """
    Generate a treemap A/E report and open it in the browser.

    Parameters
    ----------
    data_path : str
        Path to the aggregated sweep CSV.
    metric : str
        Either 'amount' or 'count'. Controls box size and coloring metric.
    """
    if metric not in {"amount", "count"}:
        raise ValueError("metric must be 'amount' or 'count'")

    df = pd.read_csv(data_path)

    cohort_col = "Dimensions"
    if cohort_col not in df.columns:
        raise ValueError(f"Missing required column '{cohort_col}' in {data_path}")

    # Box size (values)
    if metric == "count":
        if "sample_size" in df.columns:
            value_col = "sample_size"
        elif "Sum_MOC" in df.columns:
            value_col = "Sum_MOC"
        else:
            raise ValueError("Treemap count metric requires 'sample_size' or 'Sum_MOC' in the data.")
        color_col = "AE_Ratio_Count"
    else:
        if "Total_Face_Amount" in df.columns:
            value_col = "Total_Face_Amount"
        elif "Sum_MAF" in df.columns:
            value_col = "Sum_MAF"
        else:
            raise ValueError("Treemap amount metric requires 'Total_Face_Amount' or 'Sum_MAF' in the data.")
        color_col = "AE_Ratio_Amount"

    if color_col not in df.columns:
        raise ValueError(f"Missing required A/E column '{color_col}' in {data_path}")

    labels = df[cohort_col].astype(str)
    parents = ["Overall"] * len(df)
    values = df[value_col].astype(float)
    colors = df[color_col].astype(float)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(
                colors=colors,
                colorscale="RdYlGn_r",
                cmid=1.0,
            ),
            hovertemplate=(
                "Cohort: %{label}<br>"
                f"Exposure: %{ { 'value' } }<br>"
                "A/E: %{customdata:.1%}<extra></extra>"
            ),
            customdata=colors,
        )
    )

    fig.update_layout(
        height=800,
        title_text=(
            "Risk Treemap (A/E by Amount)" if metric == "amount" else "Risk Treemap (A/E by Count)"
        ),
    )

    out_path = os.path.abspath("data/temp_treemap_report.html")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)
    webbrowser.open("file://" + out_path)

    return f"Treemap report generated and opened: {out_path}"


