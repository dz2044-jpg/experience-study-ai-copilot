"""
Insight Engine: Mathematical core for A/E mortality ratio calculations with Bayesian CIs.

Uses Jeffreys Prior (Beta distribution) for mortality rate credible intervals.
NEVER allow the LLM to perform actuarial math—all calculations are deterministic.
"""

import json
import re
from itertools import combinations
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from scipy import stats


def _compute_mortality_rate_ci(
    mac: float,
    moc: float,
    confidence_level: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute credible interval for mortality rate using Jeffreys Prior (Beta distribution).

    Returns (lower_rate, upper_rate) or (None, None) if inputs are invalid.
    """
    if pd.isna(mac) or pd.isna(moc) or moc <= 0 or mac < 0 or mac > moc:
        return (None, None)

    alpha_beta = mac + 0.5
    beta_beta = moc - mac + 0.5

    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile

    lower_rate = stats.beta.ppf(lower_quantile, alpha_beta, beta_beta)
    upper_rate = stats.beta.ppf(upper_quantile, alpha_beta, beta_beta)

    return (lower_rate, upper_rate)


def compute_ae_ci(
    mac: float,
    moc: float,
    mec: float,
    confidence_level: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute A/E ratio credible interval (Count-based).

    Multiplies rate CI by moc to get credible death counts, then divides by mec.
    Returns (ae_lower, ae_upper).
    """
    rate_lower, rate_upper = _compute_mortality_rate_ci(mac, moc, confidence_level)

    if rate_lower is None or rate_upper is None or mec <= 0:
        return (None, None)

    credible_deaths_lower = rate_lower * moc
    credible_deaths_upper = rate_upper * moc

    ae_lower = credible_deaths_lower / mec
    ae_upper = credible_deaths_upper / mec

    return (ae_lower, ae_upper)


def compute_ae_ci_amount(
    mac: float,
    moc: float,
    mec: float,
    actual_amount: float,
    expected_amount: float,
    confidence_level: float = 0.95,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Compute A/E ratio credible interval (Amount-based).

    Uses hybrid avg_claim: if mac > 0 use actual_amount/mac; else use expected_amount/mec.
    Returns (ae_lower, ae_upper).
    """
    if (
        pd.isna(mac)
        or pd.isna(moc)
        or pd.isna(mec)
        or pd.isna(actual_amount)
        or pd.isna(expected_amount)
        or mec <= 0
        or expected_amount <= 0
    ):
        return (None, None)

    if mac > 0:
        avg_claim = actual_amount / mac
    else:
        avg_claim = expected_amount / mec

    rate_lower, rate_upper = _compute_mortality_rate_ci(mac, moc, confidence_level)

    if rate_lower is None or rate_upper is None:
        return (None, None)

    credible_amount_lower = rate_lower * moc * avg_claim
    credible_amount_upper = rate_upper * moc * avg_claim

    ae_lower = credible_amount_lower / expected_amount
    ae_upper = credible_amount_upper / expected_amount

    return (ae_lower, ae_upper)


EXCLUDED_DIMENSIONS = {
    "Policy_Number",
    "MAC",
    "MOC",
    "MEC",
    "MAF",
    "MEF",
    "COLA",
    "Duration",
}

CORE_COLUMNS = ["MAC", "MOC", "MEC", "MAF", "MEF"]


def _identify_categorical_columns(df: pd.DataFrame) -> List[str]:
    """Identify columns suitable as dimensions: object/string or numeric with <= 20 unique values."""
    categorical = []
    for col in df.columns:
        if col in EXCLUDED_DIMENSIONS:
            continue
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            categorical.append(col)
        elif pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique() <= 20:
                categorical.append(col)
    return categorical


VALID_SORT_COLUMNS = {
    "AE_Ratio_Count",
    "AE_Ratio_Amount",
    "Sum_MAC",
    "Sum_MOC",
    "Sum_MEC",
    "Sum_MAF",
    "Sum_MEF",
}

SWEEP_OUTPUT_PATH = "data/output/sweep_summary.csv"


def run_dimensional_sweep(
    depth: int = 1,
    filters: Optional[List[str]] = None,
    selected_columns: Optional[List[str]] = None,
    min_mac: int = 1,
    top_n: Optional[int] = 20,
    sort_by: str = "AE_Ratio_Amount",
    data_path: str = "data/output/analysis_inforce.csv",
    confidence_level: float = 0.95,
) -> str:
    """
    Run a dimensional sweep: aggregate by categorical dimension combinations,
    compute A/E ratios with Bayesian CIs, and return top N results as JSON.
    """
    path = Path(data_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {data_path}"}, indent=2)

    df = pd.read_csv(data_path)

    if filters:
        query_str = " and ".join(filters)
        df = df.query(query_str)

    # Ensure core columns exist
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        return json.dumps({"error": f"Missing required columns: {missing}"}, indent=2)

    # Identify dimension columns
    if selected_columns:
        dim_columns = [c for c in selected_columns if c in df.columns and c not in EXCLUDED_DIMENSIONS]
    else:
        dim_columns = _identify_categorical_columns(df)

    if len(dim_columns) < depth:
        return json.dumps(
            {"error": f"Need at least {depth} dimension columns, found {len(dim_columns)}"},
            indent=2,
        )

    if sort_by not in VALID_SORT_COLUMNS:
        return json.dumps(
            {"error": f"sort_by must be one of {sorted(VALID_SORT_COLUMNS)}"},
            indent=2,
        )

    combo_list = list(combinations(dim_columns, depth))
    all_results: List[dict] = []

    for dim_cols in combo_list:
        agg_dict = {
            "MAC": "sum",
            "MOC": "sum",
            "MEC": "sum",
            "MAF": "sum",
            "MEF": "sum",
        }
        grouped = df.groupby(list(dim_cols), as_index=False).agg(agg_dict)

        grouped = grouped.rename(
            columns={
                "MAC": "Sum_MAC",
                "MOC": "Sum_MOC",
                "MEC": "Sum_MEC",
                "MAF": "Sum_MAF",
                "MEF": "Sum_MEF",
            }
        )

        # A/E point estimates
        grouped["AE_Ratio_Count"] = grouped["Sum_MAC"] / grouped["Sum_MEC"]
        grouped["AE_Ratio_Amount"] = grouped["Sum_MAF"] / grouped["Sum_MEF"]

        # Beta CIs for each row
        def _row_ci_count(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
            return compute_ae_ci(
                mac=row["Sum_MAC"],
                moc=row["Sum_MOC"],
                mec=row["Sum_MEC"],
                confidence_level=confidence_level,
            )

        def _row_ci_amount(row: pd.Series) -> Tuple[Optional[float], Optional[float]]:
            return compute_ae_ci_amount(
                mac=row["Sum_MAC"],
                moc=row["Sum_MOC"],
                mec=row["Sum_MEC"],
                actual_amount=row["Sum_MAF"],
                expected_amount=row["Sum_MEF"],
                confidence_level=confidence_level,
            )

        ci_count = grouped.apply(_row_ci_count, axis=1)
        ci_amount = grouped.apply(_row_ci_amount, axis=1)

        grouped["AE_Count_CI_Lower"] = [x[0] for x in ci_count]
        grouped["AE_Count_CI_Upper"] = [x[1] for x in ci_count]
        grouped["AE_Amount_CI_Lower"] = [x[0] for x in ci_amount]
        grouped["AE_Amount_CI_Upper"] = [x[1] for x in ci_amount]

        # Visibility filter: keep only cohorts with at least min_mac deaths
        grouped = grouped[grouped["Sum_MAC"] >= min_mac].copy()

        grouped["Dimensions"] = grouped.apply(
            lambda r: " | ".join(f"{c}={r[c]}" for c in dim_cols),
            axis=1,
        )

        for _, row in grouped.iterrows():
            all_results.append(
                {
                    "Dimensions": row["Dimensions"],
                    "Sum_MAC": int(row["Sum_MAC"]),
                    "Sum_MOC": float(row["Sum_MOC"]),
                    "Sum_MEC": float(row["Sum_MEC"]),
                    "Sum_MAF": float(row["Sum_MAF"]),
                    "Sum_MEF": float(row["Sum_MEF"]),
                    "AE_Ratio_Count": float(row["AE_Ratio_Count"]),
                    "AE_Ratio_Amount": float(row["AE_Ratio_Amount"]),
                    "AE_Count_CI": [row["AE_Count_CI_Lower"], row["AE_Count_CI_Upper"]],
                    "AE_Amount_CI": [row["AE_Amount_CI_Lower"], row["AE_Amount_CI_Upper"]],
                }
            )

    if not all_results:
        return json.dumps({"results": [], "message": "No rows passed visibility filter."}, indent=2)

    result_df = pd.DataFrame(all_results)
    result_df = result_df.sort_values(sort_by, ascending=False)
    top_results = result_df.copy() if top_n is None else result_df.head(top_n)

    # Persist sweep summary for downstream visualization tools.
    summary_df = top_results.copy()
    summary_df["AE_Count_CI_Lower"] = summary_df["AE_Count_CI"].apply(lambda x: x[0] if x else None)
    summary_df["AE_Count_CI_Upper"] = summary_df["AE_Count_CI"].apply(lambda x: x[1] if x else None)
    summary_df["AE_Amount_CI_Lower"] = summary_df["AE_Amount_CI"].apply(lambda x: x[0] if x else None)
    summary_df["AE_Amount_CI_Upper"] = summary_df["AE_Amount_CI"].apply(lambda x: x[1] if x else None)
    summary_df = summary_df.drop(columns=["AE_Count_CI", "AE_Amount_CI"])
    # Name output by sweep depth + involved columns for differentiation.
    involved_columns = selected_columns if selected_columns else dim_columns
    columns_slug = "_".join(
        re.sub(r"[^a-zA-Z0-9]+", "_", col).strip("_").lower() for col in involved_columns
    )
    if not columns_slug:
        columns_slug = "all_dimensions"
    if len(columns_slug) > 120:
        columns_slug = columns_slug[:120].rstrip("_")

    dynamic_out_path = Path(f"data/output/sweep_summary_{depth}_{columns_slug}.csv")
    dynamic_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(dynamic_out_path, index=False)

    # Keep a stable latest alias for downstream defaults.
    latest_out_path = Path(SWEEP_OUTPUT_PATH)
    latest_out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(latest_out_path, index=False)

    # Convert to JSON-serializable format
    out_list = top_results.to_dict(orient="records")
    # Fix CI tuples (ensure floats)
    for rec in out_list:
        rec["AE_Count_CI"] = [
            float(rec["AE_Count_CI"][0]) if rec["AE_Count_CI"][0] is not None else None,
            float(rec["AE_Count_CI"][1]) if rec["AE_Count_CI"][1] is not None else None,
        ]
        rec["AE_Amount_CI"] = [
            float(rec["AE_Amount_CI"][0]) if rec["AE_Amount_CI"][0] is not None else None,
            float(rec["AE_Amount_CI"][1]) if rec["AE_Amount_CI"][1] is not None else None,
        ]

    return json.dumps(
        {
            "results": out_list,
            "output_path": str(dynamic_out_path),
            "latest_output_path": str(latest_out_path),
        },
        indent=2,
    )


if __name__ == "__main__":
    confidence = 0.95

    print("=== Task 1: Test 1 (Standard): mac=5, moc=1000, mec=4, maf=500000, mef=400000 ===\n")

    count_lower, count_upper = compute_ae_ci(mac=5, moc=1000, mec=4, confidence_level=confidence)
    print(f"Count A/E CI: ({count_lower}, {count_upper})")

    amount_lower, amount_upper = compute_ae_ci_amount(
        mac=5,
        moc=1000,
        mec=4,
        actual_amount=500000,
        expected_amount=400000,
        confidence_level=confidence,
    )
    print(f"Amount A/E CI: ({amount_lower}, {amount_upper})")

    print("\n=== Task 1: Test 2 (Zero Claims): mac=0, moc=1000, mec=4, maf=0, mef=400000 ===\n")

    count_lower2, count_upper2 = compute_ae_ci(mac=0, moc=1000, mec=4, confidence_level=confidence)
    print(f"Count A/E CI: ({count_lower2}, {count_upper2})")
    assert count_upper2 is not None and count_upper2 > 0, "Zero-claims case must return non-zero upper bound"

    amount_lower2, amount_upper2 = compute_ae_ci_amount(
        mac=0,
        moc=1000,
        mec=4,
        actual_amount=0,
        expected_amount=400000,
        confidence_level=confidence,
    )
    print(f"Amount A/E CI: ({amount_lower2}, {amount_upper2})")
    assert amount_upper2 is not None and amount_upper2 > 0, "Zero-claims case must return non-zero upper bound"

    print("\n✓ Task 1 tests passed.")

    print("\n" + "=" * 60)
    print("=== Dimensional Sweep: depth=2, min_mac=1 ===")
    print("=" * 60 + "\n")

    sweep_json = run_dimensional_sweep(depth=2, min_mac=1)
    print(sweep_json)

    print("\n" + "=" * 60)
    print("=== High Credibility Sweep: depth=2, min_mac=5, top_n=5, sort_by=AE_Ratio_Count ===")
    print("=" * 60 + "\n")

    high_cred = run_dimensional_sweep(
        depth=2,
        min_mac=5,
        top_n=5,
        sort_by="AE_Ratio_Count",
    )
    print(high_cred)

    parsed = json.loads(high_cred)
    results = parsed.get("results", [])
    assert all(r["Sum_MAC"] >= 5 for r in results), "All cohorts must have at least 5 deaths"
    ae_counts = [r["AE_Ratio_Count"] for r in results]
    assert ae_counts == sorted(ae_counts, reverse=True), "Results must be sorted by AE_Ratio_Count descending"
    assert len(results) <= 5, "Must return at most 5 results"
    print("\n✓ High Credibility Sweep verified: all cohorts have Sum_MAC >= 5, sorted by AE_Ratio_Count.")
