"""
Insight Engine: Mathematical core for A/E mortality ratio calculations with Bayesian CIs.

Uses Jeffreys Prior (Beta distribution) for mortality rate credible intervals.
NEVER allow the LLM to perform actuarial math; all calculations are deterministic.
"""

import difflib
import json
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import pandas as pd
from scipy import stats

from tools.data_io import CANONICAL_ANALYSIS_OUTPUT_PATH, resolve_prepared_analysis_path

try:  # pragma: no cover - exercised in runtime and integration tests
    import duckdb
except ImportError:  # pragma: no cover - dependency installed in normal project envs
    duckdb = None  # type: ignore[assignment]


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
}
SEMANTIC_NUMERIC_NON_DIMENSIONS = {
    "Face_Amount",
    "Issue_Age",
    "Age",
}

CORE_COLUMNS = ["MAC", "MOC", "MEC", "MAF", "MEF"]
VALID_SORT_COLUMNS = {
    "AE_Ratio_Count",
    "AE_Ratio_Amount",
    "Sum_MAC",
    "Sum_MOC",
    "Sum_MEC",
    "Sum_MAF",
    "Sum_MEF",
}
VALID_FILTER_OPERATORS = {"=", "!=", ">", ">=", "<", "<="}
STRING_LIKE_TYPES = {"VARCHAR", "TEXT", "STRING"}
BOOLEAN_TYPES = {"BOOLEAN", "BOOL"}
NUMERIC_TYPE_PREFIXES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "HUGEINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "REAL",
)

SWEEP_OUTPUT_PATH = "data/output/sweep_summary.csv"
AUTO_DIMENSION_SCREEN_LIMIT = 12
MAX_EXPLICIT_COMBINATIONS = 100


def _latest_sweep_output_path_for_depth(depth: int) -> Path:
    """Return the stable latest-alias path for a given sweep depth."""
    return Path(f"data/output/sweep_summary_latest_{depth}.csv")


def _quote_identifier(identifier: str) -> str:
    """Safely quote a SQL identifier for DuckDB."""
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _error_payload(
    message: str,
    *,
    available_columns: Optional[Sequence[str]] = None,
    suggested_columns: Optional[Sequence[str]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Create a standardized JSON error payload."""
    payload: dict[str, Any] = {"error": message}
    if available_columns is not None:
        payload["available_columns"] = list(available_columns)
    if suggested_columns:
        payload["suggested_columns"] = list(suggested_columns)
    if extra:
        payload.update(extra)
    return json.dumps(payload, indent=2)


def _suggest_columns(requested_column: str, available_columns: Sequence[str]) -> list[str]:
    """Return close column-name matches for graceful recovery guidance."""
    lookup = {col.lower(): col for col in available_columns}
    matches = difflib.get_close_matches(requested_column.lower(), lookup.keys(), n=5, cutoff=0.4)
    return [lookup[match] for match in matches]


def _create_analysis_source_view(connection: Any, data_path: Path) -> Optional[str]:
    """Create a temporary DuckDB view over the prepared analysis artifact."""
    escaped_path = str(data_path).replace("'", "''")
    suffix = data_path.suffix.lower()

    if suffix == ".parquet":
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW analysis_source AS "
            f"SELECT * FROM read_parquet('{escaped_path}')"
        )
        return None

    if suffix == ".csv":
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW analysis_source AS "
            f"SELECT * FROM read_csv_auto('{escaped_path}', header=true)"
        )
        return None

    return (
        f"Unsupported prepared analysis format: `{data_path.suffix or '<none>'}`. "
        "Prepared analysis sweeps support `.parquet` and legacy `.csv` artifacts."
    )


def _describe_analysis_source(connection: Any) -> pd.DataFrame:
    """Return DuckDB schema metadata for the prepared analysis view."""
    return connection.execute("DESCRIBE analysis_source").df()


def _column_names(schema_df: pd.DataFrame) -> list[str]:
    """Extract column names from DuckDB schema metadata."""
    return schema_df["column_name"].astype(str).tolist()


def _is_numeric_type(type_name: str) -> bool:
    """Return True for numeric DuckDB type names."""
    upper_type = type_name.upper()
    return upper_type.startswith(NUMERIC_TYPE_PREFIXES)


def _identify_categorical_columns(connection: Any, schema_df: pd.DataFrame) -> list[str]:
    """Identify sweep-eligible dimensions using DuckDB schema metadata and distinct counts."""
    candidate_rows = []
    for row in schema_df.itertuples(index=False):
        column_name = str(row.column_name)
        column_type = str(row.column_type).upper()
        if column_name in EXCLUDED_DIMENSIONS or column_name in SEMANTIC_NUMERIC_NON_DIMENSIONS:
            continue
        candidate_rows.append((column_name, column_type))

    categorical = [
        column_name
        for column_name, column_type in candidate_rows
        if column_type in STRING_LIKE_TYPES or column_type in BOOLEAN_TYPES
    ]

    numeric_candidates = [
        column_name for column_name, column_type in candidate_rows if _is_numeric_type(column_type)
    ]
    if not numeric_candidates:
        return categorical

    alias_map = {column_name: f"distinct_{idx}" for idx, column_name in enumerate(numeric_candidates)}
    distinct_sql = ", ".join(
        f"COUNT(DISTINCT {_quote_identifier(column_name)}) AS {alias_map[column_name]}"
        for column_name in numeric_candidates
    )
    distinct_counts = connection.execute(f"SELECT {distinct_sql} FROM analysis_source").df().iloc[0].to_dict()

    for column_name in numeric_candidates:
        if int(distinct_counts[alias_map[column_name]]) <= 20:
            categorical.append(column_name)

    return categorical


def _validate_filters(
    filters: Optional[List[dict[str, Any]]],
    available_columns: Sequence[str],
) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
    """Validate and normalize structured sweep filters."""
    normalized_filters: list[dict[str, Any]] = []

    for raw_filter in filters or []:
        if not isinstance(raw_filter, dict):
            return (None, _error_payload("Each filter must be an object with column, operator, and value."))

        column = str(raw_filter.get("column", "")).strip()
        operator = str(raw_filter.get("operator", "")).strip()
        value = raw_filter.get("value")

        if not column:
            return (
                None,
                _error_payload(
                    "Each filter must include a non-empty `column`.",
                    available_columns=sorted(available_columns),
                ),
            )

        if column not in available_columns:
            return (
                None,
                _error_payload(
                    f"Column '{column}' not found.",
                    available_columns=sorted(available_columns),
                    suggested_columns=_suggest_columns(column, available_columns),
                ),
            )

        if operator not in VALID_FILTER_OPERATORS:
            return (
                None,
                _error_payload(
                    f"Unsupported operator '{operator}'. Supported operators: {sorted(VALID_FILTER_OPERATORS)}.",
                    available_columns=sorted(available_columns),
                ),
            )

        if isinstance(value, (list, dict)):
            return (
                None,
                _error_payload(
                    f"Filter value for column '{column}' must be a scalar.",
                    available_columns=sorted(available_columns),
                ),
            )

        normalized_filters.append({"column": column, "operator": operator, "value": value})

    return (normalized_filters, None)


def _build_where_clause(filters: Sequence[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Build a parameterized DuckDB WHERE clause from validated filters."""
    if not filters:
        return ("1=1", [])

    clauses = [
        f"{_quote_identifier(filter_spec['column'])} {filter_spec['operator']} ?"
        for filter_spec in filters
    ]
    params = [filter_spec["value"] for filter_spec in filters]
    return (" AND ".join(clauses), params)


def _validate_selected_columns(
    selected_columns: Optional[List[str]],
    available_columns: Sequence[str],
) -> tuple[Optional[list[str]], Optional[str]]:
    """Validate requested sweep columns against the prepared dataset."""
    if not selected_columns:
        return (None, None)

    missing_columns = [column for column in selected_columns if column not in available_columns]
    if missing_columns:
        missing_column = missing_columns[0]
        return (
            None,
            _error_payload(
                f"Column '{missing_column}' not found.",
                available_columns=sorted(available_columns),
                suggested_columns=_suggest_columns(missing_column, available_columns),
            ),
        )

    invalid_dimensions = [
        column
        for column in selected_columns
        if column in EXCLUDED_DIMENSIONS or column in SEMANTIC_NUMERIC_NON_DIMENSIONS
    ]
    if invalid_dimensions:
        return (
            None,
            _error_payload(
                f"Column '{invalid_dimensions[0]}' is not eligible as a sweep dimension.",
                available_columns=sorted(
                    column
                    for column in available_columns
                    if column not in EXCLUDED_DIMENSIONS and column not in SEMANTIC_NUMERIC_NON_DIMENSIONS
                ),
            ),
        )

    deduped_columns: list[str] = []
    for column in selected_columns:
        if column not in deduped_columns:
            deduped_columns.append(column)
    return (deduped_columns, None)


def _aggregate_dimension_combination(
    connection: Any,
    dim_cols: Sequence[str],
    where_sql: str,
    where_params: Sequence[Any],
    min_mac: int,
) -> pd.DataFrame:
    """Return grouped sweep rows for one dimension combination."""
    dim_select = ", ".join(_quote_identifier(column) for column in dim_cols)
    aggregate_sql = f"""
        SELECT
            {dim_select},
            SUM({_quote_identifier("MAC")}) AS Sum_MAC,
            SUM({_quote_identifier("MOC")}) AS Sum_MOC,
            SUM({_quote_identifier("MEC")}) AS Sum_MEC,
            SUM({_quote_identifier("MAF")}) AS Sum_MAF,
            SUM({_quote_identifier("MEF")}) AS Sum_MEF
        FROM analysis_source
        WHERE {where_sql}
        GROUP BY {dim_select}
        HAVING SUM({_quote_identifier("MAC")}) >= ?
    """
    return connection.execute(aggregate_sql, [*where_params, min_mac]).df()


def _auto_screen_metric(sort_by: str) -> str:
    """Choose the metric used to rank auto-discovered dimensions."""
    return "AE_Ratio_Count" if sort_by == "AE_Ratio_Count" else "AE_Ratio_Amount"


def _score_auto_screen_dimension(
    connection: Any,
    column_name: str,
    where_sql: str,
    where_params: Sequence[Any],
    min_mac: int,
    metric_name: str,
) -> Optional[float]:
    """Score a dimension by the largest absolute A/E deviation from 1.0."""
    grouped = _aggregate_dimension_combination(connection, [column_name], where_sql, where_params, min_mac)
    if grouped.empty:
        return None

    grouped["AE_Ratio_Count"] = grouped["Sum_MAC"] / grouped["Sum_MEC"]
    grouped["AE_Ratio_Amount"] = grouped["Sum_MAF"] / grouped["Sum_MEF"]
    metric_series = grouped[metric_name].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if metric_series.empty:
        return None

    return float((metric_series.astype(float) - 1.0).abs().max())


def _rank_auto_screened_dimensions(
    connection: Any,
    candidate_columns: Sequence[str],
    where_sql: str,
    where_params: Sequence[Any],
    min_mac: int,
    sort_by: str,
    limit: int = AUTO_DIMENSION_SCREEN_LIMIT,
) -> list[str]:
    """Return the top-scoring auto-discovered dimensions for multiway sweeps."""
    metric_name = _auto_screen_metric(sort_by)
    scored_dimensions: list[tuple[int, float, str]] = []

    for index, column_name in enumerate(candidate_columns):
        score = _score_auto_screen_dimension(
            connection,
            column_name,
            where_sql,
            where_params,
            min_mac,
            metric_name,
        )
        if score is None:
            continue
        scored_dimensions.append((index, score, column_name))

    scored_dimensions.sort(key=lambda item: (-item[1], item[0]))
    return [column_name for _, _, column_name in scored_dimensions[:limit]]


def run_dimensional_sweep(
    depth: int = 1,
    filters: Optional[List[dict[str, Any]]] = None,
    selected_columns: Optional[List[str]] = None,
    min_mac: int = 0,
    top_n: int = 20,
    sort_by: str = "AE_Ratio_Amount",
    data_path: str = CANONICAL_ANALYSIS_OUTPUT_PATH,
    confidence_level: float = 0.95,
) -> str:
    """
    Run a dimensional sweep: aggregate by categorical dimension combinations,
    compute A/E ratios with Bayesian CIs, and return top N results as JSON.
    """
    if duckdb is None:
        return _error_payload("DuckDB is not installed. Add the `duckdb` dependency before running sweeps.")

    resolved_path = resolve_prepared_analysis_path(data_path)
    if not resolved_path.exists():
        return _error_payload(
            f"Prepared analysis dataset not found. Checked `{data_path}`.",
            extra={"checked_paths": [str(resolve_prepared_analysis_path(data_path))]},
        )

    if sort_by not in VALID_SORT_COLUMNS:
        return _error_payload(f"sort_by must be one of {sorted(VALID_SORT_COLUMNS)}")

    connection = duckdb.connect(database=":memory:")
    try:
        source_error = _create_analysis_source_view(connection, resolved_path)
        if source_error:
            return _error_payload(source_error)

        schema_df = _describe_analysis_source(connection)
        available_columns = _column_names(schema_df)

        missing_core_columns = [column for column in CORE_COLUMNS if column not in available_columns]
        if missing_core_columns:
            return _error_payload(
                f"Missing required columns: {missing_core_columns}",
                available_columns=sorted(available_columns),
            )

        normalized_selected_columns, selected_error = _validate_selected_columns(
            selected_columns,
            available_columns,
        )
        if selected_error:
            return selected_error

        normalized_filters, filters_error = _validate_filters(filters, available_columns)
        if filters_error:
            return filters_error

        if normalized_selected_columns:
            dim_columns = normalized_selected_columns
        else:
            dim_columns = _identify_categorical_columns(connection, schema_df)

        if len(dim_columns) < depth:
            return _error_payload(
                f"Need at least {depth} dimension columns, found {len(dim_columns)}.",
                available_columns=sorted(dim_columns),
            )

        where_sql, where_params = _build_where_clause(normalized_filters or [])
        if normalized_selected_columns:
            requested_combinations = math.comb(len(dim_columns), depth)
            if requested_combinations > MAX_EXPLICIT_COMBINATIONS:
                return _error_payload(
                    "Requested explicit sweep dimensions would generate too many combinations.",
                    available_columns=sorted(dim_columns),
                    extra={
                        "requested_combination_count": requested_combinations,
                        "max_supported_combinations": MAX_EXPLICIT_COMBINATIONS,
                    },
                )
        elif depth >= 2:
            screened_columns = _rank_auto_screened_dimensions(
                connection,
                dim_columns,
                where_sql,
                where_params,
                min_mac,
                sort_by,
            )
            if len(screened_columns) >= depth:
                dim_columns = screened_columns

        combo_list = list(combinations(dim_columns, depth))
        all_results: List[dict[str, Any]] = []

        for dim_cols in combo_list:
            grouped = _aggregate_dimension_combination(connection, dim_cols, where_sql, where_params, min_mac)
            if grouped.empty:
                continue

            grouped["AE_Ratio_Count"] = grouped["Sum_MAC"] / grouped["Sum_MEC"]
            grouped["AE_Ratio_Amount"] = grouped["Sum_MAF"] / grouped["Sum_MEF"]

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

            grouped["AE_Count_CI_Lower"] = [ci[0] for ci in ci_count]
            grouped["AE_Count_CI_Upper"] = [ci[1] for ci in ci_count]
            grouped["AE_Amount_CI_Lower"] = [ci[0] for ci in ci_amount]
            grouped["AE_Amount_CI_Upper"] = [ci[1] for ci in ci_amount]
            grouped["Dimensions"] = grouped.apply(
                lambda row: " | ".join(f"{column}={row[column]}" for column in dim_cols),
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
        top_results = result_df.head(top_n)

        summary_df = result_df.copy()
        summary_df["AE_Count_CI_Lower"] = summary_df["AE_Count_CI"].apply(lambda ci: ci[0] if ci else None)
        summary_df["AE_Count_CI_Upper"] = summary_df["AE_Count_CI"].apply(lambda ci: ci[1] if ci else None)
        summary_df["AE_Amount_CI_Lower"] = summary_df["AE_Amount_CI"].apply(lambda ci: ci[0] if ci else None)
        summary_df["AE_Amount_CI_Upper"] = summary_df["AE_Amount_CI"].apply(lambda ci: ci[1] if ci else None)
        summary_df = summary_df.drop(columns=["AE_Count_CI", "AE_Amount_CI"])

        involved_columns = normalized_selected_columns if normalized_selected_columns else dim_columns
        columns_slug = "_".join(
            re.sub(r"[^a-zA-Z0-9]+", "_", column).strip("_").lower() for column in involved_columns
        )
        if not columns_slug:
            columns_slug = "all_dimensions"
        if len(columns_slug) > 120:
            columns_slug = columns_slug[:120].rstrip("_")

        dynamic_out_path = Path(f"data/output/sweep_summary_{depth}_{columns_slug}.csv")
        dynamic_out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(dynamic_out_path, index=False)

        latest_out_path = Path(SWEEP_OUTPUT_PATH)
        latest_out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(latest_out_path, index=False)

        latest_depth_out_path = _latest_sweep_output_path_for_depth(depth)
        latest_depth_out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(latest_depth_out_path, index=False)

        out_list = top_results.to_dict(orient="records")
        for record in out_list:
            record["AE_Count_CI"] = [
                float(record["AE_Count_CI"][0]) if record["AE_Count_CI"][0] is not None else None,
                float(record["AE_Count_CI"][1]) if record["AE_Count_CI"][1] is not None else None,
            ]
            record["AE_Amount_CI"] = [
                float(record["AE_Amount_CI"][0]) if record["AE_Amount_CI"][0] is not None else None,
                float(record["AE_Amount_CI"][1]) if record["AE_Amount_CI"][1] is not None else None,
            ]

        return json.dumps(
            {
                "results": out_list,
                "depth": depth,
                "output_path": str(dynamic_out_path),
                "latest_output_path": str(latest_out_path),
                "latest_depth_output_path": str(latest_depth_out_path),
            },
            indent=2,
        )
    finally:
        connection.close()


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
    assert all(row["Sum_MAC"] >= 5 for row in results), "All cohorts must have at least 5 deaths"
    ae_counts = [row["AE_Ratio_Count"] for row in results]
    assert ae_counts == sorted(ae_counts, reverse=True), "Results must be sorted by AE_Ratio_Count descending"
    assert len(results) <= 5, "Must return at most 5 results"
    print("\n✓ High Credibility Sweep verified: all cohorts have Sum_MAC >= 5, sorted by AE_Ratio_Count.")
