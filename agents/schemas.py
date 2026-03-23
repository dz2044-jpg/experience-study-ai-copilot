"""Pydantic schemas defining tool-call contracts for AI agents."""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tools.data_io import CANONICAL_ANALYSIS_OUTPUT_PATH


class ProfileDatasetSchema(BaseModel):
    """Schema for profiling raw inforce data."""

    data_path: str = Field(
        default="data/input/synthetic_inforce.csv",
        description="Path to the source raw input file to profile (.csv, .parquet, or .xlsx).",
    )
    sheet_name: Optional[str] = Field(
        default=None,
        description="Optional worksheet name when data_path points to an .xlsx workbook.",
    )


class FeatureEngineeringSchema(BaseModel):
    """Schema for creating bands or regrouping categories."""

    data_path: str = Field(
        default="data/input/synthetic_inforce.csv",
        description="Source raw input path for feature engineering (alias of source_path).",
    )
    operation: str = Field(
        ...,
        description="Feature engineering operation. Use 'create_bands' or 'regroup_categories'.",
    )
    source_column: str = Field(
        ...,
        description="Column name to transform (e.g., Issue_Age, Risk_Class).",
    )
    strategy: Optional[str] = Field(
        default=None,
        description="Banding strategy for create_bands: quantiles, equal_width, or custom.",
    )
    bins: Optional[int] = Field(
        default=None,
        description="Number of bins for quantiles/equal_width strategies.",
    )
    custom_bins: Optional[List[float]] = Field(
        default=None,
        description="Custom bin edges for strategy='custom' (e.g., [0, 25, 45, 65, 100]).",
    )
    mapping_dict: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Mapping dictionary for regroup_categories (e.g., {'Standard Plus': 'Standard'}).",
    )
    source_path: str = Field(
        default="data/input/synthetic_inforce.csv",
        description="Input raw path for transformation (.csv, .parquet, or .xlsx).",
    )
    output_path: str = Field(
        default=CANONICAL_ANALYSIS_OUTPUT_PATH,
        description="Output prepared-analysis path to save transformed data.",
    )
    sheet_name: Optional[str] = Field(
        default=None,
        description="Optional worksheet name when source_path points to an .xlsx workbook.",
    )


class CategoricalBandingSchema(FeatureEngineeringSchema):
    """Alias schema used by steward tool-calling for categorical banding/regrouping."""


class RegroupCategoricalSchema(BaseModel):
    """Schema for regrouping categorical values into a derived feature."""

    data_path: str = Field(
        default="data/input/synthetic_inforce.csv",
        description="Source raw input path for regrouping (alias of source_path).",
    )
    source_column: str = Field(
        ...,
        description="Categorical column name to regroup (for example Risk_Class).",
    )
    mapping_dict: Dict[str, Any] = Field(
        ...,
        description="Mapping dictionary used to regroup source values into broader categories.",
    )
    source_path: str = Field(
        default="data/input/synthetic_inforce.csv",
        description="Input raw path for regrouping (.csv, .parquet, or .xlsx).",
    )
    output_path: str = Field(
        default=CANONICAL_ANALYSIS_OUTPUT_PATH,
        description="Output prepared-analysis path to save transformed data.",
    )
    sheet_name: Optional[str] = Field(
        default=None,
        description="Optional worksheet name when source_path points to an .xlsx workbook.",
    )


class FilterClauseSchema(BaseModel):
    """Schema for a single structured sweep filter."""

    column: str = Field(..., description="Dataset column name to filter on.")
    operator: Literal["=", "!=", ">", ">=", "<", "<="] = Field(
        ...,
        description="Scalar comparison operator.",
    )
    value: str | int | float = Field(
        ...,
        description="Scalar filter value to compare against.",
    )


class DimensionalSweepSchema(BaseModel):
    """Schema for actuarial dimensional sweep and Bayesian A/E outputs."""

    depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Combination depth for cohort intersections (1-3). "
            "Tool returns a JSON list of objects; each object includes: "
            "Dimensions, Sum_MAC, Sum_MEC, Sum_MAF, Sum_MEF, AE_Ratio_Count, "
            "AE_Ratio_Amount, AE_Count_95_Lower, AE_Count_95_Upper, "
            "AE_Amount_95_Lower, and AE_Amount_95_Upper."
        ),
    )
    min_mac: int = Field(
        default=0,
        ge=0,
        description=(
            "Visibility floor: only include cohorts with Sum_MAC >= min_mac. "
            "Output is a JSON list of cohort objects with Actual vs Expected counts/amounts and 95% CIs."
        ),
    )
    top_n: int = Field(
        default=20,
        ge=1,
        description=(
            "Maximum number of ranked cohort rows to return from the JSON list output."
        ),
    )
    sort_by: str = Field(
        default="AE_Ratio_Amount",
        description=(
            "Metric used to rank results (e.g., AE_Ratio_Amount, AE_Ratio_Count, Sum_MAF). "
            "Returned objects contain Dimensions (cohort definition such as 'Gender=F | Smoker=No'), "
            "Sum_MAC/Sum_MEC, Sum_MAF/Sum_MEF, AE_Ratio_Count/AE_Ratio_Amount, "
            "and 95% credible bounds for count and amount."
        ),
    )
    filters: List[FilterClauseSchema] = Field(
        default_factory=list,
        description=(
            "Structured scalar filters applied before aggregation. "
            "Each filter uses column/operator/value and all filters combine with logical AND."
        ),
    )
    selected_columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of dimension columns to sweep. When omitted, the tool auto-detects eligible dimensions."
        ),
    )
    data_path: str = Field(
        default=CANONICAL_ANALYSIS_OUTPUT_PATH,
        description="Prepared analysis dataset used for actuarial dimensional sweeps.",
    )


class VisualizationSchema(BaseModel):
    """Schema for visualization tool calls (scatter vs treemap)."""

    chart_type: Literal["scatter", "treemap"] = Field(
        ...,
        description="Visualization type to generate: 'scatter' (univariate A/E) or 'treemap' (hierarchical risk view).",
    )
    metric: Literal["count", "amount"] = Field(
        default="amount",
        description="Metric to visualize: 'count' uses MAC/MEC, 'amount' uses MAF/MEF.",
    )
    data_path: str = Field(
        default="data/output/sweep_summary.csv",
        description="Aggregated sweep CSV used as the visualization data source.",
    )
