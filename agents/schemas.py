"""Pydantic schemas defining tool-call contracts for AI agents."""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ProfileDatasetSchema(BaseModel):
    """Schema for profiling raw inforce data."""

    data_path: str = Field(
        default="data/uploaded_inforce.csv",
        description="Path to the source CSV file to profile.",
    )


class FeatureEngineeringSchema(BaseModel):
    """Schema for creating bands or regrouping categories."""

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
        default="data/uploaded_inforce.csv",
        description="Input CSV path for transformation.",
    )
    output_path: str = Field(
        default="data/analysis_inforce.csv",
        description="Output CSV path to save transformed data.",
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
        default=1,
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
    filters: List[str] = Field(
        default_factory=list,
        description=(
            "Pandas query filters applied before aggregation (e.g., [\"Gender == 'F'\", \"Smoker == 'No'\"]). "
            "The tool returns JSON objects where each object represents one cohort with "
            "Dimensions, actual/expected counts and amounts, A/E ratios, and 95% credible interval bounds."
        ),
    )
