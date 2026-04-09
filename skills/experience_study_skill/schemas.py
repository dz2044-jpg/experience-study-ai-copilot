"""Pydantic tool contracts for the experience study skill."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProfileDatasetInput(BaseModel):
    """Profile a raw or prepared tabular dataset."""

    data_path: str = Field(
        ...,
        description="Path to a supported source file (.csv, .parquet, or .xlsx).",
    )
    sheet_name: str | None = Field(
        default=None,
        description="Optional worksheet name when data_path points to an XLSX workbook.",
    )


class InspectDatasetSchemaInput(BaseModel):
    """Inspect the ordered columns and data types for a supported tabular dataset."""

    data_path: str | None = Field(
        default=None,
        description=(
            "Optional source path. When omitted, the current session's prepared dataset "
            "is inspected first, then the raw input dataset."
        ),
    )
    sheet_name: str | None = Field(
        default=None,
        description="Optional worksheet name when data_path points to an XLSX workbook.",
    )


class ActuarialDataChecksInput(BaseModel):
    """Run actuarial validation checks against a dataset."""

    data_path: str | None = Field(
        default=None,
        description=(
            "Optional supported source path. When omitted, the current session's prepared "
            "dataset is used."
        ),
    )
    sheet_name: str | None = Field(
        default=None,
        description="Optional worksheet name when data_path points to an XLSX workbook.",
    )


class CreateCategoricalBandsInput(BaseModel):
    """Create categorical bands for a numeric source column."""

    source_column: str = Field(
        ...,
        description="Numeric column to band, such as Issue_Age or Face_Amount.",
    )
    strategy: Literal["quantiles", "equal_width", "custom"] = Field(
        ...,
        description="Banding strategy to use for the source column.",
    )
    bins: int | None = Field(
        default=None,
        ge=1,
        description="Number of bins for quantiles or equal_width strategies.",
    )
    custom_bins: list[float] | None = Field(
        default=None,
        description="Custom bin edges for strategy='custom'.",
    )
    data_path: str | None = Field(
        default=None,
        description=(
            "Optional explicit source path. When omitted, the current session's prepared "
            "dataset is updated in place."
        ),
    )
    sheet_name: str | None = Field(
        default=None,
        description="Optional worksheet name when data_path points to an XLSX workbook.",
    )


class RegroupCategoricalFeaturesInput(BaseModel):
    """Regroup source categorical values into a derived feature."""

    source_column: str = Field(
        ...,
        description="Categorical column to regroup, such as Risk_Class.",
    )
    mapping_dict: dict[str, Any] = Field(
        ...,
        description="Mapping dictionary used to regroup source values into broader categories.",
    )
    data_path: str | None = Field(
        default=None,
        description=(
            "Optional explicit source path. When omitted, the current session's prepared "
            "dataset is updated in place."
        ),
    )
    sheet_name: str | None = Field(
        default=None,
        description="Optional worksheet name when data_path points to an XLSX workbook.",
    )


class FilterClauseInput(BaseModel):
    """Structured scalar filter applied before aggregation."""

    column: str = Field(..., description="Dataset column name to filter on.")
    operator: Literal["=", "!=", ">", ">=", "<", "<="] = Field(
        ...,
        description="Scalar comparison operator.",
    )
    value: str | int | float = Field(
        ...,
        description="Scalar filter value used in the comparison.",
    )


class RunDimensionalSweepInput(BaseModel):
    """Run a dimensional sweep using the prepared analysis dataset."""

    depth: int = Field(
        default=1,
        ge=1,
        le=3,
        description="Dimensional interaction depth to evaluate.",
    )
    min_mac: int = Field(
        default=0,
        ge=0,
        description="Visibility floor requiring Sum_MAC >= min_mac.",
    )
    top_n: int = Field(
        default=20,
        ge=1,
        le=20,
        description="Maximum number of ranked cohort rows returned in the response payload.",
    )
    sort_by: Literal[
        "AE_Ratio_Amount",
        "AE_Ratio_Count",
        "Sum_MAC",
        "Sum_MOC",
        "Sum_MEC",
        "Sum_MAF",
        "Sum_MEF",
    ] = Field(
        default="AE_Ratio_Amount",
        description="Metric used to rank the resulting cohorts.",
    )
    filters: list[FilterClauseInput] = Field(
        default_factory=list,
        description="Structured scalar filters applied before aggregation.",
    )
    selected_columns: list[str] | None = Field(
        default=None,
        description="Optional list of explicit sweep dimension columns.",
    )
    data_path: str | None = Field(
        default=None,
        description=(
            "Optional prepared analysis path. When omitted, the current session's prepared "
            "dataset is used."
        ),
    )


class GenerateVisualizationInput(BaseModel):
    """Generate the combined A/E report from a sweep summary CSV."""

    metric: Literal["count", "amount"] = Field(
        default="amount",
        description="Metric to visualize: amount or count.",
    )
    data_path: str | None = Field(
        default=None,
        description=(
            "Optional sweep summary CSV path. When omitted, the current session's latest "
            "sweep artifact is used."
        ),
    )


_TOOL_MODELS: dict[str, tuple[str, type[BaseModel]]] = {
    "profile_dataset": (
        "Profile a supported source dataset and create the session-local prepared dataset.",
        ProfileDatasetInput,
    ),
    "inspect_dataset_schema": (
        "Inspect the ordered columns and data types for a supported tabular dataset.",
        InspectDatasetSchemaInput,
    ),
    "run_actuarial_data_checks": (
        "Run deterministic actuarial validation checks on a dataset.",
        ActuarialDataChecksInput,
    ),
    "create_categorical_bands": (
        "Create categorical bands for a numeric source column and save the prepared dataset.",
        CreateCategoricalBandsInput,
    ),
    "regroup_categorical_features": (
        "Regroup source categorical values into a derived analysis feature.",
        RegroupCategoricalFeaturesInput,
    ),
    "run_dimensional_sweep": (
        "Run a deterministic dimensional sweep with A/E ratios and credibility intervals.",
        RunDimensionalSweepInput,
    ),
    "generate_combined_report": (
        "Generate the combined visualization report from a sweep summary CSV.",
        GenerateVisualizationInput,
    ),
}


def get_tool_specs(enabled_tools: set[str] | None = None) -> list[dict[str, Any]]:
    """Return OpenAI tool definitions for the enabled subset."""
    tool_names = enabled_tools or set(_TOOL_MODELS)
    specs: list[dict[str, Any]] = []
    for tool_name, (description, model) in _TOOL_MODELS.items():
        if tool_name not in tool_names:
            continue
        specs.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": model.model_json_schema(),
                },
            }
        )
    return specs
