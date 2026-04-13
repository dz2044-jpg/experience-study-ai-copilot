from skills.experience_study_skill.schemas import (
    CreateCategoricalBandsInput,
    FilterClauseInput,
    ProfileDatasetInput,
    RunDimensionalSweepInput,
)


def test_high_risk_schema_descriptions_include_negative_constraints():
    profile_schema = ProfileDatasetInput.model_json_schema()
    sweep_schema = RunDimensionalSweepInput.model_json_schema()
    filter_schema = FilterClauseInput.model_json_schema()
    band_schema = CreateCategoricalBandsInput.model_json_schema()

    assert "DO NOT invent file paths" in profile_schema["properties"]["data_path"]["description"]
    assert "DO NOT invent column names" in sweep_schema["properties"]["selected_columns"]["description"]
    assert "NEVER select a metric" in sweep_schema["properties"]["sort_by"]["description"]
    assert "DO NOT invent file paths" in sweep_schema["properties"]["data_path"]["description"]
    assert "DO NOT invent column names" in filter_schema["properties"]["column"]["description"]
    assert "DO NOT invent column names" in band_schema["properties"]["source_column"]["description"]
