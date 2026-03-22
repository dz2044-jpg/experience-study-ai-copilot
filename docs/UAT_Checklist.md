# Experience Study AI Copilot: Master UAT Checklist

This document defines the manual user-acceptance testing protocol for the multi-agent pipeline:

- Orchestrator
- Data Steward
- Lead Actuary
- Data Analyst

The goal is to validate four things:

- domain boundaries stay intact
- conversation state is handled correctly
- actuarial math comes from deterministic Python tools rather than LLM invention
- pipeline handoffs create the expected disk artifacts

## Test Setup

Run the CLI copilot from the project root:

```bash
cd /Users/amberwang/Desktop/dzw/experience-study-ai-copilot
python chat.py
```

If your virtual environment is not already active:

```bash
cd /Users/amberwang/Desktop/dzw/experience-study-ai-copilot
.venv/bin/python chat.py
```

Core files to inspect during UAT:

- `data/input/synthetic_inforce.csv`
- `data/output/analysis_inforce.parquet`
- `data/output/sweep_summary.csv`
- `data/output/temp_univariate_report.html`
- `data/output/temp_treemap_report.html`

## Pass / Fail Recording Template

Use this simple format while testing:

```text
Test ID:
Prompt(s):
Expected:
Actual:
Status: PASS / FAIL
Notes:
```

## Phase 1: Data Integrity & State Management

### Test 1: The Hard Schema Check

- Action:
  Prompt the Copilot:
  `Profile the raw synthetic inforce data and list the data types for MAC, MEC, MAF, and MEF.`
- Expected Pass:
  The Data Steward explicitly identifies `MAC`, `MEC`, `MAF`, and `MEF` as numerical fields, typically `float64`.
  It must not misclassify them as boolean or categorical just because many rows contain `0`.

### Test 2: The Session-Stacking Check

- Action:
  Prompt the Copilot:
  `Group Face_Amount into 4 equal-width bands, and then group Issue_Age into 4 equal-width bands.`
- Expected Pass:
  Inspect `data/output/analysis_inforce.parquet`.
  Both `Face_Amount_band` and `Issue_Age_band` must exist.
  The second feature-engineering step must append to the current session output instead of overwriting the first derived column.

### Test 3: The Null Preservation Check

- Action:
  Prompt the Copilot:
  `Check the data for missing values and summarize them.`
- Expected Pass:
  The Steward reports missing values in `COLA`, but explains that null `COLA` values are expected for non-claim rows.
  It must not silently drop rows or describe the null pattern as a generic data corruption issue.

### Test 3A: The Multi-Format Input Check

- Action:
  Prompt the Copilot using explicit raw input paths, for example:
  `Profile data/input/example.parquet`
  and
  `Profile data/input/example.xlsx`
- Expected Pass:
  The Steward reads both formats successfully without requiring CSV conversion.
  For `.xlsx`, it should read the default worksheet unless a specific sheet is provided through a steward tool call.

## Phase 2: Orchestration & Workflow Guardrails

### Test 4: The Guardrail Check

- Action:
  Prompt the Copilot:
  `Check the data for errors and then run a 1-way sweep on Gender.`
- Expected Pass:
  The Orchestrator should stop after the Steward step and queue the next action rather than chaining directly into analysis without user permission.
  The user should still control whether the next step runs.

### Test 5: The Continuation Routing Check

- Action:
  After Test 4 pauses at the Steward step, type:
  `proceed`
  or
  `yes`
- Expected Pass:
  The Orchestrator classifies the reply as `CONTINUE`, remembers the pending next step, and routes into the Actuary flow.
  It must not fall back to a generic message such as “please specify data prep or analysis.”

### Test 6: The Continuation-to-Visualization Check

- Action:
  After a successful actuarial sweep, type:
  `proceed`
- Expected Pass:
  The Orchestrator routes into the Analyst step and attempts to generate a visualization from the latest `data/output/sweep_summary.csv`.
  A report file should be created even if browser auto-open fails in the environment.

## Phase 3: Actuarial Math & Verification

### Test 7: The Actuarial Dictionary Check

- Action:
  Prompt the Copilot:
  `Run a 1-way dimensional sweep on Gender to calculate the A/E ratio by Count and Amount.`
- Expected Pass:
  The Lead Actuary executes the sweep immediately.
  It must not ask the user to define `MAC`, `MEC`, `MAF`, `MEF`, or the A/E formulas.

### Test 8: The Pairwise 2-Way Sweep Check

- Action:
  Prompt the Copilot:
  `Run 2-way dimensional sweeps for all pairs between Smoker, Risk_Class, and Issue_Age_band, then rank the results by AE_Ratio_Amount.`
- Expected Pass:
  The Actuary interprets this as the three pairwise combinations:
  `Smoker x Risk_Class`, `Smoker x Issue_Age_band`, and `Risk_Class x Issue_Age_band`.
  It must update `data/output/sweep_summary.csv` on disk with the latest ranked 2-way results.

### Test 9: The Physical Artifact Check

- Action:
  After Test 8 completes, inspect:
  `data/output/sweep_summary.csv`
- Expected Pass:
  The file exists and contains the latest sweep output rather than stale data from a prior run.
  It should include flattened CI columns required by visualization, such as:
  `AE_Count_CI_Lower`, `AE_Count_CI_Upper`, `AE_Amount_CI_Lower`, and `AE_Amount_CI_Upper`.

### Test 9A: Filter Validation - Numeric

- Action:
  Prompt the Copilot:
  `Run a 1-way sweep on Gender where Duration < 5.`
- Expected Pass:
  The sweep completes successfully.
  The results must reflect only rows where `Duration < 5`.
  The terminal session must not crash or emit a traceback.

### Test 9B: Filter Validation - String

- Action:
  Prompt the Copilot:
  `Run a 1-way sweep on Gender where Smoker = Yes.`
- Expected Pass:
  The sweep completes successfully.
  The results must reflect only rows where `Smoker` equals `Yes`.
  The terminal session must not crash or emit a traceback.

### Test 9C: Filter Validation - Invalid Column

- Action:
  Prompt the Copilot:
  `Run a 1-way sweep on Gender where State = California.`
- Expected Pass:
  The Copilot politely rejects the request with a controlled missing-column explanation.
  The response should indicate that `State` is unavailable in the prepared dataset and should not expose a Python traceback.

## Phase 4: Error Handling & Resilience

### Test 10: The Bad Feature Check

- Action:
  Prompt the Copilot:
  `Run a sweep on the column NonExistent_Status.`
- Expected Pass:
  The tool returns a controlled error to the agent, and the user sees a graceful explanation that the column does not exist in the current dataset.
  The terminal session must not crash.

### Test 11: The Missing Visualization Input Check

- Action:
  Remove or rename `data/output/sweep_summary.csv`, then prompt:
  `Create a treemap using the latest sweep summary.`
- Expected Pass:
  The Analyst fails gracefully with a clear explanation that the summary file is missing or unavailable.
  It must not hang indefinitely or produce a misleading “success” message.

### Test 12: The Visualization Handoff Check

- Action:
  Prompt the Copilot:
  `Generate a treemap of the 2-way sweep we just ran.`
- Expected Pass:
  The Analyst reads `data/output/sweep_summary.csv`, parses the `Dimensions` strings, and writes a treemap artifact to:
  `data/output/temp_treemap_report.html`
  If browser auto-open is blocked by the environment, file creation still counts as the required artifact-level success.

## Known Caveats to Record During UAT

Document these explicitly if they appear:

- the orchestrator misroutes analysis requests as data prep
- the browser-open step fails even though the HTML file is generated
- the agent responds with a generic help message instead of using the pending continuation state
- a later tool call overwrites `analysis_inforce.parquet` incorrectly and drops previously engineered columns

## Recommended UAT Run Order

Use this order when testing the repo manually:

1. Test 1
2. Test 2
3. Test 3
4. Test 4
5. Test 5
6. Test 7
7. Test 8
8. Test 9
9. Test 12
10. Test 10
11. Test 11

## Exit Criteria

The build is ready for broader review only if all of the following are true:

- Steward data profiling and validation behave deterministically
- multi-step feature engineering preserves prior engineered columns in the current session
- continuation prompts reliably advance the queued workflow step
- actuarial sweeps update `data/output/sweep_summary.csv`
- analyst visualizations are generated from the latest sweep artifact
- invalid inputs fail gracefully without crashing the CLI
