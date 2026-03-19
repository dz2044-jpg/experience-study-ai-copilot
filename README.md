# Experience Study AI Copilot

Experience Study AI Copilot is a multi-agent actuarial analysis system for actuaries, data scientists, and technical stakeholders working with life insurance inforce data.

It is built around a strict separation of responsibilities:

- Deterministic Python tools perform validation, feature engineering, aggregation, Bayesian credibility, and A/E calculations.
- AI agents decide which deterministic tool to call, manage workflow handoffs, and explain results in natural language.

That split keeps the math reproducible and auditable while still giving users a flexible conversational interface.

How to open the stakeholder deck:
- Open [`docs/stakeholder_presentation.html`](docs/stakeholder_presentation.html) directly in a browser. No server, package install, or external assets are required.

## What This Repo Is For

This repository is for teams who want to:

- profile and validate raw inforce data before analysis
- engineer analysis-ready cohort features such as age or face-amount bands
- run dimensional A/E sweeps with deterministic actuarial math
- generate browser-based visual reports from the latest sweep artifact
- demonstrate that LLMs are orchestrating work, not inventing actuarial results

The current implementation supports a thin Streamlit chat shell, a CLI copilot flow, deterministic actuarial tooling, and a supervisor that routes work across specialist agents.

## Why The Architecture Is Split

The project follows a "Muscles and Brains" pattern.

### Muscles: Deterministic Tools

The `tools/` layer owns all numerical and data-manipulation work:

- [`tools/data_steward.py`](tools/data_steward.py) profiles datasets, runs actuarial validation checks, and creates engineered analysis features
- [`tools/insight_engine.py`](tools/insight_engine.py) runs dimensional sweeps and computes Bayesian A/E confidence intervals
- [`tools/visualization.py`](tools/visualization.py) turns sweep outputs into interactive HTML reports

These functions are the source of truth for actuarial math and file artifacts.

### Brains: AI Agents

The `agents/` layer wraps those tools with schema-bound tool calling:

- [`agents/agent_steward.py`](agents/agent_steward.py) handles data profiling, validation, and feature engineering
- [`agents/agent_actuary.py`](agents/agent_actuary.py) runs and interprets dimensional sweeps
- [`agents/agent_analyst.py`](agents/agent_analyst.py) generates visualization artifacts from sweep summaries
- [`agents/orchestrator.py`](agents/orchestrator.py) classifies intent, routes the request, and manages continuation-based handoffs

Pydantic contracts in [`agents/schemas.py`](agents/schemas.py) define the tool-call interfaces used by the agents.

## End-To-End Workflow

The system is designed as a staged workflow rather than an unrestricted agent chain.

### 1. Data Prep

The Data Steward reads a source CSV, profiles it, validates actuarial rules, and writes engineered output to:

- `data/output/analysis_inforce.csv`

Guardrails already enforced in code:

- raw input under `data/input/` is not overwritten
- null rows are not silently dropped or imputed
- actuarial checks validate MAC, MEC, MOC, COLA, duplicate exposure rows, and death-exposure continuity rules

### 2. Actuarial Sweep

The Lead Actuary runs a deterministic dimensional sweep against:

- `data/output/analysis_inforce.csv`

The sweep ranks cohorts and writes CSV artifacts to `data/output/`, including:

- a dynamically named sweep summary for the exact run
- `data/output/sweep_summary.csv` as the latest alias used by downstream visualization

### 3. Visualization

The Analyst reads the latest sweep artifact and generates browser-openable HTML outputs such as:

- `data/output/temp_univariate_report.html`
- `data/output/temp_treemap_report.html`

Visualization depends on a fresh sweep summary. If no recent analysis artifact exists, the orchestrator returns a controlled message instead of pretending the chart succeeded.

### 4. Continue / Handoff Behavior

The orchestrator intentionally pauses after major stages and queues the next sensible step:

- after data prep, `continue` routes to actuarial analysis
- after analysis, `continue` routes to visualization
- continuation is session-aware and backed by tested pending-state logic

This preserves user control while still enabling a guided multi-step workflow.

## System Map

```text
User
  |
  v
StudyOrchestrator
  |-- DATA_PREP  --> DataStewardAgent --> tools/data_steward.py --> data/output/analysis_inforce.csv
  |-- ANALYSIS   --> ActuaryAgent     --> tools/insight_engine.py --> data/output/sweep_summary*.csv
  |-- VISUALIZE  --> AnalystAgent     --> tools/visualization.py --> data/output/*.html
  |
  `-- CONTINUE   --> pending next step based on prior artifact + session state
```

Important implementation boundaries:

- the orchestrator routes; it does not perform actuarial math
- the steward can engineer features but does not run sweeps
- the actuary can run sweeps but does not create missing feature columns
- the analyst visualizes aggregated sweep output rather than raw inforce data

## Project Structure

```text
experience-study-ai-copilot/
├── agents/
│   ├── schemas.py
│   ├── agent_steward.py
│   ├── agent_actuary.py
│   ├── agent_analyst.py
│   └── orchestrator.py
├── tools/
│   ├── data_steward.py
│   ├── insight_engine.py
│   └── visualization.py
├── data/
│   ├── input/
│   └── output/
├── docs/
│   ├── UAT_Checklist.md
│   └── stakeholder_presentation.html
├── chat.py
├── main.py
└── tests/
```

## Setup And Run

### Install Dependencies

```bash
uv sync
```

### Configure OpenAI Access

Agents use the OpenAI Python SDK when `OPENAI_API_KEY` is available. Create a `.env` file with:

```env
OPENAI_API_KEY=sk-...
```

If the key is unavailable, parts of the system fall back to deterministic routing behavior where implemented.

### Run The Streamlit App

```bash
uv run streamlit run main.py
```

### Run The CLI Copilot

```bash
uv run python chat.py
```

### Run Focused Agent Entry Points

```bash
uv run python agents/agent_steward.py
uv run python agents/agent_actuary.py
uv run python agents/orchestrator.py
```

### Run Tests

```bash
uv run pytest
```

## Output Artifacts

The repo uses file artifacts as explicit handoff points between stages.

### Prepared Analysis Dataset

- `data/output/analysis_inforce.csv`
- engineered dataset created by the steward
- input to the actuarial sweep engine

### Sweep Summaries

- `data/output/sweep_summary.csv`
- `data/output/sweep_summary_<...>.csv`

These files contain ranked cohort outputs with:

- `Dimensions`
- `Sum_MAC`, `Sum_MOC`, `Sum_MEC`
- `Sum_MAF`, `Sum_MEF`
- `AE_Ratio_Count`, `AE_Ratio_Amount`
- `AE_Count_CI_Lower`, `AE_Count_CI_Upper`
- `AE_Amount_CI_Lower`, `AE_Amount_CI_Upper`

### Visualization Outputs

- `data/output/temp_univariate_report.html`
- `data/output/temp_treemap_report.html`
- temporary filtered sources such as `data/output/temp_treemap_source_<...>.csv`

## Sweep Controls And Output Contract

The dimensional sweep interface is defined in [`agents/schemas.py`](agents/schemas.py) and executed by [`tools/insight_engine.py`](tools/insight_engine.py).

Supported runtime controls include:

- `depth`: 1-way, 2-way, or explicit 3-way cohort intersections
- `selected_columns`: constrain the sweep to named dimensions
- `filters`: apply pandas-query style row filters before aggregation
- `min_mac`: visibility floor for minimum actual death count
- `top_n`: limit the JSON response while keeping full ranked CSV artifacts on disk
- `sort_by`: rank by `AE_Ratio_Amount`, `AE_Ratio_Count`, or supported aggregate fields

Typical result shape:

```json
{
  "results": [
    {
      "Dimensions": "Gender=F | Smoker=Yes",
      "Sum_MAC": 9,
      "Sum_MOC": 71.45,
      "Sum_MEC": 3.75,
      "Sum_MAF": 2100000.0,
      "Sum_MEF": 1160000.0,
      "AE_Ratio_Count": 2.4,
      "AE_Ratio_Amount": 1.8,
      "AE_Count_CI_Lower": 1.1,
      "AE_Count_CI_Upper": 3.6,
      "AE_Amount_CI_Lower": 0.9,
      "AE_Amount_CI_Upper": 2.7
    }
  ],
  "output_path": "data/output/sweep_summary_1_gender_smoker.csv",
  "latest_output_path": "data/output/sweep_summary.csv"
}
```

## Methodology And Trust

The repo is opinionated about actuarial control.

### Deterministic Math Guardrails

- LLMs do not compute A/E ratios, exposures, or confidence intervals themselves
- all quantitative results come from deterministic Python functions
- agents are instructed to explain tool outputs, not replace them

### MOC Integrity Rules

- active policy exposure may be fractional
- `0 < MOC <= 1.0`
- if `MAC == 1`, then `MOC` must be exactly `1.0`

### Bayesian Credibility

Mortality-rate credibility uses a Jeffreys-prior Beta posterior:

- prior: `Beta(0.5, 0.5)`
- posterior parameters: `alpha = MAC + 0.5`, `beta = MOC - MAC + 0.5`
- 95% credible intervals propagate into count-based and amount-based A/E bounds

### Amount-Based A/E Fallback

For amount-based A/E intervals:

- if `MAC > 0`, average claim severity uses `MAF / MAC`
- if `MAC == 0`, the fallback uses `MEF / MEC`

That prevents zero-claim cohorts from collapsing into unusable amount-based bounds.

## Testing And UAT

The current test suite covers the behaviors this README describes, including:

- continuation routing from steward to actuary to analyst
- deterministic sweep artifact creation and latest-output aliasing
- visualization dependence on fresh analysis artifacts
- schema exposure for selected sweep dimensions
- handling of missing engineered columns

Run the automated suite with:

```bash
uv run pytest
```

For manual validation, use the UAT checklist:

- [`docs/UAT_Checklist.md`](docs/UAT_Checklist.md)

For stakeholder walkthroughs, open:

- [`docs/stakeholder_presentation.html`](docs/stakeholder_presentation.html)
