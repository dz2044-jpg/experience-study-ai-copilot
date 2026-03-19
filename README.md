## Experience Study AI Copilot

Experience Study AI Copilot is a multi-agent system for actuaries and data scientists analyzing life insurance inforce data.  
It follows a **Muscles and Brains** architecture:

- **Muscles (Deterministic Tools):** Python functions perform all validation, aggregation, actuarial math, and credibility calculations.
- **Brains (AI Agents):** LLM agents orchestrate workflows, select tool calls, and explain results without performing raw actuarial computation.

This design ensures reproducibility, auditability, and actuarial control over all quantitative outputs.

---

## Actuarial Methodology

### Mortality Exposure Count (MOC) Logic

The platform supports fractional exposure and claim-state logic:

- **Active policy periods:** `MOC` can be fractional (e.g., partial-year exposure).
- **Death periods (`MAC = 1`):** `MOC` must be exactly `1.0`.
- Validation enforces:
  - `0 < MOC <= 1.0`
  - `MAC == 1  =>  MOC == 1.0`

This allows clean integration of exposure-based mortality studies while preserving actuarial integrity for claim rows.

### Bayesian Credibility (Jeffreys Prior)

Credible intervals are computed with a Beta posterior using **Jeffreys Prior**:

- Prior: `Beta(0.5, 0.5)`
- Posterior parameters:
  - `alpha = MAC + 0.5`
  - `beta = MOC - MAC + 0.5`
- 95% interval is derived from:
  - `stats.beta.ppf(0.025, alpha, beta)`
  - `stats.beta.ppf(0.975, alpha, beta)`

These intervals are propagated into A/E confidence bounds for both count and amount studies.

### Hybrid A/E by Amount Logic

For amount-based A/E, average claim severity is handled robustly:

- If `MAC > 0`: `avg_claim = MAF / MAC`
- If `MAC == 0`: `avg_claim = MEF / MEC` (expected severity fallback)

This prevents degenerate zero-claim cohorts from collapsing to unusable bounds and preserves a meaningful upper credible limit.

---

## Project Structure

```text
experience-study-ai-copilot/
├── agents/
│   ├── schemas.py              # Pydantic contracts between tools and AI agents
│   ├── agent_steward.py        # Data Steward agent (profiling/validation/banding)
│   ├── agent_actuary.py        # Lead Actuary agent (A/E sweep interpretation)
│   ├── agent_analyst.py        # Data Analyst agent (visualization & reporting)
│   └── orchestrator.py         # Supervisor that routes GENERAL / DATA_PREP / ANALYSIS / VISUALIZE / CONTINUE
├── data/
│   ├── input/
│   │   └── synthetic_inforce.csv   # synthetic actuarial dataset (raw inforce)
│   └── output/
│       ├── analysis_inforce.csv    # engineered dataset used for actuarial sweeps
│       └── sweep_summary.csv       # latest aggregated A/E summary used for visualization
├── tools/
│   ├── data_steward.py         # data validation + feature engineering
│   ├── insight_engine.py       # actuarial sweep engine + Bayesian A/E math core
│   └── visualization.py        # Plotly-based scatter/treemap visualization utilities
├── chat.py                     # terminal copilot entrypoint
├── main.py                     # thin Streamlit shell around the orchestrator
├── pyproject.toml
├── uv.lock
└── README.md
```

### Tool Roles

- **`tools/data_steward.py` — Sanitizer/Engineer**
  - Profiles raw inforce data (by default `data/input/synthetic_inforce.csv`)
  - Runs actuarial validation checks (types, value ranges, logic checks)
  - Creates categorical bands and regrouped features
  - Writes transformed output to `data/output/analysis_inforce.csv` (never overwriting raw source input)

- **`tools/insight_engine.py` — Actuarial Math Core**
  - Computes Bayesian mortality-rate credible intervals
  - Computes A/E confidence intervals (count and amount)
  - Executes dimensional sweeps across cohort intersections using the processed dataset (`data/output/analysis_inforce.csv`)
  - Returns structured JSON for downstream agent interpretation and UI rendering

### Agent Roles

- **`agents/agent_steward.py` — Data Steward Agent**
  - Uses schema-driven tool calling for:
    - dataset profiling
    - actuarial data checks (including MOC integrity)
    - categorical banding and regrouping for engineered analysis features
  - Confirms applied transformations and output paths

- **`agents/agent_actuary.py` — Lead Actuary Agent**
  - Calls `run_dimensional_sweep` via schema-bound tools
  - Compares `AE_Ratio_Amount` (financial risk) vs `AE_Ratio_Count` (selection risk)
  - Uses cautious language when 95% CIs are wide or span 1.0

- **`agents/agent_analyst.py` — Data Analyst Agent**
  - Uses `VisualizationSchema` with `chart_type` and `metric` to choose tools
  - Calls `generate_univariate_report` (scatter + table) or `generate_treemap_report`
  - Always reads from the aggregated sweep file (`data/output/sweep_summary.csv`) rather than raw inforce data
  - Returns concise confirmations that charts were generated and opened

- **`agents/orchestrator.py` — Study Supervisor**
  - Classifies requests into:
    - `GENERAL`
    - `DATA_PREP`
    - `ANALYSIS`
    - `VISUALIZE`
    - `CONTINUE` (follow-on handoff)
  - Routes to the Steward, Actuary, Analyst, or the next queued specialist step
  - Returns a consolidated natural-language response where appropriate

---

## Feature Highlights

### Multi-Level Sweeps

Supports cohort intersections at configurable depth:

- **1-way:** single-dimension analysis
- **2-way:** pairwise interactions (e.g., `Gender x Smoker`)
- **3-way:** higher-order interaction analysis

### Dynamic Sniffing

Automatically identifies candidate dimension columns by:

- object/string dtype, or
- numeric columns with `<= 20` unique values

Then excludes actuarial measure columns (`MAC`, `MOC`, `MEC`, `MAF`, `MEF`, etc.) from dimensioning.

### Visibility Floors (Credibility Filtering)

`min_mac` controls the minimum death count required for cohort visibility:

- Lower values for exploratory signal detection
- Higher values for credibility-focused, high-stability output

### Dynamic Ranking and Result Control

Dimensional sweep supports runtime ranking controls:

- `sort_by`: rank by `AE_Ratio_Amount`, `AE_Ratio_Count`, `Sum_MAF`, etc.
- `top_n`: control how many cohort rows are returned
- `filters`: apply pandas query filters before aggregation

---

## Setup

### 1) Sync environment

```bash
uv sync
```

### 2) Run Streamlit app

The current repository entrypoint is `main.py`:

```bash
uv run streamlit run main.py
```

### 3) Run Agent Layer (optional CLI tests)

Run each agent module directly to validate tool integration and orchestration:

```bash
uv run python agents/agent_steward.py
uv run python agents/agent_actuary.py
uv run python agents/orchestrator.py
```

Or run the continuous terminal copilot:

```bash
uv run python chat.py
```

### 4) Configure OpenAI access

Agents use the official OpenAI Python SDK with dotenv-based configuration.  
Set your key once in `.env`:

```env
OPENAI_API_KEY=sk-...
```

---

## Output Contract (Dimensional Sweep)

The actuarial sweep returns JSON records containing cohort definition, actual/expected values, and Bayesian confidence bounds:

- `Dimensions`
- `Sum_MAC`, `Sum_MEC`
- `Sum_MAF`, `Sum_MEF`
- `AE_Ratio_Count`, `AE_Ratio_Amount`
- Count CI lower/upper and Amount CI lower/upper (95%)

Typical sweep payload shape:

```json
{
  "results": [
    {
      "Dimensions": "Risk_Class=Standard Plus | Issue_Age_band=(45.0, 65.0]",
      "Sum_MAC": 6,
      "Sum_MOC": 481.67,
      "Sum_MEC": 1.97,
      "Sum_MAF": 4300000.0,
      "Sum_MEF": 1338984.63,
      "AE_Ratio_Count": 3.63,
      "AE_Ratio_Amount": 3.21,
      "AE_Count_CI": [1.52, 7.44],
      "AE_Amount_CI": [1.34, 6.57]
    }
  ]
}
```

This contract is defined in `agents/schemas.py` and consumed by AI agents for deterministic, auditable analysis workflows.
