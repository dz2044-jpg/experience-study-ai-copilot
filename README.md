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
│   └── schemas.py              # Pydantic contracts between tools and AI agents
├── data/
│   ├── synthetic_inforce.csv   # synthetic actuarial dataset
│   └── analysis_inforce.csv    # engineered dataset used for actuarial sweeps
├── tools/
│   ├── data_steward.py         # data validation + feature engineering
│   └── insight_engine.py       # actuarial sweep engine + Bayesian A/E math core
├── main.py                     # current app/entry script
├── pyproject.toml
├── uv.lock
└── README.md
```

### Tool Roles

- **`tools/data_steward.py` — Sanitizer/Engineer**
  - Profiles uploaded inforce data
  - Runs actuarial validation checks (types, value ranges, logic checks)
  - Creates categorical bands and regrouped features
  - Writes transformed output to `data/analysis_inforce.csv` (never overwriting raw source input)

- **`tools/insight_engine.py` — Actuarial Math Core**
  - Computes Bayesian mortality-rate credible intervals
  - Computes A/E confidence intervals (count and amount)
  - Executes dimensional sweeps across cohort intersections
  - Returns structured JSON for downstream agent interpretation and UI rendering

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

---

## Setup

### 1) Sync environment

```bash
uv sync
```

### 2) Run Streamlit app

If your app entrypoint is `app.py`:

```bash
uv run streamlit run app.py
```

If your current entrypoint is `main.py` (as in this repository state):

```bash
uv run streamlit run main.py
```

---

## Output Contract (Dimensional Sweep)

The actuarial sweep returns JSON records containing cohort definition, actual/expected values, and Bayesian confidence bounds:

- `Dimensions`
- `Sum_MAC`, `Sum_MEC`
- `Sum_MAF`, `Sum_MEF`
- `AE_Ratio_Count`, `AE_Ratio_Amount`
- Count CI lower/upper and Amount CI lower/upper (95%)

This contract is defined in `agents/schemas.py` and consumed by AI agents for deterministic, auditable analysis workflows.

