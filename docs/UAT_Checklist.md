# Experience Study AI Copilot: Master UAT Checklist

This document contains the standard testing protocol to ensure the multi-agent pipeline (Orchestrator, Steward, Actuary, Analyst) maintains strict domain boundaries, manages state correctly, and calculates actuarial math without hallucinations.

## Phase 1: Data Integrity & State Management

### Test 1: The Hard Schema & Aggregation Check (Domain Blindness)
* **Action:** Prompt the Copilot: *"Profile the raw synthetic inforce data. Give me the total unique policy count, the sum of all actual claims (MAF), and list the exact data types for MAC, MEC, MAF, and MEF."*
* **Expected Pass:** 1. The Steward reports the correct number of unique policies.
    2. It accurately calculates the sum of all claims (`MAF`) without throwing a type error.
    3. It explicitly identifies `MAC`, `MEC`, `MAF`, and `MEF` as numerical (`float64`). It must **not** flag the high concentration of `0`s in `MAC` or `MAF` as a categorical anomaly.

### Test 2: The Session-Stacking Check (File Overwrite Bug)
* **Action:** Prompt: *"Group Face_Amount into 4 percentile bands, and then group Issue_Age into 4 percentile bands."*
* **Expected Pass:** Inspect `data/output/analysis_inforce.csv`. Both `Face_Amount_band` and `Issue_Age_band` must exist at the end of the columns. The second tool call must append to, not overwrite, the first tool call's output.

### Test 3: The Null Preservation Check
* **Action:** Prompt: *"Check the data for missing values and summarize them."*
* **Expected Pass:** The Steward notes nulls in the `COLA` (Cause of Death) column but specifically identifies this as expected actuarial behavior for active policies. It does **not** auto-drop these rows.

## Phase 2: Orchestration & Workflow Guardrails

### Test 4: The Guardrail Check (The Runaway Actuary)
* **Action:** Prompt: *"Check the data for errors and then run a 1-way sweep on Gender."*
* **Expected Pass:** The Orchestrator halts the pipeline after the Data Steward finishes the error check. It requires explicit user permission before routing the next step to the Lead Actuary.

### Test 5: The Continuation Routing (The "Proceed" Bug)
* **Action:** Following a pause in Test 4, type: *"proceed"* (or *"yes"*).
* **Expected Pass:** The Orchestrator correctly classifies this as a `CONTINUE` intent, remembers the context, and executes the sweep without throwing a generic "Please specify DATA_PREP or ANALYSIS" error.

## Phase 3: Actuarial Math & Verification

### Test 6: The Actuarial Dictionary Check (Clarification Loop)
* **Action:** Prompt: *"Run a 1-way dimensional sweep on Gender to calculate the A/E ratio by Count and Amount."*
* **Expected Pass:** The Lead Actuary executes the calculation immediately. It does not pause to ask the user to define `MEC`, `MAC`, or the A/E formula.

### Test 7: The Pairwise 2-Way Sweep Check (Iterative State)
* **Action:** Prompt: *"Run 2-way dimensional sweeps for all pairs between Smoker, Risk_Class, and Issue_Age_band, and output them as one combined ranked table."*
* **Expected Pass:** 1. The Lead Actuary correctly recognizes this as three distinct pairs (Smoker × Risk_Class, Smoker × Age, Risk_Class × Age).
    2. It iteratively calculates all three pairs without hitting a context window limit.
    3. `data/output/sweep_summary.csv` is physically updated to cleanly hold this combined, ranked pairwise data.

## Phase 4: Error Handling & Resilience

### Test 8: The "Bad Feature" Check (Graceful Failure)
* **Action:** Prompt: *"Run a sweep on the column 'NonExistent_Status'."*
* **Expected Pass:** The Python tool catches the `KeyError`, returns the traceback to the LLM as text, and the Agent responds: *"That column doesn't exist in the current dataset,"* rather than the terminal crashing completely.

### Test 9: The Visualization Handoff
* **Action:** Prompt: *"Generate a Treemap of the pairwise sweep we just ran in Test 7."*
* **Expected Pass:** The Analyst Agent successfully reads from `data/output/sweep_summ
