# Data Privacy

Experience Study AI Copilot should minimize what reaches any LLM-backed interpretation or drafting layer.

The LLM layer should receive only aggregated, sanitized, cohort-level packets derived from deterministic artifacts. It must not receive raw row-level data.

## Data That Must Not Be Sent To The LLM

- Policy-level rows.
- Claim-level rows.
- Raw application-level records.
- Raw claim-level records.
- `Policy_Number`.
- Names.
- DOBs.
- Addresses.
- Emails.
- Phone numbers.
- SSNs.
- Any unknown or unapproved source columns.

## Sanitization Standard

AI packets should use allowlist-based sanitization. A denylist alone is not sufficient because sensitive fields can appear under unexpected column names.

Allowed fields should be limited to aggregated cohort-level metrics and workflow metadata needed for interpretation, such as dimensions, selected columns, filters, A/E ratios, confidence intervals, sweep depth, ranking metric, and minimum credibility thresholds.

## Rare Cohorts

Rare cohorts should be masked or suppressed before AI interpretation when volume or credibility is too low. Low-credibility results should be flagged with cautionary language and should not be over-interpreted.

## Audit Expectation

Future AI packet builders should be test-covered to prove that raw row-level data and sensitive identifiers do not reach the LLM packet. Privacy must be enforced by code and tests, not only by documentation.
