"""A/E confidence interval math for the experience study skill."""

from __future__ import annotations

import pandas as pd
from scipy import stats


def _compute_mortality_rate_ci(
    mac: float,
    moc: float,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    if pd.isna(mac) or pd.isna(moc) or moc <= 0 or mac < 0 or mac > moc:
        return (None, None)
    alpha_beta = mac + 0.5
    beta_beta = moc - mac + 0.5
    lower_quantile = (1 - confidence_level) / 2
    upper_quantile = 1 - lower_quantile
    lower_rate = stats.beta.ppf(lower_quantile, alpha_beta, beta_beta)
    upper_rate = stats.beta.ppf(upper_quantile, alpha_beta, beta_beta)
    return (lower_rate, upper_rate)


def compute_ae_ci(
    mac: float,
    moc: float,
    mec: float,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    rate_lower, rate_upper = _compute_mortality_rate_ci(mac, moc, confidence_level)
    if rate_lower is None or rate_upper is None or mec <= 0:
        return (None, None)
    credible_deaths_lower = rate_lower * moc
    credible_deaths_upper = rate_upper * moc
    return (credible_deaths_lower / mec, credible_deaths_upper / mec)


def compute_ae_ci_amount(
    mac: float,
    moc: float,
    mec: float,
    actual_amount: float,
    expected_amount: float,
    confidence_level: float = 0.95,
) -> tuple[float | None, float | None]:
    if (
        pd.isna(mac)
        or pd.isna(moc)
        or pd.isna(mec)
        or pd.isna(actual_amount)
        or pd.isna(expected_amount)
        or mec <= 0
        or expected_amount <= 0
    ):
        return (None, None)

    average_claim = actual_amount / mac if mac > 0 else expected_amount / mec
    rate_lower, rate_upper = _compute_mortality_rate_ci(mac, moc, confidence_level)
    if rate_lower is None or rate_upper is None:
        return (None, None)

    credible_amount_lower = rate_lower * moc * average_claim
    credible_amount_upper = rate_upper * moc * average_claim
    return (credible_amount_lower / expected_amount, credible_amount_upper / expected_amount)
