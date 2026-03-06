"""
utils/ab_stats.py
Statistical functions for A/B test analysis.
"""

import numpy as np


def two_proportion_ztest(conversions_a, n_a, conversions_b, n_b, confidence=0.95):
    """
    Two-proportion z-test comparing conversion rates between control (A) and treatment (B).

    Returns a dict with:
        - rate_a, rate_b: conversion rates
        - lift: relative lift (B over A)
        - abs_diff: absolute difference (B - A)
        - z_stat: z-statistic
        - p_value: two-tailed p-value
        - ci_low, ci_high: confidence interval for the absolute difference
        - significant: whether p < alpha
    """
    rate_a = conversions_a / n_a
    rate_b = conversions_b / n_b

    # Pooled proportion under null hypothesis
    p_pool = (conversions_a + conversions_b) / (n_a + n_b)

    # Standard error for test statistic
    se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    # Z-statistic
    z_stat = (rate_b - rate_a) / se_test if se_test > 0 else 0.0

    # Two-tailed p-value using normal CDF approximation
    # (avoids scipy dependency — uses the complementary error function)
    p_value = 2 * _normal_sf(abs(z_stat))

    # Confidence interval for the difference (unpooled SE)
    alpha = 1 - confidence
    z_crit = _normal_ppf(1 - alpha / 2)
    se_diff = np.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    abs_diff = rate_b - rate_a
    ci_low = abs_diff - z_crit * se_diff
    ci_high = abs_diff + z_crit * se_diff

    # Relative lift
    lift = (rate_b - rate_a) / rate_a if rate_a > 0 else 0.0

    return {
        "rate_a": rate_a,
        "rate_b": rate_b,
        "lift": lift,
        "abs_diff": abs_diff,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "significant": p_value < alpha,
        "confidence": confidence,
    }


def required_sample_size(baseline_rate, mde, power=0.80, confidence=0.95):
    """
    Calculate the required sample size per group for a two-proportion test.

    Args:
        baseline_rate: expected conversion rate of the control group
        mde: minimum detectable effect (absolute, e.g., 0.03 for 3pp)
        power: statistical power (default 0.80)
        confidence: confidence level (default 0.95)

    Returns:
        Required sample size per group (integer).
    """
    alpha = 1 - confidence
    z_alpha = _normal_ppf(1 - alpha / 2)
    z_beta = _normal_ppf(power)

    p1 = baseline_rate
    p2 = baseline_rate + mde
    p_avg = (p1 + p2) / 2

    numerator = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
                 z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    denominator = mde ** 2

    return int(np.ceil(numerator / denominator))


# ─────────────────────────────────────────────
# Internal helpers (avoid scipy dependency)
# ─────────────────────────────────────────────

def _normal_sf(x):
    """Survival function (1 - CDF) of the standard normal distribution."""
    # Uses the complementary error function from math, available in numpy
    from math import erfc
    return 0.5 * erfc(x / np.sqrt(2))


def _normal_ppf(p):
    """
    Percent-point function (inverse CDF) of the standard normal distribution.
    Rational approximation accurate to ~4.5e-4 (Abramowitz & Stegun 26.2.23).
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be between 0 and 1 (exclusive)")

    # Work in the upper half
    if p < 0.5:
        return -_normal_ppf(1 - p)

    t = np.sqrt(-2 * np.log(1 - p))

    # Coefficients for rational approximation
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    return t - (c0 + c1 * t + c2 * t ** 2) / (1 + d1 * t + d2 * t ** 2 + d3 * t ** 3)