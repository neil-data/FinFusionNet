"""Unit tests for Statistical Engine & Hypothesis Testing."""

import pytest
import numpy as np
import pandas as pd

from finfusionnet.evaluation.stats_tests import (
    paired_date_block_bootstrap,
    block_length_sensitivity_analysis,
    apply_holm_bonferroni,
    date_level_mcnemar_test,
)


def test_paired_bootstrap_identical_models():
    """Verify that identical model predictions yield zero difference and non-significance."""
    np.random.seed(42)
    acc = np.random.uniform(0.48, 0.56, size=100)
    res = paired_date_block_bootstrap(acc, acc, block_length=5, n_bootstraps=1000)

    assert res["observed_mean_difference"] == 0.0
    assert res["ci_lower"] <= 0.0 <= res["ci_upper"]
    assert res["is_significant_05"] is False
    assert res["p_value_two_sided"] >= 0.95


def test_paired_bootstrap_detects_significant_lift():
    """Verify that a consistent, substantial lift is detected as statistically significant."""
    np.random.seed(42)
    # Model A is consistently 5 percentage points better on average with noise
    acc_b = np.random.uniform(0.45, 0.55, size=100)
    acc_a = acc_b + 0.05 + np.random.normal(0, 0.01, size=100)

    res = paired_date_block_bootstrap(acc_a, acc_b, block_length=5, n_bootstraps=2000)

    assert res["observed_mean_difference"] > 0.04
    assert res["ci_lower"] > 0.0
    assert res["is_significant_05"] is True
    assert res["p_value_one_sided"] < 0.01


def test_block_length_sensitivity():
    """Verify that block-length sensitivity runs smoothly across diverse block sizes."""
    np.random.seed(42)
    acc_b = np.random.uniform(0.45, 0.55, size=100)
    acc_a = acc_b + 0.02 + np.random.normal(0, 0.02, size=100)

    sens_df = block_length_sensitivity_analysis(
        acc_a, acc_b, block_lengths=[3, 5, 10], n_bootstraps=500
    )

    assert len(sens_df) == 3
    assert list(sens_df["block_length"]) == [3, 5, 10]
    assert "ci_lower" in sens_df.columns
    assert "ci_upper" in sens_df.columns


def test_holm_bonferroni_correction():
    """Verify Holm-Bonferroni step-down correction logic."""
    raw_p = {
        "X2_vs_X1": 0.005,
        "X3_vs_X1": 0.040,
        "X4_vs_X3": 0.020,
        "X6_vs_X3": 0.060,
    }

    adj = apply_holm_bonferroni(raw_p, family_alpha=0.05)
    assert len(adj) == 4

    # Lowest p-value (0.005) multiplied by m=4 -> 0.020 <= 0.05 -> significant
    assert adj["X2_vs_X1"]["is_significant"] is True
    # Highest p-value (0.060) remains non-significant
    assert adj["X6_vs_X3"]["is_significant"] is False


def test_date_level_mcnemar():
    """Verify supplementary date-level McNemar test."""
    acc_a = np.array([0.6, 0.7, 0.5, 0.8, 0.6])
    acc_b = np.array([0.5, 0.6, 0.5, 0.7, 0.7])  # A wins 3, B wins 1, 1 tie

    res = date_level_mcnemar_test(acc_a, acc_b)
    assert res["wins_model_a"] == 3
    assert res["wins_model_b"] == 1
    assert res["ties"] == 1
    assert "p_value" in res
