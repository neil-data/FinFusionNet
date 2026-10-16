"""FinFusionNet: Rigorous Statistical Significance & Hypothesis Testing Engine.

Implements:
1. Paired date-block bootstrap (resampling contiguous calendar blocks of trading dates,
   keeping all stocks on the same date strictly together).
2. Pre-registered primary hypothesis testing: X5 vs X3 on Protocol B date-level accuracy.
3. Block-length sensitivity analysis (L in [3, 5, 10, 15, 20]).
4. Holm-Bonferroni step-down correction for secondary model comparisons.
5. Supplementary date-level McNemar and binomial tests.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from scipy import stats


def paired_date_block_bootstrap(
    daily_acc_model_a: np.ndarray,
    daily_acc_model_b: np.ndarray,
    block_length: int = 10,
    n_bootstraps: int = 10000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Dict[str, Union[float, List[float]]]:
    """Perform paired date-block bootstrap comparing two models on identical dates.

    Resamples contiguous blocks of trading dates to preserve temporal autocorrelation
    and cross-sectional stock correlation.
    
    Delta_d = Acc_A,d - Acc_B,d
    Tests H0: Mean(Delta) <= 0 vs H1: Mean(Delta) > 0 (or two-sided != 0).
    """
    assert len(daily_acc_model_a) == len(daily_acc_model_b), "Input arrays must have identical length"
    n_dates = len(daily_acc_model_a)
    if n_dates < block_length:
        block_length = max(1, n_dates // 2)

    deltas = np.asarray(daily_acc_model_a) - np.asarray(daily_acc_model_b)
    observed_mean_diff = float(np.mean(deltas))

    # Construct overlapping contiguous blocks of indices
    n_blocks = n_dates - block_length + 1
    blocks = [np.arange(i, i + block_length) for i in range(n_blocks)]

    rng = np.random.RandomState(random_state)
    boot_means = np.empty(n_bootstraps)

    num_blocks_needed = int(np.ceil(n_dates / block_length))

    for b in range(n_bootstraps):
        # Sample block starting positions with replacement
        chosen_block_indices = rng.choice(n_blocks, size=num_blocks_needed, replace=True)
        sample_indices = np.concatenate([blocks[i] for i in chosen_block_indices])[:n_dates]
        boot_means[b] = np.mean(deltas[sample_indices])

    # 95% Percentile Confidence Interval
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))

    # Two-sided empirical p-value under null H0: mean_diff = 0
    # Center bootstrap distribution under null
    boot_means_centered = boot_means - observed_mean_diff
    p_two_sided = float(np.mean(np.abs(boot_means_centered) >= np.abs(observed_mean_diff)))

    # One-sided empirical p-value for H1: mean_diff > 0
    p_one_sided = float(np.mean(boot_means <= 0.0))

    return {
        "observed_mean_difference": observed_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "block_length": block_length,
        "n_bootstraps": n_bootstraps,
        "n_dates": n_dates,
        "is_significant_05": bool((ci_lower > 0.0) or (ci_upper < 0.0)),
    }


def block_length_sensitivity_analysis(
    daily_acc_model_a: np.ndarray,
    daily_acc_model_b: np.ndarray,
    block_lengths: List[int] = [3, 5, 10, 15, 20],
    n_bootstraps: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Analyze bootstrap CI and p-value sensitivity across different block lengths."""
    results = []
    for L in block_lengths:
        res = paired_date_block_bootstrap(
            daily_acc_model_a,
            daily_acc_model_b,
            block_length=L,
            n_bootstraps=n_bootstraps,
            random_state=random_state,
        )
        results.append({
            "block_length": L,
            "mean_diff": res["observed_mean_difference"],
            "ci_lower": res["ci_lower"],
            "ci_upper": res["ci_upper"],
            "p_two_sided": res["p_value_two_sided"],
            "p_one_sided": res["p_value_one_sided"],
            "significant_05": res["is_significant_05"],
        })
    return pd.DataFrame(results)


def apply_holm_bonferroni(
    p_values: Dict[str, float],
    family_alpha: float = 0.05,
) -> Dict[str, Dict[str, Union[float, bool]]]:
    """Apply Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER).

    Pre-registered primary hypothesis is excluded from this family.
    """
    sorted_comparisons = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_comparisons)

    adjusted_results = {}
    running_max_p = 0.0

    for rank, (name, raw_p) in enumerate(sorted_comparisons):
        # Step-down threshold: alpha / (m - rank)
        k = m - rank
        thresh = family_alpha / k
        adj_p = min(1.0, raw_p * k)
        running_max_p = max(running_max_p, adj_p)

        adjusted_results[name] = {
            "raw_p": float(raw_p),
            "adjusted_p": float(running_max_p),
            "threshold": float(thresh),
            "is_significant": bool(raw_p <= thresh),
        }

    return adjusted_results


def date_level_mcnemar_test(
    daily_acc_model_a: np.ndarray,
    daily_acc_model_b: np.ndarray,
) -> Dict[str, float]:
    """Supplementary paired test at date level: count dates where A wins vs B wins."""
    a = np.asarray(daily_acc_model_a)
    b = np.asarray(daily_acc_model_b)

    wins_a = np.sum(a > b)
    wins_b = np.sum(b > a)
    ties = np.sum(a == b)

    n_discordant = wins_a + wins_b
    if n_discordant > 0:
        # Binomial test on discordant pairs (McNemar with continuity correction or exact binomial)
        p_val = float(stats.binomtest(wins_a, n_discordant, p=0.5).pvalue)
    else:
        p_val = 1.0

    return {
        "wins_model_a": int(wins_a),
        "wins_model_b": int(wins_b),
        "ties": int(ties),
        "p_value": p_val,
    }
