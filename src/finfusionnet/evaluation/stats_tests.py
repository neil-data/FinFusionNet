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

    # Two-sided empirical p-value strictly inverted from the bootstrap distribution
    # Guarantees exact mathematical agreement between p < alpha and 0 not in [ci_lower, ci_upper]
    p_tail_le_0 = float(np.mean(boot_means <= 0.0))
    p_tail_ge_0 = float(np.mean(boot_means >= 0.0))
    p_two_sided = min(1.0, 2.0 * min(p_tail_le_0, p_tail_ge_0))

    # One-sided empirical p-value for H1: mean_diff > 0
    p_one_sided = p_tail_le_0

    # Decision rule: significant at alpha=0.05 strictly when p_two_sided < alpha
    is_sig = bool(p_two_sided < alpha)

    return {
        "observed_mean_difference": observed_mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value_two_sided": p_two_sided,
        "p_value_one_sided": p_one_sided,
        "block_length": block_length,
        "n_bootstraps": n_bootstraps,
        "n_dates": n_dates,
        "is_significant_05": is_sig,
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


def diebold_mariano_test(
    actual: np.ndarray,
    pred_model_a: np.ndarray,
    pred_model_b: np.ndarray,
    loss_type: str = "squared",
    h: int = 1,
) -> Dict[str, Union[float, bool]]:
    """Diebold-Mariano (1995) test for comparing predictive accuracy of two forecasts.

    loss_type: "squared" (MSE) or "absolute" (MAE).
    h: forecast horizon (default 1 for next-session).

    Tests H0: E[d_t] = 0 (equal predictive accuracy)
    vs   H1: E[d_t] != 0 (differing predictive accuracy).
    d_t = Loss(e_A,t) - Loss(e_B,t).
    Negative d_t indicates model A has lower loss (superior accuracy).
    """
    y = np.asarray(actual)
    y_a = np.asarray(pred_model_a)
    y_b = np.asarray(pred_model_b)

    valid_mask = np.isfinite(y) & np.isfinite(y_a) & np.isfinite(y_b)
    y = y[valid_mask]
    y_a = y_a[valid_mask]
    y_b = y_b[valid_mask]

    T = len(y)
    if T < 2:
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_loss_diff": 0.0, "is_significant": False}

    e_a = y - y_a
    e_b = y - y_b

    if loss_type == "squared":
        d = (e_a ** 2) - (e_b ** 2)
    elif loss_type == "absolute":
        d = np.abs(e_a) - np.abs(e_b)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    d_bar = float(np.mean(d))

    # Auto-covariance estimation with Bartlett kernel (Newey-West) up to lag h-1
    gamma0 = float(np.var(d, ddof=1))
    var_d = gamma0
    for k in range(1, h):
        gamma_k = float(np.cov(d[:-k], d[k:])[0, 1])
        weight = 1.0 - (k / h)
        var_d += 2.0 * weight * gamma_k

    var_d_bar = max(var_d / T, 1e-12)
    dm_stat = d_bar / np.sqrt(var_d_bar)

    # Harvey, Leybourne, Newbold (1997) small-sample modification
    hln_factor = np.sqrt((T + 1 - 2 * h + (h * (h - 1)) / T) / T)
    dm_stat_adj = float(dm_stat * hln_factor)

    p_value = float(2.0 * (1.0 - stats.t.cdf(np.abs(dm_stat_adj), df=T - 1)))
    is_sig = bool(p_value < 0.05)
    if np.isclose(d_bar, 0.0):
        better_model = "equal"
    elif d_bar < 0:
        better_model = "model_a"
    else:
        better_model = "model_b"

    return {
        "dm_stat": dm_stat_adj,
        "p_value": p_value,
        "mean_loss_diff": d_bar,
        "better_model": better_model,
        "loss_type": loss_type,
        "is_significant": is_sig,
        "sample_size": T,
    }
