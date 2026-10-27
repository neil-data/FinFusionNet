"""FinFusionNet: Experiment 5 Comprehensive Statistical Comparison & Subgroup Analysis.

Evaluates Temporal Cross-Attention Fusion against:
1. Audited Price-Only Baselines (Exp1: XGB-F0, LR-F0, LSTM-F0)
2. VADER Sentiment Fusion (Exp3B: XGB-VADER, LSTM-VADER)
3. FinBERT Concatenation (Exp4: XGB-FinBERT, LR-FinBERT, LSTM-FinBERT)
4. Trailing Volatility Benchmarks (volatility_10, volatility_20)

Statistical Rigor:
- 10,000 paired date-block bootstrap samples (L=10) on date-level accuracy.
- Date-level McNemar test.
- Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction for volatility forecasts.
- Holm-Bonferroni correction across hypotheses.
- Subgroup analyses: Active News vs No-News, Heavyweight 10 vs Expanded 40.
- Sector-level performance breakdown across all NIFTY sectors.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finfusionnet.evaluation.stats_tests import (
    paired_date_block_bootstrap,
    date_level_mcnemar_test,
    diebold_mariano_test,
    apply_holm_bonferroni,
)

EXPERIMENT_DIR = Path("experiments/experiment_5")
PRED_PATH = EXPERIMENT_DIR / "predictions" / "experiment_5_common_cohort_predictions.parquet"
UNIVERSE_CSV = Path("data/metadata/universe.csv")

HEAVYWEIGHT_10 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "ITC",
    "TCS", "LT", "BHARTIARTL", "SBIN", "HINDUNILVR",
]


def compute_cls_metrics(y_true, y_prob, dates_series):
    y_pred = (y_prob >= 0.50).astype(int)
    df_eval = pd.DataFrame({"date": dates_series, "y_true": y_true, "y_pred": y_pred})
    date_acc = df_eval.groupby("date").apply(
        lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False
    ).mean()
    pooled_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.50
    brier = brier_score_loss(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    pred_up = float(np.mean(y_pred == 1))

    return {
        "date_level_acc": round(float(date_acc), 4),
        "pooled_acc": round(float(pooled_acc), 4),
        "balanced_acc": round(float(balanced_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "predicted_up_share": round(pred_up, 4),
    }


def compute_vol_metrics(y_true, y_pred):
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[valid]
    yp = y_pred[valid]

    mse = np.mean((yt - yp) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(yt - yp))
    corr = np.corrcoef(yt, yp)[0, 1] if len(yt) > 1 else 0.0
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    ss_res = np.sum((yt - yp) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "rmse": round(float(rmse), 4),
        "mae": round(float(mae), 4),
        "pearson_r": round(float(corr), 4),
        "oos_r2": round(float(r2), 4),
    }


def run_experiment_5_analysis():
    if not PRED_PATH.exists():
        raise FileNotFoundError(f"Predictions file not found at {PRED_PATH}. Ensure walk-forward training has completed.")

    print(f"Loading Experiment 5 consolidated predictions from {PRED_PATH}...", flush=True)
    df = pd.read_parquet(PRED_PATH)
    print(f"Loaded {len(df):,} observations across {df['date'].nunique()} test dates and {df['ticker'].nunique()} tickers.", flush=True)

    y_true_dir = df["y_true"].values
    dates = df["date"].values
    y_true_vol = df["target_realized_vol"].values

    # -------------------------------------------------------------
    # 1. DIRECTION CLASSIFICATION BENCHMARK TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1. NEXT-DAY DIRECTION CLASSIFICATION PERFORMANCE (N=45,136)")
    print("=" * 80, flush=True)

    cls_models = [
        ("Cross-Attention Net (Exp 5)", "prob_cross_attn"),
        ("LSTM-F1_FinBERT (Concatenation, Exp 4)", "prob_lstm_f1_finbert"),
        ("LSTM-F1_VADER (Exp 3B)", "prob_lstm_f1_vader"),
        ("LSTM-F0 (Price-Only, Exp 1)", "prob_lstm_f0"),
        ("XGB-F1_FinBERT (Exp 4)", "prob_xgb_f1_finbert"),
        ("XGB-F1_VADER (Exp 3B)", "prob_xgb_f1_vader"),
        ("XGB-F0 (Price-Only, Exp 1)", "prob_xgb_f0"),
        ("LR-F1_FinBERT (Exp 4)", "prob_lr_f1_finbert"),
        ("LR-F0 (Price-Only, Exp 1)", "prob_lr_f0"),
        ("Baseline: Always-Up", "baseline_always_up"),
        ("Baseline: Prev-Day Direction", "baseline_prev_dir"),
    ]

    cls_rows = []
    daily_accuracies = {}

    for name, col in cls_models:
        if col not in df.columns:
            continue
        prob = df[col].values
        m = compute_cls_metrics(y_true_dir, prob, dates)
        m["model_configuration"] = name
        cls_rows.append(m)

        pred = (prob >= 0.50).astype(int)
        df_d = pd.DataFrame({"date": dates, "correct": (y_true_dir == pred).astype(float)})
        daily_accuracies[name] = df_d.groupby("date")["correct"].mean().sort_index().values

    df_cls = pd.DataFrame(cls_rows)[
        ["model_configuration", "date_level_acc", "pooled_acc", "balanced_acc", "precision", "recall", "f1_score", "roc_auc", "brier_score", "log_loss", "predicted_up_share"]
    ]
    df_cls.to_csv(EXPERIMENT_DIR / "experiment_5_classification_results.csv", index=False)
    print(df_cls.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 2. REALIZED VOLATILITY FORECASTING TABLE
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2. NEXT-DAY REALIZED VOLATILITY REGRESSION PERFORMANCE")
    print("=" * 80, flush=True)

    vol_models = [
        ("Cross-Attention Net Head 2 (Exp 5)", df["pred_cross_attn_vol"].values),
        ("Benchmark: Trailing Volatility 20 (volatility_20)", df["volatility_20"].values),
        ("Benchmark: Trailing Volatility 10 (volatility_10)", df["volatility_10"].values),
        ("Benchmark: Expanding Historical Mean Vol", np.full_like(y_true_vol, np.nanmean(y_true_vol))),
    ]

    vol_rows = []
    for name, preds in vol_models:
        vm = compute_vol_metrics(y_true_vol, preds)
        vm["model_configuration"] = name
        vol_rows.append(vm)

    df_vol = pd.DataFrame(vol_rows)[
        ["model_configuration", "rmse", "mae", "pearson_r", "oos_r2"]
    ]
    df_vol.to_csv(EXPERIMENT_DIR / "experiment_5_volatility_results.csv", index=False)
    print(df_vol.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 3. STATISTICAL SIGNIFICANCE TESTS (Classification)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3. STATISTICAL SIGNIFICANCE: PAIRED DATE-BLOCK BOOTSTRAP & MCNEMAR")
    print("=" * 80, flush=True)

    cls_comparisons = [
        ("Cross-Attention vs LSTM-F0 (Price-Only)", "LSTM-F0 (Price-Only, Exp 1)", "Cross-Attention Net (Exp 5)"),
        ("Cross-Attention vs LSTM-F1_FinBERT (Concat)", "LSTM-F1_FinBERT (Concatenation, Exp 4)", "Cross-Attention Net (Exp 5)"),
        ("Cross-Attention vs LSTM-F1_VADER", "LSTM-F1_VADER (Exp 3B)", "Cross-Attention Net (Exp 5)"),
        ("Cross-Attention vs XGB-F0 (Price-Only)", "XGB-F0 (Price-Only, Exp 1)", "Cross-Attention Net (Exp 5)"),
        ("Cross-Attention vs XGB-F1_FinBERT", "XGB-F1_FinBERT (Exp 4)", "Cross-Attention Net (Exp 5)"),
    ]

    stat_rows = []
    p_dict = {}

    for test_label, base_name, test_name in cls_comparisons:
        if base_name not in daily_accuracies or test_name not in daily_accuracies:
            continue
        acc_base = daily_accuracies[base_name]
        acc_test = daily_accuracies[test_name]

        boot_res = paired_date_block_bootstrap(
            acc_test, acc_base, block_length=10, n_bootstraps=10000, random_state=42
        )
        diff = boot_res["observed_mean_difference"]
        ci_lower = boot_res["ci_lower"]
        ci_upper = boot_res["ci_upper"]
        p_val = boot_res["p_value_one_sided"]
        p_dict[test_label] = p_val

        mc_res = date_level_mcnemar_test(acc_test, acc_base)
        mc_diff = mc_res.get("wins_model_a", 0) - mc_res.get("wins_model_b", 0)
        mc_pval = mc_res.get("p_value", 1.0)

        stat_rows.append({
            "comparison": test_label,
            "baseline": base_name,
            "alternative": test_name,
            "acc_diff": round(float(diff), 4),
            "bootstrap_95_ci": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "bootstrap_p_value": round(float(p_val), 4),
            "mcnemar_wins_diff": int(mc_diff),
            "mcnemar_p_value": round(float(mc_pval), 4),
        })

    hb_results = apply_holm_bonferroni(p_dict, family_alpha=0.05)
    for res in stat_rows:
        label = res["comparison"]
        hb_info = hb_results.get(label, {})
        res["holm_bonferroni_p_value"] = round(float(hb_info.get("adjusted_p", 1.0)), 4)
        res["statistically_significant"] = bool(hb_info.get("is_significant", False) and res["acc_diff"] > 0)

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(EXPERIMENT_DIR / "experiment_5_statistical_significance.csv", index=False)
    print(df_stats.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 4. DIEBOLD-MARIANO TESTS (Volatility Forecasting Accuracy)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4. DIEBOLD-MARIANO (1995) TESTS FOR REALIZED VOLATILITY PREDICTIVE ACCURACY")
    print("=" * 80, flush=True)

    dm_benchmarks = [
        ("Cross-Attn vs Trailing Volatility 20", df["volatility_20"].values),
        ("Cross-Attn vs Trailing Volatility 10", df["volatility_10"].values),
    ]

    dm_rows = []
    p_dm_dict = {}

    for bench_label, bench_preds in dm_benchmarks:
        # Squared loss (MSE)
        res_sq = diebold_mariano_test(
            actual=y_true_vol,
            pred_model_a=bench_preds,
            pred_model_b=df["pred_cross_attn_vol"].values,
            loss_type="squared",
        )
        dm_rows.append({
            "comparison": bench_label,
            "loss_metric": "Squared Error (MSE)",
            "mean_loss_diff": round(float(res_sq["mean_loss_diff"]), 6),
            "dm_stat": round(float(res_sq["dm_stat"]), 4),
            "p_value": round(float(res_sq["p_value"]), 4),
            "better_model": "Cross-Attention" if res_sq["better_model"] == "model_b" else "Benchmark",
            "is_significant_05": res_sq["is_significant"],
        })
        p_dm_dict[f"{bench_label}_MSE"] = res_sq["p_value"]

        # Absolute loss (MAE)
        res_abs = diebold_mariano_test(
            actual=y_true_vol,
            pred_model_a=bench_preds,
            pred_model_b=df["pred_cross_attn_vol"].values,
            loss_type="absolute",
        )
        dm_rows.append({
            "comparison": bench_label,
            "loss_metric": "Absolute Error (MAE)",
            "mean_loss_diff": round(float(res_abs["mean_loss_diff"]), 6),
            "dm_stat": round(float(res_abs["dm_stat"]), 4),
            "p_value": round(float(res_abs["p_value"]), 4),
            "better_model": "Cross-Attention" if res_abs["better_model"] == "model_b" else "Benchmark",
            "is_significant_05": res_abs["is_significant"],
        })
        p_dm_dict[f"{bench_label}_MAE"] = res_abs["p_value"]

    df_dm = pd.DataFrame(dm_rows)
    df_dm.to_csv(EXPERIMENT_DIR / "experiment_5_diebold_mariano_tests.csv", index=False)
    print(df_dm.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 5. SUBGROUP ANALYSES
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("5. SUBGROUP ANALYSES (Active News vs No-News, Heavyweight vs Expanded)")
    print("=" * 80, flush=True)

    active_mask = (df["has_news_all50"] == 1)
    no_news_mask = (df["has_news_all50"] == 0)
    heavyweight_mask = df["ticker"].isin(HEAVYWEIGHT_10)
    expanded_mask = ~heavyweight_mask

    subgroups = [
        ("Active News Days", active_mask),
        ("No-News Days", no_news_mask),
        ("Original 10 Heavyweights", heavyweight_mask),
        ("Expanded 40 Stocks", expanded_mask),
    ]

    subgroup_rows = []
    models_to_track = [
        ("Cross-Attention Net (Exp 5)", "prob_cross_attn"),
        ("LSTM-F1_FinBERT (Exp 4)", "prob_lstm_f1_finbert"),
        ("LSTM-F0 (Price-Only, Exp 1)", "prob_lstm_f0"),
        ("XGB-F1_FinBERT (Exp 4)", "prob_xgb_f1_finbert"),
        ("XGB-F0 (Price-Only, Exp 1)", "prob_xgb_f0"),
    ]

    for sub_name, mask in subgroups:
        sub_df = df[mask]
        n_obs = len(sub_df)
        pct = (n_obs / len(df)) * 100
        y_sub_dir = sub_df["y_true"].values
        y_sub_vol = sub_df["target_realized_vol"].values
        d_sub = sub_df["date"].values

        vm = compute_vol_metrics(y_sub_vol, sub_df["pred_cross_attn_vol"].values)

        for m_name, col in models_to_track:
            if col not in sub_df.columns:
                continue
            prob = sub_df[col].values
            m = compute_cls_metrics(y_sub_dir, prob, d_sub)
            m["subgroup"] = sub_name
            m["n_obs"] = n_obs
            m["pct_of_total"] = round(pct, 1)
            m["model_configuration"] = m_name
            m["cross_attn_vol_rmse"] = vm["rmse"]
            m["cross_attn_vol_mae"] = vm["mae"]
            subgroup_rows.append(m)

    df_subgroups = pd.DataFrame(subgroup_rows)[
        ["subgroup", "n_obs", "pct_of_total", "model_configuration", "date_level_acc", "pooled_acc", "balanced_acc", "roc_auc", "brier_score", "cross_attn_vol_rmse", "cross_attn_vol_mae"]
    ]
    df_subgroups.to_csv(EXPERIMENT_DIR / "experiment_5_subgroups.csv", index=False)
    print(df_subgroups.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 6. SECTOR-LEVEL BREAKDOWN
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("6. SECTOR-LEVEL PERFORMANCE BREAKDOWN")
    print("=" * 80, flush=True)

    if UNIVERSE_CSV.exists():
        u_df = pd.read_csv(UNIVERSE_CSV)
        sector_map = dict(zip(u_df["ticker"], u_df["sector"]))
    else:
        sector_map = {}

    df["sector"] = df["ticker"].map(sector_map).fillna("Other")
    sector_rows = []

    for sector, grp in df.groupby("sector"):
        if len(grp) < 100:
            continue
        y_sec_dir = grp["y_true"].values
        y_sec_vol = grp["target_realized_vol"].values

        acc_attn = accuracy_score(y_sec_dir, (grp["prob_cross_attn"] >= 0.50).astype(int))
        acc_lstm_finbert = accuracy_score(y_sec_dir, (grp["prob_lstm_f1_finbert"] >= 0.50).astype(int)) if "prob_lstm_f1_finbert" in grp.columns else np.nan
        acc_lstm_f0 = accuracy_score(y_sec_dir, (grp["prob_lstm_f0"] >= 0.50).astype(int)) if "prob_lstm_f0" in grp.columns else np.nan

        acc_xgb_finbert = accuracy_score(y_sec_dir, (grp["prob_xgb_f1_finbert"] >= 0.50).astype(int)) if "prob_xgb_f1_finbert" in grp.columns else np.nan
        acc_xgb_f0 = accuracy_score(y_sec_dir, (grp["prob_xgb_f0"] >= 0.50).astype(int)) if "prob_xgb_f0" in grp.columns else np.nan

        vm = compute_vol_metrics(y_sec_vol, grp["pred_cross_attn_vol"].values)
        vm_bench = compute_vol_metrics(y_sec_vol, grp["volatility_20"].values)

        sector_rows.append({
            "sector": sector,
            "n_obs": len(grp),
            "news_coverage_pct": round(float((grp["has_news_all50"] == 1).mean() * 100), 1),
            "cross_attn_acc": round(float(acc_attn), 4),
            "lstm_finbert_acc": round(float(acc_lstm_finbert), 4),
            "lstm_f0_acc": round(float(acc_lstm_f0), 4),
            "xgb_finbert_acc": round(float(acc_xgb_finbert), 4),
            "xgb_f0_acc": round(float(acc_xgb_f0), 4),
            "cross_attn_vs_lstm_f0": round(float(acc_attn - acc_lstm_f0), 4),
            "cross_attn_vol_rmse": vm["rmse"],
            "bench_vol20_rmse": vm_bench["rmse"],
            "vol_rmse_improvement": round(float(vm_bench["rmse"] - vm["rmse"]), 4),
        })

    df_sectors = pd.DataFrame(sector_rows).sort_values("n_obs", ascending=False)
    df_sectors.to_csv(EXPERIMENT_DIR / "experiment_5_sector_breakdown.csv", index=False)
    print(df_sectors.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 7. PER-FOLD BREAKDOWN
    # -------------------------------------------------------------
    fold_records = []
    for f_idx, grp in df.groupby("fold_idx"):
        f_y_dir = grp["y_true"].values
        f_y_vol = grp["target_realized_vol"].values

        vm = compute_vol_metrics(f_y_vol, grp["pred_cross_attn_vol"].values)
        acc_attn = accuracy_score(f_y_dir, (grp["prob_cross_attn"] >= 0.50).astype(int))
        acc_lstm_f1 = accuracy_score(f_y_dir, (grp["prob_lstm_f1_finbert"] >= 0.50).astype(int)) if "prob_lstm_f1_finbert" in grp.columns else np.nan
        acc_lstm_f0 = accuracy_score(f_y_dir, (grp["prob_lstm_f0"] >= 0.50).astype(int)) if "prob_lstm_f0" in grp.columns else np.nan

        fold_records.append({
            "fold_idx": f_idx,
            "test_start": str(grp["date"].min()),
            "test_end": str(grp["date"].max()),
            "n_samples": len(grp),
            "cross_attn_acc": round(float(acc_attn), 4),
            "lstm_finbert_acc": round(float(acc_lstm_f1), 4),
            "lstm_f0_acc": round(float(acc_lstm_f0), 4),
            "cross_attn_vol_rmse": vm["rmse"],
            "cross_attn_vol_mae": vm["mae"],
        })

    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv(EXPERIMENT_DIR / "experiment_5_fold_results.csv", index=False)

    print("\nExperiment 5 complete analysis finished successfully.", flush=True)


if __name__ == "__main__":
    run_experiment_5_analysis()
