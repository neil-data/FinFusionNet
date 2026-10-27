"""FinFusionNet: Experiment 4 - FinBERT + Price Fusion.

Evaluates whether domain-specific FinBERT sentiment features provide incremental
predictive value over:
1. Audited Price-Only Baselines (F0)
2. Experiment 3B All-50 VADER Sentiment Fusion (F1_VADER)

Protocol:
- Protocol B: 45 monthly expanding folds, 24-month initial training window.
- Strict 1-session embargo.
- Common evaluation cohort across all model comparisons (inner join on fold_idx, date, ticker).
- Models:
  - XGBoost (F0, F1_VADER, F1_FinBERT)
  - Logistic Regression (F0, F1_VADER, F1_FinBERT)
  - LSTM (F0, F1_VADER, F1_FinBERT, Seed 42)
- Statistical testing: Paired date-block bootstrap (10,000 resamples, L=10), McNemar test, Holm-Bonferroni correction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

torch.set_num_threads(6)

from finfusionnet.evaluation.pipeline import FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.evaluation.lstm_pipeline import extract_boundary_context_sequences
from finfusionnet.data.sequence_generator import generate_ticker_sequences
from finfusionnet.models.xgboost_model import FinFusionXGBoost
from finfusionnet.models.lstm_model import FinFusionLSTMClassifier
from finfusionnet.evaluation.stats_tests import (
    paired_date_block_bootstrap,
    date_level_mcnemar_test,
    apply_holm_bonferroni,
)
from finfusionnet.features.sentiment import integrate_sentiment_features


EXPERIMENT_DIR = Path("experiments/experiment_4")
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
FOLD_CKPT_DIR = PREDICTIONS_DIR / "exp4_folds"


def compute_comprehensive_metrics(y_true, y_pred, y_prob, dates_series):
    df_eval = pd.DataFrame({
        "date": dates_series,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    })
    date_acc = df_eval.groupby("date").apply(
        lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False
    ).mean()
    pooled_acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    brier = float(np.mean((y_prob - y_true) ** 2))
    pred_up_share = float(np.mean(y_pred == 1))

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
        "predicted_up_share": round(pred_up_share, 4),
    }


def run_experiment_4():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINFUSIONNET EXPERIMENT 4: FINBERT + PRICE FUSION EVALUATION")
    print("=" * 80, flush=True)

    # 1. Load Audited Base Dataset
    master_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    if not master_path.exists():
        master_path = Path("data/processed/master_dataset/master_ticker_day.parquet")
    print(f"Loading base price dataset from {master_path}...", flush=True)
    df_price = pd.read_parquet(master_path)
    df_price["date"] = pd.to_datetime(df_price["date"])

    # 2. Load Daily GNews All-50 FinBERT Sentiment Features
    finbert_path = Path("data/processed/gnews_all50/daily_gnews_all50_finbert_features.parquet")
    if not finbert_path.exists():
        raise FileNotFoundError(f"All-50 FinBERT features not found at {finbert_path}. Run processing script first.")
    print(f"Loading All-50 FinBERT daily sentiment features from {finbert_path}...", flush=True)
    df_finbert = pd.read_parquet(finbert_path)
    df_finbert["date"] = pd.to_datetime(df_finbert["date"])

    # 3. Integrate Sentiment Features (enforcing strict missing-news = null policy)
    print("Integrating All-50 FinBERT sentiment with price dataset (strict missing-news = null)...", flush=True)
    df_merged = integrate_sentiment_features(df_price, df_finbert, date_col="date", ticker_col="ticker")

    # Feature groups
    f0_features = FEATURE_GROUPS["group_a_technical"]
    news_controls = ["n_articles", "has_news"]
    finbert_features = [
        "finbert_positive_mean",
        "finbert_negative_mean",
        "finbert_neutral_mean",
        "finbert_sentiment_score",
        "finbert_dispersion",
    ]
    f1_features = f0_features + news_controls + finbert_features

    print(f"F0 Features ({len(f0_features)}): {f0_features}", flush=True)
    print(f"F1 Features ({len(f1_features)}): {f1_features}", flush=True)

    # Build cohort
    mask_date = (df_merged["date"] >= pd.Timestamp("2021-01-01")) & (df_merged["date"] <= pd.Timestamp("2026-09-30"))
    mask_target = df_merged["target"].notna()
    mask_f0 = df_merged[f0_features].notna().all(axis=1)

    cohort_df = df_merged[mask_date & mask_target & mask_f0].copy()
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Baselines
    cohort_df["baseline_always_up"] = 1
    cohort_df["baseline_prev_dir"] = (cohort_df["return_1d"] > 0).astype(int)

    # Walk-Forward Protocol B Folds
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(cohort_df)
    print(f"\nTotal Protocol B Expanding Folds: {len(folds)} | Total Cohort Ticker-Days: {len(cohort_df):,}", flush=True)

    pred_file = PREDICTIONS_DIR / "experiment_4_common_cohort_predictions.parquet"

    if not pred_file.exists():
        print("\nStarting Walk-Forward Training of FinBERT F1 Models across 45 Folds...", flush=True)

        for fold in folds:
            fold_ckpt = FOLD_CKPT_DIR / f"fold_{fold.fold_idx:02d}.parquet"
            if fold_ckpt.exists():
                continue

            train_df = cohort_df.iloc[fold.train_indices].copy()
            test_df = cohort_df.iloc[fold.test_indices].copy()

            y_tr = train_df["target"].values.astype(int)
            y_te = test_df["target"].values.astype(int)
            dates_te = test_df["date"].dt.strftime("%Y-%m-%d").values
            tickers_te = test_df["ticker"].values

            # -------------------------------------------------------------
            # A. TABULAR MODEL: XGBoost F1_FinBERT
            # -------------------------------------------------------------
            xgb_f1 = FinFusionXGBoost(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
            xgb_f1.fit(train_df, f1_features, "target", ticker_col="ticker")
            prob_xgb_f1 = xgb_f1.predict_proba(test_df, f1_features)

            # -------------------------------------------------------------
            # B. TABULAR MODEL: Logistic Regression F1_FinBERT
            # -------------------------------------------------------------
            mean_f1 = train_df[f1_features].mean()
            std_f1 = train_df[f1_features].std().replace(0, 1)
            X_tr_f1 = ((train_df[f1_features] - mean_f1) / std_f1).fillna(0).values
            X_te_f1 = ((test_df[f1_features] - mean_f1) / std_f1).fillna(0).values

            lr_f1 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            lr_f1.fit(X_tr_f1, y_tr)
            prob_lr_f1 = lr_f1.predict_proba(X_te_f1)[:, 1]

            # -------------------------------------------------------------
            # C. SEQUENTIAL MODEL: LSTM F1_FinBERT (Seed 42)
            # -------------------------------------------------------------
            prior_dates = fold.val_dates if (fold.val_dates and len(fold.val_dates) > 0) else fold.train_dates

            train_df_f1 = train_df.copy()
            train_df_f1[f1_features] = train_df[f1_features].fillna(mean_f1)

            cohort_df_f1 = cohort_df.copy()
            cohort_df_f1[f1_features] = cohort_df_f1[f1_features].fillna(mean_f1)

            X_tr_seq_f1, y_tr_seq_f1, _, _, _ = generate_ticker_sequences(train_df_f1, f1_features, sequence_length=20)
            X_te_seq_f1, _, d_te_f1, t_te_f1, _, _ = extract_boundary_context_sequences(
                cohort_df_f1, prior_dates[-19:], fold.test_dates, f1_features, sequence_length=20
            )

            # Train LSTM-F1 (Seed 42)
            clf_f1 = FinFusionLSTMClassifier(
                input_size=len(f1_features), hidden_dim=64, num_layers=2, batch_size=512, max_epochs=4, random_state=42
            )
            clf_f1.fit(X_tr_seq_f1, y_tr_seq_f1)
            prob_lstm_f1 = clf_f1.predict_proba(X_te_seq_f1)

            # Build fold dataframes
            fold_tab_df = pd.DataFrame({
                "fold_idx": fold.fold_idx,
                "date": dates_te,
                "ticker": tickers_te,
                "y_true": y_te,
                "baseline_always_up": test_df["baseline_always_up"].values,
                "baseline_prev_dir": test_df["baseline_prev_dir"].values,
                "has_news_all50": test_df["has_news"].values,
                "n_articles_all50": test_df["n_articles"].values,
                "prob_xgb_f1_finbert": prob_xgb_f1,
                "prob_lr_f1_finbert": prob_lr_f1,
            })

            fold_lstm_df = pd.DataFrame({
                "date": pd.to_datetime(d_te_f1).strftime("%Y-%m-%d"),
                "ticker": t_te_f1,
                "prob_lstm_f1_finbert": prob_lstm_f1,
            })

            merged_fold = fold_tab_df.merge(fold_lstm_df, on=["date", "ticker"], how="inner")
            merged_fold.to_parquet(fold_ckpt, index=False)

            if (fold.fold_idx + 1) % 5 == 0 or (fold.fold_idx + 1) == len(folds):
                print(f"  Completed Fold {fold.fold_idx + 1:02d} / {len(folds)} ({len(merged_fold):,} rows)", flush=True)

        # Merge all fold checkpoints
        all_fold_files = sorted(FOLD_CKPT_DIR.glob("fold_*.parquet"))
        f1_all_preds = pd.concat([pd.read_parquet(f) for f in all_fold_files], ignore_index=True)

        # Load Audited Frozen F0 Predictions
        print("\nLoading and joining audited frozen F0 predictions...", flush=True)
        df_xgb_f0 = pd.read_parquet("experiments/audit/predictions/xgb_p_predictions.parquet")
        df_lr_f0 = pd.read_parquet("experiments/audit/predictions/lr_p_predictions.parquet")
        df_lstm_f0 = pd.read_parquet("experiments/audit/predictions/lstm_p_predictions.parquet")

        f0_merged = df_xgb_f0[["fold_idx", "date", "ticker", "y_prob"]].rename(columns={"y_prob": "prob_xgb_f0"})
        f0_merged = f0_merged.merge(
            df_lr_f0[["fold_idx", "date", "ticker", "y_prob"]].rename(columns={"y_prob": "prob_lr_f0"}),
            on=["fold_idx", "date", "ticker"]
        )
        f0_merged = f0_merged.merge(
            df_lstm_f0[["fold_idx", "date", "ticker", "y_prob_seed_42"]].rename(columns={"y_prob_seed_42": "prob_lstm_f0"}),
            on=["fold_idx", "date", "ticker"]
        )

        # Load Exp 3B (All-50 VADER news) predictions
        print("Loading and joining Experiment 3B (All-50 VADER news) predictions...", flush=True)
        df_exp3b = pd.read_parquet("experiments/experiment_3b/predictions/experiment_3b_common_cohort_predictions.parquet")
        df_exp3b_sub = df_exp3b[["fold_idx", "date", "ticker", "prob_xgb_f1_all50", "prob_lr_f1_all50", "prob_lstm_f1_all50"]].rename(
            columns={
                "prob_xgb_f1_all50": "prob_xgb_f1_vader",
                "prob_lr_f1_all50": "prob_lr_f1_vader",
                "prob_lstm_f1_all50": "prob_lstm_f1_vader",
            }
        )

        eval_cohort = f1_all_preds.merge(f0_merged, on=["fold_idx", "date", "ticker"], how="inner")
        eval_cohort = eval_cohort.merge(df_exp3b_sub, on=["fold_idx", "date", "ticker"], how="inner")
        eval_cohort.to_parquet(pred_file, index=False)
        print(f"Saved Experiment 4 consolidated predictions to {pred_file} ({len(eval_cohort):,} rows across {eval_cohort['date'].nunique()} dates).", flush=True)
    else:
        print(f"\nLoading existing Experiment 4 predictions from {pred_file}...", flush=True)
        eval_cohort = pd.read_parquet(pred_file)

    # -------------------------------------------------------------
    # 5. COMPUTE MODEL METRICS
    # -------------------------------------------------------------
    y_true = eval_cohort["y_true"].values
    dates = eval_cohort["date"].values

    models = [
        ("XGB-F0 (Price-Only)", "prob_xgb_f0"),
        ("XGB-F1_VADER (All-50 VADER)", "prob_xgb_f1_vader"),
        ("XGB-F1_FinBERT (All-50 FinBERT)", "prob_xgb_f1_finbert"),
        ("LR-F0 (Price-Only)", "prob_lr_f0"),
        ("LR-F1_VADER (All-50 VADER)", "prob_lr_f1_vader"),
        ("LR-F1_FinBERT (All-50 FinBERT)", "prob_lr_f1_finbert"),
        ("LSTM-F0 (Price-Only, Seed 42)", "prob_lstm_f0"),
        ("LSTM-F1_VADER (All-50 VADER, Seed 42)", "prob_lstm_f1_vader"),
        ("LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)", "prob_lstm_f1_finbert"),
    ]

    metrics_rows = []
    daily_accuracies = {}

    for name, col_p in models:
        prob = eval_cohort[col_p].values
        pred = (prob >= 0.50).astype(int)
        m = compute_comprehensive_metrics(y_true, pred, prob, dates)
        m["configuration"] = name
        metrics_rows.append(m)

        # Date-level accuracy series for bootstrapping
        df_m = pd.DataFrame({"date": dates, "correct": (y_true == pred).astype(float)})
        daily_accuracies[name] = df_m.groupby("date")["correct"].mean().sort_index().values

    # Naive Baselines
    up_m = compute_comprehensive_metrics(y_true, eval_cohort["baseline_always_up"].values, np.ones_like(y_true, dtype=float), dates)
    up_m["configuration"] = "Baseline: Always-Up"
    metrics_rows.append(up_m)

    prev_m = compute_comprehensive_metrics(y_true, eval_cohort["baseline_prev_dir"].values, eval_cohort["baseline_prev_dir"].values.astype(float), dates)
    prev_m["configuration"] = "Baseline: Prev-Day Direction"
    metrics_rows.append(prev_m)

    df_metrics = pd.DataFrame(metrics_rows)
    print("\n--- EXPERIMENT 4 CLASSIFICATION PERFORMANCE METRICS ---", flush=True)
    print(df_metrics.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 6. STATISTICAL SIGNIFICANCE TESTS (Paired Date-Block Bootstrap & McNemar)
    # -------------------------------------------------------------
    print("\nRunning Paired Date-Block Bootstrap (10,000 Resamples, L=10 Days) & McNemar Tests...", flush=True)

    comparisons = [
        # Primary: FinBERT News vs Price-Only F0
        ("XGB: FinBERT vs Price-Only", "XGB-F0 (Price-Only)", "XGB-F1_FinBERT (All-50 FinBERT)"),
        ("LR: FinBERT vs Price-Only", "LR-F0 (Price-Only)", "LR-F1_FinBERT (All-50 FinBERT)"),
        ("LSTM: FinBERT vs Price-Only", "LSTM-F0 (Price-Only, Seed 42)", "LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)"),
        # Secondary: FinBERT News vs VADER News
        ("XGB: FinBERT vs VADER News", "XGB-F1_VADER (All-50 VADER)", "XGB-F1_FinBERT (All-50 FinBERT)"),
        ("LR: FinBERT vs VADER News", "LR-F1_VADER (All-50 VADER)", "LR-F1_FinBERT (All-50 FinBERT)"),
        ("LSTM: FinBERT vs VADER News", "LSTM-F1_VADER (All-50 VADER, Seed 42)", "LSTM-F1_FinBERT (All-50 FinBERT, Seed 42)"),
    ]

    stat_results = []
    p_dict = {}

    for test_label, base_name, test_name in comparisons:
        acc_base_daily = daily_accuracies[base_name]
        acc_test_daily = daily_accuracies[test_name]

        boot_res = paired_date_block_bootstrap(
            acc_test_daily, acc_base_daily, block_length=10, n_bootstraps=10000, random_state=42
        )
        diff = boot_res["observed_mean_difference"]
        ci_lower = boot_res["ci_lower"]
        ci_upper = boot_res["ci_upper"]
        p_val = boot_res["p_value_one_sided"]
        p_dict[test_label] = p_val

        mc_res = date_level_mcnemar_test(acc_test_daily, acc_base_daily)
        mc_diff = mc_res.get("wins_model_a", 0) - mc_res.get("wins_model_b", 0)
        mc_pval = mc_res.get("p_value", 1.0)

        stat_results.append({
            "comparison": test_label,
            "baseline": base_name,
            "alternative": test_name,
            "acc_diff": round(float(diff), 4),
            "bootstrap_95_ci": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "bootstrap_p_value": round(float(p_val), 4),
            "mcnemar_wins_diff": int(mc_diff),
            "mcnemar_p_value": round(float(mc_pval), 4),
        })

    # Holm-Bonferroni correction (family of tests)
    hb_results = apply_holm_bonferroni(p_dict, family_alpha=0.05)
    for res in stat_results:
        label = res["comparison"]
        hb_info = hb_results.get(label, {})
        res["holm_bonferroni_p_value"] = round(float(hb_info.get("adjusted_p", 1.0)), 4)
        res["statistically_significant"] = bool(hb_info.get("is_significant", False) and res["acc_diff"] > 0)

    df_stats = pd.DataFrame(stat_results)
    print("\n--- STATISTICAL SIGNIFICANCE SUMMARY ---", flush=True)
    print(df_stats.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 7. PER-FOLD METRICS BREAKDOWN
    # -------------------------------------------------------------
    fold_records = []
    for f_idx, grp in eval_cohort.groupby("fold_idx"):
        f_y = grp["y_true"].values
        f_dates = grp["date"].values
        row = {
            "fold_idx": f_idx,
            "test_start": str(grp["date"].min()),
            "test_end": str(grp["date"].max()),
            "n_samples": len(grp),
        }
        for name, col_p in models:
            f_prob = grp[col_p].values
            f_pred = (f_prob >= 0.50).astype(int)
            row[f"acc_{col_p}"] = round(float(accuracy_score(f_y, f_pred)), 4)
        fold_records.append(row)

    df_folds = pd.DataFrame(fold_records)

    # -------------------------------------------------------------
    # 8. SAVE OUTPUTS
    # -------------------------------------------------------------
    metrics_csv = EXPERIMENT_DIR / "experiment_4_results.csv"
    stats_csv = EXPERIMENT_DIR / "experiment_4_statistical_significance.csv"
    folds_csv = EXPERIMENT_DIR / "experiment_4_fold_results.csv"

    df_metrics.to_csv(metrics_csv, index=False)
    df_stats.to_csv(stats_csv, index=False)
    df_folds.to_csv(folds_csv, index=False)

    print(f"\nSaved Experiment 4 results to {metrics_csv}", flush=True)
    print(f"Saved Statistical Significance results to {stats_csv}", flush=True)
    print(f"Saved Fold-level results to {folds_csv}", flush=True)


if __name__ == "__main__":
    run_experiment_4()
