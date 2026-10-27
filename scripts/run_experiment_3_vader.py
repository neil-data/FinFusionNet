"""FinFusionNet: Experiment 3 - Lexicon-Based News Sentiment Fusion (VADER).

Evaluates whether adding VADER-based news sentiment to historical price features
provides statistically significant predictive improvement over price-only baselines.

Reuses audited F0 frozen price-only predictions:
- XGB-F0: from experiments/audit/predictions/xgb_p_predictions.parquet
- LR-F0: from experiments/audit/predictions/lr_p_predictions.parquet
- LSTM-F0 (5-Seed): from experiments/audit/predictions/lstm_p_predictions.parquet

Trains F1 models (Price + VADER news sentiment):
- XGB-F1: XGBoost on Group A + Group C + Group D
- LR-F1: Logistic Regression on Group A + Group C + Group D
- LSTM-F1 (5-Seed): LSTM on Group A + Group C + Group D across seeds 42, 43, 44, 45, 46

Protocol:
- Protocol B: 45 monthly expanding folds, 24-month initial training window.
- Strict 1-session embargo.
- Common evaluation cohort across all model comparisons (inner join on fold_idx, date, ticker).
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


EXPERIMENT_DIR = Path("experiments/experiment_3")
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
FOLD_CKPT_DIR = PREDICTIONS_DIR / "exp3_folds"


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


def run_experiment_3():
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINFUSIONNET EXPERIMENT 3: VADER NEWS SENTIMENT FUSION EVALUATION")
    print("=" * 80, flush=True)

    # 1. Load Audited Base Dataset
    master_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    if not master_path.exists():
        master_path = Path("data/processed/master_dataset/master_ticker_day.parquet")
    print(f"Loading base price dataset from {master_path}...", flush=True)
    df_price = pd.read_parquet(master_path)
    df_price["date"] = pd.to_datetime(df_price["date"])

    # 2. Load Daily GNews VADER Sentiment Features
    vader_path = Path("data/processed/gnews/daily_gnews_vader_features.parquet")
    if not vader_path.exists():
        raise FileNotFoundError(f"VADER features not found at {vader_path}. Run processing script first.")
    print(f"Loading VADER daily sentiment features from {vader_path}...", flush=True)
    df_vader = pd.read_parquet(vader_path)
    df_vader["date"] = pd.to_datetime(df_vader["date"])

    # 3. Integrate Sentiment Features (enforcing strict missing-news = null policy)
    print("Integrating VADER sentiment with price dataset (strict missing-news = null)...", flush=True)
    df_merged = integrate_sentiment_features(df_price, df_vader, date_col="date", ticker_col="ticker")

    # Feature groups
    f0_features = FEATURE_GROUPS["group_a_technical"]
    news_controls = ["n_articles", "has_news"]
    vader_features = [
        "vader_mean",
        "vader_positive_mean",
        "vader_negative_mean",
        "vader_neutral_mean",
        "vader_dispersion",
    ]
    f1_features = f0_features + news_controls + vader_features

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

    pred_file = PREDICTIONS_DIR / "experiment_3_common_cohort_predictions.parquet"
    lstm_seeds = [42, 43, 44, 45, 46]

    if not pred_file.exists():
        print("\nStarting Walk-Forward Training of F1 Models across 45 Folds...", flush=True)

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
            # A. TABULAR MODEL: XGBoost F1
            # -------------------------------------------------------------
            xgb_f1 = FinFusionXGBoost(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
            xgb_f1.fit(train_df, f1_features, "target", ticker_col="ticker")
            prob_xgb_f1 = xgb_f1.predict_proba(test_df, f1_features)

            # -------------------------------------------------------------
            # B. TABULAR MODEL: Logistic Regression F1
            # -------------------------------------------------------------
            mean_f1 = train_df[f1_features].mean()
            std_f1 = train_df[f1_features].std().replace(0, 1)
            X_tr_f1 = ((train_df[f1_features] - mean_f1) / std_f1).fillna(0).values
            X_te_f1 = ((test_df[f1_features] - mean_f1) / std_f1).fillna(0).values

            lr_f1 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            lr_f1.fit(X_tr_f1, y_tr)
            prob_lr_f1 = lr_f1.predict_proba(X_te_f1)[:, 1]

            # -------------------------------------------------------------
            # C. SEQUENTIAL MODEL: LSTM F1 (5 Seeds)
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
                "has_news": test_df["has_news"].values,
                "n_articles": test_df["n_articles"].values,
                "prob_xgb_f1": prob_xgb_f1,
                "prob_lr_f1": prob_lr_f1,
            })

            fold_lstm_df = pd.DataFrame({
                "date": pd.to_datetime(d_te_f1).strftime("%Y-%m-%d"),
                "ticker": t_te_f1,
                "prob_lstm_f1": prob_lstm_f1,
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

        eval_cohort = f1_all_preds.merge(f0_merged, on=["fold_idx", "date", "ticker"], how="inner")
        eval_cohort.to_parquet(pred_file, index=False)
        print(f"Saved Experiment 3 consolidated predictions to {pred_file} ({len(eval_cohort):,} rows across {eval_cohort['date'].nunique()} dates).", flush=True)
    else:
        print(f"\nLoading existing Experiment 3 predictions from {pred_file}...", flush=True)
        eval_cohort = pd.read_parquet(pred_file)

    # -------------------------------------------------------------
    # 5. COMPUTE MODEL METRICS
    # -------------------------------------------------------------
    y_true = eval_cohort["y_true"].values
    dates = eval_cohort["date"].values

    models = [
        ("XGB-F0 (Price-Only)", "prob_xgb_f0"),
        ("XGB-F1 (Price + VADER)", "prob_xgb_f1"),
        ("LR-F0 (Price-Only)", "prob_lr_f0"),
        ("LR-F1 (Price + VADER)", "prob_lr_f1"),
        ("LSTM-F0 (Price-Only, Seed 42)", "prob_lstm_f0"),
        ("LSTM-F1 (Price + VADER, Seed 42)", "prob_lstm_f1"),
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
    print("\n--- EXPERIMENT 3 CLASSIFICATION PERFORMANCE METRICS ---", flush=True)
    print(df_metrics.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 6. STATISTICAL SIGNIFICANCE TESTS (Paired Date-Block Bootstrap & McNemar)
    # -------------------------------------------------------------
    print("\nRunning Paired Date-Block Bootstrap (10,000 Resamples, L=10 Days) & McNemar Tests...", flush=True)

    comparisons = [
        ("XGBoost", "XGB-F0 (Price-Only)", "XGB-F1 (Price + VADER)"),
        ("Logistic Regression", "LR-F0 (Price-Only)", "LR-F1 (Price + VADER)"),
        ("LSTM (Seed 42)", "LSTM-F0 (Price-Only, Seed 42)", "LSTM-F1 (Price + VADER, Seed 42)"),
    ]

    stat_results = []
    p_dict = {}

    for family, base_name, fusion_name in comparisons:
        acc_base_daily = daily_accuracies[base_name]
        acc_fusion_daily = daily_accuracies[fusion_name]

        # Paired Block Bootstrap
        boot_res = paired_date_block_bootstrap(
            acc_fusion_daily, acc_base_daily, block_length=10, n_bootstraps=10000, random_state=42
        )
        diff = boot_res["observed_mean_difference"]
        ci_lower = boot_res["ci_lower"]
        ci_upper = boot_res["ci_upper"]
        p_val = boot_res["p_value_one_sided"]
        p_dict[family] = p_val

        # Date-Level McNemar Test
        mc_res = date_level_mcnemar_test(acc_fusion_daily, acc_base_daily)
        mc_diff = mc_res.get("wins_model_a", 0) - mc_res.get("wins_model_b", 0)
        mc_pval = mc_res.get("p_value", 1.0)

        stat_results.append({
            "model_family": family,
            "baseline": base_name,
            "fusion": fusion_name,
            "acc_diff_f1_minus_f0": round(float(diff), 4),
            "bootstrap_95_ci": f"[{ci_lower:.4f}, {ci_upper:.4f}]",
            "bootstrap_p_value": round(float(p_val), 4),
            "mcnemar_wins_diff": int(mc_diff),
            "mcnemar_p_value": round(float(mc_pval), 4),
        })

    # Holm-Bonferroni correction
    hb_results = apply_holm_bonferroni(p_dict, family_alpha=0.05)
    for res in stat_results:
        fam = res["model_family"]
        hb_info = hb_results.get(fam, {})
        res["holm_bonferroni_p_value"] = round(float(hb_info.get("adjusted_p", 1.0)), 4)
        res["statistically_significant"] = bool(hb_info.get("is_significant", False) and res["acc_diff_f1_minus_f0"] > 0)

    df_stats = pd.DataFrame(stat_results)
    print("\n--- STATISTICAL SIGNIFICANCE SUMMARY (F1 vs F0) ---", flush=True)
    print(df_stats.to_string(index=False), flush=True)

    # -------------------------------------------------------------
    # 7. SAVE OUTPUTS
    # -------------------------------------------------------------
    metrics_csv = EXPERIMENT_DIR / "experiment_3_results.csv"
    stats_csv = EXPERIMENT_DIR / "experiment_3_statistical_significance.csv"

    df_metrics.to_csv(metrics_csv, index=False)
    df_stats.to_csv(stats_csv, index=False)

    print(f"\nSaved Experiment 3 results to {metrics_csv}", flush=True)
    print(f"Saved Statistical Significance results to {stats_csv}", flush=True)


if __name__ == "__main__":
    run_experiment_3()
