"""Audit B3 & B4: Price-Only Models on Full NIFTY 50 Universe.

Executes:
1. Primary models:
   - XGB-P: FinFusionXGBoost (locked configs: 500 trees, lr=0.05, max_depth=6, early stopping 50).
   - Logistic Regression baseline (C=0.1, L2).
   - LSTM-P: FinFusionLSTMClassifier (5 seeds: 42, 43, 44, 45, 46; average probabilities, report seed spread).
2. Features: Group A technical features only (24 indicators from stock's own OHLCV). NO market context, NO news, NO sentiment, NO ticker one-hot.
3. Protocol B: 45 monthly expanding folds, 1-day embargo.
4. Comprehensive Metrics:
   - Date-level accuracy, pooled accuracy, balanced accuracy, ROC-AUC, log-loss
   - Predicted-up share, actual up-rate, mean predicted probability
   - Baselines: always-up, previous-day direction
   - Paired date-block bootstrap CIs (10,000 reps, L=10) for model minus baselines
   - Flag LSTM collapse if predicted-up share is outside 40-60%.
5. Subset comparison (B4): 10 Heavyweights vs other names.
Outputs:
- experiments/audit/model_metrics_full_universe.csv
- experiments/audit/heavyweights_vs_others.csv
- Serialized predictions under experiments/audit/predictions/
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss

torch.set_num_threads(4)

from finfusionnet.evaluation.pipeline import FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.evaluation.lstm_pipeline import extract_boundary_context_sequences
from finfusionnet.data.sequence_generator import generate_ticker_sequences
from finfusionnet.models.xgboost_model import FinFusionXGBoost
from finfusionnet.models.lstm_model import FinFusionLSTMClassifier
from finfusionnet.evaluation.stats_tests import paired_date_block_bootstrap


HEAVYWEIGHTS = {
    "BHARTIARTL", "HDFCBANK", "HINDUNILVR", "ICICIBANK", "INFY",
    "ITC", "LT", "RELIANCE", "SBIN", "TCS"
}


def compute_metrics_dict(y_true, y_pred, y_prob, dates_series):
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
    auc = roc_auc_score(y_true, y_prob)
    ll = log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7))
    brier = float(np.mean((y_prob - y_true) ** 2))
    pred_up_share = float(np.mean(y_pred == 1))
    actual_up_rate = float(np.mean(y_true == 1))
    mean_prob = float(np.mean(y_prob))

    return {
        "date_level_acc": round(float(date_acc), 4),
        "pooled_acc": round(float(pooled_acc), 4),
        "balanced_acc": round(float(balanced_acc), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "predicted_up_share": round(pred_up_share, 4),
        "actual_up_rate": round(actual_up_rate, 4),
        "mean_predicted_prob": round(mean_prob, 4),
        "is_lstm_collapsed": bool(pred_up_share < 0.40 or pred_up_share > 0.60),
    }


def run_audit_models():
    print("=" * 80, flush=True)
    print("AUDIT B3 & B4: PRICE-ONLY MODELS ON FULL NIFTY 50 UNIVERSE", flush=True)
    print("=" * 80, flush=True)

    master_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    df = pd.read_parquet(master_path)
    df["date"] = pd.to_datetime(df["date"])

    tech_cols = FEATURE_GROUPS["group_a_technical"]
    mask_date = (df["date"] >= pd.Timestamp("2021-01-01")) & (df["date"] <= pd.Timestamp("2026-09-30"))
    mask_target = df["target"].notna()
    mask_tech = df[tech_cols].notna().all(axis=1)

    cohort_df = df[mask_date & mask_target & mask_tech].copy()
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Compute naive baselines
    cohort_df["baseline_always_up"] = 1
    cohort_df["baseline_prev_dir"] = (cohort_df["return_1d"] > 0).astype(int)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(cohort_df)
    print(f"Total Protocol B Folds: {len(folds)}", flush=True)

    pred_dir = Path("experiments/audit/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    xgb_pred_file = pred_dir / "xgb_p_predictions.parquet"
    lr_pred_file = pred_dir / "lr_p_predictions.parquet"
    lstm_pred_file = pred_dir / "lstm_p_predictions.parquet"

    # =========================================================
    # Step 1: Run XGB-P and Logistic Regression (All 45 Folds)
    # =========================================================
    if not xgb_pred_file.exists() or not lr_pred_file.exists():
        print("\n--- Running XGB-P and Logistic Regression Baseline (45 Folds) ---", flush=True)
        xgb_preds = []
        lr_preds = []

        for fold in folds:
            train_df = cohort_df.iloc[fold.train_indices].copy()
            val_df = cohort_df.iloc[fold.val_indices].copy() if fold.val_indices is not None else None
            test_df = cohort_df.iloc[fold.test_indices].copy()

            y_true_test = test_df["target"].values.astype(int)
            dates_test = test_df["date"].dt.strftime("%Y-%m-%d").values
            tickers_test = test_df["ticker"].values

            # XGBoost
            xgb = FinFusionXGBoost(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                tree_method="hist",
                n_jobs=4,
            )
            xgb.fit(
                train_df=train_df,
                feature_cols=tech_cols,
                target_col="target",
                ticker_col="ticker",
                val_df=val_df,
                early_stopping_rounds=50,
            )
            probs_xgb = xgb.predict_proba(test_df, feature_cols=tech_cols, ticker_col="ticker")
            preds_xgb = (probs_xgb >= 0.50).astype(int)

            xgb_preds.append(pd.DataFrame({
                "fold_idx": fold.fold_idx,
                "date": dates_test,
                "ticker": tickers_test,
                "y_true": y_true_test,
                "y_pred": preds_xgb,
                "y_prob": probs_xgb,
                "baseline_always_up": test_df["baseline_always_up"].values,
                "baseline_prev_dir": test_df["baseline_prev_dir"].values,
            }))

            # Logistic Regression
            X_tr = train_df[tech_cols].values
            y_tr = train_df["target"].values.astype(int)
            X_te = test_df[tech_cols].values

            mu = X_tr.mean(axis=0)
            sig = X_tr.std(axis=0)
            sig[sig == 0] = 1.0

            lr = LogisticRegression(C=0.1, max_iter=500, random_state=42)
            lr.fit((X_tr - mu) / sig, y_tr)
            probs_lr = lr.predict_proba((X_te - mu) / sig)[:, 1]
            preds_lr = (probs_lr >= 0.50).astype(int)

            lr_preds.append(pd.DataFrame({
                "fold_idx": fold.fold_idx,
                "date": dates_test,
                "ticker": tickers_test,
                "y_true": y_true_test,
                "y_pred": preds_lr,
                "y_prob": probs_lr,
                "baseline_always_up": test_df["baseline_always_up"].values,
                "baseline_prev_dir": test_df["baseline_prev_dir"].values,
            }))

            if (fold.fold_idx + 1) % 10 == 0 or (fold.fold_idx + 1) == len(folds):
                print(f"  Completed XGB/LR Fold {fold.fold_idx + 1:02d} / {len(folds)}", flush=True)

        all_xgb = pd.concat(xgb_preds, ignore_index=True)
        all_lr = pd.concat(lr_preds, ignore_index=True)
        all_xgb.to_parquet(xgb_pred_file, index=False)
        all_lr.to_parquet(lr_pred_file, index=False)
        print("Saved XGB and LR predictions to parquet.", flush=True)
    else:
        print(f"Loading existing XGB and LR predictions from {pred_dir}...", flush=True)
        all_xgb = pd.read_parquet(xgb_pred_file)
        all_lr = pd.read_parquet(lr_pred_file)

    # =========================================================
    # Step 2: Run LSTM-P (5 Seeds with Checkpointing)
    # =========================================================
    lstm_ckpt_dir = pred_dir / "lstm_folds"
    lstm_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not lstm_pred_file.exists():
        print("\n--- Running LSTM-P across 5 Seeds (Seeds 42, 43, 44, 45, 46) ---", flush=True)
        seeds = [42, 43, 44, 45, 46]

        for fold in folds:
            fold_ckpt = lstm_ckpt_dir / f"lstm_fold_{fold.fold_idx:02d}.parquet"
            if fold_ckpt.exists():
                continue

            print(f"  Fitting LSTM-P Fold {fold.fold_idx + 1:02d} / {len(folds)}...", flush=True)
            train_df = cohort_df.iloc[fold.train_indices].copy()
            val_df = cohort_df.iloc[fold.val_indices].copy() if fold.val_indices is not None else None

            prior_dates = fold.val_dates if (fold.val_dates and len(fold.val_dates) > 0) else fold.train_dates
            X_tr_seq, y_tr_seq, _, _, _ = generate_ticker_sequences(train_df, tech_cols, sequence_length=20)
            X_te_seq, y_te_seq, d_te, t_te, _, _ = extract_boundary_context_sequences(
                cohort_df,
                context_dates=prior_dates[-19:],
                eval_dates=fold.test_dates,
                feature_cols=tech_cols,
                sequence_length=20,
            )

            if len(X_te_seq) == 0 or len(X_tr_seq) == 0:
                continue

            fold_prob_dict = {}
            for s in seeds:
                lstm = FinFusionLSTMClassifier(
                    input_size=len(tech_cols),
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.20,
                    dense_dim=32,
                    learning_rate=0.001,
                    batch_size=512,
                    max_epochs=4,
                    random_state=s,
                )
                lstm.fit(X_tr_seq, y_tr_seq)
                probs_s = lstm.predict_proba(X_te_seq)
                fold_prob_dict[f"y_prob_seed_{s}"] = probs_s

            fold_df = pd.DataFrame({
                "fold_idx": fold.fold_idx,
                "date": pd.to_datetime(d_te).strftime("%Y-%m-%d"),
                "ticker": t_te,
                "y_true": y_te_seq,
                **fold_prob_dict,
            })
            # Average probability
            probs_matrix = np.column_stack([fold_df[f"y_prob_seed_{s}"].values for s in seeds])
            fold_df["y_prob"] = probs_matrix.mean(axis=1)
            fold_df["y_pred"] = (fold_df["y_prob"] >= 0.50).astype(int)

            fold_df.to_parquet(fold_ckpt, index=False)
            print(f"    Saved Fold {fold.fold_idx + 1:02d} predictions ({len(fold_df)} rows).", flush=True)

        # Merge all fold checkpoints
        all_fold_files = sorted(lstm_ckpt_dir.glob("lstm_fold_*.parquet"))
        all_lstm = pd.concat([pd.read_parquet(f) for f in all_fold_files], ignore_index=True)
        all_lstm.to_parquet(lstm_pred_file, index=False)
        print(f"Compiled all LSTM fold predictions to {lstm_pred_file} ({len(all_lstm)} rows).", flush=True)
    else:
        print(f"Loading existing LSTM predictions from {lstm_pred_file}...", flush=True)
        all_lstm = pd.read_parquet(lstm_pred_file)

    # =========================================================
    # Step 3: Compute Comprehensive Performance Metrics
    # =========================================================
    print("\n--- COMPUTING METRICS (FULL NIFTY 50 POOL) ---", flush=True)

    m_xgb = compute_metrics_dict(all_xgb["y_true"].values, all_xgb["y_pred"].values, all_xgb["y_prob"].values, all_xgb["date"].values)
    m_lr = compute_metrics_dict(all_lr["y_true"].values, all_lr["y_pred"].values, all_lr["y_prob"].values, all_lr["date"].values)
    m_lstm = compute_metrics_dict(all_lstm["y_true"].values, all_lstm["y_pred"].values, all_lstm["y_prob"].values, all_lstm["date"].values)

    # Baselines
    acc_always_up = float(np.mean(all_xgb["y_true"].values == all_xgb["baseline_always_up"].values))
    acc_prev_dir = float(np.mean(all_xgb["y_true"].values == all_xgb["baseline_prev_dir"].values))

    # LSTM Seed Spread Analysis
    seeds = [42, 43, 44, 45, 46]
    seed_accs = [accuracy_score(all_lstm["y_true"].values, (all_lstm[f"y_prob_seed_{s}"].values >= 0.50).astype(int)) for s in seeds]
    seed_aucs = [roc_auc_score(all_lstm["y_true"].values, all_lstm[f"y_prob_seed_{s}"].values) for s in seeds]
    seed_spread = {
        "min_acc": round(float(np.min(seed_accs)), 4),
        "max_acc": round(float(np.max(seed_accs)), 4),
        "std_acc": round(float(np.std(seed_accs)), 4),
        "min_auc": round(float(np.min(seed_aucs)), 4),
        "max_auc": round(float(np.max(seed_aucs)), 4),
    }

    # Date-level accuracies
    daily_xgb = all_xgb.groupby("date").apply(lambda g: np.mean(g["y_true"] == g["y_pred"]), include_groups=False).sort_index()
    daily_up = all_xgb.groupby("date").apply(lambda g: np.mean(g["y_true"] == g["baseline_always_up"]), include_groups=False).sort_index()
    daily_prev = all_xgb.groupby("date").apply(lambda g: np.mean(g["y_true"] == g["baseline_prev_dir"]), include_groups=False).sort_index()
    daily_lr = all_lr.groupby("date").apply(lambda g: np.mean(g["y_true"] == g["y_pred"]), include_groups=False).sort_index()

    daily_lstm = all_lstm.groupby("date").apply(lambda g: np.mean(g["y_true"] == g["y_pred"]), include_groups=False).sort_index()
    common_lstm_dates = sorted(list(set(daily_lstm.index).intersection(set(daily_up.index))))

    daily_lstm_aligned = daily_lstm.loc[common_lstm_dates].values
    daily_up_lstm = daily_up.loc[common_lstm_dates].values
    daily_prev_lstm = daily_prev.loc[common_lstm_dates].values

    # Fractions of days beating Always-Up
    frac_beat_up_xgb = float(np.mean(daily_xgb.values > daily_up.values))
    frac_beat_up_lr = float(np.mean(daily_lr.values > daily_up.values))
    frac_beat_up_lstm = float(np.mean(daily_lstm_aligned > daily_up_lstm))

    # Statistical Bootstrapping: Model minus baselines (10,000 reps, L=10)
    print("Running Paired Date-Block Bootstraps (10,000 resamples, L=10)...", flush=True)
    boot_xgb_vs_up = paired_date_block_bootstrap(daily_xgb.values, daily_up.values, block_length=10, n_bootstraps=10000)
    boot_xgb_vs_prev = paired_date_block_bootstrap(daily_xgb.values, daily_prev.values, block_length=10, n_bootstraps=10000)

    boot_lr_vs_up = paired_date_block_bootstrap(daily_lr.values, daily_up.values, block_length=10, n_bootstraps=10000)
    boot_lr_vs_prev = paired_date_block_bootstrap(daily_lr.values, daily_prev.values, block_length=10, n_bootstraps=10000)

    boot_lstm_vs_up = paired_date_block_bootstrap(daily_lstm_aligned, daily_up_lstm, block_length=10, n_bootstraps=10000)
    boot_lstm_vs_prev = paired_date_block_bootstrap(daily_lstm_aligned, daily_prev_lstm, block_length=10, n_bootstraps=10000)

    # Format master metrics table
    metrics_records = [
        {
            "model": "XGB-P (Price-Only)",
            **m_xgb,
            "baseline_always_up": round(acc_always_up, 4),
            "baseline_prev_dir": round(acc_prev_dir, 4),
            "frac_days_beat_always_up": round(frac_beat_up_xgb, 4),
            "lift_over_always_up": round(m_xgb["pooled_acc"] - acc_always_up, 4),
            "ci_vs_always_up": f"[{boot_xgb_vs_up['ci_lower']:.4f}, {boot_xgb_vs_up['ci_upper']:.4f}]",
            "p_val_vs_always_up": round(boot_xgb_vs_up["p_value_one_sided"], 4),
            "lift_over_prev_dir": round(m_xgb["pooled_acc"] - acc_prev_dir, 4),
            "ci_vs_prev_dir": f"[{boot_xgb_vs_prev['ci_lower']:.4f}, {boot_xgb_vs_prev['ci_upper']:.4f}]",
            "p_val_vs_prev_dir": round(boot_xgb_vs_prev["p_value_one_sided"], 4),
        },
        {
            "model": "LSTM-P (5-Seed Avg)",
            **m_lstm,
            "baseline_always_up": round(acc_always_up, 4),
            "baseline_prev_dir": round(acc_prev_dir, 4),
            "frac_days_beat_always_up": round(frac_beat_up_lstm, 4),
            "lift_over_always_up": round(m_lstm["pooled_acc"] - acc_always_up, 4),
            "ci_vs_always_up": f"[{boot_lstm_vs_up['ci_lower']:.4f}, {boot_lstm_vs_up['ci_upper']:.4f}]",
            "p_val_vs_always_up": round(boot_lstm_vs_up["p_value_one_sided"], 4),
            "lift_over_prev_dir": round(m_lstm["pooled_acc"] - acc_prev_dir, 4),
            "ci_vs_prev_dir": f"[{boot_lstm_vs_prev['ci_lower']:.4f}, {boot_lstm_vs_prev['ci_upper']:.4f}]",
            "p_val_vs_prev_dir": round(boot_lstm_vs_prev["p_value_one_sided"], 4),
        },
        {
            "model": "Logistic Regression (Linear)",
            **m_lr,
            "baseline_always_up": round(acc_always_up, 4),
            "baseline_prev_dir": round(acc_prev_dir, 4),
            "frac_days_beat_always_up": round(frac_beat_up_lr, 4),
            "lift_over_always_up": round(m_lr["pooled_acc"] - acc_always_up, 4),
            "ci_vs_always_up": f"[{boot_lr_vs_up['ci_lower']:.4f}, {boot_lr_vs_up['ci_upper']:.4f}]",
            "p_val_vs_always_up": round(boot_lr_vs_up["p_value_one_sided"], 4),
            "lift_over_prev_dir": round(m_lr["pooled_acc"] - acc_prev_dir, 4),
            "ci_vs_prev_dir": f"[{boot_lr_vs_prev['ci_lower']:.4f}, {boot_lr_vs_prev['ci_upper']:.4f}]",
            "p_val_vs_prev_dir": round(boot_lr_vs_prev["p_value_one_sided"], 4),
        },
    ]

    metrics_df = pd.DataFrame(metrics_records)
    metrics_csv = Path("experiments/audit/model_metrics_full_universe.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    print("\n--- FULL UNIVERSE MODEL PERFORMANCE ---", flush=True)
    print(metrics_df.to_string(index=False), flush=True)
    print(f"\nLSTM Seed Spread (5 Seeds): {seed_spread}", flush=True)

    # =========================================================
    # Step 4: Heavyweights (10) vs Others (40) Stratification
    # =========================================================
    print("\n--- B4: STRATIFIED BREAKDOWN (HEAVYWEIGHTS VS REST) ---", flush=True)

    all_xgb["is_heavyweight"] = all_xgb["ticker"].isin(HEAVYWEIGHTS)
    all_lstm["is_heavyweight"] = all_lstm["ticker"].isin(HEAVYWEIGHTS)

    hw_records = []
    for grp_name, flag in [("10 Heavyweights", True), ("Non-Heavyweights (Rest)", False)]:
        sub_xgb = all_xgb[all_xgb["is_heavyweight"] == flag]
        sub_lstm = all_lstm[all_lstm["is_heavyweight"] == flag]

        m_hw_xgb = compute_metrics_dict(sub_xgb["y_true"].values, sub_xgb["y_pred"].values, sub_xgb["y_prob"].values, sub_xgb["date"].values)
        m_hw_lstm = compute_metrics_dict(sub_lstm["y_true"].values, sub_lstm["y_pred"].values, sub_lstm["y_prob"].values, sub_lstm["date"].values)

        hw_records.append({
            "cohort_subset": grp_name,
            "model": "XGB-P",
            "n_observations": len(sub_xgb),
            "date_level_acc": m_hw_xgb["date_level_acc"],
            "pooled_acc": m_hw_xgb["pooled_acc"],
            "roc_auc": m_hw_xgb["roc_auc"],
            "predicted_up_share": m_hw_xgb["predicted_up_share"],
        })
        hw_records.append({
            "cohort_subset": grp_name,
            "model": "LSTM-P",
            "n_observations": len(sub_lstm),
            "date_level_acc": m_hw_lstm["date_level_acc"],
            "pooled_acc": m_hw_lstm["pooled_acc"],
            "roc_auc": m_hw_lstm["roc_auc"],
            "predicted_up_share": m_hw_lstm["predicted_up_share"],
        })

    hw_df = pd.DataFrame(hw_records)
    hw_csv = Path("experiments/audit/heavyweights_vs_others.csv")
    hw_df.to_csv(hw_csv, index=False)
    print(hw_df.to_string(index=False), flush=True)

    return metrics_df, hw_df


if __name__ == "__main__":
    run_audit_models()
