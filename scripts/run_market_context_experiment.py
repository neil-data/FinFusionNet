"""Task 6: Complete Market-Level Context Experiment (X2 & L2).

Evaluates the marginal predictive power of Group B (Market Context) over Group A (Technical Indicators):
- X1 (XGBoost, Group A) vs X2 (XGBoost, Group A + Group B)
- Linear-1 (Logistic Regression, Group A) vs Linear-2 (Logistic Regression, Group A + Group B)
- L1 (LSTM, Group A) vs L2 (LSTM, Group A + Group B)
- Baselines: Always-Up, Previous-Day-Direction

Statistical Testing:
- Primary Hypothesis: H0: Acc(X2) <= Acc(X1) vs H1: Acc(X2) > Acc(X1)
- Secondary Hypothesis: H0: Acc(L2) <= Acc(L1) vs H1: Acc(L2) > Acc(L1)
- Paired Date-Block Bootstrap (10,000 resamples, L=10 days).

Outputs:
- experiments/audit/market_context_experiment_results.csv
- experiments/audit/predictions/market_context_predictions.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss

torch.set_num_threads(6)

from finfusionnet.evaluation.pipeline import FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.evaluation.lstm_pipeline import extract_boundary_context_sequences
from finfusionnet.data.sequence_generator import generate_ticker_sequences
from finfusionnet.models.xgboost_model import FinFusionXGBoost
from finfusionnet.models.lstm_model import FinFusionLSTMClassifier
from finfusionnet.evaluation.stats_tests import paired_date_block_bootstrap


def compute_metrics(y_true, y_pred, y_prob, dates_series):
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

    return {
        "date_level_acc": round(float(date_acc), 4),
        "pooled_acc": round(float(pooled_acc), 4),
        "balanced_acc": round(float(balanced_acc), 4),
        "roc_auc": round(float(auc), 4),
        "brier_score": round(float(brier), 4),
        "log_loss": round(float(ll), 4),
        "predicted_up_share": round(pred_up_share, 4),
    }


def run_market_context_experiment():
    print("=" * 80)
    print("TASK 6: MARKET-LEVEL CONTEXT EXPERIMENT (X2 & L2 vs X1 & L1)")
    print("=" * 80)

    dataset_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    df = pd.read_parquet(dataset_path)
    df["date"] = pd.to_datetime(df["date"])

    group_a = FEATURE_GROUPS["group_a_technical"]
    group_b = FEATURE_GROUPS["group_b_market_context"]
    group_ab = group_a + group_b

    mask_date = (df["date"] >= pd.Timestamp("2021-01-01")) & (df["date"] <= pd.Timestamp("2026-09-30"))
    mask_target = df["target"].notna()
    mask_features = df[group_ab].notna().all(axis=1)

    cohort_df = df[mask_date & mask_target & mask_features].copy()
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    cohort_df["baseline_always_up"] = 1
    cohort_df["baseline_prev_dir"] = (cohort_df["return_1d"] > 0).astype(int)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(cohort_df)
    print(f"Total Protocol B Folds: {len(folds)} | Total Cohort Rows: {len(cohort_df)}")

    pred_dir = Path("experiments/audit/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_file = pred_dir / "market_context_predictions.parquet"
    fold_ckpt_dir = pred_dir / "market_context_folds"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not pred_file.exists():
        records = []
        print("\n--- Training Walk-Forward Models across 45 Folds ---")
        for fold in folds:
            fold_ckpt = fold_ckpt_dir / f"fold_{fold.fold_idx:02d}.parquet"
            if fold_ckpt.exists():
                continue

            train_df = cohort_df.iloc[fold.train_indices].copy()
            val_df = cohort_df.iloc[fold.val_indices].copy() if fold.val_indices is not None else None
            test_df = cohort_df.iloc[fold.test_indices].copy()

            y_tr = train_df["target"].values.astype(int)
            y_te = test_df["target"].values.astype(int)
            dates_te = test_df["date"].dt.strftime("%Y-%m-%d").values
            tickers_te = test_df["ticker"].values

            # 1. XGBoost X1 (Group A)
            xgb_x1 = FinFusionXGBoost(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
            xgb_x1.fit(train_df, group_a, "target", ticker_col="ticker", val_df=val_df)
            prob_x1 = xgb_x1.predict_proba(test_df, group_a)

            # 2. XGBoost X2 (Group A + Group B)
            xgb_x2 = FinFusionXGBoost(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=4)
            xgb_x2.fit(train_df, group_ab, "target", ticker_col="ticker", val_df=val_df)
            prob_x2 = xgb_x2.predict_proba(test_df, group_ab)

            # 3. Linear-1 (Group A)
            mean_a = train_df[group_a].mean()
            std_a = train_df[group_a].std().replace(0, 1)
            X_tr_a = ((train_df[group_a] - mean_a) / std_a).fillna(0).values
            X_te_a = ((test_df[group_a] - mean_a) / std_a).fillna(0).values

            lr_1 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            lr_1.fit(X_tr_a, y_tr)
            prob_lr1 = lr_1.predict_proba(X_te_a)[:, 1]

            # 4. Linear-2 (Group A + Group B)
            mean_ab = train_df[group_ab].mean()
            std_ab = train_df[group_ab].std().replace(0, 1)
            X_tr_ab = ((train_df[group_ab] - mean_ab) / std_ab).fillna(0).values
            X_te_ab = ((test_df[group_ab] - mean_ab) / std_ab).fillna(0).values

            lr_2 = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
            lr_2.fit(X_tr_ab, y_tr)
            prob_lr2 = lr_2.predict_proba(X_te_ab)[:, 1]

            # 5. LSTM L1 (Group A) & L2 (Group A + Group B)
            prior_dates = fold.val_dates if (fold.val_dates and len(fold.val_dates) > 0) else fold.train_dates
            X_tr_seq_a, y_tr_seq_a, _, _, _ = generate_ticker_sequences(train_df, group_a, sequence_length=20)
            X_te_seq_a, _, d_te_a, t_te_a, _, _ = extract_boundary_context_sequences(
                cohort_df, prior_dates[-19:], fold.test_dates, group_a, sequence_length=20
            )

            X_tr_seq_ab, y_tr_seq_ab, _, _, _ = generate_ticker_sequences(train_df, group_ab, sequence_length=20)
            X_te_seq_ab, _, _, _, _, _ = extract_boundary_context_sequences(
                cohort_df, prior_dates[-19:], fold.test_dates, group_ab, sequence_length=20
            )

            lstm_l1 = FinFusionLSTMClassifier(input_size=len(group_a), hidden_dim=64, num_layers=2, batch_size=512, max_epochs=4, random_state=42)
            lstm_l1.fit(X_tr_seq_a, y_tr_seq_a)
            prob_l1 = lstm_l1.predict_proba(X_te_seq_a)

            lstm_l2 = FinFusionLSTMClassifier(input_size=len(group_ab), hidden_dim=64, num_layers=2, batch_size=512, max_epochs=4, random_state=42)
            lstm_l2.fit(X_tr_seq_ab, y_tr_seq_ab)
            prob_l2 = lstm_l2.predict_proba(X_te_seq_ab)

            fold_df = pd.DataFrame({
                "fold_idx": fold.fold_idx,
                "date": dates_te,
                "ticker": tickers_te,
                "y_true": y_te,
                "prob_xgb_x1": prob_x1,
                "prob_xgb_x2": prob_x2,
                "prob_lr_1": prob_lr1,
                "prob_lr_2": prob_lr2,
                "baseline_always_up": test_df["baseline_always_up"].values,
                "baseline_prev_dir": test_df["baseline_prev_dir"].values,
            })

            lstm_df = pd.DataFrame({
                "date": pd.to_datetime(d_te_a).strftime("%Y-%m-%d"),
                "ticker": t_te_a,
                "prob_lstm_l1": prob_l1,
                "prob_lstm_l2": prob_l2,
            })

            merged_fold_df = fold_df.merge(lstm_df, on=["date", "ticker"], how="inner")
            merged_fold_df.to_parquet(fold_ckpt, index=False)
            if (fold.fold_idx + 1) % 5 == 0 or (fold.fold_idx + 1) == len(folds):
                print(f"  Completed Fold {fold.fold_idx + 1:02d} / {len(folds)} ({len(merged_fold_df)} rows)", flush=True)

        all_fold_files = sorted(fold_ckpt_dir.glob("fold_*.parquet"))
        all_preds = pd.concat([pd.read_parquet(f) for f in all_fold_files], ignore_index=True)
        all_preds.to_parquet(pred_file, index=False)
        print(f"Saved market context predictions to {pred_file} ({len(all_preds)} rows).")
    else:
        print(f"Loading existing market context predictions from {pred_file}...")
        all_preds = pd.read_parquet(pred_file)

    # Compute Metrics for all configurations
    configs = [
        ("XGB-P (X1: Group A Only)", "prob_xgb_x1"),
        ("XGB-P (X2: Group A + Group B)", "prob_xgb_x2"),
        ("Linear-1 (Group A Only)", "prob_lr_1"),
        ("Linear-2 (Group A + Group B)", "prob_lr_2"),
        ("LSTM-P (L1: Group A Only)", "prob_lstm_l1"),
        ("LSTM-P (L2: Group A + Group B)", "prob_lstm_l2"),
    ]

    results = []
    daily_acc_dict = {}

    for name, col_p in configs:
        p = all_preds[col_p].values
        pred = (p >= 0.50).astype(int)
        m = compute_metrics(all_preds["y_true"].values, pred, p, all_preds["date"].values)
        results.append({"configuration": name, **m})

        daily_acc = all_preds.groupby("date").apply(
            lambda g: accuracy_score(g["y_true"], (g[col_p] >= 0.50).astype(int)),
            include_groups=False
        ).sort_index()
        daily_acc_dict[name] = daily_acc

    # Baseline daily accuracy
    daily_up = all_preds.groupby("date").apply(
        lambda g: accuracy_score(g["y_true"], g["baseline_always_up"]),
        include_groups=False
    ).sort_index()

    # Paired Date-Block Bootstraps (X2 vs X1 and L2 vs L1)
    print("\nRunning Paired Date-Block Bootstraps (10,000 resamples, L=10)...")
    boot_x2_vs_x1 = paired_date_block_bootstrap(
        daily_acc_dict["XGB-P (X2: Group A + Group B)"].values,
        daily_acc_dict["XGB-P (X1: Group A Only)"].values,
        block_length=10, n_bootstraps=10000
    )
    boot_l2_vs_l1 = paired_date_block_bootstrap(
        daily_acc_dict["LSTM-P (L2: Group A + Group B)"].values,
        daily_acc_dict["LSTM-P (L1: Group A Only)"].values,
        block_length=10, n_bootstraps=10000
    )
    boot_lr2_vs_lr1 = paired_date_block_bootstrap(
        daily_acc_dict["Linear-2 (Group A + Group B)"].values,
        daily_acc_dict["Linear-1 (Group A Only)"].values,
        block_length=10, n_bootstraps=10000
    )

    res_df = pd.DataFrame(results)
    
    # Add Bootstrap lift comparisons
    res_df["lift_over_base"] = [
        "Base",
        f"{boot_x2_vs_x1['observed_mean_difference']:+.4f} [{boot_x2_vs_x1['ci_lower']:.4f}, {boot_x2_vs_x1['ci_upper']:.4f}] (p={boot_x2_vs_x1['p_value_one_sided']:.4f})",
        "Base",
        f"{boot_lr2_vs_lr1['observed_mean_difference']:+.4f} [{boot_lr2_vs_lr1['ci_lower']:.4f}, {boot_lr2_vs_lr1['ci_upper']:.4f}] (p={boot_lr2_vs_lr1['p_value_one_sided']:.4f})",
        "Base",
        f"{boot_l2_vs_l1['observed_mean_difference']:+.4f} [{boot_l2_vs_l1['ci_lower']:.4f}, {boot_l2_vs_l1['ci_upper']:.4f}] (p={boot_l2_vs_l1['p_value_one_sided']:.4f})",
    ]

    out_csv = Path("experiments/audit/market_context_experiment_results.csv")
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved market context experiment results to {out_csv}")
    print(res_df.to_string(index=False))

    return res_df


if __name__ == "__main__":
    run_market_context_experiment()
