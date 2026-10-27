"""FinFusionNet: Experiment 5 - Temporal Cross-Attention Multimodal Fusion.

Trains and evaluates a dual-stream Cross-Attention architecture:
- Query stream: Market sequence (20 sessions x 24 features).
- Key/Value stream: FinBERT daily features (20 sessions x 8 features) + learnable null-news sentinel token.
- Multi-task heads:
  1. Next-day direction classification (BCE loss, Sigmoid).
  2. Next-day Parkinson realized-volatility regression (Huber loss, Softplus).

Protocol B:
- 45 monthly expanding folds (2021-01 to 2026-09).
- 24-month initial training window.
- 1-session embargo.
- Boundary context preservation (zero test dates lost).
- In-fold scaling fit strictly on training data.
- Checkpointed per-fold predictions.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(8)

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finfusionnet.evaluation.pipeline import FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.features.sentiment import integrate_sentiment_features
from finfusionnet.features.volatility_target import compute_next_session_parkinson_volatility
from finfusionnet.data.sequence_generator import (
    generate_multimodal_sequences,
    extract_boundary_multimodal_sequences,
)
from finfusionnet.models.cross_attention_model import FinFusionCrossAttentionClassifier


EXPERIMENT_DIR = Path("experiments/experiment_5")
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
FOLD_CKPT_DIR = PREDICTIONS_DIR / "exp5_folds"


def run_experiment_5(
    smoke_test: bool = False,
    single_fold: int = -1,
    max_epochs: int = 5,
    batch_size: int = 512,
):
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    FOLD_CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINFUSIONNET EXPERIMENT 5: TEMPORAL CROSS-ATTENTION FUSION")
    print(f"Settings: max_epochs={max_epochs}, batch_size={batch_size}, smoke_test={smoke_test}, single_fold={single_fold}")
    print("=" * 80, flush=True)

    # 1. Load Audited Base Dataset
    master_path = Path("experiments/audit/master_universe_ticker_day.parquet")
    if not master_path.exists():
        master_path = Path("data/processed/master_dataset/master_ticker_day.parquet")
    print(f"Loading base price dataset from {master_path}...", flush=True)
    df_price = pd.read_parquet(master_path)
    df_price["date"] = pd.to_datetime(df_price["date"])

    # 2. Compute Next-Session Parkinson Realized Volatility Target
    print("Computing Parkinson annualized realized volatility targets (sigma_Parkinson, t+1)...", flush=True)
    df_price_vol = compute_next_session_parkinson_volatility(
        df_price,
        date_col="date",
        ticker_col="ticker",
        high_col="high",
        low_col="low",
        target_date_col="target_date",
        max_vol_clip=2.0,
    )

    # 3. Load Daily All-50 FinBERT Sentiment Features
    finbert_path = Path("data/processed/gnews_all50/daily_gnews_all50_finbert_features.parquet")
    if not finbert_path.exists():
        raise FileNotFoundError(f"All-50 FinBERT features not found at {finbert_path}. Run processing script first.")
    print(f"Loading All-50 FinBERT daily features from {finbert_path}...", flush=True)
    df_finbert = pd.read_parquet(finbert_path)
    df_finbert["date"] = pd.to_datetime(df_finbert["date"])

    # 4. Integrate Features Enforcing Missing-News Null Policy
    print("Integrating FinBERT sentiment with price & volatility targets...", flush=True)
    df_merged = integrate_sentiment_features(df_price_vol, df_finbert, date_col="date", ticker_col="ticker")

    # Define Feature Sets
    market_features = FEATURE_GROUPS["group_a_technical"]
    news_features = [
        "n_articles",
        "has_news",
        "finbert_positive_mean",
        "finbert_negative_mean",
        "finbert_neutral_mean",
        "finbert_sentiment_score",
        "finbert_dispersion",
        "finbert_confidence",
    ]

    print(f"Market Stream Features ({len(market_features)}): {market_features[:4]} ... {market_features[-2:]}", flush=True)
    print(f"News Stream Features ({len(news_features)}): {news_features}", flush=True)

    # 5. Build Common Evaluation Cohort Filter
    mask_date = (df_merged["date"] >= pd.Timestamp("2021-01-01")) & (df_merged["date"] <= pd.Timestamp("2026-09-30"))
    mask_target = df_merged["target"].notna()
    mask_market = df_merged[market_features].notna().all(axis=1)

    cohort_df = df_merged[mask_date & mask_target & mask_market].copy()
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Protocol B Walk-Forward Engine
    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    folds = engine.generate_protocol_b_folds(cohort_df)
    print(f"\nTotal Protocol B Expanding Folds: {len(folds)} | Common Cohort Rows: {len(cohort_df):,}", flush=True)

    if smoke_test:
        folds = [folds[0]]
        max_epochs = 2
        print(f"Smoke Test Active: Running Fold 0 only for {max_epochs} epochs.", flush=True)
    elif single_fold >= 0:
        folds = [f for f in folds if f.fold_idx == single_fold]
        print(f"Single Fold Active: Running Fold {single_fold} only.", flush=True)

    # 6. Walk-Forward Loop
    import time
    for fold in folds:
        t0 = time.time()
        fold_ckpt = FOLD_CKPT_DIR / f"fold_{fold.fold_idx:02d}.parquet"
        if fold_ckpt.exists() and not smoke_test and single_fold < 0:
            print(f"Fold {fold.fold_idx + 1:02d}/{len(folds)} already checkpointed. Skipping.", flush=True)
            continue

        train_df = cohort_df.iloc[fold.train_indices].copy()
        test_df = cohort_df.iloc[fold.test_indices].copy()

        prior_dates = fold.val_dates if (fold.val_dates and len(fold.val_dates) > 0) else fold.train_dates

        # Extract multimodal training sequences
        X_m_tr, X_n_tr, y_dir_tr, y_vol_tr, _, _, _ = generate_multimodal_sequences(
            train_df,
            market_cols=market_features,
            news_cols=news_features,
            sequence_length=20,
        )

        # Extract multimodal test sequences with boundary context (no lost test dates)
        X_m_te, X_n_te, y_dir_te, y_vol_te, d_te, t_te, td_te = extract_boundary_multimodal_sequences(
            cohort_df,
            context_dates=prior_dates[-19:],
            eval_dates=fold.test_dates,
            market_cols=market_features,
            news_cols=news_features,
            sequence_length=20,
        )

        # Initialize and train Cross-Attention Model
        clf = FinFusionCrossAttentionClassifier(
            d_market=len(market_features),
            d_news=len(news_features),
            d_model=64,
            n_heads=4,
            seq_len=20,
            dropout=0.20,
            dense_dim=32,
            learning_rate=0.001,
            weight_decay=1e-5,
            vol_loss_weight=1.0,
            batch_size=batch_size,
            max_epochs=max_epochs,
            early_stopping_patience=3,
            random_state=42,
            device="cpu",
        )

        clf.fit(X_m_tr, X_n_tr, y_dir_tr, y_vol_tr, verbose=smoke_test)

        # Generate test predictions
        prob_dir, pred_vol, pred_dir = clf.predict(X_m_te, X_n_te)

        # Build fold predictions dataframe
        fold_res_df = pd.DataFrame({
            "fold_idx": fold.fold_idx,
            "date": pd.to_datetime(d_te).strftime("%Y-%m-%d"),
            "ticker": t_te,
            "y_true": y_dir_te.astype(int),
            "target_realized_vol": y_vol_te,
            "prob_cross_attn": prob_dir,
            "pred_cross_attn_vol": pred_vol,
            "pred_cross_attn_dir": pred_dir,
        })

        # Attach test_df controls
        test_controls = test_df[["date", "ticker", "has_news", "n_articles"]].copy()
        test_controls["date"] = test_controls["date"].dt.strftime("%Y-%m-%d")
        test_controls = test_controls.rename(columns={"has_news": "has_news_all50", "n_articles": "n_articles_all50"})

        fold_res_df = fold_res_df.merge(test_controls, on=["date", "ticker"], how="left")

        if not smoke_test:
            fold_res_df.to_parquet(fold_ckpt, index=False)

        acc_dir = (fold_res_df["pred_cross_attn_dir"] == fold_res_df["y_true"]).mean()
        valid_vol = fold_res_df["target_realized_vol"].notna()
        rmse_vol = np.sqrt(np.mean((fold_res_df.loc[valid_vol, "pred_cross_attn_vol"] - fold_res_df.loc[valid_vol, "target_realized_vol"]) ** 2))
        mae_vol = np.mean(np.abs(fold_res_df.loc[valid_vol, "pred_cross_attn_vol"] - fold_res_df.loc[valid_vol, "target_realized_vol"]))

        print(
            f"  Fold {fold.fold_idx + 1:02d} / 45 Done | Samples: {len(fold_res_df):,} | "
            f"Dir Acc: {acc_dir:.4f} | Vol RMSE: {rmse_vol:.4f} | Vol MAE: {mae_vol:.4f} | ({time.time() - t0:.1f}s)",
            flush=True,
        )

    if smoke_test:
        print("\nSmoke test completed successfully! Architecture, pipeline, and predictions validated.")
        return

    # 7. Merge All 45 Folds with Prior Experiments
    all_fold_files = sorted(FOLD_CKPT_DIR.glob("fold_*.parquet"))
    if len(all_fold_files) < 45:
        print(f"\nWarning: Only {len(all_fold_files)}/45 folds completed so far.")
        return

    print("\nMerging all 45 fold predictions into Experiment 5 consolidated dataset...", flush=True)
    df_exp5_preds = pd.concat([pd.read_parquet(f) for f in all_fold_files], ignore_index=True)

    # Load Exp 4 consolidated predictions (contains F0, F1_VADER, F1_FinBERT)
    exp4_pred_path = Path("experiments/experiment_4/predictions/experiment_4_common_cohort_predictions.parquet")
    if not exp4_pred_path.exists():
        raise FileNotFoundError(f"Exp 4 predictions not found at {exp4_pred_path}.")

    print(f"Joining with Exp4 predictions from {exp4_pred_path}...", flush=True)
    df_exp4 = pd.read_parquet(exp4_pred_path)

    # Columns to bring from Exp4:
    # prob_xgb_f0, prob_lr_f0, prob_lstm_f0,
    # prob_xgb_f1_vader, prob_lr_f1_vader, prob_lstm_f1_vader,
    # prob_xgb_f1_finbert, prob_lr_f1_finbert, prob_lstm_f1_finbert
    common_cols = [
        "fold_idx", "date", "ticker", "y_true",
        "baseline_always_up", "baseline_prev_dir",
        "prob_xgb_f0", "prob_lr_f0", "prob_lstm_f0",
        "prob_xgb_f1_vader", "prob_lr_f1_vader", "prob_lstm_f1_vader",
        "prob_xgb_f1_finbert", "prob_lr_f1_finbert", "prob_lstm_f1_finbert",
    ]
    df_exp4_sub = df_exp4[[c for c in common_cols if c in df_exp4.columns]]

    df_consolidated = df_exp5_preds.merge(
        df_exp4_sub,
        on=["fold_idx", "date", "ticker"],
        how="inner",
        suffixes=("", "_exp4"),
    )

    if "y_true_exp4" in df_consolidated.columns:
        df_consolidated = df_consolidated.drop(columns=["y_true_exp4"])

    # Also bring trailing volatility features for volatility baselines
    df_vol_benchmarks = cohort_df[["date", "ticker", "volatility_10", "volatility_20"]].copy()
    df_vol_benchmarks["date"] = df_vol_benchmarks["date"].dt.strftime("%Y-%m-%d")
    df_consolidated = df_consolidated.merge(df_vol_benchmarks, on=["date", "ticker"], how="left")

    out_file = PREDICTIONS_DIR / "experiment_5_common_cohort_predictions.parquet"
    df_consolidated.to_parquet(out_file, index=False)
    print(f"Consolidated predictions saved to {out_file} ({len(df_consolidated):,} rows across {df_consolidated['date'].nunique()} dates).", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiment 5 Cross-Attention")
    parser.add_argument("--smoke-test", action="store_true", help="Run 1 fold for 2 epochs as a smoke test")
    parser.add_argument("--single-fold", type=int, default=-1, help="Run a specific fold index")
    parser.add_argument("--max-epochs", type=int, default=5, help="Max training epochs per fold")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    args = parser.parse_args()

    run_experiment_5(
        smoke_test=args.smoke_test,
        single_fold=args.single_fold,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
    )
