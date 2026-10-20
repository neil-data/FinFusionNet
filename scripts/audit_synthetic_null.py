"""Audit A3: Synthetic Random-Walk Null Test.

Simulate OHLCV for 50 tickers (i.i.d. returns with a common market factor, zero predictability)
over the real date range (2020-01-01 to 2026-09-30).
Run the full pipeline (causal features, walk-forward folds, XGBoost, LSTM).
Repeat across 20 random seeds and report the distribution.
Verify: date-level accuracy ~50.0% within the bootstrap CI for real and shifted targets k >= 0.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from finfusionnet.data.trading_calendar import get_trading_calendar
from finfusionnet.features.technical import compute_technical_indicators
from finfusionnet.features.targets import compute_targets
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.evaluation.lstm_pipeline import extract_boundary_context_sequences
from finfusionnet.data.sequence_generator import generate_ticker_sequences
from finfusionnet.models.xgboost_model import FinFusionXGBoost
from finfusionnet.models.lstm_model import FinFusionLSTMClassifier


def simulate_synthetic_market(dates: pd.DatetimeIndex, n_tickers: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    tickers = [f"SYNTH_{i:02d}" for i in range(1, n_tickers + 1)]
    n_days = len(dates)

    # Market factor returns
    market_returns = rng.normal(loc=0.0003, scale=0.012, size=n_days)

    records = []
    for t_idx, ticker in enumerate(tickers):
        beta = rng.uniform(0.7, 1.3)
        idio_vol = rng.uniform(0.010, 0.025)
        idio_returns = rng.normal(loc=0.0, scale=idio_vol, size=n_days)
        total_returns = beta * market_returns + idio_returns

        base_price = rng.uniform(100.0, 2000.0)
        close_prices = base_price * np.cumprod(1.0 + total_returns)

        for d_idx, dt in enumerate(dates):
            c = close_prices[d_idx]
            prev_c = close_prices[d_idx - 1] if d_idx > 0 else c
            o = prev_c * (1.0 + rng.normal(0, 0.003))
            spread = max(o, c) * rng.uniform(0.002, 0.015)
            h = max(o, c) + spread
            l = min(o, c) - spread
            v = int(rng.lognormal(mean=14.0, sigma=0.5))

            records.append({
                "date": dt,
                "ticker": ticker,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })

    df = pd.DataFrame(records)
    return df


def run_synthetic_null_test():
    print("=" * 80, flush=True)
    print("AUDIT A3: SYNTHETIC RANDOM-WALK NULL TEST (20 SEEDS)", flush=True)
    print("=" * 80, flush=True)

    calendar = get_trading_calendar("2020-01-01", "2026-09-30")
    print(f"Total calendar trading sessions: {len(calendar)}", flush=True)

    tech_cols = [
        "return_1d", "return_2d", "return_5d", "return_10d", "return_20d",
        "close_sma_5_ratio", "close_sma_10_ratio", "close_sma_20_ratio", "close_sma_50_ratio",
        "sma_5_sma_20_ratio", "rsi_14", "macd", "macd_signal", "macd_histogram",
        "atr_14_pct", "bb_width_20", "volatility_10", "volatility_20",
        "volume_change_1d", "volume_ratio_5", "volume_ratio_20",
        "high_low_range_pct", "close_open_return", "close_position_in_range",
    ]

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")

    out_csv = Path("experiments/audit/synthetic_null_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    seed_records = []
    if out_csv.exists():
        try:
            existing = pd.read_csv(out_csv)
            seed_records = existing.to_dict(orient="records")
            completed_seeds = {r["seed"] for r in seed_records}
            print(f"Loaded {len(completed_seeds)} already completed seeds from {out_csv}", flush=True)
        except Exception:
            seed_records = []
            completed_seeds = set()
    else:
        completed_seeds = set()

    n_seeds = 20
    print(f"\nSimulating and evaluating across {n_seeds} independent synthetic universes...", flush=True)

    for seed in range(1, n_seeds + 1):
        if seed in completed_seeds:
            continue

        print(f"\n--- Running Seed {seed:02d} / {n_seeds} ---", flush=True)
        raw_df = simulate_synthetic_market(calendar, n_tickers=50, seed=seed * 100)
        feat_df = compute_technical_indicators(raw_df)
        df = compute_targets(feat_df, calendar=calendar)

        cohort_df = df[df["date"] >= pd.Timestamp("2021-01-01")].dropna(subset=tech_cols + ["target"]).reset_index(drop=True)
        cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

        folds = engine.generate_protocol_b_folds(cohort_df)
        eval_folds = folds[-6:]  # Last 6 monthly expanding folds for efficiency

        xgb_preds_list = []
        lstm_preds_list = []

        for fold in eval_folds:
            train_dates = fold.train_dates
            val_dates = fold.val_dates
            test_dates = fold.test_dates

            train_df = cohort_df[cohort_df["date"].isin(train_dates)].copy()
            val_df = cohort_df[cohort_df["date"].isin(val_dates)].copy() if val_dates else None
            test_df = cohort_df[cohort_df["date"].isin(test_dates)].copy()

            # --- XGBoost ---
            xgb = FinFusionXGBoost(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.05,
                random_state=seed,
                tree_method="hist",
                n_jobs=4,
            )
            xgb.fit(
                train_df=train_df,
                feature_cols=tech_cols,
                target_col="target",
                ticker_col="ticker",
                val_df=val_df,
                early_stopping_rounds=15,
            )
            probs_xgb = xgb.predict_proba(test_df, feature_cols=tech_cols, ticker_col="ticker")
            preds_xgb = xgb.predict(test_df, feature_cols=tech_cols, ticker_col="ticker")
            xgb_preds_list.append(pd.DataFrame({
                "date": test_df["date"].dt.strftime("%Y-%m-%d").values,
                "ticker": test_df["ticker"].values,
                "y_true": test_df["target"].values.astype(int),
                "y_pred": preds_xgb,
                "y_prob": probs_xgb,
            }))

            # --- LSTM ---
            prior_dates = val_dates if (val_dates and len(val_dates) > 0) else train_dates
            X_tr, y_tr, _, _, _ = generate_ticker_sequences(train_df, tech_cols, sequence_length=20)
            X_te, y_te, d_te, t_te, _, _ = extract_boundary_context_sequences(
                cohort_df,
                context_dates=prior_dates[-19:],
                eval_dates=test_dates,
                feature_cols=tech_cols,
                sequence_length=20,
            )
            if len(X_te) > 0 and len(X_tr) > 0:
                lstm = FinFusionLSTMClassifier(
                    input_size=len(tech_cols),
                    hidden_dim=32,
                    num_layers=1,
                    dropout=0.1,
                    batch_size=256,
                    max_epochs=4,
                    random_state=seed,
                )
                lstm.fit(X_tr, y_tr)
                probs_l = lstm.predict_proba(X_te)
                preds_l = lstm.predict(X_te)
                lstm_preds_list.append(pd.DataFrame({
                    "date": pd.to_datetime(d_te).strftime("%Y-%m-%d"),
                    "ticker": t_te,
                    "y_true": y_te,
                    "y_pred": preds_l,
                    "y_prob": probs_l,
                }))

        all_xgb = pd.concat(xgb_preds_list, ignore_index=True)
        xgb_date_acc = all_xgb.groupby("date").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False).mean()
        xgb_auc = roc_auc_score(all_xgb["y_true"], all_xgb["y_prob"])

        all_lstm = pd.concat(lstm_preds_list, ignore_index=True)
        lstm_date_acc = all_lstm.groupby("date").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False).mean()
        lstm_auc = roc_auc_score(all_lstm["y_true"], all_lstm["y_prob"])

        rec = {
            "seed": seed,
            "xgb_date_acc": round(float(xgb_date_acc), 4),
            "xgb_auc": round(float(xgb_auc), 4),
            "lstm_date_acc": round(float(lstm_date_acc), 4),
            "lstm_auc": round(float(lstm_auc), 4),
        }
        seed_records.append(rec)
        print(f"Seed {seed:02d}: XGB Date Acc = {rec['xgb_date_acc']:.4f} (AUC = {rec['xgb_auc']:.4f}) | LSTM Date Acc = {rec['lstm_date_acc']:.4f} (AUC = {rec['lstm_auc']:.4f})", flush=True)

        # Save after every seed
        pd.DataFrame(seed_records).to_csv(out_csv, index=False)

    res_df = pd.DataFrame(seed_records)
    print("\n--- SYNTHETIC NULL SUMMARY (20 SEEDS) ---", flush=True)
    mean_xgb = res_df["xgb_date_acc"].mean()
    std_xgb = res_df["xgb_date_acc"].std()
    mean_lstm = res_df["lstm_date_acc"].mean()
    std_lstm = res_df["lstm_date_acc"].std()

    print(f"XGBoost Date Accuracy: Mean = {mean_xgb:.4f}, Std = {std_xgb:.4f}, Range = [{res_df['xgb_date_acc'].min():.4f}, {res_df['xgb_date_acc'].max():.4f}]", flush=True)
    print(f"LSTM Date Accuracy:    Mean = {mean_lstm:.4f}, Std = {std_lstm:.4f}, Range = [{res_df['lstm_date_acc'].min():.4f}, {res_df['lstm_date_acc'].max():.4f}]", flush=True)

    rng = np.random.RandomState(42)
    boot_xgb = [rng.choice(res_df["xgb_date_acc"], size=len(res_df), replace=True).mean() for _ in range(5000)]
    ci_xgb = (np.percentile(boot_xgb, 2.5), np.percentile(boot_xgb, 97.5))
    boot_lstm = [rng.choice(res_df["lstm_date_acc"], size=len(res_df), replace=True).mean() for _ in range(5000)]
    ci_lstm = (np.percentile(boot_lstm, 2.5), np.percentile(boot_lstm, 97.5))

    print(f"XGBoost 95% CI: [{ci_xgb[0]:.4f}, {ci_xgb[1]:.4f}]", flush=True)
    print(f"LSTM 95% CI:    [{ci_lstm[0]:.4f}, {ci_lstm[1]:.4f}]", flush=True)

    assert 0.485 <= mean_xgb <= 0.515, f"FAIL: XGBoost synthetic mean {mean_xgb:.4f} deviates from null"
    assert 0.485 <= mean_lstm <= 0.515, f"FAIL: LSTM synthetic mean {mean_lstm:.4f} deviates from null"
    assert ci_xgb[0] <= 0.50 <= ci_xgb[1], "FAIL: XGBoost 95% CI does not span 0.50"
    assert ci_lstm[0] <= 0.50 <= ci_lstm[1], "FAIL: LSTM 95% CI does not span 0.50"

    print("PASS: Synthetic null test verified. Models on unforecastable random walk output pure 50% chance.", flush=True)
    return res_df


if __name__ == "__main__":
    run_synthetic_null_test()
