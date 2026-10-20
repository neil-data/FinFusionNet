"""Audit A4: Positive Controls (Signal Recovery Verification).

1. Perfect Leak Control: Add feature equal to the true next-day direction.
   Expected accuracy: ~100%.
2. Weak AR(1) Signal Control: Simulate returns with known autocorrelation rho = 0.20.
   Theoretical directional accuracy = 0.50 + (1/pi)*arcsin(0.20) ~= 56.41%.
   Verify the pipeline recovers accuracy centered near the theoretical expectation.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from finfusionnet.evaluation.pipeline import build_common_evaluation_cohort, FEATURE_GROUPS
from finfusionnet.evaluation.walk_forward import WalkForwardEngine
from finfusionnet.models.xgboost_model import FinFusionXGBoost
from finfusionnet.features.technical import compute_technical_indicators
from finfusionnet.features.targets import compute_targets
from finfusionnet.data.trading_calendar import get_trading_calendar


def run_positive_controls():
    print("=" * 80, flush=True)
    print("AUDIT A4: POSITIVE CONTROLS & SIGNAL RECOVERY VERIFICATION", flush=True)
    print("=" * 80, flush=True)

    dataset_path = Path("data/processed/master_dataset/master_ticker_day.parquet")
    raw_df = pd.read_parquet(dataset_path)
    cohort_df, _ = build_common_evaluation_cohort(raw_df)
    cohort_df = cohort_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    engine = WalkForwardEngine(date_col="date", target_date_col="target_date")
    tech_cols = FEATURE_GROUPS["group_a_technical"]

    records = []

    # ---------------------------------------------------------
    # Control 1: Perfect Next-Day Direction Canary Feature
    # ---------------------------------------------------------
    print("\n[Control 1] Perfect Next-Day Direction Feature Injection...", flush=True)
    leak_df = cohort_df.copy()
    leak_df["future_leak_target"] = leak_df["target"].values
    leak_features = tech_cols + ["future_leak_target"]

    folds = engine.generate_protocol_b_folds(leak_df)
    # Evaluate across last 10 folds
    eval_folds = folds[-10:]

    xgb_leak_preds = []
    for fold in eval_folds:
        train_df = leak_df.iloc[fold.train_indices].copy()
        val_df = leak_df.iloc[fold.val_indices].copy() if fold.val_indices is not None else None
        test_df = leak_df.iloc[fold.test_indices].copy()

        xgb = FinFusionXGBoost(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            random_state=42,
            tree_method="hist",
        )
        xgb.fit(
            train_df=train_df,
            feature_cols=leak_features,
            target_col="target",
            ticker_col="ticker",
            val_df=val_df,
            early_stopping_rounds=15,
        )
        probs = xgb.predict_proba(test_df, feature_cols=leak_features, ticker_col="ticker")
        preds = xgb.predict(test_df, feature_cols=leak_features, ticker_col="ticker")
        xgb_leak_preds.append(pd.DataFrame({
            "date": test_df["date"].dt.strftime("%Y-%m-%d").values,
            "y_true": test_df["target"].values.astype(int),
            "y_pred": preds,
            "y_prob": probs,
        }))

    all_leak = pd.concat(xgb_leak_preds, ignore_index=True)
    leak_acc = accuracy_score(all_leak["y_true"], all_leak["y_pred"])
    leak_date_acc = all_leak.groupby("date").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False).mean()
    leak_auc = roc_auc_score(all_leak["y_true"], all_leak["y_prob"])

    print(f"  Leak Feature Date Accuracy:   {leak_date_acc:.4f} (Pooled: {leak_acc:.4f})", flush=True)
    print(f"  Leak Feature ROC-AUC:         {leak_auc:.4f}", flush=True)

    records.append({
        "test_type": "perfect_leak_canary",
        "description": "True next-day direction feature",
        "date_level_acc": round(float(leak_date_acc), 4),
        "pooled_acc": round(float(leak_acc), 4),
        "roc_auc": round(float(leak_auc), 4),
        "expected_benchmark": 1.0000,
    })

    assert leak_date_acc >= 0.99, f"FAIL: Perfect leak accuracy {leak_date_acc} < 0.99"
    assert leak_auc >= 0.99, f"FAIL: Perfect leak AUC {leak_auc} < 0.99"
    print("  PASS: Canary feature successfully recovers ~100% accuracy.", flush=True)

    # ---------------------------------------------------------
    # Control 2: Weak AR(1) Signal Recovery
    # ---------------------------------------------------------
    print("\n[Control 2] Weak AR(1) Predictable Process (rho = 0.20)...", flush=True)
    rho = 0.20
    theoretical_acc = 0.50 + (1.0 / np.pi) * np.arcsin(rho)
    print(f"  Theoretical Directional Accuracy: {theoretical_acc:.4f} ({theoretical_acc*100:.2f}%)", flush=True)

    calendar = get_trading_calendar("2020-01-01", "2026-09-30")
    n_days = len(calendar)
    rng = np.random.RandomState(42)

    # Simulate 10 tickers with AR(1) return dynamics
    ar_records = []
    tickers = [f"AR_{i:02d}" for i in range(1, 11)]

    for t in tickers:
        innovations = rng.normal(0, 0.015, size=n_days)
        returns = np.zeros(n_days)
        returns[0] = innovations[0]
        for step in range(1, n_days):
            returns[step] = rho * returns[step - 1] + np.sqrt(1 - rho**2) * innovations[step]

        base_p = 1000.0
        close_p = base_p * np.cumprod(1.0 + returns)

        for d_idx, dt in enumerate(calendar):
            c = close_p[d_idx]
            prev_c = close_p[d_idx - 1] if d_idx > 0 else c
            o = prev_c * (1.0 + rng.normal(0, 0.002))
            spread = max(o, c) * rng.uniform(0.002, 0.01)
            h = max(o, c) + spread
            l = min(o, c) - spread
            v = int(rng.lognormal(mean=14.0, sigma=0.5))
            ar_records.append({
                "date": dt,
                "ticker": t,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            })

    ar_df_raw = pd.DataFrame(ar_records)
    ar_feat = compute_technical_indicators(ar_df_raw)
    ar_df = compute_targets(ar_feat, calendar=calendar)

    ar_cohort = ar_df[ar_df["date"] >= pd.Timestamp("2021-01-01")].dropna(subset=tech_cols + ["target"]).reset_index(drop=True)
    ar_cohort = ar_cohort.sort_values(["date", "ticker"]).reset_index(drop=True)

    ar_folds = engine.generate_protocol_b_folds(ar_cohort)
    eval_ar_folds = ar_folds[-15:]

    ar_preds = []
    for fold in eval_ar_folds:
        train_df = ar_cohort.iloc[fold.train_indices].copy()
        val_df = ar_cohort.iloc[fold.val_indices].copy() if fold.val_indices is not None else None
        test_df = ar_cohort.iloc[fold.test_indices].copy()

        xgb = FinFusionXGBoost(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            tree_method="hist",
        )
        xgb.fit(
            train_df=train_df,
            feature_cols=tech_cols,
            target_col="target",
            ticker_col="ticker",
            val_df=val_df,
            early_stopping_rounds=20,
        )
        probs = xgb.predict_proba(test_df, feature_cols=tech_cols, ticker_col="ticker")
        preds = xgb.predict(test_df, feature_cols=tech_cols, ticker_col="ticker")
        ar_preds.append(pd.DataFrame({
            "date": test_df["date"].dt.strftime("%Y-%m-%d").values,
            "y_true": test_df["target"].values.astype(int),
            "y_pred": preds,
            "y_prob": probs,
        }))

    all_ar = pd.concat(ar_preds, ignore_index=True)
    ar_acc = accuracy_score(all_ar["y_true"], all_ar["y_pred"])
    ar_date_acc = all_ar.groupby("date").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"]), include_groups=False).mean()
    ar_auc = roc_auc_score(all_ar["y_true"], all_ar["y_prob"])

    print(f"  AR(1) Recovered Date Accuracy: {ar_date_acc:.4f} (Pooled: {ar_acc:.4f})", flush=True)
    print(f"  AR(1) Recovered ROC-AUC:       {ar_auc:.4f}", flush=True)
    print(f"  Theoretical Target:            {theoretical_acc:.4f}", flush=True)

    records.append({
        "test_type": "weak_ar1_recovery",
        "description": "Simulated AR(1) process with rho=0.20",
        "date_level_acc": round(float(ar_date_acc), 4),
        "pooled_acc": round(float(ar_acc), 4),
        "roc_auc": round(float(ar_auc), 4),
        "expected_benchmark": round(float(theoretical_acc), 4),
    })

    # Assert model recovers signal within acceptable sampling bounds around theoretical 0.564
    assert 0.535 <= ar_date_acc <= 0.595, f"FAIL: AR(1) accuracy {ar_date_acc} outside theoretical bounds [0.535, 0.595]"
    assert ar_auc > 0.53, f"FAIL: AR(1) ROC-AUC {ar_auc} failed to detect signal"
    print("  PASS: Pipeline successfully detected and recovered weak AR(1) predictive signal.", flush=True)

    out_csv = Path("experiments/audit/positive_controls.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\nSaved positive control results to {out_csv}", flush=True)
    return records


if __name__ == "__main__":
    run_positive_controls()
