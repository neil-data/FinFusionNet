"""Compute fold-level stability and sector breakdown for Experiment 3."""

import pandas as pd
import numpy as np

df = pd.read_parquet("experiments/experiment_3/predictions/experiment_3_common_cohort_predictions.parquet")

# 1. Fold-level metrics
folds = []
for f_idx, grp in df.groupby("fold_idx"):
    y = grp["y_true"].values
    acc_xgb_f0 = float(np.mean(y == (grp["prob_xgb_f0"] >= 0.5).astype(int)))
    acc_xgb_f1 = float(np.mean(y == (grp["prob_xgb_f1"] >= 0.5).astype(int)))
    acc_lr_f0 = float(np.mean(y == (grp["prob_lr_f0"] >= 0.5).astype(int)))
    acc_lr_f1 = float(np.mean(y == (grp["prob_lr_f1"] >= 0.5).astype(int)))
    acc_lstm_f0 = float(np.mean(y == (grp["prob_lstm_f0"] >= 0.5).astype(int)))
    acc_lstm_f1 = float(np.mean(y == (grp["prob_lstm_f1"] >= 0.5).astype(int)))
    
    folds.append({
        "fold_idx": f_idx,
        "test_start": str(grp["date"].min()),
        "test_end": str(grp["date"].max()),
        "n_samples": len(grp),
        "acc_xgb_f0": round(acc_xgb_f0, 4),
        "acc_xgb_f1": round(acc_xgb_f1, 4),
        "diff_xgb": round(acc_xgb_f1 - acc_xgb_f0, 4),
        "acc_lr_f0": round(acc_lr_f0, 4),
        "acc_lr_f1": round(acc_lr_f1, 4),
        "diff_lr": round(acc_lr_f1 - acc_lr_f0, 4),
        "acc_lstm_f0": round(acc_lstm_f0, 4),
        "acc_lstm_f1": round(acc_lstm_f1, 4),
        "diff_lstm": round(acc_lstm_f1 - acc_lstm_f0, 4),
    })

df_folds = pd.DataFrame(folds)
df_folds.to_csv("experiments/experiment_3/experiment_3_fold_results.csv", index=False)

print("=== FOLD-LEVEL STABILITY SUMMARY (45 FOLDS) ===")
print(f"XGB-F0 Mean Fold Acc: {df_folds['acc_xgb_f0'].mean():.4f} (std: {df_folds['acc_xgb_f0'].std():.4f})")
print(f"XGB-F1 Mean Fold Acc: {df_folds['acc_xgb_f1'].mean():.4f} (std: {df_folds['acc_xgb_f1'].std():.4f})")
print(f"LR-F0 Mean Fold Acc:  {df_folds['acc_lr_f0'].mean():.4f} (std: {df_folds['acc_lr_f0'].std():.4f})")
print(f"LR-F1 Mean Fold Acc:  {df_folds['acc_lr_f1'].mean():.4f} (std: {df_folds['acc_lr_f1'].std():.4f})")
print(f"LSTM-F0 Mean Fold Acc: {df_folds['acc_lstm_f0'].mean():.4f} (std: {df_folds['acc_lstm_f0'].std():.4f})")
print(f"LSTM-F1 Mean Fold Acc: {df_folds['acc_lstm_f1'].mean():.4f} (std: {df_folds['acc_lstm_f1'].std():.4f})")

# 2. News Days vs No-News Days Breakdown
print("\n=== NEWS PRESENCE SUBSET BREAKDOWN ===")
news_mask = (df["has_news"] == 1)
print(f"Active News Observations: {news_mask.sum():,} ({news_mask.mean() * 100:.1f}%)")
print(f"No-News Observations:     {(~news_mask).sum():,} ({(~news_mask).mean() * 100:.1f}%)")

for subset_name, mask in [("Active News Days", news_mask), ("No-News Days", ~news_mask)]:
    sub = df[mask]
    y_s = sub["y_true"].values
    print(f"\n--- {subset_name} (N={len(sub):,}) ---")
    print(f"XGB-F0 Acc:  {np.mean(y_s == (sub['prob_xgb_f0'] >= 0.5).astype(int)):.4f}")
    print(f"XGB-F1 Acc:  {np.mean(y_s == (sub['prob_xgb_f1'] >= 0.5).astype(int)):.4f} (Diff: {np.mean(y_s == (sub['prob_xgb_f1'] >= 0.5).astype(int)) - np.mean(y_s == (sub['prob_xgb_f0'] >= 0.5).astype(int)):+.4f})")
    print(f"LR-F0 Acc:   {np.mean(y_s == (sub['prob_lr_f0'] >= 0.5).astype(int)):.4f}")
    print(f"LR-F1 Acc:   {np.mean(y_s == (sub['prob_lr_f1'] >= 0.5).astype(int)):.4f} (Diff: {np.mean(y_s == (sub['prob_lr_f1'] >= 0.5).astype(int)) - np.mean(y_s == (sub['prob_lr_f0'] >= 0.5).astype(int)):+.4f})")
    print(f"LSTM-F0 Acc: {np.mean(y_s == (sub['prob_lstm_f0'] >= 0.5).astype(int)):.4f}")
    print(f"LSTM-F1 Acc: {np.mean(y_s == (sub['prob_lstm_f1'] >= 0.5).astype(int)):.4f} (Diff: {np.mean(y_s == (sub['prob_lstm_f1'] >= 0.5).astype(int)) - np.mean(y_s == (sub['prob_lstm_f0'] >= 0.5).astype(int)):+.4f})")
