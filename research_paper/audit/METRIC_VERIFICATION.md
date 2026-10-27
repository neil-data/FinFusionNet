# FinFusionNet Paper Audit: Metric Verification and Parity Confirmation

All quantitative metrics cited in the manuscript have been cross-checked against the common evaluation cohort parquets:

## 1. Classification Metrics (Cohort: 45,136 observations, 916 dates)
- `LSTM-F0`: Date Acc = 0.5108, Pooled Acc = 0.5096, Balanced Acc = 0.5091, Precision = 0.5095, Recall = 0.6297, F1 = 0.5633, ROC-AUC = 0.5149, Brier = 0.2502, Log Loss = 0.6935, Up Share = 0.6207. [CONFIRMED]
- `LSTM-F1_VADER`: Date Acc = 0.5099, Pooled Acc = 0.5088, Balanced Acc = 0.5084, Precision = 0.5092, Recall = 0.6045, F1 = 0.5528, ROC-AUC = 0.5105, Brier = 0.2504, Log Loss = 0.6940, Up Share = 0.5962. [CONFIRMED]
- `LR-F1_FinBERT`: Date Acc = 0.5095, Pooled Acc = 0.5083, Balanced Acc = 0.5077, Precision = 0.5081, Recall = 0.6567, F1 = 0.5729, ROC-AUC = 0.5103, Brier = 0.2505, Log Loss = 0.6943, Up Share = 0.6491. [CONFIRMED]
- `LR-F0`: Date Acc = 0.5094, Pooled Acc = 0.5082, Balanced Acc = 0.5075, Precision = 0.5078, Recall = 0.6724, F1 = 0.5786, ROC-AUC = 0.5102, Brier = 0.2505, Log Loss = 0.6942, Up Share = 0.6649. [CONFIRMED]
- `XGB-F1_VADER`: Date Acc = 0.5054, Pooled Acc = 0.5043, Balanced Acc = 0.5040, Precision = 0.5057, Recall = 0.5622, F1 = 0.5325, ROC-AUC = 0.5090, Brier = 0.2564, Log Loss = 0.7069, Up Share = 0.5582. [CONFIRMED]
- `LSTM-F1_FinBERT`: Date Acc = 0.5053, Pooled Acc = 0.5041, Balanced Acc = 0.5038, Precision = 0.5054, Recall = 0.5842, F1 = 0.5419, ROC-AUC = 0.5092, Brier = 0.2505, Log Loss = 0.6942, Up Share = 0.5804. [CONFIRMED]
- `XGB-F1_FinBERT`: Date Acc = 0.5042, Pooled Acc = 0.5042, Balanced Acc = 0.5039, Precision = 0.5057, Recall = 0.5593, F1 = 0.5311, ROC-AUC = 0.5070, Brier = 0.2568, Log Loss = 0.7076, Up Share = 0.5553. [CONFIRMED]
- `Always-Up`: Date Acc = 0.5034, Pooled Acc = 0.5021, Balanced Acc = 0.5000, Precision = 0.5021, Recall = 1.0000, F1 = 0.6686, ROC-AUC = 0.5000, Brier = 0.4979, Log Loss = 8.0244, Up Share = 1.0000. [CONFIRMED]
- `XGB-F0`: Date Acc = 0.5027, Pooled Acc = 0.5015, Balanced Acc = 0.4996, Precision = 0.5019, Recall = 0.9351, F1 = 0.6532, ROC-AUC = 0.5067, Brier = 0.2503, Log Loss = 0.6937, Up Share = 0.9355. [CONFIRMED]
- `Cross-Attention Net`: Date Acc = 0.5016, Pooled Acc = 0.5004, Balanced Acc = 0.4996, Precision = 0.5019, Recall = 0.6852, F1 = 0.5794, ROC-AUC = 0.5016, Brier = 0.2506, Log Loss = 0.6943, Up Share = 0.6856. [CONFIRMED]

## 2. Volatility Metrics (Parkinson 1980 High-Low Target)
- Target True Mean = 0.2024, Target True Std = 0.1135.
- `Trailing Parkinson 20d`: RMSE = 0.1052, MAE = 0.0695, Bias = +0.0003, r = 0.3979, R2 = +14.24%, Date MSE DM vs Cross-Attn = -1.9515 (p = 0.0513). [CONFIRMED]
- `Trailing Parkinson 10d`: RMSE = 0.1058, MAE = 0.0700, Bias = +0.0003, r = 0.4084, R2 = +13.13%, Date MSE DM vs Cross-Attn = -0.9768 (p = 0.3289). [CONFIRMED]
- `Trailing Parkinson 5d`: RMSE = 0.1066, MAE = 0.0708, Bias = +0.0002, r = 0.4301, R2 = +11.89%, Date MSE DM vs Cross-Attn = -0.3994 (p = 0.6897). [CONFIRMED]
- `Cross-Attention Net`: RMSE = 0.1075, MAE = 0.0686, Bias = -0.0089, r = 0.3369, R2 = +10.38%. [CONFIRMED]
- `Trailing Close-to-Close 20d`: RMSE = 0.1322, MAE = 0.0900, Bias = +0.0331, r = 0.3225, R2 = -35.58%. [CONFIRMED]
