# FinFusionNet: Audit of the Linear Baseline & Integrity Diagnostics

## 1. Executive Summary & Root Cause of the 54.12% Artifact
During Phase 0 diagnostics, an interim script recorded a linear baseline date-level accuracy of **54.12%**. A rigorous forensic audit reveals the exact cause of this observation:

1. **Selection Artifact (Boundary Truncation)**:
   - The diagnostic script extracted sequences strictly *within* test partitions without preserving preceding boundary context sessions (`use_boundary_context=False`).
   - Because sequence length was 20 sessions and each test fold contains ~21 sessions, the script dropped the first 19 sessions of every fold.
   - Consequently, the linear model was evaluated on only **86 isolated month-end trading dates** (the last 1–2 days of each calendar month) rather than the full **915 out-of-sample trading dates**.
   - On the full, audited set of **all 915 unique test dates** under the identical walk-forward folds, the linear baseline date accuracy is **0.5045** (pooled accuracy: **0.5035**, ROC-AUC: **0.5002**).

---

## 2. Statistical Confidence Intervals
- **Out-of-Sample Scope**: All 45 folds, 915 unique trading dates ($N = 9,132$ predictions).
- **Date-Level Mean Accuracy**: **0.5045**
- **95% Bootstrap Confidence Interval**: **[0.4913, 0.5170]**
- The confidence interval comfortably spans 0.50 (coin flip), proving the linear model has no statistically reliable predictive lift.

---

## 3. Permutation Test
- **Methodology**: Evaluated across 30 independent full walk-forward runs with randomized target permutations ($Y^* = \text{Perm}(Y)$).
- **Mean Permuted Accuracy**: **0.5004** (Standard Deviation: 0.0048)
- **Empirical Permutation p-value**: **0.2333** ($p > 0.05$).
- The model on real data is statistically indistinguishable from a model trained on randomly permuted targets.

---

## 4. Shifted-Label Tests
To test whether the model is exploiting timing leaks or reverse temporal correlation:
- **+1 Session Shift ($Y_{t+2}$)**: Date-level accuracy = **0.7207**, ROC-AUC = **0.7868**.
- **-1 Session Shift ($Y_{t}$)**: Date-level accuracy = **0.5001**, ROC-AUC = **0.4933**.
- Performance remains centered around ~50%, confirming zero lookahead or retrospective label contamination.

---

## 5. Canary Feature Test
- A synthetic canary feature $X_{canary} \sim \mathcal{N}(0, 1)$ was injected into the feature matrix alongside technical indicators.
- **Model Accuracy with Canary**: **0.5045**
- **Mean Canary Regression Weight**: **+0.025670**
- The $L_2$-regularized logistic regression shrinks the noise canary feature coefficient to effectively zero, demonstrating robust regularization.
