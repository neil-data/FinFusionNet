"""FinFusionNet: XGBoost Model Implementation for Financial Direction Prediction.

Implements pooled XGBoost binary classifier with:
1. Strict in-fold OneHotEncoder for ticker identity (fitted ONLY on training partition).
2. Native missing-value handling for sentiment features.
3. Fixed 0.50 probability threshold for class assignment.
4. Gain-based feature importance extraction.
5. Early stopping on chronological validation partition (Protocol B).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder


class FinFusionXGBoost:
    """XGBoost classifier configured for leakage-safe financial market prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.03,
        max_depth: int = 4,
        min_child_weight: float = 5.0,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 2.0,
        gamma: float = 0.0,
        tree_method: str = "hist",
        n_jobs: int = 4,
        random_state: int = 42,
        threshold: float = 0.50,
        use_ticker_encoding: bool = True,
    ):
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": ["auc", "logloss"],
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "gamma": gamma,
            "tree_method": tree_method,
            "n_jobs": n_jobs,
            "random_state": random_state,
        }
        self.threshold = threshold
        self.use_ticker_encoding = use_ticker_encoding
        self.encoder: Optional[OneHotEncoder] = None
        self.feature_names: List[str] = []
        self.model: Optional[xgb.XGBClassifier] = None

    def _prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ticker_col: str = "ticker",
        is_training: bool = False,
    ) -> np.ndarray:
        """Extract numeric features and append in-fold one-hot encoded ticker columns."""
        X_num = df[feature_cols].copy().values.astype(np.float32)

        if not self.use_ticker_encoding or ticker_col not in df.columns:
            if is_training:
                self.feature_names = list(feature_cols)
            return X_num

        tickers = df[[ticker_col]].copy()

        if is_training:
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_encoded = self.encoder.fit_transform(tickers)
            encoded_names = [f"ticker_{c}" for c in self.encoder.categories_[0]]
            self.feature_names = list(feature_cols) + encoded_names
        else:
            if self.encoder is None:
                raise ValueError("Encoder must be fitted on training fold first.")
            X_encoded = self.encoder.transform(tickers)

        return np.hstack([X_num, X_encoded]).astype(np.float32)

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target",
        ticker_col: str = "ticker",
        val_df: Optional[pd.DataFrame] = None,
        early_stopping_rounds: Optional[int] = None,
    ):
        """Fit model strictly on training data with optional early stopping on validation partition."""
        X_train = self._prepare_features(
            train_df, feature_cols, ticker_col=ticker_col, is_training=True
        )
        y_train = train_df[target_col].values.astype(int)

        self.model = xgb.XGBClassifier(**self.params)

        if val_df is not None and early_stopping_rounds is not None and early_stopping_rounds > 0:
            X_val = self._prepare_features(
                val_df, feature_cols, ticker_col=ticker_col, is_training=False
            )
            y_val = val_df[target_col].values.astype(int)

            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train, verbose=False)

        return self

    def predict_proba(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ticker_col: str = "ticker",
    ) -> np.ndarray:
        """Predict out-of-sample positive class probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        X = self._prepare_features(
            df, feature_cols, ticker_col=ticker_col, is_training=False
        )
        # Probability for class 1
        return self.model.predict_proba(X)[:, 1]

    def predict(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        ticker_col: str = "ticker",
    ) -> np.ndarray:
        """Predict binary classes using fixed out-of-sample threshold (0.50)."""
        probs = self.predict_proba(df, feature_cols, ticker_col=ticker_col)
        return np.where(probs > self.threshold, 1, 0)

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract XGBoost gain, weight, and cover feature importance."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        booster = self.model.get_booster()
        gains = booster.get_score(importance_type="gain")
        weights = booster.get_score(importance_type="weight")
        covers = booster.get_score(importance_type="cover")

        records = []
        for i, fname in enumerate(self.feature_names):
            f_key = f"f{i}"
            records.append({
                "feature": fname,
                "gain": gains.get(f_key, 0.0),
                "weight": weights.get(f_key, 0.0),
                "cover": covers.get(f_key, 0.0),
            })
        df_imp = pd.DataFrame(records).sort_values("gain", ascending=False).reset_index(drop=True)
        return df_imp
