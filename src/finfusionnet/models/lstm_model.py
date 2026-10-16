"""FinFusionNet: PyTorch LSTM Sequential Classifier.

Implements:
1. Parimonious 2-layer LSTM with dropout regularization and dense classification head.
2. In-fold feature scaling strictly using training data parameters.
3. Chronological early stopping on validation loss.
4. Fixed 0.50 classification threshold.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class FinFusionLSTMNet(nn.Module):
    """PyTorch sequential LSTM architecture for financial direction prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
        dense_dim: int = 32,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_dim, dense_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dense_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len=20, input_size)
        lstm_out, _ = self.lstm(x)
        # Select representation from the final time-step T
        last_step = lstm_out[:, -1, :]
        h = self.dropout(self.relu(self.fc1(last_step)))
        out = self.sigmoid(self.fc2(h)).squeeze(-1)
        return out


class FinFusionLSTMClassifier:
    """Scikit-learn style wrapper for PyTorch FinFusionLSTMNet."""

    def __init__(
        self,
        input_size: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.20,
        dense_dim: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        batch_size: int = 64,
        max_epochs: int = 30,
        early_stopping_patience: int = 5,
        random_state: int = 42,
        threshold: float = 0.50,
        device: Optional[str] = None,
    ):
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.dense_dim = dense_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.threshold = threshold

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[FinFusionLSTMNet] = None
        self.best_epoch: int = 0
        self.best_val_loss: float = float("inf")

    def _fit_transform_scaler(self, X: np.ndarray, is_training: bool = False) -> np.ndarray:
        """Fit scaler strictly on training sequence steps; transform validation/test."""
        N, S, F = X.shape
        X_reshaped = X.reshape(-1, F)

        if is_training:
            self.scaler = StandardScaler()
            X_scaled_2d = self.scaler.fit_transform(X_reshaped)
        else:
            if self.scaler is None:
                raise ValueError("Scaler must be fitted on training data first.")
            X_scaled_2d = self.scaler.transform(X_reshaped)

        return X_scaled_2d.reshape(N, S, F).astype(np.float32)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Fit model strictly on training data with optional early stopping on validation partition."""
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Scale features strictly using train parameters
        X_tr = self._fit_transform_scaler(X_train, is_training=True)
        y_tr = torch.tensor(y_train, dtype=torch.float32)
        train_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr), y_tr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        has_val = X_val is not None and y_val is not None and len(X_val) > 0
        if has_val:
            X_v = self._fit_transform_scaler(X_val, is_training=False)
            y_v = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            val_tensor = torch.tensor(X_v).to(self.device)

        self.model = FinFusionLSTMNet(
            input_size=self.input_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            dense_dim=self.dense_dim,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.BCELoss()

        best_weights = None
        self.best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss = 0.0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                preds = self.model(bx)
                loss = criterion(preds, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(by)

            train_loss /= len(y_tr)

            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_preds = self.model(val_tensor)
                    val_loss = float(criterion(val_preds, y_v).item())

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} (best epoch: {self.best_epoch})")
                        break
            else:
                self.best_epoch = epoch
                best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict out-of-sample positive class probabilities."""
        if self.model is None:
            raise ValueError("Model is not fitted.")
        self.model.eval()
        X_scaled = self._fit_transform_scaler(X, is_training=False)
        dataset = TensorDataset(torch.tensor(X_scaled))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs_list = []
        with torch.no_grad():
            for (bx,) in loader:
                bx = bx.to(self.device)
                p = self.model(bx).cpu().numpy()
                probs_list.append(p)

        return np.concatenate(probs_list)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary classes using fixed out-of-sample threshold (0.50)."""
        probs = self.predict_proba(X)
        return np.where(probs > self.threshold, 1, 0)
