"""FinFusionNet: Multimodal Temporal Cross-Attention Fusion Architecture.

Combines historical market sequences (Query) and temporally aligned FinBERT
news feature representations (Key & Value) through multi-head cross-attention.

Features:
1. Numerically safe cross-attention with learnable sentinel (null-news) token
   guaranteeing zero NaNs on sequences with no news (100% masked days).
2. Dual prediction heads:
   - Head 1: Next-day direction classification (BCE loss, Sigmoid).
   - Head 2: Next-day realized-volatility regression (Huber/MSE loss, Softplus).
3. In-fold feature scaling strictly using training data parameters.
4. Chronological validation loss early stopping (Protocol B).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler


class TemporalCrossAttentionNet(nn.Module):
    """PyTorch Multimodal Cross-Attention Architecture with Dual Heads."""

    def __init__(
        self,
        d_market: int = 24,
        d_news: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        seq_len: int = 20,
        dropout: float = 0.20,
        dense_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len

        # 1. Market Stream Encoder
        self.market_proj = nn.Linear(d_market, d_model)
        self.market_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # 2. News Stream Encoder
        self.news_proj = nn.Linear(d_news, d_model)
        self.news_gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=1,
            batch_first=True,
        )

        # 3. Sentinel (Null-News) Learnable Token
        # Prepended to news key/value sequence to prevent all-masked attention collapse
        self.sentinel_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4. Positional Embeddings
        self.pos_emb_market = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.pos_emb_news = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        # 5. Multi-Head Temporal Cross-Attention
        # Query: Market Temporal Representation
        # Key & Value: News Temporal Representation + Sentinel Token
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 6. Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 7. Temporal Aggregation / Pooling Head
        self.temporal_attn_pool = nn.Linear(d_model, 1)

        # 8. Dual Prediction Heads
        # Head 1: Binary Direction Classification
        self.head_cls = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
        )

        # Head 2: Realized Volatility Regression
        self.head_vol = nn.Sequential(
            nn.Linear(d_model, dense_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1),
            nn.Softplus(),  # Guarantees strictly positive volatility prediction
        )

    def forward(
        self,
        x_market: torch.Tensor,
        x_news: torch.Tensor,
        news_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x_market: (B, seq_len=20, d_market=24)
            x_news: (B, seq_len=20, d_news=8)
            news_mask: (B, seq_len=20) boolean tensor where True indicates NO NEWS.

        Returns:
            prob_dir: (B,) class probability of next-day up direction
            pred_vol: (B,) predicted next-day annualized realized volatility
            attn_weights: (B, seq_len, seq_len+1) cross-attention weights
        """
        B, T, _ = x_market.shape

        # A. Encode Market Price Stream
        h_m = self.market_proj(x_market) + self.pos_emb_market[:, :T, :]
        h_m, _ = self.market_gru(h_m)  # (B, T, d_model)

        # B. Encode News Stream
        h_n = self.news_proj(x_news) + self.pos_emb_news[:, :T, :]
        h_n, _ = self.news_gru(h_n)  # (B, T, d_model)

        # C. Numerically Safe Key Augmentation with Sentinel Token
        # Prepend sentinel token to news stream so keys sequence length = T + 1
        sentinel_expanded = self.sentinel_token.expand(B, 1, self.d_model)
        h_n_aug = torch.cat([sentinel_expanded, h_n], dim=1)  # (B, T+1, d_model)

        # Augment key_padding_mask: sentinel token is NEVER masked (False)
        sentinel_mask = torch.zeros((B, 1), dtype=torch.bool, device=news_mask.device)
        mask_aug = torch.cat([sentinel_mask, news_mask], dim=1)  # (B, T+1)

        # D. Temporal Cross-Attention
        # Q = h_m, K = h_n_aug, V = h_n_aug
        attn_out, attn_weights = self.cross_attn(
            query=h_m,
            key=h_n_aug,
            value=h_n_aug,
            key_padding_mask=mask_aug,
        )

        h_fused = self.norm1(h_m + self.dropout1(attn_out))
        h_fused = self.norm2(h_fused + self.dropout2(self.ffn(h_fused)))  # (B, T, d_model)

        # E. Temporal Attention Pooling
        pool_weights = F.softmax(self.temporal_attn_pool(h_fused), dim=1)  # (B, T, 1)
        z = torch.sum(h_fused * pool_weights, dim=1)  # (B, d_model)

        # F. Dual Heads
        logits_dir = self.head_cls(z).squeeze(-1)  # (B,)
        prob_dir = torch.sigmoid(logits_dir)

        pred_vol = self.head_vol(z).squeeze(-1)  # (B,)

        return prob_dir, pred_vol, attn_weights


class FinFusionCrossAttentionClassifier:
    """Scikit-learn style wrapper for training and inference."""

    def __init__(
        self,
        d_market: int = 24,
        d_news: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        seq_len: int = 20,
        dropout: float = 0.20,
        dense_dim: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        vol_loss_weight: float = 1.0,
        batch_size: int = 256,
        max_epochs: int = 25,
        early_stopping_patience: int = 5,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.d_market = d_market
        self.d_news = d_news
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.dropout = dropout
        self.dense_dim = dense_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.vol_loss_weight = vol_loss_weight
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model: Optional[TemporalCrossAttentionNet] = None
        self.scaler_market: Optional[StandardScaler] = None
        self.scaler_news: Optional[StandardScaler] = None
        self.scaler_vol: Optional[StandardScaler] = None
        self.best_val_loss = float("inf")

    def _init_model(self):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.model = TemporalCrossAttentionNet(
            d_market=self.d_market,
            d_news=self.d_news,
            d_model=self.d_model,
            n_heads=self.n_heads,
            seq_len=self.seq_len,
            dropout=self.dropout,
            dense_dim=self.dense_dim,
        ).to(self.device)

    def fit(
        self,
        X_m_tr: np.ndarray,
        X_n_tr: np.ndarray,
        y_dir_tr: np.ndarray,
        y_vol_tr: np.ndarray,
        X_m_val: Optional[np.ndarray] = None,
        X_n_val: Optional[np.ndarray] = None,
        y_dir_val: Optional[np.ndarray] = None,
        y_vol_val: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        """Fit model strictly using in-fold training data scalers."""
        self._init_model()

        # 1. Fit scalers on training sequence steps
        N_tr, T, D_m = X_m_tr.shape
        _, _, D_n = X_n_tr.shape

        self.scaler_market = StandardScaler()
        X_m_tr_scaled = self.scaler_market.fit_transform(X_m_tr.reshape(-1, D_m)).reshape(N_tr, T, D_m)

        # Scale news features (only on active-news steps to prevent zero inflation)
        self.scaler_news = StandardScaler()
        X_n_tr_flat = X_n_tr.reshape(-1, D_n)
        has_news_flat = X_n_tr_flat[:, 1] > 0  # index 1 is has_news
        if has_news_flat.sum() > 0:
            self.scaler_news.fit(X_n_tr_flat[has_news_flat])
        else:
            self.scaler_news.fit(X_n_tr_flat)

        X_n_tr_scaled = self.scaler_news.transform(X_n_tr_flat).reshape(N_tr, T, D_n)

        # Missing news mask: True where has_news == 0
        mask_n_tr = (X_n_tr[:, :, 1] == 0)

        # Target volatility: replace any NaNs with median of training set
        vol_median = float(np.nanmedian(y_vol_tr)) if np.isnan(y_vol_tr).any() else 0.20
        y_vol_tr_clean = np.where(np.isnan(y_vol_tr), vol_median, y_vol_tr)

        # Prepare PyTorch Tensors
        t_xm_tr = torch.tensor(X_m_tr_scaled, dtype=torch.float32)
        t_xn_tr = torch.tensor(X_n_tr_scaled, dtype=torch.float32)
        t_mask_tr = torch.tensor(mask_n_tr, dtype=torch.bool)
        t_ydir_tr = torch.tensor(y_dir_tr, dtype=torch.float32)
        t_yvol_tr = torch.tensor(y_vol_tr_clean, dtype=torch.float32)

        dataset_tr = TensorDataset(t_xm_tr, t_xn_tr, t_mask_tr, t_ydir_tr, t_yvol_tr)
        loader_tr = DataLoader(dataset_tr, batch_size=self.batch_size, shuffle=True)

        # Validation set preparation
        has_val = X_m_val is not None and len(X_m_val) > 0
        if has_val:
            N_val = len(X_m_val)
            X_m_val_scaled = self.scaler_market.transform(X_m_val.reshape(-1, D_m)).reshape(N_val, T, D_m)
            X_n_val_scaled = self.scaler_news.transform(X_n_val.reshape(-1, D_n)).reshape(N_val, T, D_n)
            mask_n_val = (X_n_val[:, :, 1] == 0)
            y_vol_val_clean = np.where(np.isnan(y_vol_val), vol_median, y_vol_val)

            t_xm_val = torch.tensor(X_m_val_scaled, dtype=torch.float32, device=self.device)
            t_xn_val = torch.tensor(X_n_val_scaled, dtype=torch.float32, device=self.device)
            t_mask_val = torch.tensor(mask_n_val, dtype=torch.bool, device=self.device)
            t_ydir_val = torch.tensor(y_dir_val, dtype=torch.float32, device=self.device)
            t_yvol_val = torch.tensor(y_vol_val_clean, dtype=torch.float32, device=self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        bce_loss_fn = nn.BCELoss()
        vol_loss_fn = nn.SmoothL1Loss()  # Huber loss for robust volatility regression

        patience_counter = 0
        best_state = None

        for epoch in range(self.max_epochs):
            self.model.train()
            train_loss = 0.0

            for bx_m, bx_n, b_mask, by_dir, by_vol in loader_tr:
                bx_m = bx_m.to(self.device)
                bx_n = bx_n.to(self.device)
                b_mask = b_mask.to(self.device)
                by_dir = by_dir.to(self.device)
                by_vol = by_vol.to(self.device)

                optimizer.zero_grad()
                p_dir, p_vol, _ = self.model(bx_m, bx_n, b_mask)

                loss_cls = bce_loss_fn(p_dir, by_dir)
                loss_vol = vol_loss_fn(p_vol, by_vol)
                total_loss = loss_cls + (self.vol_loss_weight * loss_vol)

                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += total_loss.item() * len(by_dir)

            train_loss /= N_tr

            # Validation step
            if has_val:
                self.model.eval()
                with torch.inference_mode():
                    val_p_dir, val_p_vol, _ = self.model(t_xm_val, t_xn_val, t_mask_val)
                    val_loss_cls = bce_loss_fn(val_p_dir, t_ydir_val)
                    val_loss_vol = vol_loss_fn(val_p_vol, t_yvol_val)
                    val_loss = (val_loss_cls + self.vol_loss_weight * val_loss_vol).item()

                if verbose and (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1:02d}/{self.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

    def predict(
        self,
        X_m: np.ndarray,
        X_n: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate predictions for test sequences.

        Returns:
            prob_dir: (N,) predicted direction probabilities
            pred_vol: (N,) predicted annualized realized volatility
            pred_dir: (N,) binary direction predictions (threshold=0.50)
        """
        self.model.eval()
        N, T, D_m = X_m.shape
        _, _, D_n = X_n.shape

        X_m_scaled = self.scaler_market.transform(X_m.reshape(-1, D_m)).reshape(N, T, D_m)
        X_n_scaled = self.scaler_news.transform(X_n.reshape(-1, D_n)).reshape(N, T, D_n)
        mask_n = (X_n[:, :, 1] == 0)

        t_xm = torch.tensor(X_m_scaled, dtype=torch.float32)
        t_xn = torch.tensor(X_n_scaled, dtype=torch.float32)
        t_mask = torch.tensor(mask_n, dtype=torch.bool)

        dataset = TensorDataset(t_xm, t_xn, t_mask)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        probs_dir_list = []
        preds_vol_list = []

        with torch.inference_mode():
            for bx_m, bx_n, b_mask in loader:
                bx_m = bx_m.to(self.device)
                bx_n = bx_n.to(self.device)
                b_mask = b_mask.to(self.device)

                p_dir, p_vol, _ = self.model(bx_m, bx_n, b_mask)
                probs_dir_list.append(p_dir.cpu().numpy())
                preds_vol_list.append(p_vol.cpu().numpy())

        prob_dir = np.concatenate(probs_dir_list)
        pred_vol = np.concatenate(preds_vol_list)
        pred_dir = (prob_dir >= 0.50).astype(int)

        return prob_dir, pred_vol, pred_dir
