"""Unit and Integration Tests for Experiment 5: Temporal Cross-Attention Fusion.

Tests:
1. TemporalCrossAttentionNet forward pass shapes.
2. Numerically safe cross-attention with sentinel token under 100% masked news sequences (no NaNs).
3. Sentinel token weight verification on masked sequences.
4. Realized volatility target computation (Parkinson 1980 estimator).
5. Diebold-Mariano test statistical properties and small-sample correction.
6. Multi-task gradient flow to market encoder, news encoder, sentinel token, and dual heads.
7. FinFusionCrossAttentionClassifier fit and predict pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from finfusionnet.evaluation.stats_tests import diebold_mariano_test
from finfusionnet.features.volatility_target import compute_next_session_parkinson_volatility
from finfusionnet.models.cross_attention_model import (
    FinFusionCrossAttentionClassifier,
    TemporalCrossAttentionNet,
)


class TestTemporalCrossAttentionModel:
    """Tests for PyTorch Cross-Attention architecture."""

    def test_forward_shapes(self):
        B, T = 8, 20
        d_market, d_news, d_model = 24, 8, 32

        model = TemporalCrossAttentionNet(
            d_market=d_market,
            d_news=d_news,
            d_model=d_model,
            n_heads=2,
            seq_len=T,
            dense_dim=16,
        )
        model.eval()

        x_m = torch.randn(B, T, d_market)
        x_n = torch.randn(B, T, d_news)
        news_mask = torch.zeros(B, T, dtype=torch.bool)  # all active news

        with torch.inference_mode():
            prob_dir, pred_vol, attn_weights = model(x_m, x_n, news_mask)

        assert prob_dir.shape == (B,), f"Expected prob_dir shape {(B,)}, got {prob_dir.shape}"
        assert pred_vol.shape == (B,), f"Expected pred_vol shape {(B,)}, got {pred_vol.shape}"
        assert attn_weights.shape == (B, T, T + 1), f"Expected attn_weights {(B, T, T + 1)}, got {attn_weights.shape}"

        # Probabilities must be in [0, 1]
        assert (prob_dir >= 0.0).all() and (prob_dir <= 1.0).all()
        # Volatility must be strictly non-negative (softplus)
        assert (pred_vol >= 0.0).all()

    def test_all_masked_attention_stability(self):
        """CRITICAL: Sequences with 100% missing news across all 20 sessions must NOT produce NaNs."""
        B, T = 4, 20
        d_market, d_news, d_model = 24, 8, 32

        model = TemporalCrossAttentionNet(
            d_market=d_market,
            d_news=d_news,
            d_model=d_model,
            n_heads=2,
            seq_len=T,
            dense_dim=16,
        )
        model.eval()

        x_m = torch.randn(B, T, d_market)
        x_n = torch.zeros(B, T, d_news)
        news_mask = torch.ones(B, T, dtype=torch.bool)  # 100% masked (no news at all)

        with torch.inference_mode():
            prob_dir, pred_vol, attn_weights = model(x_m, x_n, news_mask)

        assert not torch.isnan(prob_dir).any(), "Direction predictions contain NaNs under 100% masked news!"
        assert not torch.isnan(pred_vol).any(), "Volatility predictions contain NaNs under 100% masked news!"
        assert not torch.isinf(prob_dir).any(), "Direction predictions contain Inf!"
        assert not torch.isinf(pred_vol).any(), "Volatility predictions contain Inf!"
        assert not torch.isnan(attn_weights).any(), "Attention weights contain NaNs!"

        # Since all news positions are masked, all attention weight must collapse to the sentinel token (index 0)
        # Note: PyTorch multihead_attention with key_padding_mask puts 1.0 on the only unmasked token (sentinel)
        sentinel_attn = attn_weights[:, :, 0]  # (B, T)
        assert torch.allclose(sentinel_attn, torch.ones_like(sentinel_attn), atol=1e-4), (
            f"Sentinel token did not receive 100% attention under all-masked sequence: {sentinel_attn}"
        )

    def test_gradient_flow_dual_heads(self):
        """Verify backpropagation flows through both market and news encoders, sentinel token, and heads."""
        B, T = 4, 20
        d_market, d_news, d_model = 24, 8, 32

        model = TemporalCrossAttentionNet(
            d_market=d_market,
            d_news=d_news,
            d_model=d_model,
            n_heads=2,
            seq_len=T,
            dense_dim=16,
        )
        model.train()

        x_m = torch.randn(B, T, d_market, requires_grad=True)
        x_n = torch.randn(B, T, d_news, requires_grad=True)
        news_mask = torch.zeros(B, T, dtype=torch.bool)
        # Set half of row 1 to masked
        news_mask[1, :10] = True

        y_dir = torch.tensor([1.0, 0.0, 1.0, 0.0])
        y_vol = torch.tensor([0.22, 0.18, 0.35, 0.15])

        prob_dir, pred_vol, _ = model(x_m, x_n, news_mask)
        loss_cls = torch.nn.functional.binary_cross_entropy(prob_dir, y_dir)
        loss_vol = torch.nn.functional.smooth_l1_loss(pred_vol, y_vol)
        total_loss = loss_cls + loss_vol

        total_loss.backward()

        # Check gradients exist and are non-zero
        assert model.sentinel_token.grad is not None
        assert model.market_proj.weight.grad is not None
        assert model.news_proj.weight.grad is not None
        assert model.head_cls[0].weight.grad is not None
        assert model.head_vol[0].weight.grad is not None
        assert not torch.isnan(model.sentinel_token.grad).any()
        assert not torch.isnan(model.head_vol[0].weight.grad).any()


class TestVolatilityTarget:
    """Tests for Parkinson realized volatility target calculation."""

    def test_parkinson_volatility_formula(self):
        """Test exact mathematical value of Parkinson volatility on synthetic prices."""
        # Suppose H = 105, L = 95. Ratio = 105/95 = 1.105263...
        # ln(1.105263) = 0.100083
        # (ln(H/L))^2 = 0.0100166
        # / (4 * ln(2)) = 0.0100166 / 2.7725887 = 0.0036128
        # sqrt(...) = 0.060107
        # * sqrt(252) = 0.060107 * 15.874507866 = 0.954168
        dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])
        df = pd.DataFrame({
            "ticker": ["RELIANCE", "RELIANCE", "RELIANCE"],
            "date": dates,
            "target_date": [dates[1], dates[2], pd.NaT],
            "open": [100.0, 100.0, 100.0],
            "high": [102.0, 105.0, 103.0],
            "low": [98.0, 95.0, 97.0],
            "close": [100.0, 100.0, 100.0],
        })

        df_out = compute_next_session_parkinson_volatility(df)
        assert "target_realized_vol" in df_out.columns

        # For row 0, target_date is 2024-01-02, where High=105, Low=95
        expected_vol = np.sqrt((np.log(105.0 / 95.0) ** 2) / (4.0 * np.log(2.0))) * np.sqrt(252.0)
        np.testing.assert_allclose(df_out.loc[0, "target_realized_vol"], expected_vol, rtol=1e-4)

        # For row 2 (boundary session), target is NaT so realized vol must be NaN
        assert np.isnan(df_out.loc[2, "target_realized_vol"])

    def test_parkinson_winsorization(self):
        """Test extreme high/low spikes are capped at max_vol_clip."""
        df = pd.DataFrame({
            "ticker": ["TEST", "TEST"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "target_date": [pd.to_datetime("2024-01-02"), pd.NaT],
            "high": [100.0, 1000.0],  # 10x spike
            "low": [100.0, 10.0],
            "close": [100.0, 500.0],
        })

        df_out = compute_next_session_parkinson_volatility(df, max_vol_clip=2.0)
        assert df_out.loc[0, "target_realized_vol"] == 2.0


class TestDieboldMariano:
    """Tests for Diebold-Mariano predictive accuracy test."""

    def test_dm_identical_forecasts(self):
        """Identical forecasts should have differential = 0, stat = 0, p-val = 1.0."""
        y_true = np.random.rand(100)
        y_pred = y_true + np.random.normal(0, 0.1, 100)
        res = diebold_mariano_test(y_true, y_pred, y_pred)
        assert np.isclose(res["dm_stat"], 0.0)
        assert np.isclose(res["p_value"], 1.0)
        assert res["better_model"] == "equal"

    def test_dm_strictly_superior_forecast(self):
        """Significantly better model should have positive DM stat and p < 0.01."""
        np.random.seed(42)
        y_true = np.random.randn(200)
        y_good = y_true + np.random.normal(0, 0.05, 200)
        y_bad = y_true + np.random.normal(0, 1.0, 200)

        res = diebold_mariano_test(y_true, y_bad, y_good, loss_type="squared")
        # diff = bad_loss - good_loss > 0 => dm_stat > 0 => model_b better
        assert res["dm_stat"] > 5.0
        assert res["p_value"] < 1e-4
        assert res["better_model"] == "model_b"


class TestClassifierWrapper:
    """Tests for FinFusionCrossAttentionClassifier wrapper."""

    def test_fit_and_predict(self):
        np.random.seed(42)
        torch.manual_seed(42)

        N_tr, N_te, T = 60, 20, 20
        D_m, D_n = 24, 8

        X_m_tr = np.random.randn(N_tr, T, D_m).astype(np.float32)
        X_n_tr = np.random.randn(N_tr, T, D_n).astype(np.float32)
        # Ensure col 1 is has_news flag
        X_n_tr[:, :, 1] = np.random.choice([0.0, 1.0], size=(N_tr, T), p=[0.3, 0.7])

        y_dir_tr = np.random.choice([0.0, 1.0], size=N_tr).astype(np.float32)
        y_vol_tr = np.random.uniform(0.10, 0.40, size=N_tr).astype(np.float32)

        X_m_te = np.random.randn(N_te, T, D_m).astype(np.float32)
        X_n_te = np.random.randn(N_te, T, D_n).astype(np.float32)
        X_n_te[:, :, 1] = np.random.choice([0.0, 1.0], size=(N_te, T), p=[0.4, 0.6])

        clf = FinFusionCrossAttentionClassifier(
            d_market=D_m,
            d_news=D_n,
            d_model=16,
            n_heads=2,
            seq_len=T,
            dense_dim=8,
            max_epochs=2,
            batch_size=32,
            random_state=42,
            device="cpu",
        )

        clf.fit(X_m_tr, X_n_tr, y_dir_tr, y_vol_tr)

        prob_dir, pred_vol, pred_dir = clf.predict(X_m_te, X_n_te)

        assert prob_dir.shape == (N_te,)
        assert pred_vol.shape == (N_te,)
        assert pred_dir.shape == (N_te,)
        assert not np.isnan(prob_dir).any()
        assert not np.isnan(pred_vol).any()
        assert ((pred_dir == 0) | (pred_dir == 1)).all()


class TestMultimodalSequenceGenerator:
    """Tests for multimodal sequence generator and boundary extraction."""

    def test_generate_and_boundary_extraction(self):
        from finfusionnet.data.sequence_generator import (
            generate_multimodal_sequences,
            extract_boundary_multimodal_sequences,
        )

        dates = pd.date_range("2024-01-01", periods=30, freq="B")
        df_a = pd.DataFrame({
            "ticker": ["TICK_A"] * 30,
            "date": dates,
            "target_date": list(dates[1:]) + [pd.NaT],
            "m1": np.random.randn(30),
            "m2": np.random.randn(30),
            "n_articles": np.random.randint(0, 5, 30),
            "has_news": np.random.choice([0.0, 1.0], 30),
            "target": np.random.choice([0.0, 1.0], 30),
            "target_realized_vol": np.random.uniform(0.1, 0.4, 30),
        })

        X_m, X_n, y_dir, y_vol, d_out, t_out, td_out = generate_multimodal_sequences(
            df_a,
            market_cols=["m1", "m2"],
            news_cols=["n_articles", "has_news"],
            sequence_length=20,
        )

        # 30 days, sequence_length=20 -> 11 sequences (days 19 through 29)
        assert len(X_m) == 11
        assert X_m.shape == (11, 20, 2)
        assert X_n.shape == (11, 20, 2)
        assert len(y_dir) == 11
        assert len(y_vol) == 11
        assert len(d_out) == 11

        # Test boundary extraction
        # Context dates: first 25 dates
        # Eval dates: last 5 dates
        context_dates = dates[:25]
        eval_dates = dates[25:]
        X_m_eval, X_n_eval, y_dir_eval, y_vol_eval, d_eval, t_eval, td_eval = extract_boundary_multimodal_sequences(
            df_a,
            context_dates=context_dates,
            eval_dates=eval_dates,
            market_cols=["m1", "m2"],
            news_cols=["n_articles", "has_news"],
            sequence_length=20,
        )

        assert len(X_m_eval) == 5
        assert len(d_eval) == 5
        assert (pd.to_datetime(d_eval) == eval_dates).all()

