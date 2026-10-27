"""FinFusionNet: Reproducible LSTM Sequence Generation Engine.

Converts tabular ticker-day observations into rolling 20-session sequences.
Strictly guarantees:
1. Sequences NEVER cross ticker boundaries.
2. Temporal ordering is strictly preserved.
3. Preceding context from training partitions is preserved for test boundaries.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def generate_ticker_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
    target_col: str = "target",
    target_date_col: str = "target_date",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate rolling 20-session sequences for deep learning models.

    Input shape: Tabular DataFrame
    Output:
    - X_seq: 3D float32 array of shape (N, sequence_length, len(feature_cols))
    - y: 1D int array of shape (N,)
    - dates: 1D Timestamp array of shape (N,) representing prediction session T
    - tickers: 1D string array of shape (N,)
    - target_dates: 1D Timestamp array of shape (N,) representing target session T+1
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if target_date_col in df.columns:
        df[target_date_col] = pd.to_datetime(df[target_date_col])

    # Enforce numeric finiteness: fill missing sentiment with 0.0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    x_list = []
    y_list = []
    dates_list = []
    tickers_list = []
    target_dates_list = []

    # Group by ticker independently to strictly isolate stocks
    for ticker, group in df.groupby(ticker_col):
        grp = group.sort_values(date_col).reset_index(drop=True)
        n_obs = len(grp)

        if n_obs < sequence_length:
            continue

        feat_vals = grp[feature_cols].values.astype(np.float32)
        y_vals = grp[target_col].values.astype(int) if target_col in grp.columns else np.zeros(n_obs, dtype=int)
        d_vals = grp[date_col].values
        td_vals = grp[target_date_col].values if target_date_col in grp.columns else grp[date_col].values

        # Rolling window generation
        for t in range(sequence_length - 1, n_obs):
            # Window covers [t - 19, ..., t]
            seq_window = feat_vals[t - sequence_length + 1 : t + 1]
            x_list.append(seq_window)
            y_list.append(y_vals[t])
            dates_list.append(d_vals[t])
            tickers_list.append(ticker)
            target_dates_list.append(td_vals[t])

    if not x_list:
        return (
            np.empty((0, sequence_length, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=int),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    X_seq = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=int)
    dates = np.array(dates_list, dtype="datetime64[ns]")
    tickers = np.array(tickers_list, dtype=object)
    target_dates = np.array(target_dates_list, dtype="datetime64[ns]")

    return X_seq, y, dates, tickers, target_dates


def generate_multimodal_sequences(
    df: pd.DataFrame,
    market_cols: List[str],
    news_cols: List[str],
    sequence_length: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
    target_cls_col: str = "target",
    target_vol_col: str = "target_realized_vol",
    target_date_col: str = "target_date",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate rolling dual-stream sequences for cross-attention models.

    Returns:
    - X_market: (N, seq_len, len(market_cols))
    - X_news: (N, seq_len, len(news_cols))
    - y_dir: (N,) binary direction target
    - y_vol: (N,) annualized realized volatility target
    - dates: (N,) prediction session dates
    - tickers: (N,) tickers
    - target_dates: (N,) target session dates
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if target_date_col in df.columns:
        df[target_date_col] = pd.to_datetime(df[target_date_col])

    # Clean market features
    for col in market_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    # Clean news features (missing news filled with 0.0)
    for col in news_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)

    xm_list = []
    xn_list = []
    ydir_list = []
    yvol_list = []
    dates_list = []
    tickers_list = []
    target_dates_list = []

    for ticker, group in df.groupby(ticker_col):
        grp = group.sort_values(date_col).reset_index(drop=True)
        n_obs = len(grp)

        if n_obs < sequence_length:
            continue

        m_vals = grp[market_cols].values.astype(np.float32)
        n_vals = grp[news_cols].values.astype(np.float32)
        ydir_vals = grp[target_cls_col].values.astype(np.float32) if target_cls_col in grp.columns else np.zeros(n_obs, dtype=np.float32)
        yvol_vals = grp[target_vol_col].values.astype(np.float32) if target_vol_col in grp.columns else np.full(n_obs, np.nan, dtype=np.float32)
        d_vals = grp[date_col].values
        td_vals = grp[target_date_col].values if target_date_col in grp.columns else grp[date_col].values

        for t in range(sequence_length - 1, n_obs):
            xm_list.append(m_vals[t - sequence_length + 1 : t + 1])
            xn_list.append(n_vals[t - sequence_length + 1 : t + 1])
            ydir_list.append(ydir_vals[t])
            yvol_list.append(yvol_vals[t])
            dates_list.append(d_vals[t])
            tickers_list.append(ticker)
            target_dates_list.append(td_vals[t])

    if not xm_list:
        return (
            np.empty((0, sequence_length, len(market_cols)), dtype=np.float32),
            np.empty((0, sequence_length, len(news_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype="datetime64[ns]"),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    return (
        np.array(xm_list, dtype=np.float32),
        np.array(xn_list, dtype=np.float32),
        np.array(ydir_list, dtype=np.float32),
        np.array(yvol_list, dtype=np.float32),
        np.array(dates_list, dtype="datetime64[ns]"),
        np.array(tickers_list, dtype=object),
        np.array(target_dates_list, dtype="datetime64[ns]"),
    )


def extract_boundary_multimodal_sequences(
    full_df: pd.DataFrame,
    context_dates: List[pd.Timestamp],
    eval_dates: List[pd.Timestamp],
    market_cols: List[str],
    news_cols: List[str],
    sequence_length: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
    target_cls_col: str = "target",
    target_vol_col: str = "target_realized_vol",
    target_date_col: str = "target_date",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract rolling multimodal sequences for eval_dates using boundary context."""
    all_dates = list(context_dates) + list(eval_dates)
    sub_df = full_df[full_df[date_col].isin(all_dates)].copy()

    X_m, X_n, y_dir, y_vol, dates, tickers, target_dates = generate_multimodal_sequences(
        sub_df,
        market_cols=market_cols,
        news_cols=news_cols,
        sequence_length=sequence_length,
        date_col=date_col,
        ticker_col=ticker_col,
        target_cls_col=target_cls_col,
        target_vol_col=target_vol_col,
        target_date_col=target_date_col,
    )

    eval_dates_set = set(pd.to_datetime(eval_dates))
    keep_mask = np.array([pd.Timestamp(d) in eval_dates_set for d in dates])

    return (
        X_m[keep_mask],
        X_n[keep_mask],
        y_dir[keep_mask],
        y_vol[keep_mask],
        dates[keep_mask],
        tickers[keep_mask],
        target_dates[keep_mask],
    )

